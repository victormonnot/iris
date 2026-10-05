"""Frozen native model packages and imported target measurements, without inference."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import zipfile
from copy import deepcopy
from pathlib import Path

from iris.datasets import load_manifest
from iris.evaluation import evaluation_detail
from iris.evaluation_analysis import _analyze
from iris.model_taxonomy import class_contract
from iris.models import TRAINING_ARCHITECTURE
from iris.projects import project_records, record_project
from iris.store import DEFAULT_PROJECT_ID, Store, _decode, new_id, now

MAX_BUNDLE_BYTES = 1280 * 1024**2
MAX_MEASUREMENT_BYTES = 8 * 1024**2
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
INPUT_TRANSFORM = {
    "color": "RGB",
    "tensor_range": [0, 1],
    "exif_transpose": True,
    "image_mean": [0.485, 0.456, 0.406],
    "image_std": [0.229, 0.224, 0.225],
    "min_size": [320],
    "max_size": 640,
    "fixed_size": None,
    "size_divisible": 32,
}
NATIVE_FILTERING = {
    "score_threshold": 0.001,
    "nms_iou_threshold": 0.5,
    "max_detections_per_image": 100,
    "ssdlite_topk_candidates_per_class": None,
    "rpn": {
        "score_threshold": 0.05,
        "nms_iou_threshold": 0.7,
        "pre_nms_top_n": 150,
        "post_nms_top_n": 150,
    },
}
SSDLITE_INPUT_TRANSFORM = {
    **INPUT_TRANSFORM,
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
    "max_size": 320,
    "fixed_size": [320, 320],
    "size_divisible": 1,
}
SSDLITE_NATIVE_FILTERING = {
    "score_threshold": 0.001,
    "nms_iou_threshold": 0.5,
    "max_detections_per_image": 100,
    "ssdlite_topk_candidates_per_class": 300,
}
WARNINGS = [
    "The package includes the selected frozen images and saved native predictions. "
    "It does not include human annotation boxes or reviewer notes.",
    "Packaging copies files only. Model loading, real parity and target performance "
    "are not validated by creating this export.",
    "The target is native PyTorch float32, batch one, independently of the training device. "
    "Exact parity is checked without widening tolerances, including across CPU and CUDA.",
    "CUDA requires compatible NVIDIA drivers and PyTorch/Torchvision builds. "
    "ARM and Jetson support depends on the board's vendor runtime; this is not a TensorRT export.",
    "Imported measurements are declarations from their producer. Checksums and "
    "consistency checks do not independently prove execution or authenticity.",
]
REQUIREMENTS = b"torch==2.10.0\ntorchvision==0.25.0\nPillow==12.3.0\n"
README = b"""# IRIS native model export

This bundle runs independently of IRIS and its database. Extract it into a new
directory. It contains a full trained state_dict, frozen classes and 1-8 selected
parity images with saved IRIS native predictions (no human annotations).

Use Python 3.12 or 3.13 and the versions in requirements.txt, with PyTorch and
Torchvision wheels compatible with the target CPU or NVIDIA CUDA runtime.
The frozen target is manifest.json profile.device; it is independent of the
training device. Dependency installation is a separate, explicit action on the
target machine. Nothing here automatically installs packages or downloads weights.

From the extracted directory:

    python run.py inspect
    python run.py check-runtime
    python run.py predict /path/to/image.png --output ../prediction.json
    python run.py measure --repeats 3 --output ../measurement.json

Inspect checks integrity without importing ML libraries or deserializing weights.
Check-runtime explicitly imports the installed runtime and probes the selected
device without constructing a model or loading weights; success does not prove
that inference works. For CUDA targets, --device cuda:N chooses a visible GPU
on check-runtime, predict or measure. The default uses the current CUDA device.
The device family must match the export's frozen target.
Predict loads weights on CPU using weights_only=True and strict state_dict
matching, then transfers the model and inputs to the selected device.
The profile and class mapping are frozen in manifest.json.
Only use bundles from a trusted source; hashes check integrity, not authenticity.

Measure performs one warmup on the first parity image, then measures every bundled
image for each repetition. Exact box, score, label, dimension and order parity is
compared to saved IRIS outputs. A failure remains a failure: no tolerance widening.
CPU/CUDA numerical or detection-order differences may cause exact parity failure.
CUDA timing synchronizes the selected device at stage boundaries and includes
input transfer in preprocessing. TF32 and cuDNN benchmarking are disabled.
An empty reference only checks empty output, not positive detection quality.
Import measurement.json through IRIS Model exports to save consistency-checked,
declared parity and timings. This is not an independently verified execution claim.

Predict total_ms covers preprocessing, forward/postprocessing in Torchvision and
result serialization; it excludes decoding, checkpoint/hash loading and warmup.
Those costs are recorded separately by measure. Saved IRIS evaluation total_ms
includes decoding and must not be compared directly with this predict-only total.

Packaging does not execute the model. Real execution, parity and target performance
remain untested until you run the bundle. ARM and Jetson boards need vendor
runtime versions compatible with this pinned profile; older boards may not
support them. This is native PyTorch, not ONNX or TensorRT, and provides no
universal hardware or accuracy/generalization guarantee. Selected images may
contain private data: review the selection before sharing the bundle elsewhere.
"""


def _runtime():
    from iris import export_runner

    return export_runner


def _canonical(value):
    return _runtime().canonical_bytes(value)


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_digest(path, *, maximum=MAX_BUNDLE_BYTES, cancelled=lambda: False):
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= maximum:
        raise ValueError("Export artifact is missing, unsafe or exceeds its size limit")
    digest, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        while block := stream.read(1024**2):
            if cancelled():
                raise ExportCancelled()
            size += len(block)
            if size > maximum:
                raise ValueError("Export artifact exceeds its size limit")
            digest.update(block)
    return {"sha256": digest.hexdigest(), "size": size}


def _resources():
    return {
        "run.py": Path(_runtime().__file__).read_bytes(),
        "requirements.txt": REQUIREMENTS,
        "README.md": README,
    }


class ExportCancelled(RuntimeError):
    """The copy attempt stopped before publication."""


class _View:
    """Read a single SQLite snapshot without opening or migrating a workspace."""

    def __init__(self, root, connection):
        self.root, self.connection = Path(root).resolve(), connection

    def get(self, table, identifier):
        from iris.store import TABLES

        if table not in TABLES:
            raise ValueError("Unsupported table")
        return _decode(
            self.connection.execute(f"SELECT * FROM {table} WHERE id=?", (identifier,)).fetchone()
        )

    def list(self, table, **filters):
        from iris.store import TABLES

        if table not in TABLES or any(not _ID.fullmatch(key) for key in filters):
            raise ValueError("Unsupported table or field")
        where = " WHERE " + " AND ".join(f"{key}=?" for key in filters) if filters else ""
        return [
            _decode(row)
            for row in self.connection.execute(
                f"SELECT * FROM {table}{where} ORDER BY created_at,id", list(filters.values())
            )
        ]

    def artifact_path(self, relative):
        return Store.artifact_path(self, relative)


def _source(store, model_id, evaluation_id):
    model = store.get("trained_models", model_id)
    if model is None:
        raise ValueError("Choose a locally trained checkpoint")
    architecture = model["architecture"]
    if architecture not in (TRAINING_ARCHITECTURE, _runtime().SSDLITE_ARCHITECTURE):
        raise ValueError("This export supports trained Faster R-CNN and SSDLite MobileNetV3 only")
    ssdlite = architecture == _runtime().SSDLITE_ARCHITECTURE
    if model["metadata"].get("architecture", None if ssdlite else architecture) != architecture:
        raise ValueError("Checkpoint metadata architecture differs from the trained model")
    contract = class_contract(model["metadata"])
    detail = evaluation_detail(store, evaluation_id)
    if record_project(store, "trained_models", model) != record_project(
        store, "evaluations", detail
    ):
        raise ValueError("The checkpoint and evaluation must belong to the same project")
    _analyze(detail, evaluation_id)
    runs = [
        run for run in detail["models"] if run["model_id"] == model_id and run["variant"] == "full"
    ]
    if len(runs) != 1 or runs[0]["metrics"] is None:
        raise ValueError("Choose a completed full-image evaluation of this checkpoint")
    run, metadata = runs[0], runs[0]["metadata"]
    reference_device = _runtime().device_family(metadata.get("device"))
    if (
        metadata.get("weight_sha256") != model["weight_sha256"]
        or metadata.get("architecture") != architecture
        or metadata.get("precision") != "float32"
        or class_contract(metadata) != contract
        or metadata.get("head_class_slots") != len(contract["class_mapping"]) + 1
        or metadata.get("input_transform")
        != (SSDLITE_INPUT_TRANSFORM if ssdlite else INPUT_TRANSFORM)
        or metadata.get("native_filtering")
        != (SSDLITE_NATIVE_FILTERING if ssdlite else NATIVE_FILTERING)
        or not isinstance(metadata.get("torch_version"), str)
        or metadata["torch_version"].split("+")[0] != "2.10.0"
        or not isinstance(metadata.get("torchvision_version"), str)
        or metadata["torchvision_version"].split("+")[0] != "0.25.0"
        or (
            reference_device == "cuda"
            and any(
                metadata[key].endswith("+cpu") for key in ("torch_version", "torchvision_version")
            )
        )
    ):
        raise ValueError("Saved evaluation does not match this native float32 export profile")
    return model, detail, run, load_manifest(store, detail["dataset_id"])


def candidates(store, project_id=DEFAULT_PROJECT_ID):
    evaluations = project_records(store, "evaluations", project_id)
    models = []
    for model in project_records(store, "trained_models", project_id):
        choices, failures = [], []
        for evaluation in evaluations:
            if model["id"] not in evaluation["model_ids"]:
                continue
            try:
                _, detail, run, _ = _source(store, model["id"], evaluation["id"])
                choices.append(
                    {
                        "id": detail["id"],
                        "name": detail["name"],
                        "device": run["metadata"]["device"],
                        "frames": [
                            {
                                "frame_id": frame["frame_id"],
                                "width": frame["width"],
                                "height": frame["height"],
                                "image_url": frame["image_url"],
                            }
                            for frame in detail["frames"]
                        ],
                    }
                )
            except (ValueError, KeyError, TypeError, OSError) as exc:
                failures.append(str(exc))
        models.append(
            {
                "id": model["id"],
                "name": model["name"],
                "architecture": model["architecture"],
                "eligible": bool(choices),
                "evaluations": choices,
                "reason": ""
                if choices
                else (failures[0] if failures else "A completed full-image evaluation is required"),
            }
        )
    return {
        "models": models,
        "profile": deepcopy(_runtime().PROFILE),
        "target_devices": ["cpu", "cuda"],
    }


def _plan(
    store,
    *,
    trained_model_id,
    evaluation_id,
    frame_ids,
    name,
    request_id,
    target_device="cpu",
    frozen=None,
):
    if target_device not in ("cpu", "cuda"):
        raise ValueError("Choose a CPU or CUDA export target")
    if (
        not isinstance(name, str)
        or not 1 <= len(name.strip()) <= 160
        or not isinstance(request_id, str)
        or not _ID.fullmatch(request_id)
        or not isinstance(frame_ids, list)
        or not 1 <= len(frame_ids) <= 8
        or any(not isinstance(item, str) or not _ID.fullmatch(item) for item in frame_ids)
        or len(set(frame_ids)) != len(frame_ids)
    ):
        raise ValueError("Choose a name and 1–8 distinct parity images")
    model, detail, run, dataset = _source(store, trained_model_id, evaluation_id)
    ssdlite = model["architecture"] == _runtime().SSDLITE_ARCHITECTURE
    modern = ssdlite or target_device == "cuda" or run["metadata"]["device"] != "cpu"
    available = {
        frame["frame_id"]: frame for frame in dataset["frames"] if frame["split"] == detail["split"]
    }
    if not set(frame_ids) <= set(available):
        raise ValueError("Parity images must belong to the selected evaluation")
    if frozen is None:
        weights = _file_digest(store.artifact_path(model["path"]), maximum=1024**3)
        resources = {
            key: {"sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw)}
            for key, raw in _resources().items()
        }
    else:
        weights = {"sha256": frozen["model"]["sha256"], "size": frozen["model"]["size"]}
        resources = frozen["resources"]
    if weights["sha256"] != model["weight_sha256"]:
        raise ValueError("Checkpoint bytes changed since training")
    predictions = {
        row["frame_id"]: row
        for row in detail["predictions"]
        if row["evaluation_model_id"] == run["id"]
    }
    frames, references = [], []
    frozen_frames = {}
    if frozen is not None:
        frozen_frames = {frame["frame_id"]: frame for frame in frozen["frames"]}
        if set(frozen_frames) != set(frame_ids) or len(frozen["frames"]) != len(frame_ids):
            raise ValueError("Frozen export frame inventory is inconsistent")
    for frame_id in frame_ids:
        frame, prediction = available[frame_id], predictions[frame_id]
        identity = (
            _file_digest(store.artifact_path(frame["image_path"]), maximum=32 * 1024**2)
            if frozen is None
            else {
                "sha256": frozen_frames[frame_id]["image_file_sha256"],
                "size": frozen_frames[frame_id]["size"],
            }
        )
        if identity["sha256"] != frame["image_file_sha256"]:
            raise ValueError("A frozen parity image changed since evaluation")
        frames.append(
            {
                "frame_id": frame_id,
                "image_path": frame["image_path"],
                "image_file_sha256": identity["sha256"],
                "size": identity["size"],
                "width": frame["width"],
                "height": frame["height"],
            }
        )
        references.append(
            {
                "frame_id": frame_id,
                "path": f"parity/images/{frame_id}.png",
                "sha256": identity["sha256"],
                "input_size": prediction["input_size"],
                "detections": prediction["detections"],
            }
        )
    plan = {
        "format": "iris-model-export-plan-v3"
        if ssdlite
        else "iris-model-export-plan-v2"
        if modern
        else "iris-model-export-plan-v1",
        "request_id": request_id,
        "name": name.strip(),
        "trained_model_id": trained_model_id,
        "evaluation_id": evaluation_id,
        "frame_ids": frame_ids,
        "model": {
            "id": model["id"],
            "name": model["name"],
            "architecture": model["architecture"],
            **weights,
            "class_contract": class_contract(model["metadata"]),
        },
        "checkpoint_path": model["path"],
        "frames": frames,
        "resources": resources,
        "source": {
            "evaluation_id": evaluation_id,
            "evaluation_model_id": run["id"],
            "dataset_id": detail["dataset_id"],
            "dataset_manifest_sha256": detail["config"]["dataset_manifest_sha256"],
        },
        "evaluation_metadata_sha256": _digest(run["metadata"]),
        "reference": {
            "format": "iris-export-reference-v1",
            "model_id": model["id"],
            "weight_sha256": model["weight_sha256"],
            "frames": references,
        },
        "profile": _runtime().ssdlite_profile(target_device)
        if ssdlite
        else _runtime().native_profile(target_device)
        if modern
        else deepcopy(_runtime().PROFILE),
    }
    if modern:
        plan["target_device"] = target_device
        plan["source"]["reference_device"] = run["metadata"]["device"]
    manifest = _manifest(plan, request_id, "2000-01-01T00:00:00+00:00")
    _runtime().validate_manifest(manifest)
    _runtime().validate_reference(manifest, plan["reference"])
    return plan


def _manifest(plan, identifier, created_at):
    files = {
        "model.pth": {key: plan["model"][key] for key in ("sha256", "size")},
        **plan["resources"],
        "parity/reference.json": {
            "sha256": _digest(plan["reference"]),
            "size": len(_canonical(plan["reference"])),
        },
    }
    for frame in plan["frames"]:
        files[f"parity/images/{frame['frame_id']}.png"] = {
            "sha256": frame["image_file_sha256"],
            "size": frame["size"],
        }
    return {
        "format": {
            "iris-model-export-plan-v1": "iris-model-export-v1",
            "iris-model-export-plan-v2": "iris-model-export-v2",
            "iris-model-export-plan-v3": "iris-model-export-v3",
        }[plan["format"]],
        "id": identifier,
        "name": plan["name"],
        "created_at": created_at,
        "model": plan["model"],
        "source": plan["source"],
        "profile": plan["profile"],
        "files": files,
        "validation": {"real_execution": "not_run", "reference_kind": "saved_iris_evaluation"},
    }


def preview_export(store, **options):
    request_id = new_id()
    with store.connect() as connection:
        connection.execute("BEGIN")
        plan = _plan(_View(store.root, connection), request_id=request_id, **options)
    return {
        "plan": plan,
        "request_id": request_id,
        "fingerprint": _digest(plan),
        "warnings": WARNINGS,
    }


def create_export(store, *, request_id, expected_fingerprint, **options):
    if not isinstance(expected_fingerprint, str) or not _SHA.fullmatch(expected_fingerprint):
        raise ValueError("Preview this export before creating it")
    with store.connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        view = _View(store.root, connection)
        previous = view.list("model_exports", request_id=request_id)
        if previous:
            row = previous[0]
            if _digest(row["config"]) != expected_fingerprint or any(
                (
                    row["config"]["profile"]["device"]
                    if key == "target_device"
                    else row["config"].get(key)
                )
                != value
                for key, value in options.items()
            ):
                raise ValueError("This preview request was already used with different options")
        else:
            plan = _plan(view, request_id=request_id, **options)
            if _digest(plan) != expected_fingerprint:
                raise ValueError("Export inputs changed; inspect a fresh preview")
            identifier, job_id, created_at = new_id(), new_id(), now()
            connection.execute(
                "INSERT INTO jobs (id,kind,status,params,message,created_at) VALUES (?,?,?,?,?,?)",
                (
                    job_id,
                    "model_export",
                    "queued",
                    json.dumps({"export_id": identifier}),
                    "Waiting to copy the standalone model package",
                    created_at,
                ),
            )
            connection.execute(
                "INSERT INTO model_exports "
                "(id,trained_model_id,evaluation_id,name,config,request_id,job_id,created_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (
                    identifier,
                    options["trained_model_id"],
                    options["evaluation_id"],
                    plan["name"],
                    _canonical(plan).decode(),
                    request_id,
                    job_id,
                    created_at,
                ),
            )
            row = view.get("model_exports", identifier)
    return export_detail(store, row["id"])


def export_detail(store, identifier):
    row = store.get("model_exports", identifier)
    if row is None:
        raise KeyError(identifier)
    return {
        **row,
        "ready": row["path"] is not None,
        "job": store.get("jobs", row["job_id"]),
        "measurements": [
            {key: item[key] for key in ("id", "summary", "fingerprint", "created_at")}
            for item in store.list("model_export_measurements", export_id=identifier)
        ],
    }


def list_exports(store, project_id=DEFAULT_PROJECT_ID):
    return [
        export_detail(store, row["id"])
        for row in project_records(store, "model_exports", project_id)
    ]


def _options(plan):
    return {
        "target_device": plan["profile"]["device"],
        **{
            key: plan[key]
            for key in ("trained_model_id", "evaluation_id", "frame_ids", "name", "request_id")
        },
    }


def run_export(store, identifier, progress, cancelled):
    row = store.get("model_exports", identifier)
    if row is None:
        raise ValueError("Export not found")
    if row["path"]:
        raise ValueError("This immutable export is already published")
    directory = store.root / "model_exports"
    directory.mkdir(exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".building-", dir=directory))
    published = False
    target = directory / identifier
    try:
        if cancelled():
            raise ExportCancelled()
        with store.connect() as connection:
            connection.execute("BEGIN")
            plan = _plan(_View(store.root, connection), **_options(row["config"]))
        if plan != row["config"]:
            raise ValueError("Export sources or runtime files changed; prepare a new export")
        manifest = _manifest(plan, identifier, row["created_at"])
        archive_path = staging / "model.zip"
        sources = {"model.pth": store.artifact_path(plan["checkpoint_path"])}
        sources.update(
            {
                f"parity/images/{item['frame_id']}.png": store.artifact_path(item["image_path"])
                for item in plan["frames"]
            }
        )
        total = sum(item["size"] for item in manifest["files"].values())
        copied = 0
        with zipfile.ZipFile(
            archive_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True
        ) as output:
            output.writestr("manifest.json", _canonical(manifest))
            for path, raw in {
                **_resources(),
                "parity/reference.json": _canonical(plan["reference"]),
            }.items():
                output.writestr(path, raw)
            for name, path in sources.items():
                digest, size = hashlib.sha256(), 0
                with (
                    path.open("rb") as source,
                    output.open(name, "w", force_zip64=True) as destination,
                ):
                    while block := source.read(1024**2):
                        if cancelled():
                            raise ExportCancelled()
                        size += len(block)
                        if size > manifest["files"][name]["size"]:
                            raise ValueError("Source file grew while the export was being copied")
                        digest.update(block)
                        destination.write(block)
                        copied += len(block)
                        progress(min(0.9, copied / total * 0.9), f"Copying {name}")
                if {"sha256": digest.hexdigest(), "size": size} != manifest["files"][name]:
                    raise ValueError("Source file changed while the export was being copied")
        # Re-read the completed ZIP; this never loads or interprets model tensors.
        read_bundle(archive_path, manifest)
        identity = _file_digest(archive_path, cancelled=cancelled)
        with store.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = _View(store.root, connection).get("model_exports", identifier)
            job = _View(store.root, connection).get("jobs", row["job_id"])
            if cancelled() or job["status"] not in {"queued", "running"} or job["cancel_requested"]:
                raise ExportCancelled()
            if current["path"] is not None or target.exists():
                raise ValueError("Another attempt already published this export")
            os.rename(staging, target)
            try:
                connection.execute(
                    "UPDATE model_exports SET path=?,manifest=?,manifest_sha256=?,archive_sha256=? "
                    "WHERE id=? AND path IS NULL",
                    (
                        f"model_exports/{identifier}/model.zip",
                        _canonical(manifest).decode(),
                        _digest(manifest),
                        identity["sha256"],
                        identifier,
                    ),
                )
                connection.execute(
                    "UPDATE jobs SET status='succeeded',progress=1,message=?,"
                    "finished_at=?,result=? "
                    "WHERE id=?",
                    (
                        "Standalone package copied; real execution is not validated",
                        now(),
                        json.dumps({"export_id": identifier, "published": True}),
                        row["job_id"],
                    ),
                )
                connection.commit()
                published = True
            except BaseException:
                shutil.rmtree(target, ignore_errors=True)
                raise
        return {"export_id": identifier, "published": True}
    except ExportCancelled:
        return {"export_id": identifier, "published": False, "cancelled": True}
    finally:
        if not published:
            shutil.rmtree(staging, ignore_errors=True)


def _json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON keys are not supported")
            result[key] = value
        return result

    try:
        return json.loads(
            raw,
            object_pairs_hook=pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Non-finite JSON")),
        )
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("Expected finite JSON data") from exc


def read_bundle(path, expected_manifest=None):
    """Bounded ZIP inspection without extraction, execution or checkpoint deserialization."""
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= MAX_BUNDLE_BYTES:
        raise ValueError("Standalone package is missing or exceeds its size limit")
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            if len(infos) > 14 or len({item.filename for item in infos}) != len(infos):
                raise ValueError("Standalone package has duplicate or excessive entries")
            manifest_info = archive.getinfo("manifest.json")
            if not 0 < manifest_info.file_size <= 2 * 1024**2:
                raise ValueError("Standalone manifest exceeds its size limit")
            manifest = _json(archive.read(manifest_info))
            _runtime().validate_manifest(manifest)
            if expected_manifest is not None and manifest != expected_manifest:
                raise ValueError("Standalone manifest differs from the saved export")
            expected = {"manifest.json", *manifest["files"]}
            if {item.filename for item in infos} != expected:
                raise ValueError("Standalone package file inventory is inconsistent")
            reference = None
            for info in infos:
                if (
                    info.is_dir()
                    or info.flag_bits & 1
                    or info.compress_type != zipfile.ZIP_STORED
                    or (info.external_attr >> 16) & 0o170000 not in {0, 0o100000}
                ):
                    raise ValueError("Unsupported standalone ZIP entry")
                if info.filename == "manifest.json":
                    continue
                identity = manifest["files"][info.filename]
                if info.file_size != identity["size"]:
                    raise ValueError("Standalone file size is inconsistent")
                digest, chunks = hashlib.sha256(), []
                with archive.open(info) as stream:
                    while block := stream.read(1024**2):
                        digest.update(block)
                        if info.filename == "parity/reference.json":
                            chunks.append(block)
                if digest.hexdigest() != identity["sha256"]:
                    raise ValueError("Standalone file checksum is inconsistent")
                if info.filename == "parity/reference.json":
                    reference = _json(b"".join(chunks))
            _runtime().validate_reference(manifest, reference)
            return manifest, reference
    except (zipfile.BadZipFile, KeyError, RuntimeError, OverflowError) as exc:
        raise ValueError("Standalone ZIP is invalid or incomplete") from exc


def download_path(store, identifier):
    row = store.get("model_exports", identifier)
    if row is None:
        raise KeyError(identifier)
    if not row["path"]:
        raise ValueError("This package has not been published")
    if row["path"] != f"model_exports/{row['id']}/model.zip":
        raise ValueError("Invalid standalone package path")
    path = store.artifact_path(row["path"])
    if _file_digest(path)["sha256"] != row["archive_sha256"]:
        raise ValueError("Standalone package changed since publication")
    return path


def preview_measurement(store, identifier, payload):
    row = store.get("model_exports", identifier)
    if row is None:
        raise KeyError(identifier)
    if len(_canonical(payload)) > MAX_MEASUREMENT_BYTES:
        raise ValueError("Measurement JSON exceeds the 8 MiB limit")
    manifest, reference = read_bundle(download_path(store, identifier), row["manifest"])
    summary = _runtime().validate_measurement(manifest, reference, payload)
    return {"fingerprint": _digest(payload), "summary": summary}


def save_measurement(store, identifier, payload, expected_fingerprint):
    preview = preview_measurement(store, identifier, payload)
    if preview["fingerprint"] != expected_fingerprint:
        raise ValueError("Measurement changed; inspect a fresh preview")
    with store.connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        previous = _View(store.root, connection).list(
            "model_export_measurements", export_id=identifier, fingerprint=expected_fingerprint
        )
        if previous:
            return previous[0]
        measurement_id = new_id()
        connection.execute(
            "INSERT INTO model_export_measurements "
            "(id,export_id,payload,summary,fingerprint,created_at) VALUES (?,?,?,?,?,?)",
            (
                measurement_id,
                identifier,
                _canonical(payload).decode(),
                _canonical(preview["summary"]).decode(),
                expected_fingerprint,
                now(),
            ),
        )
    return store.get("model_export_measurements", measurement_id)


def validate_export_archive(row, *, connection, root):
    """Validate frozen source links and nested evidence from a read-only archive snapshot."""
    view = _View(root, connection)
    plan = _plan(view, **_options(row["config"]), frozen=row["config"])
    if (
        plan != row["config"]
        or plan["trained_model_id"] != row["trained_model_id"]
        or plan["evaluation_id"] != row["evaluation_id"]
        or plan["name"] != row["name"]
        or plan["request_id"] != row["request_id"]
    ):
        raise ValueError("Export source records disagree with the frozen copy plan")
    measurements = view.list("model_export_measurements", export_id=row["id"])
    if row["path"] is None:
        if measurements:
            raise ValueError("An unpublished export cannot have measurements")
        return
    manifest, reference = read_bundle(view.artifact_path(row["path"]), row["manifest"])
    if (
        manifest != _manifest(plan, row["id"], row["created_at"])
        or _digest(manifest) != row["manifest_sha256"]
        or reference != plan["reference"]
    ):
        raise ValueError("Published export differs from its frozen source plan")
    for item in measurements:
        if (
            len(_canonical(item["payload"])) > MAX_MEASUREMENT_BYTES
            or item["fingerprint"] != _digest(item["payload"])
            or item["summary"]
            != _runtime().validate_measurement(manifest, reference, item["payload"])
        ):
            raise ValueError("Saved external measurement evidence is inconsistent")
