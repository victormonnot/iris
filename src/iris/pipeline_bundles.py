"""Copy frozen detector/tracker policies into portable experimental ZIP packages.

Packaging validates bytes and contracts without loading weights, importing a
native tracker, executing inference or changing the source project's settings.
"""

import hashlib
import os
import re
import shutil
import tempfile
import zipfile
from copy import deepcopy
from pathlib import Path

from iris import __version__, models
from iris.pipeline_bundle_contracts import (
    INTERFACE,
    VALIDATION,
    canonical,
    canonicalize_request,
    detector_contract,
    digest,
    inspect_bundle,
    license_contract,
    required_paths,
    validate_manifest,
)
from iris.pipeline_bundle_runtime_contracts import (
    FORMAT_V2,
    RUNTIME_MODULES,
    TRACKER_FILES,
    YOLOX_FILES,
    portable_tracking_bytes,
    requirements_bytes,
    runtime_descriptor,
)
from iris.prediction_taxonomy import output_contract
from iris.store import DEFAULT_PROJECT_ID, _encode, new_id, now
from iris.temporal import _insert, _row
from iris.tracking_selections import _checked_selection, _source
from iris.tracking_selections import catalogue as selection_catalogue
from iris.tracking_selections import public_job as selection_public_job

KIND = "pipeline_bundle"
MAX_CHECKPOINT_BYTES = 1024**3
LIMITATIONS = [
    "Experimental package: pipeline execution, parity and independent quality are not qualified.",
    "Packaging copies verified native checkpoint bytes; it does not deserialize or run them.",
    "The target CPU/CUDA family is separate from the historical detector runtime device.",
    "The bundle includes a standalone native runtime; packaging does not execute it.",
    "No source images, reference identities, selected object IDs "
    "or ground-truth boxes are included.",
    "Code licences do not establish rights to model weights or training data.",
]
README = b"""# IRIS experimental pipeline bundle

This package freezes native detector weights, their complete preprocessing and
output mapping, a tracker profile and optional guarded selected-object policy.
It contains no source images, selected person identities or annotation boxes.

This format milestone does not include a detector/tracker execution runtime.
Creating or inspecting this bundle does not run, convert, deserialize or qualify
its checkpoint. CPU/CUDA is a requested target family, not a hardware measurement.

Inspect the ZIP with IRIS (`iris pipeline inspect /path/to/pipeline.zip`), or copy
inspect.py and the iris_bundle directory into an empty directory and run:

    python inspect.py /path/to/pipeline.zip

The inspector uses only Python's standard library and never executes files from
the ZIP. Only execute copied scripts from sources you trust. Hashes establish
integrity, not publisher authenticity. The manifest records experimental status,
historical runtime declarations, exact input/output assumptions and limitations.

Review licenses/NOTICE.txt and the detector and tracker licence texts. Code
licences do not imply rights to checkpoints or their training data. Nothing
installs dependencies, downloads files or applies the policy to another project.
"""
INSPECT = b"""import argparse
import json
from pathlib import Path
from iris_bundle.pipeline_bundle_contracts import inspect_bundle

parser = argparse.ArgumentParser(description='Inspect an experimental IRIS pipeline ZIP')
parser.add_argument('bundle', type=Path)
args = parser.parse_args()
try:
    print(json.dumps(inspect_bundle(args.bundle), indent=2))
except (OSError, ValueError, KeyError, TypeError) as exc:
    parser.exit(1, f'Pipeline inspection failed: {exc}\\n')
"""
NOTICE = b"""IRIS experimental pipeline format

The original detector and tracker licence notices are retained in this folder.
YOLOX detector bundles also retain the upstream detector NOTICE. Detector code
licence terms do not automatically grant rights to checkpoints or training data;
consult the weight terms links recorded in manifest.json.

The IRIS contract, inspection and runtime modules are covered by the MIT licence
below. Third-party code retains its own licence notices. Nothing in this package
represents a claim of independent quality, device compatibility, physical identity
certainty or publisher authentication.

IRIS code licence
=================

"""


RUNTIME_README = b"""# IRIS experimental native pipeline runtime

Version two includes native detector weights, the frozen recipe, native trackers
and optional guarded selected-object policy. It contains no source images,
selected person identities or ground-truth boxes.

Inspect and extract using trusted installed IRIS code:

    iris pipeline inspect pipeline.zip
    iris pipeline extract pipeline.zip --to ./new-runtime

Extraction creates a new directory and never executes packaged code. inspect_bundle.py
also checks a ZIP using only Python's standard library. Hashes verify integrity,
not publisher authenticity. Execute code only from sources you trust.

Prepare Python 3.12 or 3.13 on Linux with the exact requirements.txt packages and
Torch/Torchvision builds appropriate to the frozen CPU/CUDA target. No command
automatically installs dependencies or downloads files.

    python -B run.py --bundle . check-runtime
    python -B run.py --bundle . frames /data/inputs.json --output /data/frame-results.json
    python -B run.py --bundle . video /data/input.mp4 --output /data/video-results.json
    python -B run.py --bundle . compare /data/ref.json /data/actual.json --output /data/parity.json

Keep all inputs and outputs outside the extracted bundle. Output files must be
new. A frame input manifest has this shape; image paths are relative to its own
directory, cannot escape it and must name regular files without symbolic links:

    {"schema":"iris-pipeline-input-v1","sequence_id":"flight-01",
     "clock_kind":"provided","frames":[
       {"frame_id":"frame-0000","frame_index":0,"timestamp_seconds":0.0,
        "path":"frames/0000.png","file_sha256":"REPLACE_WITH_IMAGE_SHA256",
        "input_size":{"width":640,"height":480}}]}

The hash is the complete 64-character SHA256 of the encoded image file. Image
dimensions describe RGB pixels after EXIF orientation. Frame IDs must be unique;
source indices and finite nonnegative timestamps must strictly increase. With
clock_kind "unknown", all timestamp_seconds must instead be null. Optional
select_detection_index and release fields express explicit per-frame decisions.

For video, --events /data/events.json accepts zero-based decoded frame indices:

    {"schema":"iris-pipeline-events-v1","events":[
       {"frame_index":0,"select_detection_index":0},
       {"frame_index":120,"release":true}]}

Video timestamps default to index / nominal FPS, not certified capture time;
--clock unknown emits null timestamps. --max-frames 500 processes a bounded
prefix. Selection addresses a confirmed measured detection's detection_index,
not a track number. No target is selected automatically. Pipelines are bounded
to 10000 updates per reset; reset clears identities and selected-object state.

Use --device cuda:0 only for a CUDA-target bundle; there is no CPU fallback.
Historical detector device/runtime metadata is separate from the target runtime.
See example.py for a frame-by-frame consumer. Direct imports and example.py must
run with python -B or PYTHONDONTWRITEBYTECODE=1 set before importing the package.
All __pycache__ directories and .pyc files are rejected: validating source hashes
cannot authenticate cached bytecode. run.py disables bytecode before its imports.
Reset starts a fresh sequence,
predictions remain separate from measured observations, and selecting an object
requires an explicit measured box. A track number does not prove identity.

Runtime/parity reports are separate evidence and never promote this manifest to
qualified status. Performance and independent quality require their own tests.
Review licence texts and licenses/NOTICE.txt, which includes the MIT licence for
IRIS code. Code licences do not establish checkpoint or training-data rights.
"""
RUNTIME_LAUNCHER = b"""import sys
sys.dont_write_bytecode = True
from iris_bundle.pipeline_runner import main

if __name__ == '__main__':
    raise SystemExit(main())
"""
RUNTIME_INIT = b"""import sys
sys.dont_write_bytecode = True
"""


def _resources(detector, profile, *, bundle_format=FORMAT_V2):
    base = Path(__file__).parent
    architecture = detector["architecture"]
    result = {"README.md": README, "inspect.py": INSPECT, "iris_bundle/__init__.py": b""}
    for name in (
        "pipeline_bundle_contracts.py",
        "pipeline_detector_contracts.py",
        "tracking_contracts.py",
        "tracking_selection_contracts.py",
    ):
        result[f"iris_bundle/{name}"] = (base / name).read_bytes()
    result["licenses/tracker-LICENSE"] = (
        base / "_vendor" / profile["algorithm"] / "LICENSE"
    ).read_bytes()
    result["licenses/detector-LICENSE"] = (
        base
        / "_vendor"
        / ("yolox/LICENSE" if architecture == "yolox_nano" else "torchvision-LICENSE")
    ).read_bytes()
    result["licenses/NOTICE.txt"] = NOTICE + (base / "LICENSE.txt").read_bytes()
    if architecture == "yolox_nano":
        result["licenses/detector-NOTICE"] = (base / "_vendor/yolox/NOTICE").read_bytes()
    if bundle_format == FORMAT_V2:
        result["README.md"] = RUNTIME_README
        result.pop("inspect.py")
        result["inspect_bundle.py"] = b"import sys\nsys.dont_write_bytecode = True\n" + INSPECT
        result["iris_bundle/__init__.py"] = RUNTIME_INIT
        result["run.py"] = RUNTIME_LAUNCHER
        result["example.py"] = (base / "pipeline_example.py").read_bytes()
        result["requirements.txt"] = requirements_bytes()
        for name in RUNTIME_MODULES:
            raw = (base / name).read_bytes()
            result[f"iris_bundle/{name}"] = (
                portable_tracking_bytes(raw) if name == "tracking.py" else raw
            )
        result["provenance/tracking-original.py.txt"] = (base / "tracking.py").read_bytes()
        result["iris_bundle/_vendor/__init__.py"] = b""
        result["iris_bundle/_vendor/tracking-provenance.json"] = (
            base / "_vendor/tracking-provenance.json"
        ).read_bytes()
        for algorithm, names in TRACKER_FILES.items():
            for name in names:
                result[f"iris_bundle/_vendor/{algorithm}/{name}"] = (
                    base / "_vendor" / algorithm / name
                ).read_bytes()
        if architecture == "yolox_nano":
            for name in YOLOX_FILES:
                result[f"iris_bundle/_vendor/yolox/{name}"] = (
                    base / "_vendor/yolox" / name
                ).read_bytes()
    if set(result) | {"detector/model.pth"} != required_paths(architecture, format=bundle_format):
        raise ValueError("Pipeline package resources do not match the versioned inventory")
    return result


def _facts(raw):
    return {"sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw)}


def _checkpoint_path(root, relative):
    base = Path(root).resolve()
    path = base / relative
    if not path.resolve().is_relative_to(base) or any(
        item.is_symlink() for item in (path, *path.parents) if item != base.parent
    ):
        raise ValueError("Pipeline checkpoint path is unsafe")
    return path


def _file_facts(path, checkpoint=None):
    if (
        path.is_symlink()
        or not path.is_file()
        or not 0 < path.stat().st_size <= MAX_CHECKPOINT_BYTES
    ):
        raise ValueError("Pipeline checkpoint is missing, unsafe or exceeds 1 GiB")
    result, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        while block := stream.read(1024**2):
            if checkpoint:
                checkpoint()
            size += len(block)
            if size > MAX_CHECKPOINT_BYTES:
                raise ValueError("Pipeline checkpoint exceeded 1 GiB")
            result.update(block)
    return {"sha256": result.hexdigest(), "size": size}


def _model(conn, config, project_id):
    if config["origin"] == "official":
        spec = models.get_spec(config["model_id"])
        path = "models/" + spec["weight_filename"]
    else:
        row = _row(conn, "trained_models", config["model_id"])
        owner = conn.execute(
            "SELECT d.project_id FROM training_runs t "
            "JOIN dataset_versions d ON d.id=t.dataset_id WHERE t.id=?",
            (row["training_id"],),
        ).fetchone()
        if owner is None or owner["project_id"] != project_id:
            raise ValueError("The trained checkpoint must belong to the source project")
        spec = models._trained_spec(row)
        path = row["path"]
        parts = Path(path).parts
        if len(parts) != 3 or parts[:2] != ("models", "trained") or not parts[-1].endswith(".pth"):
            raise ValueError("Trained checkpoint path is outside its managed directory")
        if spec["weight_sha256"] != config["weight_sha256"]:
            raise ValueError("Trained checkpoint hash differs from the frozen detector")
    if any(spec[key] != config[key] for key in ("architecture", "origin", "classes")) or digest(
        output_contract(spec)
    ) != digest(config["class_contract"]):
        raise ValueError("Detector architecture or output classes differ from the frozen model")
    return spec, path


def _evidence(conn, payload, *, store=None, project_id=None):
    request = canonicalize_request(payload)
    source = _source(conn, request["source"], store=store, project_id=project_id)
    config = source["replay"]["cache"]["config"]["detector"]
    if config["inference"] != {"mode": "full"}:
        raise ValueError(
            "This bundle format supports full-image recipes only; tiled recipes are not converted"
        )
    spec, path = _model(conn, config, source["sequence"]["project_id"])
    selected = None
    if request["selection_id"] is not None:
        selection, _ = _checked_selection(
            conn, request["selection_id"], store=store, project_id=source["sequence"]["project_id"]
        )
        if (
            selection["status"] != "succeeded"
            or selection["params"]["request"]["source"] != request["source"]
        ):
            raise ValueError(
                "Choose a completed selected-object scenario from the exact source and profile"
            )
        policy = selection["params"]["request"]["policy"]
        selected = {
            "algorithm": "guarded_geometry",
            "policy": deepcopy(policy),
            "policy_sha256": digest(policy),
            "source_job_id": selection["id"],
            "source_report_sha256": digest(selection["result"]),
        }
    return {
        **source,
        "request": request,
        "spec": spec,
        "checkpoint_path": path,
        "selection": selected,
    }


def _manifest(evidence, identifier, created_at, *, files, producer, bundle_format=FORMAT_V2):
    replay = evidence["replay"]
    binding = evidence["source_binding"]
    config = replay["cache"]["config"]["detector"]
    source = {
        key: deepcopy(binding[key])
        for key in (
            "source_report_sha256",
            "sequence_sha256",
            "cache_fingerprint",
            "result_sha256",
            "replay_sha256",
            "first_pass_semantic_sha256",
            "inherited_dataset",
        )
    }
    source["descriptor"] = deepcopy(evidence["request"]["source"])
    source["repeatability"] = replay["repeatability"]["status"]
    result = {
        "format": bundle_format,
        "id": identifier,
        "name": evidence["request"]["name"],
        "created_at": created_at,
        "producer": producer,
        "detector": detector_contract(
            config,
            evidence["request"]["target_device"],
            checkpoint_size=files["detector/model.pth"]["size"],
        ),
        "tracker": {
            "profile": deepcopy(replay["profile"]),
            "profile_sha256": replay["profile_sha256"],
            "runtime": deepcopy(replay["passes"][0]["metadata"]),
            "runtime_sha256": replay["passes"][0]["runtime_sha256"],
        },
        "selection": deepcopy(evidence["selection"]),
        "interface": deepcopy(INTERFACE),
        "source": source,
        "validation": deepcopy(VALIDATION),
        "licenses": license_contract(config),
        "files": files,
    }
    if bundle_format == FORMAT_V2:
        result["deployment_runtime"] = runtime_descriptor(files)
    return validate_manifest(result)


def _fingerprint(request, manifest, path):
    content = {key: value for key, value in manifest.items() if key not in ("id", "created_at")}
    return digest({"request": request, "manifest": content, "checkpoint_path": path})


def _prepare(
    conn,
    payload,
    store,
    project_id,
    *,
    identifier="preview",
    created_at="2000-01-01T00:00:00+00:00",
):
    evidence = _evidence(conn, payload, store=store, project_id=project_id)
    path = _checkpoint_path(store.root, evidence["checkpoint_path"])
    # Explicit file identity is checked again while copying, not just via the
    # catalog's cached official prefix/size verification.
    models._verified_digest(path, evidence["spec"])
    weights = _file_facts(path)
    if weights["sha256"] != evidence["replay"]["cache"]["config"]["detector"]["weight_sha256"]:
        raise ValueError("Checkpoint bytes differ from the full frozen detector SHA-256")
    resources = _resources(
        evidence["replay"]["cache"]["config"]["detector"], evidence["replay"]["profile"]
    )
    files = {key: _facts(raw) for key, raw in resources.items()}
    files["detector/model.pth"] = weights
    manifest = _manifest(
        evidence,
        identifier,
        created_at,
        files=files,
        producer={"name": "IRIS", "version": __version__},
    )
    return (
        evidence,
        manifest,
        _fingerprint(evidence["request"], manifest, evidence["checkpoint_path"]),
    )


def status():
    return {
        "format": FORMAT_V2,
        "supported_formats": ["iris-pipeline-bundle-v1", FORMAT_V2],
        "target_devices": ["cpu", "cuda"],
        "max_checkpoint_bytes": MAX_CHECKPOINT_BYTES,
        "validation": deepcopy(VALIDATION),
        "limitations": LIMITATIONS,
    }


def catalogue(store, *, project_id=DEFAULT_PROJECT_ID):
    rows = selection_catalogue(store, project_id=project_id)["sources"]
    with store.connect() as conn:
        conn.execute("BEGIN")
        selections = []
        for raw in conn.execute(
            "SELECT id FROM jobs WHERE kind='tracking_selection' "
            "AND status='succeeded' ORDER BY created_at,id"
        ):
            row = _row(conn, "jobs", raw["id"])
            seq = _row(conn, "temporal_sequences", row["params"]["sequence_id"])
            if seq["project_id"] == project_id:
                checked, _ = _checked_selection(conn, row["id"], store=store, project_id=project_id)
                selections.append(checked)
        result = []
        for row in rows:
            item = {
                **row,
                "available": False,
                "reason": None,
                "selections": [
                    {
                        "id": job["id"],
                        "name": job["params"]["name"],
                        "policy": job["params"]["request"]["policy"],
                    }
                    for job in selections
                    if job["params"]["request"]["source"] == row["source"]
                ],
            }
            try:
                evidence = _evidence(
                    conn,
                    {
                        "name": "Availability",
                        "source": row["source"],
                        "selection_id": None,
                        "target_device": "cpu",
                    },
                    store=store,
                    project_id=project_id,
                )
                config = evidence["replay"]["cache"]["config"]["detector"]
                item["detector"] = {
                    key: config[key]
                    for key in (
                        "architecture",
                        "model_id",
                        "classes",
                        "min_score",
                        "inference",
                        "device",
                    )
                }
                path = _checkpoint_path(store.root, evidence["checkpoint_path"])
                if path.is_symlink() or not path.is_file():
                    raise ValueError("The frozen detector checkpoint is not available locally")
                if models._verified_digest(path, evidence["spec"]) != config["weight_sha256"]:
                    raise ValueError("Local checkpoint differs from the frozen detector SHA-256")
                item["available"] = True
            except (KeyError, ValueError, OSError) as exc:
                item["reason"] = str(exc)
            result.append(item)
        return {"sources": result}


def preview_bundle(store, payload, *, project_id=DEFAULT_PROJECT_ID):
    with store.connect() as conn:
        conn.execute("BEGIN")
        evidence, manifest, fingerprint = _prepare(conn, payload, store, project_id)
        return {
            "request": evidence["request"],
            "fingerprint": fingerprint,
            "manifest": manifest,
            "weight": manifest["files"]["detector/model.pth"],
            "limitations": LIMITATIONS,
            "scope": "Copy frozen bytes and contracts only; no pipeline execution.",
        }


def _checked_bundle(conn, job_id, *, store=None, root=None, project_id=None):
    if not isinstance(job_id, str) or re.fullmatch("[0-9a-f]{32}", job_id) is None:
        raise ValueError("Pipeline package job ID is invalid")
    job = _row(conn, "jobs", job_id)
    if job["kind"] != KIND:
        raise KeyError(job_id)
    params = job["params"]
    if not isinstance(params, dict) or set(params) != {
        "name",
        "source_job_id",
        "sequence_id",
        "request",
        "fingerprint",
        "checkpoint_path",
        "manifest",
    }:
        raise ValueError("Pipeline package request is invalid")
    evidence = _evidence(conn, params["request"], store=store, project_id=project_id)
    saved = validate_manifest(params["manifest"])
    expected = _manifest(
        evidence,
        job["id"],
        job["created_at"],
        files=saved["files"],
        producer=saved["producer"],
        bundle_format=saved["format"],
    )
    if (
        digest(saved) != digest(expected)
        or digest(params["request"]) != digest(evidence["request"])
        or params["name"] != evidence["request"]["name"]
        or params["source_job_id"] != evidence["request"]["source"]["job_id"]
        or params["sequence_id"] != evidence["sequence"]["id"]
        or params["checkpoint_path"] != evidence["checkpoint_path"]
        or params["fingerprint"]
        != _fingerprint(evidence["request"], expected, evidence["checkpoint_path"])
    ):
        raise ValueError("Pipeline package changed its frozen source or manifest")
    if job["status"] == "succeeded":
        result = job["result"]
        if (
            job["cancel_requested"]
            or not isinstance(result, dict)
            or set(result)
            != {
                "complete",
                "path",
                "manifest",
                "manifest_sha256",
                "archive_sha256",
                "archive_bytes",
            }
        ):
            raise ValueError("Pipeline package publication is incomplete")
        relative = f"pipeline_bundles/{job_id}/pipeline.zip"
        if (
            result["complete"] is not True
            or result["path"] != relative
            or digest(result["manifest"]) != digest(expected)
        ):
            raise ValueError("Pipeline package publication changed its manifest or path")
        checked = inspect_bundle(
            (Path(root) if root is not None else store.root) / relative, expected_manifest=expected
        )
        if any(
            result[key] != checked[key]
            for key in ("manifest_sha256", "archive_sha256", "archive_bytes")
        ):
            raise ValueError("Pipeline package bytes no longer match their publication")
    elif job["result"] is not None:
        raise ValueError("An unfinished package cannot publish a complete bundle")
    return job, evidence


def public_job(job):
    job = selection_public_job(job)
    if job["kind"] != KIND:
        return job
    params = {key: value for key, value in job["params"].items() if key != "manifest"}
    result = job["result"]
    return {
        **job,
        "params": params,
        "result": None
        if result is None
        else {
            "complete": True,
            "archive_sha256": result["archive_sha256"],
            "archive_bytes": result["archive_bytes"],
        },
    }


def _public(job, *, detail):
    return {
        "id": job["id"],
        "name": job["params"]["name"],
        "sequence_id": job["params"]["sequence_id"],
        "source_job_id": job["params"]["source_job_id"],
        "job": public_job(job),
        "bundle": job["result"] if detail and job["status"] == "succeeded" else None,
    }


def create_bundle(store, jobs, payload, *, expected_fingerprint, project_id=DEFAULT_PROJECT_ID):
    identifier, created_at = new_id(), now()
    with jobs.guard, store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        evidence, manifest, fingerprint = _prepare(
            conn, payload, store, project_id, identifier=identifier, created_at=created_at
        )
        if fingerprint != expected_fingerprint:
            raise ValueError(
                "The pipeline preview changed; inspect a fresh preview before packaging"
            )
        _insert(
            conn,
            "jobs",
            {
                "id": identifier,
                "kind": KIND,
                "status": "queued",
                "params": {
                    "name": evidence["request"]["name"],
                    "source_job_id": evidence["request"]["source"]["job_id"],
                    "sequence_id": evidence["sequence"]["id"],
                    "request": evidence["request"],
                    "fingerprint": fingerprint,
                    "checkpoint_path": evidence["checkpoint_path"],
                    "manifest": manifest,
                },
                "result": None,
                "created_at": created_at,
                "message": "Waiting to package frozen bytes; execution remains untested",
            },
        )
    return get_bundle(store, identifier, project_id=project_id)


def get_bundle(store, job_id, *, project_id=DEFAULT_PROJECT_ID):
    with store.connect() as conn:
        conn.execute("BEGIN")
        job, _ = _checked_bundle(conn, job_id, store=store, project_id=project_id)
        return _public(job, detail=True)


def list_bundles(store, *, project_id=DEFAULT_PROJECT_ID):
    with store.connect() as conn:
        conn.execute("BEGIN")
        result = []
        for row in conn.execute(
            "SELECT j.id FROM jobs j JOIN temporal_sequences s "
            "ON s.id=json_extract(j.params,'$.sequence_id') "
            "WHERE j.kind=? AND s.project_id=? ORDER BY j.created_at,j.id",
            (KIND, project_id),
        ):
            job, _ = _checked_bundle(conn, row["id"], store=store, project_id=project_id)
            result.append(_public(job, detail=False))
        return result


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def run_pipeline_bundle(store, job_id, progress, cancelled):
    with store.connect() as conn:
        conn.execute("BEGIN")
        job, evidence = _checked_bundle(conn, job_id, store=store)
    if job["status"] != "running":
        raise ValueError("Pipeline package is no longer running")
    manifest = job["params"]["manifest"]

    def checkpoint():
        if cancelled():
            raise RuntimeError("Pipeline packaging cancelled before publication")

    checkpoint()
    resources = _resources(
        evidence["replay"]["cache"]["config"]["detector"],
        evidence["replay"]["profile"],
        bundle_format=manifest["format"],
    )
    if any(_facts(raw) != manifest["files"][name] for name, raw in resources.items()):
        raise ValueError("Packaging resources changed after preview; prepare a new package")
    weight_path = _checkpoint_path(store.root, evidence["checkpoint_path"])
    checkpoint()
    if _file_facts(weight_path, checkpoint) != manifest["files"]["detector/model.pth"]:
        raise ValueError("Checkpoint bytes changed after preview")
    directory = store.root / "pipeline_bundles"
    if directory.is_symlink():
        raise ValueError("Pipeline output directory must not be a symlink")
    directory.mkdir(exist_ok=True)
    _sync_directory(store.root)
    staging = Path(tempfile.mkdtemp(prefix=f".partial-{job_id}-", dir=directory))
    target = directory / job_id
    published = moved = False
    try:
        package = staging / "pipeline.zip"
        with zipfile.ZipFile(
            package, "w", compression=zipfile.ZIP_STORED, allowZip64=False
        ) as output:
            for name, raw in {"manifest.json": canonical(manifest), **resources}.items():
                checkpoint()
                output.writestr(name, raw)
            identity, size = hashlib.sha256(), 0
            with (
                weight_path.open("rb") as source,
                output.open("detector/model.pth", "w") as destination,
            ):
                while block := source.read(1024**2):
                    checkpoint()
                    size += len(block)
                    if size > manifest["files"]["detector/model.pth"]["size"]:
                        raise ValueError("Checkpoint grew during packaging")
                    identity.update(block)
                    destination.write(block)
                    progress(
                        0.85 * size / manifest["files"]["detector/model.pth"]["size"],
                        "Copying frozen native checkpoint bytes",
                    )
            if {"size": size, "sha256": identity.hexdigest()} != manifest["files"][
                "detector/model.pth"
            ]:
                raise ValueError("Checkpoint changed while packaging")
        with package.open("rb") as stream:
            os.fsync(stream.fileno())
        inspected = inspect_bundle(package, expected_manifest=manifest, checkpoint=checkpoint)
        checkpoint()
        result = {"complete": True, "path": f"pipeline_bundles/{job_id}/pipeline.zip", **inspected}
        with store.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            current, _ = _checked_bundle(conn, job_id, store=store)
            checkpoint()
            if (
                current["status"] != "running"
                or current["cancel_requested"]
                or current["params"] != job["params"]
                or target.exists()
                or target.is_symlink()
            ):
                raise ValueError("Pipeline package stopped or changed before publication")
            os.rename(staging, target)
            moved = True
            _sync_directory(target)
            _sync_directory(directory)
            values = _encode(
                {
                    "status": "succeeded",
                    "result": result,
                    "finished_at": now(),
                    "progress": 1.0,
                    "message": "Experimental pipeline package ready; execution remains untested",
                }
            )
            changed = conn.execute(
                f"UPDATE jobs SET {','.join(f'{key}=?' for key in values)} "
                "WHERE id=? AND status='running' AND cancel_requested=0",
                (*values.values(), job_id),
            ).rowcount
            if changed != 1:
                raise ValueError("Pipeline publication lost its uncancelled job claim")
            conn.commit()
            published = True
        return result
    finally:
        if not published:
            shutil.rmtree(staging, ignore_errors=True)
            if moved:
                shutil.rmtree(target, ignore_errors=True)


def cleanup_unpublished(store, job_id=None):
    """Remove only managed, unpublished job outputs after worker/restart recovery."""
    rows = [store.get("jobs", job_id)] if job_id else store.list("jobs", kind=KIND)
    base = store.root / "pipeline_bundles"
    if not base.exists() or base.is_symlink():
        return
    for row in rows:
        if (
            not row
            or not isinstance(row["id"], str)
            or re.fullmatch("[0-9a-f]{32}", row["id"]) is None
            or row["kind"] != KIND
            or row["status"] in {"queued", "running", "succeeded"}
        ):
            continue
        for path in [base / row["id"], *base.glob(f".partial-{row['id']}-*")]:
            if path.is_symlink():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)


def validate_pipeline_bundle_records(connection, root, require):
    try:
        for raw in connection.execute("SELECT id FROM jobs WHERE kind=?", (KIND,)):
            job, _ = _checked_bundle(connection, raw["id"], root=root)
            if job["status"] == "succeeded":
                row = job["result"]
                require(
                    row["path"],
                    (f"pipeline_bundles/{job['id']}/",),
                    row["archive_sha256"],
                    row["archive_bytes"],
                )
    except (KeyError, TypeError, IndexError, OSError, OverflowError, RecursionError) as exc:
        raise ValueError("Pipeline bundle evidence is invalid or missing") from exc
