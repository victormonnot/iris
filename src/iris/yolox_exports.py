"""Versioned YOLOX ONNX conversion integrated with the existing export job lifecycle."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

from iris import yolox_export_runner as runner
from iris.datasets import load_manifest
from iris.evaluation import evaluation_detail
from iris.evaluation_analysis import _analyze
from iris.model_taxonomy import class_contract
from iris.projects import record_project
from iris.store import now

PLAN_FORMAT = "iris-yolox-export-plan-v1"
WARNINGS = [
    "This job converts the trained YOLOX checkpoint to ONNX and runs a CPU conversion check. "
    "It does not train, contact a provider or change another application's model.",
    "The package includes selected reference images and saved predictions. Review these "
    "images before sharing the package.",
    "The standalone destination is OpenCV CPU. GPU training is supported independently; "
    "this profile does not provide TensorRT or validate an embedded board.",
    "Conversion uses a declared numerical tolerance on raw outputs. Exact parity of saved "
    "PyTorch predictions and destination predictions is measured separately and can fail.",
]
README = b"""# IRIS YOLOX ONNX bundle

This generic bundle contains a custom YOLOX-Nano ONNX graph, an explicit class
mapping and input/output contract, reference images and a standalone CPU runner.
Install requirements.txt separately. No IRIS installation, cloud service or
automatic model download is needed. From the extracted directory:

    python run.py inspect
    python run.py predict image.png --output ../prediction.json
    python run.py measure --repeats 3 --output ../measurement.json

The output graph uses raw YOLOX grid coordinates (decode once), sigmoid objectness
and sigmoid class probabilities. Input is BGR float32, 0..255, top-left letterbox
with 114 padding and OpenCV linear resize, fixed 1x3x416x416. Classes are explicitly
zero-based in the manifest; downstream consumers must not guess a person slot.
This runner reports detections with IRIS's 1-based native labels and category IDs.

Conversion checks raw PyTorch/ONNX outputs on the bundled images against the
fixed rtol=.001,atol=.001 tolerance. This is separate from exact saved-prediction
parity, detection quality and tracking quality. measure exits 3 when exact parity
fails, retaining the report. Import that report into IRIS to retain the result.
The measurements describe the executing host, not a drone or another computer.
Hashes check integrity, not publisher authenticity. Only load trusted bundles.
"""


def resources():
    return {
        "run.py": Path(runner.__file__).read_bytes(),
        "README.md": README,
        "requirements.txt": b"opencv-python-headless==4.14.0.94\nnumpy==2.5.3\nPillow==12.3.0\n",
    }


def source(store, model_id, evaluation_id):
    from iris.yolox_spec import INPUT_TRANSFORM, NATIVE_FILTERING

    model = store.get("trained_models", model_id)
    if model is None or model["architecture"] != "yolox_nano":
        raise ValueError("Choose an IRIS-trained YOLOX-Nano checkpoint")
    contract = class_contract(model["metadata"])
    detail = evaluation_detail(store, evaluation_id)
    if record_project(store, "trained_models", model) != record_project(
        store, "evaluations", detail
    ):
        raise ValueError("Checkpoint and evaluation must belong to the same project")
    _analyze(detail, evaluation_id)
    runs = [
        item
        for item in detail["models"]
        if item["model_id"] == model_id and item["variant"] == "full"
    ]
    if len(runs) != 1 or runs[0]["metrics"] is None:
        raise ValueError("Choose a completed full-image YOLOX evaluation")
    metadata = runs[0]["metadata"]
    if (
        metadata.get("weight_sha256") != model["weight_sha256"]
        or metadata.get("architecture") != "yolox_nano"
        or metadata.get("precision") != "float32"
        or class_contract(metadata) != contract
        or metadata.get("input_transform") != INPUT_TRANSFORM
        or metadata.get("native_filtering") != NATIVE_FILTERING
    ):
        raise ValueError("Saved YOLOX evaluation differs from the export recipe")
    return model, detail, runs[0], load_manifest(store, detail["dataset_id"])


def plan(
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
    from iris import model_exports as common

    if target_device != "cpu":
        raise ValueError("YOLOX ONNX currently exports for OpenCV CPU; training may use CUDA")
    if (
        not isinstance(name, str)
        or not 1 <= len(name.strip()) <= 160
        or not isinstance(request_id, str)
        or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", request_id)
        or not isinstance(frame_ids, list)
        or not 1 <= len(frame_ids) <= 8
        or any(not isinstance(item, str) for item in frame_ids)
        or len(set(frame_ids)) != len(frame_ids)
    ):
        raise ValueError("Choose a name and 1–8 distinct reference images")
    model, detail, run, dataset = source(store, trained_model_id, evaluation_id)
    available = {
        item["frame_id"]: item for item in dataset["frames"] if item["split"] == detail["split"]
    }
    if not set(frame_ids) <= available.keys():
        raise ValueError("Reference images must belong to the selected evaluation split")
    weights = (
        common._file_digest(store.artifact_path(model["path"]), maximum=1024**3)
        if frozen is None
        else {key: frozen["model"][key] for key in ("size", "sha256")}
    )
    if weights["sha256"] != model["weight_sha256"]:
        raise ValueError("Trained YOLOX checkpoint bytes changed")
    predictions = {
        item["frame_id"]: item
        for item in detail["predictions"]
        if item["evaluation_model_id"] == run["id"]
    }
    frames, references = [], []
    frozen_frames = {item["frame_id"]: item for item in frozen["frames"]} if frozen else {}
    if frozen is not None and (
        set(frozen_frames) != set(frame_ids) or len(frozen["frames"]) != len(frame_ids)
    ):
        raise ValueError("Frozen ONNX reference inventory differs from its selection")
    for frame_id in frame_ids:
        frame, prediction = available[frame_id], predictions[frame_id]
        identity = (
            common._file_digest(store.artifact_path(frame["image_path"]), maximum=32 * 1024**2)
            if frozen is None
            else {
                "sha256": frozen_frames[frame_id]["image_file_sha256"],
                "size": frozen_frames[frame_id]["size"],
            }
        )
        if identity["sha256"] != frame["image_file_sha256"]:
            raise ValueError("Reference image bytes changed")
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
    contract = class_contract(model["metadata"])
    classes = [
        {
            "index": contract["class_mapping"][item["id"]] - 1,
            "id": item["id"],
            "name": item["name"],
            "category_id": contract["output_class_mapping"][item["id"]],
        }
        for item in contract["taxonomy"]["classes"]
    ]
    classes.sort(key=lambda item: item["index"])
    files = (
        {
            key: {"size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
            for key, raw in resources().items()
        }
        if frozen is None
        else frozen["resources"]
    )
    return {
        "format": PLAN_FORMAT,
        "request_id": request_id,
        "name": name.strip(),
        "trained_model_id": trained_model_id,
        "evaluation_id": evaluation_id,
        "frame_ids": frame_ids,
        "checkpoint_path": model["path"],
        "frames": frames,
        "resources": files,
        "classes": classes,
        "taxonomy_id": contract["taxonomy_id"],
        "model": {
            "id": model["id"],
            "name": model["name"],
            "architecture": "yolox_nano",
            **weights,
            "class_contract": contract,
        },
        "source": {
            "model_id": model["id"],
            "training_id": model["training_id"],
            "checkpoint_sha256": weights["sha256"],
            "evaluation_id": evaluation_id,
            "evaluation_model_id": run["id"],
            "dataset_id": detail["dataset_id"],
            "dataset_manifest_sha256": detail["config"]["dataset_manifest_sha256"],
            "reference_device": run["metadata"]["device"],
        },
        "evaluation_metadata_sha256": runner.digest(run["metadata"]),
        "reference": {
            "format": "iris-export-reference-v1",
            "model_id": model["id"],
            "weight_sha256": weights["sha256"],
            "frames": references,
        },
        "profile": {
            "id": runner.FORMAT,
            "device": "cpu",
            "runtime": {"opencv": "4.14.0"},
            "conversion": {"opset": 11, "raw_rtol": 0.001, "raw_atol": 0.001},
        },
    }


def _convert(store, plan, directory, progress, cancelled):
    import cv2
    import numpy as np
    import onnx
    import torch
    from PIL import Image, ImageOps

    from iris.models import TorchvisionDetector
    from iris.yolox_runtime import preprocess

    from .model_exports import ExportCancelled

    detector = TorchvisionDetector(store.root, plan["trained_model_id"], device="cpu")
    model = detector.model.eval()
    model.head.decode_in_inference = False
    output = directory / "model.onnx"
    progress(0.12, "Converting YOLOX to raw-grid ONNX on CPU")
    sample = torch.full((1, 3, 416, 416), 114, dtype=torch.float32)
    with torch.inference_mode():
        torch.onnx.export(
            model,
            sample,
            output,
            input_names=["images"],
            output_names=["output"],
            opset_version=11,
            dynamo=False,
            external_data=False,
        )
    graph = onnx.load(output, load_external_data=False)
    onnx.checker.check_model(graph)

    def shape(value):
        return [item.dim_value for item in value.type.tensor_type.shape.dim]

    if (
        len(graph.graph.input) != 1
        or len(graph.graph.output) != 1
        or shape(graph.graph.input[0]) != [1, 3, 416, 416]
        or shape(graph.graph.output[0]) != [1, 3549, 5 + len(plan["classes"])]
        or any(item.data_location == onnx.TensorProto.EXTERNAL for item in graph.graph.initializer)
    ):
        raise ValueError("Converted graph has an unsupported shape or external tensor data")
    cv2.setNumThreads(2)
    net = cv2.dnn.readNetFromONNX(str(output))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    checks = []
    for index, frame in enumerate(plan["frames"]):
        if cancelled():
            raise ExportCancelled()
        with Image.open(store.artifact_path(frame["image_path"])) as image:
            tensor, _ = preprocess(ImageOps.exif_transpose(image).convert("RGB"), "cpu")
        with torch.inference_mode():
            expected = model(tensor).cpu().numpy()
        net.setInput(tensor.numpy())
        actual = net.forward("output")
        passed = (
            actual.shape == expected.shape
            and np.isfinite(actual).all()
            and np.isfinite(expected).all()
            and np.allclose(actual, expected, rtol=0.001, atol=0.001)
        )
        checks.append(
            {
                "frame_id": frame["frame_id"],
                "passed": bool(passed),
                "max_absolute_error": float(np.max(np.abs(actual - expected))),
            }
        )
        if not passed:
            raise ValueError("ONNX conversion differs from PyTorch beyond the frozen tolerance")
        progress(0.25 + 0.4 * (index + 1) / len(plan["frames"]), "Checking raw ONNX outputs")
    return {
        "raw_output_equivalence": True,
        "rtol": 0.001,
        "atol": 0.001,
        "frames": checks,
        "torch": torch.__version__,
        "onnx": onnx.__version__,
        "opencv": cv2.__version__,
        "device": "cpu",
        "exact_saved_prediction_parity": "not_run",
    }


def manifest(plan, identifier, created_at, files, validation):
    return {
        "format": runner.FORMAT,
        "id": identifier,
        "name": plan["name"],
        "created_at": created_at,
        "architecture": "yolox_nano",
        "input": {
            "name": "images",
            "shape": [1, 3, 416, 416],
            "dtype": "float32",
            "color": "BGR",
            "range": [0, 255],
            "letterbox": {"alignment": "top_left", "value": 114, "interpolation": "opencv_linear"},
        },
        "output": {
            "name": "output",
            "shape": [1, 3549, 5 + len(plan["classes"])],
            "encoding": "yolox_raw_grid",
            "strides": [8, 16, 32],
        },
        "classes": plan["classes"],
        "taxonomy_id": plan["taxonomy_id"],
        "model": {
            "path": "model.onnx",
            "size_bytes": files["model.onnx"]["size"],
            "sha256": files["model.onnx"]["sha256"],
        },
        "source": plan["source"],
        "profile": plan["profile"],
        "files": files,
        "validation": validation,
    }


def run_export(store, identifier, progress, cancelled):
    from iris import model_exports as common

    row = store.get("model_exports", identifier)
    if row is None or row["path"]:
        raise ValueError("Export is missing or already published")
    directory = store.root / "model_exports"
    directory.mkdir(exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".building-", dir=directory))
    target, published = directory / identifier, False
    try:
        with store.connect() as connection:
            connection.execute("BEGIN")
            frozen = plan(common._View(store.root, connection), **common._options(row["config"]))
        if frozen != row["config"]:
            raise ValueError("Export sources changed; create a new preview")
        if cancelled():
            raise common.ExportCancelled()
        bundle = staging / "bundle"
        bundle.mkdir()
        validation = _convert(store, frozen, bundle, progress, cancelled)
        for name, raw in {
            **resources(),
            "parity/reference.json": runner.canonical(frozen["reference"]),
        }.items():
            path = bundle / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        for frame in frozen["frames"]:
            path = bundle / "parity/images" / f"{frame['frame_id']}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(store.artifact_path(frame["image_path"]), path)
            if common._file_digest(path)["sha256"] != frame["image_file_sha256"]:
                raise ValueError("Reference image changed while packaging")
        files = {
            path.relative_to(bundle).as_posix(): common._file_digest(path)
            for path in bundle.rglob("*")
            if path.is_file()
        }
        contract = manifest(frozen, identifier, row["created_at"], files, validation)
        runner.validate_manifest(contract)
        (bundle / "manifest.json").write_bytes(runner.canonical(contract))
        runner.inspect_bundle(bundle)
        archive = staging / "model.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as output:
            for path in sorted(bundle.rglob("*")):
                if path.is_file():
                    output.write(path, path.relative_to(bundle).as_posix())
        common.read_bundle(archive, contract)
        identity = common._file_digest(archive)
        shutil.rmtree(bundle)
        with store.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            view = common._View(store.root, connection)
            current, job = view.get("model_exports", identifier), view.get("jobs", row["job_id"])
            if cancelled() or job["cancel_requested"] or job["status"] not in {"running", "queued"}:
                raise common.ExportCancelled()
            if current["path"] or target.exists():
                raise ValueError("Another attempt published this export")
            os.rename(staging, target)
            try:
                connection.execute(
                    "UPDATE model_exports SET path=?,manifest=?,manifest_sha256=?,archive_sha256=? "
                    "WHERE id=? AND path IS NULL",
                    (
                        f"model_exports/{identifier}/model.zip",
                        runner.canonical(contract).decode(),
                        runner.digest(contract),
                        identity["sha256"],
                        identifier,
                    ),
                )
                connection.execute(
                    "UPDATE jobs SET status='succeeded',progress=1,message=?,"
                    "finished_at=?,result=? "
                    "WHERE id=?",
                    (
                        "ONNX conversion checked on CPU; "
                        "target quality and exact parity remain separate",
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
    except common.ExportCancelled:
        return {"export_id": identifier, "published": False, "cancelled": True}
    finally:
        if not published:
            shutil.rmtree(staging, ignore_errors=True)


def validate_reference(manifest, reference):
    return runner.validate_reference(manifest, reference)


def validate_archive(row, *, connection, root):
    from iris import model_exports as common

    view = common._View(root, connection)
    frozen = plan(view, **common._options(row["config"]), frozen=row["config"])
    if frozen != row["config"] or any(
        row[key] != frozen[key]
        for key in ("trained_model_id", "evaluation_id", "name", "request_id")
    ):
        raise ValueError("ONNX export source records differ from its frozen plan")
    measurements = view.list("model_export_measurements", export_id=row["id"])
    if not row["path"]:
        if measurements:
            raise ValueError("An unpublished ONNX export cannot have measurements")
        return
    contract, reference = common.read_bundle(view.artifact_path(row["path"]), row["manifest"])
    expected = manifest(
        frozen, row["id"], row["created_at"], contract["files"], contract["validation"]
    )
    if (
        contract != expected
        or runner.digest(contract) != row["manifest_sha256"]
        or reference != frozen["reference"]
    ):
        raise ValueError("Published ONNX export differs from its source plan")
    if any(contract["files"].get(key) != value for key, value in frozen["resources"].items()):
        raise ValueError("Published ONNX runtime differs from its frozen plan")
    for item in measurements:
        if item["fingerprint"] != runner.digest(item["payload"]) or item[
            "summary"
        ] != runner.validate_measurement(contract, reference, item["payload"]):
            raise ValueError("Saved ONNX measurement is inconsistent")
