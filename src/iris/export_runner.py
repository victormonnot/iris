"""Standalone, offline IRIS trained-model bundle runner (copied verbatim as run.py).

Inspection and evidence validation use only the Python standard library. The
optional inference dependencies are imported only for explicit execution.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import math
import os
import pickle
import platform
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path, PurePosixPath

ARCHITECTURE = "fasterrcnn_mobilenet_v3_large_320_fpn"
MAX_JSON_BYTES = 2 * 1024 * 1024
MAX_CHECKPOINT_BYTES = 1024 * 1024 * 1024
MAX_IMAGE_BYTES = 32 * 1024 * 1024
MAX_PIXELS = 64 * 1024 * 1024
PROFILE = {
    "id": "iris-torchvision-trained-cpu-v1",
    "architecture": ARCHITECTURE,
    "format": "pytorch_state_dict",
    "runtime": {
        "python": ">=3.12,<3.14",
        "torch": "2.10.0",
        "torchvision": "0.25.0",
        "pillow": "12.3.0",
    },
    "device": "cpu",
    "precision": "float32",
    "batch_size": 1,
    "threads": "min(4, os.cpu_count() or 1)",
    "builder": {
        "weights": None,
        "weights_backbone": None,
        "num_classes": "class_count + 1",
        "box_score_thresh": 0.001,
        "box_nms_thresh": 0.5,
        "box_detections_per_img": 100,
        "min_size": 320,
        "max_size": 640,
        "rpn_pre_nms_top_n_test": 150,
        "rpn_post_nms_top_n_test": 150,
        "rpn_score_thresh": 0.05,
        "rpn_nms_thresh": 0.7,
    },
    "backbone_normalization": {"type": "FrozenBatchNorm2d", "eps": 1e-5},
    "input": {
        "exif_transpose": True,
        "color": "RGB",
        "layout": "CHW",
        "tensor_range": [0, 1],
        "image_mean": [0.485, 0.456, 0.406],
        "image_std": [0.229, 0.224, 0.225],
        "size_divisible": 32,
        "fixed_size": None,
        "resize": "inside_torchvision_forward",
    },
    "output": {
        "coordinates": "xyxy pixels, original oriented image, exclusive right/bottom edge",
        "order": "native",
        "additional_filtering": False,
    },
    "parity": {"box_atol": 0, "score_atol": 0, "rtol": 0, "order": "exact"},
    "timing": {
        "warmup_passes": 1,
        "warmup_in_samples": False,
        "load_ms": (
            "Model construction, checkpoint loading and runtime setup; excludes bundle validation"
        ),
        "decode_ms": "Image file read, SHA-256 verification and Pillow decode",
        "preprocess_ms": "EXIF orientation, RGB conversion and float32 CHW tensor divided by 255",
        "inference_ms": (
            "Full forward including normalization, resize, proposals, "
            "NMS and coordinate restoration"
        ),
        "postprocess_ms": "Tensor transfer to CPU, output validation and JSON-compatible values",
        "total_ms": (
            "Preprocess plus inference plus postprocess; "
            "excludes decode, load, warmup and JSON writing"
        ),
    },
}
_BUILTIN_ID = "iris-objects-v1"
_BUILTIN_TAXONOMY = {
    "id": _BUILTIN_ID,
    "box_format": "xyxy_pixels",
    "classes": [
        {
            "id": "person",
            "name": "Person",
            "definition": (
                "A visible human, including a rider. Enclose the visible extent of each person; "
                "do not infer a box for a fully occluded person."
            ),
            "coco_id": 1,
        },
        {
            "id": "car",
            "name": "Car",
            "definition": (
                "A passenger car, including an SUV or passenger minivan. Exclude buses, trucks, "
                "motorcycles and bicycles. Enclose the visible extent of each car."
            ),
            "coco_id": 3,
        },
    ],
    "review_guidance": (
        "Review the whole image for missing objects and imprecise boxes. Only validate when "
        "all visible target objects are annotated. A validated empty image is an explicit "
        "negative example. Automatic proposals are never reference annotations by themselves."
    ),
}
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
_CLASS_ID = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z")
_TAXONOMY_ID = re.compile(r"taxonomy-[0-9a-f]{32}\Z")
_COCO_IDS = set(range(1, 91)) - {12, 26, 29, 30, 45, 66, 68, 69, 71, 83}
_REQUIRED_FILES = {"model.pth", "run.py", "requirements.txt", "README.md", "parity/reference.json"}


def canonical_bytes(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _object(value, keys, context):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"Invalid {context} fields")


def _text(value, context, maximum=256):
    if not isinstance(value, str) or not value or value != value.strip() or len(value) > maximum:
        raise ValueError(f"Invalid {context}")
    return value


def _identifier(value, context):
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError(f"Invalid {context}")


def _integer(value, context, minimum=0, maximum=1_000_000):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"Invalid {context}")


def _number(value, context, maximum=86_400_000):
    if type(value) not in (float, int) or not 0 <= value <= maximum or not math.isfinite(value):
        raise ValueError(f"Invalid {context}")


def _hash(value):
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise ValueError("Invalid SHA-256")


def _timestamp(value):
    _text(value, "timestamp", 64)
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("Invalid timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Timestamp must include its timezone")


def _class_contract(contract):
    _object(
        contract,
        {"taxonomy", "taxonomy_id", "class_mapping", "output_class_mapping"},
        "class contract",
    )
    taxonomy = contract["taxonomy"]
    if not isinstance(taxonomy, dict):
        raise ValueError("Invalid frozen taxonomy")
    builtin = contract["taxonomy_id"] == _BUILTIN_ID
    expected = {"id", "classes", "box_format", "review_guidance"}
    if not builtin:
        expected |= {"version", "parent_id", "created_at"}
    _object(taxonomy, expected, "taxonomy")
    if taxonomy["id"] != contract["taxonomy_id"] or taxonomy["box_format"] != "xyxy_pixels":
        raise ValueError("Invalid frozen taxonomy identity or coordinates")
    if not builtin:
        if not isinstance(taxonomy["id"], str) or not _TAXONOMY_ID.fullmatch(taxonomy["id"]):
            raise ValueError("Invalid custom taxonomy identity")
        _integer(taxonomy["version"], "taxonomy version", 2)
        parent = taxonomy["parent_id"]
        if (
            not isinstance(parent, str)
            or parent == taxonomy["id"]
            or not (parent == _BUILTIN_ID or _TAXONOMY_ID.fullmatch(parent))
            or (taxonomy["version"] == 2) != (parent == _BUILTIN_ID)
        ):
            raise ValueError("Invalid custom taxonomy parent")
        _timestamp(taxonomy["created_at"])
    _text(taxonomy["review_guidance"], "review guidance", 4000)
    if taxonomy["review_guidance"] != _BUILTIN_TAXONOMY["review_guidance"] or (
        builtin and taxonomy != _BUILTIN_TAXONOMY
    ):
        raise ValueError("Frozen taxonomy differs from its supported definition")
    classes = taxonomy["classes"]
    if not isinstance(classes, list) or not 1 <= len(classes) <= 100:
        raise ValueError("Expected 1–100 frozen classes")
    identifiers, coco_ids = [], set()
    for item in classes:
        if not isinstance(item, dict) or set(item) not in (
            {"id", "name", "definition"},
            {"id", "name", "definition", "coco_id"},
        ):
            raise ValueError("Invalid frozen class fields")
        identifier = item["id"]
        if (
            not isinstance(identifier, str)
            or not _CLASS_ID.fullmatch(identifier)
            or identifier == "exclude"
            or identifier in identifiers
        ):
            raise ValueError("Invalid or repeated class ID")
        identifiers.append(identifier)
        _text(item["name"], "class name", 120)
        _text(item["definition"], "class definition", 2000)
        if "coco_id" in item:
            coco_id = item["coco_id"]
            if type(coco_id) is not int or coco_id not in _COCO_IDS or coco_id in coco_ids:
                raise ValueError("Invalid or repeated explicit COCO category")
            coco_ids.add(coco_id)
    internal = {identifier: slot for slot, identifier in enumerate(identifiers, 1)}
    output = internal.copy()
    if builtin:
        if identifiers != ["person", "car"] or [item.get("coco_id") for item in classes] != [1, 3]:
            raise ValueError("Builtin classes must preserve person/car slots")
        output = {"person": 1, "car": 3}
    for key, mapping in (("class_mapping", internal), ("output_class_mapping", output)):
        if (
            not isinstance(contract[key], dict)
            or any(type(slot) is not int for slot in contract[key].values())
            or contract[key] != mapping
        ):
            raise ValueError(f"Invalid frozen {key}")
    return contract


def _relative_path(value):
    if not isinstance(value, str) or "\\" in value or ":" in value or len(value) > 256:
        raise ValueError("Invalid bundle path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in value.split("/"))
        or str(path) != value
    ):
        raise ValueError("Invalid bundle path")
    return path


def _file_limit(path):
    return (
        MAX_CHECKPOINT_BYTES
        if path == "model.pth"
        else MAX_IMAGE_BYTES
        if path.startswith("parity/images/")
        else MAX_JSON_BYTES
    )


def validate_manifest(manifest):
    _object(
        manifest,
        {"format", "id", "name", "created_at", "model", "source", "profile", "files", "validation"},
        "manifest",
    )
    if manifest["format"] != "iris-model-export-v1":
        raise ValueError("Unsupported model export format")
    _identifier(manifest["id"], "export ID")
    _text(manifest["name"], "export name", 200)
    _timestamp(manifest["created_at"])
    if canonical_bytes(manifest["profile"]) != canonical_bytes(PROFILE):
        raise ValueError("Unsupported runtime profile")
    if manifest["validation"] != {
        "real_execution": "not_run",
        "reference_kind": "saved_iris_evaluation",
    }:
        raise ValueError("Invalid export validation declaration")
    model = manifest["model"]
    _object(model, {"id", "name", "architecture", "sha256", "size", "class_contract"}, "model")
    _identifier(model["id"], "model ID")
    _text(model["name"], "model name", 200)
    if model["architecture"] != ARCHITECTURE:
        raise ValueError("Unsupported trained model architecture")
    _hash(model["sha256"])
    _integer(model["size"], "checkpoint size", 1, MAX_CHECKPOINT_BYTES)
    _class_contract(model["class_contract"])
    source = manifest["source"]
    _object(
        source,
        {"evaluation_id", "evaluation_model_id", "dataset_id", "dataset_manifest_sha256"},
        "source",
    )
    for key in ("evaluation_id", "evaluation_model_id", "dataset_id"):
        _identifier(source[key], key)
    _hash(source["dataset_manifest_sha256"])
    files = manifest["files"]
    if (
        not isinstance(files, dict)
        or not _REQUIRED_FILES <= files.keys()
        or not 6 <= len(files) <= 13
    ):
        raise ValueError("Invalid bundle file inventory")
    for path, evidence in files.items():
        _relative_path(path)
        if path not in _REQUIRED_FILES:
            if not re.fullmatch(r"parity/images/[A-Za-z0-9_-]{1,128}\.png", path):
                raise ValueError("Unexpected bundle file")
        _object(evidence, {"sha256", "size"}, "file evidence")
        _hash(evidence["sha256"])
        _integer(evidence["size"], "file size", 1, _file_limit(path))
    if files["model.pth"] != {"sha256": model["sha256"], "size": model["size"]}:
        raise ValueError("Checkpoint inventory differs from the model identity")
    return manifest


def _input_size(value):
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("Invalid image dimensions")
    for dimension in value:
        _integer(dimension, "image dimension", 1, MAX_PIXELS)
    if value[0] * value[1] > MAX_PIXELS:
        raise ValueError("Image exceeds the pixel limit")


def _detections(value, size, contract):
    if not isinstance(value, list) or len(value) > 100:
        raise ValueError("Invalid detection list")
    labels = {slot: label for label, slot in contract["class_mapping"].items()}
    builtin = contract["taxonomy_id"] == _BUILTIN_ID
    keys = {"box", "label", "label_id", "native_label_id", "score"}
    if not builtin:
        keys.add("taxonomy_id")
    for detection in value:
        _object(detection, keys, "detection")
        native = detection["native_label_id"]
        if type(native) is not int or native not in labels:
            raise ValueError("Invalid native class slot")
        label = labels[native]
        if (
            type(detection["label_id"]) is not int
            or detection["label_id"] != contract["output_class_mapping"][label]
            or detection["label"] != label
        ):
            raise ValueError("Detection class differs from its frozen mapping")
        if not builtin and detection["taxonomy_id"] != contract["taxonomy_id"]:
            raise ValueError("Detection taxonomy differs from its frozen mapping")
        box = detection["box"]
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError("Invalid box")
        for coordinate in box:
            _number(coordinate, "box coordinate", MAX_PIXELS)
        if not 0 <= box[0] < box[2] <= size[0] or not 0 <= box[1] < box[3] <= size[1]:
            raise ValueError("Box outside the oriented image")
        _number(detection["score"], "detection score", 1)


def validate_reference(manifest, reference):
    validate_manifest(manifest)
    _object(reference, {"format", "model_id", "weight_sha256", "frames"}, "reference")
    if (
        reference["format"] != "iris-export-reference-v1"
        or reference["model_id"] != manifest["model"]["id"]
        or reference["weight_sha256"] != manifest["model"]["sha256"]
    ):
        raise ValueError("Reference differs from the exported model")
    frames = reference["frames"]
    if not isinstance(frames, list) or not 1 <= len(frames) <= 8:
        raise ValueError("Expected 1–8 reference frames")
    ids, paths = set(), set()
    for frame in frames:
        _object(
            frame, {"frame_id", "path", "sha256", "input_size", "detections"}, "reference frame"
        )
        _identifier(frame["frame_id"], "frame ID")
        path = f"parity/images/{frame['frame_id']}.png"
        if frame["path"] != path or path in paths or frame["frame_id"] in ids:
            raise ValueError("Invalid or repeated reference frame")
        ids.add(frame["frame_id"])
        paths.add(path)
        _hash(frame["sha256"])
        if path not in manifest["files"] or frame["sha256"] != manifest["files"][path]["sha256"]:
            raise ValueError("Reference image differs from the bundle inventory")
        _input_size(frame["input_size"])
        _detections(frame["detections"], frame["input_size"], manifest["model"]["class_contract"])
    if set(manifest["files"]) != _REQUIRED_FILES | paths:
        raise ValueError("Bundle inventory differs from the reference images")
    if (
        digest_bytes(canonical_bytes(reference))
        != manifest["files"]["parity/reference.json"]["sha256"]
        or len(canonical_bytes(reference)) != manifest["files"]["parity/reference.json"]["size"]
    ):
        raise ValueError("Reference JSON differs from its canonical file evidence")
    return reference


def _environment(value):
    keys = {
        "python",
        "torch",
        "torchvision",
        "pillow",
        "platform",
        "machine",
        "processor",
        "cpu_count",
        "threads",
        "interop_threads",
        "device",
        "precision",
        "batch_size",
    }
    _object(value, keys, "measurement environment")
    for key in ("python", "torch", "torchvision", "pillow", "platform", "machine", "processor"):
        _text(value[key], f"environment {key}", 512)
    if not re.fullmatch(r"3\.(12|13)\.\d+", value["python"]):
        raise ValueError("Unsupported Python runtime")
    for key in ("torch", "torchvision"):
        if not re.fullmatch(
            re.escape(PROFILE["runtime"][key]) + r"(?:\+[A-Za-z0-9_.-]+)?", value[key]
        ):
            raise ValueError(f"Unsupported {key} runtime")
    if value["pillow"] != PROFILE["runtime"]["pillow"]:
        raise ValueError("Unsupported Pillow runtime")
    for key in ("cpu_count", "threads", "interop_threads"):
        _integer(value[key], key, 1, 65536)
    if (
        value["device"] != "cpu"
        or value["precision"] != "float32"
        or type(value["batch_size"]) is not int
        or value["batch_size"] != 1
        or value["threads"] != min(4, value["cpu_count"])
    ):
        raise ValueError("Environment differs from the CPU float32 batch-one profile")


def _timing(value):
    _object(
        value, {"preprocess_ms", "inference_ms", "postprocess_ms", "total_ms"}, "prediction timing"
    )
    for duration in value.values():
        _number(duration, "duration")
    total = sum(value[key] for key in ("preprocess_ms", "inference_ms", "postprocess_ms"))
    if not math.isclose(total, value["total_ms"], rel_tol=1e-9, abs_tol=1e-6):
        raise ValueError("Prediction timing total differs from its stages")


def validate_measurement(manifest, reference, payload):
    """Recompute evidence; a valid imported declaration is not authenticated execution."""
    validate_reference(manifest, reference)
    _object(
        payload,
        {
            "format",
            "manifest_sha256",
            "environment",
            "repeats",
            "warmup",
            "load_ms",
            "samples",
            "declaration",
        },
        "measurement",
    )
    if payload["format"] != "iris-export-measurement-v1" or payload[
        "manifest_sha256"
    ] != digest_bytes(canonical_bytes(manifest)):
        raise ValueError("Measurement differs from the export manifest")
    if not isinstance(payload["declaration"], str) or payload["declaration"] not in {
        "simulation",
        "external_execution",
    }:
        raise ValueError("Unknown execution declaration")
    _environment(payload["environment"])
    _integer(payload["repeats"], "repeats", 1, 10)
    _number(payload["load_ms"], "load duration")
    _object(payload["warmup"], {"frame_id", "duration_ms"}, "warmup")
    if payload["warmup"]["frame_id"] != reference["frames"][0]["frame_id"]:
        raise ValueError("Warmup must use the first reference image")
    _number(payload["warmup"]["duration_ms"], "warmup duration")
    samples = payload["samples"]
    expected = [
        (frame, repeat)
        for repeat in range(1, payload["repeats"] + 1)
        for frame in reference["frames"]
    ]
    if not isinstance(samples, list) or len(samples) != len(expected):
        raise ValueError("Missing or additional measurement samples")
    mismatched = []
    for sample, (frame, repeat) in zip(samples, expected, strict=True):
        _object(
            sample,
            {"frame_id", "repeat", "input_size", "detections", "timing", "decode_ms"},
            "measurement sample",
        )
        if (
            sample["frame_id"] != frame["frame_id"]
            or type(sample["repeat"]) is not int
            or sample["repeat"] != repeat
        ):
            raise ValueError("Measurement sample order or repetition differs from the protocol")
        _input_size(sample["input_size"])
        _detections(sample["detections"], sample["input_size"], manifest["model"]["class_contract"])
        _timing(sample["timing"])
        _number(sample["decode_ms"], "decode duration")
        if (
            sample["input_size"] != frame["input_size"]
            or sample["detections"] != frame["detections"]
        ):
            mismatched.append({"frame_id": frame["frame_id"], "repeat": repeat})

    def distribution(values):
        return {"min": min(values), "median": statistics.median(values), "max": max(values)}

    return {
        "parity_passed": not mismatched,
        "frames": len(reference["frames"]),
        "repeats": payload["repeats"],
        "sample_count": len(samples),
        "mismatched_samples": mismatched,
        "timing_ms": {
            key: distribution([sample["timing"][key] for sample in samples])
            for key in ("preprocess_ms", "inference_ms", "postprocess_ms", "total_ms")
        },
        "decode_ms": distribution([sample["decode_ms"] for sample in samples]),
        "load_ms": payload["load_ms"],
        "warmup_ms": payload["warmup"]["duration_ms"],
        "declaration": payload["declaration"],
        "execution_verified": False,
    }


def _safe_file(directory, relative):
    path = directory
    for part in _relative_path(relative).parts:
        path = path / part
        if path.is_symlink():
            raise ValueError("Bundle files and directories must not be symbolic links")
    if not path.is_file() or not path.resolve().is_relative_to(directory.resolve()):
        raise ValueError(f"Missing or unsafe bundle file: {relative}")
    return path


def _json_file(path):
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError("JSON exceeds the size limit")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    return json.loads(
        path.read_bytes(),
        object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"Invalid JSON number: {value}")
        ),
    )


def validate_bundle(directory):
    directory = Path(directory).resolve()
    manifest = validate_manifest(_json_file(_safe_file(directory, "manifest.json")))
    for relative, evidence in manifest["files"].items():
        path = _safe_file(directory, relative)
        if path.stat().st_size != evidence["size"]:
            raise ValueError(f"File size mismatch: {relative}")
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != evidence["sha256"]:
            raise ValueError(f"File SHA-256 mismatch: {relative}")
    reference = validate_reference(
        manifest, _json_file(_safe_file(directory, "parity/reference.json"))
    )
    return manifest, reference


def _restore_frozen_batchnorm(module, torch, torchvision):
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.BatchNorm2d):
            setattr(
                module, name, torchvision.ops.misc.FrozenBatchNorm2d(child.num_features, eps=1e-5)
            )
        else:
            _restore_frozen_batchnorm(child, torch, torchvision)


class Detector:
    """Same bounded construction and image contract as IRIS; no installation or download."""

    def __init__(self, directory, manifest):
        validate_manifest(manifest)
        for package in ("torch", "torchvision", "pillow"):
            installed = importlib.metadata.version(package)
            if installed.split("+", 1)[0] != PROFILE["runtime"][package]:
                raise RuntimeError(
                    f"Expected {package} {PROFILE['runtime'][package]}; found {installed}"
                )
        if sys.version_info[:2] not in ((3, 12), (3, 13)):
            raise RuntimeError("This profile requires Python 3.12 or 3.13")
        import torch
        import torchvision

        self.torch, self.torchvision = torch, torchvision
        self.contract = manifest["model"]["class_contract"]
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        options = dict(PROFILE["builder"])
        options["num_classes"] = len(self.contract["class_mapping"]) + 1
        self.model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(**options)
        _restore_frozen_batchnorm(self.model.backbone, torch, torchvision)
        checkpoint = _safe_file(Path(directory).resolve(), "model.pth")
        with checkpoint.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
            if digest != manifest["model"]["sha256"] or source.tell() != manifest["model"]["size"]:
                raise ValueError("Checkpoint changed before loading")
            source.seek(0)
            weights = torch.load(source, map_location="cpu", weights_only=True)
        self.model.load_state_dict(weights, strict=True)
        self.model = self.model.eval().to("cpu")

    def environment(self):
        return {
            "python": platform.python_version(),
            "torch": self.torch.__version__,
            "torchvision": self.torchvision.__version__,
            "pillow": importlib.metadata.version("pillow"),
            "platform": platform.platform(),
            "machine": platform.machine() or "unknown",
            "processor": platform.processor() or platform.machine() or "unknown",
            "cpu_count": os.cpu_count() or 1,
            "threads": self.torch.get_num_threads(),
            "interop_threads": self.torch.get_num_interop_threads(),
            "device": "cpu",
            "precision": "float32",
            "batch_size": 1,
        }

    def predict(self, image):
        from PIL import ImageOps

        started = time.perf_counter()
        oriented = ImageOps.exif_transpose(image).convert("RGB")
        _input_size(list(oriented.size))
        tensor = (
            self.torchvision.transforms.functional.pil_to_tensor(oriented).to(
                device="cpu", dtype=self.torch.float32
            )
            / 255.0
        )
        preprocessed = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model([tensor])[0]
        inferred = time.perf_counter()
        boxes, labels, scores = [
            output[key].detach().cpu().tolist() for key in ("boxes", "labels", "scores")
        ]
        if not len(boxes) == len(labels) == len(scores):
            raise ValueError("Detector returned mismatched output lengths")
        names = {slot: label for label, slot in self.contract["class_mapping"].items()}
        detections = []
        for box, native_id, score in zip(boxes, labels, scores, strict=True):
            if type(native_id) is not int or native_id not in names:
                raise ValueError("Detector returned an invalid native class slot")
            label = names[native_id]
            detection = {
                "box": box,
                "label_id": self.contract["output_class_mapping"][label],
                "native_label_id": native_id,
                "label": label,
                "score": score,
            }
            if self.contract["taxonomy_id"] != _BUILTIN_ID:
                detection["taxonomy_id"] = self.contract["taxonomy_id"]
            detections.append(detection)
        _detections(detections, list(oriented.size), self.contract)
        finished = time.perf_counter()
        return {
            "input_size": list(oriented.size),
            "detections": detections,
            "timing": {
                "preprocess_ms": (preprocessed - started) * 1000,
                "inference_ms": (inferred - preprocessed) * 1000,
                "postprocess_ms": (finished - inferred) * 1000,
                "total_ms": (finished - started) * 1000,
            },
        }


def _image(path, expected_hash=None):
    from PIL import Image

    if path.stat().st_size > MAX_IMAGE_BYTES:
        raise ValueError("Image exceeds the file size limit")
    content = path.read_bytes()
    if expected_hash is not None and digest_bytes(content) != expected_hash:
        raise ValueError("Reference image changed before decoding")
    with Image.open(io.BytesIO(content)) as source:
        _input_size(list(source.size))
        source.load()
        return source.copy()


def measure(directory, manifest, reference, repeats):
    validate_reference(manifest, reference)
    _integer(repeats, "repeats", 1, 10)
    started = time.perf_counter()
    detector = Detector(directory, manifest)
    load_ms = (time.perf_counter() - started) * 1000
    first = reference["frames"][0]
    with _image(_safe_file(Path(directory), first["path"]), first["sha256"]) as image:
        started = time.perf_counter()
        detector.predict(image)
        warmup_ms = (time.perf_counter() - started) * 1000
    samples = []
    for repeat in range(1, repeats + 1):
        for frame in reference["frames"]:
            started = time.perf_counter()
            with _image(_safe_file(Path(directory), frame["path"]), frame["sha256"]) as image:
                decode_ms = (time.perf_counter() - started) * 1000
                result = detector.predict(image)
            samples.append(
                {"frame_id": frame["frame_id"], "repeat": repeat, **result, "decode_ms": decode_ms}
            )
    payload = {
        "format": "iris-export-measurement-v1",
        "manifest_sha256": digest_bytes(canonical_bytes(manifest)),
        "environment": detector.environment(),
        "repeats": repeats,
        "warmup": {"frame_id": first["frame_id"], "duration_ms": warmup_ms},
        "load_ms": load_ms,
        "samples": samples,
        "declaration": "external_execution",
    }
    validate_measurement(manifest, reference, payload)
    return payload


def _output_path(path, directory):
    destination = Path(path).resolve()
    if destination.is_relative_to(Path(directory).resolve()):
        raise ValueError("Write results outside the immutable bundle directory")
    if destination.exists():
        raise FileExistsError("The output file already exists")
    if not destination.parent.is_dir():
        raise ValueError("The output directory does not exist")
    return destination


def _write_new_json(path, value, directory):
    destination = _output_path(path, directory)
    with destination.open("xb") as output:
        output.write(canonical_bytes(value))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Inspect or explicitly run a local IRIS model bundle. Never installs or downloads."
        )
    )
    parser.add_argument("--bundle", type=Path, default=Path(__file__).resolve().parent)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "inspect", help="Check inventory and hashes without loading ML dependencies"
    )
    predict = commands.add_parser("predict", help="Run one local image on CPU")
    predict.add_argument("image", type=Path)
    predict.add_argument("--output", required=True, type=Path)
    measurement = commands.add_parser(
        "measure", help="Compare saved references and measure CPU inference"
    )
    measurement.add_argument("--repeats", type=int, default=3)
    measurement.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        manifest, reference = validate_bundle(args.bundle)
        if args.command != "inspect":
            _output_path(args.output, args.bundle)
        if args.command == "inspect":
            result = {
                "format": manifest["format"],
                "id": manifest["id"],
                "name": manifest["name"],
                "model_id": manifest["model"]["id"],
                "files_verified": len(manifest["files"]),
                "reference_frames": len(reference["frames"]),
                "real_execution": "not_run",
                "runtime_loaded": False,
            }
        elif args.command == "predict":
            detector = Detector(args.bundle, manifest)
            with _image(args.image) as image:
                result = {
                    "manifest_sha256": digest_bytes(canonical_bytes(manifest)),
                    "environment": detector.environment(),
                    **detector.predict(image),
                }
            _write_new_json(args.output, result, args.bundle)
            result = {"output": str(args.output.resolve()), "detections": len(result["detections"])}
        else:
            payload = measure(args.bundle, manifest, reference, args.repeats)
            _write_new_json(args.output, payload, args.bundle)
            result = {
                "output": str(args.output.resolve()),
                **validate_measurement(manifest, reference, payload),
            }
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 3 if args.command == "measure" and not result["parity_passed"] else 0
    except (
        ValueError,
        OSError,
        RuntimeError,
        ImportError,
        EOFError,
        pickle.UnpicklingError,
        importlib.metadata.PackageNotFoundError,
    ) as exc:
        print(f"Export runner: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
