"""Frozen local detector recipes for temporal caches, without implicit ML loading.

Cached detections are post-native-filtering outputs, never raw logits/proposals.
The storage threshold can only remove more outputs; it cannot undo native score,
class, NMS, proposal or detection caps. Preparation reads local weights and package
metadata but does not import Torch, construct a model, download or execute it.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import re
from copy import deepcopy
from pathlib import Path

from iris import models
from iris.model_taxonomy import class_contract
from iris.prediction_taxonomy import COCO_TAXONOMY, output_contract
from iris.tiling import validate_tiling_config
from iris.training_architectures import FRCNN, SSDLITE, YOLOX
from iris.yolox_spec import INPUT_TRANSFORM, NATIVE_FILTERING, SOURCE_COMMIT

SCHEMA = "iris-temporal-detector-v1"
ADAPTER_REVISION = "iris-temporal-detector-v1"
EXECUTION_SCHEMA = "iris-temporal-execution-v1"
PACKAGES = ("torch", "torchvision", "pillow", "numpy", "opencv-python-headless")
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_FIELDS = {
    "schema",
    "model_id",
    "architecture",
    "origin",
    "weight_sha256",
    "classes",
    "class_contract",
    "device",
    "min_score",
    "inference",
    "native_filtering",
    "preprocessing",
    "output_policy",
    "runtime",
}


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _json_value(value, *, depth=0):
    if depth > 20:
        raise ValueError("Detector configuration exceeds the supported nesting depth")
    if isinstance(value, dict):
        if len(value) > 1000 or any(type(key) is not str for key in value):
            raise ValueError("Detector configuration requires bounded JSON objects")
        for item in value.values():
            _json_value(item, depth=depth + 1)
    elif isinstance(value, list):
        if len(value) > 1000:
            raise ValueError("Detector configuration requires bounded JSON arrays")
        for item in value:
            _json_value(item, depth=depth + 1)
    elif type(value) is str:
        if len(value) > 20_000:
            raise ValueError("Detector configuration contains oversized text")
        try:
            value.encode("utf-8")
        except UnicodeError as exc:
            raise ValueError("Detector configuration requires valid UTF-8") from exc
    elif value is not None and type(value) not in (bool, int, float):
        raise ValueError("Detector configuration must contain JSON values only")
    elif type(value) in (int, float):
        try:
            if not math.isfinite(value):
                raise ValueError("Detector configuration contains a nonfinite number")
        except OverflowError as exc:
            raise ValueError("Detector configuration contains an oversized number") from exc


def _object(value, fields, label):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ValueError(f"{label} must contain exactly its supported fields")


def _text(value, label, maximum=160):
    if (
        type(value) is not str
        or not value.strip()
        or len(value) > maximum
        or any(ord(char) < 32 for char in value)
    ):
        raise ValueError(f"{label} must be bounded nonempty text")


def _same(actual, expected, label):
    # JSON equality rejects bool/int and tuple/list substitutions in frozen configs.
    if _json(actual) != _json(expected):
        raise ValueError(f"{label} differs from the frozen detector contract")


def _sources(architecture, mode):
    names = {
        "temporal_detector.py",
        "temporal_detection_worker.py",
        "temporal_detection_contracts.py",
        "models.py",
        "model_taxonomy.py",
        "prediction_taxonomy.py",
        "dataset_manifest.py",
        "taxonomies.py",
        "inference.py",
        "media.py",
    }
    if mode == "tiled":
        names.add("tiling.py")
    if architecture == YOLOX:
        names.update({"yolox_runtime.py", "yolox_spec.py"})
        names.update(
            f"_vendor/yolox/{name}.py"
            for name in (
                "__init__",
                "darknet",
                "losses",
                "network_blocks",
                "utils",
                "yolo_head",
                "yolo_pafpn",
                "yolox",
            )
        )
    return sorted(names)


def _runtime(architecture, mode):
    try:
        versions = {name: importlib.metadata.version(name) for name in PACKAGES}
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "The local detector runtime is incomplete; install IRIS ML dependencies"
        ) from exc
    for name, required in models.RUNTIME_VERSIONS.items():
        if versions[name].split("+", 1)[0] != required:
            raise RuntimeError(f"The local detector runtime requires {name} {required}")
    directory = Path(__file__).parent
    return {
        "adapter_revision": ADAPTER_REVISION,
        "python": platform.python_version(),
        "packages": versions,
        "source_sha256": {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in _sources(architecture, mode)
        },
    }


def _native_profile(architecture):
    if architecture == YOLOX:
        return deepcopy(NATIVE_FILTERING), deepcopy(INPUT_TRANSFORM)
    filtering = {
        "score_threshold": 0.001,
        "nms_iou_threshold": 0.5,
        "max_detections_per_image": 100,
        "ssdlite_topk_candidates_per_class": 300 if architecture == SSDLITE else None,
    }
    if architecture == FRCNN:
        filtering["rpn"] = {
            "score_threshold": 0.05,
            "nms_iou_threshold": 0.7,
            "pre_nms_top_n": 150,
            "post_nms_top_n": 150,
        }
    preprocessing = {
        "color": "RGB",
        "tensor_range": [0, 1],
        "exif_transpose": True,
        "image_mean": [0.5] * 3 if architecture == SSDLITE else [0.485, 0.456, 0.406],
        "image_std": [0.5] * 3 if architecture == SSDLITE else [0.229, 0.224, 0.225],
        "min_size": [320],
        "max_size": 320 if architecture == SSDLITE else 640,
        "fixed_size": [320, 320] if architecture == SSDLITE else None,
        "size_divisible": 1 if architecture == SSDLITE else 32,
    }
    return filtering, preprocessing


def _output_policy(architecture):
    return {
        "stage": "post_native_filtering",
        "native_score_comparison": "gte" if architecture == YOLOX else "gt",
        "storage_score_comparison": "gte",
        "class_selection": "all_detector_output_classes",
        "native_class_selection": "best_class_per_anchor"
        if architecture == YOLOX
        else "all_foreground_classes",
        "rpn_score_comparison": "gte" if architecture == FRCNN else None,
        "rpn_min_box_size": 0.001 if architecture == FRCNN else None,
        "native_min_box_size": 0.01 if architecture == FRCNN else None,
        "cap_scope": "all_classes_per_image_or_tile",
        "suppressed_candidates_recoverable": False,
        "complete_above_score_floor": False,
        "coordinates": "xyxy pixels, original oriented image, exclusive right/bottom edge",
    }


def validate_detector_config(config):
    """Validate a historical JSON snapshot without files, packages, Torch or network.

    Historical package versions/source hashes remain readable. Execution checks them
    against the current environment separately and refuses a changed recipe.
    """
    _json_value(config)
    _object(config, _FIELDS, "Temporal detector configuration")
    if config["schema"] != SCHEMA:
        raise ValueError("Unsupported temporal detector schema")
    _text(config["model_id"], "Detector model ID", 128)
    architecture = config["architecture"]
    if type(architecture) is not str or architecture not in {FRCNN, SSDLITE, YOLOX}:
        raise ValueError("Unsupported temporal detector architecture")
    if config["origin"] not in ("official", "trained"):
        raise ValueError("Unsupported temporal detector origin")
    if type(config["weight_sha256"]) is not str or not _HASH.fullmatch(config["weight_sha256"]):
        raise ValueError("Detector weights require a full SHA-256")
    if config["device"] not in ("cpu", "cuda"):
        raise ValueError("Temporal detector device must be cpu or cuda")
    if type(config["min_score"]) not in (int, float) or not 0.001 <= config["min_score"] <= 1:
        raise ValueError("Storage min_score must be between the native floor 0.001 and 1")
    if config["origin"] == "official":
        if config["model_id"] != architecture:
            raise ValueError("Official model ID must equal its detector architecture")
        expected_contract = {"taxonomy_id": COCO_TAXONOMY}
        classes = [
            {"id": index, "name": name}
            for index, name in enumerate(models.COCO_CATEGORIES)
            if index and name != "N/A"
        ]
    else:
        expected_contract = class_contract(config["class_contract"])
        classes = [
            {"id": expected_contract["output_class_mapping"][item["id"]], "name": item["id"]}
            for item in expected_contract["taxonomy"]["classes"]
        ]
    _same(config["class_contract"], expected_contract, "Detector class definitions")
    _same(config["classes"], classes, "Complete detector class list")
    inference = config["inference"]
    if not isinstance(inference, dict) or inference.get("mode") not in ("full", "tiled"):
        raise ValueError("Temporal inference must use full or tiled images")
    if inference["mode"] == "full":
        _object(inference, {"mode"}, "Full-image inference")
    else:
        _object(inference, {"mode", "algorithm", "tiling"}, "Tiled inference")
        if inference["algorithm"] != "iris-tiling-v1":
            raise ValueError("Unsupported temporal tiling algorithm")
        _object(
            inference["tiling"], {"tile_size", "overlap", "merge_iou", "max_detections"}, "Tiling"
        )
        _same(
            inference["tiling"],
            validate_tiling_config(
                inference["tiling"]["tile_size"], inference["tiling"]["overlap"]
            ),
            "Tiling settings",
        )
    filtering, preprocessing = _native_profile(architecture)
    _same(config["native_filtering"], filtering, "Native filtering")
    _same(config["preprocessing"], preprocessing, "Preprocessing")
    _same(config["output_policy"], _output_policy(architecture), "Output policy")
    runtime = config["runtime"]
    _object(
        runtime, {"adapter_revision", "python", "packages", "source_sha256"}, "Detector runtime"
    )
    if runtime["adapter_revision"] != ADAPTER_REVISION:
        raise ValueError("Unsupported temporal detector adapter revision")
    _text(runtime["python"], "Python version", 80)
    _object(runtime["packages"], PACKAGES, "Runtime package versions")
    for version in runtime["packages"].values():
        _text(version, "Package version", 100)
    _object(runtime["source_sha256"], _sources(architecture, inference["mode"]), "Adapter sources")
    if any(
        type(value) is not str or not _HASH.fullmatch(value)
        for value in runtime["source_sha256"].values()
    ):
        raise ValueError("Adapter sources require full SHA-256 hashes")
    result = deepcopy(config)
    result["min_score"] = float(config["min_score"])
    return result


def prepare_detector(
    root,
    model_id,
    *,
    device="cpu",
    inference_mode="full",
    tile_size=640,
    overlap=0.2,
    min_score=0.001,
):
    """Freeze a verified local recipe; never provision weights or import optional ML."""
    _text(model_id, "Detector model ID", 128)
    if device not in ("cpu", "cuda"):
        raise ValueError("Temporal detector device must be cpu or cuda")
    if inference_mode not in ("full", "tiled"):
        raise ValueError("Temporal inference must use full or tiled images")
    if type(min_score) not in (int, float) or not 0.001 <= min_score <= 1:
        raise ValueError("Storage min_score must be between the native floor 0.001 and 1")
    root = Path(root)
    spec = models.get_spec(model_id, root)
    if spec["architecture"] not in (FRCNN, SSDLITE, YOLOX):
        raise ValueError("Unsupported temporal detector architecture")
    digest = models._verified_digest(models.checkpoint_path(root, spec), spec)
    filtering, preprocessing = _native_profile(spec["architecture"])
    inference = {"mode": inference_mode}
    if inference_mode == "tiled":
        inference.update(
            algorithm="iris-tiling-v1", tiling=validate_tiling_config(tile_size, overlap)
        )
    return validate_detector_config(
        {
            "schema": SCHEMA,
            "model_id": model_id,
            "architecture": spec["architecture"],
            "origin": spec["origin"],
            "weight_sha256": digest,
            "classes": deepcopy(spec["classes"]),
            "class_contract": output_contract(spec),
            "device": device,
            "min_score": min_score,
            "inference": inference,
            "native_filtering": filtering,
            "preprocessing": preprocessing,
            "output_policy": _output_policy(spec["architecture"]),
            "runtime": _runtime(spec["architecture"], inference_mode),
        }
    )


def _verify_runtime(config):
    _same(
        config["runtime"],
        _runtime(config["architecture"], config["inference"]["mode"]),
        "Execution runtime or adapter sources",
    )


def _native_mapping(config):
    if config["origin"] == "official" and config["architecture"] == YOLOX:
        return {str(index): item["id"] for index, item in enumerate(config["classes"], 1)}
    if (
        config["origin"] == "trained"
        and config["architecture"] != YOLOX
        and config["class_contract"]["taxonomy_id"] == "iris-objects-v1"
    ):
        return {str(key): value for key, value in models.IRIS_NATIVE_TO_COCO.items()}
    return None


def _signature_from_metadata(config, metadata):
    """Pure comparison of observed facts against a frozen configuration."""
    if not isinstance(metadata, dict):
        raise ValueError("Loaded detector metadata must be an object")
    # Model metadata contains tuples (SSD transform) and integer mapping keys;
    # normalize those known JSON representations before comparing/persisting.
    try:
        metadata = json.loads(_json(metadata))
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("Loaded detector metadata is not finite JSON") from exc
    for field in ("model_id", "architecture", "weight_sha256"):
        _same(metadata.get(field), config[field], f"Loaded detector {field}")
    for name in ("torch", "torchvision"):
        _same(
            metadata.get(f"{name}_version"),
            config["runtime"]["packages"][name],
            f"Loaded {name} version",
        )
    for actual, frozen in (
        ("input_transform", "preprocessing"),
        ("native_filtering", "native_filtering"),
    ):
        _same(metadata.get(actual), config[frozen], f"Loaded {actual}")
    _same(metadata.get("coordinates"), config["output_policy"]["coordinates"], "Output coordinates")
    _same(metadata.get("precision"), "float32", "Runtime precision")
    if config["origin"] == "trained":
        _same(class_contract(metadata), config["class_contract"], "Loaded class definitions")
    elif any(key in metadata for key in ("taxonomy", "class_mapping", "output_class_mapping")):
        raise ValueError("Official detector metadata contains custom class definitions")
    slots = (
        len(config["class_contract"]["class_mapping"]) + (config["architecture"] != YOLOX)
        if config["origin"] == "trained"
        else (80 if config["architecture"] == YOLOX else 91)
    )
    _same(metadata.get("head_class_slots"), slots, "Native class slots")
    _same(metadata.get("native_to_coco"), _native_mapping(config), "Native output mapping")
    if config["architecture"] == YOLOX:
        for key, expected in {
            "source_commit": SOURCE_COMMIT,
            "background_class": False,
            "internal_class_index_base": 0,
            "native_class_index_base": 1,
        }.items():
            _same(metadata.get(key), expected, f"YOLOX {key}")
    device = metadata.get("device")
    if config["device"] == "cpu":
        if device != "cpu" or metadata.get("cuda") is not None:
            raise ValueError("CPU execution cannot contain a CUDA runtime")
    elif type(device) is not str or not re.fullmatch(r"cuda:\d+", device):
        raise ValueError("CUDA execution must report its actual indexed device; no CPU fallback")
    for field in ("hardware", "platform"):
        _text(metadata.get(field), f"Runtime {field}", 1000)
    for field in ("threads", "interop_threads"):
        if type(metadata.get(field)) is not int or not 1 <= metadata[field] <= 4096:
            raise ValueError("Runtime thread counts must be positive integers")
    cuda = metadata.get("cuda")
    if config["device"] == "cuda":
        _object(
            cuda,
            {
                "runtime",
                "cudnn",
                "index",
                "name",
                "capability",
                "total_memory",
                "uuid",
                "tf32_matmul",
                "tf32_cudnn",
                "cudnn_benchmark",
            },
            "CUDA runtime",
        )
        if type(cuda["index"]) is not int or device != f"cuda:{cuda['index']}":
            raise ValueError("CUDA metadata does not match the actual device")
        if cuda["name"] != metadata["hardware"]:
            raise ValueError("CUDA hardware does not match its runtime metadata")
        _text(cuda["runtime"], "CUDA version", 100)
        _text(cuda["name"], "CUDA hardware", 1000)
        if (
            (cuda["cudnn"] is not None and (type(cuda["cudnn"]) is not int or cuda["cudnn"] < 1))
            or type(cuda["total_memory"]) is not int
            or cuda["total_memory"] < 1
            or not isinstance(cuda["capability"], list)
            or len(cuda["capability"]) != 2
            or any(type(value) is not int or value < 0 for value in cuda["capability"])
        ):
            raise ValueError("CUDA capabilities, cuDNN and memory must be explicit")
        if cuda["uuid"] is not None:
            _text(cuda["uuid"], "CUDA UUID", 200)
        if any(cuda[key] is not False for key in ("tf32_matmul", "tf32_cudnn", "cudnn_benchmark")):
            raise ValueError("CUDA execution must retain the frozen float32 policy")
    return {
        "schema": EXECUTION_SCHEMA,
        "config_sha256": hashlib.sha256(_json(config).encode()).hexdigest(),
        "runtime": deepcopy(config["runtime"]),
        **{
            key: deepcopy(metadata[key])
            for key in (
                "model_id",
                "architecture",
                "weight_sha256",
                "device",
                "hardware",
                "platform",
                "threads",
                "interop_threads",
                "precision",
                "head_class_slots",
                "input_transform",
                "native_filtering",
            )
        },
        "native_to_coco": _native_mapping(config),
        "cuda": deepcopy(cuda),
    }


def verify_detector_metadata(config, metadata):
    """Verify loaded-model facts and return a deterministic signature for resume.

    Loading durations, timestamps and other attempt-specific instrumentation are
    excluded. Actual device/hardware/threads remain part of the signature, so a
    resumed job cannot mix CPU/GPU or changed runtime settings under one identity.
    """
    config = validate_detector_config(config)
    _verify_runtime(config)
    return _signature_from_metadata(config, metadata)


def saved_execution_signature(config, metadata):
    """Reconcile saved metadata with its signature without inspecting this machine."""
    return _signature_from_metadata(validate_detector_config(config), metadata)


def validate_execution_signature(config, signature):
    """Validate saved execution evidence on any computer, without local ML/files.

    This establishes consistency, not proof that a historical worker actually ran
    on the declared hardware. Current execution uses verify_detector_metadata.
    """
    config = validate_detector_config(config)
    _json_value(signature)
    _object(
        signature,
        {
            "schema",
            "config_sha256",
            "runtime",
            "model_id",
            "architecture",
            "weight_sha256",
            "device",
            "hardware",
            "platform",
            "threads",
            "interop_threads",
            "precision",
            "head_class_slots",
            "input_transform",
            "native_filtering",
            "native_to_coco",
            "cuda",
        },
        "Execution signature",
    )
    metadata = {
        **signature,
        "torch_version": config["runtime"]["packages"]["torch"],
        "torchvision_version": config["runtime"]["packages"]["torchvision"],
        "coordinates": config["output_policy"]["coordinates"],
    }
    if config["origin"] == "trained":
        metadata.update(config["class_contract"])
    if config["architecture"] == YOLOX:
        metadata.update(
            source_commit=SOURCE_COMMIT,
            background_class=False,
            internal_class_index_base=0,
            native_class_index_base=1,
        )
    checked = _signature_from_metadata(config, metadata)
    _same(signature, checked, "Saved execution signature")
    return checked


def detector_factory(root, config):
    """Construct the ordinary local detector with the frozen CPU/CUDA choice."""
    config = validate_detector_config(config)
    _verify_runtime(config)
    return models.TorchvisionDetector(Path(root), config["model_id"], device=config["device"])
