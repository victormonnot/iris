"""Pinned full-image detector recipes for portable inspection without ML imports.

These frozen v1 definitions mirror temporal_detector and the standalone trained
model class contract. Tests enforce parity; changing a recipe needs a new format.
"""

import json
import math
import re
from copy import deepcopy

FRCNN = "fasterrcnn_mobilenet_v3_large_320_fpn"
SSDLITE = "ssdlite320_mobilenet_v3_large"
YOLOX = "yolox_nano"
ARCHITECTURES = {FRCNN, SSDLITE, YOLOX}
SCHEMA = "iris-temporal-detector-v1"
ADAPTER_REVISION = SCHEMA
PACKAGES = ("torch", "torchvision", "pillow", "numpy", "opencv-python-headless")
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_BUILTIN_ID = "iris-objects-v1"
_CLASS_ID = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z")
_TAXONOMY_ID = re.compile(r"taxonomy-[0-9a-f]{32}\Z")
COCO_TAXONOMY = "coco-2017-v1"

COCO_CATEGORIES = (
    "__background__",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)

_BUILTIN_TAXONOMY = {
    "id": "iris-objects-v1",
    "box_format": "xyxy_pixels",
    "classes": [
        {
            "id": "person",
            "name": "Person",
            "definition": "A visible human, including a rider. Enclose the visible extent of "
            "each person; do not infer a box for a fully occluded person.",
            "coco_id": 1,
        },
        {
            "id": "car",
            "name": "Car",
            "definition": "A passenger car, including an SUV or passenger minivan. Exclude "
            "buses, trucks, motorcycles and bicycles. Enclose the visible "
            "extent of each car.",
            "coco_id": 3,
        },
    ],
    "review_guidance": "Review the whole image for missing objects and imprecise boxes. Only "
    "validate when all visible target objects are annotated. A validated empty "
    "image is an explicit negative example. Automatic proposals are never "
    "reference annotations by themselves.",
}

INPUT_TRANSFORM = {
    "color": "BGR",
    "tensor_range": [0, 255],
    "exif_transpose": True,
    "fixed_size": [416, 416],
    "dtype": "float32",
    "layout": "NCHW",
    "letterbox": {"alignment": "top_left", "value": 114, "interpolation": "opencv_linear"},
}

NATIVE_FILTERING = {
    "score_threshold": 0.001,
    "nms_iou_threshold": 0.5,
    "max_detections_per_image": 100,
    "score": "objectness multiplied by best class probability",
    "nms": "class_aware",
}

_FIELDS = {
    "architecture",
    "class_contract",
    "classes",
    "device",
    "inference",
    "min_score",
    "model_id",
    "native_filtering",
    "origin",
    "output_policy",
    "preprocessing",
    "runtime",
    "schema",
    "weight_sha256",
}

_COCO_IDS = {i for i, name in enumerate(COCO_CATEGORIES) if i and name != "N/A"}


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


def _integer(value, context, minimum=0, maximum=1_000_000):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"Invalid {context}")


def _class_contract(contract):
    """Mirror frozen dataset taxonomy validation, without database imports."""
    _object(
        contract,
        {"taxonomy", "taxonomy_id", "class_mapping", "output_class_mapping"},
        "class contract",
    )
    taxonomy = contract["taxonomy"]
    builtin = contract["taxonomy_id"] == _BUILTIN_ID
    fields = {"id", "classes", "box_format", "review_guidance"}
    _object(
        taxonomy, fields if builtin else fields | {"version", "parent_id", "created_at"}, "taxonomy"
    )
    if taxonomy["id"] != contract["taxonomy_id"] or taxonomy["box_format"] != "xyxy_pixels":
        raise ValueError("Invalid frozen taxonomy identity or coordinates")
    if builtin:
        _same(taxonomy, _BUILTIN_TAXONOMY, "Builtin taxonomy")
    else:
        identifier, parent = taxonomy["id"], taxonomy["parent_id"]
        if (
            not isinstance(identifier, str)
            or not _TAXONOMY_ID.fullmatch(identifier)
            or type(taxonomy["version"]) is not int
            or taxonomy["version"] < 2
            or not isinstance(parent, str)
            or parent == identifier
            or not (parent == _BUILTIN_ID or _TAXONOMY_ID.fullmatch(parent))
            or (taxonomy["version"] == 2) != (parent == _BUILTIN_ID)
            or not isinstance(taxonomy["created_at"], str)
            or not taxonomy["created_at"]
        ):
            raise ValueError("Invalid frozen custom taxonomy")
    if taxonomy["review_guidance"] != _BUILTIN_TAXONOMY["review_guidance"]:
        raise ValueError("Unsupported frozen review guidance")
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
        for key, maximum in (("name", 120), ("definition", 2000)):
            value = item[key]
            if (
                not isinstance(value, str)
                or not 1 <= len(value) <= maximum
                or value != value.strip()
            ):
                raise ValueError("Frozen class definitions must retain their normalized values")
        if "coco_id" in item:
            coco_id = item["coco_id"]
            if type(coco_id) is not int or coco_id not in _COCO_IDS or coco_id in coco_ids:
                raise ValueError("Invalid or repeated explicit COCO category")
            coco_ids.add(coco_id)
    internal = {identifier: slot for slot, identifier in enumerate(identifiers, 1)}
    output = {"person": 1, "car": 3} if builtin else internal.copy()
    for key, expected in (("class_mapping", internal), ("output_class_mapping", output)):
        mapping = contract[key]
        if (
            not isinstance(mapping, dict)
            or mapping != expected
            or any(type(slot) is not int for slot in mapping.values())
        ):
            raise ValueError("Frozen class mapping differs from its definitions")
    return deepcopy(contract)


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
            for index, name in enumerate(COCO_CATEGORIES)
            if index and name != "N/A"
        ]
    else:
        expected_contract = _class_contract(config["class_contract"])
        classes = [
            {"id": expected_contract["output_class_mapping"][item["id"]], "name": item["id"]}
            for item in expected_contract["taxonomy"]["classes"]
        ]
    _same(config["class_contract"], expected_contract, "Detector class definitions")
    _same(config["classes"], classes, "Complete detector class list")
    inference = config["inference"]
    _object(inference, {"mode"}, "Full-image inference")
    if inference["mode"] != "full":
        raise ValueError("Pipeline bundles v1 support full-image inference only")
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


def detector_contract(config, target_device, *, checkpoint_size):
    checked = validate_detector_config(config)
    if target_device not in ("cpu", "cuda"):
        raise ValueError("Target device must be cpu or cuda")
    _integer(checkpoint_size, "checkpoint size", 1, 1024**3)
    yolo = checked["architecture"] == YOLOX
    trained = checked["origin"] == "trained"
    entries = []
    for position, item in enumerate(checked["classes"]):
        internal = position if yolo else position + 1 if trained else item["id"]
        entries.append(
            {
                "internal_index": internal,
                "native_label_id": internal + 1 if yolo else internal,
                "output_id": item["id"],
                "label": item["name"],
            }
        )
    return {
        "config": checked,
        "checkpoint": {
            "path": "detector/model.pth",
            "sha256": checked["weight_sha256"],
            "size": checkpoint_size,
            "encoding": "pytorch_model_envelope" if yolo and not trained else "pytorch_state_dict",
        },
        "target_device": target_device,
        "precision": "float32",
        "output_mapping": {
            "head_slots": len(entries) + (not yolo) if trained else 80 if yolo else 91,
            "background_index": None if yolo else 0,
            "entries": entries,
        },
    }


def license_contract(config):
    checked = validate_detector_config(config)
    yolo = checked["architecture"] == YOLOX
    source = "https://github.com/Megvii-BaseDetection/YOLOX/tree/419778480ab6ec0590e5d3831b3afb3b46ab2aa3"
    return {
        "detector": {
            "code_license": "Apache-2.0" if yolo else "BSD-3-Clause",
            "code_license_path": "licenses/detector-LICENSE",
            "code_source_url": source if yolo else "https://github.com/pytorch/vision/tree/v0.25.0",
            "weights_terms_url": source + "/LICENSE"
            if yolo
            else "https://docs.pytorch.org/vision/0.25/models.html#general-information-on-pre-trained-weights",
            "weights_rights": "not_inferred_from_code_license",
            "notice_path": "licenses/detector-NOTICE" if yolo else None,
        },
        "tracker": {"code_license": "MIT", "license_path": "licenses/tracker-LICENSE"},
        "notice_path": "licenses/NOTICE.txt",
    }
