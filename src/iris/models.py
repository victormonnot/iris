"""Offline official and locally trained detectors with explicit weight provisioning.

The catalog never imports the ML runtime or performs a network request. Only
``download_model`` accesses the network, after a caller explicitly requests it.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import sqlite3
import tempfile
import time
from collections.abc import Callable
from copy import deepcopy
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from urllib.request import Request, urlopen

from PIL import Image, ImageOps

RUNTIME_VERSIONS = {"torch": "2.10.0", "torchvision": "0.25.0"}
# COCO category IDs contain gaps; never use a compact 0..79 class index.
# Source: torchvision v0.25.0, models/_meta.py (_COCO_CATEGORIES).
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

_BASE_URL = "https://download.pytorch.org/models/"
_SPECS = {
    "ssdlite320_mobilenet_v3_large": {
        "name": "SSDLite320 MobileNetV3-Large",
        "architecture": "ssdlite320_mobilenet_v3_large",
        "weight_filename": "ssdlite320_mobilenet_v3_large_coco-a79551df.pth",
        "expected_hash_prefix": "a79551df",
        "download_bytes": 14069355,
    },
    "fasterrcnn_mobilenet_v3_large_320_fpn": {
        "name": "Faster R-CNN MobileNetV3-Large 320 FPN",
        "architecture": "fasterrcnn_mobilenet_v3_large_320_fpn",
        "weight_filename": "fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth",
        "expected_hash_prefix": "907ea3f9",
        "download_bytes": 77844807,
    },
}

TIMING_PROTOCOL = {
    "preprocess_ms": (
        "EXIF orientation, RGB conversion, float32 tensor creation and device transfer"
    ),
    "inference_ms": (
        "Complete Torchvision forward: normalization, resize, backbone, detection heads, "
        "proposal filtering, NMS and restoration to original oriented image coordinates"
    ),
    "postprocess_ms": "Result transfer to CPU, validation and conversion to JSON-compatible values",
    "total_ms": "Preprocessing + complete forward + result serialization; excludes image file I/O",
    "synchronization": "CUDA synchronized at each timing boundary when using CUDA",
    "warmup": "One complete forward on the first image; excluded from measured predictions",
    "batch_size": 1,
}


TRAINING_ARCHITECTURE = "fasterrcnn_mobilenet_v3_large_320_fpn"
IRIS_NATIVE_TO_COCO = {0: 0, 1: 1, 2: 3}


def _trained_rows(root: Path) -> list[dict]:
    database = root / "iris.sqlite3"
    if not database.exists():
        return []
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        if not connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='trained_models'"
        ).fetchone():
            return []
        rows = connection.execute("SELECT * FROM trained_models ORDER BY created_at,id").fetchall()
    return [{**dict(row), "metadata": json.loads(row["metadata"])} for row in rows]


def _trained_spec(row: dict) -> dict:
    return {
        "id": row["id"],
        "name": row["name"],
        "architecture": row["architecture"],
        "origin": "trained",
        "checkpoint_path": row["path"],
        "weight_filename": Path(row["path"]).name,
        "weight_sha256": row["weight_sha256"],
        "weights_name": "IRIS fine-tuned",
        "weight_url": None,
        "license_url": get_spec(TRAINING_ARCHITECTURE)["license_url"],
        "task": "object_detection",
        "inference": True,
        "training": row["architecture"] == TRAINING_ARCHITECTURE,
        "classes": [{"id": 1, "name": "person"}, {"id": 3, "name": "car"}],
        "taxonomy_id": "iris-objects-v1",
        "native_to_coco": dict(IRIS_NATIVE_TO_COCO),
        "training_id": row["training_id"],
        "parent_model_id": row["parent_model_id"],
        "provenance": row["metadata"],
        "created_at": row["created_at"],
    }


def get_spec(model_id: str, root: Path | None = None) -> dict:
    """Return independent catalog metadata without importing Torchvision."""
    if model_id not in _SPECS:
        if root is not None:
            for row in _trained_rows(root.resolve()):
                if row["id"] == model_id:
                    return _trained_spec(row)
        raise ValueError(f"Unknown detector: {model_id}")
    spec = deepcopy(_SPECS[model_id])
    spec.update(
        id=model_id,
        weights_name="COCO_V1",
        weight_url=_BASE_URL + spec["weight_filename"],
        license_url=(
            "https://docs.pytorch.org/vision/0.25/models.html"
            "#general-information-on-pre-trained-weights"
        ),
        code_license_url="https://github.com/pytorch/vision/blob/v0.25.0/LICENSE",
        classes=[
            {"id": i, "name": label}
            for i, label in enumerate(COCO_CATEGORIES)
            if i and label != "N/A"
        ],
        task="object_detection",
        inference=True,
        training=spec["architecture"] == TRAINING_ARCHITECTURE,
        origin="official",
    )
    return spec


def _runtime_problem() -> str | None:
    problems = []
    for name, expected in RUNTIME_VERSIONS.items():
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            problems.append(f"{name} {expected} is not installed")
        else:
            if installed.split("+", 1)[0] != expected:
                problems.append(f"{name} {installed} is installed; expected {expected}")
    return "; ".join(problems) or None


@lru_cache(maxsize=16)
def _cached_digest(path: str, signature: tuple) -> str:
    # The stat signature invalidates cache entries on replacement or modification.
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_digest(path: Path, spec: dict) -> str:
    stat = path.stat()
    trained = spec.get("origin") == "trained"
    if not trained and stat.st_size != spec["download_bytes"]:
        raise ValueError("Checkpoint size does not match the official artifact.")
    signature = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino)
    digest = _cached_digest(str(path.resolve()), signature)
    if trained and digest != spec["weight_sha256"]:
        raise ValueError("Checkpoint SHA-256 does not match its training record.")
    if not trained and not digest.startswith(spec["expected_hash_prefix"]):
        raise ValueError("Checkpoint SHA-256 does not match the official hash prefix.")
    return digest


def checkpoint_path(root: Path, spec: dict) -> Path:
    if spec.get("origin") != "trained":
        return root / "models" / spec["weight_filename"]
    path = (root / spec["checkpoint_path"]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Trained checkpoint path escapes the workspace")
    return path


def catalog(root: Path) -> list[dict]:
    """Report dependency/file readiness; actual runtime loading happens in the worker."""
    runtime_problem = _runtime_problem()
    result = []
    items = [get_spec(model_id) for model_id in _SPECS]
    items.extend(_trained_spec(row) for row in _trained_rows(root.resolve()))
    for item in items:
        item.update(status="ready", reason=None, runtime_load_verified=False)
        try:
            path = checkpoint_path(root, item)
            if not path.exists():
                item.update(status="missing_weights", reason="Checkpoint is not installed.")
            else:
                item["weight_sha256"] = _verified_digest(path, item)
        except (OSError, ValueError) as exc:
            item.update(status="invalid_weights", reason=str(exc))
        if runtime_problem and item["status"] != "invalid_weights":
            item.update(status="missing_runtime", reason=runtime_problem)
        item["runtime_versions_required"] = dict(RUNTIME_VERSIONS)
        result.append(item)
    return result


def _receipt(root: Path, spec: dict, digest: str, *, downloaded: bool) -> dict:
    record = {
        "model_id": spec["id"],
        "weights_name": spec["weights_name"],
        "weight_sha256": digest,
        "weight_source": spec["weight_url"],
        "weight_filename": spec["weight_filename"],
        "download_bytes": spec["download_bytes"],
        "license_url": spec["license_url"],
        "verified_at": datetime.now(UTC).isoformat(),
        "downloaded": downloaded,
    }
    path = root / "models" / (spec["weight_filename"] + ".json")
    if not downloaded and path.exists():
        try:
            existing = json.loads(path.read_text())
            if existing.get("weight_sha256") == digest:
                return existing
        except (OSError, ValueError, AttributeError):
            pass
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=path.name + ".", suffix=".part", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
        try:
            json.dump(record, temporary, indent=2)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)
    return record


def download_model(
    root: Path,
    model_id: str,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    """Explicit download action; no user image or workspace metadata is transmitted.

    Existing valid files are reused. A corrupt existing file is only replaced
    after its replacement has passed size and SHA-256 verification.
    """
    spec = get_spec(model_id)
    directory = root / "models"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / spec["weight_filename"]
    if destination.exists():
        try:
            digest = _verified_digest(destination, spec)
        except (OSError, ValueError):
            pass
        else:
            return _receipt(root, spec, digest, downloaded=False)
    with tempfile.NamedTemporaryFile(
        dir=directory, prefix=destination.name + ".", suffix=".part", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
        try:
            request = Request(
                spec["weight_url"], headers={"User-Agent": "IRIS-checkpoint-download"}
            )
            digest = hashlib.sha256()
            count = 0
            with urlopen(request, timeout=30) as response:
                length = response.headers.get("Content-Length")
                if length is not None and int(length) != spec["download_bytes"]:
                    raise ValueError("Download size does not match the official artifact.")
                while chunk := response.read(1024 * 1024):
                    count += len(chunk)
                    if count > spec["download_bytes"]:
                        raise ValueError("Download exceeds the official artifact size.")
                    temporary.write(chunk)
                    digest.update(chunk)
                    if progress:
                        progress(count, spec["download_bytes"])
            if count != spec["download_bytes"]:
                raise ValueError("Checkpoint download is incomplete.")
            sha256 = digest.hexdigest()
            if not sha256.startswith(spec["expected_hash_prefix"]):
                raise ValueError(
                    "Downloaded checkpoint SHA-256 does not match the official prefix."
                )
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path.replace(destination)
        finally:
            temporary_path.unlink(missing_ok=True)
    return _receipt(root, spec, sha256, downloaded=True)


def _restore_frozen_batchnorm(module, torch, torchvision) -> None:
    # With weights=None Torchvision chooses ordinary BatchNorm. Official Faster
    # R-CNN weights use FrozenBatchNorm, whose evaluation epsilon must be retained.
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.BatchNorm2d):
            replacement = torchvision.ops.misc.FrozenBatchNorm2d(child.num_features, eps=1e-5)
            setattr(module, name, replacement)
        else:
            _restore_frozen_batchnorm(child, torch, torchvision)


def _serialize_predictions(
    output: dict, size: tuple[int, int], native_to_coco: dict | None = None
) -> list[dict]:
    """Preserve model outputs, including classes outside the current project taxonomy."""
    boxes = output["boxes"].detach().cpu().tolist()
    labels = output["labels"].detach().cpu().tolist()
    scores = output["scores"].detach().cpu().tolist()
    if not len(boxes) == len(labels) == len(scores):
        raise ValueError("Detector returned mismatched boxes, labels and scores.")
    width, height = size
    detections = []
    for box, label_id, score in zip(boxes, labels, scores, strict=True):
        native_id = label_id
        if native_to_coco is not None:
            if type(label_id) is not int or label_id not in native_to_coco:
                raise ValueError("Detector returned an unknown native class ID.")
            label_id = native_to_coco[label_id]
        if (
            len(box) != 4
            or not all(math.isfinite(value) for value in [*box, score])
            or not 0 <= score <= 1
            or not isinstance(label_id, int)
            or not 0 <= label_id < len(COCO_CATEGORIES)
            or not 0 <= box[0] < box[2] <= width
            or not 0 <= box[1] < box[3] <= height
        ):
            raise ValueError("Detector returned invalid original-image coordinates or scores.")
        detections.append(
            {"box": box, "label_id": label_id, "label": COCO_CATEGORIES[label_id], "score": score}
        )
        if native_to_coco is not None:
            detections[-1]["native_label_id"] = native_id
    return detections


def _cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


class TorchvisionDetector:
    """Load a verified official or trained checkpoint, without implicit downloads."""

    def __init__(self, root: Path, model_id: str, device: str = "cpu"):
        self.spec = get_spec(model_id, root)
        self.native_to_coco = self.spec.get("native_to_coco")
        if not re.fullmatch(r"cpu|cuda(?::\d+)?", device):
            raise ValueError("Device must be cpu, cuda, or cuda:<index>.")
        problem = _runtime_problem()
        if problem:
            raise RuntimeError(problem)
        path = checkpoint_path(root, self.spec)
        digest = _verified_digest(path, self.spec)
        try:
            import torch
            import torchvision
        except (ImportError, OSError, RuntimeError) as exc:
            raise RuntimeError(f"The optional Torchvision runtime cannot load: {exc}") from exc
        self.torch = torch
        self.functional = torchvision.transforms.functional
        self.device = torch.device(device)
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA is unavailable; select CPU or install a compatible runtime."
                )
            if self.device.index is None:
                self.device = torch.device("cuda", torch.cuda.current_device())
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        architecture = self.spec["architecture"]
        builder = getattr(torchvision.models.detection, architecture)
        num_classes = 3 if self.native_to_coco else 91
        options = {"weights": None, "weights_backbone": None, "num_classes": num_classes}
        if architecture.startswith("ssdlite"):
            options.update(score_thresh=0.001, nms_thresh=0.5, detections_per_img=100)
        else:
            options.update(box_score_thresh=0.001, box_nms_thresh=0.5, box_detections_per_img=100)
        model = builder(**options)
        if architecture.startswith("fasterrcnn"):
            _restore_frozen_batchnorm(model.backbone, torch, torchvision)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        self.model = model.eval().to(self.device)
        transform = self.model.transform
        native_filtering = {
            "score_threshold": 0.001,
            "nms_iou_threshold": 0.5,
            "max_detections_per_image": 100,
            "ssdlite_topk_candidates_per_class": getattr(model, "topk_candidates", None),
        }
        if hasattr(model, "rpn"):
            native_filtering["rpn"] = {
                "score_threshold": model.rpn.score_thresh,
                "nms_iou_threshold": model.rpn.nms_thresh,
                "pre_nms_top_n": model.rpn.pre_nms_top_n(),
                "post_nms_top_n": model.rpn.post_nms_top_n(),
            }
        self.metadata = {
            "model_id": model_id,
            "architecture": self.spec["architecture"],
            "weights_name": self.spec["weights_name"],
            "weight_sha256": digest,
            "weight_source": self.spec["weight_url"],
            "license_url": self.spec["license_url"],
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "device": str(self.device),
            "hardware": (
                torch.cuda.get_device_name(self.device)
                if self.device.type == "cuda"
                else _cpu_name()
            ),
            "platform": platform.platform(),
            "threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "precision": "float32",
            "head_class_slots": num_classes,
            "input_transform": {
                "color": "RGB",
                "tensor_range": [0, 1],
                "exif_transpose": True,
                "image_mean": list(transform.image_mean),
                "image_std": list(transform.image_std),
                "min_size": list(transform.min_size),
                "max_size": transform.max_size,
                "fixed_size": getattr(transform, "fixed_size", None),
                "size_divisible": transform.size_divisible,
            },
            "native_filtering": native_filtering,
            "timing_protocol": deepcopy(TIMING_PROTOCOL),
            "coordinates": "xyxy pixels, original oriented image, exclusive right/bottom edge",
        }
        if self.native_to_coco:
            self.metadata.update(
                native_to_coco=self.native_to_coco,
                taxonomy_id="iris-objects-v1",
                training_id=self.spec["training_id"],
                parent_model_id=self.spec["parent_model_id"],
                training_provenance=self.spec["provenance"],
            )

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def warmup(self, image: Image.Image) -> None:
        self.predict(image)

    def predict(self, image: Image.Image) -> dict:
        self._synchronize()
        started = time.perf_counter()
        oriented = ImageOps.exif_transpose(image).convert("RGB")
        tensor = (
            self.functional.pil_to_tensor(oriented).to(device=self.device, dtype=self.torch.float32)
            / 255.0
        )
        self._synchronize()
        preprocessed = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model([tensor])[0]
        self._synchronize()
        inferred = time.perf_counter()
        detections = _serialize_predictions(
            output, oriented.size, getattr(self, "native_to_coco", None)
        )
        self._synchronize()
        finished = time.perf_counter()
        return {
            "detections": detections,
            "input_size": list(oriented.size),
            "timing": {
                "preprocess_ms": (preprocessed - started) * 1000,
                "inference_ms": (inferred - preprocessed) * 1000,
                "postprocess_ms": (finished - inferred) * 1000,
                "total_ms": (finished - started) * 1000,
            },
        }
