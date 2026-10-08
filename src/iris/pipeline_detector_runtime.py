"""Native full-image detector for a trusted, unpacked portable pipeline bundle.

Importing this module uses only the standard library and portable JSON contracts.
Construction explicitly loads verified local weights. It never installs packages,
downloads a checkpoint, changes device family or substitutes another architecture.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import math
import os
import platform
import re
import stat
import sys
import time
from copy import deepcopy
from pathlib import Path

from .pipeline_bundle_contracts import canonical, digest, validate_manifest
from .pipeline_bundle_runtime_contracts import (
    FORMAT_V2,
    OPERATING_SYSTEMS,
    PACKAGE_VERSIONS,
    PYTHON_VERSIONS,
    VERSION_POLICY,
)
from .pipeline_detector_contracts import FRCNN, YOLOX

DETECTOR_PACKAGES = ("torch", "torchvision", "pillow", "numpy", "opencv-python-headless")
MAX_IMAGE_PIXELS = 100_000_000


def _device(manifest, requested=None):
    selected = requested if requested is not None else manifest["detector"]["target_device"]
    if not isinstance(selected, str) or not re.fullmatch(r"cpu|cuda(?::[0-9]{1,4})?", selected):
        raise ValueError("Detector device must be cpu, cuda, or cuda:<index>")
    if selected.split(":", 1)[0] != manifest["detector"]["target_device"]:
        raise ValueError("Detector device must match the bundle's frozen target family")
    return selected


def _packages():
    versions = {}
    for package in DETECTOR_PACKAGES:
        required = PACKAGE_VERSIONS[package]
        try:
            installed = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                f"Install the declared runtime separately: {package} {required}"
            ) from exc
        actual = (
            installed.split("+", 1)[0] if VERSION_POLICY[package] == "base_version" else installed
        )
        if actual != required:
            raise RuntimeError(f"Expected {package} {required}; found {installed}")
        versions[package] = installed
    return versions


def _runtime_modules(manifest, requested=None):
    selected = _device(manifest, requested)
    packages = _packages()
    if ".".join(str(value) for value in sys.version_info[:2]) not in PYTHON_VERSIONS:
        raise RuntimeError("This detector runtime requires Python 3.12 or 3.13")
    if platform.system() not in OPERATING_SYSTEMS:
        raise RuntimeError("This detector runtime currently supports Linux only")
    try:
        import torch
        import torchvision
    except (ImportError, OSError, RuntimeError) as exc:
        raise RuntimeError(f"The declared detector runtime cannot load: {exc}") from exc
    family = selected.split(":", 1)[0]
    operators = ["torchvision::nms"]
    if manifest["detector"]["config"]["architecture"] == FRCNN:
        operators.append("torchvision::roi_align")
    dispatch = "CUDA" if family == "cuda" else "CPU"
    if not torchvision.extension._has_ops() or any(
        not torch._C._dispatch_has_kernel_for_dispatch_key(name, dispatch) for name in operators
    ):
        raise RuntimeError(f"Matching Torchvision {dispatch} detection operators are unavailable")
    if family == "cuda":
        if not getattr(torch.version, "cuda", None) or not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; this target cannot fall back to CPU")
        index = int(selected.split(":")[1]) if ":" in selected else torch.cuda.current_device()
        if index >= torch.cuda.device_count():
            raise RuntimeError("Selected CUDA device does not exist")
        major, minor = torch.cuda.get_device_capability(index)
        capability = major * 10 + minor
        supported = any(
            (item.startswith("compute_") and int(item[8:]) <= capability)
            or (
                item.startswith("sm_")
                and int(item[3:]) // 10 == major
                and int(item[3:]) <= capability
            )
            for item in torch.cuda.get_arch_list()
            if re.fullmatch(r"(?:sm|compute)_[0-9]+", item)
        )
        if not supported:
            raise RuntimeError("This PyTorch build does not support the selected GPU architecture")
        selected = f"cuda:{index}"
        torch.cuda.set_device(index)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    return torch, torchvision, selected, packages


def _checkpoint(directory, identity):
    """Open the exact regular file; hash and deserialize through this same handle."""
    base = Path(directory)
    if base.is_symlink() or not base.is_dir():
        raise ValueError("Detector bundle directory must be an existing regular directory")
    base = base.resolve()
    path = base / identity["path"]
    if not path.resolve().is_relative_to(base) or any(
        ancestor.is_symlink() for ancestor in (path, *path.parents) if ancestor.is_relative_to(base)
    ):
        raise ValueError("Detector checkpoint path is unsafe")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError("Detector checkpoint is missing or unsafe") from exc
    try:
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode) or details.st_size != identity["size"]:
            raise ValueError("Detector checkpoint type or size differs from its manifest")
        stream = os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise
    try:
        hashed, count = hashlib.sha256(), 0
        while block := stream.read(1024**2):
            count += len(block)
            if count > identity["size"]:
                raise ValueError("Detector checkpoint grew before loading")
            hashed.update(block)
        if count != identity["size"] or hashed.hexdigest() != identity["sha256"]:
            raise ValueError("Detector checkpoint SHA-256 differs from its manifest")
        stream.seek(0)
        return stream, details
    except BaseException:
        stream.close()
        raise


def _restore_frozen_batchnorm(module, torch, torchvision):
    # weights=None otherwise changes the official Faster R-CNN normalization.
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.BatchNorm2d):
            setattr(
                module, name, torchvision.ops.misc.FrozenBatchNorm2d(child.num_features, eps=1e-5)
            )
        else:
            _restore_frozen_batchnorm(child, torch, torchvision)


def _build_yolox(count, torch):
    from ._vendor.yolox.yolo_head import YOLOXHead
    from ._vendor.yolox.yolo_pafpn import YOLOPAFPN
    from ._vendor.yolox.yolox import YOLOX as YOLOXModel

    model = YOLOXModel(
        YOLOPAFPN(0.33, 0.25, in_channels=[256, 512, 1024], depthwise=True),
        YOLOXHead(count, 0.25, in_channels=[256, 512, 1024], depthwise=True),
    )
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eps = 1e-3
            module.momentum = 0.03
    model.head.initialize_biases(1e-2)
    return model


def _preprocess_yolox(image, torch, device):
    import cv2
    import numpy as np

    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = rgb.shape[:2]
    ratio = min(416 / height, 416 / width)
    resized = cv2.resize(
        rgb[:, :, ::-1],
        (max(1, int(width * ratio)), max(1, int(height * ratio))),
        interpolation=cv2.INTER_LINEAR,
    )
    padded = np.full((416, 416, 3), 114, dtype=np.uint8)
    padded[: resized.shape[0], : resized.shape[1]] = resized
    array = np.ascontiguousarray(padded.transpose(2, 0, 1), dtype=np.float32)
    return torch.from_numpy(array).unsqueeze(0).to(device), ratio


def _postprocess_yolox(output, size, ratio, torch, torchvision):
    if output.ndim != 3 or output.shape[0] != 1 or output.shape[2] < 6:
        raise ValueError("YOLOX returned an invalid prediction shape")
    prediction = output[0]
    if not torch.isfinite(prediction).all().item():
        raise ValueError("YOLOX returned nonfinite predictions")
    class_scores, class_indices = prediction[:, 5:].max(dim=1)
    scores = prediction[:, 4] * class_scores
    boxes = (
        torch.cat(
            (
                prediction[:, :2] - prediction[:, 2:4] / 2,
                prediction[:, :2] + prediction[:, 2:4] / 2,
            ),
            dim=1,
        )
        / ratio
    )
    width, height = size
    boxes[:, (0, 2)] = boxes[:, (0, 2)].clamp(0, width)
    boxes[:, (1, 3)] = boxes[:, (1, 3)].clamp(0, height)
    keep = (scores >= 0.001) & (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes, scores, class_indices = boxes[keep], scores[keep], class_indices[keep]
    chosen = torchvision.ops.batched_nms(boxes, scores, class_indices, 0.5)[:100]
    return {"boxes": boxes[chosen], "scores": scores[chosen], "labels": class_indices[chosen] + 1}


def _canonical_detections(boxes, labels, scores, size, detector):
    if not len(boxes) == len(labels) == len(scores) or len(boxes) > 100:
        raise ValueError("Detector output lengths or native detection cap are invalid")
    mapping = {entry["native_label_id"]: entry for entry in detector["output_mapping"]["entries"]}
    width, height = size
    config = detector["config"]
    floor = config["native_filtering"]["score_threshold"]
    rows = []
    for index, (box, label, score) in enumerate(zip(boxes, labels, scores, strict=True)):
        if type(label) is not int or label not in mapping:
            raise ValueError("Detector returned an unknown native class slot")
        if (
            not isinstance(box, list)
            or len(box) != 4
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in box)
            or not 0 <= box[0] < box[2] <= width
            or not 0 <= box[1] < box[3] <= height
            or type(score) not in (int, float)
            or not math.isfinite(score)
            or not floor <= score <= 1
            or (score == floor and config["output_policy"]["native_score_comparison"] == "gt")
        ):
            raise ValueError("Detector output violates its frozen coordinates or score contract")
        if score >= config["min_score"]:
            entry = mapping[label]
            rows.append(
                {
                    "detection_index": index,
                    "label_id": entry["output_id"],
                    "label": entry["label"],
                    "score": score,
                    "box": list(box),
                }
            )
    return rows


class Detector:
    """Verified native checkpoint execution with the bundle's fixed target family."""

    def __init__(self, directory, manifest, device=None):
        self.manifest = validate_manifest(manifest)
        if self.manifest["format"] != FORMAT_V2:
            raise ValueError("Detector execution requires a version-two runtime bundle")
        self.detector = deepcopy(self.manifest["detector"])
        self.config = self.detector["config"]
        torch, torchvision, self.device, packages = _runtime_modules(self.manifest, device)
        self.torch, self.torchvision = torch, torchvision
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        architecture = self.config["architecture"]
        count = self.detector["output_mapping"]["head_slots"]
        with torch.device("cpu"):
            if architecture == YOLOX:
                model = _build_yolox(count, torch)
            else:
                builder = getattr(torchvision.models.detection, architecture)
                options = {"weights": None, "weights_backbone": None, "num_classes": count}
                if architecture == FRCNN:
                    options.update(
                        box_score_thresh=0.001, box_nms_thresh=0.5, box_detections_per_img=100
                    )
                else:
                    options.update(score_thresh=0.001, nms_thresh=0.5, detections_per_img=100)
                model = builder(**options)
                if architecture == FRCNN:
                    _restore_frozen_batchnorm(model.backbone, torch, torchvision)
            model = model.float()
        stream, details = _checkpoint(directory, self.detector["checkpoint"])
        with stream:
            weights = torch.load(stream, map_location="cpu", weights_only=True)
            current = os.fstat(stream.fileno())
            if (current.st_size, current.st_mtime_ns, current.st_ctime_ns) != (
                details.st_size,
                details.st_mtime_ns,
                details.st_ctime_ns,
            ):
                raise ValueError("Detector checkpoint changed while loading")
        if self.detector["checkpoint"]["encoding"] == "pytorch_model_envelope":
            if not isinstance(weights, dict) or "model" not in weights:
                raise ValueError("Official YOLOX checkpoint has no model state")
            weights = weights["model"]
        model.load_state_dict(weights, strict=True)
        self.model = model.eval().to(self.device)
        self.synchronize()
        self._metadata = {
            "schema": "iris-pipeline-detector-runtime-v1",
            "model_id": self.config["model_id"],
            "architecture": architecture,
            "weight_sha256": self.detector["checkpoint"]["sha256"],
            "detector_contract_sha256": digest(self.detector),
            "device": self.device,
            "target_device": self.detector["target_device"],
            "precision": "float32",
            "batch_size": 1,
            "packages": packages,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "input_transform": deepcopy(self.config["preprocessing"]),
            "native_filtering": deepcopy(self.config["native_filtering"]),
            "output_mapping": deepcopy(self.detector["output_mapping"]),
            "coordinates": self.config["output_policy"]["coordinates"],
            "cuda": self._cuda_metadata(),
        }

    def _cuda_metadata(self):
        if not self.device.startswith("cuda:"):
            return None
        torch = self.torch
        index = int(self.device.split(":")[1])
        properties = torch.cuda.get_device_properties(index)
        return {
            "runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "index": index,
            "name": properties.name,
            "capability": list(torch.cuda.get_device_capability(index)),
            "total_memory": properties.total_memory,
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
        }

    @property
    def metadata(self):
        return deepcopy(self._metadata)

    def verify_runtime(self):
        if (
            _packages() != self._metadata["packages"]
            or self.torch.get_num_threads() != self._metadata["threads"]
            or self.torch.get_num_interop_threads() != self._metadata["interop_threads"]
            or self._cuda_metadata() != self._metadata["cuda"]
            or digest(self.detector) != self._metadata["detector_contract_sha256"]
        ):
            raise RuntimeError("Detector runtime or contract changed; construct a fresh pipeline")

    def synchronize(self):
        if self.device.startswith("cuda:"):
            self.torch.cuda.synchronize(self.device)

    def predict(self, image):
        from PIL import Image, ImageOps

        if not isinstance(image, Image.Image):
            raise ValueError("Detector input must be a Pillow image")
        width, height = image.size
        if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
            raise ValueError("Detector input exceeds the supported image size")
        self.synchronize()
        started = time.perf_counter()
        oriented = ImageOps.exif_transpose(image).convert("RGB")
        architecture = self.config["architecture"]
        if architecture == YOLOX:
            tensor, ratio = _preprocess_yolox(oriented, self.torch, self.device)
        else:
            tensor = (
                self.torchvision.transforms.functional.pil_to_tensor(oriented).to(
                    device=self.device, dtype=self.torch.float32
                )
                / 255.0
            )
        self.synchronize()
        preprocessed = time.perf_counter()
        with (
            self.torch.inference_mode(),
            self.torch.autocast(device_type=self.detector["target_device"], enabled=False),
        ):
            output = self.model(tensor) if architecture == YOLOX else self.model([tensor])[0]
        self.synchronize()
        inferred = time.perf_counter()
        if architecture == YOLOX:
            output = _postprocess_yolox(output, oriented.size, ratio, self.torch, self.torchvision)
        boxes, labels, scores = [
            output[key].detach().cpu().tolist() for key in ("boxes", "labels", "scores")
        ]
        detections = _canonical_detections(boxes, labels, scores, oriented.size, self.detector)
        self.synchronize()
        finished = time.perf_counter()
        result = {
            "input_size": list(oriented.size),
            "detections": detections,
            "native_detection_count": len(boxes),
            "timing": {
                "preprocess_ms": (preprocessed - started) * 1000,
                "inference_ms": (inferred - preprocessed) * 1000,
                "postprocess_ms": (finished - inferred) * 1000,
                "total_ms": (finished - started) * 1000,
            },
        }
        canonical(result)
        return result
