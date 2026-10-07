"""Offline YOLOX-Nano runtime with the official BGR letterbox contract.

The native module remains available as ``detector.model``. Set its head's
``decode_in_inference`` to False for portable raw-grid ONNX export.
"""

from __future__ import annotations

import os
import platform
import re
import time
from copy import deepcopy
from pathlib import Path

from PIL import Image, ImageOps

from iris.yolox_spec import INPUT_SIZE, INPUT_TRANSFORM, NATIVE_FILTERING, SOURCE_COMMIT


def coco_class_ids():
    from iris.models import COCO_CATEGORIES

    return [index for index, label in enumerate(COCO_CATEGORIES) if index and label != "N/A"]


def build_model(num_classes: int = 80):
    """Build the pinned upstream Nano architecture without loading or downloading weights."""
    import torch

    from iris._vendor.yolox.yolo_head import YOLOXHead
    from iris._vendor.yolox.yolo_pafpn import YOLOPAFPN
    from iris._vendor.yolox.yolox import YOLOX

    if type(num_classes) is not int or not 1 <= num_classes <= 1000:
        raise ValueError("YOLOX needs between 1 and 1000 foreground classes")
    model = YOLOX(
        YOLOPAFPN(0.33, 0.25, in_channels=[256, 512, 1024], depthwise=True),
        YOLOXHead(num_classes, 0.25, in_channels=[256, 512, 1024], depthwise=True),
    )
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eps = 1e-3
            module.momentum = 0.03
    model.head.initialize_biases(1e-2)
    return model


def preprocess(image: Image.Image, device="cpu"):
    """Return a [1,3,416,416] float32 tensor and resize ratio for an oriented image.

    Callers apply EXIF orientation before this helper so returned predictions and
    reference annotation coordinates use the same oriented image size.
    """
    import cv2
    import numpy as np
    import torch

    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = rgb.shape[:2]
    ratio = min(INPUT_SIZE / height, INPUT_SIZE / width)
    resized = cv2.resize(
        rgb[:, :, ::-1],
        (max(1, int(width * ratio)), max(1, int(height * ratio))),
        interpolation=cv2.INTER_LINEAR,
    )
    padded = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
    padded[: resized.shape[0], : resized.shape[1]] = resized
    array = np.ascontiguousarray(padded.transpose(2, 0, 1), dtype=np.float32)
    return torch.from_numpy(array).unsqueeze(0).to(device), ratio


def postprocess(output, size, ratio):
    """Restore decoded center/size predictions; native labels are one-based foreground."""
    import torch
    from torchvision.ops import batched_nms

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
    keep = (
        (scores >= NATIVE_FILTERING["score_threshold"])
        & (boxes[:, 2] > boxes[:, 0])
        & (boxes[:, 3] > boxes[:, 1])
    )
    boxes, scores, class_indices = boxes[keep], scores[keep], class_indices[keep]
    chosen = batched_nms(boxes, scores, class_indices, NATIVE_FILTERING["nms_iou_threshold"])
    chosen = chosen[: NATIVE_FILTERING["max_detections_per_image"]]
    return {"boxes": boxes[chosen], "scores": scores[chosen], "labels": class_indices[chosen] + 1}


class YOLOXDetector:
    """Verified local official or trained YOLOX-Nano weights; no implicit network access."""

    def __init__(self, root: Path, model_id: str, device: str = "cpu"):
        from iris.model_taxonomy import class_contract
        from iris.models import (
            _configure_cuda,
            _cpu_name,
            _runtime_problem,
            _verified_digest,
            checkpoint_path,
            get_spec,
        )

        self.spec = get_spec(model_id, root)
        if self.spec["architecture"] != "yolox_nano":
            raise ValueError("Unsupported YOLOX architecture")
        if not re.fullmatch(r"cpu|cuda(?::\d+)?", device):
            raise ValueError("Device must be cpu, cuda, or cuda:<index>.")
        problem = _runtime_problem()
        if problem:
            raise RuntimeError(problem)
        path = checkpoint_path(root, self.spec)
        digest = _verified_digest(path, self.spec)
        import torch
        import torchvision

        self.torch = torch
        self.functional = torchvision.transforms.functional
        self.device = torch.device(device)
        cuda_metadata = None
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable; this attempt cannot fall back to CPU.")
            if self.device.index is None:
                self.device = torch.device("cuda", torch.cuda.current_device())
            cuda_metadata = _configure_cuda(torch, self.device)
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        self.class_contract = (
            class_contract(self.spec) if self.spec.get("origin") == "trained" else None
        )
        count = len(self.class_contract["class_mapping"]) if self.class_contract else 80
        self.native_to_coco = (
            None if self.class_contract else dict(enumerate(coco_class_ids(), start=1))
        )
        self.model = build_model(count)
        state = torch.load(path, map_location="cpu", weights_only=True)
        if self.spec["origin"] == "official":
            if not isinstance(state, dict) or "model" not in state:
                raise ValueError("Official YOLOX checkpoint has no model state")
            state = state["model"]
        self.model.load_state_dict(state, strict=True)
        self.model.eval().to(self.device)
        self.metadata = {
            "model_id": model_id,
            "architecture": "yolox_nano",
            "weights_name": self.spec["weights_name"],
            "weight_sha256": digest,
            "weight_source": self.spec["weight_url"],
            "license_url": self.spec["license_url"],
            "source_commit": SOURCE_COMMIT,
            "torch_version": str(torch.__version__),
            "torchvision_version": str(torchvision.__version__),
            "device": str(self.device),
            "hardware": torch.cuda.get_device_name(self.device) if cuda_metadata else _cpu_name(),
            "platform": platform.platform(),
            "threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "precision": "float32",
            "head_class_slots": count,
            "background_class": False,
            "internal_class_index_base": 0,
            "native_class_index_base": 1,
            "input_transform": deepcopy(INPUT_TRANSFORM),
            "native_filtering": deepcopy(NATIVE_FILTERING),
            "timing_protocol": {
                "preprocess_ms": "EXIF orientation, BGR OpenCV letterbox, float32 device transfer",
                "inference_ms": "YOLOX forward and grid decoding; excludes NMS",
                "postprocess_ms": "Class scores, clipping, class-aware NMS and JSON serialization",
                "total_ms": "Preprocessing + forward + postprocessing; excludes image file I/O",
                "synchronization": "CUDA synchronized at each timing boundary when using CUDA",
                "warmup": "One complete prediction on the first image; excluded from measurements",
                "batch_size": 1,
            },
            "coordinates": "xyxy pixels, original oriented image, exclusive right/bottom edge",
        }
        if cuda_metadata is not None:
            self.metadata["cuda"] = cuda_metadata
        if self.class_contract:
            self.metadata.update(
                **self.class_contract,
                training_id=self.spec["training_id"],
                parent_model_id=self.spec["parent_model_id"],
                training_provenance=self.spec["provenance"],
            )
        else:
            self.metadata["native_to_coco"] = dict(self.native_to_coco)

    def _synchronize(self):
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def warmup(self, image):
        self.predict(image)

    def predict(self, image):
        from iris.models import _serialize_predictions

        self._synchronize()
        started = time.perf_counter()
        oriented = ImageOps.exif_transpose(image).convert("RGB")
        tensor, ratio = preprocess(oriented, self.device)
        self._synchronize()
        preprocessed = time.perf_counter()
        with self.torch.inference_mode():
            prediction = self.model(tensor)
        self._synchronize()
        inferred = time.perf_counter()
        output = postprocess(prediction, oriented.size, ratio)
        detections = _serialize_predictions(
            output,
            oriented.size,
            self.native_to_coco,
            contract=self.class_contract,
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
