"""Bounded sliced inference over original pixels, with deterministic global merging."""

import math
import time
from collections.abc import Callable
from copy import deepcopy

import numpy as np
from PIL import Image

MAX_TILES_PER_FRAME = 64
MAX_DETECTIONS_PER_TILE = 1000
_TIMING_FIELDS = ("preprocess_ms", "inference_ms", "postprocess_ms", "total_ms")


class TiledInferenceCancelled(Exception):
    """A cancelled image has no complete prediction to publish."""


def _number(value) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def validate_tiling_config(
    tile_size: int = 640,
    overlap: float = 0.2,
    *,
    merge_iou: float = 0.5,
    max_detections: int = 300,
) -> dict:
    """Normalize the small, explicit set of parameters used by saved runs."""
    if (
        isinstance(tile_size, bool)
        or not isinstance(tile_size, int)
        or not 128 <= tile_size <= 2048
    ):
        raise ValueError("Tile size must be an integer between 128 and 2048 pixels")
    if not _number(overlap) or not 0 <= overlap <= 0.5:
        raise ValueError("Tile overlap must be a finite number between 0 and 0.5")
    if not _number(merge_iou) or not 0.1 <= merge_iou <= 0.9:
        raise ValueError("Merge IoU must be a finite number between 0.1 and 0.9")
    if (
        isinstance(max_detections, bool)
        or not isinstance(max_detections, int)
        or not 1 <= max_detections <= 1000
    ):
        raise ValueError("The merged detection limit must be an integer between 1 and 1000")
    return {
        "tile_size": tile_size,
        "overlap": float(overlap),
        "merge_iou": float(merge_iou),
        "max_detections": max_detections,
    }


def _config(config: dict) -> dict:
    if not isinstance(config, dict):
        raise ValueError("Tiling configuration must be an object")
    allowed = {"tile_size", "overlap", "merge_iou", "max_detections"}
    if set(config) - allowed:
        raise ValueError("Unknown tiling configuration field")
    return validate_tiling_config(**config)


def tile_boxes(width: int, height: int, config: dict) -> list[list[int]]:
    """Return row-major original-pixel crops, including both far image edges.

    The stride is floor(tile_size * (1 - overlap)). The final crop on each
    axis is anchored to the edge and can therefore overlap more than requested.
    The count is checked before building coordinate lists, even for huge images.
    """
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in (width, height)
    ):
        raise ValueError("Image dimensions must be positive integers")
    settings = _config(config)
    size = settings["tile_size"]
    stride = math.floor(size * (1 - settings["overlap"]))

    def count(length: int) -> int:
        distance = max(0, length - size)
        return (distance + stride - 1) // stride + 1

    columns, rows = count(width), count(height)
    if columns * rows > MAX_TILES_PER_FRAME:
        raise ValueError(
            f"This image needs {columns * rows} tiles; the limit is {MAX_TILES_PER_FRAME}. "
            "Increase tile size or reduce overlap."
        )
    xs = [min(index * stride, max(0, width - size)) for index in range(columns)]
    ys = [min(index * stride, max(0, height - size)) for index in range(rows)]
    return [[x, y, min(x + size, width), min(y + size, height)] for y in ys for x in xs]


def _validate_tile_prediction(prediction: dict, width: int, height: int) -> None:
    if not isinstance(prediction, dict):
        raise ValueError("Detector returned an invalid tile prediction")
    size = prediction.get("input_size")
    if (
        not isinstance(size, (list, tuple))
        or len(size) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in size)
        or list(size) != [width, height]
    ):
        raise ValueError("Detector returned predictions for unexpected tile dimensions")
    detections = prediction.get("detections")
    if not isinstance(detections, list) or len(detections) > MAX_DETECTIONS_PER_TILE:
        raise ValueError(f"Detector must return at most {MAX_DETECTIONS_PER_TILE} boxes per tile")
    for detection in detections:
        if not isinstance(detection, dict):
            raise ValueError("Detector returned an invalid tile detection")
        box, score, label = (
            detection.get("box"),
            detection.get("score"),
            detection.get("label_id"),
        )
        if (
            not isinstance(box, (list, tuple))
            or len(box) != 4
            or not all(_number(value) for value in box)
        ):
            raise ValueError("Detector returned invalid tile coordinates")
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("Detector returned a box outside its tile")
        if not _number(score) or not 0 <= score <= 1:
            raise ValueError("Detector returned an invalid tile confidence")
        if isinstance(label, bool) or not isinstance(label, int) or label < 1:
            raise ValueError("Detector returned an invalid tile category")
        native = detection.get("native_label_id")
        if "native_label_id" in detection and (
            isinstance(native, bool) or not isinstance(native, int) or native < 0
        ):
            raise ValueError("Detector returned an invalid native category")
    timing = prediction.get("timing")
    if not isinstance(timing, dict) or any(
        not _number(timing.get(key)) or timing[key] < 0 for key in _TIMING_FIELDS
    ):
        raise ValueError("Detector returned an invalid tile timing measurement")


def _merge(detections: list[dict], threshold: float, cancelled: Callable[[], bool]) -> list[dict]:
    """Class-aware greedy NMS; ties retain tile order, then detector output order."""
    if not detections:
        return []
    by_class = {}
    for index, detection in enumerate(detections):
        by_class.setdefault(detection["label_id"], []).append(index)
    boxes = np.asarray([detection["box"] for detection in detections], dtype=np.float64)
    scores = [detection["score"] for detection in detections]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    kept = []
    for indices in by_class.values():
        remaining = np.asarray(sorted(indices, key=lambda index: (-scores[index], index)))
        while remaining.size:
            if cancelled():
                raise TiledInferenceCancelled()
            current = int(remaining[0])
            kept.append(current)
            remaining = remaining[1:]
            if not remaining.size:
                break
            left_top = np.maximum(boxes[current, :2], boxes[remaining, :2])
            right_bottom = np.minimum(boxes[current, 2:], boxes[remaining, 2:])
            intersection_size = np.maximum(0, right_bottom - left_top)
            intersection = intersection_size[:, 0] * intersection_size[:, 1]
            iou = intersection / (areas[current] + areas[remaining] - intersection)
            remaining = remaining[iou <= threshold]
    kept.sort(key=lambda index: (-scores[index], index))
    return [detections[index] for index in kept]


def tiled_predict(
    detector,
    image: Image.Image,
    config: dict,
    *,
    cancelled: Callable[[], bool] = lambda: False,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    """Predict complete crops, map to the source image and merge duplicate boxes.

    No padding, upscaling or resizing happens here. Each detector retains its
    usual internal image transform. Cancellation discards the incomplete image.
    Detector timings are summed; wall time also includes crop/validation/merge,
    but excludes progress callbacks. Raw tile outputs remain in metadata.
    """
    started = time.perf_counter()
    settings = _config(config)
    boxes = tile_boxes(*image.size, settings)
    timing = {key: 0.0 for key in _TIMING_FIELDS}
    timing.update(crop_ms=0.0, merge_ms=0.0, forward_passes=0, tile_count=len(boxes))
    detections = []
    tiles = []
    progress_seconds = 0.0
    for tile_index, box in enumerate(boxes):
        if cancelled():
            raise TiledInferenceCancelled()
        crop_started = time.perf_counter()
        with image.crop(tuple(box)) as crop:
            timing["crop_ms"] += (time.perf_counter() - crop_started) * 1000
            prediction = detector.predict(crop)
            if cancelled():
                raise TiledInferenceCancelled()
            _validate_tile_prediction(prediction, *crop.size)
        for key in _TIMING_FIELDS:
            timing[key] += prediction["timing"][key]
        timing["forward_passes"] += 1
        tiles.append({**deepcopy(prediction), "tile_index": tile_index, "box": box})
        mapping_started = time.perf_counter()
        x, y = box[:2]
        for detection in prediction["detections"]:
            x1, y1, x2, y2 = detection["box"]
            detections.append(
                {
                    **detection,
                    "box": [x1 + x, y1 + y, x2 + x, y2 + y],
                    "tile_index": tile_index,
                }
            )
        timing["merge_ms"] += (time.perf_counter() - mapping_started) * 1000
        if progress is not None:
            progress_started = time.perf_counter()
            progress(tile_index + 1, len(boxes))
            progress_seconds += time.perf_counter() - progress_started
    if cancelled():
        raise TiledInferenceCancelled()
    merge_started = time.perf_counter()
    merged = _merge(detections, settings["merge_iou"], cancelled)
    timing["merge_ms"] += (time.perf_counter() - merge_started) * 1000
    if cancelled():
        raise TiledInferenceCancelled()
    limited = merged[: settings["max_detections"]]
    timing.update(
        raw_detection_count=len(detections),
        merged_detection_count=len(merged),
        kept_detection_count=len(limited),
        truncated_detection_count=len(merged) - len(limited),
        total_ms=(time.perf_counter() - started - progress_seconds) * 1000,
    )
    return {
        "detections": limited,
        "input_size": list(image.size),
        "timing": timing,
        "metadata": {"tiles": tiles},
    }
