"""Frozen detector cache identities and observed outputs, without loading models.

Detection indices identify positions in the detector's final native output before
the cache score floor. They are neither temporal identities nor tracker output IDs.
"""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from types import MappingProxyType
from typing import Any

from iris.temporal_contracts import (
    MAX_SEQUENCE_FRAMES,
    _digest,
    _integer,
    _number,
    _object,
    _text,
    sequence_hash,
    validate_sequence_manifest,
)
from iris.temporal_detector import validate_detector_config
from iris.tiling import tile_boxes

CACHE_SCHEMA = "iris-temporal-detection-cache-v1"
FRAME_SCHEMA = "iris-temporal-detection-frame-v1"
MAX_FULL_DETECTIONS = 100
MAX_TILED_DETECTIONS = 300
TIMING_FIELDS = {
    "decode_ms",
    "preprocess_ms",
    "inference_ms",
    "postprocess_ms",
    "crop_ms",
    "merge_ms",
    "filter_ms",
    "total_ms",
}
_GENERATION = re.compile(r"[0-9a-f]{32}\Z")


@dataclass(frozen=True, slots=True)
class _FrameValidationContext:
    """Internal validated snapshot, bound to one unchanged input object pair."""

    original_config: dict = dataclass_field(repr=False, compare=False)
    original_sequence: dict = dataclass_field(repr=False, compare=False)
    fingerprint: str
    frames: MappingProxyType[str, Any] = dataclass_field(repr=False)
    classes: MappingProxyType[int, str] = dataclass_field(repr=False)
    work: MappingProxyType[tuple[int, int], tuple[int, int]] = dataclass_field(repr=False)
    min_score: float
    native_score_floor: float
    native_score_strict: bool
    maximum_detections: int


def _canonical_hash(value: dict) -> str:
    try:
        raw = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ValueError("Detection cache documents must be finite UTF-8 JSON") from exc
    return hashlib.sha256(raw).hexdigest()


def _cache_config(config: dict) -> dict:
    _object(
        config,
        {"schema", "sequence_id", "sequence_sha256", "detector", "generation", "frame_ids"},
        "Temporal detection cache",
    )
    if config["schema"] != CACHE_SCHEMA:
        raise ValueError("Unsupported temporal detection cache schema")
    _text(config["sequence_id"], "Detection cache sequence ID")
    _digest(config["sequence_sha256"], "Detection cache sequence hash")
    generation = config["generation"]
    if generation is not None and (
        not isinstance(generation, str) or not _GENERATION.fullmatch(generation)
    ):
        raise ValueError("Cache generation must be null or 32 lowercase hexadecimal characters")
    frame_ids = config["frame_ids"]
    if not isinstance(frame_ids, list) or not 1 <= len(frame_ids) <= MAX_SEQUENCE_FRAMES:
        raise ValueError("Detection cache requires 1–10000 sequence frame IDs")
    for frame_id in frame_ids:
        _text(frame_id, "Detection cache frame ID")
    if len(set(frame_ids)) != len(frame_ids):
        raise ValueError("Detection cache frame IDs must be unique")
    detector = validate_detector_config(config["detector"])
    result = deepcopy(config)
    result["detector"] = detector
    return result


def validate_cache_config(config: dict, sequence_manifest: dict) -> dict:
    """Bind a frozen detector to every available source frame in source order."""
    sequence = validate_sequence_manifest(sequence_manifest)
    config = _cache_config(config)
    if config["sequence_id"] != sequence["id"] or config["sequence_sha256"] != sequence_hash(
        sequence
    ):
        raise ValueError("Detection cache must bind the exact frozen sequence")
    if config["frame_ids"] != [frame["frame_id"] for frame in sequence["frames"]]:
        raise ValueError("Detection cache must include every sequence frame in its original order")
    detector = config["detector"]
    if (
        detector["origin"] == "trained"
        and detector["class_contract"]["taxonomy"] != sequence["taxonomy"]
    ):
        raise ValueError("Trained detector classes must match the frozen sequence taxonomy exactly")
    return config


def cache_fingerprint(config: dict) -> str:
    """Hash a canonical self-contained cache configuration, including rerun generation."""
    return _canonical_hash(_cache_config(config))


def frame_validation_context(config: dict, sequence_manifest: dict) -> _FrameValidationContext:
    """Validate/cache immutable inputs once for a sequence-sized output loop.

    This object is internal execution state, never a serialized or HTTP input.
    Reuse it only with the exact same config/manifest objects, kept unchanged for
    the lifetime of the loop. Source/class facts are independently frozen here.
    """
    sequence = validate_sequence_manifest(sequence_manifest)
    validated_config = validate_cache_config(config, sequence)
    detector = validated_config["detector"]
    inference = detector["inference"]
    tiled = inference["mode"] == "tiled"
    work = {}
    for frame in sequence["frames"]:
        size = (frame["width"], frame["height"])
        if size not in work:
            count = len(tile_boxes(*size, inference["tiling"])) if tiled else 0
            work[size] = (count if tiled else 1, count)
    return _FrameValidationContext(
        original_config=config,
        original_sequence=sequence_manifest,
        fingerprint=cache_fingerprint(validated_config),
        frames=MappingProxyType(
            {frame["frame_id"]: MappingProxyType(frame) for frame in sequence["frames"]}
        ),
        classes=MappingProxyType({item["id"]: item["name"] for item in detector["classes"]}),
        work=MappingProxyType(work),
        min_score=detector["min_score"],
        native_score_floor=float(detector["native_filtering"]["score_threshold"]),
        native_score_strict=detector["output_policy"]["native_score_comparison"] == "gt",
        maximum_detections=MAX_TILED_DETECTIONS if tiled else MAX_FULL_DETECTIONS,
    )


def _box(value, width: int, height: int) -> list[float]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("Cached boxes require four xyxy original-pixel coordinates")
    coordinates = [_number(item, "Cached box coordinate") for item in value]
    x1, y1, x2, y2 = coordinates
    if not (x1 < x2 <= width and y1 < y2 <= height):
        raise ValueError("Cached boxes must have positive area inside their source frame")
    return coordinates


def validate_frame_payload(
    payload: dict,
    config: dict,
    sequence_manifest: dict,
    *,
    context: _FrameValidationContext | None = None,
) -> dict:
    """Validate complete observed outputs with frozen classes, order, clock and work.

    The native count refers to the detector's post-NMS output, or the final merged
    tiled output, before the cache floor. Timing components may overlap, so this
    contract checks finite nonnegative measurements rather than summing them.
    """
    if context is None:
        context = frame_validation_context(config, sequence_manifest)
    elif (
        not isinstance(context, _FrameValidationContext)
        or context.original_config is not config
        or context.original_sequence is not sequence_manifest
    ):
        raise ValueError("Frame validation context must belong to these exact input objects")
    _object(
        payload,
        {
            "schema",
            "cache_fingerprint",
            "frame_id",
            "frame_index",
            "timestamp_seconds",
            "frame_sha256",
            "file_sha256",
            "input_size",
            "detections",
            "native_detection_count",
            "execution_signature_sha256",
            "timing",
            "work",
        },
        "Temporal detection frame",
    )
    if payload["schema"] != FRAME_SCHEMA:
        raise ValueError("Unsupported temporal detection frame schema")
    _digest(payload["cache_fingerprint"], "Frame cache fingerprint")
    if payload["cache_fingerprint"] != context.fingerprint:
        raise ValueError("Frame output belongs to a different detection cache")
    _text(payload["frame_id"], "Cached source frame ID")
    source = context.frames.get(payload["frame_id"])
    if source is None:
        raise ValueError("Cached frame does not belong to the frozen sequence")
    _integer(payload["frame_index"], "Cached source frame index")
    if payload["frame_index"] != source["frame_index"]:
        raise ValueError("Cached frame index does not match its source")
    for field, source_field in (("frame_sha256", "sha256"), ("file_sha256", "file_sha256")):
        _digest(payload[field], f"Cached {field}")
        if payload[field] != source[source_field]:
            raise ValueError("Cached image hashes do not match their frozen source")
    timestamp = payload["timestamp_seconds"]
    if source["timestamp_seconds"] is None:
        if timestamp is not None:
            raise ValueError("An unknown source clock cannot acquire a cached timestamp")
    else:
        timestamp = _number(timestamp, "Cached source timestamp")
        if timestamp != source["timestamp_seconds"]:
            raise ValueError("Cached timestamp must retain the exact source time")
    size = payload["input_size"]
    if not isinstance(size, list) or len(size) != 2:
        raise ValueError("Cached input size must contain original width and height")
    for dimension in size:
        _integer(dimension, "Cached input dimension", minimum=1)
    if size != [source["width"], source["height"]]:
        raise ValueError("Cached input dimensions must match the original source image")
    _digest(payload["execution_signature_sha256"], "Execution signature hash")
    native_count = _integer(
        payload["native_detection_count"],
        "Native detection count",
        maximum=context.maximum_detections,
    )
    detections = payload["detections"]
    if not isinstance(detections, list) or len(detections) > native_count:
        raise ValueError("Cached detections cannot exceed the bounded native output count")
    classes = context.classes
    previous_index, normalized = None, []
    for detection in detections:
        _object(
            detection,
            {"detection_index", "label_id", "label", "score", "box"},
            "Cached detection",
        )
        index = _integer(
            detection["detection_index"], "Native detection index", maximum=native_count - 1
        )
        if previous_index is not None and index <= previous_index:
            raise ValueError("Cached detection indices must preserve unique native output order")
        previous_index = index
        label_id = _integer(detection["label_id"], "Cached detection class ID")
        if label_id not in classes or detection["label"] != classes[label_id]:
            raise ValueError("Cached class ID and label must match the frozen detector classes")
        score = _number(detection["score"], "Cached detection score")
        if not context.min_score <= score <= 1:
            raise ValueError("Cached scores must be between the cache floor and one")
        if score < context.native_score_floor or (
            context.native_score_strict and score == context.native_score_floor
        ):
            raise ValueError("Cached score violates the frozen native threshold comparison")
        normalized.append({**detection, "score": score, "box": _box(detection["box"], *size)})
    _object(payload["timing"], TIMING_FIELDS, "Cached frame timing")
    timing = {
        field: _number(payload["timing"][field], f"Cached {field}") for field in TIMING_FIELDS
    }
    work = _object(payload["work"], {"forward_passes", "tile_count"}, "Cached frame work")
    passes, tile_count = context.work[tuple(size)]
    expected_work = {"forward_passes": passes, "tile_count": tile_count}
    for field, expected in expected_work.items():
        if _integer(work[field], f"Cached {field}") != expected:
            raise ValueError(
                "Cached work must match the frozen inference mode and source dimensions"
            )
    result = deepcopy(payload)
    result.update(timestamp_seconds=timestamp, detections=normalized, timing=timing)
    return result


def payload_hash(payload: dict) -> str:
    """Hash persisted normalized output; source-aware validation is a separate step."""
    if not isinstance(payload, dict):
        raise ValueError("Detection frame payload must be an object")
    return _canonical_hash(payload)
