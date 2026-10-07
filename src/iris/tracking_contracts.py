"""Strict, JSON-only tracker profiles and per-frame observations.

These contracts intentionally distinguish measured detector boxes from tracker
predictions. They do not import a tracker, an ML runtime, or image libraries.
"""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy

PROFILE_SCHEMA = "iris-tracker-profile-v1"
FRAME_SCHEMA = "iris-tracking-frame-v1"
TIME_POLICY = "one_update_per_available_frame"
MAX_DETECTIONS = 300
ALGORITHMS = {"bytetrack", "botsort"}
UNASSIGNED_REASONS = {
    "below_low_threshold",
    "strict_high_boundary",
    "unmatched_low_confidence",
    "below_birth_threshold",
    "native_unconfirmed",
    "native_suppressed",
}
PROFILE_FIELDS = {
    "schema",
    "algorithm",
    "class_ids",
    "high_threshold",
    "low_threshold",
    "new_track_threshold",
    "match_threshold",
    "buffer_updates",
    "fuse_score",
    "gmc_method",
    "gmc_downscale",
    "seed",
    "opencv_threads",
    "with_reid",
    "time_policy",
}
SOURCE_FIELDS = {
    "frame_id",
    "frame_index",
    "timestamp_seconds",
    "input_size",
    "detections",
}
SOURCE_METADATA = {
    "schema",
    "cache_fingerprint",
    "frame_sha256",
    "file_sha256",
    "native_detection_count",
    "execution_signature_sha256",
    "timing",
    "work",
    "stored_payload_sha256",
    "producer_job_id",
}
DETECTION_FIELDS = {"detection_index", "label_id", "label", "score", "box"}
OBSERVATION_FIELDS = DETECTION_FIELDS | {"track_id", "confirmed", "estimated_box"}
PREDICTION_FIELDS = {
    "track_id",
    "label_id",
    "label",
    "box",
    "confirmed",
    "last_observed_frame_id",
    "last_observed_frame_index",
    "last_observed_timestamp_seconds",
    "last_observed_update_index",
    "age_updates",
    "age_seconds",
}
FRAME_FIELDS = {
    "schema",
    "sequence_id",
    "frame_id",
    "frame_index",
    "timestamp_seconds",
    "input_size",
    "update_index",
    "observations",
    "predictions",
    "unassigned",
    "gmc",
    "timing",
}


def _object(value, fields: set[str], name: str) -> dict:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{name} must contain exactly its documented fields")
    return value


def _text(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 1024:
        raise ValueError(f"{name} must be a nonempty string of at most 1024 characters")
    # JSON permits escaped lone surrogates, but they are not valid UTF-8 text.
    try:
        value.encode("utf-8")
    except UnicodeError as exc:
        raise ValueError(f"{name} must be valid UTF-8 text") from exc
    return value


def _integer(value, name: str, *, minimum=0, maximum=None) -> int:
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} must be an integer in its documented range")
    return value


def _number(value, name: str, *, minimum=None, maximum=None):
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if (
        not finite
        or (minimum is not None and value < minimum)
        or (maximum is not None and value > maximum)
    ):
        raise ValueError(f"{name} must be a finite number in its documented range")
    return value


def _boolean(value, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")
    return value


def _timestamp(value, name: str):
    return None if value is None else _number(value, name, minimum=0)


def _box(value, name: str, *, size: list[int] | None = None) -> list:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{name} must contain four xyxy coordinates")
    x1, y1, x2, y2 = [_number(item, name) for item in value]
    if x1 >= x2 or y1 >= y2:
        raise ValueError(f"{name} must have positive area")
    if size is not None and not (0 <= x1 < x2 <= size[0] and 0 <= y1 < y2 <= size[1]):
        raise ValueError(f"{name} must be inside the source frame")
    return value


def _class_ids(value) -> list[int]:
    if not isinstance(value, list) or not 1 <= len(value) <= 100:
        raise ValueError("Tracker class_ids requires 1–100 native class IDs")
    for class_id in value:
        _integer(class_id, "Tracker native class ID", minimum=1)
    if value != sorted(set(value)):
        raise ValueError("Tracker class_ids must be sorted and unique")
    return value


def validate_profile(profile: dict) -> dict:
    """Return a detached canonical profile with every native constraint explicit.

    ByteTrack fixes its low threshold to 0.1 and its birth threshold to the high
    threshold plus 0.1. Equivalent decimal birth values are normalized to the
    native sum so the profile hash describes the threshold actually executed.
    """
    _object(profile, PROFILE_FIELDS, "Tracker profile")
    if profile["schema"] != PROFILE_SCHEMA:
        raise ValueError("Unsupported tracker profile schema")
    if not isinstance(profile["algorithm"], str) or profile["algorithm"] not in ALGORITHMS:
        raise ValueError("Tracker algorithm must be bytetrack or botsort")
    _class_ids(profile["class_ids"])
    for key in ("low_threshold", "high_threshold", "new_track_threshold", "match_threshold"):
        _number(profile[key], key, minimum=0, maximum=1)
    high, low, birth = (
        profile["high_threshold"],
        profile["low_threshold"],
        profile["new_track_threshold"],
    )
    if profile["algorithm"] == "bytetrack":
        if (
            low != 0.1
            or not 0.1 < high <= 0.9
            or not math.isclose(birth, high + 0.1, rel_tol=0, abs_tol=1e-12)
        ):
            raise ValueError("ByteTrack requires low=0.1, 0.1<high<=0.9 and birth=high+0.1")
        if profile["gmc_method"] != "none":
            raise ValueError("ByteTrack does not provide camera motion compensation")
    elif not 0 <= low < high <= birth <= 1:
        raise ValueError("BoT-SORT thresholds must satisfy 0<=low<high<=birth<=1")
    if not isinstance(profile["gmc_method"], str) or profile["gmc_method"] not in {
        "none",
        "sparseOptFlow",
    }:
        raise ValueError("Camera motion compensation must be none or sparseOptFlow")
    _integer(profile["gmc_downscale"], "GMC downscale", minimum=2, maximum=2)
    _integer(profile["buffer_updates"], "Tracker lost buffer", minimum=0, maximum=10000)
    _integer(profile["seed"], "Tracker seed", minimum=0, maximum=2**31 - 1)
    _integer(profile["opencv_threads"], "OpenCV threads", minimum=1, maximum=32)
    _boolean(profile["fuse_score"], "Tracker score fusion")
    if _boolean(profile["with_reid"], "Tracker appearance embeddings"):
        raise ValueError("This tracker profile does not support learned ReID")
    if profile["time_policy"] != TIME_POLICY:
        raise ValueError("Tracker time policy must be one_update_per_available_frame")
    checked = deepcopy(profile)
    if checked["algorithm"] == "bytetrack":
        checked["new_track_threshold"] = high + 0.1
    return checked


def make_profile(algorithm: str, *, class_ids: list[int], **overrides) -> dict:
    """Build a complete strict profile; unknown options never silently disappear."""
    allowed = PROFILE_FIELDS - {"schema", "algorithm", "class_ids"}
    if set(overrides) - allowed:
        raise ValueError(f"Unknown tracker profile options: {sorted(set(overrides) - allowed)}")
    profile = {
        "schema": PROFILE_SCHEMA,
        "algorithm": algorithm,
        "class_ids": class_ids,
        "high_threshold": 0.5,
        "low_threshold": 0.1,
        "new_track_threshold": 0.6,
        "match_threshold": 0.8,
        "buffer_updates": 30,
        "fuse_score": True,
        "gmc_method": "sparseOptFlow" if algorithm == "botsort" else "none",
        "gmc_downscale": 2,
        "seed": 0,
        "opencv_threads": 1,
        "with_reid": False,
        "time_policy": TIME_POLICY,
        **overrides,
    }
    if algorithm == "bytetrack" and "new_track_threshold" not in overrides:
        high = _number(profile["high_threshold"], "high_threshold", minimum=0, maximum=1)
        profile["new_track_threshold"] = high + 0.1
    return validate_profile(profile)


def profile_hash(profile: dict) -> str:
    """SHA-256 of the canonical executable JSON profile, independent of key order."""
    canonical = json.dumps(
        validate_profile(profile),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _detection(value: dict, size: list[int], class_ids: set[int]) -> None:
    _integer(value["detection_index"], "Native detection index")
    _integer(value["label_id"], "Native detection class ID", minimum=1)
    if value["label_id"] not in class_ids:
        raise ValueError("Input detections must already be restricted to tracker class_ids")
    _text(value["label"], "Native detection label")
    _number(value["score"], "Native detection score", minimum=0, maximum=1)
    _box(value["box"], "Observed detection box", size=size)


def validate_update_input(frame: dict, profile: dict) -> dict:
    """Clone the source facts needed for one update; retain native detection indices.

    T2 cache/provenance metadata is accepted without interpreting it here. Its
    integrity is checked by the cache service before an update reaches a tracker.
    """
    checked_profile = validate_profile(profile)
    if (
        not isinstance(frame, dict)
        or not SOURCE_FIELDS <= set(frame)
        or set(frame) - SOURCE_FIELDS - SOURCE_METADATA
    ):
        raise ValueError("Tracker input requires source frame facts and known cache metadata")
    _text(frame["frame_id"], "Source frame ID")
    _integer(frame["frame_index"], "Source frame index")
    _timestamp(frame["timestamp_seconds"], "Source timestamp")
    size = frame["input_size"]
    if not isinstance(size, list) or len(size) != 2:
        raise ValueError("Source input_size must contain width and height")
    for dimension in size:
        _integer(dimension, "Source dimension", minimum=1)
    detections = frame["detections"]
    if not isinstance(detections, list) or len(detections) > MAX_DETECTIONS:
        raise ValueError("Tracker input accepts at most 300 native detections")
    indices = []
    labels = {}
    classes = set(checked_profile["class_ids"])
    for detection in detections:
        _object(detection, DETECTION_FIELDS, "Native input detection")
        _detection(detection, size, classes)
        indices.append(detection["detection_index"])
        label_id, label = detection["label_id"], detection["label"]
        if label_id in labels and labels[label_id] != label:
            raise ValueError("A native class ID must have one consistent label")
        labels[label_id] = label
    if indices != sorted(set(indices)):
        raise ValueError("Native detection indices must be unique and in their original order")
    return deepcopy({key: frame[key] for key in SOURCE_FIELDS})


def _validate_gmc(value: dict, profile: dict) -> None:
    _object(value, {"method", "status", "matrix", "downscale"}, "GMC result")
    if value["method"] != profile["gmc_method"]:
        raise ValueError("GMC result must use the frozen profile method")
    _integer(value["downscale"], "GMC downscale", minimum=2, maximum=2)
    if value["method"] == "none":
        if value["status"] != "disabled" or value["matrix"] is not None:
            raise ValueError("Disabled GMC requires status=disabled and a null matrix")
        return
    if not isinstance(value["status"], str) or value["status"] not in {
        "initialized",
        "estimated",
        "identity_insufficient_matches",
    }:
        raise ValueError("Unsupported camera motion result status")
    matrix = value["matrix"]
    if (
        not isinstance(matrix, list)
        or len(matrix) != 2
        or any(not isinstance(row, list) or len(row) != 3 for row in matrix)
    ):
        raise ValueError("Enabled GMC requires a finite 2×3 affine matrix")
    for row in matrix:
        for coordinate in row:
            _number(coordinate, "GMC affine coefficient")
    if value["status"] in {"initialized", "identity_insufficient_matches"} and matrix != [
        [1, 0, 0],
        [0, 1, 0],
    ]:
        raise ValueError("GMC initialization and insufficient matches must report identity motion")


def validate_tracking_frame(result: dict, source: dict, profile: dict) -> dict:
    """Check that every detector observation has exactly one explicit disposition.

    Track identities are unique across observations and predictions. Predicted
    boxes can leave the image, but can never masquerade as new detections or
    claim a detector score. Cross-frame identity continuity belongs to the runner.
    """
    profile = validate_profile(profile)
    source = validate_update_input(source, profile)
    _object(result, FRAME_FIELDS, "Tracking frame")
    if result["schema"] != FRAME_SCHEMA:
        raise ValueError("Unsupported tracking frame schema")
    _text(result["sequence_id"], "Tracking sequence ID")
    # Validate types before equality: Python would otherwise accept True == 1.
    _text(result["frame_id"], "Tracking source frame ID")
    _integer(result["frame_index"], "Tracking source frame index")
    _timestamp(result["timestamp_seconds"], "Tracking source timestamp")
    if not isinstance(result["input_size"], list) or len(result["input_size"]) != 2:
        raise ValueError("Tracking input_size must contain width and height")
    for dimension in result["input_size"]:
        _integer(dimension, "Tracking source dimension", minimum=1)
    for key in SOURCE_FIELDS - {"detections"}:
        if result[key] != source[key]:
            raise ValueError("Tracking result must preserve source frame identity, clock and size")
    update = _integer(result["update_index"], "Tracking update index", minimum=1)
    for key in ("observations", "predictions", "unassigned"):
        if not isinstance(result[key], list):
            raise ValueError(f"Tracking {key} must be a list")
    inputs = {item["detection_index"]: item for item in source["detections"]}
    classes = set(profile["class_ids"])
    labels = {item["label_id"]: item["label"] for item in source["detections"]}
    accounted, track_ids = set(), set()
    for key, fields in (
        ("observations", OBSERVATION_FIELDS),
        ("unassigned", DETECTION_FIELDS | {"reason"}),
    ):
        for item in result[key]:
            _object(item, fields, f"Tracking {key} item")
            _detection(item, source["input_size"], classes)
            index = item["detection_index"]
            if (
                index not in inputs
                or {name: item[name] for name in DETECTION_FIELDS} != inputs[index]
            ):
                raise ValueError("Tracking outputs must preserve each source detection unchanged")
            if index in accounted:
                raise ValueError(
                    "A source detection cannot appear more than once in a tracking result"
                )
            accounted.add(index)
            if key == "observations":
                track_id = _integer(item["track_id"], "Observation track ID", minimum=1)
                if track_id in track_ids:
                    raise ValueError("A track ID cannot appear more than once in a frame")
                track_ids.add(track_id)
                _boolean(item["confirmed"], "Observation confirmed state")
                _box(item["estimated_box"], "Estimated observation box")
            elif not isinstance(item["reason"], str) or item["reason"] not in UNASSIGNED_REASONS:
                raise ValueError("Unsupported unassigned detection reason")
    if accounted != set(inputs):
        raise ValueError("Every source detection requires an observation or an unassigned reason")
    for prediction in result["predictions"]:
        _object(prediction, PREDICTION_FIELDS, "Predicted track")
        track_id = _integer(prediction["track_id"], "Prediction track ID", minimum=1)
        if track_id in track_ids:
            raise ValueError("A track ID cannot appear more than once in a frame")
        track_ids.add(track_id)
        label_id = _integer(prediction["label_id"], "Prediction native class ID", minimum=1)
        label = _text(prediction["label"], "Prediction native label")
        if label_id not in classes or (label_id in labels and label != labels[label_id]):
            raise ValueError("Predictions must preserve the selected native class identity")
        labels[label_id] = label
        _boolean(prediction["confirmed"], "Prediction confirmed state")
        _box(prediction["box"], "Predicted track box")
        previous_id = _text(prediction["last_observed_frame_id"], "Last observed frame ID")
        previous_frame = _integer(
            prediction["last_observed_frame_index"], "Last observed frame index"
        )
        previous_update = _integer(
            prediction["last_observed_update_index"], "Last observed update index", minimum=1
        )
        previous_time = _timestamp(
            prediction["last_observed_timestamp_seconds"], "Last observed timestamp"
        )
        age = _integer(prediction["age_updates"], "Prediction age in updates", minimum=1)
        age_seconds = _timestamp(prediction["age_seconds"], "Prediction age in seconds")
        if (
            previous_id == source["frame_id"]
            or previous_frame >= source["frame_index"]
            or previous_update >= update
            or age != update - previous_update
        ):
            raise ValueError(
                "A prediction must reference an earlier observation with its actual age"
            )
        current_time = source["timestamp_seconds"]
        if current_time is None or previous_time is None:
            if age_seconds is not None:
                raise ValueError(
                    "Prediction age_seconds is unknown when either timestamp is unknown"
                )
        elif (
            previous_time > current_time
            or age_seconds is None
            or not math.isclose(
                age_seconds, current_time - previous_time, rel_tol=1e-9, abs_tol=1e-9
            )
        ):
            raise ValueError("Prediction age_seconds must match the source timestamps")
    _validate_gmc(result["gmc"], profile)
    _object(result["timing"], {"gmc_ms", "association_ms", "total_ms"}, "Tracking timing")
    for value in result["timing"].values():
        _number(value, "Tracking duration", minimum=0)
    return deepcopy(result)


def semantic_frame(result: dict) -> dict:
    """Detach semantic tracking outputs, excluding measured wall-clock durations.

    The caller validates the frame against its input/profile before publishing or
    comparing it. No frame identities, GMC estimates or dispositions are removed.
    """
    _object(result, FRAME_FIELDS, "Tracking frame")
    return deepcopy({key: value for key, value in result.items() if key != "timing"})
