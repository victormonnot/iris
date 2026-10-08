"""Strict JSON requests and explicit experimental selected-object policy limits."""

import hashlib
import json
import math
import re
from copy import deepcopy

REPORT_SCHEMA = "iris-tracking-selection-v1"
POLICY_SCHEMA = "iris-selection-policy-v1"
MAX_FRAMES = 500
MAX_SECONDS = 120
MAX_REPORT_BYTES = 48 * 1024**2
STATES = ("idle", "observed", "lost", "recovering", "ambiguous", "recovered", "expired", "released")
DEFAULT_POLICY = {
    "schema": POLICY_SCHEMA,
    "min_score": 0.3,
    "min_iou": 0.05,
    "max_center_distance": 1.0,
    "max_area_ratio": 3.0,
    "max_lost_seconds": 1.0,
    "max_lost_updates": 15,
    "recovery_confirmation_updates": 2,
}
POLICY_BOUNDS = {
    "min_score": [0, 1],
    "min_iou": [0, 1],
    "max_center_distance": [0, 10],
    "max_area_ratio": [1, 100],
    "max_lost_seconds": [0.001, 60],
    "max_lost_updates": [1, 1000],
    "recovery_confirmation_updates": [2, 10],
}
LIMITATIONS = [
    "Geometry and local track numbers are association evidence, not proof of physical identity. "
    "A sole lookalike at compatible coordinates can still be selected incorrectly.",
    "Both diagnostic policies use the same saved first-pass measured observations. "
    "They do not execute a detector, tracker, appearance model or application control loop.",
    "The Track ID only lane is a diagnostic counterfactual, not a reproduction of another app.",
    "Predictions and unassigned detections never refresh selected-object observation memory.",
    "Recovery confirmation needs consecutive source frames with the same candidate track number. "
    "Skipped frames are not replayed and clear confirmation; "
    "arbitrary ID churn may prevent recovery.",
    "Timeouts use source timestamps when known and an available-update bound. "
    "Unknown clocks or a null seconds limit use updates only; "
    "no capture or wall clock is inferred.",
    "Defaults are visible experimental assumptions, "
    "not settings qualified for every scene or device.",
    "Quality requires an explicitly selected, frozen human-reviewed reference identity. "
    "Assisted reference seeds and reused development footage are not independent qualification.",
    "Sampled outage durations count only consecutive evaluable source-frame intervals. "
    "They exclude unknown gaps and do not claim exact physical loss or recovery times.",
    "No policy is automatically applied to another application or exported as a deployed runtime.",
]


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _object(value, fields, name):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ValueError(f"{name} must contain exactly its documented fields")
    return value


def _text(value, name, maximum=128):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f"{name} must be a nonempty bounded string")
    try:
        value.encode("utf-8")
    except UnicodeError as exc:
        raise ValueError(f"{name} must be valid UTF-8") from exc
    return value


def _integer(value, name, minimum=0, maximum=2**53 - 1):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in its documented range")
    return value


def _number(value, name, minimum, maximum):
    try:
        valid = type(value) in (int, float) and math.isfinite(value) and minimum <= value <= maximum
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{name} must be a finite number in its documented range")
    return float(value)


def validate_policy(payload):
    _object(payload, DEFAULT_POLICY, "Selection policy")
    if payload["schema"] != POLICY_SCHEMA:
        raise ValueError("Unsupported selected-object policy schema")
    result = {"schema": POLICY_SCHEMA}
    for key, bounds in POLICY_BOUNDS.items():
        value = payload[key]
        if key == "max_lost_seconds" and value is None:
            result[key] = None
        elif key in {"max_lost_updates", "recovery_confirmation_updates"}:
            result[key] = _integer(value, key, *bounds)
        else:
            result[key] = _number(value, key, *bounds)
    return result


def canonicalize_source(payload):
    _object(payload, {"kind", "job_id", "sequence_id", "profile_sha256"}, "Selection source")
    if not isinstance(payload["kind"], str) or payload["kind"] not in {"comparison", "study"}:
        raise ValueError("Selection source kind must be comparison or study")
    result = {"kind": payload["kind"]}
    for key in ("job_id", "sequence_id"):
        result[key] = _text(payload[key], key)
    value = payload["profile_sha256"]
    if not isinstance(value, str) or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise ValueError("Selection source profile hash must be a lowercase SHA-256")
    result["profile_sha256"] = value
    return result


def canonicalize_request(payload):
    _object(
        payload,
        {"name", "source", "selection", "release_frame_id", "policy", "evaluation", "max_seconds"},
        "Selected-object request",
    )
    selection = _object(payload["selection"], {"frame_id", "detection_index"}, "Initial selection")
    result = {
        "name": _text(payload["name"], "Scenario name", 160).strip(),
        "source": canonicalize_source(payload["source"]),
        "selection": {
            "frame_id": _text(selection["frame_id"], "Selected frame ID"),
            "detection_index": _integer(selection["detection_index"], "Selected detection index"),
        },
        "release_frame_id": None,
        "policy": validate_policy(payload["policy"]),
        "evaluation": None,
        "max_seconds": _number(payload["max_seconds"], "Execution time budget", 1, MAX_SECONDS),
    }
    if payload["release_frame_id"] is not None:
        result["release_frame_id"] = _text(payload["release_frame_id"], "Release frame ID")
    evaluation = payload["evaluation"]
    if evaluation is not None:
        _object(
            evaluation,
            {"reference_id", "identity_id", "class_mapping", "iou_threshold"},
            "Evaluation",
        )
        mapping = evaluation["class_mapping"]
        if not isinstance(mapping, dict) or not 1 <= len(mapping) <= 100:
            raise ValueError("Class mapping requires 1–100 native classes")
        for key, value in mapping.items():
            if (
                not isinstance(key, str)
                or len(key) > 16
                or re.fullmatch("[1-9][0-9]*", key) is None
            ):
                raise ValueError("Class mapping keys must be positive native class IDs")
            if value is not None:
                _text(value, "Mapped taxonomy class")
        if not any(value is not None for value in mapping.values()):
            raise ValueError("Evaluation requires at least one included class")
        result["evaluation"] = {
            "reference_id": _text(evaluation["reference_id"], "Reference ID"),
            "identity_id": _text(evaluation["identity_id"], "Reference identity ID"),
            "class_mapping": {key: mapping[key] for key in sorted(mapping, key=int)},
            "iou_threshold": _number(evaluation["iou_threshold"], "Evaluation IoU", 1e-12, 1),
        }
    return result


def selection_status():
    return {
        "schema": REPORT_SCHEMA,
        "policy_schema": POLICY_SCHEMA,
        "default_policy": deepcopy(DEFAULT_POLICY),
        "policy_bounds": deepcopy(POLICY_BOUNDS),
        "limits": {
            "max_frames": MAX_FRAMES,
            "max_seconds": MAX_SECONDS,
            "max_report_bytes": MAX_REPORT_BYTES,
        },
        "states": list(STATES),
        "limitations": list(LIMITATIONS),
    }
