"""Deterministic selected-object state from frozen measured tracker observations.

A local track number is an association hint, never an identity guarantee. The
policy has no reference-annotation input and cannot renew its memory from a
prediction, a rejected candidate or a polling call.
"""

import math
from copy import deepcopy


def _iou(first, second):
    intersection = max(0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0, min(first[3], second[3]) - max(first[1], second[1])
    )
    area1 = (first[2] - first[0]) * (first[3] - first[1])
    area2 = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / (area1 + area2 - intersection) if intersection else 0.0


def _geometry(anchor, candidate):
    width, height = anchor[2] - anchor[0], anchor[3] - anchor[1]
    width2, height2 = candidate[2] - candidate[0], candidate[3] - candidate[1]
    center = math.hypot(
        (candidate[0] + candidate[2] - anchor[0] - anchor[2]) / 2,
        (candidate[1] + candidate[3] - anchor[1] - anchor[3]) / 2,
    ) / math.hypot(width, height)
    ratio = max(width * height, width2 * height2) / min(width * height, width2 * height2)
    return {"iou": _iou(anchor, candidate), "center_distance": center, "area_ratio": ratio}


def _candidate(anchor, observation, policy):
    geometry = _geometry(anchor["box"], observation["box"])
    reasons = []
    if not observation["confirmed"]:
        reasons.append("unconfirmed")
    if observation["label_id"] != anchor["label_id"]:
        reasons.append("different_class")
    if observation["score"] < policy["min_score"]:
        reasons.append("low_score")
    if geometry["iou"] < policy["min_iou"]:
        reasons.append("insufficient_overlap")
    if geometry["center_distance"] > policy["max_center_distance"]:
        reasons.append("center_distance")
    if geometry["area_ratio"] > policy["max_area_ratio"]:
        reasons.append("area_ratio")
    return {
        "detection_index": observation["detection_index"],
        "track_id": observation["track_id"],
        "eligible": not reasons,
        "reasons": reasons,
        **geometry,
    }


def _anchor(frame, observation):
    return {
        "frame_id": frame["frame_id"],
        "frame_index": frame["frame_index"],
        "timestamp_seconds": frame["timestamp_seconds"],
        "update_index": frame["update_index"],
        "observation": deepcopy(observation),
    }


def _advance(frames, request, mode, checkpoint, charge):
    """Apply one policy without access to annotations, hashes or other lane state."""
    policy = request["policy"]
    last = pending = previous = None
    state = "idle"
    outputs = []
    for frame in frames:
        if checkpoint:
            checkpoint()
        gap = previous is not None and frame["frame_index"] != previous["frame_index"] + 1
        age = {"updates": None, "seconds": None}
        if last is not None:
            age["updates"] = frame["update_index"] - last["update_index"]
            if frame["timestamp_seconds"] is not None and last["timestamp_seconds"] is not None:
                age["seconds"] = frame["timestamp_seconds"] - last["timestamp_seconds"]
        selected, event, checks = None, None, []
        reason = "before_selection"
        if frame["frame_id"] == request["selection"]["frame_id"]:
            selected = next(
                item
                for item in frame["observations"]
                if item["detection_index"] == request["selection"]["detection_index"]
            )
            state, reason, event = "observed", "explicit_selection", "selected"
            pending = None
        elif frame["frame_id"] == request["release_frame_id"]:
            state, reason, event = "released", "explicit_release", "released"
            pending = None
        elif state == "released":
            reason = "selection_released"
        elif state == "expired":
            reason = "selection_expired"
        elif last is not None:
            expired_updates = age["updates"] > policy["max_lost_updates"]
            expired_seconds = (
                policy["max_lost_seconds"] is not None
                and age["seconds"] is not None
                and age["seconds"] > policy["max_lost_seconds"] + 1e-12
            )
            if expired_updates or expired_seconds:
                state, reason, event, pending = (
                    "expired",
                    "seconds_limit" if expired_seconds else "update_limit",
                    "expired",
                    None,
                )
            else:
                candidates = []
                for observation in frame["observations"]:
                    if mode == "guarded_geometry":
                        check = _candidate(last["observation"], observation, policy)
                    else:
                        reasons = []
                        for condition, text in (
                            (not observation["confirmed"], "unconfirmed"),
                            (
                                observation["label_id"] != last["observation"]["label_id"],
                                "different_class",
                            ),
                            (
                                observation["track_id"] != last["observation"]["track_id"],
                                "different_track_id",
                            ),
                            (observation["score"] < policy["min_score"], "low_score"),
                        ):
                            if condition:
                                reasons.append(text)
                        check = {
                            "detection_index": observation["detection_index"],
                            "track_id": observation["track_id"],
                            "eligible": not reasons,
                            "reasons": reasons,
                            "iou": None,
                            "center_distance": None,
                            "area_ratio": None,
                        }
                    checks.append(check)
                    if check["eligible"]:
                        candidates.append(observation)
                if not candidates:
                    state, reason, pending = "lost", "no_compatible_observation", None
                elif len(candidates) > 1:
                    state, reason, pending = "ambiguous", "multiple_compatible_observations", None
                else:
                    candidate = candidates[0]
                    continuous = (
                        state in {"observed", "recovered"}
                        and not gap
                        and candidate["track_id"] == last["observation"]["track_id"]
                    )
                    if mode == "track_id_only" or continuous:
                        selected = candidate
                        event = None if continuous else "recovered"
                        state = "observed" if continuous else "recovered"
                        reason, pending = (
                            "continued_observation" if continuous else "same_track_returned",
                            None,
                        )
                    else:
                        coherent = (
                            pending is not None
                            and not gap
                            and candidate["track_id"] == pending["observation"]["track_id"]
                            and _candidate(pending["observation"], candidate, policy)["eligible"]
                        )
                        pending = {
                            "observation": deepcopy(candidate),
                            "count": pending["count"] + 1 if coherent else 1,
                        }
                        if pending["count"] >= policy["recovery_confirmation_updates"]:
                            selected = candidate
                            state, reason, event, pending = (
                                "recovered",
                                "unique_candidate_confirmed",
                                "recovered",
                                None,
                            )
                        else:
                            state, reason = "recovering", "candidate_needs_confirmation"
        if selected is not None:
            last = _anchor(frame, selected)
            age = {"updates": 0, "seconds": 0.0 if frame["timestamp_seconds"] is not None else None}
        output = {
            **{
                key: deepcopy(frame[key])
                for key in (
                    "frame_id",
                    "frame_index",
                    "timestamp_seconds",
                    "input_size",
                    "update_index",
                )
            },
            "state": state,
            "reason": reason,
            "event": event,
            "logical_object_id": "selection-1" if last is not None else None,
            "selected": deepcopy(selected),
            "last_observed": deepcopy(last),
            "age": age,
            "pending": None
            if pending is None
            else {
                "track_id": pending["observation"]["track_id"],
                "observations": pending["count"],
                "required": policy["recovery_confirmation_updates"],
            },
            "candidates": checks,
            "source_gap": gap,
        }
        charge(output)
        outputs.append(output)
        previous = frame
    return outputs


def _checked_inputs(bundle, request, checkpoint):
    from iris.temporal_contracts import validate_sequence_manifest
    from iris.tracking_contracts import (
        DETECTION_FIELDS,
        profile_hash,
        validate_profile,
        validate_tracking_frame,
    )
    from iris.tracking_selection_contracts import MAX_FRAMES, digest

    sequence = validate_sequence_manifest(bundle["sequence"]["manifest"])
    replay = bundle["replay"]
    if digest(replay["sequence"]) != digest(sequence):
        raise ValueError("Selected-object replay must use the exact frozen sequence")
    profile = validate_profile(replay["profile"])
    if (
        profile_hash(profile) != request["source"]["profile_sha256"]
        or replay["profile_sha256"] != request["source"]["profile_sha256"]
    ):
        raise ValueError("Selected-object source must use the exact frozen tracker profile")
    if request["source"]["sequence_id"] != sequence["id"]:
        raise ValueError("Selected-object source sequence does not match")
    frames = replay["passes"][0]["frames"]
    if not 1 <= len(frames) <= MAX_FRAMES or len(frames) != len(sequence["frames"]):
        raise ValueError("Selected-object replay needs every available frame, at most 500")
    checked = []
    for index, (frame, source) in enumerate(zip(frames, sequence["frames"], strict=True), 1):
        if checkpoint:
            checkpoint()
        if frame["update_index"] != index or frame["sequence_id"] != sequence["id"]:
            raise ValueError("Selected-object source updates must be ordered within one sequence")
        detections = sorted(
            (
                {key: item[key] for key in DETECTION_FIELDS}
                for item in [*frame["observations"], *frame["unassigned"]]
            ),
            key=lambda item: item["detection_index"],
        )
        native = {
            "frame_id": source["frame_id"],
            "frame_index": source["frame_index"],
            "timestamp_seconds": source["timestamp_seconds"],
            "input_size": [source["width"], source["height"]],
            "detections": detections,
        }
        checked.append(validate_tracking_frame(frame, native, profile))
    by_id = {frame["frame_id"]: frame for frame in checked}
    initial = by_id.get(request["selection"]["frame_id"])
    if initial is None:
        raise ValueError("Initial selection must use an available source frame")
    choices = [
        item
        for item in initial["observations"]
        if item["detection_index"] == request["selection"]["detection_index"]
    ]
    if (
        len(choices) != 1
        or not choices[0]["confirmed"]
        or choices[0]["score"] < request["policy"]["min_score"]
    ):
        raise ValueError(
            "Initial selection must be an exact confirmed measured observation above min_score"
        )
    release = request["release_frame_id"]
    if release is not None and (
        release not in by_id or by_id[release]["frame_index"] <= initial["frame_index"]
    ):
        raise ValueError("Release must use an available frame strictly after initial selection")
    return sequence, checked, _anchor(initial, choices[0])


def _summary(frames):
    from iris.tracking_selection_contracts import STATES

    observed = [frame["selected"] for frame in frames if frame["selected"] is not None]
    active = [frame for frame in frames if frame["state"] not in {"idle", "released"}]
    return {
        "states": {state: sum(frame["state"] == state for frame in frames) for state in STATES},
        "selected_frames": len(observed),
        "recovery_events": sum(frame["event"] == "recovered" for frame in frames),
        "track_id_changes": sum(
            first["track_id"] != second["track_id"]
            for first, second in zip(observed, observed[1:], strict=False)
        ),
        "active_frames": len(active),
        "unobserved_frames": sum(frame["selected"] is None for frame in active),
    }


def run_selection(bundle, checkpoint=None):
    """Replay and assess two fixed policies from already verified frozen evidence."""
    import json

    from iris.tracking_selection_contracts import (
        LIMITATIONS,
        MAX_REPORT_BYTES,
        REPORT_SCHEMA,
        canonicalize_request,
    )
    from iris.tracking_selection_metrics import evaluate_selection, prepare_evaluation

    if checkpoint:
        checkpoint()
    request = canonicalize_request(bundle["request"])
    sequence, frames, anchor = _checked_inputs(bundle, request, checkpoint)
    context = prepare_evaluation(request, sequence, bundle["replay"], bundle["reference"], anchor)
    size = 0

    def charge(value):
        nonlocal size
        size += len(json.dumps(value, allow_nan=False).encode("utf-8"))
        if size > MAX_REPORT_BYTES:
            raise ValueError(
                "Selected-object report exceeds its bounded size; choose a shorter sequence"
            )

    mismatch = bundle["replay"]["repeatability"]["status"] == "observed_mismatch"
    lanes = []
    # Only operational fields enter the state machine; evaluation is kept outside.
    decisions = {key: request[key] for key in ("selection", "release_frame_id", "policy")}
    for identifier, name in (
        ("track_id_only", "Track ID only"),
        ("guarded_geometry", "Guarded geometry"),
    ):
        outputs = _advance(frames, decisions, identifier, checkpoint, charge)
        quality = evaluate_selection(
            outputs,
            context,
            unavailable_reason="source_replay_semantic_mismatch" if mismatch else None,
            checkpoint=checkpoint,
        )
        charge(quality)
        lanes.append(
            {
                "id": identifier,
                "name": name,
                "frames": outputs,
                "summary": _summary(outputs),
                "quality": quality,
            }
        )
    reference = bundle["reference"]
    report = {
        "schema": REPORT_SCHEMA,
        "complete": True,
        "request": request,
        "source_binding": deepcopy(bundle["source_binding"]),
        "fingerprint": bundle["fingerprint"],
        "sequence": sequence,
        "profile": deepcopy(bundle["replay"]["profile"]),
        "profile_sha256": bundle["replay"]["profile_sha256"],
        "reference": None
        if reference is None
        else {
            "id": reference["id"],
            "revision": reference["revision"],
            "payload_sha256": reference["payload_sha256"],
            "origin": deepcopy(reference["payload"].get("provenance", {}).get("origin")),
        },
        "repeatability": deepcopy(bundle["replay"]["repeatability"]),
        "lanes": lanes,
        "limitations": list(LIMITATIONS),
    }
    if len(json.dumps(report, allow_nan=False).encode("utf-8")) > MAX_REPORT_BYTES:
        raise ValueError(
            "Selected-object report exceeds its bounded size; choose a shorter sequence"
        )
    if checkpoint:
        checkpoint()
    return report


def validate_report(bundle, report, checkpoint=None):
    """Recompute every decision and metric; a rehashed edit is not valid evidence."""
    from iris.tracking_selection_contracts import digest

    expected = run_selection(bundle, checkpoint=checkpoint)
    if digest(report) != digest(expected):
        raise ValueError("Selected-object report does not match its frozen evidence")
    return expected
