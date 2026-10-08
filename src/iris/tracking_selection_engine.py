"""Deterministic selected-object state from frozen measured tracker observations.

A local track number is an association hint, never an identity guarantee. The
policy has no reference-annotation input and cannot renew its memory from a
prediction, a rejected candidate or a polling call.
"""

from copy import deepcopy

from .pipeline_selection import SelectionState, _anchor


def _advance(frames, request, mode, checkpoint, charge):
    """Use the same bounded transition machine as portable inference."""
    machine = SelectionState(request["policy"], mode=mode)
    outputs = []
    for frame in frames:
        if checkpoint:
            checkpoint()
        output = machine.update(
            frame,
            select_detection_index=request["selection"]["detection_index"]
            if frame["frame_id"] == request["selection"]["frame_id"]
            else None,
            release=frame["frame_id"] == request["release_frame_id"],
        )
        charge(output)
        outputs.append(output)
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
