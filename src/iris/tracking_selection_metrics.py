"""Selected-object measurements against an explicit frozen reference identity.

Reference annotations are consumed only here, after policy decisions. Unknown
frames and unlocalized occlusion cannot become negative examples or durations.
"""

from copy import deepcopy

from iris.temporal_contracts import validate_reference
from iris.tracking_metrics import _iou
from iris.tracking_selection_contracts import digest

CATEGORIES = (
    "correct_target",
    "wrong_other_identity",
    "unmatched_selected_box",
    "reference_ambiguous",
    "abstained_visible",
    "abstained_absent",
)


def _eligible_reference(frame, label):
    reasons = []
    if frame is None:
        return ["missing_reference_frame"]
    if frame["review"]["status"] != "human_reviewed":
        reasons.append("not_human_reviewed")
    if frame["coverage"] != "complete":
        reasons.append("incomplete_frame_review")
    objects = [obj for obj in frame["objects"] if obj["label"] == label]
    if any(
        obj["certainty"] != "certain"
        or obj["identity_id"] is None
        or obj["visibility"] == "unknown"
        or (obj["visibility"] == "occluded" and obj["box"] is None)
        for obj in objects
    ):
        reasons.append("unlocalized_or_uncertain_reference")
    return reasons


def _boxes(frame, label):
    return [
        obj
        for obj in frame["objects"]
        if obj["label"] == label
        and obj["visibility"] in {"visible", "occluded"}
        and obj["box"] is not None
    ]


def prepare_evaluation(request, sequence, replay, reference, anchor):
    """Check the independent assessment configuration and initial selected box."""
    config = request["evaluation"]
    if config is None:
        if reference is not None:
            raise ValueError("An unevaluated scenario must not attach a reference")
        return None
    if not isinstance(reference, dict) or reference.get("id") != config["reference_id"]:
        raise ValueError("Evaluation must pin the requested reference revision")
    payload = validate_reference(reference["payload"], sequence)
    if digest(reference["payload"]) != reference["payload_sha256"]:
        raise ValueError("Reference checksum does not match the frozen payload")
    expected = {str(value) for value in replay["profile"]["class_ids"]}
    if set(config["class_mapping"]) != expected:
        raise ValueError("Map every native tracker class explicitly")
    labels = {item["id"] for item in sequence["taxonomy"]["classes"]}
    if any(value is not None and value not in labels for value in config["class_mapping"].values()):
        raise ValueError("Evaluation mapping must use the frozen sequence taxonomy")
    identities = {item["id"]: item["label"] for item in payload["identities"]}
    identity = config["identity_id"]
    if identity not in identities:
        raise ValueError("Choose an identity declared by the frozen reference")
    label = identities[identity]
    if config["class_mapping"][str(anchor["observation"]["label_id"])] != label:
        raise ValueError(
            "Selected observation class does not map to the requested reference identity"
        )
    reference_frames = {frame["frame_index"]: frame for frame in payload["frames"]}
    initial = reference_frames.get(anchor["frame_index"])
    if _eligible_reference(initial, label):
        raise ValueError(
            "Initial selection requires a human-complete geometrically evaluable reference frame"
        )
    matches = [
        obj["identity_id"]
        for obj in _boxes(initial, label)
        if _iou(anchor["observation"]["box"], obj["box"]) >= config["iou_threshold"]
    ]
    if matches != [identity]:
        raise ValueError(
            "Initial selected box must uniquely match the requested reference identity"
        )
    return {
        "payload": payload,
        "reference_frames": reference_frames,
        "identity_id": identity,
        "label": label,
        "iou_threshold": config["iou_threshold"],
    }


def evaluate_selection(frames, context, *, unavailable_reason=None, checkpoint=None):
    active = [row for row in frames if row["state"] not in {"idle", "released"}]
    target = context["identity_id"] if context else None
    reason = unavailable_reason or ("no_reference_selected" if context is None else None)
    counts = dict.fromkeys(CATEGORIES, 0)
    evaluated, excluded, rows = [], [], []
    visible_frames = 0
    for output in active:
        if checkpoint:
            checkpoint()
        index = output["frame_index"]
        reference = context["reference_frames"].get(index) if context else None
        reasons = [reason] if reason else _eligible_reference(reference, context["label"])
        if reasons:
            excluded.append({"frame_index": index, "reasons": reasons})
            rows.append(
                {
                    "frame_index": index,
                    "category": "unavailable",
                    "matched_identity": None,
                    "reasons": reasons,
                }
            )
            continue
        objects = _boxes(reference, context["label"])
        target_visible = any(obj["identity_id"] == target for obj in objects)
        selected = output["selected"]
        matches = (
            []
            if selected is None
            else [
                obj["identity_id"]
                for obj in objects
                if _iou(selected["box"], obj["box"]) >= context["iou_threshold"]
            ]
        )
        if selected is None:
            category = "abstained_visible" if target_visible else "abstained_absent"
        elif len(matches) > 1:
            category = "reference_ambiguous"
        elif not matches:
            category = "unmatched_selected_box"
        else:
            category = "correct_target" if matches[0] == target else "wrong_other_identity"
        counts[category] += 1
        reasons = ["geometric_reference_ambiguity"] if category == "reference_ambiguous" else []
        rows.append(
            {
                "frame_index": index,
                "category": category,
                "matched_identity": matches[0] if len(matches) == 1 else None,
                "reasons": reasons,
            }
        )
        if reasons:
            excluded.append({"frame_index": index, "reasons": reasons})
        else:
            evaluated.append(index)
            visible_frames += int(target_visible)
    by_index = {row["frame_index"]: row for row in rows}
    evaluated_set = set(evaluated)
    supported, excluded_intervals, elapsed, unobserved, wrong = 0, 0, 0.0, 0.0, 0.0
    for previous, current in zip(active, active[1:], strict=False):
        first, second = previous["frame_index"], current["frame_index"]
        start, end = previous["timestamp_seconds"], current["timestamp_seconds"]
        if (
            first not in evaluated_set
            or second not in evaluated_set
            or second != first + 1
            or start is None
            or end is None
        ):
            excluded_intervals += 1
            continue
        supported += 1
        duration = end - start
        elapsed += duration
        if previous["selected"] is None:
            unobserved += duration
        if by_index[first]["category"] == "wrong_other_identity":
            wrong += duration
    events = []
    previous_selected = None
    for position, output in enumerate(active):
        if output["event"] == "recovered":
            episode = (
                active[previous_selected : position + 1] if previous_selected is not None else []
            )
            supported_episode = (
                bool(episode)
                and all(row["frame_index"] in evaluated_set for row in episode)
                and all(
                    second["frame_index"] == first["frame_index"] + 1
                    for first, second in zip(episode, episode[1:], strict=False)
                )
            )
            category = by_index[output["frame_index"]]["category"]
            outcome = "unavailable"
            if supported_episode:
                if category == "correct_target":
                    outcome = "correct"
                elif category in {"wrong_other_identity", "unmatched_selected_box"}:
                    outcome = "wrong"
            events.append({"frame_index": output["frame_index"], "outcome": outcome})
        if output["selected"] is not None:
            previous_selected = position
    selected_count = sum(
        counts[key] for key in ("correct_target", "wrong_other_identity", "unmatched_selected_box")
    )
    return {
        "status": "available" if evaluated else "unavailable",
        "reason": None if evaluated else reason or "no_evaluable_frames",
        "target_identity": target,
        "coverage": {
            "active_frames": len(active),
            "evaluated_frames": len(evaluated),
            "excluded_frames": len(excluded),
            "evaluated_frame_indices": evaluated,
            "excluded": excluded,
            "reference_origin": deepcopy(context["payload"].get("provenance", {}).get("origin"))
            if context
            else None,
        },
        "counts": counts,
        "rates": {
            "target_agreement": counts["correct_target"] / visible_frames
            if visible_frames
            else None,
            "wrong_identity_among_selected": counts["wrong_other_identity"] / selected_count
            if selected_count
            else None,
            "visible_abstention": counts["abstained_visible"] / visible_frames
            if visible_frames
            else None,
        },
        "durations": {
            "supported_intervals": supported,
            "excluded_intervals": excluded_intervals,
            "evaluated_seconds": elapsed if supported else None,
            "unobserved_seconds": unobserved if supported else None,
            "wrong_identity_seconds": wrong if supported else None,
            "convention": "left_sample_on_consecutive_evaluable_source_frames",
        },
        "recoveries": {
            "total": len(events),
            **{
                key: sum(event["outcome"] == key for event in events)
                for key in ("correct", "wrong", "unavailable")
            },
            "events": events,
        },
        "frames": rows,
    }
