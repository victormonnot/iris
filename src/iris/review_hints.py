"""Read-only review hints from pending proposals and bounded saved detector evidence."""

from iris.annotations import _finite_number
from iris.selection_insights import MAX_PREDICTION_SOURCES, _prediction_signal
from iris.taxonomies import TAXONOMY


def review_hints(conn, frame, taxonomy, suggestions, decisions, disagreement):
    pending = [
        item
        for item in suggestions
        if item["id"] not in decisions
        and isinstance(item.get("metadata"), dict)
        and item["metadata"].get("target_taxonomy", TAXONOMY["id"]) == taxonomy["id"]
        and item["metadata"].get("frame_sha256", frame["sha256"]) == frame["sha256"]
    ]
    uncertain = sum(item["metadata"].get("recommendation") == "uncertain" for item in pending)
    low = sum(
        _finite_number(item["metadata"].get("score")) and 0 <= item["metadata"]["score"] < 0.5
        for item in pending
    )
    candidate_ids = [
        row[0]
        for row in conn.execute(
            "SELECT p.id FROM predictions p JOIN comparisons c ON c.id=p.comparison_id "
            "JOIN jobs j ON j.id=c.job_id WHERE p.frame_id=? AND c.session_id=? "
            "AND j.status='succeeded' ORDER BY p.created_at DESC,p.id DESC LIMIT ?",
            (frame["id"], frame["session_id"], MAX_PREDICTION_SOURCES + 1),
        )
    ]
    saved = _prediction_signal(conn, frame, taxonomy, candidate_ids)
    no_targets = saved.get("no_target_predictions") is True
    unmatched = disagreement["status"] == "disagreement"
    reasons = []
    invalid = sum(not isinstance(item.get("metadata"), dict) for item in suggestions)
    if invalid:
        reasons.append(
            "Some saved proposal metadata is unreadable; inspect those proposals manually."
        )
    if uncertain:
        reasons.append(f"{uncertain} pending candidate reviews marked uncertain by their provider.")
    if low:
        reasons.append(f"{low} pending proposals have a native score below 0.5.")
    if no_targets:
        reasons.append(
            "A compatible saved detector returned no target boxes. Inspect the whole image; "
            "an empty result does not establish an object's absence."
        )
    if unmatched:
        reasons.append(
            "Saved detectors leave boxes unmatched. Inspect for missed objects, false positives "
            "or class and geometry differences; disagreement does not identify the correct result."
        )
    return {
        "protocol": "iris-review-hints-v1",
        "low_confidence_count": low,
        "uncertain_count": uncertain,
        "possible_omission": no_targets or unmatched,
        "reasons": reasons,
        "unreadable_proposal_count": invalid,
        "prediction_id": saved.get("prediction_source_id"),
        "prediction_status": saved["prediction_signal_status"],
        "prediction_reason": saved["prediction_signal_reason"],
        "sources_inspected": saved["prediction_sources_inspected"],
        "sources_truncated": saved["prediction_sources_truncated"],
        "score_note": "Native scores below 0.5 are review hints, not calibrated uncertainty.",
    }
