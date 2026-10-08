"""Immutable quality measurements of saved tracking and reference revisions.

Only frozen JSON evidence is evaluated. This service never imports an inference
runtime, executes a tracker, changes a reference, or follows its latest revision.
"""

import json

from iris.store import DEFAULT_PROJECT_ID, _decode, new_id, now
from iris.temporal import _digest, _insert, _reference_record, _row
from iris.tracking_comparisons import _checked_job, _detail
from iris.tracking_metrics import evaluate_quality, quality_status

# Leave room below the archive's 64 MiB JSON-column bound, including the escaping
# performed by Store._encode. Oversized results fail without any published row.
MAX_REPORT_BYTES = 48 * 1024**2


def status():
    return {**quality_status(), "max_report_bytes": MAX_REPORT_BYTES}


def _sources(conn, comparison_id, reference_id, *, project_id=None):
    job, _, sequence = _checked_job(conn, comparison_id)
    if project_id is not None and sequence["project_id"] != project_id:
        raise KeyError(comparison_id)
    if job["status"] != "succeeded":
        raise ValueError("Tracking quality requires a complete saved comparison")
    reference_row = _row(conn, "temporal_references", reference_id)
    if reference_row["sequence_id"] != sequence["id"]:
        raise ValueError("Choose a reference revision from the exact comparison sequence")
    reference = _reference_record(conn, reference_row, sequence)
    return _detail(job, include_report=True), reference, sequence


def _bounded(report):
    if len(json.dumps(report, allow_nan=False).encode("utf-8")) > MAX_REPORT_BYTES:
        raise ValueError("Tracking quality report is too large; use a shorter frozen sequence")


def _record(conn, row):
    """Reproduce the saved protocol result; a rehashed edit is not valid evidence."""
    try:
        config, report = row["config"], row["report"]
        if not isinstance(config, dict) or set(config) != {"class_mapping", "iou_threshold"}:
            raise ValueError("Tracking quality configuration is invalid")
        _bounded(report)
        if _digest(report) != row["report_sha256"]:
            raise ValueError("Tracking quality report checksum is invalid")
        comparison, reference, sequence = _sources(conn, row["comparison_id"], row["reference_id"])
        expected = evaluate_quality(comparison, reference, **config)
        canonical_config = {
            "class_mapping": expected["protocol"]["class_mapping"],
            "iou_threshold": expected["protocol"]["iou_threshold"],
        }
        if (
            row["sequence_id"] != sequence["id"]
            or _digest(config) != _digest(canonical_config)
            or _digest(expected) != row["report_sha256"]
        ):
            raise ValueError("Tracking quality report does not match its frozen evidence")
        return row
    except (KeyError, TypeError, IndexError, OverflowError) as exc:
        raise ValueError("Tracking quality evidence is invalid or missing") from exc


def create_quality_report(
    store,
    comparison_id,
    *,
    reference_id,
    class_mapping,
    iou_threshold=0.5,
    project_id=DEFAULT_PROJECT_ID,
):
    # The bounded pure computation and publication share one write transaction:
    # interruption or validation failure leaves no partial row, and no source can
    # change between validation and publication.
    with store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        comparison, reference, sequence = _sources(
            conn, comparison_id, reference_id, project_id=project_id
        )
        report = evaluate_quality(
            comparison, reference, class_mapping=class_mapping, iou_threshold=iou_threshold
        )
        _bounded(report)
        row = {
            "id": new_id(),
            "comparison_id": comparison_id,
            "sequence_id": sequence["id"],
            "reference_id": reference_id,
            "config": {
                "class_mapping": report["protocol"]["class_mapping"],
                "iou_threshold": report["protocol"]["iou_threshold"],
            },
            "report": report,
            "report_sha256": _digest(report),
            "created_at": now(),
        }
        _insert(conn, "tracking_quality_reports", row)
        return row


def get_quality_report(store, report_id, *, project_id=DEFAULT_PROJECT_ID):
    with store.connect() as conn:
        conn.execute("BEGIN")
        row = _row(conn, "tracking_quality_reports", report_id)
        sequence = _row(conn, "temporal_sequences", row["sequence_id"])
        if sequence["project_id"] != project_id:
            raise KeyError(report_id)
        return _record(conn, row)


def list_quality_reports(store, comparison_id, *, project_id=DEFAULT_PROJECT_ID):
    with store.connect() as conn:
        conn.execute("BEGIN")
        _, _, sequence = _checked_job(conn, comparison_id)
        if sequence["project_id"] != project_id:
            raise KeyError(comparison_id)
        records = []
        for raw in conn.execute(
            "SELECT * FROM tracking_quality_reports WHERE comparison_id=? ORDER BY created_at,id",
            (comparison_id,),
        ):
            row = _record(conn, _decode(raw))
            records.append(
                {
                    **row,
                    "report": {
                        **row["report"],
                        "lanes": [
                            {key: value for key, value in lane.items() if key != "frames"}
                            for lane in row["report"]["lanes"]
                        ],
                    },
                }
            )
        return records


def validate_tracking_quality_records(connection):
    for row in connection.execute("SELECT * FROM tracking_quality_reports"):
        _record(connection, _decode(row))
