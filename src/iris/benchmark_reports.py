"""Immutable, locally saved comparison reports; no model or provider execution."""

from __future__ import annotations

import hashlib
import json
import re

from iris.benchmark import BenchmarkConflict
from iris.store import Store, _decode, new_id, now

PROTOCOL = "iris-benchmark-report-v1"
EVIDENCE_KINDS = {"not_declared", "simulation", "real_data"}
MAX_SNAPSHOT_BYTES = 16 * 1024**2
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_SNAPSHOT_FIELDS = {"protocol", "evidence_kind", "title", "objective", "conclusion", "comparison"}


def _canonical(value):
    try:
        raw = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError("The report must contain finite, valid JSON values") from exc
    if len(raw) > MAX_SNAPSHOT_BYTES:
        raise ValueError("The comparison report exceeds the 16 MiB snapshot limit")
    return raw


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _editorial(*, title, objective, conclusion, evidence_kind):
    result = {}
    for field, value, minimum, maximum in (
        ("title", title, 1, 160),
        ("objective", objective, 0, 4000),
        ("conclusion", conclusion, 0, 4000),
    ):
        if not isinstance(value, str) or not minimum <= len(value.strip()) <= maximum:
            raise ValueError(f"Report {field} must contain {minimum}–{maximum} characters")
        result[field] = value.strip()
    if not isinstance(evidence_kind, str) or evidence_kind not in EVIDENCE_KINDS:
        raise ValueError("Declare report evidence as not_declared, simulation or real_data")
    return {**result, "evidence_kind": evidence_kind}


def _preview(store, benchmark_id, connection, *, role, editorial):
    from iris.benchmark_analysis import build_comparison

    snapshot = {
        "protocol": PROTOCOL,
        **editorial,
        "comparison": build_comparison(store, benchmark_id, role=role, connection=connection),
    }
    return {"benchmark_id": benchmark_id, "snapshot": snapshot, "fingerprint": _digest(snapshot)}


def preview_report(
    store: Store,
    benchmark_id: str,
    *,
    role: str,
    title: str,
    objective: str = "",
    conclusion: str = "",
    evidence_kind: str = "not_declared",
) -> dict:
    """Read one consistent snapshot, including any currently active trial evidence."""
    editorial = _editorial(
        title=title, objective=objective, conclusion=conclusion, evidence_kind=evidence_kind
    )
    with store.connect() as connection:
        connection.execute("BEGIN")
        return _preview(store, benchmark_id, connection, role=role, editorial=editorial)


def create_report(
    store: Store,
    benchmark_id: str,
    *,
    role: str,
    title: str,
    expected_fingerprint: str,
    objective: str = "",
    conclusion: str = "",
    evidence_kind: str = "not_declared",
) -> dict:
    """Compare and save atomically; changed evidence requires a fresh preview."""
    if not isinstance(expected_fingerprint, str) or not _SHA.fullmatch(expected_fingerprint):
        raise ValueError("A report preview fingerprint is required")
    editorial = _editorial(
        title=title, objective=objective, conclusion=conclusion, evidence_kind=evidence_kind
    )
    with store.connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        preview = _preview(store, benchmark_id, connection, role=role, editorial=editorial)
        if preview["fingerprint"] != expected_fingerprint:
            raise BenchmarkConflict("Comparison evidence or report text changed; preview again")
        states = [
            row[0]
            for row in connection.execute(
                "SELECT j.status FROM benchmark_trials t JOIN jobs j ON j.id=t.job_id "
                "WHERE t.benchmark_id=? AND t.split=?",
                (benchmark_id, role),
            )
        ]
        if not states:
            raise BenchmarkConflict("Run at least one trial in this role before saving a report")
        if any(state in {"queued", "running"} for state in states):
            raise BenchmarkConflict("Wait for all trials in this role to finish before saving")
        existing = _decode(
            connection.execute(
                "SELECT * FROM benchmark_reports WHERE benchmark_id=? AND snapshot_sha256=?",
                (benchmark_id, expected_fingerprint),
            ).fetchone()
        )
        if existing is not None:
            return validate_report_row(existing, connection=connection, store=store)
        row = {
            "id": new_id(),
            "benchmark_id": benchmark_id,
            "snapshot": preview["snapshot"],
            "snapshot_sha256": preview["fingerprint"],
            "created_at": now(),
        }
        validate_report_row(row, connection=connection, store=store)
        connection.execute(
            "INSERT INTO benchmark_reports "
            "(id,benchmark_id,snapshot,snapshot_sha256,created_at) VALUES (?,?,?,?,?)",
            (
                row["id"],
                benchmark_id,
                _canonical(row["snapshot"]).decode("utf-8"),
                row["snapshot_sha256"],
                row["created_at"],
            ),
        )
        return row


def validate_report_row(row: dict, *, connection=None, manifest=None, store=None) -> dict:
    """Validate saved bytes and their original sources without consulting latest results."""
    from iris.benchmark_analysis import validate_comparison_snapshot, validate_snapshot_sources

    snapshot = row.get("snapshot")
    if (
        not isinstance(snapshot, dict)
        or set(snapshot) != _SNAPSHOT_FIELDS
        or snapshot.get("protocol") != PROTOCOL
        or not isinstance(row.get("snapshot_sha256"), str)
        or _SHA.fullmatch(row["snapshot_sha256"]) is None
        or _digest(snapshot) != row["snapshot_sha256"]
    ):
        raise ValueError("Benchmark report snapshot or checksum is invalid")
    editorial = _editorial(
        **{field: snapshot[field] for field in _SNAPSHOT_FIELDS - {"protocol", "comparison"}}
    )
    if any(snapshot[field] != value for field, value in editorial.items()):
        raise ValueError("Benchmark report text is not canonical")
    comparison = snapshot["comparison"]
    validate_comparison_snapshot(comparison)
    if comparison["benchmark"]["id"] != row.get("benchmark_id"):
        raise ValueError("Benchmark report has a different benchmark owner")
    trials = [trial for config in comparison["configs"] for trial in config["trials"]]
    if not trials or any(trial["status"] in {"queued", "running"} for trial in trials):
        raise ValueError("A saved benchmark report requires terminal trials in its selected role")
    if connection is not None:
        validate_snapshot_sources(
            connection,
            comparison,
            manifest=manifest,
            store=store,
            captured_at=row["created_at"],
        )
    return row


def get_report(store: Store, report_id: str) -> dict | None:
    with store.connect() as connection:
        connection.execute("BEGIN")
        row = _decode(
            connection.execute(
                "SELECT * FROM benchmark_reports WHERE id=?", (report_id,)
            ).fetchone()
        )
        return (
            validate_report_row(row, connection=connection, store=store)
            if row is not None
            else None
        )


def list_reports(store: Store, benchmark_id: str) -> list[dict]:
    """Return compact summaries; each historical snapshot is checked before exposure."""
    with store.connect() as connection:
        connection.execute("BEGIN")
        reports = []
        for raw in connection.execute(
            "SELECT * FROM benchmark_reports WHERE benchmark_id=? ORDER BY created_at,id",
            (benchmark_id,),
        ):
            row = validate_report_row(_decode(raw), connection=connection, store=store)
            snapshot = row["snapshot"]
            reports.append(
                {key: row[key] for key in ("id", "benchmark_id", "snapshot_sha256", "created_at")}
                | {
                    "title": snapshot["title"],
                    "role": snapshot["comparison"]["role"],
                    "evidence_kind": snapshot["evidence_kind"],
                }
            )
        return reports
