"""Durable one-attempt dispatch receipts, separate from job and publication success."""

from __future__ import annotations

import json

from iris.store import Store, _decode, new_id, now

_TABLES = {"assist": "assistance_records", "video_review": "video_reviews"}
_STATES = {"not_started", "dispatching", "response_received", "outcome_unknown"}
_MESSAGES = {
    "not_started": "No provider dispatch is recorded for this request.",
    "dispatching": "A provider request may be in progress; do not submit it again.",
    "response_received": (
        "A complete provider response was saved; "
        "this does not establish valid proposals or final billing."
    ),
    "outcome_unknown": (
        "The provider may have processed this request. Its outcome and possible charge "
        "are unknown; do not automatically resend it."
    ),
}


class DispatchConflict(ValueError):
    """A record has already claimed its sole generation attempt or is no longer active."""


def initial_dispatch(provider_config: dict) -> dict:
    """Record a known unsent request in the transaction that creates its job."""
    name = provider_config.get("provider", "ollama")
    return {
        "provider": name,
        "model": provider_config.get("model"),
        "external": name != "ollama",
        "state": "not_started",
        "attempted_at": None,
        "response_received_at": None,
    }


def _record(conn, table, record_id):
    if table not in _TABLES.values():
        raise ValueError("Unsupported dispatch record")
    row = _decode(conn.execute(f"SELECT * FROM {table} WHERE id=?", (record_id,)).fetchone())
    if row is None:
        raise ValueError("Dispatch record not found")
    return row


def _identity(record):
    provider = record["config"].get("provider", {})
    name = provider.get("provider", "ollama")
    return {"provider": name, "model": provider.get("model"), "external": name != "ollama"}


def _legacy(record, job):
    identity = _identity(record)
    metadata = record.get("metadata") or {}
    attempted = metadata.get("attempted_at") or job.get("started_at")
    evidence = attempted or record.get("raw_response") is not None or record.get("error")
    state = (
        "outcome_unknown"
        if identity["external"] and evidence
        else "response_received"
        if record.get("raw_response") is not None
        else "outcome_unknown"
        if attempted
        else "not_started"
    )
    return {
        **identity,
        "state": state,
        "attempted_at": attempted,
        "response_received_at": None,
        "legacy": True,
    }


def _ledger(record, job):
    saved = (record.get("metadata") or {}).get("dispatch")
    if saved is None:
        return _legacy(record, job)
    identity = _identity(record)
    if (
        not isinstance(saved, dict)
        or saved.get("state") not in _STATES
        or any(saved.get(key) != value for key, value in identity.items())
    ):
        return {
            **identity,
            "state": "outcome_unknown",
            "attempted_at": None,
            "response_received_at": None,
        }
    return dict(saved)


def _metadata(record, incoming=None):
    current = dict(record.get("metadata") or {})
    if incoming is not None:
        if not isinstance(incoming, dict):
            raise ValueError("Provider metadata must be an object")
        current.update({key: value for key, value in incoming.items() if key != "dispatch"})
    return current


def _write(conn, table, record_id, updates):
    allowed = {"metadata", "prompt", "raw_response", "error"}
    if not updates or not set(updates) <= allowed:
        raise ValueError("Invalid dispatch record update")
    encoded = {
        key: json.dumps(value, allow_nan=False) if key in {"metadata", "raw_response"} else value
        for key, value in updates.items()
    }
    conn.execute(
        f"UPDATE {table} SET " + ",".join(f"{key}=?" for key in encoded) + " WHERE id=?",
        (*encoded.values(), record_id),
    )


def update_dispatch_record(store: Store, table: str, record_id: str, updates: dict):
    """Save preflight evidence only while no executor owns the generation attempt."""
    with store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        record = _record(conn, table, record_id)
        metadata = _metadata(record)
        saved = metadata.get("dispatch", {})
        job = conn.execute(
            "SELECT status,cancel_requested FROM jobs WHERE id=?", (record["job_id"],)
        ).fetchone()
        if (
            not isinstance(saved, dict)
            or saved.get("attempt_id")
            or saved.get("state", "not_started") != "not_started"
            or metadata.get("attempted_at")
            or record.get("raw_response") is not None
            or job is None
            or job["status"] not in {"queued", "running"}
            or job["cancel_requested"]
        ):
            raise DispatchConflict("This request is already claimed or stopped")
        values = dict(updates)
        if "metadata" in values:
            values["metadata"] = _metadata(record, values["metadata"])
        _write(conn, table, record_id, values)


def claim_dispatch(store: Store, table: str, record_id: str) -> str:
    """Claim one generation attempt before any reviewer can dispatch its payload."""
    with store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        record = _record(conn, table, record_id)
        job = _decode(conn.execute("SELECT * FROM jobs WHERE id=?", (record["job_id"],)).fetchone())
        metadata = _metadata(record)
        saved = metadata.get("dispatch")
        if (
            job is None
            or _TABLES.get(job["kind"]) != table
            or job["status"] not in {"queued", "running"}
            or job["cancel_requested"]
            or record.get("raw_response") is not None
            or record.get("error")
            or metadata.get("attempted_at")
            or saved is not None
            and (
                not isinstance(saved, dict)
                or saved.get("state") != "not_started"
                or saved.get("attempt_id")
            )
        ):
            raise DispatchConflict(
                "This request was already attempted or stopped; it cannot be sent again"
            )
        attempt = new_id()
        metadata["dispatch"] = {
            **_identity(record),
            "state": "not_started",
            "attempt_id": attempt,
            "claimed_at": now(),
            "attempted_at": None,
            "response_received_at": None,
        }
        _write(conn, table, record_id, {"metadata": metadata})
    return attempt


def mark_dispatched(store: Store, table: str, record_id: str, attempt_id: str):
    """Commit before entering the transport; no provider-side idempotency is assumed."""
    with store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        record = _record(conn, table, record_id)
        job = conn.execute(
            "SELECT status,cancel_requested FROM jobs WHERE id=?", (record["job_id"],)
        ).fetchone()
        metadata = _metadata(record)
        saved = metadata.get("dispatch", {})
        if saved.get("attempt_id") != attempt_id or saved.get("state") != "not_started":
            raise DispatchConflict("This provider dispatch was already started")
        if job is None or job["status"] not in {"queued", "running"} or job["cancel_requested"]:
            raise DispatchConflict("The request stopped before provider dispatch")
        saved.update(state="dispatching", attempted_at=now())
        if table == "video_reviews":
            metadata["attempted_at"] = saved["attempted_at"]
        _write(conn, table, record_id, {"metadata": metadata})


def record_dispatch_outcome(
    store: Store,
    table: str,
    record_id: str,
    attempt_id: str,
    updates: dict,
    *,
    response_received: bool,
):
    """Save response evidence and its receipt together, even after cancellation."""
    with store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        record = _record(conn, table, record_id)
        values = dict(updates)
        metadata = _metadata(record, values.pop("metadata", None))
        saved = metadata.get("dispatch", {})
        if saved.get("attempt_id") != attempt_id:
            raise DispatchConflict("Dispatch receipt belongs to another attempt")
        if response_received:
            saved.update(state="response_received", response_received_at=now())
            provider_metadata = metadata.get("provider")
            request_id = metadata.get("request_id")
            if not request_id and isinstance(provider_metadata, dict):
                request_id = provider_metadata.get("request_id")
            if isinstance(request_id, str) and request_id:
                saved["request_id"] = request_id
        elif saved.get("state") == "dispatching":
            saved["state"] = "outcome_unknown"
        _write(conn, table, record_id, {**values, "metadata": metadata})


def dispatch_summary(store: Store, job: dict) -> dict | None:
    if job.get("kind") == "benchmark":
        from iris.benchmark_dispatch import dispatch_summary as benchmark_dispatch_summary

        trials = store.list("benchmark_trials", job_id=job["id"])
        if trials and trials[0]["config"].get("external_plan"):
            return benchmark_dispatch_summary(store, trials[0]["id"])
        return None
    table = _TABLES.get(job.get("kind"))
    if table is None:
        return None
    with store.connect() as conn:
        row = _decode(
            conn.execute(f"SELECT * FROM {table} WHERE job_id=?", (job["id"],)).fetchone()
        )
    if row is None:
        return None
    ledger = _ledger(row, job)
    state = ledger["state"]
    if state == "dispatching" and job["status"] not in {"queued", "running"}:
        state = "outcome_unknown"
    result = {
        key: ledger.get(key)
        for key in ("provider", "model", "external", "attempted_at", "response_received_at")
    }
    result.update(state=state, message=_MESSAGES[state])
    if ledger.get("request_id"):
        result["request_id"] = ledger["request_id"]
    return result


def reconcile_dispatches(store: Store) -> None:
    """Call after interrupting jobs on shutdown/startup; never send or clear evidence."""
    from iris.benchmark_dispatch import recover_benchmark_dispatches

    recover_benchmark_dispatches(store)
    with store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        for kind, table in _TABLES.items():
            for raw in conn.execute(
                f"SELECT r.* FROM {table} r JOIN jobs j ON j.id=r.job_id "
                "WHERE j.kind=? AND j.status NOT IN ('queued','running')",
                (kind,),
            ).fetchall():
                record = _decode(raw)
                job = _decode(
                    conn.execute("SELECT * FROM jobs WHERE id=?", (record["job_id"],)).fetchone()
                )
                saved = _ledger(record, job)
                if saved["state"] == "dispatching":
                    saved["state"] = "outcome_unknown"
                elif (record.get("metadata") or {}).get("dispatch") is not None or saved[
                    "state"
                ] != "outcome_unknown":
                    continue
                metadata = _metadata(record)
                metadata["dispatch"] = saved
                _write(conn, table, record["id"], {"metadata": metadata})
