"""Independent correction revisions and acknowledged review-time intervals.

Reference labels never enter the correction document. Timing measures explicit
review intervals, not estimated labor, and never credits an expired lease or a
server's downtime. Candidate outputs and reference snapshots remain immutable.
"""

import hashlib
import json
import time
import uuid
from datetime import UTC, datetime, timedelta

from iris.annotations import MAX_BOXES, _coordinates
from iris.store import Store, _decode, _encode, new_id, now

PROCESS_ID = uuid.uuid4().hex
LEASE_SECONDS = 30
TIMING_PROTOCOL = "iris-review-intervals-v1"


class CorrectionConflict(RuntimeError):
    """Saved corrections or timing ownership changed; refresh before another edit."""


def _clock():
    return datetime.now(UTC), time.monotonic()


def _context(store, output_id):
    from iris.benchmark import benchmark_frame, load_benchmark_manifest, open_benchmark_image

    output = store.get("benchmark_outputs", output_id)
    if output is None:
        raise KeyError(output_id)
    trial = store.get("benchmark_trials", output["trial_id"])
    manifest = load_benchmark_manifest(store, trial["benchmark_id"])
    frame = benchmark_frame(store, trial["benchmark_id"], output["frame_id"])
    with open_benchmark_image(store, frame):
        pass
    result = output["result"]
    if (
        output["error"]
        or not isinstance(result, dict)
        or not isinstance(result.get("proposals"), list)
    ):
        raise ValueError(
            "This candidate did not return valid proposals; no correction can be scored"
        )
    return output, trial, frame, manifest["taxonomy"], result["proposals"]


def _latest(conn, output_id):
    return _decode(
        conn.execute(
            "SELECT * FROM benchmark_corrections WHERE output_id=? ORDER BY revision DESC LIMIT 1",
            (output_id,),
        ).fetchone()
    )


def _timer(conn, output_id):
    return _decode(
        conn.execute("SELECT * FROM benchmark_timers WHERE output_id=?", (output_id,)).fetchone()
    )


def _expired(timer, monotonic):
    last = timer["metadata"].get("last_monotonic")
    return timer["state"] == "running" and (
        timer["metadata"].get("process_id") != PROCESS_ID
        or type(last) not in (int, float)
        or not 0 <= monotonic - last <= LEASE_SECONDS
    )


def _public_timer(timer, clock=None):
    if timer is None:
        return {
            "revision": 0,
            "state": "unmeasured",
            "elapsed_ms": None,
            "recorded_segments": 0,
            "owner_token": None,
            "lease_expires_at": None,
            "fully_timed": False,
            "protocol": TIMING_PROTOCOL,
        }
    _, monotonic = clock or _clock()
    metadata = timer["metadata"]
    expired = _expired(timer, monotonic)
    running = timer["state"] == "running" and not expired
    return {
        "revision": timer["revision"],
        "state": "running" if running else "paused",
        "elapsed_ms": timer["elapsed_ms"],
        "recorded_segments": len(timer["segments"]),
        "owner_token": metadata.get("owner_token"),
        "reviewer": timer["reviewer"],
        "lease_expires_at": metadata.get("lease_expires_at") if running else None,
        "fully_timed": not expired and not metadata.get("unmeasured_gap", False),
        "interruption_reason": "Review lease expired; only confirmed intervals were retained"
        if expired
        else metadata.get("interruption_reason"),
        "protocol": TIMING_PROTOCOL,
    }


def _write_timer(conn, timer):
    encoded = _encode({key: value for key, value in timer.items() if key != "id"})
    conn.execute(
        f"UPDATE benchmark_timers SET {','.join(f'{key}=?' for key in encoded)} WHERE id=?",
        (*encoded.values(), timer["id"]),
    )


def _credit(timer, clock):
    wall, monotonic = clock
    if timer["state"] != "running":
        return
    if _expired(timer, monotonic):
        timer["state"] = "paused"
        timer["metadata"].update(
            unmeasured_gap=True,
            interruption_reason="Review lease expired; unconfirmed time omitted",
        )
        return
    elapsed = (monotonic - timer["metadata"]["last_monotonic"]) * 1000
    timer["elapsed_ms"] += elapsed
    segment = timer["segments"][-1]
    segment["elapsed_ms"] += elapsed
    segment["confirmed_until"] = wall.isoformat()
    timer["metadata"].update(
        last_monotonic=monotonic,
        lease_expires_at=(wall + timedelta(seconds=LEASE_SECONDS)).isoformat(),
    )


def timer_action(
    store: Store,
    output_id: str,
    *,
    action: str,
    expected_revision: int,
    token: str,
    operation_id: str,
    reviewer: str = "",
    discard_unconfirmed: bool = False,
) -> dict:
    _context(store, output_id)
    if action not in {"start", "heartbeat", "pause"}:
        raise ValueError("Choose start, heartbeat or pause")
    if type(discard_unconfirmed) is not bool or (discard_unconfirmed and action != "pause"):
        raise ValueError("Unconfirmed review time can only be discarded when pausing")
    if type(expected_revision) is not int or expected_revision < 0:
        raise ValueError("A timing revision is required")
    if any(
        not isinstance(value, str) or not 16 <= len(value) <= 128 for value in (token, operation_id)
    ):
        raise ValueError("Timing needs bounded owner and operation IDs")
    if not isinstance(reviewer, str) or len(reviewer.strip()) > 120:
        raise ValueError("Reviewer must contain at most 120 characters")
    reviewer = reviewer.strip()
    if action == "start" and not reviewer:
        raise ValueError("Start review with a reviewer name")
    request_hash = hashlib.sha256(
        json.dumps(
            [action, expected_revision, token, reviewer, discard_unconfirmed],
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    clock = _clock()
    stamp = clock[0].isoformat()
    with store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        timer = _timer(conn, output_id)
        operations = timer["metadata"].get("operations", {}) if timer else {}
        if operation_id in operations:
            receipt = operations[operation_id]
            if receipt["request_hash"] != request_hash:
                raise CorrectionConflict("The timing operation ID was reused with different inputs")
            return _public_timer(timer, clock)
        if (timer["revision"] if timer else 0) != expected_revision:
            raise CorrectionConflict("Review timing changed; refresh its saved state")
        if timer is None:
            if action != "start":
                raise ValueError("Start the review timer first")
            timer = {
                "id": new_id(),
                "output_id": output_id,
                "reviewer": reviewer,
                "state": "paused",
                "revision": 0,
                "elapsed_ms": 0.0,
                "segments": [],
                "metadata": {},
                "created_at": stamp,
                "updated_at": stamp,
            }
            encoded = _encode(timer)
            conn.execute(
                f"INSERT INTO benchmark_timers ({','.join(encoded)}) "
                f"VALUES ({','.join('?' for _ in encoded)})",
                tuple(encoded.values()),
            )
        running = timer["state"] == "running" and not _expired(timer, clock[1])
        if running and timer["metadata"].get("owner_token") != token:
            raise CorrectionConflict("Another tab owns this active review timer")
        if action != "start" and timer["metadata"].get("owner_token") != token:
            raise CorrectionConflict("This tab does not own the review timer")
        if discard_unconfirmed:
            timer["state"] = "paused"
            timer["metadata"].update(
                unmeasured_gap=True,
                interruption_reason="Editor lost its timing receipt; unconfirmed time omitted",
            )
        else:
            _credit(timer, clock)
        if action == "start":
            if running:
                raise CorrectionConflict("The review is already running; use its current timer")
            if len(timer["segments"]) >= 2000:
                raise ValueError("This review reached the limit of 2000 recorded intervals")
            timer["state"] = "running"
            timer["reviewer"] = reviewer
            timer["segments"].append(
                {
                    "started_at": stamp,
                    "confirmed_until": stamp,
                    "elapsed_ms": 0.0,
                    "reviewer": reviewer,
                }
            )
            timer["metadata"].update(
                owner_token=token,
                process_id=PROCESS_ID,
                last_monotonic=clock[1],
                lease_expires_at=(clock[0] + timedelta(seconds=LEASE_SECONDS)).isoformat(),
            )
        elif action == "pause":
            timer["state"] = "paused"
        timer["revision"] += 1
        timer["updated_at"] = stamp
        response = _public_timer(timer, clock)
        operations[operation_id] = {"request_hash": request_hash, "response": response}
        timer["metadata"]["operations"] = dict(list(operations.items())[-256:])
        _write_timer(conn, timer)
    return response


def recover_timers(store: Store):
    """Keep acknowledged time only when a workspace starts or its server stops."""
    with store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        for row in conn.execute("SELECT * FROM benchmark_timers WHERE state='running'").fetchall():
            timer = _decode(row)
            timer["state"] = "paused"
            timer["revision"] += 1
            timer["metadata"].update(
                unmeasured_gap=True,
                interruption_reason="Server stopped; only confirmed review intervals were retained",
            )
            timer["updated_at"] = now()
            _write_timer(conn, timer)


def correction_document(store: Store, output_id: str) -> dict:
    output, trial, frame, taxonomy, proposals = _context(store, output_id)
    with store.connect() as conn:
        conn.execute("BEGIN")
        latest = _latest(conn, output_id)
        timer = _timer(conn, output_id)
        history = [
            dict(row)
            for row in conn.execute(
                "SELECT revision,status,reviewer,created_at FROM benchmark_corrections "
                "WHERE output_id=? ORDER BY revision DESC",
                (output_id,),
            )
        ]
    return {
        "output_id": output_id,
        "trial_id": trial["id"],
        "benchmark_id": trial["benchmark_id"],
        "frame": {
            "id": frame["frame_id"],
            "width": frame["width"],
            "height": frame["height"],
            "image_url": (
                f"/api/benchmarks/{trial['benchmark_id']}/frames/{output['frame_id']}/image"
            ),
        },
        "taxonomy": taxonomy,
        "proposals": proposals,
        "revision": latest["revision"] if latest else 0,
        "status": latest["status"] if latest else "unreviewed",
        "boxes": latest["boxes"]
        if latest
        else [
            {
                "id": proposal["id"],
                "label": proposal["label"],
                "box": proposal["box"],
                "proposal_id": proposal["id"],
            }
            for proposal in proposals
        ],
        "decisions": latest["decisions"] if latest else {},
        "reviewer": latest["reviewer"] if latest else "",
        "notes": latest["notes"] if latest else "",
        "history": history,
        "timer": _public_timer(timer),
        "timing": latest["timing"] if latest else None,
    }


def save_correction(
    store: Store,
    output_id: str,
    *,
    expected_revision: int,
    boxes: list[dict],
    status: str = "draft",
    reviewer: str = "",
    notes: str = "",
    timer_revision: int | None = None,
    timer_token: str | None = None,
) -> dict:
    output, trial, frame, taxonomy, proposals = _context(store, output_id)
    if type(expected_revision) is not int or expected_revision < 0:
        raise ValueError("A saved correction revision is required")
    if status not in {"draft", "reviewed"}:
        raise ValueError("Correction status must be draft or reviewed")
    if not isinstance(reviewer, str) or len(reviewer.strip()) > 120:
        raise ValueError("Reviewer must contain at most 120 characters")
    if not isinstance(notes, str) or len(notes) > 4000:
        raise ValueError("Review notes must contain at most 4000 characters")
    reviewer = reviewer.strip()
    if status == "reviewed" and not reviewer:
        raise ValueError("Completing a review requires a reviewer name")
    if not isinstance(boxes, list) or len(boxes) > MAX_BOXES:
        raise ValueError(f"Corrections support at most {MAX_BOXES} boxes")
    classes = {item["id"] for item in taxonomy["classes"]}
    sources = {item["id"]: item for item in proposals}
    decisions = dict.fromkeys(sources, "rejected")
    normalized, identifiers, references = [], set(), set()
    for box in boxes:
        if not isinstance(box, dict) or set(box) - {"id", "label", "box", "proposal_id"}:
            raise ValueError("Each correction box needs its ID, class and coordinates")
        identifier, label, source_id = box.get("id"), box.get("label"), box.get("proposal_id")
        if (
            not isinstance(identifier, str)
            or not 1 <= len(identifier) <= 128
            or identifier in identifiers
        ):
            raise ValueError("Correction box IDs must be distinct and bounded")
        if not isinstance(label, str) or label not in classes:
            raise ValueError("The correction label must belong to the frozen class definitions")
        coordinates = _coordinates(box.get("box"), frame)
        if source_id is not None:
            if (
                not isinstance(source_id, str)
                or source_id not in sources
                or source_id in references
            ):
                raise ValueError("A candidate proposal can belong to only one corrected box")
            references.add(source_id)
            source = sources[source_id]
            decisions[source_id] = (
                "accepted"
                if label == source["label"] and coordinates == source["box"]
                else "corrected"
            )
        identifiers.add(identifier)
        normalized.append(
            {"id": identifier, "label": label, "box": coordinates, "proposal_id": source_id}
        )
    clock = _clock()
    with store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        latest = _latest(conn, output_id)
        if (latest["revision"] if latest else 0) != expected_revision:
            raise CorrectionConflict("Corrections changed; reload before saving")
        timer = _timer(conn, output_id)
        if timer and timer["state"] == "running":
            if timer_revision != timer["revision"] or timer_token != timer["metadata"].get(
                "owner_token"
            ):
                raise CorrectionConflict("Pause or refresh the active review timer before saving")
            _credit(timer, clock)
            timer["state"] = "paused"
            timer["revision"] += 1
            timer["updated_at"] = clock[0].isoformat()
            _write_timer(conn, timer)
        public_timing = _public_timer(timer, clock)
        timing = {
            key: public_timing.get(key)
            for key in (
                "protocol",
                "elapsed_ms",
                "recorded_segments",
                "fully_timed",
                "interruption_reason",
            )
        }
        timing["changes"] = {
            "added": sum(box["proposal_id"] is None for box in normalized),
            **{
                state: sum(value == state for value in decisions.values())
                for state in ("accepted", "corrected", "rejected")
            },
        }
        if timer and any(segment["reviewer"] != reviewer for segment in timer["segments"]):
            timing["fully_timed"] = False
            timing["interruption_reason"] = "Recorded intervals include a different reviewer"
        if latest and timer and timer["elapsed_ms"] <= (latest["timing"].get("elapsed_ms") or 0):
            timing["fully_timed"] = False
            timing["interruption_reason"] = "No new review interval was recorded for this revision"
        encoded = _encode(
            {
                "id": new_id(),
                "output_id": output_id,
                "revision": expected_revision + 1,
                "status": status,
                "boxes": normalized,
                "decisions": decisions,
                "reviewer": reviewer,
                "notes": notes,
                "timing": timing,
                "created_at": clock[0].isoformat(),
            }
        )
        conn.execute(
            f"INSERT INTO benchmark_corrections ({','.join(encoded)}) "
            f"VALUES ({','.join('?' for _ in encoded)})",
            tuple(encoded.values()),
        )
    return correction_document(store, output_id)


def correction_summaries(store: Store, trial_id: str) -> dict:
    """Report only saved completed reviews, keeping missing/incomplete time explicit."""
    trial = store.get("benchmark_trials", trial_id)
    if trial is None:
        raise KeyError(trial_id)
    expected = len(trial["config"]["frame_ids"])
    with store.connect() as conn:
        rows = [
            _decode(row)
            for row in conn.execute(
                "SELECT c.* FROM benchmark_corrections c "
                "JOIN benchmark_outputs o ON o.id=c.output_id "
                "WHERE o.trial_id=? AND c.revision=(SELECT MAX(x.revision) "
                "FROM benchmark_corrections x "
                "WHERE x.output_id=c.output_id)",
                (trial_id,),
            )
        ]
        outputs = conn.execute(
            "SELECT COUNT(*) FROM benchmark_outputs WHERE trial_id=?", (trial_id,)
        ).fetchone()[0]
    reviewed = [row for row in rows if row["status"] == "reviewed"]
    timed = [row for row in reviewed if row["timing"].get("elapsed_ms") is not None]
    return {
        "reviewed_count": len(reviewed),
        "output_count": outputs,
        "planned_count": expected,
        "timed_count": len(timed),
        "fully_timed_count": sum(row["timing"].get("fully_timed") is True for row in timed),
        "recorded_review_ms": sum(row["timing"]["elapsed_ms"] for row in timed) if timed else None,
        "complete": expected > 0 and len(reviewed) == outputs == expected,
        "frames": [
            {key: row[key] for key in ("output_id", "revision", "status", "reviewer", "timing")}
            for row in rows
        ],
        "note": (
            "Recorded review intervals are measured separately from model latency. "
            "Missing time is not zero; recorded intervals do not prove uninterrupted activity."
        ),
    }
