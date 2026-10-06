"""Explicit hosted proposals, with durable requests shared by later local reviews."""

import hashlib
import json
import math
import time
from copy import deepcopy
from decimal import Decimal

from iris import dinox_provider as provider
from iris.annotations import MAX_DETECTOR_SUGGESTIONS
from iris.inference import _load_verified_frame
from iris.preannotation import _digest, _snapshot
from iris.store import _decode, new_id, now
from iris.taxonomies import get_taxonomy

PROTOCOL = "iris-dinox-batch-v1"
ACTIVE = {"queued", "running"}
MAX_FRAMES = 25


def _request_snapshot(snapshot):
    return {"frame": deepcopy(snapshot["frame"]), "taxonomy_id": snapshot["taxonomy_id"]}


def _cached(conn, key):
    return _decode(
        conn.execute(
            "SELECT * FROM dinox_requests WHERE cache_key=? "
            "ORDER BY CASE WHEN state='succeeded' THEN 0 ELSE 1 END, "
            "created_at DESC,id DESC LIMIT 1",
            (key,),
        ).fetchone()
    )


def _action(request):
    if request is None or request["state"] in {"not_started", "failed"}:
        return "submit"
    if request["state"] in {"succeeded", "response_received"}:
        return "reuse"
    if request["state"] == "submitted" and request["task_id"]:
        return "poll"
    return "blocked"


def _prepare(store, session_id, *, frame_ids, threshold=0.25, class_prompts=None):
    if (
        not isinstance(frame_ids, list)
        or not 1 <= len(frame_ids) <= MAX_FRAMES
        or any(not isinstance(x, str) or not x for x in frame_ids)
        or len(set(frame_ids)) != len(frame_ids)
    ):
        raise ValueError("Choose between 1 and 25 distinct images")
    session = store.get("sessions", session_id)
    if session is None:
        raise KeyError(session_id)
    status = provider.provider_status()
    prepared, rows, frozen = [], [], None
    with store.connect() as conn:
        conn.execute("BEGIN")
        history = [
            tuple(row)
            for row in conn.execute(
                "SELECT b.id,j.status FROM dinox_batches b JOIN jobs j ON j.id=b.job_id "
                "WHERE b.session_id=? ORDER BY b.created_at,b.id",
                (session_id,),
            )
        ]
        active_frames = set()
        for entry in conn.execute(
            "SELECT b.frame_ids FROM dinox_batches b JOIN jobs j ON j.id=b.job_id "
            "WHERE b.session_id=? AND j.status IN ('queued','running')",
            (session_id,),
        ):
            active_frames.update(json.loads(entry[0]))
        for identifier in frame_ids:
            frame, _, snapshot = _snapshot(conn, identifier)
            if frame["session_id"] != session_id:
                raise ValueError("Every image must belong to this session")
            taxonomy = get_taxonomy(store, snapshot["taxonomy_id"], session["project_id"])
            config = provider.freeze_config(taxonomy, class_prompts, threshold)
            if frozen is None:
                frozen = config
            key = _digest({"snapshot": _request_snapshot(snapshot), "config": config})
            saved = _cached(conn, key)
            action = _action(saved)
            row = {
                "frame_id": identifier,
                "source_filename": store.get("assets", frame["asset_id"])["filename"],
                "eligible": True,
                "reason": None,
                "action": action,
                "request_id": saved["id"] if saved else None,
                "proposal_count": len((saved or {}).get("result", {}).get("proposals", []))
                if (saved or {}).get("result")
                else 0,
            }
            try:
                if config != frozen:
                    raise ValueError("Choose images with the same saved class definitions")
                if identifier in active_frames:
                    raise ValueError("A DINO-X batch is already active for this image")
                if action == "blocked":
                    raise ValueError(
                        "The previous request has an unknown outcome; it will not be resent"
                    )
                if action != "reuse" and status.get("status") != "ready":
                    raise ValueError("Configure a DINO-X key before sending or polling images")
                if snapshot["annotation_frame_sha256"] != frame["sha256"]:
                    raise ValueError("Saved annotations refer to different image pixels")
                with _load_verified_frame(store, frame, frame["sha256"]) as image:
                    provider.encode_image(image)
            except (ValueError, OSError) as exc:
                row.update(eligible=False, reason=str(exc))
            rows.append(row)
            prepared.append(
                {
                    "snapshot": snapshot,
                    "row": row,
                    "cache_key": key,
                    "request_state": saved["state"] if saved else None,
                }
            )
    count = sum(row["eligible"] and row["action"] == "submit" for row in rows)
    settings = {"frame_ids": frame_ids, "threshold": threshold, "class_prompts": class_prompts}
    estimate = provider.estimate(count)
    return {
        "session_id": session_id,
        "config": settings,
        "provider_config": frozen,
        "provider": status,
        "frames": rows,
        "request_count": count,
        "reuse_count": sum(r["eligible"] and r["action"] == "reuse" for r in rows),
        "poll_count": sum(r["eligible"] and r["action"] == "poll" for r in rows),
        "eligible_count": sum(r["eligible"] for r in rows),
        "excluded_count": sum(not r["eligible"] for r in rows),
        "estimate": estimate,
        "warnings": [
            "Only image pixels and the saved class phrases are sent to DeepDataSpace.",
            "Prices are estimates, not account balances. "
            "An interrupted request may still be billed.",
            "Review proposals and the whole image before validating annotations; "
            "empty results are not validated negatives.",
        ],
        "fingerprint": _digest(
            {
                "session_id": session_id,
                "settings": settings,
                "provider_config": frozen,
                "frames": prepared,
                "history": history,
                "estimate": estimate,
            }
        ),
    }, prepared


def preview_batch(store, session_id, **settings):
    return _prepare(store, session_id, **settings)[0]


def _receipt(conn, session_id, fingerprint):
    return _decode(
        conn.execute(
            "SELECT * FROM dinox_batches WHERE session_id=? "
            "AND json_extract(config,'$.fingerprint')=? ORDER BY created_at,id LIMIT 1",
            (session_id, fingerprint),
        ).fetchone()
    )


def create_batch(
    store,
    jobs,
    session_id,
    *,
    name,
    expected_fingerprint,
    approve_external=False,
    max_cost_cny=0,
    **settings,
):
    if not isinstance(name, str) or not 1 <= len(name.strip()) <= 160:
        raise ValueError("Use a batch name of 1–160 characters")
    if not isinstance(expected_fingerprint, str) or len(expected_fingerprint) != 64:
        raise ValueError("Preview this batch before starting it")
    if (
        type(max_cost_cny) not in {int, float}
        or not math.isfinite(max_cost_cny)
        or not 0 <= max_cost_cny <= 1000
    ):
        raise ValueError("Choose a finite CNY budget between 0 and 1000")
    with jobs.guard, store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        previous = _receipt(conn, session_id, expected_fingerprint)
        if previous:
            identifier = previous["id"]
        else:
            preview, prepared = _prepare(store, session_id, **settings)
            if preview["fingerprint"] != expected_fingerprint:
                raise ValueError("The images, annotations or saved requests changed; preview again")
            eligible = [f for f in prepared if f["row"]["eligible"]]
            if not eligible:
                raise ValueError("No images are eligible for this batch")
            if preview["request_count"] or preview["poll_count"]:
                if approve_external is not True:
                    raise ValueError("Confirm processing by DeepDataSpace before starting")
            if Decimal(str(max_cost_cny)) < Decimal(str(preview["estimate"]["total"])):
                raise ValueError("The CNY budget is below this batch's estimated maximum")
            identifier, job_id, created = new_id(), new_id(), now()
            config = {
                "protocol": PROTOCOL,
                "fingerprint": expected_fingerprint,
                "settings": preview["config"],
                "provider_config": preview["provider_config"],
                "frames": eligible,
                "excluded": [r for r in preview["frames"] if not r["eligible"]],
                "estimate": preview["estimate"],
                "approve_external": approve_external,
                "max_cost_cny": float(max_cost_cny),
            }
            conn.execute(
                "INSERT INTO jobs (id,kind,status,params,message,created_at) VALUES (?,?,?,?,?,?)",
                (
                    job_id,
                    "dinox",
                    "queued",
                    json.dumps({"batch_id": identifier}),
                    "Waiting for DINO-X proposals",
                    created,
                ),
            )
            rows = []
            for item in eligible:
                snapshot, row = item["snapshot"], item["row"]
                saved = _cached(conn, item["cache_key"])
                if saved is None or saved["state"] == "failed":
                    request_id = new_id()
                    conn.execute(
                        "INSERT INTO dinox_requests "
                        "(id,frame_id,job_id,cache_key,config,snapshot,state,metadata,"
                        "created_at,updated_at) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (
                            request_id,
                            row["frame_id"],
                            job_id,
                            item["cache_key"],
                            json.dumps(config["provider_config"], allow_nan=False),
                            json.dumps(_request_snapshot(snapshot), allow_nan=False),
                            "not_started",
                            json.dumps({"previous_request_id": saved["id"] if saved else None}),
                            created,
                            created,
                        ),
                    )
                else:
                    request_id = saved["id"]
                rows.append(
                    {
                        "frame_id": row["frame_id"],
                        "source_filename": row["source_filename"],
                        "request_id": request_id,
                        "state": "queued",
                        "reason": None,
                        "proposal_count": 0,
                        "reused": row["action"] in {"reuse", "poll"},
                    }
                )
            conn.execute(
                "INSERT INTO dinox_batches "
                "(id,session_id,name,frame_ids,config,metadata,job_id,created_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (
                    identifier,
                    session_id,
                    name.strip(),
                    json.dumps([r["frame_id"] for r in rows]),
                    json.dumps(config, allow_nan=False),
                    json.dumps(
                        {
                            "frames": rows,
                            "attempted_count": 0,
                            "estimated_cny": preview["estimate"]["total"],
                        }
                    ),
                    job_id,
                    created,
                ),
            )
    return batch_detail(store, identifier)


def _write_request(conn, identifier, **changes):
    changes["updated_at"] = now()
    encoded = {
        k: json.dumps(v, allow_nan=False)
        if k in {"metadata", "raw_response", "result"} and v is not None
        else v
        for k, v in changes.items()
    }
    conn.execute(
        "UPDATE dinox_requests SET " + ",".join(f"{k}=?" for k in encoded) + " WHERE id=?",
        (*encoded.values(), identifier),
    )


def _frame_state(conn, batch_id, frame_id, **changes):
    batch = _decode(conn.execute("SELECT * FROM dinox_batches WHERE id=?", (batch_id,)).fetchone())
    metadata = batch["metadata"]
    for item in metadata["frames"]:
        if item["frame_id"] == frame_id:
            item.update(changes)
    conn.execute(
        "UPDATE dinox_batches SET metadata=? WHERE id=?",
        (json.dumps(metadata, allow_nan=False), batch_id),
    )


def _running(conn, batch):
    job = _decode(conn.execute("SELECT * FROM jobs WHERE id=?", (batch["job_id"],)).fetchone())
    return (
        job
        and job["kind"] == "dinox"
        and job["params"].get("batch_id") == batch["id"]
        and job["status"] == "running"
        and not job["cancel_requested"]
    )


def _take_request(store, batch, request_id):
    with store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        if not _running(conn, batch):
            raise InterruptedError("Batch stopped before provider processing")
        request = _decode(
            conn.execute("SELECT * FROM dinox_requests WHERE id=?", (request_id,)).fetchone()
        )
        if request is None:
            raise ValueError("Saved DINO-X request is missing")
        if request["state"] == "not_started":
            live = _decode(
                conn.execute("SELECT * FROM dinox_batches WHERE id=?", (batch["id"],)).fetchone()
            )
            metadata = live["metadata"]
            count = metadata["attempted_count"] + 1
            amount = provider.estimate(count)["total"]
            if (
                amount > batch["config"]["max_cost_cny"]
                or count > batch["config"]["estimate"]["request_count"]
            ):
                raise ValueError("The frozen DINO-X request budget is exhausted")
            metadata["attempted_count"] = count
            conn.execute(
                "UPDATE dinox_batches SET metadata=? WHERE id=?",
                (json.dumps(metadata), batch["id"]),
            )
            _write_request(
                conn,
                request_id,
                state="dispatching",
                job_id=batch["job_id"],
                metadata={**request["metadata"], "attempted_at": now()},
            )
        elif request["state"] == "submitted":
            old_job = conn.execute(
                "SELECT status FROM jobs WHERE id=?", (request["job_id"],)
            ).fetchone()
            if request["job_id"] != batch["job_id"] and old_job and old_job["status"] in ACTIVE:
                raise ValueError("Another worker owns the saved remote task")
            _write_request(conn, request_id, job_id=batch["job_id"])
        elif request["state"] not in {"succeeded", "response_received"}:
            raise ValueError("This provider request cannot be replayed")
    return request


def _publish(store, batch, item, request, result, cancelled):
    if len(result["proposals"]) > MAX_DETECTOR_SUGGESTIONS:
        raise ValueError(
            "More than 100 proposals; inspect the saved response before another request"
        )
    frame_id, snapshot = item["row"]["frame_id"], item["snapshot"]
    with store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        if cancelled() or not _running(conn, batch):
            raise InterruptedError("Batch stopped; provider response retained")
        frame, _, current = _snapshot(conn, frame_id)
        if current != snapshot:
            raise ValueError(
                "Image pixels, classes or human annotations changed; response retained"
            )
        with _load_verified_frame(store, frame, frame["sha256"]):
            pass
        for proposal in result["proposals"]:
            identity = hashlib.sha256(
                f"dinox:{request['id']}:{proposal['id']}".encode()
            ).hexdigest()
            metadata = {
                "provider": "dinox",
                "dinox_request_id": request["id"],
                "batch_id": batch["id"],
                "model_id": "DINO-X-1.0",
                "target_taxonomy": snapshot["taxonomy_id"],
                "frame_sha256": frame["sha256"],
                "base_revision": snapshot["revision"],
                "base_revision_id": snapshot["revision_id"],
                "score": proposal.get("score"),
                "threshold": batch["config"]["settings"]["threshold"],
                "source": proposal.get("source", {}),
                "geometry": proposal.get("geometry", {}),
            }
            # Stable IDs keep earlier human decisions intact when a cached response is reused.
            conn.execute(
                "INSERT OR IGNORE INTO annotation_suggestions "
                "(id,frame_id,job_id,kind,label,box,metadata,created_at) VALUES (?,?,?,?,?,?,?,?)",
                (
                    identity,
                    frame_id,
                    batch["job_id"],
                    "detector",
                    proposal["label"],
                    json.dumps(proposal["box"]),
                    json.dumps(metadata, allow_nan=False),
                    now(),
                ),
            )
        _frame_state(
            conn,
            batch["id"],
            frame_id,
            state="pending_review" if result["proposals"] else "no_proposals",
            proposal_count=len(result["proposals"]),
            reason=None,
        )


def run_batch(store, batch_id, progress, cancelled):
    batch = store.get("dinox_batches", batch_id)
    if batch is None or batch["config"].get("protocol") != PROTOCOL:
        raise ValueError("Invalid saved DINO-X batch")
    config = provider.validate_frozen_config(batch["config"]["provider_config"])
    ready, issues = 0, 0
    for item, row in zip(batch["config"]["frames"], batch["metadata"]["frames"], strict=True):
        if cancelled():
            break
        frame_id = item["row"]["frame_id"]
        request_id = row["request_id"]
        submitted = False
        try:
            request = store.get("dinox_requests", request_id)
            expected_snapshot = _request_snapshot(item["snapshot"])
            if (
                row["frame_id"] != frame_id
                or request is None
                or request["frame_id"] != frame_id
                or request["snapshot"] != expected_snapshot
                or request["config"] != config
                or request["cache_key"]
                != _digest({"snapshot": expected_snapshot, "config": config})
            ):
                raise ValueError("Frozen DINO-X request provenance changed")
            with store.connect() as conn:
                frame, _, current = _snapshot(conn, frame_id)
            if current != item["snapshot"]:
                raise ValueError("Image or human annotation revision changed before processing")
            with _load_verified_frame(store, frame, frame["sha256"]) as image:
                png = provider.encode_image(image)
            if request["state"] in {"not_started", "submitted"}:
                if provider.provider_status().get("status") != "ready":
                    raise ValueError("DINO-X credentials are unavailable; request left unsent")
            request = _take_request(store, batch, request_id)
            started = time.monotonic()
            if request["state"] == "not_started":
                submitted = True
                receipt = provider.submit(config, png, idempotency_key=request_id)
                with store.connect() as conn:
                    conn.execute("BEGIN IMMEDIATE")
                    live = _decode(
                        conn.execute(
                            "SELECT * FROM dinox_requests WHERE id=?", (request_id,)
                        ).fetchone()
                    )
                    _write_request(
                        conn,
                        request_id,
                        state="submitted",
                        task_id=receipt["task_id"],
                        metadata={**live["metadata"], "submission": receipt},
                    )
                request = store.get("dinox_requests", request_id)
            if request["state"] == "submitted":
                deadline = time.monotonic() + 300
                while True:
                    if cancelled():
                        raise InterruptedError("Remote task retained; later batches can poll it")
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Remote task still pending; later batches can poll it")
                    receipt = provider.poll(request["task_id"])
                    with store.connect() as conn:
                        conn.execute("BEGIN IMMEDIATE")
                        live = _decode(
                            conn.execute(
                                "SELECT * FROM dinox_requests WHERE id=?", (request_id,)
                            ).fetchone()
                        )
                        meta = {
                            **live["metadata"],
                            "last_poll": receipt,
                            "http_wall_ms": (time.monotonic() - started) * 1000,
                        }
                        if receipt["status"] == "succeeded":
                            _write_request(
                                conn,
                                request_id,
                                state="response_received",
                                raw_response=receipt["result"],
                                metadata=meta,
                            )
                        elif receipt["status"] == "failed":
                            _write_request(
                                conn,
                                request_id,
                                state="failed",
                                error="DINO-X task failed",
                                metadata=meta,
                            )
                        else:
                            _write_request(conn, request_id, metadata=meta)
                    if receipt["status"] == "succeeded":
                        break
                    if receipt["status"] == "failed":
                        raise ValueError("DINO-X reported a failed remote task")
                    for _ in range(20):
                        if cancelled():
                            break
                        time.sleep(0.1)
                request = store.get("dinox_requests", request_id)
            result = provider.normalize(
                request["raw_response"], config, frame["width"], frame["height"]
            )
            if request["result"] is not None and request["result"] != result:
                raise ValueError("Saved DINO-X proposals differ from their native response")
            if request["result"] is None:
                with store.connect() as conn:
                    conn.execute("BEGIN IMMEDIATE")
                    _write_request(conn, request_id, state="succeeded", result=result, error=None)
            _publish(store, batch, item, request, result, cancelled)
            ready += 1
            progress(
                (ready + issues) / len(batch["frame_ids"]),
                f"DINO-X: {ready} images ready for review",
            )
        except (Exception, KeyboardInterrupt) as exc:
            if isinstance(exc, provider.DinoXError):
                message = str(exc)
            elif isinstance(exc, (ValueError, InterruptedError, TimeoutError)):
                message = str(exc)
            else:
                message = (
                    f"DINO-X processing stopped ({type(exc).__name__}); "
                    "saved responses are retained"
                )
            with store.connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                request = _decode(
                    conn.execute(
                        "SELECT * FROM dinox_requests WHERE id=?", (request_id,)
                    ).fetchone()
                )
                if submitted and request and request["state"] == "dispatching":
                    known = isinstance(exc, provider.DinoXError) and not exc.outcome_unknown
                    state = (
                        "failed"
                        if known and exc.dispatched
                        else "not_started"
                        if known
                        else "outcome_unknown"
                    )
                    meta = {
                        **request["metadata"],
                        "submission_error": exc.raw_response
                        if isinstance(exc, provider.DinoXError)
                        else None,
                    }
                    _write_request(conn, request_id, state=state, error=message, metadata=meta)
                _frame_state(
                    conn,
                    batch_id,
                    frame_id,
                    state="cancelled" if cancelled() else "failed",
                    reason=message,
                )
            issues += 1
            # Stop this batch; later explicit previews reuse/poll saved work without a second POST.
            break
    return {"batch_id": batch_id, "preannotation": {"frames_ready": ready, "frames_issues": issues}}


def reconcile_requests(store):
    with store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "UPDATE dinox_requests SET state='outcome_unknown',error=?,updated_at=? "
            "WHERE state='dispatching' AND job_id IN "
            "(SELECT id FROM jobs WHERE status NOT IN ('queued','running'))",
            ("Worker stopped during submission; this request will not be resent", now()),
        )


def batch_detail(store, identifier):
    batch = store.get("dinox_batches", identifier)
    if batch is None:
        raise KeyError(identifier)
    job = store.get("jobs", batch["job_id"])
    rows, requests = [], []
    for saved in batch["metadata"]["frames"]:
        row = dict(saved)
        request = store.get("dinox_requests", row["request_id"])
        if row["state"] == "queued" and job["status"] not in ACTIVE:
            row.update(
                state="not_started", reason="Not processed in this attempt; preview to continue"
            )
        rows.append(row)
        requests.append(request)
    return {
        **batch,
        "job": job,
        "frames": rows,
        "requests": requests,
        "counts": {
            "total": len(rows),
            "ready": sum(r["state"] in {"pending_review", "no_proposals"} for r in rows),
            "issues": sum(r["state"] in {"failed", "cancelled"} for r in rows),
            "proposals": sum(r["proposal_count"] for r in rows),
        },
        "cost": {
            "currency": "CNY",
            "estimated_cny": provider.estimate(batch["metadata"]["attempted_count"])["total"],
            "new_requests": batch["metadata"]["attempted_count"],
            "unknown_outcome_count": sum(
                r["state"] in {"dispatching", "outcome_unknown"} for r in requests
            ),
        },
    }


def list_batches(store, session_id):
    if store.get("sessions", session_id) is None:
        raise KeyError(session_id)
    return [
        batch_detail(store, row["id"]) for row in store.list("dinox_batches", session_id=session_id)
    ]
