"""Explicit, bounded local review batches using the existing per-frame worker."""

import hashlib
import json
import math
from datetime import datetime, timedelta

from iris.annotations import TAXONOMY, AnnotationConflict
from iris.assistance import MAX_CANDIDATES, _candidates
from iris.assistance_provider import ProviderConfig, _config, provider_status
from iris.inference import _load_verified_frame, comparison_lanes
from iris.jobs import ACTIVE
from iris.store import Store, _decode, new_id, now

MAX_BATCH_FRAMES = 25
STATUSES = ("queued", "running", "succeeded", "failed", "cancelled", "interrupted")
FRAME_FIELDS = ("id", "session_id", "asset_id", "sha256", "path", "width", "height", "selected")


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _configuration(
    store,
    session_id,
    frame_ids,
    source,
    comparison_id,
    detector_model_id,
    detector_variant,
    model,
    threshold,
    instructions,
):
    if store.get("sessions", session_id) is None:
        raise KeyError(session_id)
    if (
        not isinstance(frame_ids, list)
        or not 1 <= len(frame_ids) <= MAX_BATCH_FRAMES
        or any(not isinstance(frame_id, str) or not frame_id for frame_id in frame_ids)
        or len(set(frame_ids)) != len(frame_ids)
    ):
        raise ValueError("Choose between 1 and 25 distinct selected frames")
    if source not in {"annotations", "comparison"}:
        raise ValueError("Batch source must be annotations or comparison")
    if (
        type(threshold) not in (int, float)
        or not math.isfinite(threshold)
        or not 0 <= threshold <= 1
    ):
        raise ValueError("Confidence threshold must be between 0 and 1")
    if not isinstance(instructions, str) or len(instructions) > 2000:
        raise ValueError("Additional instructions are limited to 2000 characters")
    if not isinstance(model, str) or not model:
        raise ValueError("Choose an installed local vision model")
    local = _config({"endpoint": ProviderConfig.from_env().endpoint, "model": model})
    if source == "annotations":
        if (
            comparison_id is not None
            or detector_model_id is not None
            or detector_variant is not None
        ):
            raise ValueError("Comparison fields are only valid for a comparison source")
    else:
        if not isinstance(comparison_id, str) or not isinstance(detector_model_id, str):
            raise ValueError("Choose a completed comparison and one of its detector models")
        comparison = store.get("comparisons", comparison_id)
        if comparison is None or comparison["session_id"] != session_id:
            raise ValueError("The comparison must belong to this session")
        job = store.get("jobs", comparison["job_id"])
        if job is None or job["status"] != "succeeded":
            raise ValueError("The source comparison must have completed successfully")
        if (
            not isinstance(comparison["model_ids"], list)
            or detector_model_id not in comparison["model_ids"]
        ):
            raise ValueError("The detector model must belong to the source comparison")
        if (
            not isinstance(comparison["config"], dict)
            or comparison["config"].get("taxonomy") != "coco-2017-v1"
        ):
            raise ValueError("The source comparison must use the supported COCO taxonomy")
        variants = [
            lane["variant"]
            for lane in comparison_lanes(comparison)
            if lane["model_id"] == detector_model_id
        ]
        if detector_variant is None:
            if len(variants) != 1:
                raise ValueError("Choose an explicit detector inference mode: full or tiled")
            detector_variant = variants[0]
        elif not isinstance(detector_variant, str) or detector_variant not in variants:
            raise ValueError("The detector inference mode must belong to the source comparison")
    frames = []
    for frame_id in frame_ids:
        frame = store.get("frames", frame_id)
        if frame is None or frame["session_id"] != session_id:
            raise ValueError("Every frame must belong to this session")
        frames.append(frame)
    return (
        frames,
        local,
        {
            "source": source,
            "comparison_id": comparison_id,
            "detector_model_id": detector_model_id,
            "detector_variant": detector_variant,
            "model": local.model,
            "threshold": float(threshold),
            "instructions": instructions.strip(),
        },
    )


def _frame_state(conn, frame_id):
    frame = _decode(conn.execute("SELECT * FROM frames WHERE id=?", (frame_id,)).fetchone())
    revision = _decode(
        conn.execute(
            "SELECT * FROM annotation_revisions WHERE frame_id=? ORDER BY revision DESC LIMIT 1",
            (frame_id,),
        ).fetchone()
    )
    active = [
        row[0]
        for row in conn.execute(
            "SELECT j.id FROM assistance_records a JOIN jobs j ON j.id=a.job_id "
            "WHERE a.frame_id=? AND j.status IN ('queued','running') ORDER BY j.id",
            (frame_id,),
        )
    ]
    return frame, revision, active


def _snapshot(frame, revision, active):
    return {
        "frame": {key: frame[key] for key in FRAME_FIELDS} if frame else None,
        "revision": revision,
        "active_job_ids": active,
    }


def _comparison_state(conn, comparison_id, detector_model_id, frame_id, detector_variant):
    if comparison_id is None:
        return None
    comparison = _decode(
        conn.execute(
            "SELECT * FROM comparisons WHERE id=?",
            (comparison_id,),
        ).fetchone()
    )
    job = (
        _decode(
            conn.execute(
                "SELECT * FROM jobs WHERE id=?",
                (comparison["job_id"],),
            ).fetchone()
        )
        if comparison
        else None
    )
    runs = [
        _decode(row)
        for row in conn.execute(
            "SELECT * FROM runs WHERE comparison_id=? AND model_id=? AND variant=? ORDER BY id",
            (comparison_id, detector_model_id, detector_variant),
        )
    ]
    predictions = [
        _decode(row)
        for row in conn.execute(
            "SELECT p.* FROM predictions p JOIN runs r ON r.id=p.run_id "
            "WHERE p.comparison_id=? AND p.model_id=? AND p.frame_id=? "
            "AND r.comparison_id=p.comparison_id AND r.model_id=p.model_id "
            "AND r.variant=? ORDER BY p.id",
            (comparison_id, detector_model_id, frame_id, detector_variant),
        )
    ]
    return {"comparison": comparison, "job": job, "runs": runs, "predictions": predictions}


def _prepare_batch(
    store,
    session_id,
    *,
    frame_ids,
    source,
    model,
    comparison_id=None,
    detector_model_id=None,
    detector_variant=None,
    threshold=0.5,
    instructions="",
):
    frames, local, config = _configuration(
        store,
        session_id,
        frame_ids,
        source,
        comparison_id,
        detector_model_id,
        detector_variant,
        model,
        threshold,
        instructions,
    )
    status = provider_status(local.as_dict())
    if status["status"] == "invalid_config":
        raise ValueError(status.get("reason") or "Invalid local annotation configuration")
    if status["status"] != "ready" or not status.get("model_digest"):
        raise RuntimeError(status.get("reason") or "The local annotation model is unavailable")
    provider = {
        key: status[key]
        for key in (
            "provider",
            "endpoint",
            "model",
            "model_digest",
            "status",
        )
    }
    prepared = []
    for requested in frames:
        with store.connect() as conn:
            conn.execute("BEGIN")
            frame, revision, active = _frame_state(conn, requested["id"])
            comparison_state = _comparison_state(
                conn,
                comparison_id,
                detector_model_id,
                requested["id"],
                config["detector_variant"],
            )
        if frame is None or frame["session_id"] != session_id:
            raise AnnotationConflict("A requested frame changed; preview the batch again")
        asset = store.get("assets", frame["asset_id"])
        row = {
            "frame_id": frame["id"],
            "eligible": False,
            "reason": None,
            "candidate_count": 0,
            "base_revision": revision["revision"] if revision else 0,
            "prediction_id": None,
            "frame_sha256": frame["sha256"],
            "source_filename": asset["filename"],
            "width": frame["width"],
            "height": frame["height"],
        }
        prediction = None
        if source == "comparison":
            predictions = comparison_state["predictions"]
            prediction = predictions[0] if len(predictions) == 1 else None
            row["prediction_id"] = prediction["id"] if prediction else None
        candidates = []
        try:
            if source == "comparison" and prediction is None:
                raise ValueError("No saved prediction from this comparison and detector")
            if revision and revision["frame_sha256"] != frame["sha256"]:
                raise ValueError("Saved annotations refer to a different frame revision")
            candidates = _candidates(
                store,
                frame,
                revision or {"revision": 0, "boxes": []},
                row["prediction_id"],
                threshold,
            )
            row["candidate_count"] = len(candidates)
            if not frame["selected"]:
                raise ValueError("This frame is no longer selected")
            if active:
                raise ValueError(
                    "An assistance request is already queued or running for this frame"
                )
            if not 1 <= len(candidates) <= MAX_CANDIDATES:
                raise ValueError("The source must contain 1–8 person/car boxes")
            with _load_verified_frame(store, frame, frame["sha256"]):
                pass
            row["eligible"] = True
        except (ValueError, OSError) as exc:
            row["reason"] = str(exc)
        review_config = {
            "provider": {key: provider[key] for key in ("provider", "endpoint", "model")},
            "model_digest": provider["model_digest"],
            "frame_sha256": frame["sha256"],
            "base_revision": row["base_revision"],
            "taxonomy_id": TAXONOMY["id"],
            "instructions": config["instructions"],
            "prediction_id": row["prediction_id"],
            "threshold": config["threshold"],
            "max_candidates": MAX_CANDIDATES,
        }
        prepared.append(
            {
                "row": row,
                "snapshot": _snapshot(frame, revision, active),
                "candidates": candidates,
                "config": review_config,
                "comparison_state": comparison_state,
            }
        )
    preview = {
        "session_id": session_id,
        "frame_ids": frame_ids,
        "config": config,
        "provider": provider,
        "fingerprint": _digest(
            {
                "session_id": session_id,
                "config": config,
                "provider": provider,
                "prepared": prepared,
            }
        ),
        "eligible_count": sum(item["row"]["eligible"] for item in prepared),
        "excluded_count": sum(not item["row"]["eligible"] for item in prepared),
        "candidate_count": sum(
            item["row"]["candidate_count"] for item in prepared if item["row"]["eligible"]
        ),
        "frames": [item["row"] for item in prepared],
    }
    return preview, prepared


def preview_batch(store: Store, session_id: str, **options) -> dict:
    """Read local metadata and verified pixels only; never run a reviewer."""
    return _prepare_batch(store, session_id, **options)[0]


def create_batch(
    store: Store,
    jobs,
    session_id: str,
    *,
    name: str,
    expected_fingerprint: str,
    **options,
) -> dict:
    if not isinstance(name, str) or not 1 <= len(name.strip()) <= 160:
        raise ValueError("Batch name must contain between 1 and 160 characters")
    if not isinstance(expected_fingerprint, str) or len(expected_fingerprint) != 64:
        raise ValueError("Preview this batch before queueing it")
    preview, prepared = _prepare_batch(store, session_id, **options)
    if preview["fingerprint"] != expected_fingerprint:
        raise AnnotationConflict("The batch preview changed; preview the batch again")
    eligible = [item for item in prepared if item["row"]["eligible"]]
    if not eligible:
        raise ValueError("No frames are eligible for local assistance")
    batch_id, created_at = new_id(), now()
    job_ids, frame_ids = [new_id() for _ in eligible], [r["row"]["frame_id"] for r in eligible]
    config = {
        **preview["config"],
        "provider": preview["provider"],
        "requested_frame_ids": preview["frame_ids"],
        "preview_fingerprint": expected_fingerprint,
        "excluded": [item["row"] for item in prepared if not item["row"]["eligible"]],
    }
    with jobs.guard, store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        for item in prepared:
            row = item["row"]
            frame, revision, active = _frame_state(conn, row["frame_id"])
            if _snapshot(frame, revision, active) != item["snapshot"]:
                raise AnnotationConflict("The batch inputs changed; preview the batch again")
            current_source = _comparison_state(
                conn,
                config["comparison_id"],
                config["detector_model_id"],
                row["frame_id"],
                config["detector_variant"],
            )
            if current_source != item["comparison_state"]:
                raise AnnotationConflict("A source prediction changed; preview the batch again")
            if row["eligible"]:
                try:
                    with _load_verified_frame(store, frame, row["frame_sha256"]):
                        pass
                except (ValueError, OSError) as exc:
                    raise AnnotationConflict(
                        "Frame bytes changed; preview the batch again"
                    ) from exc
        conn.execute(
            "INSERT INTO assistance_batches "
            "(id,session_id,name,frame_ids,job_ids,config,created_at) VALUES (?,?,?,?,?,?,?)",
            (
                batch_id,
                session_id,
                name.strip(),
                json.dumps(frame_ids),
                json.dumps(job_ids),
                json.dumps(config),
                created_at,
            ),
        )
        for index, (item, job_id) in enumerate(zip(eligible, job_ids, strict=True)):
            record_id, frame_id = new_id(), item["row"]["frame_id"]
            queued_at = (
                datetime.fromisoformat(created_at) + timedelta(microseconds=index)
            ).isoformat()
            conn.execute(
                "INSERT INTO jobs (id,kind,status,params,message,created_at) VALUES (?,?,?,?,?,?)",
                (
                    job_id,
                    "assist",
                    "queued",
                    json.dumps(
                        {
                            "frame_id": frame_id,
                            "assistance_id": record_id,
                            "batch_id": batch_id,
                        }
                    ),
                    "Waiting for local batch review",
                    queued_at,
                ),
            )
            conn.execute(
                "INSERT INTO assistance_records "
                "(id,frame_id,job_id,config,candidates,created_at) VALUES (?,?,?,?,?,?)",
                (
                    record_id,
                    frame_id,
                    job_id,
                    json.dumps(item["config"]),
                    json.dumps(item["candidates"]),
                    created_at,
                ),
            )
    return batch_detail(store, batch_id)


def batch_detail(store: Store, batch_id: str) -> dict:
    with store.connect() as conn:
        conn.execute("BEGIN")
        batch = _decode(
            conn.execute(
                "SELECT * FROM assistance_batches WHERE id=?",
                (batch_id,),
            ).fetchone()
        )
        if batch is None:
            raise KeyError(batch_id)
        children = []
        for frame_id, job_id in zip(batch["frame_ids"], batch["job_ids"], strict=True):
            job = _decode(conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone())
            record = _decode(
                conn.execute(
                    "SELECT * FROM assistance_records WHERE job_id=?",
                    (job_id,),
                ).fetchone()
            )
            filename = conn.execute(
                "SELECT a.filename FROM assets a JOIN frames f ON f.asset_id=a.id WHERE f.id=?",
                (frame_id,),
            ).fetchone()[0]
            count = conn.execute(
                "SELECT COUNT(*) FROM annotation_suggestions WHERE job_id=?",
                (job_id,),
            ).fetchone()[0]
            progress = 1.0 if job["status"] == "succeeded" else job["progress"]
            children.append(
                {
                    "frame_id": frame_id,
                    "job_id": job_id,
                    "assistance_id": record["id"],
                    "status": job["status"],
                    "progress": progress,
                    "message": job["message"],
                    "error": job["error"] or record["error"],
                    "cancel_requested": job["cancel_requested"],
                    "suggestions_created": count,
                    "base_revision": record["config"]["base_revision"],
                    "candidate_count": len(record["candidates"]),
                    "prediction_id": record["config"]["prediction_id"],
                    "source_filename": filename,
                }
            )
    counts = {status: sum(row["status"] == status for row in children) for status in STATUSES}
    counts["total"] = len(children)
    active_children = [row for row in children if row["status"] in ACTIVE]
    if counts["running"]:
        status = "running"
    elif counts["queued"]:
        status = "queued"
    else:
        states = {row["status"] for row in children}
        status = states.pop() if len(states) == 1 else "partial"
    return {
        **batch,
        "status": status,
        "counts": counts,
        "frames": children,
        "progress": sum(row["progress"] for row in children) / len(children),
        "finished_count": sum(row["status"] not in ACTIVE for row in children),
        "suggestions_created": sum(row["suggestions_created"] for row in children),
        "cancel_requested": (
            all(row["cancel_requested"] for row in active_children)
            if active_children
            else any(row["cancel_requested"] for row in children)
        ),
    }


def list_batches(store: Store, session_id: str) -> list[dict]:
    if store.get("sessions", session_id) is None:
        raise KeyError(session_id)
    return [
        batch_detail(store, batch["id"])
        for batch in reversed(store.list("assistance_batches", session_id=session_id))
    ]


def cancel_batch(store: Store, jobs, batch_id: str) -> dict:
    with jobs.guard, store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        batch = _decode(
            conn.execute(
                "SELECT * FROM assistance_batches WHERE id=?",
                (batch_id,),
            ).fetchone()
        )
        if batch is None:
            raise KeyError(batch_id)
        finished_at = now()
        for job_id in batch["job_ids"]:
            conn.execute(
                "UPDATE jobs SET cancel_requested=1, "
                "status=CASE WHEN status='queued' THEN 'cancelled' ELSE status END, "
                "finished_at=CASE WHEN status='queued' THEN ? ELSE finished_at END, "
                "message='Cancellation requested; saved proposals are preserved' "
                "WHERE id=? AND status IN ('queued','running')",
                (finished_at, job_id),
            )
    return batch_detail(store, batch_id)
