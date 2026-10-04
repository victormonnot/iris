"""Explicit new local review batches for unfinished images; never replay successes."""

from iris.annotations import AnnotationConflict
from iris.assistance_batches import _digest, _prepare_batch, batch_detail, create_batch
from iris.jobs import ACTIVE
from iris.store import Store, _decode


def retry_state(conn, batch_id: str) -> dict:
    batch = _decode(
        conn.execute("SELECT * FROM assistance_batches WHERE id=?", (batch_id,)).fetchone()
    )
    if batch is None:
        raise KeyError(batch_id)
    if batch["config"].get("provider", {}).get("provider") != "ollama":
        raise ValueError("Only local review batches support preparing unfinished images")
    children = []
    for frame_id, job_id in zip(batch["frame_ids"], batch["job_ids"], strict=True):
        job = conn.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
        if job is None:
            raise ValueError("A batch job is missing")
        count = conn.execute(
            "SELECT COUNT(*) FROM annotation_suggestions WHERE job_id=?", (job_id,)
        ).fetchone()[0]
        children.append(
            {"frame_id": frame_id, "job_id": job_id, "status": job["status"], "proposals": count}
        )
    if any(child["status"] in ACTIVE for child in children):
        raise RuntimeError("Wait for this batch to stop before preparing another batch")
    return {
        "id": batch_id,
        "session_id": batch["session_id"],
        "config": batch["config"],
        "children": children,
        "frame_ids": [
            child["frame_id"]
            for child in children
            if child["status"] in {"failed", "cancelled", "interrupted"} and child["proposals"] == 0
        ],
    }


def find_retry(conn, batch_id: str) -> dict | None:
    return _decode(
        conn.execute(
            "SELECT * FROM assistance_batches WHERE json_extract(config,'$.retry_of')=? "
            "ORDER BY created_at,id LIMIT 1",
            (batch_id,),
        ).fetchone()
    )


def _prepare_retry(store, batch_id):
    with store.connect() as conn:
        conn.execute("BEGIN")
        parent = retry_state(conn, batch_id)
        if find_retry(conn, batch_id):
            raise AnnotationConflict("A new batch was already created; inspect its results first")
    if not parent["frame_ids"]:
        raise ValueError("No unfinished images without saved proposals remain in this batch")
    config = parent["config"]
    options = {
        key: config[key]
        for key in (
            "source",
            "model",
            "comparison_id",
            "detector_model_id",
            "detector_variant",
            "threshold",
            "instructions",
        )
    }
    options["frame_ids"] = parent["frame_ids"]
    preview, _ = _prepare_batch(store, parent["session_id"], **options)
    fingerprint = _digest({"parent": parent, "preview": preview["fingerprint"]})
    return parent, options, preview, fingerprint


def preview_retry_batch(store: Store, batch_id: str) -> dict:
    parent, _, preview, fingerprint = _prepare_retry(store, batch_id)
    return {
        **preview,
        "fingerprint": fingerprint,
        "retry_of": batch_id,
        "retained_count": len(parent["children"]) - len(parent["frame_ids"]),
        "reason": (
            "This creates new local requests using the current saved annotations and settings. "
            "Successful images and images with saved proposals stay in the earlier batch. "
            "No completed provider computation is resumed."
        ),
    }


def retry_batch(store: Store, jobs, batch_id: str, *, name: str, expected_fingerprint: str) -> dict:
    # An HTTP response can be lost after commit. Repeating the same confirmation
    # returns its receipt without checking providers or creating another batch.
    with store.connect() as conn:
        existing = find_retry(conn, batch_id)
    if existing:
        if existing["config"].get("retry_fingerprint") != expected_fingerprint:
            raise AnnotationConflict("A new batch was already created; inspect its results first")
        return batch_detail(store, existing["id"])
    parent, options, preview, fingerprint = _prepare_retry(store, batch_id)
    if fingerprint != expected_fingerprint:
        raise AnnotationConflict("The unfinished batch inputs changed; preview them again")
    return create_batch(
        store,
        jobs,
        parent["session_id"],
        name=name,
        expected_fingerprint=preview["fingerprint"],
        _retry_parent=parent,
        _retry_fingerprint=fingerprint,
        **options,
    )
