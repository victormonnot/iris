"""Read-only job context and durable partial artifacts, without loading any model."""

from iris.projects import JOB_PARENTS, record_project
from iris.store import DEFAULT_PROJECT_ID, Store, _decode

WORKSPACES = {
    "extract": "intake",
    "infer": "comparison",
    "assist": "annotation",
    "train": "training",
    "evaluate": "evaluation",
    "video_review": "intake",
    "benchmark": "benchmark",
    "model_export": "training",
}
NAMES = {
    "extract": "Frame extraction",
    "infer": "Model comparison",
    "assist": "Annotation assistance",
    "train": "Detector training",
    "evaluate": "Quality evaluation",
    "video_review": "Video passage review",
    "benchmark": "Annotation benchmark",
    "model_export": "Standalone model export",
}


def job_detail(store: Store, job_id: str, project_id: str = DEFAULT_PROJECT_ID) -> dict:
    from iris.job_dispatch import dispatch_summary

    job = store.get("jobs", job_id)
    if job is None or record_project(store, "jobs", job) != project_id:
        raise KeyError(job_id)
    table, key = JOB_PARENTS[job["kind"]]
    target = store.get(table, job["params"].get(key, ""))
    if target is None:
        # Old jobs may have only a reverse foreign key in their associated record.
        with store.connect() as conn:
            if "job_id" in store.columns[table]:
                target = _decode(
                    conn.execute(f"SELECT * FROM {table} WHERE job_id=?", (job_id,)).fetchone()
                )
    session_id = target.get("session_id") if target else None
    if target and table == "assistance_records":
        session_id = store.get("frames", target["frame_id"])["session_id"]
    elif target and table == "video_reviews":
        session_id = store.get("assets", target["asset_id"])["session_id"]
    session = store.get("sessions", session_id) if session_id else None
    batch_id = job["params"].get("batch_id")
    if batch_id:
        batch = store.get("assistance_batches", batch_id)
        if batch is None or record_project(store, "assistance_batches", batch) != project_id:
            batch_id = None
    artifacts = []

    def add(kind, label, count, target_id=None):
        artifacts.append({"kind": kind, "label": label, "count": count, "target_id": target_id})

    with store.connect() as conn:
        conn.execute("BEGIN")
        if job["kind"] == "extract":
            count = conn.execute(
                "SELECT COUNT(*) FROM frames WHERE json_extract(extraction,'$.job_id')=?",
                (job_id,),
            ).fetchone()[0]
            add("frames", "Images saved by this attempt", count, session_id)
        elif target and job["kind"] == "infer":
            count = conn.execute(
                "SELECT COUNT(*) FROM predictions WHERE comparison_id=?", (target["id"],)
            ).fetchone()[0]
            add("predictions", "Saved image/model predictions", count, target["id"])
        elif target and job["kind"] == "assist":
            count = conn.execute(
                "SELECT COUNT(*) FROM annotation_suggestions WHERE job_id=?", (job_id,)
            ).fetchone()[0]
            add("suggestions", "Proposals saved for human review", count, target["frame_id"])
            add(
                "raw_response",
                "Saved provider response records",
                int(target["raw_response"] is not None),
            )
        elif target and job["kind"] == "train":
            add(
                "training_points",
                "Saved training history entries",
                len(target["history"]),
                target["id"],
            )
            add(
                "checkpoint",
                "Published model checkpoints",
                int(bool(target["checkpoint_id"])),
                target["checkpoint_id"],
            )
            states = conn.execute(
                "SELECT count(*) FROM training_checkpoints WHERE training_id=?", (target["id"],)
            ).fetchone()[0]
            add("optimizer_states", "Durable optimizer checkpoints", states, target["id"])
        elif target and job["kind"] == "evaluate":
            count = conn.execute(
                "SELECT COUNT(*) FROM evaluation_predictions WHERE evaluation_id=?", (target["id"],)
            ).fetchone()[0]
            add("predictions", "Saved evaluation predictions", count, target["id"])
            measured = conn.execute(
                "SELECT COUNT(*) FROM evaluation_models "
                "WHERE evaluation_id=? AND metrics IS NOT NULL",
                (target["id"],),
            ).fetchone()[0]
            add("metrics", "Saved model metric results", measured, target["id"])
        elif target and job["kind"] == "video_review":
            add(
                "raw_response",
                "Saved provider response records",
                int(target["raw_response"] is not None),
            )
            add(
                "passages",
                "Saved proposed passages",
                len((target["result"] or {}).get("passages", [])),
                target["id"],
            )
        elif target and job["kind"] == "benchmark":
            count = conn.execute(
                "SELECT COUNT(*) FROM benchmark_outputs WHERE trial_id=?", (target["id"],)
            ).fetchone()[0]
            add("benchmark_outputs", "Saved benchmark image results", count, target["id"])
        elif target and job["kind"] == "model_export":
            add(
                "model_export",
                "Published standalone model packages",
                int(bool(target["path"])),
                target["id"],
            )
        children = [
            _decode(row)
            for row in conn.execute(
                "SELECT * FROM jobs WHERE json_extract(params,'$.recovery_of')=? "
                "ORDER BY created_at,id",
                (job_id,),
            )
        ]
    parent_id = job["params"].get("recovery_of")
    preannotation = bool(
        job["kind"] == "infer" and target and target.get("config", {}).get("preannotation")
    )
    if preannotation:
        add(
            "suggestions",
            "Proposals saved for human review",
            len(store.list("annotation_suggestions", job_id=job_id)),
            target["id"],
        )
    if parent_id:
        parent = store.get("jobs", parent_id)
        if parent is None or record_project(store, "jobs", parent) != project_id:
            parent_id = None
    can_check = (
        job["kind"] == "extract"
        and job["status"] in {"failed", "cancelled", "interrupted"}
        and isinstance(job["params"].get("extraction_contract"), dict)
    )
    reason = (
        "Check the saved sampling plan and remaining positions before continuing."
        if can_check
        else "This task cannot resume in place. Prepare a new run explicitly if needed."
    )
    next_reason = {
        "train": (
            "A new training run starts from its chosen checkpoint; optimizer state is not resumed."
        ),
        "infer": (
            "A new comparison keeps the earlier predictions intact "
            "and runs its selected images again."
        ),
        "evaluate": "A new evaluation keeps the earlier predictions and metrics intact.",
        "model_export": "Preview a new copy attempt; published packages remain immutable.",
        "benchmark": (
            "Preview an explicit new benchmark trial; earlier outputs and corrections stay intact."
        ),
        "assist": (
            "Prepare a fresh review and inspect its images. "
            "External processing requires new cost approval."
        ),
        "video_review": (
            "Prepare a new storyboard. External processing requires a fresh preview and approval."
        ),
        "extract": (
            "A new extraction uses a new sampling plan; it does not resume the earlier attempt."
        ),
    }[job["kind"]]
    if preannotation:
        next_reason = (
            "Preview an explicit new proposal run. Saved predictions, proposals and human "
            "corrections from this attempt remain available."
        )
    if job["kind"] == "train" and target and target["config"].get("checkpoint_protocol"):
        reason = "Open training results to inspect saved optimizer checkpoints and resume options."
        next_reason = (
            "Preview an explicit resume in the training results. It creates a new attempt "
            "with the same settings and dataset, restoring optimizer and random state."
        )
    return {
        "job": job,
        "context": {
            "name": (target or {}).get("name")
            or (target or {}).get("filename")
            or NAMES[job["kind"]],
            "session_id": session_id,
            "session_name": session["name"] if session else None,
            "target_type": table,
            "target_id": target["id"] if target else None,
            "batch_id": batch_id,
        },
        "artifacts": artifacts,
        "dispatch": dispatch_summary(store, job),
        "lineage": {
            "parent_job_id": parent_id,
            "child_job_ids": [
                child["id"]
                for child in children
                if record_project(store, "jobs", child) == project_id
            ],
        },
        "recovery": {"can_check": can_check, "reason": reason},
        "next_action": {
            "workspace": "annotation" if preannotation else WORKSPACES[job["kind"]],
            "label": "Prepare a new run",
            "reason": next_reason,
        },
    }
