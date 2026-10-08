"""Read-only job context and durable partial artifacts, without loading any model."""

from iris.projects import JOB_PARENTS, record_project
from iris.store import DEFAULT_PROJECT_ID, Store, _decode

WORKSPACES = {
    "extract": "intake",
    "infer": "comparison",
    "assist": "annotation",
    "dinox": "annotation",
    "train": "training",
    "evaluate": "evaluation",
    "video_review": "intake",
    "benchmark": "benchmark",
    "model_export": "training",
    "temporal_detect": "tracking",
    "tracking_compare": "tracking",
    "tracking_cost": "tracking",
    "tracking_study": "tracking",
}
NAMES = {
    "extract": "Frame extraction",
    "infer": "Model comparison",
    "assist": "Annotation assistance",
    "dinox": "DINO-X cloud proposals",
    "train": "Detector training",
    "evaluate": "Quality evaluation",
    "video_review": "Video passage review",
    "benchmark": "Annotation benchmark",
    "model_export": "Standalone model export",
    "temporal_detect": "Temporal detector cache",
    "tracking_compare": "Visual tracking comparison",
    "tracking_cost": "Tracking pipeline measurement",
    "tracking_study": "Tracking profile study",
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
    elif target and table == "temporal_detection_caches":
        sequence = store.get("temporal_sequences", target["sequence_id"])
        asset = store.get("assets", sequence["asset_id"]) if sequence else None
        session_id = asset["session_id"] if asset else None
    elif target and job["kind"] == "tracking_cost":
        sequence = store.get("temporal_sequences", job["params"]["sequence_id"])
        asset = store.get("assets", sequence["asset_id"]) if sequence else None
        session_id = asset["session_id"] if asset else None
    session = store.get("sessions", session_id) if session_id else None
    batch_id = job["params"].get("batch_id")
    if batch_id and job["kind"] != "dinox":
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
        elif target and job["kind"] == "temporal_detect":
            count = conn.execute(
                "SELECT COUNT(*) FROM temporal_detection_frames WHERE cache_id=?",
                (target["id"],),
            ).fetchone()[0]
            own_count = conn.execute(
                "SELECT COUNT(*) FROM temporal_detection_frames WHERE cache_id=? AND job_id=?",
                (target["id"], job_id),
            ).fetchone()[0]
            add("temporal_detection_frames", "Saved frames in this cache", count, target["id"])
            add(
                "temporal_detection_attempt_frames",
                "Frames saved by this attempt",
                own_count,
                target["id"],
            )
        elif target and job["kind"] == "tracking_compare":
            add(
                "tracking_comparison",
                "Complete visual tracking comparisons",
                int(job["status"] == "succeeded" and job["result"] is not None),
                job["id"],
            )
        elif target and job["kind"] == "tracking_cost":
            add(
                "tracking_cost",
                "Complete pipeline measurements",
                int(job["status"] == "succeeded" and job["result"] is not None),
                job["id"],
            )
        elif target and job["kind"] == "tracking_study":
            add(
                "tracking_study",
                "Complete profile studies",
                int(job["status"] == "succeeded" and job["result"] is not None),
                job["id"],
            )
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
        elif target and job["kind"] == "dinox":
            add(
                "suggestions",
                "Proposals saved for human review",
                len(store.list("annotation_suggestions", job_id=job_id)),
                target["id"],
            )
            request_ids = {r["request_id"] for r in target["metadata"]["frames"]}
            add(
                "raw_response",
                "Saved provider results",
                sum(
                    store.get("dinox_requests", rid)["raw_response"] is not None
                    for rid in request_ids
                ),
                target["id"],
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
    can_check = job["status"] in {"failed", "cancelled", "interrupted"} and (
        job["kind"] == "temporal_detect"
        or job["kind"] == "extract"
        and isinstance(job["params"].get("extraction_contract"), dict)
    )
    reason = (
        "Check the saved sampling plan and remaining positions before continuing."
        if can_check
        else "This task cannot resume in place. Prepare a new run explicitly if needed."
    )
    if can_check and job["kind"] == "temporal_detect":
        reason = "Check frozen detector settings and the remaining frames before continuing."
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
        "dinox": (
            "Preview the selected images again to reuse saved results or poll known remote tasks. "
            "Unknown submission outcomes are never resent automatically."
        ),
        "video_review": (
            "Prepare a new storyboard. External processing requires a fresh preview and approval."
        ),
        "extract": (
            "A new extraction uses a new sampling plan; it does not resume the earlier attempt."
        ),
        "temporal_detect": (
            "Reuse a complete cache or explicitly continue its remaining frames. "
            "Changed detector settings require a separate cache."
        ),
        "tracking_study": (
            "Open the completed study or preview a new bounded run. "
            "The baseline remains unchanged; test sequences remain reserved."
        ),
        "tracking_cost": (
            "Open the complete pipeline measurement or launch a fresh run. "
            "Imported declarations do not authenticate execution on a target machine."
        ),
        "tracking_compare": (
            "Open the saved visual comparison or launch both trackers fresh from the same "
            "complete detector cache. Tracker state cannot resume in place."
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
    from iris.tracking_studies import public_job

    return {
        "job": public_job(job),
        "context": {
            "name": (
                job["params"].get("name")
                if job["kind"] in {"tracking_compare", "tracking_cost", "tracking_study"}
                else None
            )
            or (target or {}).get("name")
            or (target or {}).get("filename")
            or NAMES[job["kind"]],
            "session_id": session_id,
            "session_name": session["name"] if session else None,
            "target_type": table,
            "target_id": target["id"] if target else None,
            "batch_id": batch_id,
            **(
                {"sequence_id": target["sequence_id"]}
                if job["kind"] == "temporal_detect" and target
                else {}
            ),
            **(
                {
                    "comparison_id": job["id"],
                    "sequence_id": job["params"]["sequence_id"],
                    "cache_id": job["params"]["cache_id"],
                }
                if job["kind"] == "tracking_compare"
                else {}
            ),
            **(
                {
                    "comparison_id": job["params"]["comparison_id"],
                    "sequence_id": job["params"]["sequence_id"],
                    "tracking_cost_id": job["id"],
                }
                if job["kind"] == "tracking_cost"
                else {}
            ),
            **(
                {"tracking_study_id": job["id"], "dataset_id": job["params"]["dataset_id"]}
                if job["kind"] == "tracking_study"
                else {}
            ),
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
