"""Project ownership for the single-user workspace.

Ownership follows existing foreign keys rather than rewriting immutable experiment
snapshots. Official model weights and workspace transfers remain shared resources.
"""

from iris.store import DEFAULT_PROJECT_ID, Store, new_id, now

# Each hop uses an existing foreign key; project membership cannot be changed by
# passing a different query parameter or by editing a saved manifest.
PARENTS = {
    "assets": ("sessions", "session_id"),
    "frames": ("sessions", "session_id"),
    "comparisons": ("sessions", "session_id"),
    "runs": ("comparisons", "comparison_id"),
    "predictions": ("comparisons", "comparison_id"),
    "annotation_revisions": ("frames", "frame_id"),
    "annotation_suggestions": ("frames", "frame_id"),
    "assistance_records": ("frames", "frame_id"),
    "assistance_previews": ("frames", "frame_id"),
    "assistance_batches": ("sessions", "session_id"),
    "training_runs": ("dataset_versions", "dataset_id"),
    "trained_models": ("training_runs", "training_id"),
    "evaluations": ("dataset_versions", "dataset_id"),
    "evaluation_models": ("evaluations", "evaluation_id"),
    "evaluation_predictions": ("evaluations", "evaluation_id"),
    "model_references": ("evaluations", "evaluation_id"),
    "video_reviews": ("assets", "asset_id"),
    "experiment_reports": ("evaluations", "evaluation_id"),
    "benchmark_configs": ("benchmarks", "benchmark_id"),
    "benchmark_trials": ("benchmarks", "benchmark_id"),
    "benchmark_outputs": ("benchmark_trials", "trial_id"),
    "benchmark_corrections": ("benchmark_outputs", "output_id"),
    "benchmark_timers": ("benchmark_outputs", "output_id"),
    "benchmark_reports": ("benchmarks", "benchmark_id"),
}
DIRECT = {"sessions", "dataset_versions", "dataset_imports", "taxonomy_versions", "benchmarks"}
JOB_PARENTS = {
    "extract": ("assets", "asset_id"),
    "infer": ("comparisons", "comparison_id"),
    "assist": ("assistance_records", "assistance_id"),
    "train": ("training_runs", "training_id"),
    "evaluate": ("evaluations", "evaluation_id"),
    "video_review": ("video_reviews", "video_review_id"),
    "benchmark": ("benchmark_trials", "trial_id"),
}


def create_project(store: Store, *, name: str, description: str = "") -> dict:
    name, description = name.strip(), description.strip()
    if not name or len(name) > 160 or len(description) > 2000:
        raise ValueError("Use a project name of 1–160 characters and a description up to 2000.")
    return store.insert(
        "projects",
        {
            "id": new_id(),
            "name": name,
            "description": description,
            "taxonomy_id": "iris-objects-v1",
            "created_at": now(),
        },
    )


def record_project(store: Store, table: str, record: dict) -> str | None:
    """Resolve ownership, including old jobs with only a reverse foreign key."""
    if table in DIRECT:
        return record["project_id"]
    if table == "projects":
        return record["id"]
    if table == "jobs":
        parent = JOB_PARENTS.get(record["kind"])
        if parent:
            parent_table, key = parent
            identifier = record["params"].get(key)
            if identifier:
                row = store.get(parent_table, identifier)
                if row:
                    return record_project(store, parent_table, row)
        # Older job receipts and imported snapshots may omit params. Their
        # owning comparison/training/etc still provides an unambiguous link.
        for parent_table in (
            "comparisons",
            "assistance_records",
            "training_runs",
            "evaluations",
            "video_reviews",
            "benchmark_trials",
        ):
            rows = store.list(parent_table, job_id=record["id"])
            if rows:
                return record_project(store, parent_table, rows[0])
        return DEFAULT_PROJECT_ID
    if table in PARENTS:
        parent_table, key = PARENTS[table]
        parent = store.get(parent_table, record[key])
        return record_project(store, parent_table, parent) if parent else None
    raise ValueError(f"No project ownership rule for {table}")


def project_records(store: Store, table: str, project_id: str) -> list[dict]:
    if table in DIRECT:
        return store.list(table, project_id=project_id)
    return [row for row in store.list(table) if record_project(store, table, row) == project_id]
