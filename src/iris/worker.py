"""Subprocess entry point for local media, model and annotation jobs."""

import ctypes
import os
import signal
import sys
import traceback
from pathlib import Path

from iris.assistance import run_assistance
from iris.evaluation import run_evaluation
from iris.inference import run_comparison
from iris.jobs import update_running
from iris.media import extract_frames
from iris.store import Store, now
from iris.training import run_training
from iris.video_reviews import run_video_review


def run(root: Path, job_id: str, parent_pid: int):
    store = Store(root)
    job = store.get("jobs", job_id)
    stopping = False

    def request_stop(_signal, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    # Linux/WSL: an abruptly terminated server must not leave a worker writing
    # to a reopened workspace. Check the parent again to close the startup race.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "Cannot tie worker lifetime to server")
    if os.getppid() != parent_pid or job is None or job["status"] != "running":
        return

    def cancelled():
        current = store.get("jobs", job_id)
        return stopping or current["cancel_requested"] or current["status"] != "running"

    def progress(value: float, message: str):
        current = store.get("jobs", job_id)
        if current["status"] != "running":
            return
        update_running(
            store,
            job_id,
            {
                "progress": max(0, min(1, value)),
                "message": message,
                "logs": [*current["logs"][-199:], f"{now()} {message}"],
            },
        )
        print(message, flush=True)

    try:
        if job["kind"] == "extract":
            if "extraction_contract" in job["params"]:
                from iris.job_recovery import run_durable_extraction

                result = run_durable_extraction(store, job_id, progress, cancelled)
            else:
                result = extract_frames(
                    store,
                    job["params"]["asset_id"],
                    {**job["params"]["config"], "job_id": job_id},
                    progress,
                    cancelled,
                    plan=job["params"]["config"].get("passages_plan"),
                )
        elif job["kind"] == "infer":
            result = run_comparison(store, job["params"]["comparison_id"], progress, cancelled)
        elif job["kind"] == "assist":
            result = run_assistance(store, job["params"]["assistance_id"], progress, cancelled)
        elif job["kind"] == "train":
            result = run_training(store, job["params"]["training_id"], progress, cancelled)
        elif job["kind"] == "evaluate":
            result = run_evaluation(store, job["params"]["evaluation_id"], progress, cancelled)
        elif job["kind"] == "video_review":
            result = run_video_review(store, job["params"]["video_review_id"], progress, cancelled)
        else:
            raise ValueError(f"Unsupported job kind: {job['kind']}")
        current = store.get("jobs", job_id)
        if current["status"] != "running":
            return
        status = (
            "cancelled"
            if current["cancel_requested"]
            else ("interrupted" if stopping else "succeeded")
        )
        preannotation = result.get("preannotation")
        preannotation_failed = bool(
            status == "succeeded"
            and isinstance(preannotation, dict)
            and preannotation.get("frames_issues", 0)
            and not preannotation.get("frames_ready", 0)
        )
        if preannotation_failed:
            status = "failed"
        preannotation_message = (
            f"Preannotation complete: {preannotation['frames_ready']} images ready for review, "
            f"{preannotation['frames_issues']} need attention"
            if isinstance(preannotation, dict)
            else None
        )
        update_running(
            store,
            job_id,
            {
                "status": status,
                "result": result,
                "finished_at": now(),
                "error": "No image could publish proposals; inspect the saved per-image results"
                if preannotation_failed
                else current["error"],
                "progress": 1 if status == "succeeded" else current["progress"],
                "message": (
                    preannotation_message
                    or {
                        "extract": "Extraction complete",
                        "infer": "Comparison complete",
                        "assist": "Annotation proposals ready for human review",
                        "train": "Training complete; checkpoint available in the comparator",
                        "evaluate": "Evaluation complete; metrics and predictions saved",
                        "video_review": "Video passages ready for human selection",
                    }[job["kind"]]
                )
                if status == "succeeded" or preannotation_failed
                else "Job stopped; saved artifacts are preserved",
            },
        )
    except Exception as exc:
        traceback.print_exc()
        current = store.get("jobs", job_id)
        if current["status"] != "running":
            return
        status = (
            "cancelled"
            if current["cancel_requested"]
            else ("interrupted" if stopping else "failed")
        )
        update_running(
            store,
            job_id,
            {
                "status": status,
                "error": str(exc),
                "finished_at": now(),
                "message": "Job failed" if status == "failed" else "Job stopped",
            },
        )


if __name__ == "__main__":
    run(Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]))
