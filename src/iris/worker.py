"""Subprocess entry point for bounded media extraction."""

import ctypes
import os
import signal
import sys
import traceback
from pathlib import Path

from iris.media import extract_frames
from iris.store import Store, now


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
        store.update(
            "jobs",
            job_id,
            {
                "progress": max(0, min(1, value)),
                "message": message,
                "logs": [*current["logs"][-199:], f"{now()} {message}"],
            },
        )
        print(message, flush=True)

    try:
        result = extract_frames(
            store,
            job["params"]["asset_id"],
            {**job["params"]["config"], "job_id": job_id},
            progress,
            cancelled,
        )
        current = store.get("jobs", job_id)
        if current["status"] != "running":
            return
        status = (
            "cancelled"
            if current["cancel_requested"]
            else ("interrupted" if stopping else "succeeded")
        )
        store.update(
            "jobs",
            job_id,
            {
                "status": status,
                "result": result,
                "finished_at": now(),
                "progress": 1 if status == "succeeded" else current["progress"],
                "message": "Extraction complete"
                if status == "succeeded"
                else "Extraction stopped; existing frames are preserved",
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
        store.update(
            "jobs",
            job_id,
            {
                "status": status,
                "error": str(exc),
                "finished_at": now(),
                "message": "Extraction failed" if status == "failed" else "Extraction stopped",
            },
        )


if __name__ == "__main__":
    run(Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]))
