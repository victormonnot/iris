"""One local worker at a time; job state survives application restarts."""

import fcntl
import os
import subprocess
import sys
import threading
import time

from iris.store import Store, new_id, now

ACTIVE = {"queued", "running"}


class JobManager:
    def __init__(self, store: Store):
        self.store = store
        self.stop_event = threading.Event()
        self.thread = None
        self.guard = threading.Lock()

    def start(self):
        self.lock_file = (self.store.root / ".server.lock").open("a")
        try:
            fcntl.flock(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.lock_file.close()
            raise RuntimeError(
                "This data directory is already open in another IRIS server"
            ) from None
        self._interrupt_unfinished()
        self.thread = threading.Thread(target=self._run, name="iris-jobs", daemon=True)
        self.thread.start()

    def _interrupt_unfinished(self):
        with self.store.connect() as conn:
            conn.execute(
                "UPDATE jobs SET status='interrupted', finished_at=?, "
                "message='Server stopped; artifacts preserved. Start a new job to retry.' "
                "WHERE status IN ('queued','running')",
                (now(),),
            )

    def close(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=10)
        self._interrupt_unfinished()
        fcntl.flock(self.lock_file, fcntl.LOCK_UN)
        self.lock_file.close()

    def submit(self, asset_id: str, config: dict) -> dict:
        with self.guard:
            for job in self.store.list("jobs"):
                if (
                    job["kind"] == "extract"
                    and job["status"] in ACTIVE
                    and job["params"]["asset_id"] == asset_id
                ):
                    raise ValueError("Extraction is already queued or running for this video")
            return self.store.insert(
                "jobs",
                {
                    "id": new_id(),
                    "kind": "extract",
                    "status": "queued",
                    "params": {"asset_id": asset_id, "config": config},
                    "created_at": now(),
                    "message": "Waiting for the local worker",
                },
            )

    def cancel(self, job_id: str) -> dict:
        with self.guard, self.store.connect() as conn:
            conn.execute(
                "UPDATE jobs SET cancel_requested=1, "
                "status=CASE WHEN status='queued' THEN 'cancelled' ELSE status END, "
                "finished_at=CASE WHEN status='queued' THEN ? ELSE finished_at END, "
                "message='Cancellation requested; saved artifacts are preserved' "
                "WHERE id=? AND status IN ('queued','running')",
                (now(), job_id),
            )
        return self.store.get("jobs", job_id)

    def _run(self):
        while not self.stop_event.wait(0.2):
            queued = self.store.list("jobs", status="queued")
            if not queued:
                continue
            job = queued[0]
            with self.store.connect() as conn:
                claimed = conn.execute(
                    "UPDATE jobs SET status='running', started_at=?, message='Starting local job' "
                    "WHERE id=? AND status='queued' AND cancel_requested=0",
                    (now(), job["id"]),
                ).rowcount
            if not claimed:
                continue
            try:
                self._execute(job)
            except Exception as exc:
                self.store.update(
                    "jobs",
                    job["id"],
                    {
                        "status": "failed",
                        "error": str(exc),
                        "finished_at": now(),
                        "message": "Unable to run the local worker",
                    },
                )

    def _execute(self, job: dict):
        log_dir = self.store.root / "logs"
        log_dir.mkdir(exist_ok=True)
        with (log_dir / f"{job['id']}.log").open("ab") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "iris.worker",
                    str(self.store.root),
                    job["id"],
                    str(os.getpid()),
                ],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
            )
            terminating_at = None
            while process.poll() is None:
                current = self.store.get("jobs", job["id"])
                should_stop = self.stop_event.is_set() or current["cancel_requested"]
                if should_stop and terminating_at is None:
                    process.terminate()
                    terminating_at = time.monotonic()
                if terminating_at is not None and time.monotonic() - terminating_at > 3:
                    process.kill()
                time.sleep(0.05)
            current = self.store.get("jobs", job["id"])
            if current["status"] == "running":
                status = "failed"
                if self.stop_event.is_set():
                    status = "interrupted"
                elif current["cancel_requested"]:
                    status = "cancelled"
                self.store.update(
                    "jobs",
                    job["id"],
                    {
                        "status": status,
                        "finished_at": now(),
                        "message": f"Worker {status}; saved artifacts are preserved",
                        "error": f"Worker exited with code {process.returncode}"
                        if status == "failed"
                        else None,
                    },
                )
