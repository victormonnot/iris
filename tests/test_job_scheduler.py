"""Queue dispatch survives SQLite contention without repeating worker execution."""

import sqlite3
import threading
from contextlib import contextmanager

import pytest

from iris.jobs import JobManager, update_running
from iris.store import Store, new_id, now


def queued_job(store):
    return store.insert(
        "jobs",
        {"id": new_id(), "kind": "extract", "status": "queued", "params": {}, "created_at": now()},
    )


@pytest.mark.parametrize("action", ["release", "cancel", "stop"])
def test_dispatch_recovers_from_real_sqlite_write_lock(tmp_path, monkeypatch, action):
    store = Store(tmp_path / "workspace")
    first, second = queued_job(store), queued_job(store)
    manager = JobManager(store)
    lock_seen = threading.Event()
    finished = threading.Event()
    executed = []
    errors = []
    connect = store.connect

    @contextmanager
    def short_busy_timeout():
        with connect() as conn:
            # Exercise the real SQLite timeout without a 30-second test delay.
            conn.execute("PRAGMA busy_timeout=20")
            try:
                yield conn
            except sqlite3.OperationalError as exc:
                if exc.sqlite_errorcode == sqlite3.SQLITE_BUSY:
                    lock_seen.set()
                raise

    monkeypatch.setattr(store, "connect", short_busy_timeout)

    def execute(job):
        executed.append(job["id"])
        assert store.get("jobs", job["id"])["status"] == "running"
        update_running(store, job["id"], {"status": "succeeded", "finished_at": now()})
        if job["id"] == second["id"]:
            finished.set()

    monkeypatch.setattr(manager, "_execute", execute)

    def run():
        try:
            manager._run()
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run)
    try:
        with connect() as writer:
            writer.execute("BEGIN IMMEDIATE")
            thread.start()
            assert lock_seen.wait(5), "The scheduler never encountered the real write lock"
            assert not executed
            assert store.get("jobs", first["id"])["status"] == "queued"
            assert store.get("jobs", first["id"])["started_at"] is None
            if action == "cancel":
                # A cancellation committed before the next claim must still win.
                writer.execute(
                    "UPDATE jobs SET status='cancelled', cancel_requested=1 WHERE id=?",
                    (first["id"],),
                )
            elif action == "stop":
                manager.stop_event.set()
                thread.join(timeout=2)
                assert not thread.is_alive(), "Retrying a busy queue must allow shutdown"
        if action != "stop":
            assert finished.wait(5), f"Queue did not recover after the lock: {errors!r}"
    finally:
        manager.stop_event.set()
        if thread.ident is not None:
            thread.join(timeout=5)

    assert not thread.is_alive()
    assert not errors
    if action == "stop":
        assert not executed
        assert all(job["status"] == "queued" for job in store.list("jobs"))
    else:
        assert executed == ([first["id"], second["id"]] if action == "release" else [second["id"]])
        assert store.get("jobs", second["id"])["status"] == "succeeded"
        expected = "succeeded" if action == "release" else "cancelled"
        assert store.get("jobs", first["id"])["status"] == expected


@pytest.mark.parametrize(
    "code", [sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED, sqlite3.SQLITE_BUSY_SNAPSHOT]
)
def test_transient_queue_read_errors_are_retried(tmp_path, monkeypatch, code):
    manager = JobManager(Store(tmp_path / "workspace"))
    error = sqlite3.OperationalError("Synthetic queue read contention")
    error.sqlite_errorcode = code
    calls = []

    def read_queue(*_args, **_kwargs):
        calls.append(True)
        if len(calls) == 1:
            raise error
        manager.stop_event.set()
        return []

    monkeypatch.setattr(manager.store, "list", read_queue)
    manager._run()
    assert len(calls) == 2


def test_other_sqlite_errors_are_not_treated_as_contention(tmp_path, monkeypatch):
    manager = JobManager(Store(tmp_path / "workspace"))

    def invalid_queue(*_args, **_kwargs):
        with manager.store.connect() as conn:
            return conn.execute("SELECT * FROM missing_jobs_table").fetchall()

    monkeypatch.setattr(manager.store, "list", invalid_queue)
    with pytest.raises(sqlite3.OperationalError, match="no such table"):
        manager._run()


def test_shutdown_after_claim_commit_does_not_start_a_worker(tmp_path, monkeypatch):
    store = Store(tmp_path / "workspace")
    job = queued_job(store)
    manager = JobManager(store)
    executed = []
    connect = store.connect

    @contextmanager
    def stop_after_commit():
        with connect() as conn:
            yield conn
            wrote = conn.in_transaction
        if wrote:
            manager.stop_event.set()

    monkeypatch.setattr(store, "connect", stop_after_commit)
    monkeypatch.setattr(manager, "_execute", lambda job: executed.append(job["id"]))
    manager._run()
    assert not executed
    claimed = store.get("jobs", job["id"])
    assert claimed["status"] == "running" and claimed["started_at"]
    manager._interrupt_unfinished()
    assert store.get("jobs", job["id"])["status"] == "interrupted"


def test_worker_errors_fail_the_job_without_reexecuting_it(tmp_path, monkeypatch):
    store = Store(tmp_path / "workspace")
    first, second = queued_job(store), queued_job(store)
    manager = JobManager(store)
    executed = []

    def execute(job):
        executed.append(job["id"])
        if job["id"] == first["id"]:
            error = sqlite3.OperationalError("Worker encountered a write lock")
            error.sqlite_errorcode = sqlite3.SQLITE_BUSY
            raise error
        update_running(store, job["id"], {"status": "succeeded", "finished_at": now()})
        manager.stop_event.set()

    monkeypatch.setattr(manager, "_execute", execute)
    manager._run()
    assert executed == [first["id"], second["id"]]
    assert store.get("jobs", first["id"])["status"] == "failed"
    assert store.get("jobs", first["id"])["error"] == "Worker encountered a write lock"
    assert store.get("jobs", second["id"])["status"] == "succeeded"
