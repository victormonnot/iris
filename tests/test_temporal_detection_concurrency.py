"""Synthetic detector workers exercise publication races without loading a model."""

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Event

import pytest
from test_temporal_detection_api import client as api_client_fixture
from test_temporal_detection_api import create, sequence
from test_temporal_detector import runtime_metadata

from iris import temporal_detection_worker as worker
from iris import temporal_detections as caches
from iris.jobs import JobManager
from iris.store import now

client = api_client_fixture


class SyntheticDetector:
    def __init__(self, root, settings):
        self.metadata = runtime_metadata(settings)

    def warmup(self, image):
        pass

    def predict(self, image):
        return {
            "input_size": list(image.size),
            "detections": [
                {"label_id": 1, "label": "person", "score": 0.8, "box": [2, 3, 20, 40]},
            ],
            "timing": {
                "preprocess_ms": 1,
                "inference_ms": 2,
                "postprocess_ms": 1,
                "total_ms": 4,
            },
        }


def running_cache(client, tmp_path):
    source = sequence(client, tmp_path)
    cache = create(client, source)
    store = client.app.state.store
    store.update("jobs", cache["job_id"], {"status": "running", "started_at": now()})
    return store, cache


def run(store, job_id, *, factory=SyntheticDetector, cancelled=lambda: False):
    return worker.run(store, job_id, lambda *_args: None, cancelled, detector_factory=factory)


def test_cancellation_between_prediction_and_commit_preserves_only_the_prior_prefix(
    client, tmp_path, monkeypatch
):
    store, cache = running_cache(client, tmp_path)
    original_publish = worker._publish
    saved_prefix = []

    def cancel_before_second_commit(store, cache, job_id, token, payload, position):
        if position == 1:
            saved_prefix.extend(
                deepcopy(store.list("temporal_detection_frames", cache_id=cache["id"]))
            )
            JobManager(store).cancel(job_id)
        return original_publish(store, cache, job_id, token, payload, position)

    monkeypatch.setattr(worker, "_publish", cancel_before_second_commit)
    result = run(store, cache["job_id"])
    assert result["cancelled"]
    assert result["produced_count"] == result["completed_count"] == 1
    assert store.list("temporal_detection_frames", cache_id=cache["id"]) == saved_prefix
    assert store.get("jobs", cache["job_id"])["cancel_requested"]
    detail = caches.get_detection_cache(store, cache["id"])
    assert detail["coverage"]["state"] == "partial"
    assert detail["coverage"]["remaining_count"] == 2


def test_stopped_late_worker_cannot_publish_or_rewrite_its_terminal_attempt_after_recovery(
    client, tmp_path
):
    store, cache = running_cache(client, tmp_path)
    predicting, release = Event(), Event()

    class HeldDetector(SyntheticDetector):
        def predict(self, image):
            predicting.set()
            if not release.wait(5):
                raise AssertionError("Synthetic detector was not released")
            return super().predict(image)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(run, store, cache["job_id"], factory=HeldDetector)
        try:
            assert predicting.wait(5), "Worker did not reach the controlled inference boundary"
            store.update(
                "jobs",
                cache["job_id"],
                {
                    "status": "interrupted",
                    "finished_at": now(),
                    "message": "Synthetic supervisor stop",
                },
            )
            original = store.get("jobs", cache["job_id"])
            preview = caches.preview_detection_recovery(store, cache["job_id"])
            assert preview["available"]
            successor = caches.recover_detection_cache(
                store, JobManager(store), cache["job_id"], fingerprint=preview["fingerprint"]
            )
            assert successor["status"] == "queued"
        finally:
            release.set()
        assert future.result(timeout=5) == original["result"]
    assert store.get("jobs", cache["job_id"]) == original
    assert store.get("jobs", successor["id"]) == successor
    assert store.list("temporal_detection_frames", cache_id=cache["id"]) == []
    assert caches.get_detection_cache(store, cache["id"])["coverage"]["state"] == "empty"


def test_two_workers_cannot_claim_one_attempt_or_run_two_detectors(client, tmp_path):
    store, cache = running_cache(client, tmp_path)
    loading, release = Event(), Event()
    constructed = []

    def held_factory(root, settings):
        constructed.append(settings["model_id"])
        loading.set()
        if not release.wait(5):
            raise AssertionError("Synthetic detector factory was not released")
        return SyntheticDetector(root, settings)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(run, store, cache["job_id"], factory=held_factory)
        try:
            assert loading.wait(5), "First worker did not claim the attempt"
            token = store.get("jobs", cache["job_id"])["result"]["worker_token"]
            with pytest.raises(caches.DetectionCacheConflict, match="already claimed"):
                run(store, cache["job_id"], factory=held_factory)
            assert store.get("jobs", cache["job_id"])["result"]["worker_token"] == token
            assert len(constructed) == 1
        finally:
            release.set()
        result = future.result(timeout=5)
    assert result["completed_count"] == result["produced_count"] == 3
    assert not result["cancelled"]
    assert len(store.list("temporal_detection_frames", cache_id=cache["id"])) == 3
    assert caches.get_detection_cache(store, cache["id"])["coverage"]["state"] == "complete"


def test_frame_and_checkpoint_roll_back_together_when_publication_fails(
    client, tmp_path, monkeypatch
):
    store, cache = running_cache(client, tmp_path)
    original_write = worker._write_result
    reached_checkpoint = []

    def fail_after_checkpoint_update(conn, job_id, result):
        original_write(conn, job_id, result)
        if result["produced_count"] == 1:
            reached_checkpoint.append(result["completed_count"])
            raise RuntimeError("Synthetic failure after writing the frame and checkpoint")

    monkeypatch.setattr(worker, "_write_result", fail_after_checkpoint_update)
    with pytest.raises(RuntimeError, match="after writing the frame and checkpoint"):
        run(store, cache["job_id"])
    assert reached_checkpoint == [1]
    assert store.list("temporal_detection_frames", cache_id=cache["id"]) == []
    job = store.get("jobs", cache["job_id"])
    assert job["result"]["execution"] is not None
    assert job["result"]["produced_count"] == job["result"]["completed_count"] == 0
    assert caches.get_detection_cache(store, cache["id"])["coverage"]["state"] == "empty"


def test_cancel_cleanup_serializes_with_supervisor_before_mutating_a_terminal_result(
    client, tmp_path, monkeypatch
):
    store, cache = running_cache(client, tmp_path)
    original_row = worker._row
    supervisor_was_blocked = []

    def supervisor_attempt_after_cleanup_read(conn, table, identifier):
        row = original_row(conn, table, identifier)
        if table == "jobs" and row["result"]["worker_token"] and row["status"] == "running":
            # A zero-timeout second writer makes this race deterministic. The
            # cleanup transaction must prevent the supervisor from committing
            # a terminal status between its status check and result write.
            with store.connect() as supervisor:
                supervisor.execute("PRAGMA busy_timeout=0")
                try:
                    supervisor.execute(
                        "UPDATE jobs SET status='interrupted' WHERE id=?", (identifier,)
                    )
                except sqlite3.OperationalError as exc:
                    assert "locked" in str(exc)
                    supervisor_was_blocked.append(True)
                else:
                    supervisor_was_blocked.append(False)
        return row

    def forbidden_factory(*_args):
        pytest.fail("Cancellation before loading must not construct a model")

    monkeypatch.setattr(worker, "_row", supervisor_attempt_after_cleanup_read)
    result = run(store, cache["job_id"], factory=forbidden_factory, cancelled=lambda: True)
    assert supervisor_was_blocked == [True]
    assert result["cancelled"]
    assert result["produced_count"] == 0
    store.update("jobs", cache["job_id"], {"status": "interrupted", "finished_at": now()})
    terminal = store.get("jobs", cache["job_id"])
    assert terminal["result"] == result
    assert store.list("temporal_detection_frames", cache_id=cache["id"]) == []
