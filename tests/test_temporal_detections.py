"""Temporal detector outputs survive failures and never silently change recipes."""

from copy import deepcopy

import pytest
from test_temporal_detection_api import client as client
from test_temporal_detection_api import create, sequence
from test_temporal_detector import runtime_metadata

from iris import temporal_detections as caches
from iris.store import now
from iris.temporal_detection_worker import run


class Detector:
    def __init__(self, settings, *, calls, mutate=None, threads=None):
        self.metadata = runtime_metadata(settings)
        if threads is not None:
            self.metadata["threads"] = threads
        self.calls = calls
        self.mutate = mutate

    def warmup(self, image):
        pass

    def predict(self, image):
        marker = image.getpixel((0, 0))[0]
        self.calls.append(marker)
        result = {
            "input_size": list(image.size),
            "detections": []
            if marker == 13
            else [
                {"label_id": 1, "label": "person", "score": 0.02, "box": [2, 3, 20, 40]},
                {"label_id": 3, "label": "car", "score": 0.3, "box": [30, 5, 65, 50]},
                {"label_id": 1, "label": "person", "score": 0.8, "box": [3, 4, 21, 41]},
            ],
            "timing": {
                "preprocess_ms": 1,
                "inference_ms": 2,
                "postprocess_ms": 1,
                "total_ms": 4,
            },
        }
        if self.mutate:
            self.mutate(result)
        return result


def execute(store, cache, *, job=None, calls=None, progress=None, stop=None, **detector_args):
    identifier = job or cache["job_id"]
    store.update("jobs", identifier, {"status": "running", "started_at": now()})
    try:
        result = run(
            store,
            identifier,
            progress or (lambda *_: None),
            stop or (lambda: False),
            lambda root, settings: Detector(
                settings, calls=calls if calls is not None else [], **detector_args
            ),
        )
    except Exception:
        store.update("jobs", identifier, {"status": "failed", "finished_at": now()})
        raise
    store.update(
        "jobs",
        identifier,
        {"status": "cancelled" if result["cancelled"] else "succeeded", "finished_at": now()},
    )
    return result


def continuation(client, job_id):
    store = client.app.state.store
    preview = caches.preview_detection_recovery(store, job_id)
    assert preview["available"], preview
    return caches.recover_detection_cache(
        store, client.app.state.jobs, job_id, fingerprint=preview["fingerprint"]
    )


def entries(store, cache):
    return sorted(
        store.list("temporal_detection_frames", cache_id=cache["id"]),
        key=lambda row: row["payload"]["frame_index"],
    )


def test_repeated_continuation_keeps_empty_outputs_and_parent_attempts_byte_for_byte(
    client, tmp_path
):
    source = sequence(client, tmp_path)
    cache = create(client, source)
    store = client.app.state.store
    calls, old_rows, old_jobs = [], [], []
    job_id = cache["job_id"]
    for count in (1, 2, 3):
        execute(
            store,
            cache,
            job=job_id,
            calls=calls,
            stop=lambda count=count: len(entries(store, cache)) >= count,
        )
        assert entries(store, cache)[: len(old_rows)] == old_rows
        for old in old_jobs:
            assert store.get("jobs", old["id"]) == old
        old_rows = entries(store, cache)
        old_jobs.append(store.get("jobs", job_id))
        if count < 3:
            job_id = continuation(client, job_id)["id"]
    assert calls == [0, 13, 26]
    assert old_rows[1]["payload"]["detections"] == []
    assert [j["result"]["inherited_count"] for j in old_jobs] == [0, 1, 2]
    assert [j["result"]["produced_count"] for j in old_jobs] == [1, 1, 1]
    view = caches.read_detection_cache(store, cache["id"])
    assert len(view["frames"]) == 3
    assert not caches.preview_detection_recovery(store, job_id)["available"]


def test_crash_after_commit_resumes_only_missing_frames(client, tmp_path):
    cache = create(client, sequence(client, tmp_path))
    store, calls = client.app.state.store, []

    def crash_after_save(_fraction, message):
        if message.startswith("Saved"):
            raise RuntimeError("Simulated process exit after durable frame commit")

    with pytest.raises(RuntimeError, match="process exit"):
        execute(store, cache, calls=calls, progress=crash_after_save)
    before = entries(store, cache)
    next_job = continuation(client, cache["job_id"])
    execute(store, cache, job=next_job["id"], calls=calls)
    assert calls == [0, 13, 26]
    assert entries(store, cache)[:1] == before
    assert caches.get_detection_cache(store, cache["id"])["coverage"]["state"] == "complete"


def test_failure_after_final_commit_still_has_a_complete_reusable_cache(client, tmp_path):
    cache = create(client, sequence(client, tmp_path))
    store = client.app.state.store

    def crash_at_end(fraction, message):
        if fraction == 1 and message.startswith("Saved"):
            raise RuntimeError("Supervisor did not record completion")

    with pytest.raises(RuntimeError):
        execute(store, cache, progress=crash_at_end)
    assert len(caches.read_detection_cache(store, cache["id"])["frames"]) == 3
    assert not caches.preview_detection_recovery(store, cache["job_id"])["available"]
    assert len(store.list("jobs")) == 1


def test_storage_floor_and_read_filters_preserve_native_indices_and_all_classes(
    client, tmp_path, monkeypatch
):
    cache = create(client, sequence(client, tmp_path), min_score=0.1)
    store = client.app.state.store
    execute(store, cache)
    before = entries(store, cache)
    monkeypatch.setattr(caches, "prepare_detector", lambda *_a, **_k: pytest.fail("Model read"))
    full = caches.read_detection_cache(store, cache["id"])
    selected = caches.read_detection_cache(store, cache["id"], min_score=0.7, class_ids=[1])
    assert full["frames"][0]["native_detection_count"] == 3
    assert [d["detection_index"] for d in full["frames"][0]["detections"]] == [1, 2]
    assert [d["label"] for d in full["frames"][0]["detections"]] == ["car", "person"]
    assert [d["detection_index"] for d in selected["frames"][0]["detections"]] == [2]
    assert selected["result_sha256"] == full["result_sha256"]
    assert selected["frames"][1]["detections"] == []
    assert entries(store, cache) == before
    with pytest.raises(ValueError, match="floor"):
        caches.read_detection_cache(store, cache["id"], min_score=0.01)


@pytest.mark.parametrize("field", ["weight_sha256", "runtime", "preprocessing"])
def test_recipe_drift_blocks_recovery_without_touching_saved_prefix(
    client, tmp_path, monkeypatch, field
):
    cache = create(client, sequence(client, tmp_path))
    store = client.app.state.store
    execute(store, cache, stop=lambda: len(entries(store, cache)) == 1)
    before = entries(store, cache)
    prepare = caches.prepare_detector

    def changed(*args, **kwargs):
        config = deepcopy(prepare(*args, **kwargs))
        config[field] = "changed fixture"
        return config

    monkeypatch.setattr(caches, "prepare_detector", changed)
    preview = caches.preview_detection_recovery(store, cache["job_id"])
    assert not preview["available"] and "changed" in preview["reason"]
    assert entries(store, cache) == before
    assert len(store.list("jobs")) == 1


def test_hardware_thread_drift_cannot_mix_outputs_but_original_hardware_can_continue(
    client, tmp_path
):
    cache = create(client, sequence(client, tmp_path))
    store = client.app.state.store
    execute(store, cache, stop=lambda: len(entries(store, cache)) == 1)
    before = entries(store, cache)
    second = continuation(client, cache["job_id"])
    with pytest.raises(caches.DetectionCacheConflict, match="hardware"):
        execute(store, cache, job=second["id"], threads=17)
    assert entries(store, cache) == before
    third = continuation(client, second["id"])
    calls = []
    execute(store, cache, job=third["id"], calls=calls)
    assert calls == [13, 26]


def test_changed_source_blocks_recovery(client, tmp_path):
    source = sequence(client, tmp_path)
    cache = create(client, source)
    store = client.app.state.store
    execute(store, cache, stop=lambda: len(entries(store, cache)) == 1)
    before = entries(store, cache)
    frame = store.get("frames", source["manifest"]["frames"][-1]["frame_id"])
    store.artifact_path(frame["path"]).write_bytes(b"changed")
    preview = caches.preview_detection_recovery(store, cache["job_id"])
    assert not preview["available"]
    assert entries(store, cache) == before


def test_stale_recovery_fingerprint_does_not_create_a_job(client, tmp_path):
    cache = create(client, sequence(client, tmp_path))
    store = client.app.state.store
    execute(store, cache, stop=lambda: True)
    preview = caches.preview_detection_recovery(store, cache["job_id"])
    store.update("jobs", cache["job_id"], {"message": "Changed after preview"})
    with pytest.raises(caches.DetectionCacheConflict, match="preview changed"):
        caches.recover_detection_cache(
            store, client.app.state.jobs, cache["job_id"], fingerprint=preview["fingerprint"]
        )
    assert len(store.list("jobs")) == 1


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p.update(input_size=[81, 60]),
        lambda p: p["detections"][0].update(label="car"),
        lambda p: p["detections"][0].update(score=float("nan")),
        lambda p: p["detections"][0].update(box=[2, 3, 90, 40]),
        lambda p: p["timing"].update(inference_ms=-1),
    ],
)
def test_invalid_detector_output_never_publishes_even_below_storage_floor(
    client, tmp_path, mutation
):
    cache = create(client, sequence(client, tmp_path), min_score=0.1)
    store = client.app.state.store
    with pytest.raises(ValueError):
        execute(store, cache, mutate=mutation)
    assert entries(store, cache) == []
    assert caches.preview_detection_recovery(store, cache["job_id"])["available"]


def test_single_tile_is_counted_as_tiled_and_round_trips(client, tmp_path):
    cache = create(client, sequence(client, tmp_path), inference_mode="tiled", tile_size=640)
    store = client.app.state.store
    execute(store, cache)
    result = caches.read_detection_cache(store, cache["id"])
    assert len(result["frames"]) == 3
    assert all(f["work"] == {"forward_passes": 1, "tile_count": 1} for f in result["frames"])


def test_cancel_during_prediction_discards_only_inflight_frame(client, tmp_path):
    cache = create(client, sequence(client, tmp_path))
    store = client.app.state.store
    calls = []
    execute(store, cache, calls=calls, stop=lambda: len(calls) == 2)
    assert calls == [0, 13] and len(entries(store, cache)) == 1
    next_job = continuation(client, cache["job_id"])
    execute(store, cache, calls=calls, job=next_job["id"])
    assert calls == [0, 13, 13, 26]
    assert len(entries(store, cache)) == 3


def test_simultaneous_creation_cannot_duplicate_the_same_recipe(client, tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    from iris.jobs import JobManager
    from iris.training_architectures import SSDLITE

    source = sequence(client, tmp_path)
    store = client.app.state.store

    def launch(_):
        # Independent managers deliberately have different in-process guards.
        return caches.create_detection_cache(
            store, JobManager(store), source["id"], name="Concurrent recipe", model_id=SSDLITE
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(launch, range(2)))
    assert results[0]["id"] == results[1]["id"]
    assert sorted(result["reused"] for result in results) == [False, True]
    assert len(store.list("jobs")) == len(store.list("temporal_detection_caches")) == 1


def test_simultaneous_recovery_has_exactly_one_successor(client, tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    from iris.jobs import JobManager

    cache = create(client, sequence(client, tmp_path))
    store = client.app.state.store
    execute(store, cache, stop=lambda: len(entries(store, cache)) == 1)
    parent = store.get("jobs", cache["job_id"])
    preview = caches.preview_detection_recovery(store, cache["job_id"])

    def recover(_):
        try:
            return caches.recover_detection_cache(
                store,
                JobManager(store),
                cache["job_id"],
                fingerprint=preview["fingerprint"],
            )
        except caches.DetectionCacheConflict:
            return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(recover, range(2)))
    assert sum(result is not None for result in results) == 1
    assert len(store.list("jobs")) == 2
    assert store.get("jobs", parent["id"]) == parent
    assert caches.get_detection_cache(store, cache["id"])["coverage"]["completed_count"] == 1
