"""Recovery uses synthetic local videos, durable positions and no model/provider calls."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Barrier

import cv2
import pytest
from test_media import make_video

from iris import job_recovery
from iris.job_recovery import (
    RecoveryConflict,
    preview_job_recovery,
    recover_job,
    run_durable_extraction,
)
from iris.jobs import JobManager, update_running
from iris.media import extract_frames, import_asset
from iris.projects import create_project
from iris.store import DEFAULT_PROJECT_ID, Store, new_id, now
from iris.taxonomies import TAXONOMY, publish_taxonomy


@pytest.fixture
def workspace(tmp_path):
    store = Store(tmp_path / "workspace")
    session = store.insert(
        "sessions",
        {
            "id": new_id(),
            "name": "Durable fixture",
            "scene_group": "recovery",
            "created_at": now(),
        },
    )
    source = make_video(tmp_path / "source.avi")
    asset = import_asset(store, session["id"], source, source.name)
    return store, JobManager(store), asset


def submit(workspace, **config):
    _, manager, asset = workspace
    return manager.submit(
        asset["id"],
        {
            "sampling_mode": "uniform",
            "max_frames": 8,
            **config,
        },
    )


def partial(workspace, *, count=2, status="interrupted", **config):
    store, _, _ = workspace
    job = submit(workspace, **config)
    store.update("jobs", job["id"], {"status": "running", "started_at": now()})
    recorded = 0

    def progress(value, _message):
        nonlocal recorded
        recorded = round(value * job["params"]["extraction_contract"]["plan"]["planned_count"])

    result = run_durable_extraction(store, job["id"], progress, lambda: recorded >= count)
    store.update("jobs", job["id"], {"status": status, "finished_at": now(), "result": result})
    return store.get("jobs", job["id"])


def continue_job(workspace, parent):
    store, manager, _ = workspace
    preview = preview_job_recovery(store, parent["id"])
    assert preview["available"], preview
    return recover_job(store, manager, parent["id"], fingerprint=preview["fingerprint"])


def test_continuation_decodes_only_remaining_positions_and_preserves_parent(workspace, monkeypatch):
    store, _, asset = workspace
    parent = partial(workspace)
    original_frames = deepcopy(store.list("frames", asset_id=asset["id"]))
    original_files = {
        frame["id"]: store.artifact_path(frame["path"]).read_bytes() for frame in original_frames
    }
    preview = preview_job_recovery(store, parent["id"])
    assert (preview["completed_count"], preview["remaining_count"], preview["total_count"]) == (
        2,
        6,
        8,
    )
    assert preview["mode"] == "continue_extraction"
    child = continue_job(workspace, parent)
    assert child["status"] == "queued" and child["params"]["recovery_of"] == parent["id"]
    assert child["params"]["extraction_contract"] == parent["params"]["extraction_contract"]
    assert child["result"]["inherited_completed_count"] == 2
    assert child["progress"] == 0.25
    real_capture = cv2.VideoCapture
    seeks = []

    class RecordingCapture:
        def __init__(self, *args):
            self.capture = real_capture(*args)

        def __getattr__(self, name):
            return getattr(self.capture, name)

        def set(self, prop, value):
            if prop == cv2.CAP_PROP_POS_FRAMES:
                seeks.append(value)
            return self.capture.set(prop, value)

    monkeypatch.setattr(cv2, "VideoCapture", RecordingCapture)
    store.update("jobs", child["id"], {"status": "running"})
    result = run_durable_extraction(store, child["id"], lambda *_: None, lambda: False)
    store.update(
        "jobs", child["id"], {"status": "succeeded", "result": result, "finished_at": now()}
    )
    assert seeks == [2, 3, 4, 5, 6, 7]
    assert result["completed_count"] == result["created"] == 8 and result["remaining_count"] == 0
    assert len(store.list("frames", asset_id=asset["id"])) == 8
    assert store.get("jobs", parent["id"]) == parent
    for frame in original_frames:
        assert store.get("frames", frame["id"]) == frame
        assert store.artifact_path(frame["path"]).read_bytes() == original_files[frame["id"]]
    tail = [frame for frame in store.list("frames") if frame["id"] not in original_files]
    assert {frame["extraction"]["job_id"] for frame in tail} == {child["id"]}
    assert {frame["extraction"]["operation_id"] for frame in tail} == {parent["id"]}
    assert all("sampling_plan" in frame["extraction"] for frame in tail)
    after = preview_job_recovery(store, parent["id"])
    assert not after["available"] and after["successor_job_id"] == child["id"]


def test_frame_committed_before_checkpoint_is_recognized_without_rewriting_parent(
    workspace, monkeypatch
):
    store, _, asset = workspace
    parent = submit(workspace)
    store.update("jobs", parent["id"], {"status": "running"})
    persist = job_recovery._persist_checkpoint

    def crash_after_image(store, job_id, result):
        if result["completed_count"] == 1:
            raise SystemExit("synthetic hard stop after frame commit")
        persist(store, job_id, result)

    monkeypatch.setattr(job_recovery, "_persist_checkpoint", crash_after_image)
    with pytest.raises(SystemExit):
        run_durable_extraction(store, parent["id"], lambda *_: None, lambda: False)
    store.update("jobs", parent["id"], {"status": "interrupted", "finished_at": now()})
    parent = store.get("jobs", parent["id"])
    assert parent["result"]["completed_count"] == 0
    (first,) = store.list("frames", asset_id=asset["id"])
    preview = preview_job_recovery(store, parent["id"])
    assert preview["available"] and preview["completed_count"] == 1
    child = continue_job(workspace, parent)
    monkeypatch.setattr(job_recovery, "_persist_checkpoint", persist)
    result = run_durable_extraction(store, child["id"], lambda *_: None, lambda: False)
    assert result["created"] == 8 and result["inherited_completed_count"] == 1
    assert len(store.list("frames", asset_id=asset["id"])) == 8
    assert result["frame_ids"][0] == first["id"]
    assert store.get("jobs", parent["id"]) == parent


@pytest.mark.parametrize("dedup,expected_state", [(0, "skipped_similar"), (None, "skipped_exact")])
def test_skipped_positions_are_checkpointed_and_counted_as_completed(
    workspace, tmp_path, dedup, expected_state
):
    store, manager, asset = workspace
    if dedup is None:
        source = make_video(tmp_path / "duplicates.avi", colors=[(10, 20, 30)] * 8)
        asset = import_asset(store, asset["session_id"], source, source.name)
        workspace = (store, manager, asset)
    parent = partial(workspace, count=3, dedup_hamming=dedup)
    assert parent["result"]["created"] == 1
    assert parent["result"][expected_state] == 2
    assert preview_job_recovery(store, parent["id"])["completed_count"] == 3
    child = continue_job(workspace, parent)
    result = run_durable_extraction(store, child["id"], lambda *_: None, lambda: False)
    assert result["created"] == 1 and result[expected_state] == 7
    assert result["completed_count"] == 8 and len(store.list("frames")) == 1


def test_existing_images_remain_valid_baseline_and_are_not_new_results(workspace):
    store, _, asset = workspace
    extract_frames(
        store,
        asset["id"],
        {"interval_seconds": 0.25, "max_frames": 1},
        lambda *_: None,
        lambda: False,
    )
    parent = partial(workspace)
    assert parent["result"]["skipped_existing"] == parent["result"]["created"] == 1
    child = continue_job(workspace, parent)
    result = run_durable_extraction(store, child["id"], lambda *_: None, lambda: False)
    assert result["created"] == 7 and result["skipped_existing"] == 1
    assert len(store.list("frames")) == 8


def test_taxonomy_is_pinned_when_queued_and_retained_after_publication(workspace):
    store, _, _ = workspace
    parent = partial(workspace)
    custom = publish_taxonomy(
        store,
        DEFAULT_PROJECT_ID,
        expected_taxonomy_id=TAXONOMY["id"],
        classes=[
            {
                "id": "helmet",
                "name": "Helmet",
                "definition": "A protective helmet.",
            }
        ],
    )
    child = continue_job(workspace, parent)
    assert preview_job_recovery(store, parent["id"])["successor_job_id"] == child["id"]
    result = run_durable_extraction(store, child["id"], lambda *_: None, lambda: False)
    assert result["created"] == 8
    assert {frame["taxonomy_id"] for frame in store.list("frames")} == {TAXONOMY["id"]}
    assert store.get("projects", DEFAULT_PROJECT_ID)["taxonomy_id"] == custom["id"]


@pytest.mark.parametrize(
    "mutation",
    [
        "source",
        "missing_source",
        "plan",
        "frame_pixels",
        "missing_frame",
        "inventory",
        "checkpoint",
    ],
)
def test_changed_inputs_refuse_recovery_without_creating_attempt(workspace, mutation):
    store, manager, asset = workspace
    parent = partial(workspace)
    preview = preview_job_recovery(store, parent["id"])
    frame = store.list("frames")[0]
    if mutation == "source":
        with store.artifact_path(asset["path"]).open("ab") as handle:
            handle.write(b"changed source")
    elif mutation == "missing_source":
        store.artifact_path(asset["path"]).unlink()
    elif mutation == "plan":
        store.update("assets", asset["id"], {"metadata": {**asset["metadata"], "fps": 2}})
    elif mutation == "frame_pixels":
        from PIL import Image

        Image.new("RGB", (frame["width"], frame["height"]), "white").save(
            store.artifact_path(frame["path"])
        )
    elif mutation == "missing_frame":
        store.artifact_path(frame["path"]).unlink()
    elif mutation == "inventory":
        extract_frames(
            store,
            asset["id"],
            {"interval_seconds": 0.25, "start_seconds": 1.0, "max_frames": 1},
            lambda *_: None,
            lambda: False,
        )
    else:
        corrupt = deepcopy(parent["result"])
        corrupt["checkpoint"]["units"][0]["frame_index"] = 4
        store.update("jobs", parent["id"], {"result": corrupt})
    assert not preview_job_recovery(store, parent["id"])["available"]
    with pytest.raises(RecoveryConflict):
        recover_job(store, manager, parent["id"], fingerprint=preview["fingerprint"])
    assert len(store.list("jobs")) == 1


def test_stale_fingerprint_and_cross_project_are_rejected(workspace):
    store, manager, _ = workspace
    parent = partial(workspace)
    with pytest.raises(RecoveryConflict, match="changed since"):
        recover_job(store, manager, parent["id"], fingerprint="0" * 64)
    other = create_project(store, name="Another project")
    with pytest.raises(KeyError):
        preview_job_recovery(store, parent["id"], project_id=other["id"])
    with pytest.raises(KeyError):
        recover_job(store, manager, parent["id"], fingerprint="0" * 64, project_id=other["id"])
    assert len(store.list("jobs")) == 1


def test_concurrent_double_click_creates_one_child_across_managers(workspace):
    store, _, _ = workspace
    parent = partial(workspace)
    preview = preview_job_recovery(store, parent["id"])
    barrier = Barrier(2)

    def submit_once():
        barrier.wait(timeout=5)
        try:
            return recover_job(
                store, JobManager(store), parent["id"], fingerprint=preview["fingerprint"]
            )
        except RecoveryConflict:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: submit_once(), range(2)))
    assert sum(result is not None for result in results) == 1
    assert len(store.list("jobs")) == 2
    assert store.get("jobs", parent["id"]) == parent


def test_cancelled_successor_can_continue_as_another_immutable_attempt(workspace):
    store, manager, _ = workspace
    parent = partial(workspace)
    child = continue_job(workspace, parent)
    cancelled = manager.cancel(child["id"])
    grandchild = continue_job(workspace, cancelled)
    assert grandchild["params"]["recovery_of"] == child["id"]
    assert grandchild["params"]["extraction_contract"]["operation_id"] == parent["id"]
    result = run_durable_extraction(store, grandchild["id"], lambda *_: None, lambda: False)
    assert result["created"] == 8 and result["inherited_completed_count"] == 2
    assert store.get("jobs", parent["id"]) == parent
    assert store.get("jobs", child["id"]) == cancelled


def test_restart_marks_partial_attempt_interrupted_without_resubmitting(workspace):
    store, manager, _ = workspace
    parent = partial(workspace, status="running")
    manager.start()
    try:
        current = store.get("jobs", parent["id"])
        assert current["status"] == "interrupted"
        assert current["result"] == parent["result"]
        assert preview_job_recovery(store, parent["id"])["available"]
        assert len(store.list("jobs")) == 1
    finally:
        manager.close()


def test_legacy_and_non_extraction_attempts_never_offer_automatic_resume(workspace):
    store, _, asset = workspace
    legacy = store.insert(
        "jobs",
        {
            "id": new_id(),
            "kind": "extract",
            "status": "interrupted",
            "params": {"asset_id": asset["id"], "config": {"max_frames": 8}},
            "created_at": now(),
        },
    )
    preview = preview_job_recovery(store, legacy["id"])
    assert not preview["available"] and "historical extraction" in preview["reason"]
    # Frame ownership is recoverable for an old inference job without loading models.
    inference = store.insert(
        "jobs",
        {
            "id": new_id(),
            "kind": "infer",
            "status": "failed",
            "params": {"session_id": asset["session_id"]},
            "created_at": now(),
        },
    )
    preview = preview_job_recovery(store, inference["id"])
    assert not preview["available"] and "never resent automatically" in preview["reason"]
    assert len(store.list("jobs")) == 2


def test_late_worker_cannot_overwrite_terminal_attempt(workspace):
    store, _, _ = workspace
    parent = partial(workspace)
    assert not update_running(
        store, parent["id"], {"status": "succeeded", "result": {"late": True}}
    )
    assert store.get("jobs", parent["id"]) == parent
    with pytest.raises(ValueError, match="terminal extraction is immutable"):
        run_durable_extraction(store, parent["id"], lambda *_: None, lambda: False)


def test_fully_completed_interrupted_attempt_needs_no_continuation(workspace):
    store, _, _ = workspace
    parent = partial(workspace, count=8)
    preview = preview_job_recovery(store, parent["id"])
    assert not preview["available"] and preview["completed_count"] == 8
    assert preview["remaining_count"] == 0 and "Every planned position" in preview["reason"]
