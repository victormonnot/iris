"""Durable batch reviews on synthetic local files and explicit reviewer fixtures."""

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import pytest
from PIL import Image

from iris import assistance_batches
from iris.annotations import AnnotationConflict, save_annotation
from iris.assistance import request_assistance, run_assistance
from iris.assistance_batches import (
    batch_detail,
    cancel_batch,
    create_batch,
    list_batches,
    preview_batch,
)
from iris.jobs import JobManager
from iris.media import import_asset
from iris.store import Store, new_id, now

READY = {
    "provider": "ollama",
    "endpoint": "http://127.0.0.1:11434",
    "model": "fixture-only:latest",
    "model_digest": "a" * 64,
    "status": "ready",
}
BOX = {"id": "manual", "label": "person", "box": [2, 3, 30, 35]}


def add_session(store):
    return store.insert(
        "sessions",
        {
            "id": new_id(),
            "name": "Synthetic batch fixture",
            "scene_group": "fixture",
            "created_at": now(),
        },
    )


def add_frame(store, session_id, number):
    path = store.root / f"synthetic-{number}.png"
    Image.new("RGB", (80, 60), (number, 30, 40)).save(path)
    asset = import_asset(store, session_id, path, path.name)
    frame = store.list("frames", asset_id=asset["id"])[0]
    return store.update("frames", frame["id"], {"selected": True})


def save(store, frame_id, **changes):
    return save_annotation(
        store, frame_id, expected_revision=1, boxes=[BOX], decisions={}, status="draft", **changes
    )


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    store = Store(tmp_path / "workspace")
    session = add_session(store)
    frames = [add_frame(store, session["id"], number) for number in (10, 20, 30)]
    for frame in frames:
        save_annotation(
            store,
            frame["id"],
            expected_revision=0,
            boxes=[BOX],
            decisions={},
            status="validated",
            reviewer="Synthetic fixture",
        )
    jobs = JobManager(store)
    monkeypatch.setattr(assistance_batches, "provider_status", lambda _: deepcopy(READY))
    monkeypatch.setattr("iris.assistance.provider_status", lambda *_: deepcopy(READY))
    return store, jobs, session, frames


def options(frames, **changes):
    return {
        "frame_ids": [frame["id"] for frame in frames],
        "source": "annotations",
        "model": "fixture-only",
        **changes,
    }


def queue(workspace, **changes):
    store, jobs, session, frames = workspace
    payload = options(frames, **changes)
    preview = preview_batch(store, session["id"], **payload)
    return create_batch(
        store,
        jobs,
        session["id"],
        name="Synthetic local review",
        expected_fingerprint=preview["fingerprint"],
        **payload,
    )


def add_comparison(store, session, frames):
    job = store.insert(
        "jobs",
        {"id": new_id(), "kind": "infer", "status": "succeeded", "params": {}, "created_at": now()},
    )
    comparison = store.insert(
        "comparisons",
        {
            "id": new_id(),
            "session_id": session["id"],
            "name": "Synthetic saved detector",
            "frame_ids": [frame["id"] for frame in frames],
            "model_ids": ["fixture-detector"],
            "config": {
                "taxonomy": "coco-2017-v1",
                "frame_hashes": {frame["id"]: frame["sha256"] for frame in frames},
            },
            "job_id": job["id"],
            "created_at": now(),
        },
    )
    run = store.insert(
        "runs",
        {
            "id": new_id(),
            "comparison_id": comparison["id"],
            "model_id": "fixture-detector",
            "metadata": {"fixture": True, "model_digest": "synthetic"},
            "created_at": now(),
        },
    )
    for frame in frames:
        store.insert(
            "predictions",
            {
                "id": new_id(),
                "comparison_id": comparison["id"],
                "run_id": run["id"],
                "frame_id": frame["id"],
                "model_id": run["model_id"],
                "detections": [
                    {"box": [2, 3, 30, 35], "label_id": 1, "score": 0.9},
                    {"box": [40, 30, 70, 50], "label_id": 3, "score": 0.4},
                ],
                "timing": {},
                "input_size": [80, 60],
                "created_at": now(),
            },
        )
    return comparison


def test_preview_is_read_only_bounded_and_probes_only_once(workspace, monkeypatch):
    store, _, session, frames = workspace
    probes = []

    def status(config):
        probes.append(config)
        return deepcopy(READY)

    monkeypatch.setattr(assistance_batches, "provider_status", status)
    before = store.list("annotation_revisions")
    result = preview_batch(store, session["id"], **options(frames))
    assert probes == [{"endpoint": READY["endpoint"], "model": READY["model"]}]
    assert result["eligible_count"] == result["candidate_count"] == 3
    assert result["excluded_count"] == 0
    assert all(row["base_revision"] == 1 for row in result["frames"])
    assert store.list("jobs") == store.list("assistance_records") == []
    assert store.list("assistance_batches") == store.list("annotation_suggestions") == []
    assert store.list("annotation_revisions") == before
    assert result == preview_batch(store, session["id"], **options(frames))


@pytest.mark.parametrize(
    "changes",
    [
        {"frame_ids": []},
        {"frame_ids": ["missing"]},
        {"frame_ids": ["a"] * 26},
        {"frame_ids": ["a", "a"]},
        {"frame_ids": [None]},
        {"source": "api"},
        {"threshold": True},
        {"threshold": -0.01},
        {"threshold": float("nan")},
        {"threshold": 1.01},
        {"instructions": "x" * 2001},
        {"model": "qwen:cloud"},
        {"model": "https://example.com/model"},
        {"model": None},
        {"comparison_id": "irrelevant"},
        {"detector_model_id": "irrelevant"},
        {"source": "comparison"},
    ],
)
def test_invalid_global_inputs_never_probe_provider(workspace, monkeypatch, changes):
    store, _, session, frames = workspace
    monkeypatch.setattr(assistance_batches, "provider_status", lambda _: pytest.fail("probe"))
    with pytest.raises(ValueError):
        preview_batch(store, session["id"], **options(frames, **changes))


def test_unknown_session_and_foreign_frame_rejected(workspace):
    store, _, session, frames = workspace
    with pytest.raises(KeyError):
        preview_batch(store, "missing", **options(frames))
    foreign = add_frame(store, add_session(store)["id"], 80)
    with pytest.raises(ValueError, match="flight session"):
        preview_batch(store, session["id"], **options([*frames, foreign]))


@pytest.mark.parametrize("status", ["unavailable", "missing_model", "unsupported_model"])
def test_provider_dependency_failure_is_not_queued(workspace, monkeypatch, status):
    store, _, session, frames = workspace
    monkeypatch.setattr(
        assistance_batches,
        "provider_status",
        lambda _: {
            **READY,
            "status": status,
            "reason": "Fixture unavailable",
        },
    )
    with pytest.raises(RuntimeError, match="Fixture unavailable"):
        preview_batch(store, session["id"], **options(frames))
    assert store.list("jobs") == []


@pytest.mark.parametrize(
    "case,reason",
    [
        ("empty", "1–8"),
        ("too_many", "1–8"),
        ("missing_bytes", "No such file"),
        ("corrupt_bytes", "cannot identify"),
        ("unselected", "selected"),
        ("active", "already queued"),
    ],
)
def test_ineligible_frames_explain_exclusions_without_losing_eligible_ones(
    workspace,
    case,
    reason,
):
    store, jobs, session, frames = workspace
    first = frames[0]
    if case in {"empty", "too_many"}:
        boxes = [] if case == "empty" else [{**BOX, "id": str(i)} for i in range(9)]
        save_annotation(
            store, first["id"], expected_revision=1, boxes=boxes, decisions={}, status="draft"
        )
    elif case == "missing_bytes":
        store.artifact_path(first["path"]).unlink()
    elif case == "corrupt_bytes":
        store.artifact_path(first["path"]).write_bytes(b"corrupt synthetic fixture")
    elif case == "unselected":
        store.update("frames", first["id"], {"selected": False})
    else:
        request_assistance(store, jobs, first["id"], expected_revision=1)
    result = preview_batch(store, session["id"], **options(frames))
    assert result["eligible_count"] == 2 and result["excluded_count"] == 1
    assert reason in result["frames"][0]["reason"]
    created = queue(workspace)
    assert created["frame_ids"] == [frame["id"] for frame in frames[1:]]
    assert created["config"]["excluded"] == [result["frames"][0]]
    assert created["config"]["requested_frame_ids"] == [frame["id"] for frame in frames]


def test_atomic_queue_preserves_order_candidates_and_human_revisions(workspace):
    store, _, _, frames = workspace
    before = store.list("annotation_revisions"), store.list("frames")
    batch = queue(workspace, instructions="  Synthetic context  ")
    assert batch["status"] == "queued" and batch["counts"]["queued"] == 3
    assert batch["progress"] == batch["finished_count"] == batch["suggestions_created"] == 0
    assert batch["frame_ids"] == [frame["id"] for frame in frames]
    assert batch["job_ids"] == [job["id"] for job in store.list("jobs")]
    for frame, job in zip(frames, store.list("jobs"), strict=True):
        assert job["params"]["batch_id"] == batch["id"]
        record = store.get("assistance_records", job["params"]["assistance_id"])
        assert record["frame_id"] == frame["id"]
        assert record["candidates"][0]["source"]["target_box_id"] == BOX["id"]
        assert record["config"]["base_revision"] == 1
        assert record["config"]["frame_sha256"] == frame["sha256"]
        assert record["config"]["model_digest"] == READY["model_digest"]
        assert record["config"]["instructions"] == "Synthetic context"
        assert record["config"]["provider"]["provider"] == "ollama"
        assert "consent" not in record["config"]
    assert (store.list("annotation_revisions"), store.list("frames")) == before


def test_comparison_source_freezes_detector_candidates_and_threshold(workspace):
    store, _, session, frames = workspace
    comparison = add_comparison(store, session, frames[:2])
    payload = {
        "source": "comparison",
        "comparison_id": comparison["id"],
        "detector_model_id": "fixture-detector",
        "threshold": 0.5,
    }
    preview = preview_batch(store, session["id"], **options(frames, **payload))
    assert preview["candidate_count"] == 2 and preview["excluded_count"] == 1
    assert "No saved prediction" in preview["frames"][2]["reason"]
    batch = queue(workspace, **payload)
    for row in batch["frames"]:
        record = store.get("assistance_records", row["assistance_id"])
        candidate = record["candidates"][0]
        assert candidate["source"]["kind"] == "prediction"
        assert candidate["source"]["model_id"] == "fixture-detector"
        assert candidate["source"]["run_metadata"]["fixture"] is True
        assert candidate["source"]["score"] == 0.9
        assert record["config"]["prediction_id"] == row["prediction_id"]


@pytest.mark.parametrize("case", ["foreign", "unfinished", "detector", "taxonomy"])
def test_comparison_requires_a_completed_compatible_source(workspace, case):
    store, _, session, frames = workspace
    comparison = add_comparison(store, session, frames)
    detector = "fixture-detector"
    if case == "foreign":
        store.update("comparisons", comparison["id"], {"session_id": add_session(store)["id"]})
    elif case == "unfinished":
        store.update("jobs", comparison["job_id"], {"status": "running"})
    elif case == "detector":
        detector = "another-detector"
    else:
        store.update("comparisons", comparison["id"], {"config": {"taxonomy": "unknown"}})
    with pytest.raises(ValueError):
        preview_batch(
            store,
            session["id"],
            **options(
                frames,
                source="comparison",
                comparison_id=comparison["id"],
                detector_model_id=detector,
            ),
        )


@pytest.mark.parametrize(
    "case",
    [
        "null_detection",
        "string_detections",
        "missing_hash",
        "run_metadata",
        "invalid_score",
    ],
)
def test_corrupt_saved_predictions_are_excluded_without_hiding_good_frames(workspace, case):
    store, _, session, frames = workspace
    comparison = add_comparison(store, session, frames)
    prediction = store.list("predictions", frame_id=frames[0]["id"])[0]
    if case == "null_detection":
        store.update("predictions", prediction["id"], {"detections": [None]})
    elif case == "string_detections":
        store.update("predictions", prediction["id"], {"detections": "invalid"})
    elif case == "missing_hash":
        config = deepcopy(comparison["config"])
        del config["frame_hashes"][frames[0]["id"]]
        store.update("comparisons", comparison["id"], {"config": config})
    elif case == "run_metadata":
        store.update("runs", prediction["run_id"], {"metadata": []})
    else:
        store.update(
            "predictions",
            prediction["id"],
            {
                "detections": [
                    {"box": [2, 3, 30, 35], "label_id": 1, "score": "not a score"},
                ]
            },
        )
    result = preview_batch(
        store,
        session["id"],
        **options(
            frames,
            source="comparison",
            comparison_id=comparison["id"],
            detector_model_id="fixture-detector",
        ),
    )
    assert result["frames"][0]["eligible"] is False
    assert result["frames"][0]["reason"]
    assert result["eligible_count"] == (0 if case == "run_metadata" else 2)


def test_failed_second_child_insert_rolls_back_entire_batch(workspace):
    store, _, _, frames = workspace
    with store.connect() as conn:
        conn.execute(
            "CREATE TRIGGER reject_child BEFORE INSERT ON assistance_records "
            f"WHEN NEW.frame_id='{frames[1]['id']}' "
            "BEGIN SELECT RAISE(ABORT,'synthetic insertion failure'); END"
        )
    with pytest.raises(sqlite3.IntegrityError, match="synthetic insertion failure"):
        queue(workspace)
    assert store.list("jobs") == store.list("assistance_records") == []
    assert store.list("assistance_batches") == []


@pytest.mark.parametrize("case", ["revision", "selection", "frame_hash", "bytes", "active"])
def test_changed_inputs_after_preview_reject_stale_fingerprint(workspace, case):
    store, jobs, session, frames = workspace
    payload = options(frames)
    preview = preview_batch(store, session["id"], **payload)
    _change_input(store, jobs, frames[0], case)
    with pytest.raises(AnnotationConflict, match="preview"):
        create_batch(
            store,
            jobs,
            session["id"],
            name="stale",
            **payload,
            expected_fingerprint=preview["fingerprint"],
        )
    assert store.list("assistance_batches") == []


def _change_input(store, jobs, frame, case):
    if case == "revision":
        save(store, frame["id"], notes="late human revision")
    elif case == "selection":
        store.update("frames", frame["id"], {"selected": False})
    elif case == "frame_hash":
        store.update("frames", frame["id"], {"sha256": "b" * 64})
    elif case == "bytes":
        Image.new("RGB", (80, 60), "white").save(store.artifact_path(frame["path"]))
    else:
        request_assistance(store, jobs, frame["id"], expected_revision=1)


@pytest.mark.parametrize("case", ["revision", "selection", "frame_hash", "bytes", "active"])
def test_transaction_rechecks_all_inputs_after_preparation(workspace, monkeypatch, case):
    store, jobs, session, frames = workspace
    payload = options(frames)
    preview = preview_batch(store, session["id"], **payload)
    prepare = assistance_batches._prepare_batch

    def racing_prepare(*args, **kwargs):
        result = prepare(*args, **kwargs)
        _change_input(store, jobs, frames[1], case)
        return result

    monkeypatch.setattr(assistance_batches, "_prepare_batch", racing_prepare)
    with pytest.raises(AnnotationConflict, match="preview"):
        create_batch(
            store,
            jobs,
            session["id"],
            name="race",
            **payload,
            expected_fingerprint=preview["fingerprint"],
        )
    assert store.list("assistance_batches") == []
    assert len(store.list("jobs")) == (1 if case == "active" else 0)


def test_provider_digest_changes_invalidate_preview(workspace, monkeypatch):
    store, jobs, session, frames = workspace
    payload = options(frames)
    preview = preview_batch(store, session["id"], **payload)
    monkeypatch.setattr(
        assistance_batches,
        "provider_status",
        lambda _: {
            **READY,
            "model_digest": "b" * 64,
        },
    )
    with pytest.raises(AnnotationConflict):
        create_batch(
            store,
            jobs,
            session["id"],
            name="changed model",
            **payload,
            expected_fingerprint=preview["fingerprint"],
        )


@pytest.mark.parametrize("case", ["prediction", "run", "comparison", "missing_prediction"])
def test_transaction_rechecks_comparison_provenance(workspace, monkeypatch, case):
    store, jobs, session, frames = workspace
    comparison = add_comparison(store, session, frames[:2])
    payload = options(
        frames,
        source="comparison",
        comparison_id=comparison["id"],
        detector_model_id="fixture-detector",
    )
    preview = preview_batch(store, session["id"], **payload)
    prepare = assistance_batches._prepare_batch

    def racing_prepare(*args, **kwargs):
        result = prepare(*args, **kwargs)
        prediction = store.list("predictions", frame_id=frames[0]["id"])[0]
        if case == "prediction":
            store.update("predictions", prediction["id"], {"detections": []})
        elif case == "run":
            store.update("runs", prediction["run_id"], {"metadata": {"changed": True}})
        elif case == "comparison":
            store.update("jobs", comparison["job_id"], {"status": "failed"})
        else:
            store.insert("predictions", {**prediction, "id": new_id(), "frame_id": frames[2]["id"]})
        return result

    monkeypatch.setattr(assistance_batches, "_prepare_batch", racing_prepare)
    with pytest.raises(AnnotationConflict, match="prediction changed"):
        create_batch(
            store,
            jobs,
            session["id"],
            name="race",
            **payload,
            expected_fingerprint=preview["fingerprint"],
        )
    assert store.list("assistance_batches") == store.list("assistance_records") == []


def test_batch_children_overlap_with_single_frame_requests(workspace):
    store, jobs, session, frames = workspace
    batch = queue(workspace)
    with pytest.raises(RuntimeError, match="already queued"):
        request_assistance(store, jobs, frames[0]["id"], expected_revision=1)
    preview = preview_batch(store, session["id"], **options(frames))
    assert preview["eligible_count"] == 0
    with pytest.raises(ValueError, match="No frames"):
        create_batch(
            store,
            jobs,
            session["id"],
            name="empty",
            **options(frames),
            expected_fingerprint=preview["fingerprint"],
        )
    assert len(store.list("jobs")) == len(batch["job_ids"])


def test_duplicate_submission_does_not_queue_again(workspace):
    store, jobs, session, frames = workspace
    preview = preview_batch(store, session["id"], **options(frames))
    for attempt in range(2):
        if attempt == 0:
            create_batch(
                store,
                jobs,
                session["id"],
                name="first",
                **options(frames),
                expected_fingerprint=preview["fingerprint"],
            )
        else:
            with pytest.raises(AnnotationConflict):
                create_batch(
                    store,
                    jobs,
                    session["id"],
                    name="second",
                    **options(frames),
                    expected_fingerprint=preview["fingerprint"],
                )
    assert len(store.list("assistance_batches")) == 1 and len(store.list("jobs")) == 3


def test_concurrent_process_managers_cannot_queue_overlapping_batches(workspace, monkeypatch):
    store, _, session, frames = workspace
    preview = preview_batch(store, session["id"], **options(frames))
    prepared = assistance_batches._prepare_batch
    rendezvous = threading.Barrier(2)

    def simultaneous_prepare(*args, **kwargs):
        result = prepared(*args, **kwargs)
        rendezvous.wait(timeout=10)
        return result

    monkeypatch.setattr(assistance_batches, "_prepare_batch", simultaneous_prepare)

    def attempt(index):
        try:
            return create_batch(
                store,
                JobManager(store),
                session["id"],
                name=f"Concurrent fixture {index}",
                expected_fingerprint=preview["fingerprint"],
                **options(frames),
            )["id"]
        except AnnotationConflict:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(attempt, range(2)))
    assert sum(result is not None for result in results) == 1
    assert len(store.list("assistance_batches")) == 1 and len(store.list("jobs")) == 3


def test_individual_cancellation_keeps_batch_cancellation_available(workspace):
    store, jobs, _, _ = workspace
    batch = queue(workspace)
    jobs.cancel(batch["job_ids"][0])
    partial = batch_detail(store, batch["id"])
    assert partial["counts"]["queued"] == 2
    assert partial["cancel_requested"] is False
    cancelled = cancel_batch(store, jobs, batch["id"])
    assert cancelled["status"] == "cancelled"
    assert cancelled["counts"]["cancelled"] == 3 and cancelled["cancel_requested"] is True


def test_cancel_batch_is_durable_idempotent_and_preserves_terminal_children(workspace):
    store, jobs, _, _ = workspace
    batch = queue(workspace)
    succeeded, running, queued = batch["job_ids"]
    store.update("jobs", succeeded, {"status": "succeeded", "progress": 1, "finished_at": now()})
    store.update("jobs", running, {"status": "running", "progress": 0.2, "started_at": now()})
    before = store.get("jobs", succeeded)
    cancelled = cancel_batch(store, jobs, batch["id"])
    assert cancelled["status"] == "running" and cancelled["cancel_requested"] is True
    assert cancelled["counts"]["cancelled"] == 1 and cancelled["finished_count"] == 2
    assert store.get("jobs", queued)["finished_at"] is not None
    assert store.get("jobs", running)["status"] == "running"
    assert store.get("jobs", running)["cancel_requested"] is True
    assert store.get("jobs", succeeded) == before
    queued_before = store.get("jobs", queued)
    cancel_batch(store, jobs, batch["id"])
    assert store.get("jobs", queued) == queued_before
    assert batch_detail(Store(store.root), batch["id"]) == cancelled


class FixtureReviewer:
    def __init__(self, config):
        self.metadata = {**READY, "fixture": True}

    def review(self, image, candidates, instructions=""):
        assert image.size == (80, 60)
        return {
            "reviews": [
                {"candidate_id": row["id"], "label": "car", "reason": "Fixture only"}
                for row in candidates
            ],
            "scene_notes": "Synthetic batch fixture",
            "prompt": "Synthetic fixture prompt",
            "metadata": self.metadata,
            "raw_response": {"fixture": True},
        }


def test_cancellation_during_worker_response_prevents_publishing_proposals(workspace):
    store, jobs, _, _ = workspace
    batch = queue(workspace)
    first = batch["frames"][0]
    store.update("jobs", first["job_id"], {"status": "running"})

    class CancellingReviewer(FixtureReviewer):
        def review(self, *args, **kwargs):
            result = super().review(*args, **kwargs)
            cancel_batch(store, jobs, batch["id"])
            return result

    result = run_assistance(
        store,
        first["assistance_id"],
        lambda *_: None,
        lambda: False,
        reviewer_factory=CancellingReviewer,
    )
    assert result["cancelled"] is True and result["suggestions_created"] == 0
    assert store.list("annotation_suggestions") == []
    assert store.get("assistance_records", first["assistance_id"])["raw_response"]["fixture"]
    assert batch_detail(store, batch["id"])["cancel_requested"] is True


def test_existing_worker_publishes_reviewable_proposals_and_partial_failures(workspace):
    store, jobs, _, _ = workspace
    before = store.list("annotation_revisions")
    batch = queue(workspace)
    first = batch["frames"][0]
    store.update("jobs", first["job_id"], {"status": "running"})
    result = run_assistance(
        store,
        first["assistance_id"],
        lambda *_: None,
        lambda: False,
        reviewer_factory=FixtureReviewer,
    )
    store.update("jobs", first["job_id"], {"status": "succeeded", "result": result, "progress": 1})
    store.update("jobs", batch["job_ids"][1], {"status": "failed", "error": "Fixture failure"})
    jobs._interrupt_unfinished()
    detail = batch_detail(Store(store.root), batch["id"])
    assert detail["status"] == "partial" and detail["finished_count"] == 3
    assert detail["counts"]["succeeded"] == detail["counts"]["failed"] == 1
    assert detail["counts"]["interrupted"] == 1
    assert detail["progress"] == pytest.approx(1 / 3)
    assert detail["suggestions_created"] == 1
    assert detail["frames"][1]["error"] == "Fixture failure"
    assert store.list("annotation_revisions") == before
    suggestion = store.list("annotation_suggestions")[0]
    assert suggestion["metadata"]["base_revision"] == 1
    assert suggestion["metadata"]["target_box_id"] == BOX["id"]
    cancel_batch(store, jobs, batch["id"])
    assert store.list("annotation_suggestions") == [suggestion]
    assert batch_detail(store, batch["id"]) == detail


@pytest.mark.parametrize("terminal", ["succeeded", "failed", "cancelled", "interrupted"])
def test_uniform_terminal_status_is_not_conflated_with_success(workspace, terminal):
    store, _, _, _ = workspace
    batch = queue(workspace)
    for job_id in batch["job_ids"]:
        store.update("jobs", job_id, {"status": terminal})
    detail = batch_detail(store, batch["id"])
    assert detail["status"] == terminal and detail["finished_count"] == 3
    assert detail["counts"][terminal] == 3
    assert detail["progress"] == (1 if terminal == "succeeded" else 0)


def test_list_is_session_scoped_and_restart_does_not_retry(workspace):
    store, jobs, session, _ = workspace
    batch = queue(workspace)
    assert list_batches(store, session["id"]) == [batch]
    assert list_batches(store, add_session(store)["id"]) == []
    with pytest.raises(KeyError):
        list_batches(store, "missing")
    with pytest.raises(KeyError):
        batch_detail(store, "missing")
    with pytest.raises(KeyError):
        cancel_batch(store, jobs, "missing")
    jobs._interrupt_unfinished()
    reopened = Store(store.root)
    assert list_batches(reopened, session["id"])[0]["status"] == "interrupted"
    assert reopened.list("jobs", status="queued") == []


def test_schema_seven_migration_preserves_existing_data(workspace):
    store, _, _, _ = workspace
    before = {
        table: store.list(table)
        for table in (
            "sessions",
            "assets",
            "frames",
            "annotation_revisions",
        )
    }
    with store.connect() as conn:
        conn.execute("DROP TABLE assistance_batches")
        conn.execute("PRAGMA user_version=7")
    reopened = Store(store.root)
    with reopened.connect() as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 10
    assert reopened.list("assistance_batches") == []
    assert {table: reopened.list(table) for table in before} == before
