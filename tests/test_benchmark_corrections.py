"""Durable correction revisions and real server-clock review intervals."""

from copy import deepcopy
from datetime import UTC, datetime, timedelta

import pytest
from PIL import Image

from iris import benchmark_corrections as review
from iris.media import import_asset
from iris.store import Store, new_id, now
from iris.taxonomies import TAXONOMY


@pytest.fixture
def context(tmp_path, monkeypatch):
    store = Store(tmp_path / "workspace")
    session = store.insert(
        "sessions", dict(id=new_id(), name="Independent", scene_group="scene", created_at=now())
    )
    path = tmp_path / "image.png"
    Image.new("RGB", (100, 80)).save(path)
    asset = import_asset(store, session["id"], path, path.name)
    frame = store.list("frames", asset_id=asset["id"])[0]
    benchmark = store.insert(
        "benchmarks",
        dict(
            id=new_id(),
            project_id="default",
            name="Fixture",
            path="fixture",
            manifest_sha256="a" * 64,
            summary={},
            created_at=now(),
        ),
    )
    config = store.insert(
        "benchmark_configs",
        dict(
            id=new_id(),
            benchmark_id=benchmark["id"],
            name="Fixture",
            approach="local_detector",
            config={},
            fingerprint="b" * 64,
            created_at=now(),
        ),
    )
    job = store.insert(
        "jobs", dict(id=new_id(), kind="benchmark", status="succeeded", params={}, created_at=now())
    )
    trial = store.insert(
        "benchmark_trials",
        dict(
            id=new_id(),
            benchmark_id=benchmark["id"],
            config_id=config["id"],
            split="tuning",
            config={"frame_ids": [frame["id"]]},
            job_id=job["id"],
            created_at=now(),
        ),
    )
    proposals = [
        dict(id="p1", label="person", box=[1.0, 2.0, 20.0, 30.0], score=0.8),
        dict(id="p2", label="car", box=[40.0, 40.0, 60.0, 70.0], score=None),
    ]
    output = store.insert(
        "benchmark_outputs",
        dict(
            id=new_id(),
            trial_id=trial["id"],
            frame_id=frame["id"],
            result={"proposals": proposals},
            metadata={},
            created_at=now(),
        ),
    )
    frozen_frame = dict(frame_id=frame["id"], width=100, height=80)
    monkeypatch.setattr(
        review,
        "_context",
        lambda *_: (
            deepcopy(output),
            deepcopy(trial),
            frozen_frame,
            deepcopy(TAXONOMY),
            deepcopy(proposals),
        ),
    )
    clock = [100.0]
    monkeypatch.setattr(
        review,
        "_clock",
        lambda: (datetime(2026, 10, 4, tzinfo=UTC) + timedelta(seconds=clock[0]), clock[0]),
    )
    return store, output, trial, clock


def tick(
    context, action, revision, *, token="owner-abcdefghijkl", operation=None, reviewer="Victor"
):
    return review.timer_action(
        context[0],
        context[1]["id"],
        action=action,
        expected_revision=revision,
        token=token,
        operation_id=operation or new_id(),
        reviewer=reviewer,
    )


def save(context, **kwargs):
    return review.save_correction(
        context[0],
        context[1]["id"],
        expected_revision=kwargs.pop("expected_revision", 0),
        boxes=kwargs.pop("boxes", []),
        reviewer=kwargs.pop("reviewer", "Victor"),
        **kwargs,
    )


def test_never_started_has_missing_time_and_no_reference(context):
    doc = review.correction_document(context[0], context[1]["id"])
    assert doc["timer"]["elapsed_ms"] is None
    assert "annotation" not in doc and "reference" not in doc
    assert doc["boxes"][0]["proposal_id"] == "p1"
    saved = save(context, status="reviewed")
    assert saved["timing"]["elapsed_ms"] is None
    assert saved["timing"]["fully_timed"] is False
    assert saved["decisions"] == {"p1": "rejected", "p2": "rejected"}
    assert context[0].get("benchmark_outputs", context[1]["id"]) == context[1]


def test_acknowledged_intervals_exclude_paused_time_and_save_atomic(context):
    start = tick(context, "start", 0)
    context[3][0] += 4
    heartbeat = tick(context, "heartbeat", start["revision"])
    assert heartbeat["elapsed_ms"] == 4000
    context[3][0] += 2
    paused = tick(context, "pause", heartbeat["revision"])
    assert paused["elapsed_ms"] == 6000
    context[3][0] += 3600
    restart = tick(context, "start", paused["revision"])
    context[3][0] += 3
    saved = save(
        context,
        status="reviewed",
        timer_revision=restart["revision"],
        timer_token="owner-abcdefghijkl",
    )
    assert saved["timer"]["state"] == "paused"
    assert saved["timing"]["elapsed_ms"] == 9000
    assert saved["timing"]["recorded_segments"] == 2
    assert saved["timing"]["fully_timed"] is True


def test_expiry_retains_confirmed_time_and_restart_never_credits_downtime(context):
    tick(context, "start", 0)
    context[3][0] += 4
    tick(context, "heartbeat", 1)
    context[3][0] += 31
    timer = review.correction_document(context[0], context[1]["id"])["timer"]
    assert timer["state"] == "paused" and timer["elapsed_ms"] == 4000
    assert timer["fully_timed"] is False
    tick(context, "start", 2)
    context[3][0] += 8
    review.recover_timers(context[0])
    context[3][0] += 86400
    restored = review.correction_document(context[0], context[1]["id"])["timer"]
    assert restored["state"] == "paused" and restored["elapsed_ms"] == 4000
    assert restored["fully_timed"] is False


def test_foreign_process_cannot_keep_restored_timer_running(context, monkeypatch):
    tick(context, "start", 0)
    monkeypatch.setattr(review, "PROCESS_ID", "different-server")
    doc = review.correction_document(context[0], context[1]["id"])
    assert doc["timer"]["state"] == "paused"
    assert not doc["timer"]["fully_timed"]


def test_clock_going_back_does_not_produce_negative_duration(context):
    tick(context, "start", 0)
    context[3][0] -= 10
    timer = tick(context, "pause", 1)
    assert timer["elapsed_ms"] == 0 and not timer["fully_timed"]


def test_cas_ownership_and_idempotency(context):
    operation = new_id()
    first = tick(context, "start", 0, operation=operation)
    context[3][0] += 4
    assert tick(context, "start", 0, operation=operation) == first
    with pytest.raises(review.CorrectionConflict):
        tick(context, "heartbeat", 1, token="another-owner-abcdef")
    with pytest.raises(review.CorrectionConflict):
        tick(context, "heartbeat", 0)
    with pytest.raises(review.CorrectionConflict):
        tick(context, "pause", 1, operation=operation)
    with pytest.raises(review.CorrectionConflict):
        save(context, timer_revision=1, timer_token="another-owner-abcdef")
    assert context[0].list("benchmark_corrections") == []
    review.recover_timers(context[0])
    replay = tick(context, "start", 0, operation=operation)
    assert replay["state"] == "paused" and replay["revision"] > first["revision"]


def test_corrections_keep_immutable_history_and_derive_changes(context):
    boxes = [
        dict(id="unchanged", label="person", box=[1, 2, 20, 30], proposal_id="p1"),
        dict(id="fix", label="person", box=[40, 40, 65, 70], proposal_id="p2"),
        dict(id="new", label="car", box=[60, 1, 80, 20]),
    ]
    one = save(context, boxes=boxes, status="draft")
    assert one["decisions"] == {"p1": "accepted", "p2": "corrected"}
    assert one["timing"]["changes"] == dict(added=1, accepted=1, corrected=1, rejected=0)
    two = save(context, expected_revision=1, status="reviewed")
    assert two["revision"] == 2 and len(two["history"]) == 2
    history = context[0].list("benchmark_corrections", output_id=context[1]["id"])
    assert len(history) == 2
    assert next(row for row in history if row["revision"] == 1)["boxes"] == one["boxes"]
    with pytest.raises(review.CorrectionConflict):
        save(context, expected_revision=1)


@pytest.mark.parametrize(
    "box",
    [
        dict(id="a", label="unknown", box=[1, 2, 3, 4]),
        dict(id="a", label="person", box=[-1, 2, 3, 4]),
        dict(id="a", label="person", box=[1, 2, float("nan"), 4]),
        dict(id="a", label="person", box=[1, 2, 3, 4], proposal_id="not-a-proposal"),
        dict(id="a", label="person", box=[1, 2, 3, 4], reference_id="hidden"),
    ],
)
def test_invalid_geometry_provenance_and_classes_do_not_save(context, box):
    with pytest.raises(ValueError):
        save(context, boxes=[box])
    assert context[0].list("benchmark_corrections") == []


def test_reviewer_switch_and_unrecorded_revision_are_incomplete(context):
    tick(context, "start", 0)
    context[3][0] += 1
    tick(context, "pause", 1)
    tick(context, "start", 2, reviewer="Second reviewer")
    context[3][0] += 1
    saved = save(
        context, reviewer="Second reviewer", timer_revision=3, timer_token="owner-abcdefghijkl"
    )
    assert not saved["timing"]["fully_timed"]
    assert "different reviewer" in saved["timing"]["interruption_reason"]
    next_save = save(context, expected_revision=1)
    assert not next_save["timing"]["fully_timed"]
    assert "No new review interval" in next_save["timing"]["interruption_reason"]


def test_concurrent_saves_publish_one_revision(context):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    barrier = Barrier(2)

    def attempt():
        barrier.wait()
        try:
            return save(context, status="reviewed")["revision"]
        except review.CorrectionConflict:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: attempt(), range(2)))
    assert sorted(results, key=str) == [1, "conflict"]
    assert len(context[0].list("benchmark_corrections")) == 1


def test_partial_output_set_cannot_claim_complete_correction(context):
    store, _, trial, _ = context
    save(context, status="reviewed")
    store.update(
        "benchmark_trials",
        trial["id"],
        {"config": {"frame_ids": [trial["config"]["frame_ids"][0], "unreturned-frame"]}},
    )
    summary = review.correction_summaries(store, trial["id"])
    assert summary["reviewed_count"] == 1 and summary["planned_count"] == 2
    assert summary["complete"] is False
    assert summary["recorded_review_ms"] is None


def test_lost_receipt_reconciliation_can_discard_unconfirmed_masked_interval(context):
    tick(context, "start", 0)
    context[3][0] += 5
    tick(context, "heartbeat", 1)
    context[3][0] += 10
    timer = review.timer_action(
        context[0],
        context[1]["id"],
        action="pause",
        expected_revision=2,
        token="owner-abcdefghijkl",
        operation_id=new_id(),
        reviewer="Victor",
        discard_unconfirmed=True,
    )
    assert timer["elapsed_ms"] == 5000 and timer["state"] == "paused"
    assert timer["fully_timed"] is False
    assert "receipt" in timer["interruption_reason"]
