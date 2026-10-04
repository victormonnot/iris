"""Comparison evidence comes from offline production paths, not model calls."""

import json
from copy import deepcopy
from datetime import UTC, datetime, timedelta

import pytest
from test_benchmark import config, freeze
from test_benchmark import workspace as workspace
from test_benchmark_combined import configured, launch
from test_benchmark_combined import providers as providers
from test_benchmark_combined import run as run_combined
from test_benchmark_runs import Detector, queued, run
from test_benchmark_segmentation import runtime as runtime
from test_multimodal_provider import transport as transport

from iris.benchmark_analysis import (
    build_comparison,
    validate_comparison_snapshot,
    validate_snapshot_sources,
)
from iris.benchmark_corrections import save_correction
from iris.benchmark_runs import create_benchmark_trial, preview_benchmark_trial
from iris.store import new_id, now


def finish(store, trial, status="succeeded"):
    store.update("jobs", trial["job_id"], {"status": status, "finished_at": now()})


def compare(workspace, reference):
    return build_comparison(workspace[0], reference["id"], role="tuning")


def repeat(workspace, reference, candidate):
    preview = preview_benchmark_trial(
        workspace[0], reference["id"], config_id=candidate["id"], role="tuning"
    )
    trial = create_benchmark_trial(
        workspace[0],
        workspace[1],
        reference["id"],
        config_id=candidate["id"],
        role="tuning",
        expected_fingerprint=preview["fingerprint"],
    )
    run(workspace[0], trial)
    finish(workspace[0], trial)
    return trial


def test_empty_comparison_is_deterministic_and_offline(workspace, monkeypatch):
    reference = freeze(workspace)
    config(workspace, reference)
    from iris import benchmark, sam_provider

    def forbidden(*args, **kwargs):
        raise AssertionError("No provider probe or runtime in a report")

    monkeypatch.setattr(benchmark, "benchmark_approaches", forbidden)
    monkeypatch.setattr(sam_provider, "provider_status", forbidden)
    before = workspace[0].list("jobs")
    report = compare(workspace, reference)
    assert report == compare(workspace, reference)
    assert report["coverage"]["missing_approaches"] == ["multimodal", "segmentation", "combined"]
    assert report["reference"]["image_count"] == report["reference"]["object_count"] == 2
    assert report["configs"][0]["repeatability"]["measured"] is False
    assert report["configs"][0]["repeatability"]["identical_geometry"] is None
    assert workspace[0].list("jobs") == before
    assert validate_comparison_snapshot(report) == report


def test_quality_recomputed_and_requires_terminal_success(workspace):
    reference, _, _, trial = queued(workspace)
    run(workspace[0], trial)
    active = compare(workspace, reference)["configs"][0]["trials"][0]
    assert active["coverage"] == {"planned": 2, "ready": 2, "failed": 0, "missing": 0}
    assert active["quality"]["complete"] is False
    assert active["frames"][0]["quality"]["tp"] == 1
    job = workspace[0].get("jobs", trial["job_id"])
    job["result"]["quality"] = {"invented": True}
    workspace[0].update("jobs", job["id"], {"result": job["result"]})
    finish(workspace[0], trial)
    report = compare(workspace, reference)
    completed = report["configs"][0]["trials"][0]
    assert completed["quality"]["metrics"]["summary"]["tp"] == 2
    assert completed["cost"]["usage_cost_usd"] is None
    assert completed["cost"]["external"] is False
    assert completed["frames"][0]["raw_response_sha256"]
    assert validate_comparison_snapshot(report) == report
    with workspace[0].connect() as conn:
        assert validate_snapshot_sources(conn, report, store=workspace[0]) == report
    finish(workspace[0], trial, "failed")
    assert compare(workspace, reference)["configs"][0]["trials"][0]["quality"]["complete"] is False


def test_all_repetitions_and_pending_configs_visible(workspace):
    reference, candidate, _, trial = queued(workspace)
    run(workspace[0], trial)
    finish(workspace[0], trial)
    repeat(workspace, reference, candidate)
    config(workspace, reference, threshold=0.8)
    report = compare(workspace, reference)
    assert len(report["configs"]) == 2
    repeated = next(row for row in report["configs"] if row["id"] == candidate["id"])
    assert len(repeated["trials"]) == 2
    assert repeated["repeatability"]["measured"] is True
    assert repeated["repeatability"]["identical_geometry"] is True
    assert repeated["repeatability"]["metrics"]["tp"] == {"count": 2, "min": 2, "mean": 2, "max": 2}
    assert report["coverage"]["complete_trial_count"] == 2
    assert validate_comparison_snapshot(report) == report


def test_missing_and_failed_outputs_never_become_empty_predictions(workspace):
    reference, _, _, trial = queued(workspace)
    finish(workspace[0], trial, "interrupted")
    report = compare(workspace, reference)
    entry = report["configs"][0]["trials"][0]
    assert entry["coverage"] == {"planned": 2, "ready": 0, "failed": 0, "missing": 2}
    assert entry["quality"]["metrics"] is None
    assert all(frame["proposals"] is None and frame["quality"] is None for frame in entry["frames"])
    assert entry["latency"]["total_ms"] is None


def test_latest_correction_and_historical_revision_validation(workspace):
    reference, _, _, trial = queued(workspace)
    run(workspace[0], trial)
    finish(workspace[0], trial)
    output = workspace[0].list("benchmark_outputs", trial_id=trial["id"])[0]
    save_correction(
        workspace[0],
        output["id"],
        expected_revision=0,
        boxes=[],
        status="reviewed",
        reviewer="Fixture reviewer",
    )
    saved = compare(workspace, reference)
    entry = saved["configs"][0]["trials"][0]
    assert entry["corrections"]["reviewed_count"] == 1
    assert entry["corrections"]["recorded_review_ms"] is None
    assert entry["corrections"]["changes"]["rejected"] == 1
    save_correction(
        workspace[0],
        output["id"],
        expected_revision=1,
        boxes=[],
        status="draft",
        reviewer="Fixture reviewer",
    )
    current = compare(workspace, reference)
    assert current["configs"][0]["trials"][0]["corrections"]["reviewed_count"] == 0
    with workspace[0].connect() as conn:
        assert validate_snapshot_sources(conn, saved, store=workspace[0]) == saved


def test_combined_known_cost_and_timing_scope(workspace, providers):
    reference, candidate, preview = configured(workspace)
    trial = launch(workspace, reference, candidate, preview)
    run_combined(workspace, trial)
    finish(workspace[0], trial)
    report = compare(workspace, reference)
    entry = report["configs"][0]["trials"][0]
    assert entry["quality"]["complete"] is True
    assert entry["cost"]["usage_cost_usd"] > 0
    assert entry["cost"]["request_counts"]["response_received"] == 4
    assert entry["latency"]["model_load_included"] is True
    assert entry["identity"]["returned_models"] == ["gpt-6-astra"]
    assert "offline-combined-fixture-key" not in json.dumps(report)
    with workspace[0].connect() as conn:
        assert validate_snapshot_sources(conn, report, store=workspace[0]) == report


def test_combined_unknown_delivery_retains_cost_uncertainty(workspace, providers):
    reference, candidate, preview = configured(workspace)
    providers["fail"] = "review"
    trial = launch(workspace, reference, candidate, preview)
    run_combined(workspace, trial)
    finish(workspace[0], trial, "failed")
    entry = compare(workspace, reference)["configs"][0]["trials"][0]
    assert entry["cost"]["usage_cost_usd"] is None
    assert entry["cost"]["known_usage_cost_usd"] > 0
    assert entry["cost"]["unknown_outcome_count"] == 1
    assert entry["coverage"]["failed"] == 1 and entry["coverage"]["missing"] == 1
    assert entry["quality"]["metrics"] is None


@pytest.mark.parametrize("kind", ["metric", "reference", "correction", "hash"])
def test_snapshot_tampering_is_rejected(workspace, kind):
    reference, _, _, trial = queued(workspace)
    run(workspace[0], trial)
    finish(workspace[0], trial)
    report = deepcopy(compare(workspace, reference))
    entry = report["configs"][0]["trials"][0]
    if kind == "metric":
        entry["quality"]["metrics"]["summary"]["tp"] = 100
    elif kind == "reference":
        report["reference"]["frames"][0]["source_filename"] = "changed.png"
    elif kind == "correction":
        entry["corrections"]["reviewed_count"] = 2
    else:
        entry["frames"][0]["raw_response_sha256"] = "0" * 64
    with workspace[0].connect() as conn, pytest.raises(ValueError):
        validate_snapshot_sources(conn, report, store=workspace[0])


def test_bounded_without_silent_truncation_and_invalid_role(workspace):
    reference, candidate, _, trial = queued(workspace)
    original = workspace[0].get("benchmark_trials", trial["id"])
    finish(workspace[0], trial, "cancelled")
    for _ in range(100):
        trial_id, job_id = new_id(), new_id()
        workspace[0].insert(
            "jobs",
            {
                "id": job_id,
                "kind": "benchmark",
                "status": "cancelled",
                "params": {"trial_id": trial_id},
                "created_at": now(),
            },
        )
        workspace[0].insert("benchmark_trials", {**original, "id": trial_id, "job_id": job_id})
    with pytest.raises(ValueError, match="100"):
        compare(workspace, reference)
    with pytest.raises(ValueError, match="role"):
        build_comparison(workspace[0], reference["id"], role="all")


def test_supplied_transaction_owns_every_read(workspace, monkeypatch):
    reference, _, _, trial = queued(workspace)
    run(workspace[0], trial)
    finish(workspace[0], trial)
    with workspace[0].connect() as conn:
        with pytest.raises(ValueError, match="transaction"):
            build_comparison(workspace[0], reference["id"], role="tuning", connection=conn)
        conn.execute("BEGIN")
        monkeypatch.setattr(
            workspace[0], "connect", lambda: pytest.fail("Opened another transaction")
        )
        assert (
            build_comparison(workspace[0], reference["id"], role="tuning", connection=conn)[
                "coverage"
            ]["trial_count"]
            == 1
        )


def test_sam_scope_and_historical_source_validation(workspace, runtime):
    from test_benchmark_segmentation import configured, launch, run

    reference, candidate, preview = configured(workspace)
    trial = launch(workspace, reference, candidate, preview)
    run(workspace, trial)
    finish(workspace[0], trial)
    report = compare(workspace, reference)
    entry = report["configs"][0]["trials"][0]
    assert entry["quality"]["complete"] is True
    assert entry["latency"]["model_load_included"] is False
    assert entry["latency"]["model_load_ms"] is not None
    assert entry["identity"]["runtime"]["cuda"]["device"] == "Synthetic CUDA fixture"
    with workspace[0].connect() as conn:
        assert validate_snapshot_sources(conn, report, manifest=reference["manifest"]) == report


def test_astra_repeats_surface_resolved_model_changes_without_new_holdout(workspace, transport):
    from test_benchmark_multimodal import configured, launch, run

    reference, candidate, preview = configured(workspace)
    first = launch(workspace, reference, candidate, preview)
    run(workspace, first)
    finish(workspace[0], first)
    saved = compare(workspace, reference)
    transport["response"]["model"] = "gpt-6-astra-2026-10-04"
    preview = preview_benchmark_trial(
        workspace[0], reference["id"], config_id=candidate["id"], role="tuning"
    )
    second = launch(workspace, reference, candidate, preview)
    run(workspace, second)
    finish(workspace[0], second)
    report = compare(workspace, reference)
    repeated = report["configs"][0]["repeatability"]
    assert repeated["mixed_returned_models"] is True
    assert repeated["identical_geometry"] is True
    assert len(transport["requests"]) == 4
    with workspace[0].connect() as conn:
        assert validate_snapshot_sources(conn, saved, manifest=reference["manifest"]) == saved
        assert validate_snapshot_sources(conn, report, manifest=reference["manifest"]) == report


def test_repetition_spread_includes_empty_success_but_excludes_failure(workspace):
    reference, candidate, _, trial = queued(workspace)
    run(workspace[0], trial)
    finish(workspace[0], trial)
    preview = preview_benchmark_trial(
        workspace[0], reference["id"], config_id=candidate["id"], role="tuning"
    )
    empty = create_benchmark_trial(
        workspace[0],
        workspace[1],
        reference["id"],
        config_id=candidate["id"],
        role="tuning",
        expected_fingerprint=preview["fingerprint"],
    )

    class Empty(Detector):
        def predict(self, image):
            raw = super().predict(image)
            raw["detections"] = []
            return raw

    run(workspace[0], empty, factory=Empty)
    finish(workspace[0], empty)
    failed = repeat(workspace, reference, candidate)
    finish(workspace[0], failed, "failed")
    repeated = compare(workspace, reference)["configs"][0]["repeatability"]
    assert repeated["trial_count"] == 3 and repeated["complete_count"] == 2
    assert repeated["identical_geometry"] is False
    assert repeated["metrics"]["tp"] == {"count": 2, "min": 0, "mean": 1, "max": 2}
    assert repeated["metrics"]["precision"]["count"] == 1


def test_latest_review_time_is_cumulative_not_sum_of_revisions(workspace, monkeypatch):
    from iris import benchmark_corrections as review

    reference, _, _, trial = queued(workspace)
    run(workspace[0], trial)
    finish(workspace[0], trial)
    output = workspace[0].list("benchmark_outputs", trial_id=trial["id"])[0]
    clock = [100.0]
    monkeypatch.setattr(
        review,
        "_clock",
        lambda: (datetime(2026, 10, 4, tzinfo=UTC) + timedelta(seconds=clock[0]), clock[0]),
    )
    token = "comparison-offline-owner"
    timer = review.timer_action(
        workspace[0],
        output["id"],
        action="start",
        expected_revision=0,
        token=token,
        operation_id=new_id(),
        reviewer="Fixture reviewer",
    )
    clock[0] += 5
    first = save_correction(
        workspace[0],
        output["id"],
        expected_revision=0,
        boxes=[],
        status="reviewed",
        reviewer="Fixture reviewer",
        timer_revision=timer["revision"],
        timer_token=token,
    )
    timer = review.timer_action(
        workspace[0],
        output["id"],
        action="start",
        expected_revision=first["timer"]["revision"],
        token=token,
        operation_id=new_id(),
        reviewer="Fixture reviewer",
    )
    clock[0] += 7
    save_correction(
        workspace[0],
        output["id"],
        expected_revision=1,
        boxes=[],
        status="reviewed",
        reviewer="Fixture reviewer",
        timer_revision=timer["revision"],
        timer_token=token,
    )
    summary = compare(workspace, reference)["configs"][0]["trials"][0]["corrections"]
    assert summary["recorded_review_ms"] == 12000
    assert summary["timed_count"] == summary["fully_timed_count"] == 1


def test_capture_cutoff_rejects_omitted_failure_but_accepts_later_attempts(workspace):
    from iris.benchmark_analysis import _coverage, _repeatability

    reference, candidate, _, first = queued(workspace)
    run(workspace[0], first)
    finish(workspace[0], first)
    failed = repeat(workspace, reference, candidate)
    finish(workspace[0], failed, "failed")
    saved = compare(workspace, reference)
    captured_at = now()
    repeat(workspace, reference, candidate)
    config(workspace, reference, threshold=0.8)
    with workspace[0].connect() as conn:
        assert (
            validate_snapshot_sources(conn, saved, store=workspace[0], captured_at=captured_at)
            == saved
        )
        forged = deepcopy(saved)
        forged["configs"][0]["trials"].pop()
        forged["configs"][0]["repeatability"] = _repeatability(forged["configs"][0]["trials"])
        forged["coverage"] = _coverage(forged["configs"])
        assert validate_comparison_snapshot(forged) == forged
        with pytest.raises(ValueError, match="omits"):
            validate_snapshot_sources(conn, forged, store=workspace[0], captured_at=captured_at)


def test_capture_cutoff_rejects_omitted_untested_configuration(workspace):
    from iris.benchmark_analysis import _coverage

    reference, _, _, trial = queued(workspace)
    run(workspace[0], trial)
    finish(workspace[0], trial)
    config(workspace, reference, threshold=0.8)
    forged = compare(workspace, reference)
    captured_at = now()
    forged["configs"].pop()
    forged["coverage"] = _coverage(forged["configs"])
    with workspace[0].connect() as conn, pytest.raises(ValueError, match="omits"):
        validate_snapshot_sources(conn, forged, store=workspace[0], captured_at=captured_at)
