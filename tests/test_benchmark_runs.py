"""Candidate isolation, explicit launches, partial evidence and durable trial boundaries."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Barrier, Event

import pytest
from PIL import Image
from test_benchmark import DETECTION, MODEL, TIMING, config, freeze
from test_benchmark import workspace as workspace

from iris.annotations import save_annotation
from iris.benchmark import BenchmarkConflict, benchmark_detail, lock_benchmark
from iris.benchmark_runs import (
    benchmark_trial_detail,
    create_benchmark_trial,
    preview_benchmark_trial,
    run_benchmark_trial,
)
from iris.jobs import JobManager


class Detector:
    metadata = {"weight_sha256": "a" * 64}

    def __init__(self, root, model_id, *, device):
        assert root.name == "data" and model_id == MODEL and device == "cpu"

    def warmup(self, image):
        assert isinstance(image, Image.Image)

    def predict(self, image):
        assert isinstance(image, Image.Image) and image.size == (80, 60)
        return dict(
            input_size=list(image.size), timing=deepcopy(TIMING), detections=[deepcopy(DETECTION)]
        )


def queued(workspace, **changes):
    store, jobs = workspace[:2]
    reference = freeze(workspace)
    candidate = config(workspace, reference, **changes)
    preview = preview_benchmark_trial(
        store, reference["id"], config_id=candidate["id"], role="tuning"
    )
    trial = create_benchmark_trial(
        store,
        jobs,
        reference["id"],
        config_id=candidate["id"],
        role="tuning",
        expected_fingerprint=preview["fingerprint"],
    )
    return reference, candidate, preview, trial


def run(store, trial, factory=Detector, cancelled=lambda: False):
    return run_benchmark_trial(
        store, trial["id"], lambda *_: None, cancelled, detector_factory=factory
    )


def test_adapter_sees_only_pixels_and_model_settings_never_reference(workspace):
    store = workspace[0]
    reference, _, _, trial = queued(workspace)
    # The source may evolve after freezing: trial must neither read nor publish annotations.
    frame = workspace[2][0]
    save_annotation(store, frame["id"], expected_revision=1, boxes=[], decisions={})
    revisions = store.list("annotation_revisions")
    result = run(store, trial)
    assert result["frames_ready"] == 2 and result["frames_issues"] == 0
    assert result["quality"]["metrics"]["summary"]["tp"] == 2
    assert store.list("annotation_revisions") == revisions
    assert not store.list("annotation_suggestions")
    detail = benchmark_trial_detail(store, trial["id"])
    assert detail["corrections"]["planned_count"] == 2
    assert detail["corrections"]["recorded_review_ms"] is None
    assert detail["latency"]["measured_count"] == 2
    assert detail["latency"]["total_ms"] >= 0
    for output in detail["outputs"]:
        assert output["metadata"]["reference_withheld"] is True
        assert output["raw_response"]["detections"] == [DETECTION]
        assert output["result"]["proposals"][0]["id"] == "detection-0"
    assert (
        store.get("benchmarks", reference["id"])["manifest_sha256"] == reference["manifest_sha256"]
    )


def test_launch_receipt_idempotent_across_managers_and_fresh_preview_explicit(workspace):
    store, jobs = workspace[:2]
    reference = freeze(workspace)
    candidate = config(workspace, reference)
    values = dict(config_id=candidate["id"], role="tuning")
    preview = preview_benchmark_trial(store, reference["id"], **values)
    gate = Barrier(2)

    def launch(_):
        gate.wait()
        return create_benchmark_trial(
            store,
            JobManager(store),
            reference["id"],
            **values,
            expected_fingerprint=preview["fingerprint"],
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        trials = list(pool.map(launch, range(2)))
    assert trials[0]["id"] == trials[1]["id"] and len(store.list("jobs")) == 1
    result = run(store, trials[0])
    store.update("jobs", trials[0]["job_id"], {"status": "succeeded", "result": result})
    replay = create_benchmark_trial(
        store, jobs, reference["id"], **values, expected_fingerprint=preview["fingerprint"]
    )
    assert replay["id"] == trials[0]["id"]
    fresh = preview_benchmark_trial(store, reference["id"], **values)
    assert fresh["fingerprint"] != preview["fingerprint"]
    repeated = create_benchmark_trial(
        store, jobs, reference["id"], **values, expected_fingerprint=fresh["fingerprint"]
    )
    assert repeated["id"] != replay["id"] and len(store.list("jobs")) == 2


def test_only_one_executor_can_load_model_for_same_trial(workspace):
    store = workspace[0]
    _, _, _, trial = queued(workspace)
    entered, release = Event(), Event()

    class Blocking(Detector):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            entered.set()
            assert release.wait(timeout=5)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(run, store, trial, Blocking)
        assert entered.wait(timeout=5)
        try:
            with pytest.raises(BenchmarkConflict, match="claimed"):
                run(store, trial, lambda *args, **kwargs: pytest.fail("Second model load"))
        finally:
            release.set()
        assert future.result()["frames_ready"] == 2
    assert len(store.list("benchmark_outputs", trial_id=trial["id"])) == 2


@pytest.mark.parametrize("failure", ["invalid", "exception", "nonfinite", "oversize"])
def test_invalid_outputs_preserve_evidence_without_becoming_negatives(workspace, failure):
    store = workspace[0]
    _, _, _, trial = queued(workspace)

    class Invalid(Detector):
        def predict(self, image):
            if failure == "exception":
                raise RuntimeError()
            output = super().predict(image)
            if failure == "invalid":
                output["detections"][0]["box"][0] = -1
            elif failure == "nonfinite":
                output["detections"][0]["score"] = float("nan")
            else:
                output["oversized"] = "x" * (2 * 1024 * 1024)
            return output

    result = run(store, trial, Invalid)
    assert result["frames_ready"] == 0 and result["frames_issues"] == 2
    assert not result["quality"]["complete"] and result["quality"]["metrics"] is None
    outputs = store.list("benchmark_outputs", trial_id=trial["id"])
    assert all(output["result"] is None and output["error"] for output in outputs)
    if failure == "invalid":
        assert outputs[0]["raw_response"]["detections"][0]["box"][0] == -1
    elif failure == "exception":
        assert outputs[0]["raw_response"] is None and outputs[0]["error"] == "RuntimeError"
    else:
        assert len(outputs[0]["raw_response"]["non_json_output_summary"]) <= 4000


def test_cancellation_after_first_saved_image_keeps_partial_and_no_headline(workspace):
    store = workspace[0]
    _, _, _, trial = queued(workspace)
    result = run(store, trial, cancelled=lambda: bool(store.list("benchmark_outputs")))
    assert result["cancelled"] and result["outputs_created"] == 1
    output = store.list("benchmark_outputs")[0]
    assert output["raw_response"] and output["result"] is None
    assert output["metadata"]["state"] == "cancelled"
    assert not benchmark_trial_detail(store, trial["id"])["quality"]["complete"]


def test_interrupt_during_prediction_preserves_raw_without_late_publication(workspace):
    store = workspace[0]
    _, _, _, trial = queued(workspace)

    class Interrupted(Detector):
        def predict(self, image):
            store.update(
                "jobs", trial["job_id"], {"status": "interrupted", "result": {"old": True}}
            )
            return super().predict(image)

    result = run(store, trial, Interrupted)
    assert result["cancelled"] and result["frames_ready"] == 0
    output = store.list("benchmark_outputs")[0]
    assert output["raw_response"] and output["result"] is None
    assert store.get("jobs", trial["job_id"])["result"] == {"old": True}


def test_partial_failure_keeps_first_success_and_no_complete_quality(workspace):
    store = workspace[0]
    _, _, _, trial = queued(workspace)

    class Partial(Detector):
        calls = 0

        def predict(self, image):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("Second image failed")
            return super().predict(image)

    result = run(store, trial, Partial)
    assert result["frames_ready"] == result["frames_issues"] == 1
    assert not result["quality"]["complete"]
    assert sum(row["result"] is not None for row in store.list("benchmark_outputs")) == 1


def test_active_trial_blocks_lock_and_evaluation_requires_explicit_lock(workspace):
    store = workspace[0]
    reference, candidate, _, trial = queued(workspace)
    with pytest.raises(BenchmarkConflict, match="active"):
        lock_benchmark(
            store,
            reference["id"],
            expected_fingerprint=benchmark_detail(store, reference["id"])["lock_fingerprint"],
        )
    with pytest.raises(BenchmarkConflict, match="explicit lock"):
        preview_benchmark_trial(
            store, reference["id"], config_id=candidate["id"], role="evaluation"
        )
    result = run(store, trial)
    store.update("jobs", trial["job_id"], {"status": "succeeded", "result": result})
    lock_benchmark(
        store,
        reference["id"],
        expected_fingerprint=benchmark_detail(store, reference["id"])["lock_fingerprint"],
    )
    assert preview_benchmark_trial(
        store, reference["id"], config_id=candidate["id"], role="evaluation"
    )["reference_withheld"]
    with pytest.raises(BenchmarkConflict, match="explicit lock"):
        preview_benchmark_trial(store, reference["id"], config_id=candidate["id"], role="tuning")


def test_changed_frozen_pixels_never_reach_detector_or_complete_quality(workspace):
    store = workspace[0]
    reference, _, _, trial = queued(workspace)
    frame = next(row for row in reference["manifest"]["frames"] if row["role"] == "tuning")
    store.artifact_path(frame["image_path"]).write_bytes(b"changed pixels")
    result = run(store, trial)
    assert result["frames_issues"] == 1 and result["frames_ready"] == 1
    assert not result["quality"]["complete"]


def test_tiled_invalid_before_complete_return_is_error_not_empty_prediction(workspace):
    store = workspace[0]
    _, _, _, trial = queued(workspace, inference_mode="tiled", tile_size=128)

    class Outside(Detector):
        def predict(self, image):
            result = super().predict(image)
            result["detections"][0]["box"][0] = -1
            return result

    result = run(store, trial, Outside)
    assert result["frames_issues"] == 2
    for output in store.list("benchmark_outputs"):
        assert output["raw_response"] is None and output["result"] is None
        assert "outside its tile" in output["error"]


@pytest.mark.parametrize("change", ["job_kind", "job_link", "trial_snapshot", "weight"])
def test_tampered_job_or_frozen_config_refused_before_factory(workspace, change):
    store = workspace[0]
    _, _, _, trial = queued(workspace)
    if change == "job_kind":
        store.update("jobs", trial["job_id"], {"kind": "infer"})
    elif change == "job_link":
        store.update("jobs", trial["job_id"], {"params": {"trial_id": "another-trial"}})
    elif change == "trial_snapshot":
        frozen = deepcopy(trial["config"])
        frozen["candidate_config"]["threshold"] = 0.99
        store.update("benchmark_trials", trial["id"], {"config": frozen})
    else:
        workspace[4]["weight_sha256"] = "b" * 64
    with pytest.raises(ValueError):
        run(store, trial, lambda *args, **kwargs: pytest.fail("Unsafe detector load"))
    assert not store.list("benchmark_outputs")
