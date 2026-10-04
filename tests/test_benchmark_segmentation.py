"""SAM orchestration using native-output fixtures, never a model or a download."""

import json
from copy import deepcopy

import pytest
from test_benchmark import freeze
from test_benchmark import workspace as workspace

from iris import benchmark_segmentation, sam_provider
from iris.benchmark import BenchmarkConflict, create_benchmark_config, preview_benchmark_config
from iris.benchmark_runs import (
    benchmark_trial_detail,
    create_benchmark_trial,
    preview_benchmark_trial,
    run_benchmark_trial,
)

IDENTITY = {
    "python": "3.12.3",
    "isolated": True,
    "packages": {
        "torch": "2.10.0+cu126",
        "torchvision": "0.25.0+cu126",
        "numpy": "1.26.4",
        "sam3": "0.1.0",
    },
    "code_revision": sam_provider.CODE_REVISION,
    "tokenizer_sha256": "e" * 64,
    "cuda": {
        "available": True,
        "version": "12.6",
        "bfloat16": True,
        "device": "Synthetic CUDA fixture",
        "capability": [8, 0],
    },
}


def native(config, width=80, height=60):
    return {
        "protocol": sam_provider.RAW_PROTOCOL,
        "complete": True,
        "image": {"width": width, "height": height},
        "coordinates": {
            "format": "xyxy",
            "space": "normalized",
            "image_size": [width, height],
            "to_original": {"scale": [width, height], "offset": [0, 0]},
        },
        "prompts": [
            {
                **prompt,
                "boxes": [[2 / width, 3 / height, 30 / width, 35 / height]]
                if prompt["class_id"] == "person"
                else [],
                "scores": [0.85] if prompt["class_id"] == "person" else [],
                "native_indices": [2] if prompt["class_id"] == "person" else [],
                "error": None,
            }
            for prompt in config["prompts"]
        ],
        "metadata": {"fixture": True},
    }


@pytest.fixture
def runtime(monkeypatch):
    record = {"loads": [], "images": [], "closed": 0, "change_loaded": False}
    status = {
        "status": "ready",
        "ready": True,
        "reason": "Offline fixture",
        "runtime": {"ready": True, "identity": deepcopy(IDENTITY)},
    }
    record["status"] = status
    monkeypatch.setattr(sam_provider, "provider_status", lambda *a, **k: deepcopy(status))

    class Adapter:
        def __init__(self, root, config, *, cancelled):
            self.config = deepcopy(config)
            self.metadata = {"runtime_identity": deepcopy(IDENTITY), "fixture": True}
            if record["change_loaded"]:
                self.metadata["runtime_identity"]["cuda"]["device"] = "Changed GPU"
            record["loads"].append(config)

        def predict(self, image, *, cancelled):
            assert image.mode == "RGB" and not image.info
            record["images"].append(image.size)
            raw = native(self.config, *image.size)
            if record.get("action"):
                return record["action"](raw)
            return raw

        def close(self):
            record["closed"] += 1

    monkeypatch.setattr(sam_provider, "Sam3Preannotator", Adapter)
    return record


def configured(workspace, **segmentation):
    reference = freeze(workspace)
    values = {"model_id": "sam3", "approach": "segmentation", "segmentation": segmentation}
    preview = preview_benchmark_config(workspace[0], reference["id"], **values)
    candidate = create_benchmark_config(
        workspace[0],
        reference["id"],
        name="Offline SAM fixture",
        expected_fingerprint=preview["fingerprint"],
        **values,
    )
    preview = preview_benchmark_trial(
        workspace[0], reference["id"], config_id=candidate["id"], role="tuning"
    )
    return reference, candidate, preview


def launch(workspace, reference, candidate, preview):
    return create_benchmark_trial(
        workspace[0],
        workspace[1],
        reference["id"],
        config_id=candidate["id"],
        role="tuning",
        expected_fingerprint=preview["fingerprint"],
    )


def run(workspace, trial, cancelled=lambda: False):
    return run_benchmark_trial(workspace[0], trial["id"], lambda *_: None, cancelled)


def test_missing_setup_still_freezes_config_but_cannot_queue(workspace, monkeypatch):
    monkeypatch.delenv("IRIS_SAM_PYTHON", raising=False)
    reference, candidate, preview = configured(workspace)
    assert preview["launch_allowed"] is False
    assert candidate["config"]["provider_config"]["weights"]["sha256"]
    with pytest.raises(ValueError):
        launch(workspace, reference, candidate, preview)
    assert not workspace[0].list("jobs")


def test_complete_trial_persists_raw_first_and_never_sends_reference(
    workspace, runtime, monkeypatch
):
    reference, candidate, preview = configured(workspace)
    assert preview["work"]["image_encodings"] == 2
    assert preview["work"]["prompt_evaluations"] == 2 * len(
        candidate["config"]["provider_config"]["prompts"]
    )
    trial = launch(workspace, reference, candidate, preview)
    assert launch(workspace, reference, candidate, preview)["id"] == trial["id"]
    before = workspace[0].list("annotation_revisions")
    original = sam_provider.normalize_response

    def normalize(raw, *args, **kwargs):
        rows = workspace[0].list("benchmark_outputs", trial_id=trial["id"])
        saved = [row for row in rows if row["metadata"]["state"] == "raw_saved"]
        assert len(saved) == 1 and saved[0]["raw_response"] == raw
        assert saved[0]["result"] is None
        return original(raw, *args, **kwargs)

    monkeypatch.setattr(sam_provider, "normalize_response", normalize)
    result = run(workspace, trial)
    assert result["frames_ready"] == 2 and result["frames_issues"] == 0
    assert result["quality"]["complete"] is True
    assert result["quality"]["protocol"] == benchmark_segmentation.SAM_SCORING
    assert len(runtime["loads"]) == 1 and runtime["closed"] == 1
    assert len(runtime["images"]) == 2
    sent = json.dumps(runtime["loads"])
    for forbidden in ("human-box", "withheld sentinel", "Independent fixture author", "0.png"):
        assert forbidden not in sent
    assert workspace[0].list("annotation_revisions") == before
    detail = benchmark_trial_detail(workspace[0], trial["id"])
    assert detail["latency"]["model_load_ms"] >= 0
    assert detail["latency"]["measured_count"] == 2
    assert all(row["result"]["proposals"][0]["box"] == [2, 3, 30, 35] for row in detail["outputs"])
    with pytest.raises(BenchmarkConflict):
        run(workspace, trial)


@pytest.mark.parametrize("failure", ["partial", "invalid", "exception", "oversized"])
def test_failure_retains_evidence_and_stops_remaining_images(workspace, runtime, failure):
    reference, candidate, preview = configured(workspace)
    trial = launch(workspace, reference, candidate, preview)

    def action(raw):
        if failure == "exception":
            raise sam_provider.ProviderResponseError("Interrupted", raw_response={"partial": True})
        if failure == "partial":
            raw["complete"] = False
        elif failure == "invalid":
            raw["prompts"][0]["class_id"] = "not-a-class"
        elif failure == "oversized":
            raw["padding"] = "x" * (2 * 1024 * 1024)
        return raw

    runtime["action"] = action
    result = run(workspace, trial)
    assert result["frames_ready"] == 0 and result["frames_issues"] == 1
    assert result["quality"]["metrics"] is None
    assert len(runtime["images"]) == 1 and runtime["closed"] == 1
    output = workspace[0].list("benchmark_outputs", trial_id=trial["id"])[0]
    assert output["raw_response"] is not None and output["result"] is None and output["error"]


def test_cancel_after_prediction_keeps_raw_without_publishing(workspace, runtime):
    reference, candidate, preview = configured(workspace)
    trial = launch(workspace, reference, candidate, preview)

    def action(raw):
        workspace[0].update("jobs", trial["job_id"], {"cancel_requested": True})
        return raw

    runtime["action"] = action
    result = run(workspace, trial)
    output = workspace[0].list("benchmark_outputs", trial_id=trial["id"])[0]
    assert result["cancelled"] and output["metadata"]["state"] == "cancelled"
    assert output["raw_response"] and output["result"] is None
    assert len(runtime["images"]) == 1


@pytest.mark.parametrize("when", ["approval", "execution", "loaded"])
def test_runtime_drift_never_predicts(workspace, runtime, when):
    reference, candidate, preview = configured(workspace)
    if when == "approval":
        runtime["status"]["runtime"]["identity"]["cuda"]["device"] = "Changed GPU"
        with pytest.raises(BenchmarkConflict):
            launch(workspace, reference, candidate, preview)
    else:
        trial = launch(workspace, reference, candidate, preview)
        if when == "execution":
            runtime["status"]["runtime"]["identity"]["cuda"]["device"] = "Changed GPU"
        else:
            runtime["change_loaded"] = True
        with pytest.raises(BenchmarkConflict):
            run(workspace, trial)
    assert not runtime["images"] and not workspace[0].list("benchmark_outputs")
    if when == "loaded":
        assert runtime["closed"] == 1


def test_claim_prevents_second_load_even_if_initial_load_failed(workspace, runtime, monkeypatch):
    reference, candidate, preview = configured(workspace)
    trial = launch(workspace, reference, candidate, preview)

    def fail(*args, **kwargs):
        raise RuntimeError("Synthetic load failure")

    monkeypatch.setattr(sam_provider, "Sam3Preannotator", fail)
    with pytest.raises(RuntimeError, match="Synthetic load failure"):
        run(workspace, trial)
    with pytest.raises(BenchmarkConflict, match="claimed"):
        run(workspace, trial)


@pytest.mark.parametrize("phase", ["preview", "execution"])
def test_oversize_images_fail_before_queue_or_model_load(workspace, runtime, monkeypatch, phase):
    from iris import sam_runtime

    reference, candidate, preview = configured(workspace)
    trial = launch(workspace, reference, candidate, preview) if phase == "execution" else None
    monkeypatch.setattr(sam_runtime.wire, "MAX_IMAGE_PIXELS", 1)
    with pytest.raises(sam_runtime.SamRuntimeError, match="pixel"):
        if phase == "execution":
            run(workspace, trial)
        else:
            preview_benchmark_trial(
                workspace[0], reference["id"], config_id=candidate["id"], role="tuning"
            )
    assert not runtime["loads"]
    assert not workspace[0].list("benchmark_outputs")
