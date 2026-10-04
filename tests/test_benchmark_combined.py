"""Combined execution with real orchestration and exclusively simulated providers."""

import json
from copy import deepcopy

import pytest
from test_benchmark import freeze
from test_benchmark import workspace as workspace
from test_benchmark_segmentation import IDENTITY, native

from iris import (
    benchmark_combined,
    combined_provider,
    multimodal_provider,
    sam_provider,
    sam_runtime,
)
from iris.benchmark import BenchmarkConflict, create_benchmark_config, preview_benchmark_config
from iris.benchmark_runs import (
    benchmark_trial_detail,
    create_benchmark_trial,
    preview_benchmark_trial,
    run_benchmark_trial,
)


def response(content, stage):
    return {
        "id": "synthetic-" + stage,
        "model": "gpt-6-astra",
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": json.dumps(content)}],
            }
        ],
        "usage": {
            "input_tokens": 1000,
            "output_tokens": 100,
            "total_tokens": 1100,
            "input_tokens_details": {"cached_tokens": 0},
        },
    }


@pytest.fixture
def providers(monkeypatch):
    monkeypatch.setenv(multimodal_provider.KEY_ENV, "offline-combined-fixture-key")
    record = {"requests": [], "loads": [], "phrases": [], "closed": 0, "image_calls": 0}
    status = {
        "ready": True,
        "status": "ready",
        "reason": None,
        "runtime": {"ready": True, "identity": deepcopy(IDENTITY)},
    }
    record["status"] = status
    monkeypatch.setattr(sam_provider, "provider_status", lambda *a, **k: deepcopy(status))

    def request(payload, key, *, before_dispatch):
        assert key == "offline-combined-fixture-key"
        before_dispatch()
        stage = payload["text"]["format"]["name"].removeprefix("iris_combined_")
        body = json.loads(payload["input"][0]["content"][0]["text"])
        record["requests"].append({"stage": stage, "payload": deepcopy(payload), "body": body})
        if record.get("fail") == stage:
            raise TimeoutError("Simulated transport timeout")
        if stage == "planning":
            result = {
                "prompts": [
                    {
                        "class_id": item["id"],
                        "text": item["name"] + " " + str(len(record["requests"])),
                    }
                    for item in body["classes"]
                ]
            }
        else:
            result = {
                "decisions": [
                    {
                        "id": item["id"],
                        "label": item["label"],
                        "action": "accept",
                        "reason": "Synthetic review decision",
                        "uncertain": False,
                    }
                    for item in body["candidates"]
                ]
            }
        if record.get("invalid") == stage:
            result = {"invalid": True}
        raw = response(result, stage)
        if record.get("after_request"):
            record["after_request"](stage)
        return raw, {"http_status": 200, "http_request_id": "synthetic-http-" + stage}

    monkeypatch.setattr(multimodal_provider, "_request", request)

    class Runtime:
        def __init__(self, config, path, *, cancelled, allow_dynamic_prompts):
            assert allow_dynamic_prompts is True
            self.config = deepcopy(config)
            self.metadata = {"runtime_identity": deepcopy(IDENTITY), "allow_dynamic_prompts": True}
            if record.get("changed_load"):
                self.metadata["runtime_identity"]["cuda"]["device"] = "Changed device"
            record["loads"].append(deepcopy(config))

        def set_prompts(self, prompts, *, cancelled):
            self.config["prompts"] = deepcopy(prompts)

        def predict(self, image, class_prompts, threshold, cancelled):
            assert class_prompts == self.config["prompts"]
            record["phrases"].append(deepcopy(class_prompts))
            record["image_calls"] += 1
            raw = native(self.config, *image.size)
            raw["metadata"].update(self.metadata)
            if record.get("fail") == "grounding":
                raise sam_runtime.SamRuntimeError(
                    "Synthetic local failure", raw_response={"partial": True}
                )
            if record.get("empty"):
                for item in raw["prompts"]:
                    item.update(boxes=[], scores=[], native_indices=[])
            if record.get("after_grounding"):
                record["after_grounding"]()
            return raw

        def close(self):
            record["closed"] += 1

    monkeypatch.setattr(sam_runtime, "SamRuntime", Runtime)
    return record


def configured(workspace):
    reference = freeze(workspace)
    values = {"approach": "combined", "model_id": benchmark_combined.MODEL, "combined": {}}
    preview = preview_benchmark_config(workspace[0], reference["id"], **values)
    candidate = create_benchmark_config(
        workspace[0],
        reference["id"],
        name="Combined offline fixture",
        expected_fingerprint=preview["fingerprint"],
        **values,
    )
    preview = preview_benchmark_trial(
        workspace[0], reference["id"], config_id=candidate["id"], role="tuning"
    )
    return reference, candidate, preview


def launch(workspace, reference, candidate, preview, **changes):
    settings = {
        "config_id": candidate["id"],
        "role": "tuning",
        "expected_fingerprint": preview["fingerprint"],
        "approve_external": True,
        "max_cost_usd": preview["external_plan"]["estimate"]["upper_bound_usd"],
        "preview_token": preview["preview_token"],
    }
    return create_benchmark_trial(
        workspace[0], workspace[1], reference["id"], **{**settings, **changes}
    )


def run(workspace, trial, cancelled=lambda: False):
    return run_benchmark_trial(workspace[0], trial["id"], lambda *_: None, cancelled)


def test_prepare_without_setup_is_offline_and_launch_is_blocked(workspace, monkeypatch):
    for name in (
        multimodal_provider.KEY_ENV,
        multimodal_provider.FALLBACK_KEY_ENV,
        "IRIS_SAM_PYTHON",
    ):
        monkeypatch.delenv(name, raising=False)
    reference, candidate, preview = configured(workspace)
    assert preview["launch_allowed"] is False
    assert preview["work"]["request_count"] == 4
    template = preview["external_plan"]["requests"][0]["review"]["template"]
    assert template["template"] is True and template["max_dynamic_text_bytes"] == 131072
    assert "image_bytes" not in template and "request_sha256" not in template
    with pytest.raises(ValueError):
        launch(workspace, reference, candidate, preview)
    assert not workspace[0].list("jobs")


def test_two_stages_per_image_preserve_raw_and_geometry_without_reference(
    workspace, providers, monkeypatch
):
    reference, candidate, preview = configured(workspace)
    trial = launch(workspace, reference, candidate, preview)
    assert launch(workspace, reference, candidate, preview)["id"] == trial["id"]
    before = workspace[0].list("annotation_revisions")
    original = combined_provider.normalize_plan

    def normalize(raw, config):
        rows = workspace[0].list("benchmark_outputs", trial_id=trial["id"])
        assert any((row["raw_response"] or {}).get("planning") == raw for row in rows)
        return original(raw, config)

    monkeypatch.setattr(combined_provider, "normalize_plan", normalize)
    result = run(workspace, trial)
    assert result["frames_ready"] == 2 and result["frames_issues"] == 0
    assert (
        result["quality"]["complete"]
        and result["quality"]["protocol"] == benchmark_combined.COMBINED_SCORING
    )
    assert [item["stage"] for item in providers["requests"]] == ["planning", "review"] * 2
    assert len(providers["loads"]) == 1 and providers["closed"] == 1
    assert providers["phrases"][0] != providers["phrases"][1]
    for secret in ("human-box", "withheld sentinel", "Independent fixture author", "0.png"):
        assert secret not in json.dumps(providers["requests"])
    assert workspace[0].list("annotation_revisions") == before
    detail = benchmark_trial_detail(workspace[0], trial["id"])
    assert detail["external_dispatch"]["counts"]["response_received"] == 4
    assert detail["external_dispatch"]["usage_missing_count"] == 0
    assert detail["latency"]["measured_count"] == 2 and detail["latency"]["model_load_ms"] >= 0
    for output in detail["outputs"]:
        assert set(output["raw_response"]) == {"planning", "grounding", "review"}
        proposal = output["result"]["proposals"][0]
        assert proposal["box"] == [2, 3, 30, 35] and proposal["score"] is None
        assert proposal["source"]["native_score"] == 0.85
    with pytest.raises((ValueError, RuntimeError)):
        run(workspace, trial)
    assert len(providers["requests"]) == 4


@pytest.mark.parametrize("stage", ["planning", "grounding", "review"])
def test_failure_stops_later_stages_and_images_without_retry(workspace, providers, stage):
    reference, candidate, preview = configured(workspace)
    trial = launch(workspace, reference, candidate, preview)
    providers["fail"] = stage
    result = run(workspace, trial)
    assert result["frames_issues"] == 1 and result["frames_ready"] == 0
    assert result["quality"]["metrics"] is None
    detail = benchmark_trial_detail(workspace[0], trial["id"])
    assert all(row["result"] is None for row in detail["outputs"])
    assert len(providers["requests"]) == (2 if stage == "review" else 1)
    if stage != "grounding":
        assert detail["external_dispatch"]["unknown_outcome_count"] == 1
        assert detail["external_dispatch"]["usage_cost_usd"] is None
    assert detail["latency"]["measured_count"] == 1


@pytest.mark.parametrize("stage", ["planning", "review"])
def test_invalid_response_is_saved_and_not_an_empty_success(workspace, providers, stage):
    reference, candidate, preview = configured(workspace)
    trial = launch(workspace, reference, candidate, preview)
    providers["invalid"] = stage
    result = run(workspace, trial)
    assert result["frames_ready"] == 0 and result["frames_issues"] == 1
    rows = workspace[0].list("benchmark_outputs", trial_id=trial["id"])
    assert any((row["raw_response"] or {}).get(stage) for row in rows)


@pytest.mark.parametrize("stage", ["planning", "grounding", "review"])
def test_cancellation_preserves_stage_response_and_stops_progression(workspace, providers, stage):
    reference, candidate, preview = configured(workspace)
    trial = launch(workspace, reference, candidate, preview)

    def cancel(current=None):
        if current is None or current == stage:
            workspace[0].update("jobs", trial["job_id"], {"cancel_requested": True})

    providers["after_grounding" if stage == "grounding" else "after_request"] = cancel
    result = run(
        workspace, trial, lambda: workspace[0].get("jobs", trial["job_id"])["cancel_requested"]
    )
    assert result["cancelled"] and result["frames_ready"] == 0
    assert len(providers["requests"]) == (2 if stage == "review" else 1)
    assert all(
        row["result"] is None
        for row in workspace[0].list("benchmark_outputs", trial_id=trial["id"])
    )


def test_empty_sam_output_still_receives_explicit_review(workspace, providers):
    reference, candidate, preview = configured(workspace)
    trial = launch(workspace, reference, candidate, preview)
    providers["empty"] = True
    result = run(workspace, trial)
    assert result["frames_ready"] == 2 and len(providers["requests"]) == 4
    for request in providers["requests"]:
        if request["stage"] == "review":
            assert request["body"]["candidates"] == []


@pytest.mark.parametrize(
    "change",
    [{"approve_external": False}, {"max_cost_usd": 0.000001}, {"preview_token": "invalid"}],
)
def test_fresh_consent_and_full_two_stage_budget_required(workspace, providers, change):
    reference, candidate, preview = configured(workspace)
    with pytest.raises((ValueError, BenchmarkConflict)):
        launch(workspace, reference, candidate, preview, **change)
    assert not providers["requests"] and not workspace[0].list("jobs")


def test_loaded_runtime_drift_stops_before_sam_or_second_external_call(workspace, providers):
    reference, candidate, preview = configured(workspace)
    trial = launch(workspace, reference, candidate, preview)
    providers["changed_load"] = True
    result = run(workspace, trial)
    assert result["frames_issues"] == 1 and result["frames_ready"] == 0
    assert len(providers["requests"]) == 1 and providers["image_calls"] == 0
    assert providers["closed"] == 1
