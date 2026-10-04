"""External benchmark integration with an intercepted HTTP transport; never live calls."""

import hashlib
import json
from copy import deepcopy

import pytest
from test_benchmark import freeze
from test_benchmark import workspace as workspace
from test_multimodal_provider import transport as transport

from iris import benchmark_multimodal, multimodal_provider
from iris.benchmark import BenchmarkConflict, create_benchmark_config, preview_benchmark_config
from iris.benchmark_runs import (
    benchmark_trial_detail,
    create_benchmark_trial,
    preview_benchmark_trial,
    run_benchmark_trial,
)


def configured(workspace):
    reference = freeze(workspace)
    store = workspace[0]
    values = {"model_id": "gpt-6-astra", "approach": "multimodal"}
    preview = preview_benchmark_config(store, reference["id"], **values)
    candidate = create_benchmark_config(
        store,
        reference["id"],
        name="Offline Astra fixture",
        expected_fingerprint=preview["fingerprint"],
        **values,
    )
    preview = preview_benchmark_trial(
        store, reference["id"], config_id=candidate["id"], role="tuning"
    )
    return reference, candidate, preview


def launch(workspace, reference, candidate, preview, **changes):
    values = dict(
        config_id=candidate["id"],
        role="tuning",
        expected_fingerprint=preview["fingerprint"],
        approve_external=True,
        max_cost_usd=1.0,
        preview_token=preview["preview_token"],
    )
    values.update(changes)
    return create_benchmark_trial(workspace[0], workspace[1], reference["id"], **values)


def run(workspace, trial, cancelled=lambda: False):
    return run_benchmark_trial(workspace[0], trial["id"], lambda *_: None, cancelled)


def test_no_key_still_allows_local_preview_but_not_dispatch(workspace, monkeypatch):
    monkeypatch.delenv("IRIS_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    reference, candidate, preview = configured(workspace)
    assert not preview["launch_allowed"]
    request = preview["external_plan"]["requests"][0]
    image = benchmark_multimodal.input_image(workspace[0], candidate["id"], request["frame_id"])
    assert hashlib.sha256(image).hexdigest() == request["input"]["image"]["sha256"]
    with pytest.raises(ValueError, match="KEY"):
        launch(workspace, reference, candidate, preview)
    assert not workspace[0].list("jobs")
    assert not workspace[0].list("benchmark_trials")


@pytest.mark.parametrize(
    "change",
    [
        {"approve_external": False},
        {"max_cost_usd": 0},
        {"max_cost_usd": 0.000001},
        {"max_cost_usd": float("nan")},
        {"preview_token": None},
        {"preview_token": "bad"},
        {"expected_fingerprint": "a" * 64},
    ],
)
def test_explicit_current_approval_required(workspace, transport, change):
    reference, candidate, preview = configured(workspace)
    with pytest.raises((ValueError, BenchmarkConflict)):
        launch(workspace, reference, candidate, preview, **change)
    assert not transport["requests"] and not workspace[0].list("jobs")


def test_expired_preview_and_restart_receipt_rejected(workspace, transport, monkeypatch):
    reference, candidate, preview = configured(workspace)
    original_time = benchmark_multimodal.time.time()
    monkeypatch.setattr(benchmark_multimodal.time, "time", lambda: original_time + 601)
    with pytest.raises(BenchmarkConflict, match="expired"):
        launch(workspace, reference, candidate, preview)
    monkeypatch.setattr(benchmark_multimodal.time, "time", lambda: original_time)
    monkeypatch.setattr(benchmark_multimodal, "_PREVIEW_SECRET", b"another-server-process")
    with pytest.raises(BenchmarkConflict, match="expired"):
        launch(workspace, reference, candidate, preview)
    assert not transport["requests"]


def test_saved_confirmation_replay_does_not_create_another_job(workspace, transport, monkeypatch):
    reference, candidate, preview = configured(workspace)
    trial = launch(workspace, reference, candidate, preview)
    monkeypatch.setattr(benchmark_multimodal, "_PREVIEW_SECRET", b"restarted")
    replay = launch(workspace, reference, candidate, preview)
    assert replay["id"] == trial["id"]
    assert len(workspace[0].list("jobs")) == 1
    assert len(workspace[0].list("benchmark_outputs")) == 2
    assert not transport["requests"]


def test_complete_trial_keeps_reference_out_of_payload_and_raw_before_proposals(
    workspace, transport
):
    reference, candidate, preview = configured(workspace)
    trial = launch(workspace, reference, candidate, preview)
    before = workspace[0].list("annotation_revisions")
    result = run(workspace, trial)
    assert result["frames_ready"] == 2 and result["frames_issues"] == 0
    assert result["quality"]["complete"] is True
    assert result["quality"]["protocol"] == benchmark_multimodal.MULTIMODAL_SCORING
    for request, sent in zip(
        preview["external_plan"]["requests"], transport["requests"], strict=True
    ):
        body = sent[2]
        assert hashlib.sha256(body).hexdigest() == request["input"]["request_sha256"]
        for forbidden in (
            b"human-box",
            b"withheld sentinel",
            b"Independent fixture author",
            b"0.png",
        ):
            assert forbidden not in body
        payload = json.loads(body)
        assert payload["store"] is False and payload["tools"] == []
    detail = benchmark_trial_detail(workspace[0], trial["id"])
    assert detail["external_dispatch"]["counts"]["response_received"] == 2
    assert detail["external_dispatch"]["usage_cost_usd"] > 0
    assert detail["latency"]["measured_count"] == 2
    assert "OpenAI" in detail["latency"]["includes"]
    assert all(row["result"]["proposals"][0]["score"] is None for row in detail["outputs"])
    assert all(row["raw_response"] == transport["response"] for row in detail["outputs"])
    assert workspace[0].list("annotation_revisions") == before
    assert not workspace[0].list("annotation_suggestions")
    with pytest.raises((ValueError, RuntimeError), match="attempted"):
        run(workspace, trial)
    assert len(transport["requests"]) == 2


@pytest.mark.parametrize("failure", ["timeout", "http", "invalid", "incomplete", "empty"])
def test_failed_responses_stop_remaining_requests_and_are_not_empty(workspace, transport, failure):
    reference, candidate, preview = configured(workspace)
    trial = launch(workspace, reference, candidate, preview)
    if failure == "timeout":
        transport["error"] = TimeoutError("fixture transport interrupted")
    elif failure == "http":
        transport["status"] = 401
    elif failure == "invalid":
        transport["response"]["output"][0]["content"][0]["text"] = "broken JSON"
    elif failure == "incomplete":
        transport["remaining"] = 5
    else:
        transport["response"]["output"][0]["content"][0]["text"] = json.dumps(
            {"coordinate_space": "normalized", "proposals": []}
        )
    result = run(workspace, trial)
    detail = benchmark_trial_detail(workspace[0], trial["id"])
    if failure == "empty":
        assert result["quality"]["complete"] and result["quality"]["metrics"]["summary"]["fn"] == 2
        assert len(transport["requests"]) == 2
        return
    assert len(transport["requests"]) == 1
    assert result["frames_ready"] == 0 and result["frames_issues"] == 1
    assert result["quality"]["metrics"] is None
    summary = detail["external_dispatch"]
    assert summary["counts"]["not_started"] == 1
    if failure in {"timeout", "incomplete"}:
        assert summary["unknown_outcome_count"] == 1
        assert summary["usage_cost_usd"] is None
    else:
        assert summary["counts"]["response_received"] == 1


def test_cancel_after_response_saves_receipt_without_publishing(workspace, transport, monkeypatch):
    reference, candidate, preview = configured(workspace)
    trial = launch(workspace, reference, candidate, preview)
    original = multimodal_provider._request

    def cancelled_transport(*args, **kwargs):
        response = original(*args, **kwargs)
        workspace[0].update("jobs", trial["job_id"], {"cancel_requested": True})
        return response

    monkeypatch.setattr(multimodal_provider, "_request", cancelled_transport)
    result = run(
        workspace, trial, lambda: workspace[0].get("jobs", trial["job_id"])["cancel_requested"]
    )
    detail = benchmark_trial_detail(workspace[0], trial["id"])
    assert result["cancelled"] and len(transport["requests"]) == 1
    assert detail["external_dispatch"]["counts"]["response_received"] == 1
    assert detail["latency"]["measured_count"] == 1
    assert all(row["result"] is None for row in detail["outputs"])
    assert any(row["raw_response"] is not None for row in detail["outputs"])


def test_tampered_preview_payload_fails_before_network(workspace, transport):
    reference, candidate, preview = configured(workspace)
    trial = launch(workspace, reference, candidate, preview)
    altered = deepcopy(trial["config"])
    altered["external_plan"]["requests"][0]["input"]["prompt"] += " changed"
    workspace[0].update("benchmark_trials", trial["id"], {"config": altered})
    with pytest.raises(ValueError, match="approved"):
        run(workspace, trial)
    assert not transport["requests"]
