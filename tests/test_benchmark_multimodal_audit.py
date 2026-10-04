"""Independent integration audit through the actual adapter with an offline HTTPS fixture."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Event

import pytest
from test_benchmark import freeze
from test_benchmark import workspace as workspace
from test_multimodal_provider import transport as transport

from iris import benchmark_multimodal, multimodal_provider
from iris.benchmark import BenchmarkConflict, create_benchmark_config, preview_benchmark_config
from iris.benchmark_dispatch import DispatchConflict, validate_external_trial
from iris.benchmark_runs import create_benchmark_trial, preview_benchmark_trial, run_benchmark_trial


def configure(workspace):
    store = workspace[0]
    reference = freeze(workspace)
    values = {"approach": "multimodal", "model_id": "gpt-6-astra"}
    preview = preview_benchmark_config(store, reference["id"], **values)
    candidate = create_benchmark_config(
        store,
        reference["id"],
        name="Independent API audit",
        expected_fingerprint=preview["fingerprint"],
        **values,
    )
    preview = preview_benchmark_trial(
        store, reference["id"], config_id=candidate["id"], role="tuning"
    )
    return reference, candidate, preview


def launch(workspace, configured=None):
    store, jobs = workspace[:2]
    reference, candidate, preview = configured or configure(workspace)
    trial = create_benchmark_trial(
        store,
        jobs,
        reference["id"],
        config_id=candidate["id"],
        role="tuning",
        expected_fingerprint=preview["fingerprint"],
        approve_external=True,
        max_cost_usd=preview["external_plan"]["estimate"]["upper_bound_usd"] + 0.01,
        preview_token=preview["preview_token"],
    )
    return reference, candidate, preview, trial


def test_actual_adapter_callbacks_publish_without_reference_inputs(workspace, transport):
    store = workspace[0]
    _, _, _, trial = launch(workspace)
    before = store.list("annotation_revisions")
    result = run_benchmark_trial(store, trial["id"], lambda *_: None, lambda: False)
    assert result["frames_ready"] == 2 and result["quality"]["complete"]
    assert len(transport["requests"]) == 2
    for _, _, payload, _ in transport["requests"]:
        assert b"withheld sentinel" not in payload
        assert b"Independent fixture author" not in payload
        assert b"human-box" not in payload
    for output in store.list("benchmark_outputs", trial_id=trial["id"]):
        assert output["metadata"]["dispatch"]["state"] == "response_received"
        assert output["raw_response"]["id"] == "resp-fixture"
        assert all(proposal["score"] is None for proposal in output["result"]["proposals"])
    assert store.list("annotation_revisions") == before
    assert not store.list("annotation_suggestions")


def test_server_restart_invalidates_unsubmitted_preview_without_sending(
    workspace, transport, monkeypatch
):
    store, jobs = workspace[:2]
    reference, candidate, preview = configure(workspace)
    monkeypatch.setattr(benchmark_multimodal, "_PREVIEW_SECRET", b"new-process-key")
    with pytest.raises(BenchmarkConflict, match="expired or changed"):
        create_benchmark_trial(
            store,
            jobs,
            reference["id"],
            config_id=candidate["id"],
            role="tuning",
            expected_fingerprint=preview["fingerprint"],
            approve_external=True,
            max_cost_usd=10,
            preview_token=preview["preview_token"],
        )
    assert not store.list("jobs") and not transport["requests"]


def test_cancel_during_http_keeps_receipt_and_never_sends_next_image(
    workspace, transport, monkeypatch
):
    store = workspace[0]
    _, _, _, trial = launch(workspace)
    connection = multimodal_provider.http.client.HTTPSConnection

    class Interrupted(connection):
        def request(self, *args, **kwargs):
            super().request(*args, **kwargs)
            store.update("jobs", trial["job_id"], {"cancel_requested": True})

    monkeypatch.setattr(multimodal_provider.http.client, "HTTPSConnection", Interrupted)
    result = run_benchmark_trial(store, trial["id"], lambda *_: None, lambda: False)
    outputs = store.list("benchmark_outputs", trial_id=trial["id"])
    assert len(transport["requests"]) == 1
    received = [
        row for row in outputs if row["metadata"]["dispatch"]["state"] == "response_received"
    ]
    assert len(received) == 1 and received[0]["raw_response"] is not None
    assert received[0]["result"] is None and received[0]["metadata"]["state"] == "cancelled"
    assert result["frames_ready"] == 0 and not result["quality"]["complete"]


def test_normalized_result_tampering_retains_raw_and_stops_before_second_send(
    workspace, transport, monkeypatch
):
    store = workspace[0]
    _, _, _, trial = launch(workspace)
    propose = multimodal_provider.OpenAIPreannotator.propose

    def altered(self, *args, **kwargs):
        response = propose(self, *args, **kwargs)
        response["result"]["proposals"][0]["box"][0] += 1
        return response

    monkeypatch.setattr(multimodal_provider.OpenAIPreannotator, "propose", altered)
    result = run_benchmark_trial(store, trial["id"], lambda *_: None, lambda: False)
    assert result["frames_issues"] == 1 and len(transport["requests"]) == 1
    outputs = store.list("benchmark_outputs", trial_id=trial["id"])
    failed = next(row for row in outputs if row["error"] is not None)
    assert failed["raw_response"] and failed["result"] is None
    assert "normalization differs" in failed["error"]


def test_concurrent_executor_cannot_dispatch_duplicate_paid_requests(
    workspace, transport, monkeypatch
):
    store = workspace[0]
    _, _, _, trial = launch(workspace)
    connection = multimodal_provider.http.client.HTTPSConnection
    entered, resume = Event(), Event()

    class Blocked(connection):
        def request(self, *args, **kwargs):
            super().request(*args, **kwargs)
            if len(transport["requests"]) == 1:
                entered.set()
                assert resume.wait(timeout=5)

    monkeypatch.setattr(multimodal_provider.http.client, "HTTPSConnection", Blocked)
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(run_benchmark_trial, store, trial["id"], lambda *_: None, lambda: False)
        assert entered.wait(timeout=5)
        try:
            with pytest.raises(DispatchConflict, match="already attempted"):
                run_benchmark_trial(store, trial["id"], lambda *_: None, lambda: False)
        finally:
            resume.set()
        assert first.result()["frames_ready"] == 2
    assert len(transport["requests"]) == 2


@pytest.mark.parametrize("mutation", ["consent", "fingerprint"])
def test_saved_external_plan_requires_its_explicit_consent_and_fingerprint(
    workspace, transport, mutation
):
    _, candidate, _, trial = launch(workspace)
    frozen = deepcopy(trial["config"])
    if mutation == "consent":
        frozen["external_plan"]["approval"]["allow_external"] = False
    else:
        frozen["external_plan"]["approval"]["fingerprint"] = "0" * 64
    with pytest.raises(ValueError):
        validate_external_trial(
            frozen,
            candidate["config"],
            [{"frame_id": frame_id} for frame_id in frozen["frame_ids"]],
        )
