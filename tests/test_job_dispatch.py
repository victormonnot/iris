"""Durable dispatch receipts using synthetic images and in-memory providers only."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Barrier, Event

import pytest
import test_video_reviews as video_fixtures
from test_assistance import READY, FixtureReviewer, prepare, run
from test_assistance import workspace as workspace
from test_assistance_previews import consent, preview, queue
from test_assistance_previews import remote_workspace as remote_workspace
from test_remote_provider import CANDIDATES, CONFIG, KEY, run_review
from test_remote_provider import transport as transport
from test_video_reviews import FixtureReviewer as VideoFixture
from test_video_reviews import prepare as prepare_video
from test_video_reviews import start as start_video

from iris import assistance_provider, remote_provider
from iris.assistance import run_assistance
from iris.assistance_provider import ProviderResponseError
from iris.job_dispatch import (
    DispatchConflict,
    claim_dispatch,
    dispatch_summary,
    mark_dispatched,
    reconcile_dispatches,
    update_dispatch_record,
)
from iris.store import now
from iris.video_reviews import run_video_review

video_workspace = video_fixtures.workspace


def remote_request(workspace):
    quote = preview(workspace)
    response = queue(workspace, consent(quote))
    assert response.status_code == 202, response.text
    job = response.json()
    return job, job["params"]["assistance_id"]


def receipt(store, job):
    return dispatch_summary(store, store.get("jobs", job["id"]))


def test_concurrent_execution_dispatches_once_and_does_not_overwrite_evidence(workspace):
    _, store, _, _ = workspace
    _, job, record_id = prepare(workspace)
    entered, release = Event(), Event()
    calls = []

    class Blocking(FixtureReviewer):
        def review(self, *args, **kwargs):
            calls.append(record_id)
            assert receipt(store, job)["state"] == "dispatching"
            entered.set()
            assert release.wait(5)
            return super().review(*args, **kwargs)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(run, store, record_id, factory=Blocking)
        try:
            assert entered.wait(5)
            second = executor.submit(run, store, record_id, factory=Blocking)
            with pytest.raises(DispatchConflict):
                second.result(timeout=5)
            assert store.get("assistance_records", record_id)["error"] is None
        finally:
            release.set()
        assert first.result(timeout=5)["suggestions_created"] == 1
    assert calls == [record_id]
    assert len(store.list("annotation_suggestions")) == 1
    assert receipt(store, job)["state"] == "response_received"


def test_crash_after_dispatch_is_unknown_after_restart_and_cannot_be_replayed(remote_workspace):
    _, store, jobs, _ = remote_workspace
    job, record_id = remote_request(remote_workspace)
    store.update("jobs", job["id"], {"status": "running", "started_at": now()})
    calls = []

    class SimulatedProcessDeath(BaseException):
        pass

    class Crash:
        def __init__(self, config, expected_images):
            self.metadata = {**config, "fixture": True}

        def review(self, *args, **kwargs):
            calls.append(record_id)
            assert receipt(store, job)["state"] == "dispatching"
            raise SimulatedProcessDeath()

    with pytest.raises(SimulatedProcessDeath):
        run(store, record_id, factory=Crash)
    assert receipt(store, job)["state"] == "dispatching"
    jobs._interrupt_unfinished()
    assert receipt(store, job)["state"] == "outcome_unknown"
    saved = store.get("assistance_records", record_id)
    assert saved["raw_response"] is None
    assert saved["metadata"]["fixture"] is True
    reconcile_dispatches(store)
    assert store.get("assistance_records", record_id) == saved
    with pytest.raises(DispatchConflict):
        run(store, record_id, factory=Crash)
    assert calls == [record_id]
    assert not store.list("annotation_suggestions")


def test_two_executors_that_both_pass_preflight_claim_only_one_attempt(workspace):
    _, store, _, _ = workspace
    _, job, record_id = prepare(workspace)
    ready = Barrier(2)
    calls = []

    class Once(FixtureReviewer):
        def review(self, *args, **kwargs):
            calls.append("one provider call")
            return super().review(*args, **kwargs)

    def execute():
        try:
            return run_assistance(
                store,
                record_id,
                lambda value, _: ready.wait(timeout=5) if value == 0.1 else None,
                lambda: False,
                reviewer_factory=Once,
            )
        except DispatchConflict as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as executor:
        first, second = executor.submit(execute), executor.submit(execute)
        results = [first.result(timeout=10), second.result(timeout=10)]
    assert sum(isinstance(result, DispatchConflict) for result in results) == 1
    assert calls == ["one provider call"]
    assert len(store.list("annotation_suggestions")) == 1
    assert store.get("assistance_records", record_id)["error"] is None
    assert receipt(store, job)["state"] == "response_received"


@pytest.mark.parametrize("complete", [False, True])
def test_external_error_distinguishes_raw_transport_stub_from_received_response(
    remote_workspace, complete
):
    _, store, _, _ = remote_workspace
    job, record_id = remote_request(remote_workspace)

    class Failed:
        def __init__(self, config, expected_images):
            self.metadata = {**config, "request_id": "fixture-received" if complete else None}

        def review(self, *args, **kwargs):
            error = ProviderResponseError(
                "Fixture invalid response" if complete else "Fixture timeout",
                raw_response={"error": "invalid"}
                if complete
                else {"http_status": None, "body": "", "truncated": False},
                metadata={**self.metadata, "dispatch": {"state": "not_started"}},
                prompt="Saved fixture prompt",
            )
            error.response_received = complete
            raise error

    with pytest.raises(ProviderResponseError):
        run(store, record_id, factory=Failed)
    saved = store.get("assistance_records", record_id)
    summary = receipt(store, job)
    assert summary["state"] == ("response_received" if complete else "outcome_unknown")
    assert summary["external"] is True
    assert summary["attempted_at"]
    assert bool(summary["response_received_at"]) is complete
    assert summary.get("request_id") == ("fixture-received" if complete else None)
    assert saved["raw_response"] is not None and saved["prompt"] == "Saved fixture prompt"
    assert saved["metadata"]["dispatch"]["attempt_id"]
    assert not store.list("annotation_suggestions")
    with pytest.raises(ValueError, match="immutable"):
        run(store, record_id, factory=Failed)
    assert store.get("assistance_records", record_id) == saved


def test_cancelled_and_failed_preflight_are_known_unsent(remote_workspace):
    _, store, jobs, _ = remote_workspace
    job, record_id = remote_request(remote_workspace)
    store.update("jobs", job["id"], {"status": "running", "started_at": now()})
    assert receipt(store, job)["state"] == "not_started"

    def broken_factory(**kwargs):
        raise ValueError("Fixture invalid local configuration")

    with pytest.raises(ValueError, match="invalid local"):
        run(store, record_id, factory=broken_factory)
    assert receipt(store, job)["state"] == "not_started"
    assert receipt(store, job)["attempted_at"] is None
    jobs.cancel(job["id"])
    assert run(store, record_id, cancelled=lambda: True)["cancelled"]
    assert receipt(store, job)["state"] == "not_started"


def test_claim_and_dispatch_check_committed_cancellation_and_preserve_metadata(workspace):
    _, store, jobs, _ = workspace
    _, job, record_id = prepare(workspace)
    update_dispatch_record(
        store,
        "assistance_records",
        record_id,
        {"metadata": {"fixture": "kept", "dispatch": {"state": "outcome_unknown"}}},
    )
    attempt = claim_dispatch(store, "assistance_records", record_id)
    with pytest.raises(DispatchConflict, match="already claimed"):
        update_dispatch_record(
            store, "assistance_records", record_id, {"error": "Competing preflight failure"}
        )
    assert receipt(store, job)["state"] == "not_started"
    jobs.cancel(job["id"])
    with pytest.raises(DispatchConflict, match="stopped"):
        mark_dispatched(store, "assistance_records", record_id, attempt)
    assert receipt(store, job)["state"] == "not_started"
    assert store.get("assistance_records", record_id)["metadata"]["fixture"] == "kept"
    assert store.get("assistance_records", record_id)["error"] is None


def test_late_preflight_failure_cannot_write_into_a_successful_attempt(workspace):
    _, store, _, _ = workspace
    _, job, record_id = prepare(workspace)
    entered, release = Event(), Event()

    def failed_factory(config):
        entered.set()
        assert release.wait(5)
        raise ValueError("Competing preflight failure")

    with ThreadPoolExecutor(max_workers=1) as executor:
        losing = executor.submit(run, store, record_id, factory=failed_factory)
        try:
            assert entered.wait(5)
            assert run(store, record_id)["suggestions_created"] == 1
            saved = store.get("assistance_records", record_id)
        finally:
            release.set()
        with pytest.raises(DispatchConflict):
            losing.result(timeout=5)
    assert store.get("assistance_records", record_id) == saved
    assert receipt(store, job)["state"] == "response_received"
    assert saved["error"] is None


@pytest.mark.parametrize("evidence", ["started", "raw_stub", "error", "bad_receipt"])
def test_legacy_and_corrupt_external_receipts_remain_conservative(remote_workspace, evidence):
    _, store, _, _ = remote_workspace
    job, record_id = remote_request(remote_workspace)
    store.update("assistance_records", record_id, {"metadata": {"historic": "preserved"}})
    assert receipt(store, job)["state"] == "not_started"
    if evidence == "started":
        store.update("jobs", job["id"], {"started_at": now()})
    elif evidence == "raw_stub":
        store.update("assistance_records", record_id, {"raw_response": {"http_status": None}})
    elif evidence == "error":
        store.update("assistance_records", record_id, {"error": "historic failure"})
    else:
        store.update("assistance_records", record_id, {"metadata": {"dispatch": "invalid"}})
    store.update("jobs", job["id"], {"status": "interrupted"})
    summary = receipt(store, job)
    assert summary["state"] == "outcome_unknown" and summary["external"]
    assert summary["response_received_at"] is None
    before = store.get("assistance_records", record_id)
    reconcile_dispatches(store)
    after = store.get("assistance_records", record_id)
    assert after["raw_response"] == before["raw_response"]
    assert after["config"] == before["config"] and after["error"] == before["error"]
    assert receipt(store, job) == summary


@pytest.mark.parametrize("complete", [False, True])
def test_video_receipt_keeps_consent_source_and_raw_evidence(video_workspace, complete):
    store, _, _ = video_workspace
    quote = prepare_video(video_workspace, provider="alibaba")
    job = start_video(
        video_workspace,
        quote,
        allow_external=True,
        max_cost_usd=quote["config"]["estimated_cost"]["upper_bound_usd"],
    )
    original = deepcopy(store.get("video_reviews", quote["id"])["metadata"])

    class Failed(VideoFixture):
        def review(self, *args, **kwargs):
            assert receipt(store, job)["state"] == "dispatching"
            error = ProviderResponseError(
                "Fixture video failure",
                raw_response={"fixture": "received"} if complete else {"http_status": None},
                metadata={**self.metadata, "request_id": "video-fixture"},
            )
            error.response_received = complete
            raise error

    with pytest.raises(ProviderResponseError):
        run_video_review(store, quote["id"], lambda *_: None, lambda: False, Failed)
    saved = store.get("video_reviews", quote["id"])
    assert saved["metadata"]["consent"] == original["consent"]
    assert saved["metadata"]["preview_sha256"] == original["preview_sha256"]
    assert saved["metadata"]["attempted_at"] == receipt(store, job)["attempted_at"]
    assert receipt(store, job)["state"] == ("response_received" if complete else "outcome_unknown")
    assert saved["result"] is None
    with pytest.raises(DispatchConflict):
        run_video_review(store, quote["id"], lambda *_: None, lambda: False, Failed)
    assert store.get("video_reviews", quote["id"]) == saved


@pytest.mark.parametrize("failure", ["timeout", "http", "invalid_json", "null", "truncated"])
def test_remote_adapter_marks_only_complete_http_responses_received(transport, failure):
    if failure == "timeout":
        transport["error"] = TimeoutError("Fixture connection interrupted")
    elif failure == "http":
        transport["status"] = 500
        transport["response"] = {"error": "fixture"}
    elif failure == "invalid_json":
        transport["response"] = b"not-json"
    elif failure == "null":
        transport["response"] = b"null"
    else:
        transport["response"] = b"x" * (remote_provider.MAX_RESPONSE_BYTES + 1)
    with pytest.raises(ProviderResponseError) as caught:
        run_review()
    assert caught.value.response_received is (failure in {"http", "invalid_json", "null"})
    assert caught.value.raw_response is not None
    if failure == "null":
        assert caught.value.raw_response["body"] == "null"
    assert len(transport["requests"]) == 1
    assert KEY not in str(caught.value)


def test_provider_dispatch_callback_runs_before_post_and_preflight_can_block_it(transport):
    from PIL import Image

    calls = []
    reviewer = remote_provider.AlibabaReviewer(CONFIG)

    def before_dispatch():
        assert transport["requests"] == []
        calls.append("committed")

    reviewer.before_dispatch = before_dispatch
    reviewer.review(Image.new("RGB", (80, 80)), CANDIDATES)
    assert calls == ["committed"] and len(transport["requests"]) == 1
    reviewer = remote_provider.AlibabaReviewer(CONFIG, expected_images=[b"changed fixture"])
    reviewer.before_dispatch = lambda: pytest.fail("Preflight must precede dispatch")
    with pytest.raises(ValueError, match="approved preview"):
        reviewer.review(Image.new("RGB", (80, 80)), CANDIDATES)
    assert len(transport["requests"]) == 1


def test_content_length_short_read_is_not_a_complete_response(transport, monkeypatch):
    class PartialResponse:
        status = 200
        fp = None
        length = 97
        chunks = iter([b"{}", b""])

        def read1(self, size):
            return next(self.chunks)

    monkeypatch.setattr(
        remote_provider.http.client.HTTPSConnection, "getresponse", lambda self: PartialResponse()
    )
    with pytest.raises(ProviderResponseError) as caught:
        run_review()
    assert caught.value.response_received is False
    assert caught.value.raw_response["body"] == "{}"
    assert len(transport["requests"]) == 1


@pytest.mark.parametrize("failure", ["timeout", "http", "invalid_json", "schema"])
def test_local_adapter_receipt_tracks_generation_response_only(transport, monkeypatch, failure):
    from PIL import Image

    monkeypatch.setattr(assistance_provider, "provider_status", lambda _: deepcopy(READY))
    monkeypatch.setattr(
        assistance_provider.http.client,
        "HTTPConnection",
        remote_provider.http.client.HTTPSConnection,
    )
    if failure == "timeout":
        transport["error"] = TimeoutError("Fixture interrupted")
    elif failure == "http":
        transport["status"] = 500
    elif failure == "invalid_json":
        transport["response"] = b"bad-json"
    else:
        transport["response"] = {"model": READY["model"], "done": False}
    calls = []
    reviewer = assistance_provider.OllamaReviewer({k: READY[k] for k in ("endpoint", "model")})
    reviewer.before_dispatch = lambda: calls.append("attempted")
    with pytest.raises(ProviderResponseError) as caught:
        reviewer.review(Image.new("RGB", (80, 80)), CANDIDATES)
    assert caught.value.response_received is (failure != "timeout")
    assert calls == ["attempted"]
    assert len(transport["requests"]) == 1


def test_local_identity_probe_response_is_not_generation_evidence(monkeypatch):
    from PIL import Image

    monkeypatch.setattr(assistance_provider, "provider_status", lambda _: deepcopy(READY))
    reviewer = assistance_provider.OllamaReviewer({k: READY[k] for k in ("endpoint", "model")})

    def changed_identity():
        error = ProviderResponseError("Fixture identity response rejected", raw_response={})
        error.response_received = True
        raise error

    monkeypatch.setattr(reviewer, "_check_identity", changed_identity)
    reviewer.before_dispatch = lambda: pytest.fail("Identity check must precede generation")
    with pytest.raises(ProviderResponseError) as caught:
        reviewer.review(Image.new("RGB", (80, 80)), CANDIDATES)
    assert caught.value.response_received is False
