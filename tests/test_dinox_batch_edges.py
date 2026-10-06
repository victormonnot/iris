"""Hosted recovery, concurrent claims and credential HTTP boundary regression tests."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Barrier

import pytest
from test_dinox_batches import client as batch_client
from test_dinox_batches import execute, fake_provider, options, queue
from test_dinox_batches import setup as batch_setup

from iris import dinox_batches as batches
from iris import dinox_provider as provider
from iris.annotations import get_annotation

client = batch_client
setup = batch_setup


def test_known_auth_rejection_requires_new_approval_and_keeps_old_receipt(setup, monkeypatch):
    def rejected():
        raise provider.DinoXResponseError(
            "http_status",
            raw_response={"http_status": 401, "body": {"code": 401}},
            outcome_unknown=False,
        )

    calls = fake_provider(monkeypatch, on_submit=rejected)
    first = execute(setup, queue(setup))
    assert calls == {"post": 1, "get": 0}
    assert first["requests"][0]["state"] == "failed"
    first_request_id = first["requests"][0]["id"]
    store, jobs, session, _ = setup
    preview = batches.preview_batch(store, session["id"], **options(setup))
    assert preview["request_count"] == 2
    assert preview["estimate"]["total"] == 0.3
    with pytest.raises(ValueError, match="Confirm"):
        batches.create_batch(
            store,
            jobs,
            session["id"],
            name="Needs new approval",
            expected_fingerprint=preview["fingerprint"],
            max_cost_cny=0.3,
            **options(setup),
        )
    calls = fake_provider(monkeypatch)
    recovered = execute(setup, queue(setup))
    assert calls == {"post": 2, "get": 2}
    assert recovered["counts"]["ready"] == 2
    assert recovered["requests"][0]["id"] != first_request_id
    assert recovered["requests"][0]["metadata"]["previous_request_id"] == first_request_id
    assert store.get("dinox_requests", first_request_id)["state"] == "failed"


def test_two_workers_can_reserve_only_one_paid_intent(setup):
    batch = queue(setup)
    store = setup[0]
    store.update("jobs", batch["job_id"], {"status": "running"})
    identifier = batch["requests"][0]["id"]
    gate = Barrier(2)

    def claim():
        gate.wait(timeout=5)
        try:
            return batches._take_request(store, batch, identifier)["state"]
        except ValueError:
            return "already_owned"

    with ThreadPoolExecutor(max_workers=2) as pool:
        attempts = [pool.submit(claim) for _ in range(2)]
        outcomes = [attempt.result(timeout=10) for attempt in attempts]
    assert sorted(outcomes) == ["already_owned", "not_started"]
    assert store.get("dinox_batches", batch["id"])["metadata"]["attempted_count"] == 1
    assert store.get("dinox_requests", identifier)["state"] == "dispatching"


def test_claim_checks_frozen_budget_before_dispatch(setup):
    batch = queue(setup)
    store = setup[0]
    store.update("jobs", batch["job_id"], {"status": "running"})
    batch["config"]["max_cost_cny"] = 0.14
    identifier = batch["requests"][0]["id"]
    with pytest.raises(ValueError, match="budget"):
        batches._take_request(store, batch, identifier)
    assert store.get("dinox_requests", identifier)["state"] == "not_started"
    assert store.get("dinox_batches", batch["id"])["metadata"]["attempted_count"] == 0


def test_cached_result_must_match_saved_provider_evidence_before_publication(setup, monkeypatch):
    calls = fake_provider(monkeypatch)
    first = execute(setup, queue(setup))
    store = setup[0]
    request = first["requests"][0]
    changed = deepcopy(request["result"])
    changed["proposals"][0].update(id="invented", label="untrusted-class", box=[1, 1, 60, 45])
    store.update("dinox_requests", request["id"], {"result": changed})
    previous = store.list("annotation_suggestions")
    again = execute(setup, queue(setup))
    assert again["frames"][0]["state"] == "failed"
    assert calls == {"post": 2, "get": 2}
    assert store.list("annotation_suggestions") == previous


@pytest.mark.parametrize("decision", ["accepted", "rejected"])
def test_cached_reuse_preserves_explicit_human_decisions(setup, client, monkeypatch, decision):
    calls = fake_provider(monkeypatch)
    execute(setup, queue(setup))
    store, _, _, frames = setup
    frame = frames[0]
    suggestion = get_annotation(store, frame["id"])["suggestions"][0]
    boxes = (
        []
        if decision == "rejected"
        else [
            {
                "id": "reviewed-box",
                "label": suggestion["label"],
                "box": suggestion["box"],
                "suggestion_id": suggestion["id"],
            }
        ]
    )
    response = client.put(
        f"/api/frames/{frame['id']}/annotation",
        json={
            "expected_revision": 0,
            "boxes": boxes,
            "decisions": {suggestion["id"]: decision},
            "status": "validated",
            "reviewer": "Synthetic test reviewer",
        },
    )
    assert response.status_code == 200, response.text
    previous = store.list("annotation_revisions")
    again = execute(setup, queue(setup))
    assert again["cost"]["new_requests"] == 0
    assert calls == {"post": 2, "get": 2}
    assert store.list("annotation_revisions") == previous
    annotation = get_annotation(store, frame["id"])
    assert annotation["decisions"][suggestion["id"]] == decision
    assert len(annotation["boxes"]) == (decision == "accepted")


@pytest.fixture
def key_file(tmp_path, monkeypatch):
    monkeypatch.delenv(provider.KEY_ENV, raising=False)
    path = tmp_path / "private" / "iris" / "credentials.json"
    monkeypatch.setattr(provider, "CREDENTIAL_PATH", path)
    return path


def test_key_endpoint_never_echoes_secret_and_stores_outside_workspace(client, key_file):
    secret = "synthetic-private-cloud-key"
    saved = client.put("/api/dinox/key", json={"key": secret})
    assert saved.status_code == 200, saved.text
    assert secret not in saved.text
    assert saved.headers["cache-control"] == "no-store"
    assert saved.json()["key_source"] == "file"
    assert key_file.exists()
    assert not key_file.is_relative_to(client.app.state.store.root)
    status = client.get("/api/dinox/provider")
    assert secret not in status.text
    assert str(key_file) not in status.text
    assert client.delete("/api/dinox/key").json()["key_configured"] is False


@pytest.mark.parametrize(
    "payload",
    [
        {"key": "secret with whitespace"},
        {"key": {"secret": "synthetic-key"}},
        {"key": "synthetic-key", "extra": "synthetic-private-extra"},
        {"key": "synthetic-key" * 300},
        ["synthetic-key"],
    ],
)
def test_key_errors_are_generic_and_never_write(client, key_file, payload):
    response = client.put("/api/dinox/key", json=payload)
    assert response.status_code == 422
    assert response.json() == {"detail": "Unable to save this DINO-X key securely"}
    assert "synthetic" not in response.text and "whitespace" not in response.text
    assert not key_file.exists()


@pytest.mark.parametrize(
    "headers",
    [
        {"Origin": "https://external.example"},
        {"Origin": "null"},
        {"Origin": "http://127.0.0.1:9999"},
        {"Sec-Fetch-Site": "cross-site"},
    ],
)
@pytest.mark.parametrize("method", ["put", "delete"])
def test_cross_origin_credential_mutations_are_rejected(client, key_file, headers, method):
    provider.update_key("synthetic-original")
    original = key_file.read_bytes()
    kwargs = {"json": {"key": "synthetic-attacker"}} if method == "put" else {}
    response = getattr(client, method)("/api/dinox/key", headers=headers, **kwargs)
    assert response.status_code == 403
    assert key_file.read_bytes() == original


@pytest.mark.parametrize("budget", ["NaN", "Infinity", "-Infinity", "true"])
def test_nonfinite_or_boolean_api_budget_rejected_before_queue(setup, client, budget):
    import json

    store, _, session, _ = setup
    preview = batches.preview_batch(store, session["id"], **options(setup))
    content = (
        json.dumps(
            {
                **options(setup),
                "name": "Invalid budget",
                "expected_fingerprint": preview["fingerprint"],
                "approve_external": True,
            }
        )[:-1]
        + ', "max_cost_cny": '
        + budget
        + "}"
    )
    response = client.post(
        f"/api/sessions/{session['id']}/dinox-batches",
        content=content,
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422
    assert store.list("dinox_batches") == []
