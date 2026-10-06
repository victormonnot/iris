"""Hosted batch accounting/recovery and human review using synthetic provider replies."""

from copy import deepcopy

import pytest
from test_review_queue_api import client as review_client
from test_review_queue_api import session_with_frames

from iris import dinox_batches as batches
from iris import dinox_provider as provider
from iris.annotations import get_annotation
from iris.store import now

client = review_client


@pytest.fixture
def setup(client, monkeypatch):
    monkeypatch.setattr(
        provider,
        "provider_status",
        lambda: {
            "status": "ready",
            "provider": "dinox",
            "key_configured": True,
        },
    )
    session, frames = session_with_frames(client, count=2)
    return client.app.state.store, client.app.state.jobs, session, frames


def options(setup):
    return {"frame_ids": [f["id"] for f in setup[3]], "threshold": 0.25}


def queue(setup, **changes):
    store, jobs, session, _ = setup
    settings = options(setup)
    preview = batches.preview_batch(store, session["id"], **settings)
    return batches.create_batch(
        store,
        jobs,
        session["id"],
        name="Synthetic cloud batch",
        expected_fingerprint=preview["fingerprint"],
        approve_external=True,
        max_cost_cny=0.3,
        **settings,
        **changes,
    )


def fake_provider(monkeypatch, *, output=None, on_submit=None, on_poll=None):
    calls = {"post": 0, "get": 0}
    if output is None:
        output = {"objects": [{"category": "person", "bbox": [2, 3, 20, 40], "score": 0.8}]}

    def submit(config, png, *, idempotency_key):
        calls["post"] += 1
        assert png.startswith(b"\x89PNG")
        if on_submit:
            on_submit()
        return {"task_id": idempotency_key, "raw_response": {"code": 0}}

    def poll(task_id):
        calls["get"] += 1
        if on_poll:
            return on_poll(task_id)
        return {"status": "succeeded", "raw_response": {"code": 0}, "result": deepcopy(output)}

    monkeypatch.setattr(provider, "submit", submit)
    monkeypatch.setattr(provider, "poll", poll)
    return calls


def execute(setup, batch, cancelled=lambda: False):
    store = setup[0]
    store.update("jobs", batch["job_id"], {"status": "running"})
    result = batches.run_batch(store, batch["id"], lambda *_: None, cancelled)
    store.update(
        "jobs",
        batch["job_id"],
        {
            "status": "cancelled" if cancelled() else "succeeded",
            "result": result,
            "finished_at": now(),
        },
    )
    return batches.batch_detail(store, batch["id"])


def test_preview_is_readonly_and_confirmation_is_idempotent(setup):
    store, jobs, session, _ = setup
    before = {table: store.list(table) for table in store.columns}
    plan = batches.preview_batch(store, session["id"], **options(setup))
    assert plan["request_count"] == 2 and plan["estimate"]["total"] == 0.3
    assert {table: store.list(table) for table in store.columns} == before
    payload = {
        **options(setup),
        "name": "Explicit",
        "expected_fingerprint": plan["fingerprint"],
        "approve_external": True,
        "max_cost_cny": 0.3,
    }
    batch = batches.create_batch(store, jobs, session["id"], **payload)
    assert batches.create_batch(store, jobs, session["id"], **payload)["id"] == batch["id"]
    assert len(store.list("dinox_requests")) == 2 and len(store.list("jobs")) == 1
    assert not store.list("annotation_revisions")


@pytest.mark.parametrize(
    "approved,budget", [(False, 0.3), (True, 0.29), (True, float("nan")), (True, True)]
)
def test_explicit_budget_and_external_processing_required(setup, approved, budget):
    store, jobs, session, _ = setup
    plan = batches.preview_batch(store, session["id"], **options(setup))
    with pytest.raises(ValueError):
        batches.create_batch(
            store,
            jobs,
            session["id"],
            name="Unsafe",
            **options(setup),
            expected_fingerprint=plan["fingerprint"],
            approve_external=approved,
            max_cost_cny=budget,
        )
    assert not store.list("jobs")


def test_real_workflow_reuses_without_post_or_overwriting_human_decisions(
    setup, monkeypatch, client
):
    calls = fake_provider(monkeypatch)
    batch = execute(setup, queue(setup))
    store, _, session, frames = setup
    assert calls == {"post": 2, "get": 2}
    assert batch["counts"] == {"total": 2, "ready": 2, "issues": 0, "proposals": 2}
    assert batch["cost"]["estimated_cny"] == 0.3
    assert not store.list("annotation_revisions")
    document = get_annotation(store, frames[0]["id"])
    suggestion = document["suggestions"][0]
    response = client.put(
        f"/api/frames/{frames[0]['id']}/annotation",
        json={
            "expected_revision": 0,
            "boxes": [
                {
                    "id": "human-box",
                    "label": "person",
                    "box": [3, 4, 21, 41],
                    "suggestion_id": suggestion["id"],
                }
            ],
            "decisions": {suggestion["id"]: "corrected"},
            "status": "validated",
            "reviewer": "Test human",
            "notes": "Corrected",
        },
    )
    assert response.status_code == 200, response.text
    revision = store.list("annotation_revisions")
    plan = batches.preview_batch(store, session["id"], **options(setup))
    assert plan["reuse_count"] == 2 and plan["estimate"]["total"] == 0
    again = execute(setup, queue(setup))
    assert again["cost"]["new_requests"] == 0
    assert calls == {"post": 2, "get": 2}
    assert len(store.list("annotation_suggestions")) == 2
    assert store.list("annotation_revisions") == revision
    assert get_annotation(store, frames[0]["id"])["boxes"][0]["review_state"] == "corrected"


def test_empty_response_is_not_a_human_negative(setup, monkeypatch):
    fake_provider(monkeypatch, output={"objects": []})
    batch = execute(setup, queue(setup))
    assert {f["state"] for f in batch["frames"]} == {"no_proposals"}
    assert not setup[0].list("annotation_suggestions")
    assert not setup[0].list("annotation_revisions")


def test_ambiguous_post_is_not_retried_after_new_preview(setup, monkeypatch):
    def timeout():
        raise provider.DinoXTransportError("transport_error")

    calls = fake_provider(monkeypatch, on_submit=timeout)
    batch = execute(setup, queue(setup))
    plan = batches.preview_batch(setup[0], setup[2]["id"], **options(setup))
    assert calls["post"] == 1
    assert batch["requests"][0]["state"] == "outcome_unknown"
    assert plan["frames"][0]["action"] == "blocked"
    assert plan["frames"][1]["action"] == "submit"
    assert plan["request_count"] == 1 and plan["excluded_count"] == 1


def test_submitted_task_is_polled_without_resending_after_cancellation(setup, monkeypatch):
    cancelled = {"value": False}
    calls = fake_provider(monkeypatch, on_submit=lambda: cancelled.update(value=True))
    first = execute(setup, queue(setup), lambda: cancelled["value"])
    assert calls == {"post": 1, "get": 0}
    assert first["requests"][0]["state"] == "submitted"
    cancelled["value"] = False
    plan = batches.preview_batch(setup[0], setup[2]["id"], **options(setup))
    assert plan["poll_count"] == 1 and plan["request_count"] == 1
    calls = fake_provider(monkeypatch)
    result = execute(setup, queue(setup))
    assert calls == {"post": 1, "get": 2}
    assert result["counts"]["ready"] == 2


def test_raw_response_saved_before_validation_failure(setup, monkeypatch):
    calls = fake_provider(
        monkeypatch,
        output={"objects": [{"category": "unknown", "bbox": [1, 2, 4, 6], "score": 0.9}]},
    )
    first = execute(setup, queue(setup))
    request = first["requests"][0]
    assert request["state"] == "response_received" and request["raw_response"]
    assert not setup[0].list("annotation_suggestions")
    execute(setup, queue(setup))
    assert calls["post"] == 1


def test_human_edit_during_remote_request_preserves_response_without_publication(
    setup, monkeypatch, client
):
    frame_id = setup[3][0]["id"]

    def edit():
        response = client.put(
            f"/api/frames/{frame_id}/annotation",
            json={
                "expected_revision": 0,
                "boxes": [],
                "decisions": {},
                "status": "draft",
                "notes": "Concurrent edit",
            },
        )
        assert response.status_code == 200

    fake_provider(monkeypatch, on_submit=edit)
    result = execute(setup, queue(setup))
    assert result["requests"][0]["state"] == "succeeded"
    assert result["frames"][0]["state"] == "failed"
    assert not setup[0].list("annotation_suggestions")
    assert setup[0].list("annotation_revisions")[0]["notes"] == "Concurrent edit"


def test_api_project_boundary_stale_inputs_and_job_context(setup, client):
    store, _, session, frames = setup
    url = f"/api/sessions/{session['id']}/dinox-batches"
    plan = client.post(url + "/preview", json=options(setup)).json()
    payload = {
        **options(setup),
        "name": "API",
        "expected_fingerprint": plan["fingerprint"],
        "approve_external": True,
        "max_cost_cny": 0.3,
    }
    response = client.post(url, json=payload)
    assert response.status_code == 202, response.text
    batch = response.json()
    assert client.post(url, json=payload).json()["id"] == batch["id"]
    activity = client.get(f"/api/jobs/{batch['job_id']}")
    assert activity.status_code == 200, activity.text
    assert activity.json()["next_action"]["workspace"] == "annotation"
    assert activity.json()["context"]["batch_id"] == batch["id"]
    other = client.post("/api/projects", json={"name": "Other"}).json()["id"]
    for endpoint in [url, f"/api/dinox-batches/{batch['id']}", f"/api/jobs/{batch['job_id']}"]:
        assert client.get(endpoint, params={"project_id": other}).status_code == 404
    assert client.post(url, json=payload, params={"project_id": other}).status_code == 404
    store.update("jobs", batch["job_id"], {"status": "cancelled"})
    plan = client.post(url + "/preview", json=options(setup)).json()
    store.update("frames", frames[0]["id"], {"taxonomy_id": "changed"})
    assert client.post(
        url, json={**payload, "expected_fingerprint": plan["fingerprint"]}
    ).status_code in {404, 409}


def test_restart_reconciles_ambiguous_intent_but_keeps_known_task(setup):
    batch = queue(setup)
    store = setup[0]
    first, second = batch["requests"]
    store.update("dinox_requests", first["id"], {"state": "dispatching"})
    store.update("dinox_requests", second["id"], {"state": "submitted", "task_id": "known-task"})
    setup[1]._interrupt_unfinished()
    assert store.get("dinox_requests", first["id"])["state"] == "outcome_unknown"
    assert store.get("dinox_requests", second["id"])["task_id"] == "known-task"


@pytest.mark.parametrize(
    "change",
    [
        {"threshold": True},
        {"threshold": "0.25"},
        {"frame_ids": []},
        {"endpoint": "https://elsewhere"},
    ],
)
def test_api_rejects_unsafe_options(setup, client, change):
    url = f"/api/sessions/{setup[2]['id']}/dinox-batches/preview"
    assert client.post(url, json={**options(setup), **change}).status_code == 422
