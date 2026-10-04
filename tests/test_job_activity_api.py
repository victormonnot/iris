"""Project job history exposes partial evidence without resending or loading models."""

import pytest
from test_assistance_batches_api import create, payload
from test_assistance_batches_api import workspace as api_workspace

from iris.store import new_id, now

workspace = api_workspace


def test_job_detail_retains_partial_proposals_and_unknown_external_outcome(workspace, monkeypatch):
    client, store, session, frames = workspace
    batch = create(client, session, payload(frames))
    identifier = batch["job_ids"][0]
    record = store.list("assistance_records", job_id=identifier)[0]
    config = {**record["config"], "provider": {"provider": "alibaba", "model": "fixture-only"}}
    # An old transport-error envelope is evidence of an uncertain call, not a
    # confirmed provider response. No external request is executed by this test.
    store.update(
        "assistance_records",
        record["id"],
        {"config": config, "raw_response": {"http_status": None, "body": ""}},
    )
    store.update(
        "jobs",
        identifier,
        {"status": "interrupted", "started_at": now(), "finished_at": now(), "progress": 0.4},
    )
    store.insert(
        "annotation_suggestions",
        {
            "id": new_id(),
            "frame_id": frames[0]["id"],
            "job_id": identifier,
            "kind": "multimodal",
            "label": "person",
            "box": [2, 3, 30, 35],
            "metadata": {"synthetic": True},
            "created_at": now(),
        },
    )
    before = {table: store.list(table) for table in store.columns}

    def unexpected(*args, **kwargs):
        pytest.fail("Job history must not contact or load a provider")

    monkeypatch.setattr("iris.assistance.provider_status", unexpected)
    monkeypatch.setattr("iris.remote_provider.provider_status", unexpected)
    response = client.get(f"/api/jobs/{identifier}")
    assert response.status_code == 200, response.text
    detail = response.json()
    assert detail["job"]["status"] == "interrupted"
    assert detail["context"]["session_id"] == session["id"]
    assert detail["context"]["batch_id"] == batch["id"]
    counts = {row["kind"]: row["count"] for row in detail["artifacts"]}
    assert counts == {"suggestions": 1, "raw_response": 1}
    assert detail["dispatch"]["external"] is True
    assert detail["dispatch"]["state"] == "outcome_unknown"
    assert detail["recovery"]["can_check"] is False
    assert detail["next_action"]["workspace"] == "annotation"
    assert {table: store.list(table) for table in before} == before


def test_job_detail_and_recovery_are_project_scoped(workspace):
    client, store, session, frames = workspace
    batch = create(client, session, payload(frames))
    identifier = batch["job_ids"][0]
    other = client.post("/api/projects", json={"name": "Unrelated"}).json()["id"]
    assert client.get("/api/jobs", params={"project_id": other}).json() == []
    for suffix in ("", "/recovery"):
        assert (
            client.get(f"/api/jobs/{identifier}{suffix}", params={"project_id": other}).status_code
            == 404
        )
    assert (
        client.post(
            f"/api/jobs/{identifier}/recover",
            params={"project_id": other},
            json={"fingerprint": "a" * 64},
        ).status_code
        == 404
    )
    assert client.get("/api/jobs/missing").status_code == 404
    assert len(store.list("jobs")) == 2


def test_legacy_reverse_assistance_reference_is_readable(workspace):
    client, store, session, frames = workspace
    batch = create(client, session, payload(frames))
    identifier = batch["job_ids"][0]
    store.update("jobs", identifier, {"params": {}, "status": "cancelled"})
    response = client.get(f"/api/jobs/{identifier}")
    assert response.status_code == 200, response.text
    assert response.json()["context"]["session_id"] == session["id"]
    assert response.json()["dispatch"]["state"] == "not_started"


def test_basic_extraction_partial_frame_count_is_stored_evidence(workspace):
    client, store, session, frames = workspace
    identifier = new_id()
    job = store.insert(
        "jobs",
        {
            "id": identifier,
            "kind": "extract",
            "status": "interrupted",
            "params": {"asset_id": frames[0]["asset_id"], "config": {}},
            "created_at": now(),
            "finished_at": now(),
            "progress": 0.3,
        },
    )
    store.update("frames", frames[0]["id"], {"extraction": {"job_id": job["id"]}})
    response = client.get(f"/api/jobs/{identifier}")
    assert response.status_code == 200, response.text
    detail = response.json()
    assert detail["artifacts"][0]["count"] == 1
    assert detail["dispatch"] is None and detail["recovery"]["can_check"] is False
    assert detail["next_action"]["workspace"] == "intake"
    assert (
        client.post(
            f"/api/jobs/{identifier}/recover",
            json={"fingerprint": "a" * 64, "allow_external": True},
        ).status_code
        == 422
    )
