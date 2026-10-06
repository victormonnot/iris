"""Project-scoped preannotation HTTP workflow with synthetic availability, no model calls."""

import pytest
from test_review_queue_api import client as review_client
from test_review_queue_api import session_with_frames

from iris.models import get_spec

client = review_client
MODEL = "fasterrcnn_mobilenet_v3_large_320_fpn"


@pytest.fixture
def request_data(client, monkeypatch):
    spec = {**get_spec(MODEL), "status": "ready", "weight_sha256": "a" * 64}
    monkeypatch.setattr("iris.inference.catalog", lambda _: [spec])
    monkeypatch.setattr("iris.app.catalog", lambda _: [spec])
    session, frames = session_with_frames(client)
    return session, frames, {"frame_ids": [frame["id"] for frame in frames], "model_id": MODEL}


def preview(client, session, payload):
    response = client.post(f"/api/sessions/{session['id']}/preannotations/preview", json=payload)
    assert response.status_code == 200, response.text
    return response.json()


def test_preview_confirmation_receipt_history_and_activity(client, request_data):
    session, frames, payload = request_data
    store = client.app.state.store
    before = {table: store.list(table) for table in store.columns}
    plan = preview(client, session, payload)
    assert plan["eligible_count"] == 3
    assert plan["work"]["total_forward_passes"] == 4
    assert {table: store.list(table) for table in store.columns} == before
    confirmation = {
        **payload,
        "name": "Synthetic direct proposal run",
        "expected_fingerprint": plan["fingerprint"],
    }
    endpoint = f"/api/sessions/{session['id']}/preannotations"
    response = client.post(endpoint, json=confirmation)
    assert response.status_code == 202, response.text
    receipt = response.json()
    again = client.post(endpoint, json=confirmation)
    assert again.status_code == 202 and again.json()["id"] == receipt["id"]
    assert len(store.list("jobs")) == len(store.list("comparisons")) == 1
    assert store.list("annotation_revisions") == store.list("annotation_suggestions") == []
    detail = client.get(f"/api/preannotations/{receipt['id']}").json()
    assert detail["job"]["status"] == "queued"
    assert detail["counts"]["total"] == 3
    assert {row["frame_id"] for row in detail["frames"]} == {frame["id"] for frame in frames}
    assert client.get(endpoint).json()[0]["id"] == receipt["id"]
    activity = client.get(f"/api/jobs/{receipt['job_id']}").json()
    assert activity["dispatch"] is None
    assert activity["next_action"]["workspace"] == "annotation"
    assert {artifact["kind"] for artifact in activity["artifacts"]} == {
        "predictions",
        "suggestions",
    }
    assert client.get("/api/preannotations/missing").status_code == 404


def test_stale_revision_returns_conflict_without_creating_work(client, request_data):
    session, frames, payload = request_data
    plan = preview(client, session, payload)
    response = client.put(
        f"/api/frames/{frames[0]['id']}/annotation",
        json={
            "expected_revision": 0,
            "boxes": [],
            "decisions": {},
            "status": "draft",
            "notes": "Saved after preview",
        },
    )
    assert response.status_code == 200, response.text
    response = client.post(
        f"/api/sessions/{session['id']}/preannotations",
        json={
            **payload,
            "name": "Stale",
            "expected_fingerprint": plan["fingerprint"],
        },
    )
    assert response.status_code == 409, response.text
    assert client.app.state.store.list("jobs") == []


def test_other_project_cannot_read_confirm_or_preview_request(client, request_data):
    session, _, payload = request_data
    plan = preview(client, session, payload)
    endpoint = f"/api/sessions/{session['id']}/preannotations"
    confirmation = {**payload, "name": "Synthetic", "expected_fingerprint": plan["fingerprint"]}
    receipt = client.post(endpoint, json=confirmation).json()
    other = client.post("/api/projects", json={"name": "Unrelated"}).json()["id"]
    params = {"project_id": other}
    for path in [
        endpoint,
        f"/api/preannotations/{receipt['id']}",
        f"/api/jobs/{receipt['job_id']}",
    ]:
        assert client.get(path, params=params).status_code == 404
    assert client.post(endpoint, params=params, json=confirmation).status_code == 404
    assert client.post(endpoint + "/preview", params=params, json=payload).status_code == 404
    assert len(client.app.state.store.list("jobs")) == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"threshold": True},
        {"threshold": "0.5"},
        {"threshold": 2},
        {"device": "remote"},
        {"frame_ids": []},
        {"provider": "alibaba"},
        {"inference_mode": "paired"},
        {"allow_external": True},
        {"tile_size": False},
    ],
)
def test_generation_rejects_unknown_unsafe_or_coerced_options(client, request_data, changes):
    session, _, payload = request_data
    response = client.post(
        f"/api/sessions/{session['id']}/preannotations/preview", json={**payload, **changes}
    )
    assert response.status_code == 422, response.text
    assert client.app.state.store.list("jobs") == []


def test_provider_catalog_only_advertises_implemented_capabilities(
    client, request_data, monkeypatch
):
    def unexpected(*args, **kwargs):
        pytest.fail("Capability catalogue must not probe or invoke an annotation provider")

    monkeypatch.setattr("iris.assistance_catalog.catalog", unexpected)
    monkeypatch.setattr("iris.assistance_provider._request", unexpected)
    monkeypatch.setattr("iris.dinox_provider.provider_status", unexpected)
    monkeypatch.setattr("iris.dinox_provider.submit", unexpected)
    monkeypatch.setattr("iris.dinox_provider.poll", unexpected)
    providers = {
        row["id"]: row for row in client.get("/api/preannotation-providers").json()["providers"]
    }
    assert set(providers) == {"local_detector", "ollama", "alibaba", "dinox"}
    assert providers["local_detector"]["models"][0]["id"] == MODEL
    assert providers["local_detector"]["capabilities"]["creates_boxes"] is True
    assert providers["ollama"]["capabilities"]["creates_boxes"] is False
    assert providers["ollama"]["capabilities"]["requires_candidates"] is True
    assert providers["alibaba"]["capabilities"]["execution"] == "external"
    assert providers["dinox"]["local"] is False
    assert providers["dinox"]["capabilities"]["execution"] == "external"
    assert providers["dinox"]["capabilities"]["creates_boxes"] is True
    assert providers["dinox"]["capabilities"]["requires_candidates"] is False
    assert providers["dinox"]["capabilities"]["class_support"] == "explicit_text_prompts"
    assert all(not row["capabilities"]["automatic_validation"] for row in providers.values())
