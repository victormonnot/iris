"""Local batch HTTP contracts, using generated images and no model calls."""

from copy import deepcopy

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from iris.annotations import save_annotation
from iris.app import create_app
from iris.media import import_asset
from iris.store import new_id, now

READY = {
    "provider": "ollama",
    "endpoint": "http://127.0.0.1:11434",
    "model": "synthetic-reviewer",
    "model_digest": "a" * 64,
    "status": "ready",
}


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr("iris.assistance_batches.provider_status", lambda *_: deepcopy(READY))
    app = create_app(tmp_path / "workspace", run_jobs=False)
    store = app.state.store
    session = store.insert(
        "sessions",
        {"id": new_id(), "name": "Synthetic batch", "scene_group": "fixture", "created_at": now()},
    )
    frames = []
    for index in range(3):
        source = tmp_path / f"synthetic-{index}.png"
        Image.new("RGB", (80, 60), (20 + index, 60, 100)).save(source)
        asset = import_asset(store, session["id"], source, source.name)
        frame = store.list("frames", asset_id=asset["id"])[0]
        store.update("frames", frame["id"], {"selected": True})
        frames.append(frame)
        if index < 2:
            save_annotation(
                store,
                frame["id"],
                expected_revision=0,
                boxes=[{"id": f"fixture-{index}", "label": "person", "box": [2, 3, 30, 35]}],
                decisions={},
                reviewer="Automated synthetic fixture",
            )
    with TestClient(app, base_url="http://127.0.0.1") as client:
        yield client, store, session, frames


def payload(frames):
    return {
        "frame_ids": [frame["id"] for frame in frames],
        "source": "annotations",
        "model": READY["model"],
        "threshold": 0.5,
    }


def endpoint(session):
    return f"/api/sessions/{session['id']}/assistance-batches"


def create(client, session, data):
    preview = client.post(endpoint(session) + "/preview", json=data)
    assert preview.status_code == 200, preview.text
    response = client.post(
        endpoint(session),
        json={
            **data,
            "name": "Synthetic local review",
            "expected_fingerprint": preview.json()["fingerprint"],
        },
    )
    assert response.status_code == 202, response.text
    return response.json()


def test_preview_exclusions_atomic_creation_and_persistent_cancel(workspace):
    client, store, session, frames = workspace
    before = store.list("annotation_revisions")
    data = payload(frames)
    preview = client.post(endpoint(session) + "/preview", json=data)
    assert preview.status_code == 200, preview.text
    assert preview.json()["eligible_count"] == 2
    assert preview.json()["excluded_count"] == 1
    assert preview.json()["frames"][2]["reason"]
    assert not store.list("jobs") and not store.list("assistance_batches")
    batch = create(client, session, data)
    assert len(batch["job_ids"]) == 2
    assert batch["counts"]["queued"] == 2
    assert batch["counts"]["total"] == 2
    assert batch["config"]["excluded"][0]["frame_id"] == frames[2]["id"]
    assert client.get(endpoint(session)).json() == [batch]
    assert client.get(f"/api/assistance-batches/{batch['id']}").json() == batch
    cancelled = client.post(f"/api/assistance-batches/{batch['id']}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    assert cancelled.json()["counts"]["cancelled"] == 2
    assert store.list("annotation_revisions") == before
    assert not store.list("annotation_suggestions")
    with TestClient(
        create_app(store.root, run_jobs=False), base_url="http://127.0.0.1"
    ) as reopened:
        assert reopened.get(endpoint(session)).json() == [cancelled.json()]


@pytest.mark.parametrize(
    "changes",
    [
        {"provider": "alibaba"},
        {"provider": "ollama"},
        {"allow_external": True},
        {"max_cost_usd": 1.0},
        {"preview_id": "external-preview"},
        {"endpoint": "https://example.com"},
        {"source": "imported"},
        {"frame_ids": []},
        {"frame_ids": ["synthetic"] * 26},
        {"frame_ids": [42]},
        {"model": ""},
        {"threshold": -0.1},
        {"threshold": 1.1},
        {"threshold": "0.5"},
        {"instructions": "x" * 2001},
    ],
)
def test_strict_preview_contract_rejects_external_fields_and_invalid_inputs(workspace, changes):
    client, store, session, frames = workspace
    data = {**payload(frames), **changes}
    assert client.post(endpoint(session) + "/preview", json=data).status_code == 422
    assert not store.list("jobs") and not store.list("assistance_batches")


@pytest.mark.parametrize("bad_fingerprint", [None, "old", "z" * 64])
def test_queue_requires_well_formed_preview_fingerprint(workspace, bad_fingerprint):
    client, store, session, frames = workspace
    data = {**payload(frames), "name": "Review", "expected_fingerprint": bad_fingerprint}
    assert client.post(endpoint(session), json=data).status_code == 422
    assert not store.list("jobs")


def test_stale_preview_returns_conflict_without_any_children(workspace):
    client, store, session, frames = workspace
    data = payload(frames)
    preview = client.post(endpoint(session) + "/preview", json=data).json()
    save_annotation(
        store,
        frames[0]["id"],
        expected_revision=1,
        boxes=[{"id": "changed", "label": "car", "box": [2, 3, 35, 40]}],
        decisions={},
    )
    response = client.post(
        endpoint(session),
        json={**data, "name": "Review", "expected_fingerprint": preview["fingerprint"]},
    )
    assert response.status_code == 409, response.text
    assert not store.list("jobs")
    assert not store.list("assistance_records")
    assert not store.list("assistance_batches")


def test_unavailable_model_is_explicit_and_manual_labels_are_unchanged(workspace, monkeypatch):
    client, store, session, frames = workspace
    before = store.list("annotation_revisions")
    monkeypatch.setattr(
        "iris.assistance_batches.provider_status",
        lambda *_: {"status": "unavailable", "reason": "Synthetic offline provider"},
    )
    response = client.post(endpoint(session) + "/preview", json=payload(frames))
    assert response.status_code == 409
    assert "offline" in response.json()["detail"]
    assert store.list("annotation_revisions") == before
    assert not store.list("jobs")


def test_duplicate_and_foreign_frames_are_invalid(workspace):
    client, store, session, frames = workspace
    data = payload(frames)
    data["frame_ids"] = [frames[0]["id"], frames[0]["id"]]
    assert client.post(endpoint(session) + "/preview", json=data).status_code == 422
    other = client.post("/api/sessions", json={"name": "Other", "scene_group": "other"}).json()
    assert client.post(endpoint(other) + "/preview", json=payload(frames)).status_code == 422
    assert not store.list("jobs")


def test_missing_records_and_cross_origin_requests(workspace):
    client, store, _, frames = workspace
    assert client.get("/api/sessions/missing/assistance-batches").status_code == 404
    assert client.get("/api/assistance-batches/missing").status_code == 404
    assert client.post("/api/assistance-batches/missing/cancel").status_code == 404
    assert (
        client.post(
            "/api/sessions/missing/assistance-batches/preview", json=payload(frames)
        ).status_code
        == 404
    )
    assert (
        client.post(
            "/api/assistance-batches/missing/cancel", headers={"Origin": "https://example.com"}
        ).status_code
        == 403
    )
    assert not store.list("jobs")
