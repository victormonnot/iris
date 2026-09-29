"""Dataset/training HTTP contracts, using synthetic images and no downloads."""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from iris.app import create_app

BASE_URL = "http://127.0.0.1"
PARENT = "fasterrcnn_mobilenet_v3_large_320_fpn"


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path / "workspace", run_jobs=False), base_url=BASE_URL) as api:
        yield api


def prepare_dataset(client):
    frame_ids = []
    for index, group in enumerate(("fixture-train", "fixture-val", "fixture-test")):
        session = client.post("/api/sessions", json={"name": group, "scene_group": group}).json()
        output = io.BytesIO()
        Image.new("RGB", (32, 24), (40 * index, 70, 110)).save(output, format="PNG")
        uploaded = client.post(
            f"/api/sessions/{session['id']}/assets",
            files={"file": ("synthetic.png", output.getvalue(), "image/png")},
        )
        assert uploaded.status_code == 201, uploaded.text
        (frame,) = client.get(f"/api/sessions/{session['id']}/frames").json()
        client.patch(f"/api/frames/{frame['id']}", json={"selected": True})
        response = client.put(
            f"/api/frames/{frame['id']}/annotation",
            json={
                "expected_revision": 0,
                "status": "validated",
                "reviewer": "Automated fixture; not human ground truth",
                "boxes": [
                    {"id": "fixture-person", "label": "person", "box": [2, 2, 15, 20]},
                    {"id": "fixture-car", "label": "car", "box": [17, 12, 30, 22]},
                ],
                "decisions": {},
            },
        )
        assert response.status_code == 200, response.text
        frame_ids.append(frame["id"])
    payload = {
        "name": "Synthetic dataset",
        "frame_ids": frame_ids,
        "splits": {"fixture-train": "train", "fixture-val": "val", "fixture-test": "test"},
    }
    response = client.post("/api/datasets", json=payload)
    assert response.status_code == 201, response.text
    return response.json(), payload


def test_dataset_manifest_download_and_restart_preserve_frozen_revision(client):
    dataset, _ = prepare_dataset(client)
    assert "path" not in dataset
    (listed,) = client.get("/api/datasets").json()
    assert listed["manifest_sha256"] == dataset["manifest_sha256"]
    assert "path" not in listed
    detail = client.get(f"/api/datasets/{dataset['id']}").json()
    manifest = client.get(f"/api/datasets/{dataset['id']}/manifest")
    assert manifest.status_code == 200
    assert "attachment" in manifest.headers["content-disposition"]
    assert manifest.json() == detail["manifest"]
    frame_id = dataset["manifest"]["frames"][0]["frame_id"]
    edit = client.put(
        f"/api/frames/{frame_id}/annotation",
        json={"expected_revision": 1, "boxes": [], "decisions": {}, "status": "draft"},
    )
    assert edit.status_code == 200, edit.text
    with TestClient(
        create_app(client.app.state.store.root, run_jobs=False), base_url=BASE_URL
    ) as reopened:
        assert reopened.get(f"/api/datasets/{dataset['id']}").json() == detail
        assert reopened.get(f"/api/datasets/{dataset['id']}/manifest").content == manifest.content


def test_dataset_candidates_exclude_draft_and_report_reserved_groups(client):
    dataset, _ = prepare_dataset(client)
    candidates = client.get("/api/dataset-candidates")
    assert candidates.status_code == 200, candidates.text
    groups = candidates.json()["groups"]
    assert {group["reserved_split"] for group in groups} == {"train", "val", "test"}
    assert sum(group["count"] for group in groups) == 3
    first = dataset["manifest"]["frames"][0]
    client.put(
        f"/api/frames/{first['frame_id']}/annotation",
        json={"expected_revision": 1, "boxes": [], "decisions": {}, "status": "draft"},
    )
    after = client.get("/api/dataset-candidates").json()
    assert sum(group["count"] for group in after["groups"]) == 2
    assert after["excluded"]["draft"] == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"steps": True},
        {"steps": 0},
        {"steps": 201},
        {"seed": True},
        {"seed": -1},
        {"learning_rate": 0},
        {"learning_rate": 1},
        {"device": "cuda"},
        {"unexpected": "field"},
    ],
)
def test_invalid_training_configuration_never_creates_job(client, changes):
    response = client.post(
        "/api/trainings",
        json={
            "name": "Fixture",
            "dataset_id": "missing",
            "parent_model_id": PARENT,
            **changes,
        },
    )
    assert response.status_code == 422, response.text
    assert client.get("/api/jobs").json() == []


def test_missing_models_block_training_without_affecting_dataset(client):
    dataset, _ = prepare_dataset(client)
    response = client.post(
        "/api/trainings",
        json={
            "name": "Unavailable parent",
            "dataset_id": dataset["id"],
            "parent_model_id": PARENT,
        },
    )
    assert response.status_code == 409, response.text
    assert client.get("/api/jobs").json() == []
    assert client.get("/api/trainings").json() == []
    assert client.get(f"/api/datasets/{dataset['id']}").status_code == 200


def test_unknown_resources_return_not_found(client):
    for endpoint in (
        "/api/datasets/missing",
        "/api/datasets/missing/manifest",
        "/api/trainings/missing",
    ):
        assert client.get(endpoint).status_code == 404


def test_manifest_tampering_prevents_serving_and_training(client):
    dataset, _ = prepare_dataset(client)
    store = client.app.state.store
    row = store.get("dataset_versions", dataset["id"])
    store.artifact_path(row["path"]).write_text("{}")
    assert client.get(f"/api/datasets/{dataset['id']}").status_code == 409
    assert client.get(f"/api/datasets/{dataset['id']}/manifest").status_code == 409
    assert client.get("/api/dataset-candidates").status_code == 409
    response = client.post(
        "/api/trainings",
        json={
            "name": "Tampered",
            "dataset_id": dataset["id"],
            "parent_model_id": PARENT,
        },
    )
    assert response.status_code in {409, 422}, response.text
    assert client.get("/api/jobs").json() == []
