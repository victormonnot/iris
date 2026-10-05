"""Dataset/training HTTP contracts, using synthetic images and no downloads."""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from iris import training
from iris.app import create_app
from iris.models import get_spec

BASE_URL = "http://127.0.0.1"
PARENT = "fasterrcnn_mobilenet_v3_large_320_fpn"
SCOPES = {
    "prediction_head_only": ["roi_heads.box_predictor"],
    "partial_backbone": [
        "backbone.body.13",
        "backbone.body.14",
        "backbone.body.15",
        "backbone.body.16",
        "backbone.fpn",
        "rpn",
        "roi_heads",
    ],
    "full_model": ["backbone", "rpn", "roi_heads"],
}


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path / "workspace", run_jobs=False), base_url=BASE_URL) as api:
        yield api


def prepare_dataset(client, *, train_count=1, negative_train_index=None):
    frame_ids = []
    train_groups = ["fixture-train", *[f"fixture-train-{i}" for i in range(1, train_count)]]
    for index, group in enumerate((*train_groups, "fixture-val", "fixture-test")):
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
                "boxes": []
                if index == negative_train_index
                else [
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
        "splits": {
            **{group: "train" for group in train_groups},
            "fixture-val": "val",
            "fixture-test": "test",
        },
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
        {"steps": 10001},
        {"seed": True},
        {"seed": -1},
        {"learning_rate": 0},
        {"learning_rate": 1},
        {"device": "metal"},
        {"scope": "unknown"},
        {"scope": "full"},
        {"scope": True},
        {"scope": None},
        {"unexpected": "field"},
    ],
)
@pytest.mark.parametrize("endpoint", ["/api/trainings", "/api/trainings/preview"])
def test_invalid_training_configuration_never_creates_job(client, changes, endpoint):
    response = client.post(
        endpoint,
        json={
            "name": "Fixture",
            "dataset_id": "missing",
            "parent_model_id": PARENT,
            **changes,
        },
    )
    assert response.status_code == 422, response.text
    assert client.get("/api/jobs").json() == []


@pytest.fixture
def ready_parent(monkeypatch):
    """Metadata fixture only: preview and queue must not construct a detector."""
    parent = {**get_spec(PARENT), "status": "ready", "weight_sha256": "b" * 64}
    monkeypatch.setattr(training, "catalog", lambda _root: [parent])

    def no_model(*_args, **_kwargs):
        pytest.fail("Training preview and queue must not load detector weights")

    monkeypatch.setattr(training, "TorchvisionDetector", no_model)
    return parent


@pytest.mark.parametrize("scope", list(SCOPES))
@pytest.mark.parametrize("steps", [2, 3, 7])
def test_training_preview_is_readonly_and_reports_train_coverage(
    client, ready_parent, scope, steps
):
    dataset, _ = prepare_dataset(client, train_count=3)
    store = client.app.state.store
    tables = ("jobs", "training_runs", "trained_models", "dataset_versions", "frames")
    before = {table: store.list(table) for table in tables}
    files = {str(path.relative_to(store.root)) for path in store.root.rglob("*") if path.is_file()}
    response = client.post(
        "/api/trainings/preview",
        json={
            "name": "Synthetic preview",
            "dataset_id": dataset["id"],
            "parent_model_id": PARENT,
            "scope": scope,
            "steps": steps,
            "learning_rate": 0.002,
            "seed": 11,
        },
    )
    assert response.status_code == 200, response.text
    preview = response.json()
    assert preview["scope"]["id"] == scope
    assert preview["scope"]["label"] and preview["scope"]["description"]
    assert preview["scope"]["trainable_modules"] == SCOPES[scope]
    assert preview["config"]["scope"] == scope
    assert preview["config"]["learning_rate"] == 0.002
    assert preview["config"]["seed"] == 11
    assert preview["config"]["parent_weight_sha256"] == ready_parent["weight_sha256"]
    assert preview["config"]["dataset_manifest_sha256"] == dataset["manifest_sha256"]
    assert preview["dataset"] == {
        "id": dataset["id"],
        "name": dataset["name"],
        "train_images": 3,
        "positive_train_images": 3,
        "annotation_count": 6,
    }
    assert preview["workload"] == {
        "steps": steps,
        "batch_size": 1,
        "image_visits": steps,
        "unique_images_min": min(steps, 3),
        "full_passes": steps // 3,
        "remainder_images": steps % 3,
        "device": "cpu",
    }
    assert "job" not in preview
    assert {table: store.list(table) for table in tables} == before
    assert {
        str(path.relative_to(store.root)) for path in store.root.rglob("*") if path.is_file()
    } == files


@pytest.mark.parametrize("scope", [None, *SCOPES])
def test_queue_preserves_selected_scope_and_legacy_default(client, ready_parent, scope):
    dataset, _ = prepare_dataset(client)
    payload = {
        "name": "Synthetic scope fixture",
        "dataset_id": dataset["id"],
        "parent_model_id": PARENT,
        "steps": 1,
    }
    if scope is not None:
        payload["scope"] = scope
    preview = client.post("/api/trainings/preview", json=payload)
    assert preview.status_code == 200, preview.text
    queued = client.post("/api/trainings", json=payload)
    assert queued.status_code == 202, queued.text
    training_run = queued.json()
    expected = scope or "prediction_head_only"
    assert training_run["config"]["scope"] == expected
    assert training_run["config"] == preview.json()["config"]
    assert training_run["job"]["kind"] == "train"
    assert training_run["job"]["status"] == "queued"
    assert training_run["job"]["params"] == {"training_id": training_run["id"]}
    assert training_run["checkpoint_id"] is None and training_run["history"] == []
    with TestClient(
        create_app(client.app.state.store.root, run_jobs=False), base_url=BASE_URL
    ) as reopened:
        persisted = reopened.get(f"/api/trainings/{training_run['id']}").json()
        assert persisted == training_run
        assert persisted["config"]["scope"] == expected


def test_training_preview_requires_the_same_verified_parent_and_manifest(client):
    dataset, _ = prepare_dataset(client)
    payload = {
        "name": "Blocked preview",
        "dataset_id": dataset["id"],
        "parent_model_id": PARENT,
        "scope": "full_model",
    }
    missing = client.post("/api/trainings/preview", json=payload)
    assert missing.status_code == 409, missing.text
    store = client.app.state.store
    row = store.get("dataset_versions", dataset["id"])
    store.artifact_path(row["path"]).write_text("{}")
    invalid = client.post("/api/trainings/preview", json=payload)
    assert invalid.status_code in {409, 422}, invalid.text
    assert client.get("/api/jobs").json() == []
    assert client.get("/api/trainings").json() == []


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
