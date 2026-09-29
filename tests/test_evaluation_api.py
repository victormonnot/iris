"""HTTP evaluation workflows with explicitly synthetic detector outputs."""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from test_training_api import BASE_URL, prepare_dataset

from iris import evaluation
from iris.app import create_app
from iris.models import get_spec

MODEL_IDS = ["ssdlite320_mobilenet_v3_large", "fasterrcnn_mobilenet_v3_large_320_fpn"]


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path / "workspace", run_jobs=False), base_url=BASE_URL) as api:
        yield api


@pytest.fixture
def ready_models(monkeypatch):
    rows = [
        {**get_spec(model_id), "status": "ready", "weight_sha256": "b" * 64}
        for model_id in MODEL_IDS
    ]
    monkeypatch.setattr(evaluation, "catalog", lambda _root: rows)
    return rows


class FixtureDetector:
    """Perfect first model, empty second model; not actual model performance."""

    def __init__(self, _root, model_id, device="cpu"):
        self.model_id = model_id
        self.metadata = {
            "model_id": model_id,
            "weight_sha256": "b" * 64,
            "device": device,
            "runtime": "synthetic fixture only",
        }

    def warmup(self, _image):
        pass

    def predict(self, image):
        boxes = [
            {"box": [2, 2, 15, 20], "label_id": 1, "label": "person", "score": 0.9},
            {"box": [17, 12, 30, 22], "label_id": 3, "label": "car", "score": 0.8},
        ]
        return {
            "detections": boxes if self.model_id == MODEL_IDS[0] else [],
            "input_size": list(image.size),
            "timing": {
                key: 1.0 for key in ("preprocess_ms", "inference_ms", "postprocess_ms", "total_ms")
            },
        }


def evaluate_fixture(client, dataset_id, **changes):
    payload = {
        "name": "Synthetic validation comparison",
        "dataset_id": dataset_id,
        "model_ids": MODEL_IDS,
        **changes,
    }
    response = client.post("/api/evaluations", json=payload)
    assert response.status_code == 202, response.text
    row = response.json()
    evaluation.run_evaluation(
        client.app.state.store,
        row["id"],
        lambda *_: None,
        lambda: False,
        detector_factory=FixtureDetector,
    )
    response = client.get(f"/api/evaluations/{row['id']}")
    assert response.status_code == 200, response.text
    return response.json()


def test_metrics_reference_test_audit_and_restart(client, ready_models):
    dataset, _ = prepare_dataset(client)
    detail = evaluate_fixture(client, dataset["id"])
    assert detail["job"]["status"] == "succeeded"
    metrics = {row["model_id"]: row["metrics"] for row in detail["models"]}
    assert metrics[MODEL_IDS[0]]["summary"]["map"] == pytest.approx(1.0)
    assert metrics[MODEL_IDS[0]]["summary"]["precision"] == 1.0
    assert metrics[MODEL_IDS[1]]["summary"]["map"] == 0.0
    assert metrics[MODEL_IDS[1]]["summary"]["recall"] == 0.0
    assert client.get("/api/model-references").json()["current"] is None
    promoted = client.post(
        "/api/model-references",
        json={
            "evaluation_id": detail["id"],
            "model_id": MODEL_IDS[0],
            "reviewer": "Automated fixture",
            "notes": "Exercise explicit selection, not a real recommendation",
            "expected_previous_id": None,
        },
    )
    assert promoted.status_code == 201, promoted.text
    reference = client.get("/api/model-references").json()
    assert reference["current"]["model_id"] == MODEL_IDS[0]
    audit = evaluate_fixture(
        client, dataset["id"], split="test", validation_evaluation_id=detail["id"]
    )
    assert audit["split"] == "test"
    rejected = client.post(
        "/api/model-references",
        json={
            "evaluation_id": audit["id"],
            "model_id": MODEL_IDS[1],
            "reviewer": "Fixture",
            "notes": "Test must not drive model selection",
            "expected_previous_id": reference["current"]["id"],
        },
    )
    assert rejected.status_code in {409, 422}, rejected.text
    assert client.get("/api/model-references").json() == reference
    with TestClient(
        create_app(client.app.state.store.root, run_jobs=False), base_url=BASE_URL
    ) as reopened:
        assert reopened.get(f"/api/evaluations/{detail['id']}").json() == detail
        assert reopened.get("/api/model-references").json() == reference
        assert reopened.get(f"/api/evaluations/{audit['id']}").json() == audit


def test_frozen_images_survive_source_edits_and_missing_frames_are_not_served(client):
    dataset, _ = prepare_dataset(client)
    frame = dataset["manifest"]["frames"][0]
    endpoint = f"/api/datasets/{dataset['id']}/frames/{frame['frame_id']}/image"
    before = client.get(endpoint)
    assert before.status_code == 200
    store = client.app.state.store
    original = store.get("frames", frame["frame_id"])
    Image.new("RGB", (32, 24), "red").save(store.artifact_path(original["path"]))
    after = client.get(endpoint)
    assert after.content == before.content
    with Image.open(io.BytesIO(after.content)) as image:
        assert image.size == (32, 24)
    assert client.get(f"/api/datasets/{dataset['id']}/frames/missing/image").status_code == 404
    Image.new("RGB", (32, 24), "red").save(store.artifact_path(frame["image_path"]))
    assert client.get(endpoint).status_code == 409


@pytest.mark.parametrize(
    "changes",
    [
        {"split": "train"},
        {"model_ids": []},
        {"model_ids": ["a", "b", "c"]},
        {"confidence_threshold": -1},
        {"confidence_threshold": 1.1},
        {"iou_threshold": 0},
        {"iou_threshold": 1.1},
        {"device": "remote"},
        {"allow_external": True},
    ],
)
def test_invalid_evaluation_never_queues(client, changes):
    response = client.post(
        "/api/evaluations",
        json={
            "name": "Fixture",
            "dataset_id": "missing",
            "model_ids": MODEL_IDS,
            **changes,
        },
    )
    assert response.status_code == 422, response.text
    assert client.get("/api/jobs").json() == []


def test_test_split_requires_fixed_validation_protocol(client, ready_models):
    dataset, _ = prepare_dataset(client)
    payload = {
        "name": "Test audit",
        "dataset_id": dataset["id"],
        "model_ids": MODEL_IDS,
        "split": "test",
    }
    first = client.post("/api/evaluations", json=payload)
    assert first.status_code in {409, 422}, first.text
    val = evaluate_fixture(client, dataset["id"])
    changed = client.post(
        "/api/evaluations",
        json={
            **payload,
            "validation_evaluation_id": val["id"],
            "confidence_threshold": 0.3,
        },
    )
    assert changed.status_code in {409, 422}, changed.text
    assert len(client.get("/api/evaluations").json()) == 1


def test_reference_selection_requires_current_snapshot_and_reason(client, ready_models):
    dataset, _ = prepare_dataset(client)
    detail = evaluate_fixture(client, dataset["id"])
    payload = {
        "evaluation_id": detail["id"],
        "model_id": MODEL_IDS[0],
        "reviewer": "Fixture",
        "notes": "Controlled selection",
        "expected_previous_id": None,
    }
    assert client.post("/api/model-references", json={**payload, "notes": ""}).status_code == 422
    assert client.post("/api/model-references", json=payload).status_code == 201
    assert (
        client.post("/api/model-references", json={**payload, "model_id": MODEL_IDS[1]}).status_code
        == 409
    )
    assert len(client.get("/api/model-references").json()["history"]) == 1


def test_missing_evaluation_and_dataset_return_not_found(client):
    assert client.get("/api/evaluations/missing").status_code == 404
    assert client.get("/api/datasets/missing/frames/missing/image").status_code == 404
