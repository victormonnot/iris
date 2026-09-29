"""Explicit opt-in CPU fine-tuning on synthetic fixtures; no accuracy benchmark."""

import os
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from test_training_api import BASE_URL, PARENT, prepare_dataset

from iris.app import create_app
from iris.models import TorchvisionDetector, get_spec


def wait_for_job(client, job_id):
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        job = next(row for row in client.get("/api/jobs").json() if row["id"] == job_id)
        if job["status"] not in {"queued", "running"}:
            assert job["status"] == "succeeded", job
            return job
        time.sleep(0.1)
    pytest.fail("Bounded CPU fixture job did not complete within 90 seconds")


def test_live_training_checkpoint_comparison_and_restart(tmp_path):
    if os.environ.get("IRIS_TEST_TRAINING") != "1":
        pytest.skip("Set IRIS_TEST_TRAINING=1 to run three real CPU optimizer steps")
    location = os.environ.get("IRIS_TEST_MODEL_DIR")
    assert location, "IRIS_TEST_MODEL_DIR must identify already provisioned official weights"
    import torch

    root = tmp_path / "workspace"
    root.mkdir()
    (root / "models").mkdir()
    filename = get_spec(PARENT)["weight_filename"]
    (root / "models" / filename).symlink_to(Path(location).resolve() / "models" / filename)
    with TestClient(create_app(root), base_url=BASE_URL) as client:
        dataset, _ = prepare_dataset(client)
        response = client.post(
            "/api/trainings",
            json={
                "name": "Synthetic CPU test",
                "dataset_id": dataset["id"],
                "parent_model_id": PARENT,
                "steps": 2,
                "learning_rate": 0.001,
                "seed": 7,
            },
        )
        assert response.status_code == 202, response.text
        training_id = response.json()["id"]
        wait_for_job(client, response.json()["job"]["id"])
        training = client.get(f"/api/trainings/{training_id}").json()
        assert len(training["history"]) == 2
        checkpoint_id = training["checkpoint_id"]
        store = client.app.state.store
        checkpoint = store.get("trained_models", checkpoint_id)
        weights = torch.load(store.artifact_path(checkpoint["path"]), weights_only=True)
        # Torchvision upgrades historical FPN state-dict keys during loading.
        parent = TorchvisionDetector(root, PARENT).model.state_dict()
        # Real optimizer updates change the transferred head, while frozen feature
        # extraction weights remain exactly equal to the official parent.
        key = "roi_heads.box_predictor.cls_score.weight"
        assert weights[key].shape[0] == 3
        assert not torch.equal(weights[key], parent[key][[0, 1, 3]])
        for name in parent:
            if not name.startswith("roi_heads.box_predictor."):
                assert torch.equal(weights[name], parent[name]), name
        models = client.get("/api/models").json()
        trained = next(model for model in models if model["id"] == checkpoint_id)
        assert trained["status"] == "ready" and trained["training"]

        # Use the frozen test group only for inference, never optimization.
        held_out = next(
            frame for frame in dataset["manifest"]["frames"] if frame["split"] == "test"
        )
        compared = client.post(
            f"/api/sessions/{held_out['session_id']}/comparisons",
            json={
                "name": "Synthetic before/after software check",
                "frame_ids": [held_out["frame_id"]],
                "model_ids": [PARENT, checkpoint_id],
            },
        )
        assert compared.status_code == 202, compared.text
        comparison_id = compared.json()["id"]
        wait_for_job(client, compared.json()["job"]["id"])
        comparison = client.get(f"/api/comparisons/{comparison_id}").json()
        assert len(comparison["predictions"]) == 2
        prediction = next(
            row for row in comparison["predictions"] if row["model_id"] == checkpoint_id
        )
        assert all(row["label_id"] in {1, 3} for row in prediction["detections"])
        assert all(row["native_label_id"] in {1, 2} for row in prediction["detections"])

        continued = client.post(
            "/api/trainings",
            json={
                "name": "Synthetic second generation",
                "dataset_id": dataset["id"],
                "parent_model_id": checkpoint_id,
                "steps": 1,
            },
        )
        assert continued.status_code == 202, continued.text
        wait_for_job(client, continued.json()["job"]["id"])
        second = client.get(f"/api/trainings/{continued.json()['id']}").json()
        assert second["checkpoint_id"] != checkpoint_id
        assert second["parent_model_id"] == checkpoint_id

    with TestClient(create_app(root), base_url=BASE_URL) as reopened:
        assert reopened.get(f"/api/trainings/{training_id}").json() == training
        assert reopened.get(f"/api/comparisons/{comparison_id}").json() == comparison
        assert reopened.get(f"/api/datasets/{dataset['id']}").json() == dataset
        assert (
            len(
                [
                    row
                    for row in reopened.get("/api/models").json()
                    if row.get("origin") == "trained"
                ]
            )
            == 2
        )
