"""Explicit opt-in CPU fine-tuning on synthetic fixtures; no accuracy benchmark."""

import os
import random
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from test_training_api import BASE_URL, PARENT, SCOPES, prepare_dataset

from iris import training
from iris.app import create_app
from iris.models import IRIS_NATIVE_TO_COCO, TorchvisionDetector, get_spec


def wait_for_job(client, job_id):
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        job = next(row for row in client.get("/api/jobs").json() if row["id"] == job_id)
        if job["status"] not in {"queued", "running"}:
            assert job["status"] == "succeeded", job
            return job
        time.sleep(0.1)
    pytest.fail("Bounded CPU fixture job did not complete within 90 seconds")


def _matches_module(name, prefix):
    return name == prefix or name.startswith(prefix + ".")


@pytest.mark.parametrize("scope", list(SCOPES))
def test_live_selected_scope_gradients_frozen_state_and_reload(tmp_path, scope):
    """Five real CPU steps across scopes, continuation and negative-image handling.

    HTTP queues a persisted job; an instrumented real trainer executes it in this
    process so gradients can be inspected before checkpoint serialization. The
    separate pipeline test below exercises subprocess scheduling end to end.
    """
    if os.environ.get("IRIS_TEST_TRAINING") != "1":
        pytest.skip("Set IRIS_TEST_TRAINING=1 for bounded CPU training-scope fixtures")
    location = os.environ.get("IRIS_TEST_MODEL_DIR")
    assert location, "IRIS_TEST_MODEL_DIR must identify already provisioned official weights"
    import torch
    from torchvision.ops.misc import FrozenBatchNorm2d

    root = tmp_path / "workspace"
    (root / "models").mkdir(parents=True)
    filename = get_spec(PARENT)["weight_filename"]
    (root / "models" / filename).symlink_to(Path(location).resolve() / "models" / filename)

    with TestClient(create_app(root, run_jobs=False), base_url=BASE_URL) as client:
        dataset, _ = prepare_dataset(
            client,
            train_count=1 if scope == "prediction_head_only" else 2,
            negative_train_index=None if scope == "prediction_head_only" else 1,
        )
        store = client.app.state.store
        train_frames = [
            frame for frame in dataset["manifest"]["frames"] if frame["split"] == "train"
        ]
        first_index = next(
            (index for index, frame in enumerate(train_frames) if not frame["boxes"]), 0
        )
        for seed in range(100):
            shuffled = list(range(len(train_frames)))
            random.Random(seed).shuffle(shuffled)
            if shuffled[-1] == first_index:
                break
        else:
            pytest.fail("Could not arrange the bounded fixture sampling order")

        def execute(selected_scope, *, parent_id=PARENT, expected_initial=None):
            instances = []
            steps = 2 if selected_scope == "full_model" else 1
            expected_frames = [train_frames[index] for index in reversed(shuffled)][:steps]

            class RecordingTrainer(training._HeadTrainer):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    instances.append(self)
                    assert self.model.rpn.score_thresh == 0.0
                    self.before_state = {
                        name: value.detach().clone()
                        for name, value in self.model.state_dict().items()
                    }
                    if expected_initial is not None:
                        assert self.before_state.keys() == expected_initial.keys()
                        for name, value in expected_initial.items():
                            assert torch.equal(self.before_state[name], value), name
                    self.selected_names = {
                        name
                        for name, parameter in self.model.named_parameters()
                        if parameter.requires_grad
                    }
                    expected_names = {
                        name
                        for name, _ in self.model.named_parameters()
                        if any(_matches_module(name, prefix) for prefix in SCOPES[selected_scope])
                    }
                    assert self.selected_names == expected_names
                    assert self.selected_names
                    self.frozen_normalization = {
                        name + "." + key
                        for name, module in self.model.named_modules()
                        if isinstance(module, FrozenBatchNorm2d)
                        for key in module.state_dict()
                    }
                    assert self.frozen_normalization
                    self.gradient_groups = {}
                    self.seen_boxes = []

                def step(self, image, boxes):
                    assert boxes == expected_frames[len(self.seen_boxes)]["boxes"]
                    self.seen_boxes.append(boxes)
                    result = super().step(image, boxes)
                    assert torch.isfinite(torch.tensor(result["loss"]))
                    selected = {
                        name: parameter
                        for name, parameter in self.model.named_parameters()
                        if name in self.selected_names
                    }
                    for name, parameter in selected.items():
                        assert parameter.grad is not None, name
                        assert torch.isfinite(parameter.grad).all(), name
                    self.gradient_groups = {
                        prefix: any(
                            torch.count_nonzero(parameter.grad).item() > 0
                            for name, parameter in selected.items()
                            if _matches_module(name, prefix)
                        )
                        for prefix in SCOPES[selected_scope]
                    }
                    assert all(self.gradient_groups.values()), self.gradient_groups
                    return result

            response = client.post(
                "/api/trainings",
                json={
                    "name": "Synthetic CPU scope check — no quality claim",
                    "dataset_id": dataset["id"],
                    "parent_model_id": parent_id,
                    "scope": selected_scope,
                    "steps": steps,
                    "learning_rate": 0.001,
                    "seed": seed,
                },
            )
            assert response.status_code == 202, response.text
            queued = response.json()
            assert queued["config"]["scope"] == selected_scope
            outcome = training.run_training(
                store,
                queued["id"],
                lambda *_args: None,
                lambda: False,
                trainer_factory=RecordingTrainer,
            )
            assert outcome["steps_completed"] == steps and outcome["checkpoint_id"]
            assert len(instances) == 1
            engine = instances[0]
            finished = client.get(f"/api/trainings/{queued['id']}").json()
            assert finished["job"]["status"] == "succeeded"
            assert len(finished["history"]) == steps
            assert [row["frame_id"] for row in finished["history"]] == [
                frame["frame_id"] for frame in expected_frames
            ]
            metadata = finished["metadata"]
            assert metadata["scope"] == selected_scope
            assert metadata["config"]["scope"] == selected_scope
            assert metadata["trainable_modules"] == SCOPES[selected_scope]
            assert metadata["validation_consumed"] is False
            assert metadata["test_consumed"] is False
            assert metadata["frozen_parameters_unchanged"] is True
            assert metadata["frozen_batchnorm_buffers_unchanged"] is True
            assert set(metadata["changed_trainable_modules"]) == set(SCOPES[selected_scope])
            assert metadata["training_proposal_filtering"]["rpn_score_threshold"] == 0.0
            assert (
                metadata["training_proposal_filtering"]["parent_inference_rpn_score_threshold"]
                == 0.05
            )
            assert metadata["native_filtering"]["rpn"]["score_threshold"] == 0.05

            checkpoint = store.get("trained_models", outcome["checkpoint_id"])
            saved = torch.load(store.artifact_path(checkpoint["path"]), weights_only=True)
            assert saved.keys() == engine.before_state.keys()
            for name, before in engine.before_state.items():
                assert torch.isfinite(saved[name]).all(), name
                if name not in engine.selected_names:
                    assert torch.equal(saved[name], before), name
            for name in engine.frozen_normalization:
                assert torch.equal(saved[name], engine.before_state[name]), name
            for prefix in SCOPES[selected_scope]:
                assert any(
                    not torch.equal(saved[name], engine.before_state[name])
                    for name in engine.selected_names
                    if _matches_module(name, prefix)
                ), prefix

            # Construction loads the saved state with strict=True through the
            # real inference adapter; normalized label IDs must survive reload.
            detector = TorchvisionDetector(root, outcome["checkpoint_id"])
            assert detector.model.rpn.score_thresh == 0.05
            assert detector.metadata["native_filtering"]["rpn"]["score_threshold"] == 0.05
            assert detector.native_to_coco == IRIS_NATIVE_TO_COCO
            assert detector.model.roi_heads.box_predictor.cls_score.out_features == 3
            assert detector.model.roi_heads.box_predictor.bbox_pred.out_features == 12
            loaded = detector.model.state_dict()
            assert loaded.keys() == saved.keys()
            assert all(torch.equal(loaded[name], value) for name, value in saved.items())
            with Image.new("RGB", (32, 24), (0, 70, 110)) as image:
                prediction = detector.predict(image)
            assert prediction["input_size"] == [32, 24]
            assert all(row["label_id"] in {1, 3} for row in prediction["detections"])
            assert all(row["native_label_id"] in {1, 2} for row in prediction["detections"])
            return checkpoint, saved

        checkpoint, weights = execute(scope)
        if scope == "prediction_head_only":
            child, _ = execute(
                "partial_backbone", parent_id=checkpoint["id"], expected_initial=weights
            )
            assert child["parent_model_id"] == checkpoint["id"]
            assert child["metadata"]["config"]["scope"] == "partial_backbone"
            assert child["metadata"]["config"]["class_mapping"] == {"person": 1, "car": 2}
            assert (
                child["metadata"]["training_frame_hashes"]
                == checkpoint["metadata"]["training_frame_hashes"]
            )


def test_live_training_checkpoint_evaluation_and_restart(tmp_path):
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

        # A real evaluation reads the immutable validation pixels and runs COCO
        # metrics. Synthetic annotations only verify the pipeline, never quality.
        evaluated = client.post(
            "/api/evaluations",
            json={
                "name": "Synthetic validation — no accuracy claim",
                "dataset_id": dataset["id"],
                "model_ids": [PARENT, checkpoint_id],
            },
        )
        assert evaluated.status_code == 202, evaluated.text
        evaluation_id = evaluated.json()["id"]
        wait_for_job(client, evaluated.json()["job"]["id"])
        evaluation = client.get(f"/api/evaluations/{evaluation_id}").json()
        assert len(evaluation["predictions"]) == 2
        for model in evaluation["models"]:
            summary = model["metrics"]["summary"]
            assert summary["ground_truth_count"] == 2
            assert 0 <= summary["map"] <= 1
            assert model["metadata"]["device"] == "cpu"
        selected = client.post(
            "/api/model-references",
            json={
                "evaluation_id": evaluation_id,
                "model_id": checkpoint_id,
                "reviewer": "Automated fixture only",
                "notes": "Persistence check, not a recommendation or improvement claim",
                "expected_previous_id": None,
            },
        )
        assert selected.status_code == 201, selected.text
        reference = client.get("/api/model-references").json()
        audit = client.post(
            "/api/evaluations",
            json={
                "name": "Synthetic final test audit",
                "dataset_id": dataset["id"],
                "model_ids": [PARENT, checkpoint_id],
                "split": "test",
                "validation_evaluation_id": evaluation_id,
            },
        )
        assert audit.status_code == 202, audit.text
        wait_for_job(client, audit.json()["job"]["id"])
        audit_id = audit.json()["id"]
        test_evaluation = client.get(f"/api/evaluations/{audit_id}").json()
        assert len(test_evaluation["predictions"]) == 2

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
        assert client.get("/api/model-references").json() == reference

    with TestClient(create_app(root), base_url=BASE_URL) as reopened:
        assert reopened.get(f"/api/trainings/{training_id}").json() == training
        assert reopened.get(f"/api/comparisons/{comparison_id}").json() == comparison
        assert reopened.get(f"/api/datasets/{dataset['id']}").json() == dataset
        assert reopened.get(f"/api/evaluations/{evaluation_id}").json() == evaluation
        assert reopened.get(f"/api/evaluations/{audit_id}").json() == test_evaluation
        assert reopened.get("/api/model-references").json() == reference
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
