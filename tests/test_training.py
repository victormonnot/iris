"""Offline training lifecycle fixtures; synthetic engines do not claim model accuracy."""

import hashlib

import pytest
from PIL import Image

from iris import models, training
from iris.jobs import JobManager
from iris.media import _pixel_hash
from iris.store import Store, new_id, now
from iris.taxonomies import TAXONOMY


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    store = Store(tmp_path / "workspace")
    dataset = store.insert(
        "dataset_versions",
        {
            "id": new_id(),
            "name": "Synthetic frozen fixture",
            "path": "unused-fixture.json",
            "manifest_sha256": "a" * 64,
            "summary": {},
            "created_at": now(),
        },
    )
    frames = []
    for value, split in enumerate(("train", "val", "test"), 1):
        image = Image.new("RGB", (32, 24), (value, 20, 40))
        path = store.root / f"fixture-{split}.png"
        image.save(path)
        frames.append(
            {
                "frame_id": f"fixture-{split}",
                "session_id": split,
                "scene_group": split,
                "split": split,
                "sha256": _pixel_hash(image),
                "width": 32,
                "height": 24,
                "image_path": path.name,
                "image_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "annotation_revision_id": new_id(),
                "revision": 1,
                "boxes": [{"label": "person", "box": [2, 2, 12, 22]}],
            }
        )
    manifest = {
        "taxonomy": TAXONOMY,
        "class_mapping": training.CLASS_MAPPING,
        "frames": frames,
    }
    parent = {
        **models.get_spec(models.TRAINING_ARCHITECTURE),
        "status": "ready",
        "weight_sha256": "b" * 64,
    }
    monkeypatch.setattr(training, "_manifest", lambda *_args: manifest)
    monkeypatch.setattr(training, "catalog", lambda _root: [parent])
    return store, dataset, manifest, parent


def queue(workspace, **changes):
    store, dataset, _, parent = workspace
    fields = {
        "name": "Fixture head training",
        "dataset_id": dataset["id"],
        "parent_model_id": parent["id"],
        "steps": 3,
    }
    fields.update(changes)
    return training.create_training(store, JobManager(store), **fields)


class FixtureTrainer:
    """Tiny lifecycle stand-in; writes explicitly non-model fixture bytes."""

    metadata = {
        "runtime": "synthetic fixture",
        "validation_consumed": False,
        "test_consumed": False,
    }
    seen = []

    def __init__(self, root, parent, config):
        self.config = config
        self.steps = 0

    def step(self, image, boxes):
        self.steps += 1
        self.seen.append(image.getpixel((0, 0)))
        return {"loss": 1 / self.steps, "losses": {"fixture_loss": 1 / self.steps}}

    def write_checkpoint(self, path):
        path.write_bytes(b"Synthetic checkpoint fixture; these are not model parameters")
        return {"head_weights_changed": "synthetic fixture assertion only"}


def run(store, row, **kwargs):
    return training.run_training(
        store,
        row["id"],
        lambda *_args: None,
        lambda: False,
        trainer_factory=kwargs.pop("trainer_factory", FixtureTrainer),
        **kwargs,
    )


def test_queue_snapshots_parent_and_dataset_and_publishes_job_atomically(workspace):
    store, dataset, _, parent = workspace
    row = queue(workspace)
    assert row["job"]["kind"] == "train"
    assert row["job"]["params"] == {"training_id": row["id"]}
    assert row["config"]["dataset_manifest_sha256"] == dataset["manifest_sha256"]
    assert row["config"]["parent_weight_sha256"] == parent["weight_sha256"]
    assert row["config"]["device"] == "cpu"
    assert row["config"]["scope"] == "prediction_head_only"
    assert row["history"] == []
    assert row["checkpoint_id"] is None
    assert training.training_detail(Store(store.root), row["id"]) == row


@pytest.mark.parametrize(
    "changes",
    [
        {"steps": 0},
        {"steps": 201},
        {"steps": True},
        {"steps": 1.5},
        {"seed": -1},
        {"seed": True},
        {"learning_rate": float("nan")},
        {"learning_rate": 0},
        {"learning_rate": 0.2},
        {"learning_rate": True},
        {"name": " "},
        {"parent_model_id": "ssdlite320_mobilenet_v3_large"},
        {"dataset_id": "missing"},
    ],
)
def test_invalid_training_never_queues_a_job(workspace, changes):
    with pytest.raises(ValueError):
        queue(workspace, **changes)
    assert workspace[0].list("jobs") == []


def test_training_only_opens_train_pixels_and_registers_verified_checkpoint(workspace, monkeypatch):
    store, _, manifest, _ = workspace
    FixtureTrainer.seen = []
    # Held-out files are deliberately unavailable: they are not an input to training.
    for frame in manifest["frames"][1:]:
        store.artifact_path(frame["image_path"]).unlink()
    row = queue(workspace)
    result = run(store, row)
    assert result["steps_completed"] == 3
    assert result["cancelled"] is False
    assert FixtureTrainer.seen == [(1, 20, 40)] * 3
    final = training.training_detail(store, row["id"])
    assert [step["step"] for step in final["history"]] == [1, 2, 3]
    assert final["metadata"]["training_scene_groups"] == ["train"]
    assert final["metadata"]["quality_metrics"] is None
    assert final["checkpoint"]["id"] == result["checkpoint_id"]
    assert final["job"]["status"] == "succeeded"
    assert final["job"]["result"] == result
    assert final["job"]["finished_at"]
    # A cancellation after publication cannot relabel the completed checkpoint's
    # job as cancelled, even before the worker returns to its generic finalizer.
    JobManager(store).cancel(row["job_id"])
    assert store.get("jobs", row["job_id"])["status"] == "succeeded"
    monkeypatch.setattr(models, "_runtime_problem", lambda: None)
    entry = next(
        item for item in models.catalog(store.root) if item["id"] == result["checkpoint_id"]
    )
    assert entry["status"] == "ready"
    assert entry["origin"] == "trained"
    assert entry["training"] is True
    assert entry["classes"] == [{"id": 1, "name": "person"}, {"id": 3, "name": "car"}]
    checkpoint = store.get("trained_models", result["checkpoint_id"])
    path = store.artifact_path(checkpoint["path"])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == checkpoint["weight_sha256"]
    path.write_bytes(b"tampered synthetic checkpoint")
    entry = next(
        item for item in models.catalog(store.root) if item["id"] == result["checkpoint_id"]
    )
    assert entry["status"] == "invalid_weights"
    assert not list(path.parent.glob("*.part"))
    with pytest.raises(ValueError, match="immutable"):
        run(store, row)


@pytest.mark.parametrize("changed", ["parent", "dataset", "pixels"])
def test_worker_rejects_inputs_changed_after_queue(workspace, changed):
    store, dataset, manifest, parent = workspace
    row = queue(workspace)
    if changed == "parent":
        parent["weight_sha256"] = "c" * 64
    elif changed == "dataset":
        store.update("dataset_versions", dataset["id"], {"manifest_sha256": "c" * 64})
    else:
        store.artifact_path(manifest["frames"][0]["image_path"]).write_bytes(b"altered")
    with pytest.raises(ValueError, match="changed|hash"):
        run(store, row)
    assert store.list("trained_models") == []
    assert store.get("training_runs", row["id"])["history"] == []


def test_cancel_after_step_preserves_loss_history_without_checkpoint(workspace):
    store = workspace[0]
    row = queue(workspace)

    def cancelled():
        return bool(store.get("training_runs", row["id"])["history"])

    result = training.run_training(
        store, row["id"], lambda *_args: None, cancelled, trainer_factory=FixtureTrainer
    )
    assert result["cancelled"] is True
    assert result["steps_completed"] == 1
    assert len(store.get("training_runs", row["id"])["history"]) == 1
    assert store.list("trained_models") == []
    assert not (store.root / "models" / "trained").exists()


def test_cancel_during_checkpoint_write_cleans_unpublished_file(workspace):
    store = workspace[0]
    row = queue(workspace, steps=1)

    class CancelOnSave(FixtureTrainer):
        def write_checkpoint(self, path):
            info = super().write_checkpoint(path)
            JobManager(store).cancel(row["job_id"])
            return info

    result = run(store, row, trainer_factory=CancelOnSave)
    assert result["cancelled"] is True
    assert store.list("trained_models") == []
    assert list((store.root / "models" / "trained").iterdir()) == []


def test_optimizer_failure_preserves_completed_steps(workspace):
    class FailSecond(FixtureTrainer):
        def step(self, image, boxes):
            if self.steps:
                raise RuntimeError("Synthetic optimizer failure")
            return super().step(image, boxes)

    row = queue(workspace)
    with pytest.raises(RuntimeError, match="optimizer"):
        run(workspace[0], row, trainer_factory=FailSecond)
    assert len(workspace[0].get("training_runs", row["id"])["history"]) == 1
    assert workspace[0].list("trained_models") == []


def test_all_negative_dataset_is_not_queued(workspace):
    workspace[2]["frames"][0]["boxes"] = []
    with pytest.raises(ValueError, match="positive"):
        queue(workspace)


@pytest.mark.parametrize(
    "key, value",
    [
        ("training_scene_groups", "val"),
        ("training_frame_hashes", None),
    ],
)
def test_previously_trained_group_or_pixels_cannot_become_holdout(workspace, key, value):
    store, _, manifest, parent = workspace
    parent["provenance"] = {key: [value or manifest["frames"][2]["sha256"]]}
    with pytest.raises(ValueError, match="already used"):
        queue(workspace)
    assert store.list("jobs") == []


def test_trained_label_two_is_remapped_to_coco_car():
    class Tensor:
        def __init__(self, value):
            self.value = value

        def detach(self):
            return self

        def cpu(self):
            return self

        def tolist(self):
            return self.value

    output = {"boxes": Tensor([[1, 2, 11, 22]]), "labels": Tensor([2]), "scores": Tensor([0.75])}
    rows = models._serialize_predictions(output, (32, 24), models.IRIS_NATIVE_TO_COCO)
    assert rows == [
        {"box": [1, 2, 11, 22], "label_id": 3, "native_label_id": 2, "label": "car", "score": 0.75}
    ]


def test_trained_checkpoint_path_cannot_escape_workspace(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        models.checkpoint_path(tmp_path, {"origin": "trained", "checkpoint_path": "../bad.pth"})


def test_training_cannot_publish_through_a_directory_symlink(workspace, tmp_path):
    store, _, _, _ = workspace
    outside = tmp_path / "outside"
    outside.mkdir()
    (store.root / "models").mkdir()
    (store.root / "models" / "trained").symlink_to(outside, target_is_directory=True)
    row = queue(workspace)
    with pytest.raises(ValueError, match="outside the workspace"):
        run(store, row)
    assert not list(outside.iterdir())
    assert store.list("trained_models") == []
    assert not store.get("training_runs", row["id"])["checkpoint_id"]
