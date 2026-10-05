"""Long-run orchestration on synthetic images and JSON stand-in optimizer states."""

import hashlib
import json
import random
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import pytest
from fastapi.testclient import TestClient
from test_datasets import add_frame

from iris import training
from iris import training_recovery as recovery
from iris.app import create_app
from iris.datasets import create_dataset
from iris.jobs import JobManager
from iris.models import TRAINING_ARCHITECTURE, get_spec
from iris.projects import create_project
from iris.store import Store
from iris.workspace_archive import ArchiveError, create_archive
from iris.workspace_restore import inspect_archive, restore_archive


class SimulatedTrainer:
    """A seeded scalar optimizer, intentionally not a valid detector checkpoint."""

    metadata = {
        "runtime": "SIMULATION: JSON scalar optimizer, no detector execution",
        "validation_consumed": False,
        "test_consumed": False,
    }
    instances = []

    def __init__(self, root, parent, config):
        self.config, self.steps, self.weight, self.momentum = config, 0, 1.0, 0.0
        self.randomizer = random.Random(config["seed"])
        self.seen = []
        self.instances.append(self)

    def resume_runtime(self):
        return {"engine": "synthetic scalar fixture only"}

    def step(self, image, boxes):
        pixel = image.getpixel((0, 0))[0]
        self.seen.append(pixel)
        self.momentum = 0.9 * self.momentum + pixel / 255 + self.randomizer.random()
        self.weight -= self.config["learning_rate"] * self.momentum
        self.steps += 1
        loss = abs(self.weight) + 0.01
        return {"loss": loss, "losses": {"synthetic_loss": loss}}

    def write_checkpoint(self, path):
        path.write_bytes(
            recovery.canonical(
                {
                    "simulation": True,
                    "weight": self.weight,
                    "momentum": self.momentum,
                    "steps": self.steps,
                }
            )
        )
        return {"head_weights_changed": "Synthetic scalar only; not detector evidence"}

    def write_resume_state(self, path, *, binding, sampler):
        path.write_bytes(
            recovery.canonical(
                {
                    "binding": binding,
                    "sampler": sampler,
                    "steps": self.steps,
                    "weight": self.weight,
                    "momentum": self.momentum,
                    "random_state": self.randomizer.getstate(),
                }
            )
        )

    def load_resume_state(self, path, *, binding, sampler, expected_sha256):
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == expected_sha256
        state = json.loads(raw)
        assert state["binding"] == binding and state["sampler"] == sampler
        self.steps, self.weight, self.momentum = state["steps"], state["weight"], state["momentum"]
        rng = state["random_state"]
        self.randomizer.setstate((rng[0], tuple(rng[1]), rng[2]))


def fixture_workspace(tmp_path):
    store = Store(tmp_path / "workspace")
    frames = [
        add_frame(
            store,
            tmp_path,
            group=group,
            color=(color, 20, 30),
            boxes=[{"id": "person", "label": "person", "box": [2, 2, 12, 22]}],
        )
        for group, color in [("train", 5), ("train", 10), ("train", 15), ("val", 20), ("test", 25)]
    ]
    dataset = create_dataset(
        store,
        name="Synthetic recovery dataset",
        frame_ids=[f["id"] for f in frames],
        splits={"train": "train", "val": "val", "test": "test"},
    )
    parent = {**get_spec(TRAINING_ARCHITECTURE), "status": "ready", "weight_sha256": "b" * 64}
    return store, dataset, parent


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    value = fixture_workspace(tmp_path)
    monkeypatch.setattr(training, "catalog", lambda _root: [value[2]])
    return value


def queue(workspace, *, steps=25, interval=8, **changes):
    store, dataset, parent = workspace
    options = {
        "name": "Simulated long training",
        "dataset_id": dataset["id"],
        "parent_model_id": parent["id"],
        "steps": steps,
        "checkpoint_interval": interval,
        **changes,
    }
    preview = training.preview_training(store, **options)
    return training.create_training(
        store,
        JobManager(store),
        **options,
        request_id=preview["request_id"],
        expected_fingerprint=preview["fingerprint"],
    )


def execute(store, row, *, stop=None, trainer_factory=SimulatedTrainer):
    def progress(_fraction, message):
        if (
            stop is not None
            and message.startswith("CPU fine-tuning")
            and SimulatedTrainer.instances[-1].steps == stop
        ):
            store.update("jobs", row["job_id"], {"status": "interrupted"})

    return training.run_training(
        store,
        row["id"],
        progress,
        lambda: store.get("jobs", row["job_id"])["status"] == "interrupted",
        trainer_factory=trainer_factory,
    )


def interrupted(workspace):
    row = queue(workspace)
    execute(workspace[0], row, stop=10)
    return training.training_detail(workspace[0], row["id"])


def resumed(store, row):
    preview = recovery.preview_resume(store, row["id"])
    return recovery.resume_training(
        store, JobManager(store), row["id"], expected_fingerprint=preview["fingerprint"]
    )


def test_resume_matches_continuous_training_preserves_old_attempt_and_two_states(workspace):
    store = workspace[0]
    full = queue(workspace)
    execute(store, full)
    continuous = SimulatedTrainer.instances[-1]
    stopped = interrupted(workspace)
    assert len(stopped["history"]) == 10
    assert stopped["recovery"]["checkpoint_step"] == 8
    assert stopped["recovery"]["recomputed_steps"] == 2
    before = deepcopy(store.get("training_runs", stopped["id"]))
    child = resumed(store, stopped)
    assert child["job"]["params"]["recovery_of"] == stopped["job_id"]
    assert len(child["history"]) == 8
    execute(store, child)
    continued = SimulatedTrainer.instances[-1]
    assert (continued.weight, continued.momentum, continued.steps) == (
        continuous.weight,
        continuous.momentum,
        continuous.steps,
    )
    assert continued.seen == continuous.seen[8:]
    assert store.get("training_runs", stopped["id"]) == before
    final = training.training_detail(store, child["id"])
    assert final["job"]["status"] == "succeeded" and final["checkpoint_id"]
    assert [point["loss"] for point in final["history"]] == [
        point["loss"] for point in training.training_detail(store, full["id"])["history"]
    ]
    assert [state["step"] for state in final["checkpoints"]] == [24, 25]
    assert len(list((store.root / "training_checkpoints" / child["id"]).glob("*.pth"))) == 2
    assert final["recovery"]["can_resume"] is False


def test_more_than_old_limit_is_bounded_and_checkpoints_rotate(workspace):
    store = workspace[0]
    row = queue(workspace, steps=210, interval=10)
    execute(store, row)
    final = training.training_detail(store, row["id"])
    assert len(final["history"]) == 210
    assert [state["step"] for state in final["checkpoints"]] == [200, 210]
    assert len(list((store.root / "training_checkpoints" / row["id"]).iterdir())) == 2


@pytest.mark.parametrize(
    "steps,interval", [(10001, 50), (10000, 49), (1000, 0), (1000, True), (1000, 1001)]
)
def test_invalid_work_or_checkpoint_frequency_never_queues(workspace, steps, interval):
    with pytest.raises(ValueError):
        queue(workspace, steps=steps, interval=interval)
    assert not workspace[0].list("training_runs")


def test_stale_initial_preview_and_two_create_requests(workspace):
    store, dataset, parent = workspace
    options = {
        "name": "Durable",
        "dataset_id": dataset["id"],
        "parent_model_id": parent["id"],
        "steps": 20,
        "checkpoint_interval": 5,
    }
    preview = training.preview_training(store, **options)
    kwargs = {
        **options,
        "request_id": preview["request_id"],
        "expected_fingerprint": preview["fingerprint"],
    }
    with pytest.raises(ValueError, match="preview"):
        training.create_training(store, JobManager(store), **{**kwargs, "steps": 21})
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(
            pool.map(
                lambda _: training.create_training(store, JobManager(store), **kwargs), range(2)
            )
        )
    assert rows[0]["id"] == rows[1]["id"]
    assert len(store.list("training_runs")) == 1


def test_two_resume_requests_create_one_child_and_late_worker_cannot_publish(workspace):
    store = workspace[0]
    row = interrupted(workspace)
    preview = recovery.preview_resume(store, row["id"])
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(
            pool.map(
                lambda _: recovery.resume_training(
                    store, JobManager(store), row["id"], expected_fingerprint=preview["fingerprint"]
                ),
                range(2),
            )
        )
    assert rows[0]["id"] == rows[1]["id"]
    token = row["job"]["params"]["training_claim"]
    assert recovery.save_history(store, row, [], token) is False
    with pytest.raises(ValueError):
        training.run_training(
            store, row["id"], lambda *_: None, lambda: False, trainer_factory=SimulatedTrainer
        )


@pytest.mark.parametrize("damage", ["state", "sampler", "history", "config", "parent", "pixels"])
def test_changed_resume_evidence_is_rejected_before_queue(workspace, damage):
    store = workspace[0]
    row = interrupted(workspace)
    checkpoint = row["checkpoints"][0]
    if damage == "state":
        store.artifact_path(checkpoint["path"]).write_bytes(b"changed state")
    elif damage == "sampler":
        metadata = deepcopy(checkpoint["metadata"])
        metadata["sampler"]["remaining_order"] = [False]
        store.update("training_checkpoints", checkpoint["id"], {"metadata": metadata})
    elif damage == "history":
        row["history"][0]["loss"] += 1
        store.update("training_runs", row["id"], {"history": row["history"]})
    elif damage == "config":
        store.update(
            "training_runs", row["id"], {"config": {**row["config"], "learning_rate": 0.01}}
        )
    elif damage == "parent":
        workspace[2]["weight_sha256"] = "c" * 64
    else:
        path = next(f for f in workspace[1]["manifest"]["frames"] if f["split"] == "train")[
            "image_path"
        ]
        store.artifact_path(path).write_bytes(b"changed image")
    with pytest.raises((ValueError, OSError)):
        resumed(store, row)
    assert len(store.list("training_runs")) == 1


def test_failed_publication_resumes_without_optimizer_steps(workspace):
    store = workspace[0]
    row = queue(workspace, steps=3, interval=2)

    class PublicationFailure(SimulatedTrainer):
        def write_checkpoint(self, _path):
            raise OSError("simulated disk failure")

    with pytest.raises(OSError):
        execute(store, row, trainer_factory=PublicationFailure)
    store.update("jobs", row["job_id"], {"status": "failed"})
    preview = recovery.preview_resume(store, row["id"])
    assert preview["remaining_steps"] == 0
    child = resumed(store, row)
    execute(store, child)
    assert SimulatedTrainer.instances[-1].seen == []
    assert training.training_detail(store, child["id"])["checkpoint_id"]


def test_inherited_state_allows_another_explicit_attempt_after_early_failure(workspace):
    store = workspace[0]
    source = interrupted(workspace)
    child = resumed(store, source)
    store.update("jobs", child["job_id"], {"status": "failed"})
    grandchild = resumed(store, child)
    assert grandchild["config"]["resume_from"]["checkpoint_id"] == source["checkpoints"][0]["id"]
    execute(store, grandchild)
    assert training.training_detail(store, grandchild["id"])["checkpoint_id"]


def test_saved_recovery_archive_restores_and_rejects_rehashed_history(workspace, tmp_path):
    store = workspace[0]
    source = interrupted(workspace)
    child = resumed(store, source)
    store.update("jobs", child["job_id"], {"status": "interrupted"})
    archive = tmp_path / "training.zip"
    create_archive(store.root, archive)
    inspected = inspect_archive(archive)
    destination = tmp_path / "restored"
    restore_archive(archive, destination, expected_archive_sha256=inspected["archive_sha256"])
    restored = Store(destination)
    assert restored.list("training_checkpoints") == store.list("training_checkpoints")
    assert restored.list("training_runs") == store.list("training_runs")
    child["history"][0]["loss"] = 15
    store.update("training_runs", child["id"], {"history": child["history"]})
    with pytest.raises(ArchiveError):
        create_archive(store.root, tmp_path / "forged.zip")


def test_resume_api_project_scope_and_freshness(workspace):
    store = workspace[0]
    source = interrupted(workspace)
    with TestClient(create_app(store.root, run_jobs=False), base_url="http://127.0.0.1") as api:
        path = f"/api/trainings/{source['id']}"
        preview = api.post(path + "/resume-preview")
        assert preview.status_code == 200, preview.text
        assert preview.json()["recomputed_steps"] == 2
        wrong = api.post(path + "/resume", json={"expected_fingerprint": "0" * 64})
        assert wrong.status_code == 409
        created = api.post(
            path + "/resume", json={"expected_fingerprint": preview.json()["fingerprint"]}
        )
        assert created.status_code == 202, created.text
        other = create_project(store, name="Other project")
        for route in (path, path + "/resume-preview", path + "/resume"):
            response = (
                api.get(route, params={"project_id": other["id"]})
                if route == path
                else api.post(
                    route,
                    params={"project_id": other["id"]},
                    json={"expected_fingerprint": preview.json()["fingerprint"]},
                )
            )
            assert response.status_code == 404


def test_zero_work_cancel_and_stopped_checkpoint_publication(workspace):
    store = workspace[0]
    row = queue(workspace)
    result = training.run_training(
        store, row["id"], lambda *_: None, lambda: True, trainer_factory=SimulatedTrainer
    )
    assert result["cancelled"] and not store.list("training_checkpoints")

    class LateState(SimulatedTrainer):
        def write_resume_state(self, path, **kwargs):
            super().write_resume_state(path, **kwargs)
            store.update("jobs", row["job_id"], {"status": "interrupted"})

    result = execute(store, row, trainer_factory=LateState)
    assert result["cancelled"] and not store.list("training_checkpoints")
    assert not list((store.root / "training_checkpoints" / row["id"]).iterdir())
