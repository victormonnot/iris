"""Durable-training boundaries, using scalar fixtures and no detector execution."""

from copy import deepcopy
from threading import Event

import pytest
from fastapi.testclient import TestClient
from test_training_recovery import (
    SimulatedTrainer,
    execute,
    fixture_workspace,
    interrupted,
    queue,
    resumed,
)

from iris import training
from iris import training_recovery as recovery
from iris.app import create_app
from iris.jobs import JobManager
from iris.projects import create_project
from iris.store import Store


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    fixture = fixture_workspace(tmp_path)
    monkeypatch.setattr(training, "catalog", lambda _root: [fixture[2]])
    return fixture


def test_runtime_mismatch_rejects_resume_before_loading_or_optimizer_work(workspace):
    store = workspace[0]
    source = interrupted(workspace)
    original = deepcopy(store.get("training_runs", source["id"]))
    child = resumed(store, source)

    class ChangedRuntime(SimulatedTrainer):
        def resume_runtime(self):
            return {"engine": "a different synthetic scalar runtime"}

        def load_resume_state(self, *_args, **_kwargs):
            pytest.fail("An incompatible runtime must not load optimizer state")

        def step(self, *_args):
            pytest.fail("An incompatible runtime must not perform an optimizer step")

    with pytest.raises(ValueError, match="runtime changed"):
        execute(store, child, trainer_factory=ChangedRuntime)
    assert SimulatedTrainer.instances[-1].steps == 0
    assert store.get("training_runs", child["id"])["history"] == source["history"][:8]
    assert store.get("training_runs", source["id"]) == original
    assert not store.list("training_checkpoints", training_id=child["id"])
    assert not store.list("trained_models")


def test_server_restart_preserves_state_and_requires_explicit_resume(workspace, monkeypatch):
    store = workspace[0]
    source = interrupted(workspace)
    queued = queue(workspace)
    store.update("jobs", source["job_id"], {"status": "running"})
    original = deepcopy(store.get("training_runs", source["id"]))
    states = store.list("training_checkpoints")
    contents = {state["path"]: store.artifact_path(state["path"]).read_bytes() for state in states}
    polled, launched = Event(), []
    original_list = Store.list

    def observe_poll(self, table, **filters):
        rows = original_list(self, table, **filters)
        if table == "jobs" and filters.get("status") == "queued":
            polled.set()
        return rows

    monkeypatch.setattr(Store, "list", observe_poll)
    monkeypatch.setattr(JobManager, "_execute", lambda _self, job: launched.append(job))
    with TestClient(create_app(store.root), base_url="http://127.0.0.1") as api:
        assert polled.wait(timeout=2), "The real supervisor did not inspect its queue"
        assert not launched
        assert store.get("jobs", source["job_id"])["status"] == "interrupted"
        assert store.get("jobs", queued["job_id"])["status"] == "interrupted"
        detail = api.get(f"/api/trainings/{source['id']}").json()
        assert detail["recovery"]["can_resume"]
        assert detail["recovery"]["checkpoint_step"] == 8
        assert detail["recovery"]["existing_training_id"] is None
        assert len(store.list("training_runs")) == 2
        assert len(store.list("jobs")) == 2
    assert store.get("training_runs", source["id"]) == original
    assert store.list("training_checkpoints") == states
    assert all(store.artifact_path(path).read_bytes() == value for path, value in contents.items())


def test_recovery_lineage_cycle_is_rejected_even_with_equal_creation_times(workspace):
    store = workspace[0]
    source = interrupted(workspace)
    child = resumed(store, source)
    store.update("jobs", child["job_id"], {"status": "interrupted"})
    config = {
        **source["config"],
        "resume_from": {**child["config"]["resume_from"], "training_id": child["id"]},
    }
    store.update(
        "training_runs", source["id"], {"config": config, "created_at": child["created_at"]}
    )
    with store.connect() as connection, pytest.raises(ValueError, match="cycle"):
        recovery.validate_training_recoveries(connection, store.root)


def test_partial_checkpoint_write_failure_preserves_prior_usable_state(workspace):
    store = workspace[0]
    source = queue(workspace)

    class BrokenSecondWrite(SimulatedTrainer):
        def write_resume_state(self, path, **kwargs):
            if self.steps == 16:
                path.write_bytes(b"partial synthetic state")
                raise OSError("Simulated full storage during the second state write")
            return super().write_resume_state(path, **kwargs)

    with pytest.raises(OSError, match="full storage"):
        execute(store, source, trainer_factory=BrokenSecondWrite)
    store.update("jobs", source["job_id"], {"status": "failed"})
    (checkpoint,) = store.list("training_checkpoints", training_id=source["id"])
    assert checkpoint["step"] == 8
    assert recovery.file_identity(store.artifact_path(checkpoint["path"])) == (
        checkpoint["state_sha256"],
        checkpoint["size_bytes"],
    )
    assert len(store.get("training_runs", source["id"])["history"]) == 10
    assert list((store.root / "training_checkpoints" / source["id"]).iterdir()) == [
        store.artifact_path(checkpoint["path"])
    ]
    child = resumed(store, source)
    assert child["config"]["resume_from"]["step"] == 8
    execute(store, child)
    final = training.training_detail(store, child["id"])
    assert final["job"]["status"] == "succeeded"
    assert len(final["history"]) == 25
    assert store.get("training_checkpoints", checkpoint["id"]) == checkpoint


def test_durable_create_api_requires_matching_preview_and_deduplicates_within_project(workspace):
    store, dataset, parent = workspace
    payload = {
        "name": "Synthetic durable API run",
        "dataset_id": dataset["id"],
        "parent_model_id": parent["id"],
        "steps": 210,
        "checkpoint_interval": 10,
    }
    with TestClient(create_app(store.root, run_jobs=False), base_url="http://127.0.0.1") as api:
        missing = api.post("/api/trainings", json=payload)
        assert missing.status_code == 422
        assert not store.list("jobs")
        preview = api.post("/api/trainings/preview", json=payload)
        assert preview.status_code == 200, preview.text
        plan = preview.json()
        assert plan["resume_supported"] and plan["workload"]["steps"] == 210
        confirmed = {
            **payload,
            "request_id": plan["request_id"],
            "expected_fingerprint": plan["fingerprint"],
        }
        stale = api.post("/api/trainings", json={**confirmed, "checkpoint_interval": 11})
        assert stale.status_code == 422
        other = create_project(store, name="Another project")
        for path, body in (("/api/trainings/preview", payload), ("/api/trainings", confirmed)):
            response = api.post(path, params={"project_id": other["id"]}, json=body)
            assert response.status_code == 404
        assert not store.list("jobs")
        first = api.post("/api/trainings", json=confirmed)
        second = api.post("/api/trainings", json=confirmed)
        assert first.status_code == second.status_code == 202
        assert first.json()["id"] == second.json()["id"]
        assert len(store.list("jobs")) == len(store.list("training_runs")) == 1
        assert first.json()["config"]["request_id"] == plan["request_id"]
