"""Recover SAM histories without the optional runtime, GPU, weights or network."""

import hashlib
import json
from copy import deepcopy

import pytest
from test_benchmark_segmentation import configured, launch, run
from test_benchmark_segmentation import runtime as runtime
from test_benchmark_segmentation import workspace as workspace
from test_benchmarks_archive import records

from iris import sam_provider, sam_runtime
from iris.store import Store
from iris.workspace_archive import ArchiveError, allowed_artifact_path, create_archive
from iris.workspace_restore import inspect_archive, restore_archive


@pytest.mark.parametrize(
    "state", ["prepared", "queued", "ready", "raw_saved", "failed", "cancelled"]
)
def test_sam_history_round_trip_without_runtime(workspace, runtime, tmp_path, monkeypatch, state):
    store = workspace[0]
    reference, candidate, preview = configured(workspace)
    if state != "prepared":
        trial = launch(workspace, reference, candidate, preview)
        if state != "queued":
            run(workspace, trial)
            if state != "ready":
                output = store.list("benchmark_outputs", trial_id=trial["id"])[0]
                store.update(
                    "benchmark_outputs",
                    output["id"],
                    {
                        "result": None,
                        "metadata": {**output["metadata"], "state": state},
                        "error": None if state == "raw_saved" else "Synthetic interrupted output",
                    },
                )
        store.update(
            "jobs", trial["job_id"], {"status": "succeeded" if state == "ready" else "cancelled"}
        )

    def forbidden(*args, **kwargs):
        pytest.fail("Archive recovery must not probe or load a SAM runtime")

    monkeypatch.setattr(sam_provider, "provider_status", forbidden)
    monkeypatch.setattr(sam_runtime, "runtime_status", forbidden)
    monkeypatch.setattr(sam_runtime, "SamRuntime", forbidden)
    monkeypatch.delenv("IRIS_SAM_PYTHON", raising=False)
    before = records(store)
    archive = tmp_path / "sam.zip"
    create_archive(store.root, archive)
    inspected = inspect_archive(archive)
    restored = tmp_path / "restored"
    restore_archive(archive, restored, expected_archive_sha256=inspected["archive_sha256"])
    assert records(Store(restored)) == before
    assert not (restored / sam_provider.CHECKPOINT_PATH).exists()


@pytest.mark.parametrize(
    "change", ["runtime", "work", "proposal", "score", "frame", "attempt", "state"]
)
def test_archive_rejects_tampered_sam_evidence(workspace, runtime, tmp_path, change):
    store = workspace[0]
    reference, candidate, preview = configured(workspace)
    trial = launch(workspace, reference, candidate, preview)
    run(workspace, trial)
    store.update("jobs", trial["job_id"], {"status": "succeeded"})
    output = store.list("benchmark_outputs", trial_id=trial["id"])[0]
    if change in {"runtime", "work"}:
        frozen = deepcopy(trial["config"])
        if change == "runtime":
            frozen["local_plan"]["runtime_identity"]["packages"]["numpy"] = "2.0.0"
        else:
            frozen["local_plan"]["work"]["prompt_evaluations"] += 1
        store.update("benchmark_trials", trial["id"], {"config": frozen})
    elif change in {"proposal", "score"}:
        result = deepcopy(output["result"])
        if change == "proposal":
            result["proposals"][0]["box"][0] += 1
        else:
            result["proposals"][0]["score"] = 0.99
        store.update("benchmark_outputs", output["id"], {"result": result})
    else:
        metadata = deepcopy(output["metadata"])
        metadata[{"frame": "frame_sha256", "attempt": "attempt_id", "state": "state"}[change]] = (
            "wrong"
        )
        store.update("benchmark_outputs", output["id"], {"metadata": metadata})
    with pytest.raises(ArchiveError, match="SAM"):
        create_archive(store.root, tmp_path / "tampered.zip")


def test_only_expected_sam_checkpoint_path_is_archived():
    assert allowed_artifact_path("models/sam3/sam3.pt")
    for path in (
        "models/sam3/other.pt",
        "models/sam3/sam3.pt.part",
        "models/sam3/config.json",
        "models/other/sam3.pt",
        "models/sam3.pt",
    ):
        assert not allowed_artifact_path(path)


def test_present_sam_checkpoint_must_match_published_identity(workspace, tmp_path, monkeypatch):
    store = workspace[0]
    path = store.root / sam_provider.CHECKPOINT_PATH
    path.parent.mkdir(parents=True)
    fixture = b"synthetic checkpoint bytes, never loaded"
    path.write_bytes(fixture)
    with pytest.raises(ArchiveError, match="size"):
        create_archive(store.root, tmp_path / "wrong-size.zip")
    monkeypatch.setattr(sam_provider, "CHECKPOINT_SIZE", len(fixture))
    with pytest.raises(ArchiveError, match="checksum"):
        create_archive(store.root, tmp_path / "wrong-hash.zip")
    monkeypatch.setattr(sam_provider, "CHECKPOINT_SHA256", hashlib.sha256(fixture).hexdigest())
    archive = tmp_path / "fixture-weight.zip"
    create_archive(store.root, archive)
    inspected = inspect_archive(archive)
    assert "sam3" in json.dumps(inspected)
    restored = tmp_path / "restored-weight"
    restore_archive(archive, restored, expected_archive_sha256=inspected["archive_sha256"])
    assert (restored / sam_provider.CHECKPOINT_PATH).read_bytes() == fixture
