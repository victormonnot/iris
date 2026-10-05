"""Reports preserve explicitly selected simulated measurements without loading models."""

import json
import shutil
from copy import deepcopy

import pytest
from fastapi.testclient import TestClient
from test_export_cuda import cuda_environment
from test_model_exports import fixture_workspace, measurement, options

from iris import experiments
from iris import model_exports as exports
from iris.app import create_app
from iris.experiment_export import render_experiment_html
from iris.experiments import create_experiment, experiment_detail, preview_experiment
from iris.store import Store
from iris.workspace_archive import create_archive
from iris.workspace_restore import inspect_archive, restore_archive


def saved_measurement(workspace, *, target="cpu", mismatch=False):
    store = workspace[0]
    values = {**options(workspace), "target_device": target}
    preview = exports.preview_export(store, **values)
    export = exports.create_export(
        store,
        **values,
        request_id=preview["request_id"],
        expected_fingerprint=preview["fingerprint"],
    )
    exports.run_export(store, export["id"], lambda *_: None, lambda: False)
    export = exports.export_detail(store, export["id"])
    payload = measurement(export)
    if target == "cuda":
        payload["environment"] = cuda_environment()
    if mismatch:
        payload["samples"][0]["detections"][0]["score"] -= 0.01
    preview = exports.preview_measurement(store, export["id"], payload)
    item = exports.save_measurement(store, export["id"], payload, preview["fingerprint"])
    return export, item


@pytest.fixture
def evidence(tmp_path):
    workspace = fixture_workspace(tmp_path)
    export, item = saved_measurement(workspace)
    return workspace, export, item


def create(workspace, **fields):
    return create_experiment(
        workspace[0], evaluation_id=workspace[2]["id"], title="Synthetic saved evidence", **fields
    )


@pytest.mark.parametrize("target,mismatch", [("cpu", False), ("cuda", False), ("cuda", True)])
def test_measured_targets_are_explicit_frozen_and_separate(tmp_path, target, mismatch):
    workspace = fixture_workspace(tmp_path, custom=True)
    store = workspace[0]
    export, item = saved_measurement(workspace, target=target, mismatch=mismatch)
    before = {table: store.list(table) for table in store.columns}
    preview = preview_experiment(store, workspace[2]["id"])
    assert {table: store.list(table) for table in store.columns} == before
    assert preview["snapshot"]["deployments"]["measurements"] == []
    assert len(preview["available_measurements"]) == 1
    candidate = preview["available_measurements"][0]
    assert candidate["id"] == item["id"]
    assert candidate["summary"]["parity_passed"] is not mismatch
    assert candidate["summary"]["execution_verified"] is False
    assert candidate["declaration"] == "simulation"
    assert candidate["environment"]["device"].startswith(target)
    assert candidate["source"]["evaluation_model_id"] == preview["snapshot"]["lanes"][0]["id"]
    assert "detections" not in json.dumps(candidate)
    assert "checkpoint_path" not in json.dumps(candidate)
    assert "uuid" not in json.dumps(candidate)
    assert create(workspace)["snapshot"]["deployments"]["measurements"] == []
    report = create(
        workspace,
        measurement_ids=[item["id"]],
        expected_source_fingerprint=preview["source_fingerprint"],
    )
    assert report["snapshot"]["deployments"]["measurements"] == [candidate]
    assert report["snapshot"]["version"] == 2
    # No bundle, model or dataset is needed once the report is saved.
    for directory in ("model_exports", "models", "datasets"):
        shutil.rmtree(store.root / directory, ignore_errors=True)
    store.update("model_export_measurements", item["id"], {"summary": {"changed": True}})
    assert experiment_detail(Store(store.root), report["id"]) == report
    exported = render_experiment_html(store, report["id"], expected_revision=1)
    assert item["fingerprint"].encode() in exported
    assert export["archive_sha256"].encode() in exported


@pytest.mark.parametrize(
    "field,value",
    [
        ("fingerprint", "0" * 64),
        ("summary", {}),
        ("payload", {}),
    ],
)
def test_corrupt_optional_measurement_does_not_block_plain_report(evidence, field, value):
    workspace, _, item = evidence
    store = workspace[0]
    store.update("model_export_measurements", item["id"], {field: value})
    assert preview_experiment(store, workspace[2]["id"])["available_measurements"] == []
    assert create(workspace)["snapshot"]["deployments"]["measurements"] == []
    with pytest.raises(ValueError, match="measurement"):
        create(workspace, measurement_ids=[item["id"]])


@pytest.mark.parametrize(
    "part,field,value",
    [
        ("source", "evaluation_id", "different"),
        ("source", "evaluation_model_id", "different"),
        ("source", "dataset_manifest_sha256", "0" * 64),
        ("model", "sha256", "0" * 64),
        ("model", "id", "different"),
    ],
)
def test_saved_measurement_cannot_be_attached_to_unrelated_evidence(evidence, part, field, value):
    workspace, export, item = evidence
    manifest = deepcopy(export["manifest"])
    manifest[part][field] = value
    workspace[0].update("model_exports", export["id"], {"manifest": manifest})
    assert preview_experiment(workspace[0], workspace[2]["id"])["available_measurements"] == []
    with pytest.raises(ValueError, match="measurement"):
        create(workspace, measurement_ids=[item["id"]])


@pytest.mark.parametrize("selection", ["x", ["missing"], ["x", "x"], ["a"] * 5, [True]])
def test_selection_is_bounded_unique_and_scoped(evidence, selection):
    workspace = evidence[0]
    with pytest.raises(ValueError):
        create(workspace, measurement_ids=selection)
    assert workspace[0].list("experiment_reports") == []


def test_preview_fingerprint_is_stable_and_stale_creation_is_conflict(evidence, monkeypatch):
    workspace, _, _ = evidence
    store, _, evaluation, _ = workspace
    monkeypatch.setattr(experiments, "now", lambda: "2026-01-01T00:00:00+00:00")
    preview = preview_experiment(store, evaluation["id"])
    monkeypatch.setattr(experiments, "now", lambda: "2026-01-02T00:00:00+00:00")
    fresh = preview_experiment(store, evaluation["id"])
    assert preview["source_fingerprint"] == fresh["source_fingerprint"]
    store.update("evaluations", evaluation["id"], {"name": "Changed after preview"})
    with pytest.raises(experiments.ExperimentConflict, match="refresh"):
        create(workspace, expected_source_fingerprint=preview["source_fingerprint"])
    with TestClient(
        create_app(store.root, run_jobs=False), base_url="http://127.0.0.1:8010"
    ) as api:
        response = api.post(
            "/api/experiments",
            json={
                "evaluation_id": evaluation["id"],
                "title": "Conflict",
                "expected_source_fingerprint": preview["source_fingerprint"],
            },
        )
    assert response.status_code == 409, response.text
    assert store.list("experiment_reports") == []


def test_selected_measurement_rechecked_before_commit(evidence, monkeypatch):
    workspace, _, item = evidence
    store = workspace[0]
    original = experiments.deployments.available_measurements
    calls = 0

    def changing(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            # Simulate evidence changing after preparation, without a competing writer lock.
            return []
        return original(*args, **kwargs)

    monkeypatch.setattr(experiments.deployments, "available_measurements", changing)
    with pytest.raises(ValueError, match="changed"):
        create(
            workspace,
            measurement_ids=[item["id"]],
            example_frame_ids=[workspace[3]["manifest"]["frames"][1]["frame_id"]],
        )
    assert store.list("experiment_reports") == []
    assert not list((store.root / "reports").glob("*"))


def test_v1_report_stays_unchanged_and_v2_archive_is_self_contained(evidence, tmp_path):
    workspace, _, item = evidence
    store = workspace[0]
    legacy = create(workspace)
    snapshot = deepcopy(legacy["snapshot"])
    snapshot["version"] = 1
    snapshot.pop("insights")
    snapshot.pop("deployments")
    store.update(
        "experiment_reports",
        legacy["id"],
        {
            "snapshot": snapshot,
            "snapshot_sha256": experiments._digest(snapshot),
        },
    )
    legacy = experiment_detail(store, legacy["id"])
    current = create(workspace, measurement_ids=[item["id"]])
    archive = tmp_path / "reports.iris.zip"
    create_archive(store.root, archive)
    inspection = inspect_archive(archive)
    restore_archive(
        archive, tmp_path / "restored", expected_archive_sha256=inspection["archive_sha256"]
    )
    restored = Store(tmp_path / "restored")
    for original in (legacy, current):
        assert experiment_detail(restored, original["id"]) == original
        assert render_experiment_html(restored, original["id"], expected_revision=1)
