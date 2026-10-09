"""Copy-only pipeline packaging, provenance, cancellation and offline transfers."""

import hashlib
import json
import zipfile
from copy import deepcopy
from pathlib import Path

import pytest
from test_temporal_detection_api import client as client
from test_temporal_identities import comparison as comparison
from test_temporal_identities import completed as completed
from test_tracking_comparisons import run_worker
from test_tracking_quality import reviewed as reviewed
from test_tracking_replay import SyntheticTracker, forbidden
from test_tracking_selections import selection_payload as selection_payload
from test_tracking_studies import request_payload as request_payload
from test_tracking_study_runtime import synthetic_metadata

from iris import models, pipeline_bundles, temporal_detections
from iris.pipeline_bundle_contracts import inspect_bundle
from iris.projects import record_project
from iris.store import Store
from iris.workspace_archive import ArchiveError, _inventory, create_archive, validate_database
from iris.workspace_restore import inspect_archive, restore_archive


@pytest.fixture(autouse=True)
def local_fixture_weights(client, monkeypatch):
    raw = b"Synthetic checkpoint bytes; never deserialized or executed.\n" * 20
    weight_hash = hashlib.sha256(raw).hexdigest()
    identifier = "ssdlite320_mobilenet_v3_large"
    spec = {
        **models._SPECS[identifier],
        "download_bytes": len(raw),
        "expected_hash_prefix": weight_hash,
    }
    monkeypatch.setitem(models._SPECS, identifier, spec)
    path = client.app.state.store.root / "models" / spec["weight_filename"]
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(raw)
    prepare = temporal_detections.prepare_detector

    def config(*args, **kwargs):
        return {**prepare(*args, **kwargs), "weight_sha256": weight_hash}

    monkeypatch.setattr(temporal_detections, "prepare_detector", config)
    original = SyntheticTracker.__init__

    def initialize(self, profile, **kwargs):
        original(self, profile, **kwargs)
        self.metadata = synthetic_metadata(profile)

    monkeypatch.setattr(SyntheticTracker, "__init__", initialize)
    return path


@pytest.fixture
def payload(comparison):
    replay = comparison["report"]["lanes"][0]["report"]
    return {
        "name": "Portable fixture",
        "source": {
            "kind": "comparison",
            "job_id": comparison["id"],
            "sequence_id": comparison["sequence_id"],
            "profile_sha256": replay["profile_sha256"],
        },
        "selection_id": None,
        "target_device": "cpu",
    }


def preview(client, payload):
    response = client.post("/api/temporal/pipeline-bundles/preview", json=payload)
    assert response.status_code == 200, response.text
    return response.json()


def launch(client, payload):
    planned = preview(client, payload)
    response = client.post(
        "/api/temporal/pipeline-bundles",
        json={**payload, "expected_fingerprint": planned["fingerprint"]},
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_preview_copy_read_download_tasks_are_explicit_without_execution(
    client, payload, monkeypatch
):
    store = client.app.state.store
    before = {
        table: store.list(table)
        for table in ("jobs", "temporal_references", "temporal_detection_frames")
    }
    monkeypatch.setattr("iris.temporal_detections.prepare_detector", forbidden)
    monkeypatch.setattr("iris.tracking_replay._factory", forbidden)
    monkeypatch.setattr("iris.tracking.tracking_status", forbidden)
    catalogue = client.get("/api/temporal/pipeline-bundle-sources")
    assert catalogue.status_code == 200, catalogue.text
    assert len(catalogue.json()["sources"]) == 2
    assert all(row["available"] for row in catalogue.json()["sources"])
    plan = preview(client, payload)
    assert plan["manifest"]["validation"]["status"] == "experimental"
    assert plan["manifest"]["selection"] is None
    assert all(store.list(table) == value for table, value in before.items())
    assert not (store.root / "pipeline_bundles").exists()
    assert client.get("/api/temporal/pipeline-bundle-status").status_code == 200
    record = launch(client, payload)
    assert record["bundle"] is None
    job = run_worker(client, record, monkeypatch)
    assert job["status"] == "succeeded", job["error"]
    bundle = job["result"]
    assert bundle["complete"] is True
    path = store.artifact_path(bundle["path"])
    assert inspect_bundle(path)["manifest"] == bundle["manifest"]
    with zipfile.ZipFile(path) as archive:
        notice = archive.read("licenses/NOTICE.txt")
        license_text = Path(pipeline_bundles.__file__).with_name("LICENSE.txt").read_bytes()
        assert license_text in notice
        assert b"no root licence" not in notice
        assert b"no additional IRIS licence grant" not in archive.read("README.md")
    url = f"/api/temporal/pipeline-bundles/{job['id']}"
    assert client.get(url).json()["bundle"] == bundle
    assert client.get(url + "/download").content == path.read_bytes()
    assert client.get(url + "/manifest").json() == bundle["manifest"]
    assert client.get("/api/temporal/pipeline-bundles").json()[0]["bundle"] is None
    generic = next(row for row in client.get("/api/jobs").json() if row["id"] == job["id"])
    assert "manifest" not in generic["params"] and "manifest" not in generic["result"]
    activity = client.get(f"/api/jobs/{job['id']}").json()
    assert activity["context"]["pipeline_bundle_id"] == job["id"]
    assert activity["artifacts"][0]["count"] == 1
    assert activity["next_action"]["workspace"] == "tracking"
    assert record_project(store, "jobs", job) == "default"
    assert store.list("temporal_detection_frames") == before["temporal_detection_frames"]


def test_cuda_target_does_not_rewrite_historical_device(client, payload, monkeypatch):
    record = launch(client, {**payload, "target_device": "cuda"})
    job = run_worker(client, record, monkeypatch)
    assert job["status"] == "succeeded", job["error"]
    detector = job["result"]["manifest"]["detector"]
    assert detector["config"]["device"] == "cpu"
    assert detector["target_device"] == "cuda"
    assert job["result"]["manifest"]["validation"]["target_device_measurement"] == "not_run"


def test_optional_selection_exports_only_policy_from_exact_source(
    client, payload, selection_payload, monkeypatch
):
    from test_tracking_selections import launch as launch_selection

    selection = launch_selection(client, selection_payload)
    assert run_worker(client, selection, monkeypatch)["status"] == "succeeded"
    selected = {**payload, "selection_id": selection["id"]}
    package = launch(client, selected)
    job = run_worker(client, package, monkeypatch)
    assert job["status"] == "succeeded", job["error"]
    exported = job["result"]["manifest"]["selection"]
    assert set(exported) == {
        "algorithm",
        "policy",
        "policy_sha256",
        "source_job_id",
        "source_report_sha256",
    }
    assert exported["policy"] == selection_payload["policy"]
    assert "identity_id" not in json.dumps(job["result"]["manifest"])
    other = deepcopy(selected)
    other["source"]["profile_sha256"] = client.get("/api/temporal/pipeline-bundle-sources").json()[
        "sources"
    ][1]["source"]["profile_sha256"]
    assert client.post("/api/temporal/pipeline-bundles/preview", json=other).status_code == 409
    rows = client.get("/api/temporal/pipeline-bundle-sources").json()["sources"]
    assert len(rows[0]["selections"]) == 1 and rows[1]["selections"] == []


def test_t8_candidate_exports_exact_profile_without_replay(
    client, payload, request_payload, monkeypatch
):
    from test_tracking_studies import launch as launch_study

    study = launch_study(client, request_payload)
    result = run_worker(client, study, monkeypatch)
    assert result["status"] == "succeeded", result["error"]
    replay = result["result"]["runs"][0]["replays"][1]
    candidate = {
        **payload,
        "source": {
            **payload["source"],
            "kind": "study",
            "job_id": study["id"],
            "profile_sha256": replay["profile_sha256"],
        },
    }
    monkeypatch.setattr("iris.tracking_replay._factory", forbidden)
    package = launch(client, candidate)
    job = run_worker(client, package, monkeypatch)
    assert job["status"] == "succeeded", job["error"]
    assert job["result"]["manifest"]["tracker"]["profile"] == replay["profile"]
    assert job["result"]["manifest"]["source"]["inherited_dataset"]["split"] == "train"


def test_project_ownership_and_wrong_source_kind(client, payload):
    package = launch(client, payload)
    foreign = client.post("/api/projects", json={"name": "Other"}).json()["id"]
    assert client.get(
        "/api/temporal/pipeline-bundle-sources", params={"project_id": foreign}
    ).json() == {"sources": []}
    assert client.get("/api/temporal/pipeline-bundles", params={"project_id": foreign}).json() == []
    assert (
        client.get(
            f"/api/temporal/pipeline-bundles/{package['id']}", params={"project_id": foreign}
        ).status_code
        == 404
    )
    assert (
        client.post(
            "/api/temporal/pipeline-bundles/preview", json=payload, params={"project_id": foreign}
        ).status_code
        == 404
    )
    with pytest.raises(KeyError):
        pipeline_bundles.get_bundle(client.app.state.store, package["id"], project_id=foreign)
    wrong = {**payload, "source": {**payload["source"], "kind": "study"}}
    assert client.post("/api/temporal/pipeline-bundles/preview", json=wrong).status_code == 404


@pytest.mark.parametrize(
    "change,status",
    [
        ({"target_device": "jetson"}, 422),
        ({"unknown": True}, 422),
        ({"selection_id": "missing"}, 404),
        ({"name": ""}, 422),
    ],
)
def test_invalid_request_never_admits_job(client, payload, change, status):
    before = client.app.state.store.list("jobs")
    response = client.post("/api/temporal/pipeline-bundles/preview", json={**payload, **change})
    assert response.status_code == status, response.text
    assert client.app.state.store.list("jobs") == before


def test_missing_or_changed_weights_and_stale_preview_do_not_publish(
    client, payload, local_fixture_weights
):
    plan = preview(client, payload)
    changed = client.post(
        "/api/temporal/pipeline-bundles",
        json={**payload, "target_device": "cuda", "expected_fingerprint": plan["fingerprint"]},
    )
    assert changed.status_code == 409 and "preview" in changed.text
    local_fixture_weights.write_bytes(b"changed")
    assert client.post("/api/temporal/pipeline-bundles/preview", json=payload).status_code == 409
    local_fixture_weights.unlink()
    rows = client.get("/api/temporal/pipeline-bundle-sources").json()["sources"]
    assert all(not row["available"] and row["reason"] for row in rows)


def test_worker_rejects_changed_resources_and_removes_partial_outputs(client, payload, monkeypatch):
    record = launch(client, payload)
    original = pipeline_bundles._resources

    def changed(*args, **kwargs):
        resources = original(*args, **kwargs)
        resources["README.md"] += b"changed"
        return resources

    monkeypatch.setattr(pipeline_bundles, "_resources", changed)
    job = run_worker(client, record, monkeypatch)
    assert job["status"] == "failed" and job["result"] is None
    assert "resources changed" in job["error"]
    assert not (client.app.state.store.root / "pipeline_bundles" / job["id"]).exists()


def test_cancel_after_file_copy_before_publication_has_no_package(client, payload, monkeypatch):
    store = client.app.state.store
    record = launch(client, payload)
    original = pipeline_bundles.inspect_bundle

    def cancel(path, *args, **kwargs):
        result = original(path, *args, **kwargs)
        store.update("jobs", record["id"], {"cancel_requested": True})
        return result

    monkeypatch.setattr(pipeline_bundles, "inspect_bundle", cancel)
    job = run_worker(client, record, monkeypatch)
    assert job["status"] == "cancelled" and job["result"] is None
    base = store.root / "pipeline_bundles"
    assert not list(base.iterdir())
    assert client.get(f"/api/temporal/pipeline-bundles/{job['id']}/download").status_code == 409


@pytest.mark.parametrize("legacy_notice", [False, True])
def test_completed_package_read_and_archive_do_not_need_original_checkpoint(
    client, payload, monkeypatch, tmp_path, local_fixture_weights, legacy_notice
):
    store = client.app.state.store
    with monkeypatch.context() as previous_version:
        if legacy_notice:
            current_resources = pipeline_bundles._resources

            def old_resources(*args, **kwargs):
                resources = current_resources(*args, **kwargs)
                resources["licenses/NOTICE.txt"] = b"""IRIS experimental pipeline format

The original detector and tracker licence notices are retained in this folder.
YOLOX detector bundles also retain the upstream detector NOTICE. Detector code
licence terms do not automatically grant rights to checkpoints or training data;
consult the weight terms links recorded in manifest.json.

This IRIS repository has no root licence file. No additional licence grant for
IRIS code is inferred by packaging these contract/inspection modules. Nothing in
this package represents a claim of independent quality, device compatibility,
physical identity certainty or publisher authentication.
"""
                return resources

            previous_version.setattr(pipeline_bundles, "_resources", old_resources)
        record = launch(client, payload)
        job = run_worker(client, record, monkeypatch)
    assert job["status"] == "succeeded", job["error"]
    with zipfile.ZipFile(store.artifact_path(job["result"]["path"])) as archive:
        license_text = Path(pipeline_bundles.__file__).with_name("LICENSE.txt").read_bytes()
        assert (license_text in archive.read("licenses/NOTICE.txt")) is not legacy_notice
    monkeypatch.setattr("iris.tracking_replay._factory", forbidden)
    monkeypatch.setattr(pipeline_bundles, "_resources", forbidden)
    target = tmp_path / "pipeline.iris-workspace"
    create_archive(store.root, target)
    inspected = inspect_archive(target)
    restored_root = tmp_path / "restored"
    restore_archive(target, restored_root, expected_archive_sha256=inspected["archive_sha256"])
    assert (
        pipeline_bundles.get_bundle(Store(restored_root), record["id"])["bundle"] == job["result"]
    )
    local_fixture_weights.unlink()
    assert pipeline_bundles.get_bundle(store, record["id"])["bundle"] == job["result"]


def test_changed_bundle_bytes_or_rehashed_manifest_rejected(client, payload, monkeypatch):
    store = client.app.state.store
    record = launch(client, payload)
    job = run_worker(client, record, monkeypatch)
    assert job["status"] == "succeeded", job["error"]
    path = store.artifact_path(job["result"]["path"])
    path.write_bytes(path.read_bytes() + b"not an allowed archive comment")
    assert client.get(f"/api/temporal/pipeline-bundles/{job['id']}").status_code == 409
    inventory, _ = _inventory(store.root)
    with pytest.raises(ArchiveError, match="pipeline bundle"):
        validate_database(store.root, inventory)


def test_cli_inspection_does_not_open_workspace(client, payload, monkeypatch, capsys):
    import sys

    from iris.cli import main

    record = launch(client, payload)
    job = run_worker(client, record, monkeypatch)
    assert job["status"] == "succeeded", job["error"]
    capsys.readouterr()
    path = client.app.state.store.artifact_path(job["result"]["path"])
    monkeypatch.setattr(
        sys, "argv", ["iris", "--data-dir", "/nonexistent/unused", "pipeline", "inspect", str(path)]
    )
    main()
    report = json.loads(capsys.readouterr().out)
    assert report["manifest"] == job["result"]["manifest"]


def test_restart_cleanup_removes_only_unpublished_job_output(client, payload):
    store = client.app.state.store
    record = launch(client, payload)
    store.update("jobs", record["id"], {"status": "interrupted"})
    base = store.root / "pipeline_bundles"
    orphan = base / record["id"]
    partial = base / (".partial-" + record["id"] + "-fixture")
    for folder in (orphan, partial):
        folder.mkdir(parents=True)
        (folder / "pipeline.zip").write_bytes(b"partial")
    pipeline_bundles.cleanup_unpublished(store)
    assert not orphan.exists() and not partial.exists()


@pytest.mark.parametrize(
    "architecture",
    ["ssdlite320_mobilenet_v3_large", "fasterrcnn_mobilenet_v3_large_320_fpn", "yolox_nano"],
)
def test_trained_checkpoint_matches_its_project_recipe_and_native_head(
    client, tmp_path, monkeypatch, architecture
):
    from test_model_exports import fixture_workspace
    from test_temporal_detection_api import create, sequence
    from test_temporal_detections import execute
    from test_temporal_detector import frozen_config
    from test_tracking_comparisons import launch as launch_comparison
    from test_tracking_replay import synthetic

    store, model, _, dataset = fixture_workspace(tmp_path)
    model = store.update(
        "trained_models",
        model["id"],
        {
            "architecture": architecture,
            "metadata": {**model["metadata"], "architecture": architecture},
        },
    )

    def prepare(root, model_id, **settings):
        spec = models.get_spec(model_id, root)
        return {**frozen_config(model_id, spec=spec), "weight_sha256": model["weight_sha256"]}

    monkeypatch.setattr(temporal_detections, "prepare_detector", prepare)
    source = sequence(client, tmp_path)
    cache = create(client, source, model_id=model["id"])
    execute(store, cache)
    synthetic(monkeypatch)
    monkeypatch.setattr("iris.tracking.tracking_status", lambda: {"available": True})
    comparison = launch_comparison(client, cache)
    completed = run_worker(client, comparison, monkeypatch)
    assert completed["status"] == "succeeded", completed["error"]
    replay = completed["result"]["lanes"][0]["report"]
    request = {
        "name": "Trained native recipe",
        "source": {
            "kind": "comparison",
            "job_id": comparison["id"],
            "sequence_id": source["id"],
            "profile_sha256": replay["profile_sha256"],
        },
        "selection_id": None,
        "target_device": "cpu",
    }
    record = launch(client, request)
    job = run_worker(client, record, monkeypatch)
    assert job["status"] == "succeeded", job["error"]
    detector = job["result"]["manifest"]["detector"]
    assert detector["config"]["origin"] == "trained"
    assert detector["checkpoint"]["encoding"] == "pytorch_state_dict"
    assert detector["output_mapping"]["head_slots"] == (2 if architecture == "yolox_nano" else 3)
    assert [item["output_id"] for item in detector["output_mapping"]["entries"]] == [1, 3]
    foreign = client.post("/api/projects", json={"name": "Other trained owner"}).json()["id"]
    store.update("dataset_versions", dataset["id"], {"project_id": foreign})
    assert client.post("/api/temporal/pipeline-bundles/preview", json=request).status_code == 409


def test_checkpoint_full_hash_checked_after_catalog_accepts_new_bytes(
    client, payload, local_fixture_weights, monkeypatch
):
    raw = local_fixture_weights.read_bytes() + b"new checkpoint"
    local_fixture_weights.write_bytes(raw)
    identifier = "ssdlite320_mobilenet_v3_large"
    monkeypatch.setitem(
        models._SPECS,
        identifier,
        {
            **models._SPECS[identifier],
            "download_bytes": len(raw),
            "expected_hash_prefix": hashlib.sha256(raw).hexdigest(),
        },
    )
    response = client.post("/api/temporal/pipeline-bundles/preview", json=payload)
    assert response.status_code == 409 and "full frozen detector SHA-256" in response.text


def test_cleanup_ignores_untrusted_nonhex_job_identifier(client, payload):
    store = client.app.state.store
    record = launch(client, payload)
    job = store.get("jobs", record["id"])
    forged = {**job, "id": "../keep-this-folder", "status": "interrupted"}
    store.insert("jobs", forged)
    preserved = store.root / "keep-this-folder"
    preserved.mkdir()
    (preserved / "contents.txt").write_text("Preserve me")
    (store.root / "pipeline_bundles").mkdir()
    pipeline_bundles.cleanup_unpublished(store)
    assert (preserved / "contents.txt").read_text() == "Preserve me"


def test_directory_sync_failure_does_not_publish_complete_package(client, payload, monkeypatch):
    record = launch(client, payload)
    original = pipeline_bundles._sync_directory

    def failed(path):
        if path.name == record["id"]:
            raise OSError("Synthetic directory sync failure")
        original(path)

    monkeypatch.setattr(pipeline_bundles, "_sync_directory", failed)
    job = run_worker(client, record, monkeypatch)
    assert job["status"] == "failed" and job["result"] is None
    assert "sync failure" in job["error"]
    assert not list((client.app.state.store.root / "pipeline_bundles").iterdir())
