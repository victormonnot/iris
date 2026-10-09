"""Native export lifecycle using synthetic checkpoints and fake detector outputs only."""

import hashlib
import subprocess
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from test_datasets import add_frame

from iris import evaluation
from iris import model_exports as exports
from iris.app import create_app
from iris.datasets import create_dataset
from iris.job_activity import job_detail
from iris.jobs import JobManager
from iris.model_taxonomy import dataset_contract
from iris.models import TRAINING_ARCHITECTURE, get_spec
from iris.store import DEFAULT_PROJECT_ID, Store, new_id, now
from iris.taxonomies import TAXONOMY, publish_taxonomy
from iris.workspace_archive import create_archive
from iris.workspace_restore import inspect_archive, restore_archive


def fixture_workspace(tmp_path, *, custom=False):
    store = Store(tmp_path / "workspace")
    if custom:
        publish_taxonomy(
            store,
            DEFAULT_PROJECT_ID,
            expected_taxonomy_id=TAXONOMY["id"],
            classes=[
                {"id": "helmet", "name": "Helmet", "definition": "A helmet."},
                {"id": "vehicle", "name": "Vehicle", "definition": "A car.", "coco_id": 3},
            ],
        )
    frames = [
        add_frame(store, tmp_path, group=group, color=color)
        for group, color in [("train", (1, 2, 3)), ("val", (4, 5, 6)), ("val", (7, 8, 9))]
    ]
    dataset = create_dataset(
        store,
        name="Synthetic export images",
        frame_ids=[f["id"] for f in frames],
        splits={"train": "train", "val": "val"},
    )
    contract = dataset_contract(dataset["manifest"])
    model_id, training_id, job_id = "trained_" + new_id(), new_id(), new_id()
    relative = f"models/trained/{model_id}.pth"
    checkpoint = store.artifact_path(relative)
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"SYNTHETIC CHECKPOINT: never deserialize this fixture")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    store.insert(
        "jobs",
        {
            "id": job_id,
            "kind": "train",
            "status": "succeeded",
            "params": {"training_id": training_id},
            "created_at": now(),
        },
    )
    store.insert(
        "training_runs",
        {
            "id": training_id,
            "name": "Simulated training",
            "dataset_id": dataset["id"],
            "parent_model_id": TRAINING_ARCHITECTURE,
            "config": {},
            "job_id": job_id,
            "checkpoint_id": model_id,
            "created_at": now(),
        },
    )
    provenance = {
        "training_scene_groups": ["train"],
        "training_frame_hashes": [frames[0]["sha256"]],
        "parent_weight_sha256": "1" * 64,
    }
    model = store.insert(
        "trained_models",
        {
            "id": model_id,
            "name": "Synthetic trained detector",
            "training_id": training_id,
            "parent_model_id": TRAINING_ARCHITECTURE,
            "architecture": TRAINING_ARCHITECTURE,
            "path": relative,
            "weight_sha256": digest,
            "metadata": {**contract, **provenance},
            "created_at": now(),
        },
    )
    official = {**get_spec(TRAINING_ARCHITECTURE), "status": "ready", "weight_sha256": "1" * 64}
    spec = {
        **model,
        **contract,
        "origin": "trained",
        "status": "ready",
        "provenance": provenance,
        "classes": [
            {"id": value, "name": key} for key, value in contract["output_class_mapping"].items()
        ],
    }

    class FakeDetector:
        def __init__(self, *_args, **_kwargs):
            self.metadata = {
                "model_id": model_id,
                "architecture": TRAINING_ARCHITECTURE,
                "weight_sha256": digest,
                "device": "cpu",
                "precision": "float32",
                "head_class_slots": len(contract["class_mapping"]) + 1,
                "input_transform": deepcopy(exports.INPUT_TRANSFORM),
                "native_filtering": deepcopy(exports.NATIVE_FILTERING),
                "torch_version": "2.10.0+cpu",
                "torchvision_version": "0.25.0+cpu",
                "hardware": "Synthetic fixture; no detector executed",
                **contract,
            }

        def warmup(self, _image):
            pass

        def predict(self, image):
            label = "vehicle" if custom else "car"
            detection = {
                "label": label,
                "label_id": contract["output_class_mapping"][label],
                "native_label_id": contract["class_mapping"][label],
                "box": [2, 3, 14, 25],
                "score": 0.75,
            }
            if custom:
                detection["taxonomy_id"] = contract["taxonomy_id"]
            return {
                "input_size": list(image.size),
                "detections": [detection] if image.getpixel((0, 0))[0] == 4 else [],
                "timing": {
                    "preprocess_ms": 1,
                    "inference_ms": 2,
                    "postprocess_ms": 1,
                    "total_ms": 4,
                },
            }

    with patch("iris.evaluation.catalog", return_value=[official, spec]):
        row = evaluation.create_evaluation(
            store,
            JobManager(store),
            name="Synthetic CPU evaluation",
            dataset_id=dataset["id"],
            model_ids=[model_id],
        )
        evaluation.run_evaluation(
            store, row["id"], lambda *_: None, lambda: False, detector_factory=FakeDetector
        )
    return store, model, row, dataset


def options(workspace):
    _store, model, row, dataset = workspace
    return {
        "trained_model_id": model["id"],
        "evaluation_id": row["id"],
        "name": "Portable fixture",
        "frame_ids": [f["frame_id"] for f in dataset["manifest"]["frames"] if f["split"] == "val"],
    }


def queued(workspace):
    store = workspace[0]
    values = options(workspace)
    preview = exports.preview_export(store, **values)
    return exports.create_export(
        store,
        **values,
        request_id=preview["request_id"],
        expected_fingerprint=preview["fingerprint"],
    )


def published(workspace):
    row = queued(workspace)
    result = exports.run_export(workspace[0], row["id"], lambda *_: None, lambda: False)
    assert result["published"]
    return exports.export_detail(workspace[0], row["id"])


def measurement(row):
    frames = row["config"]["reference"]["frames"]
    return {
        "format": "iris-export-measurement-v1",
        "manifest_sha256": row["manifest_sha256"],
        "declaration": "simulation",
        "repeats": 2,
        "load_ms": 100,
        "environment": {
            "python": "3.12.12",
            "torch": "2.10.0+cpu",
            "torchvision": "0.25.0+cpu",
            "pillow": "12.3.0",
            "platform": "Synthetic Linux",
            "machine": "fixture",
            "processor": "Synthetic CPU",
            "cpu_count": 8,
            "threads": 4,
            "interop_threads": 8,
            "device": "cpu",
            "precision": "float32",
            "batch_size": 1,
        },
        "warmup": {"frame_id": frames[0]["frame_id"], "duration_ms": 50},
        "samples": [
            {
                "frame_id": frame["frame_id"],
                "repeat": repeat,
                "input_size": frame["input_size"],
                "detections": deepcopy(frame["detections"]),
                "decode_ms": 2,
                "timing": {
                    "preprocess_ms": 1,
                    "inference_ms": 2,
                    "postprocess_ms": 1,
                    "total_ms": 4,
                },
            }
            for repeat in (1, 2)
            for frame in frames
        ],
    }


@pytest.fixture
def workspace(tmp_path):
    return fixture_workspace(tmp_path)


@pytest.mark.parametrize("custom", [False, True])
@pytest.mark.parametrize("legacy_notice", [False, True])
def test_bundle_independent_inspect_classes_and_archive_restore(
    tmp_path, custom, legacy_notice, monkeypatch
):
    workspace = fixture_workspace(tmp_path, custom=custom)
    store = workspace[0]
    # Older bundles have a frozen README without the subsequently added MIT text.
    with monkeypatch.context() as previous_version:
        if legacy_notice:
            resources = exports._resources()
            resources["README.md"] = exports.README
            previous_version.setattr(exports, "_resources", lambda: resources)
        row = published(workspace)
    path = exports.download_path(store, row["id"])
    manifest, reference = exports.read_bundle(path)
    assert manifest["validation"]["real_execution"] == "not_run"
    assert reference["frames"][1]["detections"] == []
    detection = reference["frames"][0]["detections"][0]
    assert detection["native_label_id"] == 2
    assert detection["label_id"] == (2 if custom else 3)
    assert ("taxonomy_id" in detection) == custom
    outside = tmp_path / "independent"
    with zipfile.ZipFile(path) as archive:
        license_text = Path(exports.__file__).with_name("LICENSE.txt").read_bytes()
        assert (license_text in archive.read("README.md")) is not legacy_notice
        archive.extractall(outside)
        for name in archive.namelist():
            if name.endswith(".json"):
                assert b"reviewer" not in archive.read(name)
                assert b"annotation_revision" not in archive.read(name)
    # -I -S removes site packages and IRIS: this verifies the stdlib-only copied runner.
    result = subprocess.run(
        [sys.executable, "-I", "-S", str(outside / "run.py"), "inspect"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    payload = measurement(row)
    preview = exports.preview_measurement(store, row["id"], payload)
    saved = exports.save_measurement(store, row["id"], payload, preview["fingerprint"])
    assert saved["summary"]["parity_passed"] is True
    assert saved["summary"]["execution_verified"] is False
    assert (
        exports.save_measurement(store, row["id"], payload, preview["fingerprint"])["id"]
        == saved["id"]
    )
    archive = tmp_path / "workspace.zip"
    create_archive(store.root, archive)
    inspection = inspect_archive(archive)
    destination = tmp_path / "restored"
    restore_archive(archive, destination, expected_archive_sha256=inspection["archive_sha256"])
    restored = Store(destination)
    for table in ("model_exports", "model_export_measurements"):
        assert restored.list(table) == store.list(table)
    assert exports.download_path(restored, row["id"]).read_bytes() == path.read_bytes()
    assert job_detail(store, row["job_id"])["artifacts"][0]["count"] == 1


def test_preview_freshness_idempotent_create_and_mutated_checkpoint(workspace):
    store = workspace[0]
    values = options(workspace)
    preview = exports.preview_export(store, **values)
    kwargs = {
        **values,
        "request_id": preview["request_id"],
        "expected_fingerprint": preview["fingerprint"],
    }
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(lambda _: exports.create_export(store, **kwargs), range(2)))
    assert rows[0]["id"] == rows[1]["id"]
    with pytest.raises(ValueError, match="different"):
        exports.create_export(store, **{**kwargs, "name": "Changed"})
    checkpoint = store.artifact_path(workspace[1]["path"])
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Checkpoint bytes changed"):
        exports.run_export(store, rows[0]["id"], lambda *_: None, lambda: False)
    assert store.get("model_exports", rows[0]["id"])["path"] is None
    assert not list((store.root / "model_exports").iterdir())


@pytest.mark.parametrize("when", ["initial", "copy", "late", "stopped"])
def test_cancellation_never_publishes_partial_package(workspace, when):
    store = workspace[0]
    row = queued(workspace)
    cancelled = when == "initial"

    def progress(*_):
        nonlocal cancelled
        if when == "copy":
            cancelled = True
        if when == "late":
            store.update("jobs", row["job_id"], {"cancel_requested": True})
        if when == "stopped":
            store.update("jobs", row["job_id"], {"status": "interrupted"})

    result = exports.run_export(store, row["id"], progress, lambda: cancelled)
    assert result["cancelled"] and not result["published"]
    assert store.get("model_exports", row["id"])["path"] is None
    assert not list((store.root / "model_exports").iterdir())


@pytest.mark.parametrize("change", ["score", "empty", "box"])
def test_changed_predictions_remain_importable_as_failed_parity(workspace, change):
    store, row = workspace[0], published(workspace)
    payload = measurement(row)
    if change == "score":
        payload["samples"][0]["detections"][0]["score"] -= 1e-9
    elif change == "box":
        payload["samples"][0]["detections"][0]["box"][0] += 1e-9
    else:
        payload["samples"][0]["detections"] = []
    preview = exports.preview_measurement(store, row["id"], payload)
    assert preview["summary"]["parity_passed"] is False
    saved = exports.save_measurement(store, row["id"], payload, preview["fingerprint"])
    assert len(saved["summary"]["mismatched_samples"]) == 1


def test_api_scope_bounded_import_and_recovery(workspace, tmp_path):
    store = workspace[0]
    with TestClient(create_app(store.root, run_jobs=False), base_url="http://127.0.0.1") as api:
        response = api.get("/api/model-exports/candidates")
        assert response.status_code == 200, response.text
        assert response.json()["models"][0]["eligible"]
        preview = api.post("/api/model-exports/preview", json=options(workspace)).json()
        creation = api.post(
            "/api/model-exports",
            json={
                **options(workspace),
                "request_id": preview["request_id"],
                "expected_fingerprint": preview["fingerprint"],
            },
        )
        assert creation.status_code == 202, creation.text
        row = creation.json()
        exports.run_export(store, row["id"], lambda *_: None, lambda: False)
        row = api.get(f"/api/model-exports/{row['id']}").json()
        assert row["ready"]
        assert api.get(f"/api/model-exports/{row['id']}/download").status_code == 200
        route = f"/api/model-exports/{row['id']}/measurements"
        payload = measurement(row)
        check = api.post(route + "/preview", json=payload)
        assert check.status_code == 200, check.text
        fingerprint = check.json()["fingerprint"]
        saved = api.post(route, params={"expected_fingerprint": fingerprint}, json=payload)
        assert saved.status_code == 201, saved.text
        assert (
            api.get("/api/model-export-measurements/" + saved.json()["id"]).json()["payload"]
            == payload
        )
        assert (
            api.post(
                route + "/preview",
                content=b'{"a":1,"a":2}',
                headers={"Content-Type": "application/json"},
            ).status_code
            == 409
        )
        assert (
            api.post(
                route + "/preview",
                content=b"{}",
                headers={
                    "Content-Type": "application/json",
                    "Content-Length": str(exports.MAX_MEASUREMENT_BYTES + 1),
                },
            ).status_code
            == 413
        )
        other = api.post("/api/projects", json={"name": "Other"}).json()["id"]
        for endpoint in (
            f"/api/model-exports/{row['id']}",
            f"/api/model-exports/{row['id']}/download",
            "/api/model-export-measurements/" + saved.json()["id"],
        ):
            assert api.get(endpoint, params={"project_id": other}).status_code == 404
        assert api.get("/api/model-exports", params={"project_id": other}).json() == []
