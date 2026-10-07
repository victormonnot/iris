"""YOLOX export integrity and lifecycle; conversion/model execution are simulated."""

import hashlib
import json
import subprocess
import sys
import zipfile
from copy import deepcopy
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from test_model_exports import fixture_workspace, options

from iris import model_exports as exports
from iris import yolox_export_runner as runner
from iris import yolox_exports
from iris.app import create_app
from iris.store import Store
from iris.workspace_archive import create_archive
from iris.workspace_restore import inspect_archive, restore_archive
from iris.yolox_spec import INPUT_TRANSFORM, NATIVE_FILTERING


@pytest.fixture
def workspace(tmp_path):
    """Adapt the existing synthetic saved evaluation, never load a checkpoint."""
    store, model, evaluation, dataset = fixture_workspace(tmp_path, custom=True)
    model = store.update(
        "trained_models",
        model["id"],
        {
            "architecture": "yolox_nano",
            "parent_model_id": "yolox_nano",
            "metadata": {**model["metadata"], "architecture": "yolox_nano"},
        },
    )
    store.update("training_runs", model["training_id"], {"parent_model_id": "yolox_nano"})
    for saved in store.list("evaluation_models", evaluation_id=evaluation["id"]):
        store.update(
            "evaluation_models",
            saved["id"],
            {
                "metadata": {
                    **saved["metadata"],
                    "architecture": "yolox_nano",
                    "input_transform": deepcopy(INPUT_TRANSFORM),
                    "native_filtering": deepcopy(NATIVE_FILTERING),
                }
            },
        )
    return store, model, evaluation, dataset


def fake_conversion(_store, _plan, directory, progress, _cancelled):
    (directory / "model.onnx").write_bytes(b"SYNTHETIC ONNX: inspect only; never execute")
    progress(0.6, "Synthetic conversion fixture")
    return {
        "raw_output_equivalence": True,
        "rtol": 0.001,
        "atol": 0.001,
        "frames": [],
        "device": "cpu",
        "exact_saved_prediction_parity": "not_run",
        "evidence_kind": "test_fixture_no_conversion_executed",
    }


def publish(workspace):
    store = workspace[0]
    preview = exports.preview_export(store, **options(workspace))
    row = exports.create_export(
        store,
        **options(workspace),
        request_id=preview["request_id"],
        expected_fingerprint=preview["fingerprint"],
    )
    with patch("iris.yolox_exports._convert", side_effect=fake_conversion):
        result = exports.run_export(store, row["id"], lambda *_: None, lambda: False)
    assert result["published"]
    return exports.export_detail(store, row["id"])


def measurement(row):
    return {
        "format": "iris-yolox-measurement-v1",
        "manifest_sha256": row["manifest_sha256"],
        "device": "cpu",
        "repeats": 2,
        "evidence_kind": "simulation",
        "environment": {"python": "3.12.3", "opencv": "synthetic", "platform": "fixture"},
        "samples": [
            {
                "frame_id": frame["frame_id"],
                "repeat": repeat,
                "prediction": {
                    "input_size": frame["input_size"],
                    "detections": deepcopy(frame["detections"]),
                    "timing": {"total_ms": 4.0, "decode_ms": 1.0},
                },
            }
            for repeat in range(2)
            for frame in row["config"]["reference"]["frames"]
        ],
    }


def test_onnx_measurement_report_preserves_checkpoint_and_unknown_stage_timings(workspace):
    from iris.experiment_export import render_experiment_html
    from iris.experiments import create_experiment, preview_experiment

    store = workspace[0]
    row = publish(workspace)
    payload = measurement(row)
    preview = exports.preview_measurement(store, row["id"], payload)
    saved = exports.save_measurement(store, row["id"], payload, preview["fingerprint"])
    preview = preview_experiment(store, workspace[2]["id"])
    assert len(preview["available_measurements"]) == 1
    candidate = preview["available_measurements"][0]
    assert candidate["model_sha256"] == row["config"]["model"]["sha256"]
    assert candidate["model_sha256"] != row["manifest"]["model"]["sha256"]
    assert candidate["summary"]["frames"] == 2
    assert candidate["summary"]["timing_ms"]["total_ms"]["median"] == 4
    assert candidate["summary"]["timing_ms"]["inference_ms"]["median"] is None
    assert candidate["summary"]["execution_verified"] is False
    report = create_experiment(
        store,
        evaluation_id=workspace[2]["id"],
        title="Synthetic ONNX evidence",
        measurement_ids=[saved["id"]],
        expected_source_fingerprint=preview["source_fingerprint"],
    )
    html = render_experiment_html(store, report["id"], expected_revision=1)
    assert b"OpenCV" in html and b"SIMULATION" in html


def test_yolox_api_export_measurement_and_archive_roundtrip(workspace, tmp_path):
    store = workspace[0]
    with TestClient(create_app(store.root, run_jobs=False), base_url="http://127.0.0.1") as api:
        candidates = api.get("/api/model-exports/candidates")
        assert candidates.status_code == 200, candidates.text
        assert any(
            item["eligible"] and item["id"] == workspace[1]["id"]
            for item in candidates.json()["models"]
        )
        response = api.post("/api/model-exports/preview", json=options(workspace))
        assert response.status_code == 200, response.text
        preview = response.json()
        response = api.post(
            "/api/model-exports",
            json={
                **options(workspace),
                "request_id": preview["request_id"],
                "expected_fingerprint": preview["fingerprint"],
            },
        )
        assert response.status_code == 202, response.text
        with patch("iris.yolox_exports._convert", side_effect=fake_conversion):
            exports.run_export(store, response.json()["id"], lambda *_: None, lambda: False)
        row = api.get(f"/api/model-exports/{response.json()['id']}").json()
        assert row["ready"]
        response = api.get(f"/api/model-exports/{row['id']}/download")
        assert response.status_code == 200
        payload = measurement(row)
        route = f"/api/model-exports/{row['id']}/measurements"
        preview = api.post(route + "/preview", json=payload)
        assert preview.status_code == 200, preview.text
        response = api.post(
            route, params={"expected_fingerprint": preview.json()["fingerprint"]}, json=payload
        )
        assert response.status_code == 201, response.text
        saved = response.json()
        assert saved["summary"]["parity_passed"] is True
        assert saved["summary"]["execution_verified"] is False
        assert api.get("/api/model-export-measurements/" + saved["id"]).json()["payload"] == payload

    bundle = exports.download_path(store, row["id"])
    manifest, reference = exports.read_bundle(bundle)
    assert manifest["format"] == runner.FORMAT
    assert manifest["input"]["color"] == "BGR"
    assert manifest["output"] == {
        "name": "output",
        "shape": [1, 3549, 7],
        "encoding": "yolox_raw_grid",
        "strides": [8, 16, 32],
    }
    assert [(c["index"], c["id"], c["category_id"]) for c in manifest["classes"]] == [
        (0, "helmet", 1),
        (1, "vehicle", 2),
    ]
    assert reference["frames"][0]["detections"][0]["native_label_id"] == 2
    assert reference["frames"][1]["detections"] == []
    independent = tmp_path / "standalone"
    with zipfile.ZipFile(bundle) as archive:
        archive.extractall(independent)
    result = subprocess.run(
        [sys.executable, "-I", "-S", str(independent / "run.py"), "inspect"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == manifest

    backup = tmp_path / "workspace.zip"
    create_archive(store.root, backup)
    inspection = inspect_archive(backup)
    destination = tmp_path / "restored"
    restore_archive(backup, destination, expected_archive_sha256=inspection["archive_sha256"])
    restored = Store(destination)
    for table in ("model_exports", "model_export_measurements"):
        assert restored.list(table) == store.list(table)
    assert exports.download_path(restored, row["id"]).read_bytes() == bundle.read_bytes()


def test_yolox_exact_parity_failure_remains_saved_as_declared_evidence(workspace):
    row = publish(workspace)
    payload = measurement(row)
    payload["samples"][0]["prediction"]["detections"][0]["score"] -= 1e-9
    payload["summary"] = {"parity_passed": True, "execution_verified": True}
    preview = exports.preview_measurement(workspace[0], row["id"], payload)
    assert preview["summary"]["parity_passed"] is False
    assert preview["summary"]["execution_verified"] is False
    saved = exports.save_measurement(workspace[0], row["id"], payload, preview["fingerprint"])
    assert saved["summary"] == preview["summary"]
    assert (
        exports.save_measurement(workspace[0], row["id"], payload, preview["fingerprint"])["id"]
        == saved["id"]
    )


@pytest.mark.parametrize(
    "change", ["shape", "decoding", "color", "class_slot", "duplicate_category", "traversal"]
)
def test_yolox_contract_rejects_ambiguous_or_unsafe_mapping(workspace, change):
    row = publish(workspace)
    manifest = deepcopy(row["manifest"])
    if change == "shape":
        manifest["output"]["shape"][-1] = 85
    elif change == "decoding":
        manifest["output"]["encoding"] = "yolox_decoded"
    elif change == "color":
        manifest["input"]["color"] = "RGB"
    elif change == "class_slot":
        manifest["classes"][0]["index"] = 1
    elif change == "duplicate_category":
        manifest["classes"][1]["category_id"] = manifest["classes"][0]["category_id"]
    else:
        manifest["files"]["../outside.py"] = manifest["files"].pop("run.py")
    with pytest.raises(ValueError):
        runner.validate_manifest(manifest)


@pytest.mark.parametrize(
    "change",
    ["box", "score", "label", "native_label", "dimensions", "object", "nonfinite", "order"],
)
def test_yolox_measurement_rejects_malformed_predictions(workspace, change):
    row = publish(workspace)
    payload = measurement(row)
    prediction = payload["samples"][0]["prediction"]
    if change == "box":
        prediction["detections"][0]["box"] = [14, 3, 2, 25]
    elif change == "score":
        prediction["detections"][0]["score"] = 1.1
    elif change == "label":
        prediction["detections"][0]["label"] = "unmapped-class"
    elif change == "native_label":
        prediction["detections"][0]["native_label_id"] = 80
    elif change == "dimensions":
        prediction["input_size"] = [-1, 32]
    elif change == "object":
        prediction["detections"] = [None]
    elif change == "nonfinite":
        prediction["detections"][0]["score"] = float("nan")
    else:
        payload["samples"][0]["repeat"] = 1
    with pytest.raises(ValueError):
        exports.preview_measurement(workspace[0], row["id"], payload)
    assert workspace[0].list("model_export_measurements") == []


def test_yolox_bundle_checks_bytes_and_rejects_reference_traversal(workspace, tmp_path):
    row = publish(workspace)
    bundle = exports.download_path(workspace[0], row["id"])
    directory = tmp_path / "bundle"
    with zipfile.ZipFile(bundle) as archive:
        archive.extractall(directory)
    model = directory / "model.onnx"
    original = model.read_bytes()
    model.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    with pytest.raises(ValueError, match="changed"):
        runner.inspect_bundle(directory)
    model.write_bytes(original)
    manifest = deepcopy(row["manifest"])
    reference = deepcopy(row["config"]["reference"])
    reference["frames"][0]["path"] = "../outside.png"
    raw = runner.canonical(reference)
    (directory / "parity/reference.json").write_bytes(raw)
    manifest["files"]["parity/reference.json"] = {
        "size": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    (directory / "manifest.json").write_bytes(runner.canonical(manifest))
    with pytest.raises(ValueError):
        yolox_exports.validate_reference(manifest, reference)
    # A bundle with consistent inventory hashes still cannot redirect the standalone runner.
    with pytest.raises(ValueError):
        runner.inspect_bundle(directory)


def test_yolox_checkpoint_freshness_and_cancelled_conversion_do_not_publish(workspace):
    store = workspace[0]
    preview = exports.preview_export(store, **options(workspace))
    row = exports.create_export(
        store,
        **options(workspace),
        request_id=preview["request_id"],
        expected_fingerprint=preview["fingerprint"],
    )
    with patch("iris.yolox_exports._convert") as convert:
        result = exports.run_export(store, row["id"], lambda *_: None, lambda: True)
    assert result["cancelled"] and not result["published"]
    convert.assert_not_called()
    assert store.get("model_exports", row["id"])["path"] is None
    store.artifact_path(workspace[1]["path"]).write_bytes(b"Different checkpoint")
    with pytest.raises(ValueError, match="checkpoint bytes changed"):
        exports.run_export(store, row["id"], lambda *_: None, lambda: False)
    assert store.get("model_exports", row["id"])["path"] is None
    assert not list((store.root / "model_exports").iterdir())


@pytest.mark.parametrize("taxonomy_id", [None, "iris-objects-v1"])
def test_yolox_runner_decodes_once_and_keeps_only_best_class(
    workspace, tmp_path, monkeypatch, taxonomy_id
):
    import cv2
    import numpy as np
    from PIL import Image

    row = publish(workspace)
    directory = tmp_path / "runner"
    with zipfile.ZipFile(exports.download_path(workspace[0], row["id"])) as archive:
        archive.extractall(directory)
    if taxonomy_id:
        manifest = json.loads((directory / "manifest.json").read_text())
        manifest["taxonomy_id"] = taxonomy_id
        (directory / "manifest.json").write_bytes(runner.canonical(manifest))
    raw = np.zeros((1, 3549, 7), dtype=np.float32)
    # The first raw-grid candidate spans 0..16 in network pixels, hence 0..8 in this image.
    # Both classes exceed the threshold: only the best class may survive for one anchor.
    raw[0, 0] = [1, 1, np.log(2), np.log(2), 0.8, 0.7, 0.9]

    class Network:
        tensor = None

        def setPreferableBackend(self, _backend):
            pass

        def setInput(self, tensor):
            self.tensor = tensor

        def forward(self, name):
            assert name == "output"
            return raw.copy()

    network = Network()
    monkeypatch.setattr(cv2.dnn, "readNetFromONNX", lambda *_: network)
    image = tmp_path / "rgb.png"
    Image.new("RGB", (208, 104), (1, 2, 3)).save(image)
    result = runner.Runner(directory).predict(image)
    assert network.tensor.shape == (1, 3, 416, 416)
    assert network.tensor.dtype == np.float32
    assert network.tensor[0, :, 0, 0].tolist() == [3, 2, 1]
    assert network.tensor[0, :, 208, 0].tolist() == [114, 114, 114]
    assert len(result["detections"]) == 1
    detection = result["detections"][0]
    assert (detection["label"], detection["native_label_id"], detection["label_id"]) == (
        "vehicle",
        2,
        2,
    )
    assert detection["box"] == pytest.approx([0, 0, 8, 8])
    assert detection["score"] == pytest.approx(0.72)
    assert detection["taxonomy_id"] == (taxonomy_id or row["manifest"]["taxonomy_id"])


def test_yolox_export_supports_full_project_class_limit(workspace):
    manifest = deepcopy(publish(workspace)["manifest"])
    manifest["classes"] = [
        {"index": i, "id": f"class-{i}", "name": f"Class {i}", "category_id": i + 1}
        for i in range(100)
    ]
    manifest["output"]["shape"] = [1, 3549, 105]
    runner.validate_manifest(manifest)
    manifest["classes"].append(
        {"index": 100, "id": "overflow", "name": "Overflow", "category_id": 101}
    )
    with pytest.raises(ValueError, match="1–100"):
        runner.validate_manifest(manifest)


@pytest.mark.parametrize("environment", [None, [], {"opencv": {"nested": "invalid"}}])
def test_yolox_measurement_requires_reportable_environment(workspace, environment):
    row = publish(workspace)
    payload = measurement(row)
    if environment is None:
        del payload["environment"]
    else:
        payload["environment"] = environment
    with pytest.raises(ValueError, match="environment"):
        exports.preview_measurement(workspace[0], row["id"], payload)
