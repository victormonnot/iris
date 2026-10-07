"""Adapter/provisioning fixtures; live checkpoint tests are explicitly opt-in."""

import builtins
import hashlib
import importlib.metadata
import io
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from iris import models

SSD = "ssdlite320_mobilenet_v3_large"
FRCNN = "fasterrcnn_mobilenet_v3_large_320_fpn"


@pytest.fixture
def checkpoint(monkeypatch):
    payload = b"synthetic checkpoint for download-integrity tests, not model parameters"
    digest = hashlib.sha256(payload).hexdigest()
    specs = {name: dict(value) for name, value in models._SPECS.items()}
    for spec in specs.values():
        spec.update(download_bytes=len(payload), expected_hash_prefix=digest[:8])
    monkeypatch.setattr(models, "_SPECS", specs)
    return payload, digest


def write_checkpoint(root, payload, model_id=SSD):
    path = root / "models" / models.get_spec(model_id)["weight_filename"]
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(payload)
    return path


class Download(io.BytesIO):
    def __init__(self, payload, length=None):
        super().__init__(payload)
        self.headers = {} if length is None else {"Content-Length": str(length)}


def test_catalog_does_not_import_runtime_or_contact_network(monkeypatch, tmp_path):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert name.split(".")[0] not in {"torch", "torchvision"}
        return original_import(name, *args, **kwargs)

    def absent(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(importlib.metadata, "version", absent)
    monkeypatch.setattr(
        models, "urlopen", lambda *a, **kw: pytest.fail("Unexpected network request")
    )
    rows = models.catalog(tmp_path)
    assert [row["id"] for row in rows] == [SSD, FRCNN, "yolox_nano"]
    assert all(row["status"] == "missing_runtime" for row in rows)
    assert all(row["runtime_load_verified"] is False for row in rows)
    assert not (tmp_path / "models").exists()


def test_spec_preserves_coco_ids_and_returns_independent_values():
    spec = models.get_spec(SSD)
    assert len(spec["classes"]) == 80
    assert {row["id"]: row["name"] for row in spec["classes"]}[3] == "car"
    assert spec["classes"][-1] == {"id": 90, "name": "toothbrush"}
    spec["classes"].clear()
    assert len(models.get_spec(SSD)["classes"]) == 80
    with pytest.raises(ValueError, match="Unknown detector"):
        models.get_spec("../../other.pth")


def test_runtime_accepts_cpu_build_suffix_and_reports_incompatible_version(monkeypatch):
    monkeypatch.setattr(
        importlib.metadata, "version", lambda name: models.RUNTIME_VERSIONS[name] + "+cpu"
    )
    assert models._runtime_problem() is None
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.1")
    assert "expected 2.10.0" in models._runtime_problem()


def test_catalog_missing_valid_invalid_and_modified_weights(tmp_path, checkpoint, monkeypatch):
    monkeypatch.setattr(models, "_runtime_problem", lambda: None)
    payload, digest = checkpoint
    assert models.catalog(tmp_path)[0]["status"] == "missing_weights"
    path = write_checkpoint(tmp_path, payload)
    row = models.catalog(tmp_path)[0]
    assert row["status"] == "ready"
    assert row["weight_sha256"] == digest
    path.write_bytes(b"x" * len(payload))
    row = models.catalog(tmp_path)[0]
    assert row["status"] == "invalid_weights"
    assert "SHA-256" in row["reason"]
    path.write_bytes(b"truncated")
    assert "size" in models.catalog(tmp_path)[0]["reason"]


def test_explicit_download_is_verified_atomic_and_has_provenance(tmp_path, checkpoint, monkeypatch):
    payload, digest = checkpoint
    updates = []

    def open_download(request, timeout):
        assert request.full_url == models.get_spec(SSD)["weight_url"]
        assert request.data is None
        assert timeout == 30
        assert not (tmp_path / "models" / models.get_spec(SSD)["weight_filename"]).exists()
        return Download(payload, len(payload))

    monkeypatch.setattr(models, "urlopen", open_download)
    record = models.download_model(tmp_path, SSD, lambda done, total: updates.append((done, total)))
    assert record["weight_sha256"] == digest
    assert record["downloaded"] is True
    assert record["verified_at"]
    path = tmp_path / "models" / models.get_spec(SSD)["weight_filename"]
    assert path.read_bytes() == payload
    assert json.loads(path.with_suffix(".pth.json").read_text()) == record
    assert updates == [(len(payload), len(payload))]
    assert not list(path.parent.glob("*.part"))
    monkeypatch.setattr(models, "urlopen", lambda *a, **kw: pytest.fail("Repeated network request"))
    assert models.download_model(tmp_path, SSD) == record


@pytest.mark.parametrize("failure", ["hash", "truncated", "oversize", "header", "interrupted"])
def test_failed_download_preserves_previous_file_and_removes_part(
    tmp_path, checkpoint, monkeypatch, failure
):
    payload, _ = checkpoint
    path = write_checkpoint(tmp_path, b"previous corrupt checkpoint")
    downloads = {
        "hash": Download(b"x" * len(payload)),
        "truncated": Download(payload[:-1]),
        "oversize": Download(payload + b"x"),
        "header": Download(payload, len(payload) + 1),
        "interrupted": Download(payload),
    }
    monkeypatch.setattr(models, "urlopen", lambda *a, **kw: downloads[failure])

    def progress(done, total):
        if failure == "interrupted":
            raise RuntimeError("Cancelled fixture download")

    with pytest.raises((ValueError, RuntimeError)):
        models.download_model(tmp_path, SSD, progress)
    assert path.read_bytes() == b"previous corrupt checkpoint"
    assert not list(path.parent.glob("*.part"))
    assert not path.with_suffix(".pth.json").exists()


def test_existing_manually_installed_checkpoint_is_verified_without_network(
    tmp_path, checkpoint, monkeypatch
):
    payload, digest = checkpoint
    write_checkpoint(tmp_path, payload)
    monkeypatch.setattr(models, "urlopen", lambda *a, **kw: pytest.fail("Unexpected request"))
    record = models.download_model(tmp_path, SSD)
    assert record["weight_sha256"] == digest
    assert record["downloaded"] is False


class OutputTensor:
    def __init__(self, values):
        self.values = values

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.values


def fixture_output(boxes=None, labels=None, scores=None):
    return {
        "boxes": OutputTensor([[0.0, 0.0, 120.0, 80.0]] if boxes is None else boxes),
        "labels": OutputTensor([3] if labels is None else labels),
        "scores": OutputTensor([0.002] if scores is None else scores),
    }


def test_serialization_retains_low_scores_all_classes_and_exclusive_edges():
    output = fixture_output(
        boxes=[[0.0, 0.0, 120.0, 80.0], [1.25, 2.5, 30.0, 40.0]],
        labels=[3, 90],
        scores=[0.002, 0.7],
    )
    rows = models._serialize_predictions(output, (120, 80))
    assert rows[0] == {
        "box": [0.0, 0.0, 120.0, 80.0],
        "label_id": 3,
        "label": "car",
        "score": 0.002,
    }
    assert rows[1]["label"] == "toothbrush"
    assert rows[1]["box"] == [1.25, 2.5, 30.0, 40.0]
    assert models._serialize_predictions(fixture_output([], [], []), (120, 80)) == []


@pytest.mark.parametrize(
    "output",
    [
        fixture_output(scores=[float("nan")]),
        fixture_output(scores=[1.1]),
        fixture_output(boxes=[[0, 0, 121, 80]]),
        fixture_output(boxes=[[4, 0, 3, 80]]),
        fixture_output(labels=[91]),
        fixture_output(labels=[3.0]),
        fixture_output(scores=[]),
    ],
)
def test_serialization_rejects_invalid_outputs(output):
    with pytest.raises(ValueError):
        models._serialize_predictions(output, (120, 80))


def test_missing_runtime_fails_before_import_or_download(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "_runtime_problem", lambda: "torch is not installed")
    with pytest.raises(RuntimeError, match="not installed"):
        models.TorchvisionDetector(tmp_path, SSD)
    with pytest.raises(ValueError, match="Device must"):
        models.TorchvisionDetector(tmp_path, SSD, "arbitrary")


def test_predict_applies_orientation_once_and_reports_real_stage_durations(monkeypatch):
    image = Image.new("RGB", (120, 80))
    image.getexif()[274] = 6  # Rotated upright dimensions must be 80 x 120.
    seen_sizes = []
    calls = []

    class InputTensor:
        def to(self, **kwargs):
            return self

        def __truediv__(self, value):
            assert value == 255
            return self

    def pil_to_tensor(oriented):
        seen_sizes.append(oriented.size)
        assert oriented.mode == "RGB"
        return InputTensor()

    def forward(tensors):
        calls.append(len(tensors))
        return [fixture_output(boxes=[[0, 0, 80, 120]])]

    detector = object.__new__(models.TorchvisionDetector)
    detector.device = SimpleNamespace(type="cpu")
    detector.functional = SimpleNamespace(pil_to_tensor=pil_to_tensor)
    detector.torch = SimpleNamespace(float32="float32", inference_mode=nullcontext)
    detector.model = forward
    counter = iter([10.0, 10.01, 10.05, 10.06])
    monkeypatch.setattr(models.time, "perf_counter", lambda: next(counter))
    result = detector.predict(image)
    assert seen_sizes == [(80, 120)]
    assert result["input_size"] == [80, 120]
    assert result["detections"][0]["box"] == [0, 0, 80, 120]
    assert calls == [1]
    assert result["timing"] == pytest.approx(
        {"preprocess_ms": 10, "inference_ms": 40, "postprocess_ms": 10, "total_ms": 60}
    )


def test_cuda_synchronization_boundaries():
    calls = []
    detector = object.__new__(models.TorchvisionDetector)
    detector.device = SimpleNamespace(type="cuda")
    detector.torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=calls.append))
    detector._synchronize()
    assert calls == [detector.device]


def test_frozen_batchnorm_restoration_matches_official_architecture():
    class Node:
        def named_children(self):
            return [(name, value) for name, value in vars(self).items() if isinstance(value, Node)]

    class BatchNorm(Node):
        num_features = 12

    class FrozenBatchNorm(Node):
        def __init__(self, features, eps):
            self.num_features = features
            self.eps = eps

    backbone = Node()
    backbone.block = Node()
    backbone.block.norm = BatchNorm()
    models._restore_frozen_batchnorm(
        backbone,
        SimpleNamespace(nn=SimpleNamespace(BatchNorm2d=BatchNorm)),
        SimpleNamespace(
            ops=SimpleNamespace(misc=SimpleNamespace(FrozenBatchNorm2d=FrozenBatchNorm))
        ),
    )
    assert isinstance(backbone.block.norm, FrozenBatchNorm)
    assert backbone.block.norm.num_features == 12
    assert backbone.block.norm.eps == 1e-5


@pytest.mark.parametrize("model_id", [SSD, FRCNN])
def test_constructor_only_loads_explicit_local_weights(tmp_path, checkpoint, monkeypatch, model_id):
    payload, digest = checkpoint
    path = write_checkpoint(tmp_path, payload, model_id)
    monkeypatch.setattr(models, "_runtime_problem", lambda: None)
    captured = {}

    class Device:
        type = "cpu"

        def __init__(self, value):
            assert value == "cpu"

        def __str__(self):
            return "cpu"

    class Model:
        transform = SimpleNamespace(
            image_mean=[0.5] * 3,
            image_std=[0.5] * 3,
            min_size=(320,),
            max_size=640,
            fixed_size=None,
            size_divisible=32,
        )
        backbone = SimpleNamespace(named_children=lambda: [])
        topk_candidates = 300

        def load_state_dict(self, state, *, strict):
            assert state == {"fixture": "parameters"}
            assert strict is True
            captured["loaded"] = True

        def eval(self):
            return self

        def to(self, device):
            return self

    def builder(**kwargs):
        captured["builder"] = kwargs
        return Model()

    def torch_load(source, *, map_location, weights_only):
        assert source == path
        assert map_location == "cpu"
        assert weights_only is True
        return {"fixture": "parameters"}

    runtime = SimpleNamespace(
        __version__="2.10.0+cpu",
        device=Device,
        load=torch_load,
        set_num_threads=lambda threads: captured.update(threads=threads),
        get_num_threads=lambda: 4,
        get_num_interop_threads=lambda: 2,
    )
    vision = SimpleNamespace(
        __version__="0.25.0+cpu",
        transforms=SimpleNamespace(functional="fixture transform"),
        models=SimpleNamespace(detection=SimpleNamespace(**{model_id: builder})),
    )
    monkeypatch.setitem(sys.modules, "torch", runtime)
    monkeypatch.setitem(sys.modules, "torchvision", vision)
    monkeypatch.setattr(models, "urlopen", lambda *a, **kw: pytest.fail("Unexpected download"))
    detector = models.TorchvisionDetector(tmp_path, model_id)
    options = captured["builder"]
    assert options["weights"] is None
    assert options["weights_backbone"] is None
    assert options["num_classes"] == 91
    assert captured["loaded"] is True
    assert 1 <= captured["threads"] <= 4
    assert detector.metadata["weight_sha256"] == digest
    assert detector.metadata["weight_source"] == models.get_spec(model_id)["weight_url"]
    assert detector.metadata["native_filtering"]["score_threshold"] == 0.001
    assert "NMS" in detector.metadata["timing_protocol"]["inference_ms"]


@pytest.mark.parametrize("model_id", [SSD, FRCNN])
def test_live_local_official_checkpoint(model_id, monkeypatch):
    """Opt-in real checkpoint and reference parity on synthetic pixels, no quality claim."""
    location = os.environ.get("IRIS_TEST_MODEL_DIR")
    if not location:
        pytest.skip("Set IRIS_TEST_MODEL_DIR to an explicitly provisioned local data directory")
    root = Path(location)
    detector = models.TorchvisionDetector(root, model_id, device="cpu")
    image = Image.new("RGB", (400, 240), "gray")
    image.paste("navy", (21, 30, 125, 220))
    image.paste("orange", (210, 100, 390, 225))
    detector.warmup(image)
    result = detector.predict(image)
    assert result["input_size"] == [400, 240]
    assert result["timing"]["total_ms"] > 0
    assert len(detector.metadata["weight_sha256"]) == 64
    assert detector.metadata["head_class_slots"] == 91
    assert detector.metadata["device"] == "cpu"

    # Compare against the official pretrained builder with the same cutoffs.
    # Intercept the enum's loader so reference construction can only read the
    # already verified local checkpoint; any fallback network path fails.
    import torch
    import torchvision

    def forbid_download(*args, **kwargs):
        pytest.fail("The live parity test must never download a checkpoint")

    monkeypatch.setattr(torchvision.models._api, "load_state_dict_from_url", forbid_download)
    monkeypatch.setattr(torch.hub, "download_url_to_file", forbid_download)
    detection = torchvision.models.detection
    if model_id == SSD:
        weights = detection.SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
        options = dict(score_thresh=0.001, nms_thresh=0.5, detections_per_img=100)
    else:
        weights = detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1
        options = dict(box_score_thresh=0.001, box_nms_thresh=0.5, box_detections_per_img=100)
    checkpoint_path = root / "models" / models.get_spec(model_id)["weight_filename"]
    monkeypatch.setattr(
        weights,
        "get_state_dict",
        lambda **kwargs: torch.load(checkpoint_path, map_location="cpu", weights_only=True),
    )
    reference = getattr(detection, model_id)(
        weights=weights, weights_backbone=None, **options
    ).eval()
    assert {name: type(module) for name, module in detector.model.named_modules()} == {
        name: type(module) for name, module in reference.named_modules()
    }
    with torch.inference_mode():
        expected = reference([weights.transforms()(image)])[0]
    actual = result["detections"]
    torch.testing.assert_close(
        torch.tensor([row["label_id"] for row in actual], dtype=torch.int64),
        expected["labels"],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        torch.tensor([row["box"] for row in actual], dtype=torch.float32).reshape(-1, 4),
        expected["boxes"],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        torch.tensor([row["score"] for row in actual], dtype=torch.float32),
        expected["scores"],
        rtol=0,
        atol=0,
    )
