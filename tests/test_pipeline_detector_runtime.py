"""Portable detector semantics and local loading safety without provisioning models."""

import contextlib
import hashlib
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_temporal_detector import frozen_config
from test_training_taxonomy import contract as custom_contract

from iris import pipeline_detector_runtime as runtime
from iris.pipeline_detector_contracts import FRCNN, SSDLITE, YOLOX, detector_contract

contract = custom_contract


def detector(architecture=SSDLITE, *, trained=False, custom=False, minimum=0.001):
    config = frozen_config(architecture)
    if trained:
        from iris.model_taxonomy import class_contract

        contract = custom if isinstance(custom, dict) else class_contract({})
        config.update(
            origin="trained",
            model_id="trained-model",
            class_contract=contract,
            classes=[
                {"id": contract["output_class_mapping"][row["id"]], "name": row["id"]}
                for row in contract["taxonomy"]["classes"]
            ],
        )
    config["min_score"] = minimum
    return detector_contract(config, "cpu", checkpoint_size=12)


@pytest.mark.parametrize("architecture", [FRCNN, SSDLITE, YOLOX])
@pytest.mark.parametrize("trained", [False, True])
def test_head_slots_map_to_external_ids_without_reindexing_after_score_filter(
    architecture, trained
):
    contract = detector(architecture, trained=trained, minimum=0.2)
    rows = contract["output_mapping"]["entries"]
    car = next(row for row in rows if row["output_id"] == 3)
    person = next(row for row in rows if row["output_id"] == 1)
    result = runtime._canonical_detections(
        [[0, 0, 10, 20], [2, 3, 11, 21]],
        [person["native_label_id"], car["native_label_id"]],
        [0.19, 0.2],
        [40, 30],
        contract,
    )
    assert result == [
        {"detection_index": 1, "label_id": 3, "label": "car", "score": 0.2, "box": [2, 3, 11, 21]}
    ]
    assert contract["output_mapping"]["head_slots"] == (
        (2 if architecture == YOLOX else 3) if trained else (80 if architecture == YOLOX else 91)
    )


@pytest.mark.parametrize("architecture", [SSDLITE, YOLOX])
def test_custom_taxonomy_uses_explicit_output_names_and_slots(architecture, contract):
    spec = detector(architecture, trained=True, custom=contract)
    entries = spec["output_mapping"]["entries"]
    result = runtime._canonical_detections(
        [[0, 0, 4, 8]], [entries[-1]["native_label_id"]], [0.9], [8, 10], spec
    )
    assert result[0]["label_id"] == entries[-1]["output_id"]
    assert result[0]["label"] == entries[-1]["label"]
    assert set(result[0]) == {"detection_index", "label_id", "label", "score", "box"}


@pytest.mark.parametrize("architecture", [SSDLITE, YOLOX])
def test_native_score_boundary_preserves_the_architecture_comparison(architecture):
    args = ([[0, 0, 4, 8]], [1], [0.001], [8, 10], detector(architecture))
    if architecture == YOLOX:
        assert runtime._canonical_detections(*args)[0]["score"] == 0.001
    else:
        with pytest.raises(ValueError, match="score contract"):
            runtime._canonical_detections(*args)


@pytest.mark.parametrize(
    ("boxes", "labels", "scores"),
    [
        ([[0, 0, 2, 3]], [], [0.5]),
        ([[0, 0, 2, 3]], [True], [0.5]),
        ([[0, 0, 2, 3]], [12], [0.5]),  # COCO's unused slot must not become a class.
        ([[0, 0, 2, 3]], [1], [float("nan")]),
        ([[0, 0, 2, 3]], [1], [True]),
        ([[0, 0, 2, 3]], [1], [0.0001]),
        ([[0, 0, 2, 3]], [1], [1.1]),
        ([[0, 0, 0, 3]], [1], [0.5]),
        ([[-1, 0, 2, 3]], [1], [0.5]),
        ([[0, 0, 2, 31]], [1], [0.5]),
        ([[0, 0, float("inf"), 3]], [1], [0.5]),
        ([[0, 0, 2, 3]] * 101, [1] * 101, [0.5] * 101),
    ],
)
def test_invalid_native_outputs_fail_before_storage_filter(boxes, labels, scores):
    with pytest.raises(ValueError):
        runtime._canonical_detections(boxes, labels, scores, [40, 30], detector(minimum=0.9))


def write_checkpoint(tmp_path, raw=b"model bytes"):
    base = tmp_path / "package"
    (base / "detector").mkdir(parents=True)
    path = base / "detector/model.pth"
    path.write_bytes(raw)
    return base, {
        "path": "detector/model.pth",
        "size": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def test_checkpoint_is_verified_and_rewound_without_deserializing(tmp_path):
    base, identity = write_checkpoint(tmp_path)
    stream, details = runtime._checkpoint(base, identity)
    with stream:
        assert stream.tell() == 0
        assert stream.read() == b"model bytes"
        assert details.st_size == identity["size"]
    changed = {**identity, "sha256": "a" * 64}
    with pytest.raises(ValueError, match="SHA-256"):
        runtime._checkpoint(base, changed)
    with pytest.raises(ValueError, match="size"):
        runtime._checkpoint(base, {**identity, "size": 1})


@pytest.mark.parametrize("kind", ["file_link", "directory_link", "fifo", "directory", "escape"])
def test_checkpoint_rejects_unsafe_paths_before_opening_or_loading(tmp_path, kind):
    base, identity = write_checkpoint(tmp_path)
    path = base / identity["path"]
    raw = path.read_bytes()
    path.unlink()
    if kind == "file_link":
        other = tmp_path / "other.pth"
        other.write_bytes(raw)
        path.symlink_to(other)
    elif kind == "directory_link":
        (base / "detector").rmdir()
        other = tmp_path / "outside"
        other.mkdir()
        (other / "model.pth").write_bytes(raw)
        (base / "detector").symlink_to(other, target_is_directory=True)
    elif kind == "fifo":
        os.mkfifo(path)
    elif kind == "directory":
        path.mkdir()
    else:
        identity["path"] = "../outside.pth"
    with pytest.raises(ValueError):
        runtime._checkpoint(base, identity)


@pytest.mark.parametrize("requested", ["cuda", "cuda:1", "gpu", "cpu:0", "cuda:-1", True])
def test_device_cannot_change_frozen_family_or_accept_implicit_fallback(requested):
    with pytest.raises(ValueError):
        runtime._device({"detector": detector()}, requested)
    target = {"detector": {**detector(), "target_device": "cuda"}}
    assert runtime._device(target, "cuda:1") == "cuda:1"
    assert runtime._device(target) == "cuda"


@pytest.mark.parametrize("architecture", [FRCNN, SSDLITE, YOLOX])
@pytest.mark.parametrize("trained", [False, True])
def test_constructor_preserves_builder_flags_envelope_and_strict_weights_only_loading(
    tmp_path, monkeypatch, architecture, trained
):
    base, identity = write_checkpoint(tmp_path)
    spec = detector(architecture, trained=trained)
    spec["checkpoint"].update(identity)
    manifest = {"format": "iris-pipeline-bundle-v2", "detector": spec}
    calls = []

    class Model:
        backbone = None

        def float(self):
            calls.append("float32")
            return self

        def load_state_dict(self, value, strict):
            calls.append(("state", value, strict))

        def eval(self):
            return self

        def to(self, device):
            calls.append(("device", device))
            return self

    def builder(**options):
        calls.append(("builder", options))
        return Model()

    def load(stream, **options):
        assert stream.tell() == 0 and stream.read() == b"model bytes"
        calls.append(("load", options))
        return (
            {"model": {"parameter": "weights"}, "epoch": 3}
            if architecture == YOLOX and not trained
            else {"parameter": "weights"}
        )

    torch = SimpleNamespace(
        device=lambda _: contextlib.nullcontext(),
        set_num_threads=lambda value: None,
        get_num_threads=lambda: 4,
        get_num_interop_threads=lambda: 1,
        load=load,
    )
    torchvision = SimpleNamespace(
        models=SimpleNamespace(detection=SimpleNamespace(**{architecture: builder}))
    )
    monkeypatch.setattr(runtime, "validate_manifest", deepcopy)
    monkeypatch.setattr(runtime, "_runtime_modules", lambda *_: (torch, torchvision, "cpu", {}))
    monkeypatch.setattr(
        runtime, "_restore_frozen_batchnorm", lambda *args: calls.append("frozen_bn")
    )
    monkeypatch.setattr(
        runtime, "_build_yolox", lambda count, _: (calls.append(("yolox_slots", count)), Model())[1]
    )
    instance = runtime.Detector(base, manifest)
    assert ("load", {"map_location": "cpu", "weights_only": True}) in calls
    assert ("state", {"parameter": "weights"}, True) in calls
    if architecture == YOLOX:
        assert ("yolox_slots", spec["output_mapping"]["head_slots"]) in calls
    else:
        options = next(row[1] for row in calls if isinstance(row, tuple) and row[0] == "builder")
        assert options["weights"] is None and options["weights_backbone"] is None
        assert options["num_classes"] == spec["output_mapping"]["head_slots"]
        assert options["box_score_thresh" if architecture == FRCNN else "score_thresh"] == 0.001
        assert ("frozen_bn" in calls) == (architecture == FRCNN)
    assert instance.metadata["device"] == "cpu"
    changed = instance.metadata
    changed["device"] = "cuda"
    assert instance.metadata["device"] == "cpu"


def test_import_has_no_optional_ml_or_original_application_dependency(tmp_path):
    target = tmp_path / "iris_bundle"
    target.mkdir()
    (target / "__init__.py").write_text("")
    files = (
        "pipeline_detector_runtime.py",
        "pipeline_bundle_contracts.py",
        "pipeline_bundle_runtime_contracts.py",
        "pipeline_detector_contracts.py",
        "tracking_contracts.py",
        "tracking_selection_contracts.py",
    )
    source = Path(runtime.__file__).parent
    for filename in files:
        (target / filename).write_bytes((source / filename).read_bytes())
    script = """
import builtins, importlib, json, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'iris', 'argos', 'torch', 'torchvision', 'numpy', 'cv2', 'PIL'}:
        raise AssertionError('Unexpected runtime dependency: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
importlib.import_module('iris_bundle.pipeline_detector_runtime')
print(json.dumps({'lazy': True}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(completed.stdout) == {"lazy": True}


def test_declared_package_versions_allow_only_torch_build_suffix_changes(monkeypatch):
    versions = {name: runtime.PACKAGE_VERSIONS[name] for name in runtime.DETECTOR_PACKAGES}
    versions.update(torch="2.10.0+cu128", torchvision="0.25.0+cu128")
    monkeypatch.setattr(runtime.importlib.metadata, "version", versions.__getitem__)
    assert runtime._packages() == versions
    versions["numpy"] = "2.5.3+different"
    with pytest.raises(RuntimeError, match="Expected numpy"):
        runtime._packages()
    versions["numpy"] = "2.5.3"
    versions["torch"] = "2.11.0+cu128"
    with pytest.raises(RuntimeError, match="Expected torch"):
        runtime._packages()


@pytest.mark.parametrize("size", [(47, 23), (21, 57), (32, 32)])
def test_yolox_preprocessing_matches_original_byte_color_letterbox_contract(size):
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    from PIL import Image

    from iris.yolox_runtime import preprocess

    pixels = np.random.default_rng(49).integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    image = Image.fromarray(pixels)
    expected, expected_ratio = preprocess(image)
    actual, ratio = runtime._preprocess_yolox(image, torch, "cpu")
    assert ratio == expected_ratio
    assert torch.equal(actual, expected)
    assert actual.dtype == torch.float32 and actual.shape == (1, 3, 416, 416)


def test_yolox_decoding_filter_and_class_aware_nms_match_original_native_postprocess():
    torch = pytest.importorskip("torch")
    torchvision = pytest.importorskip("torchvision")
    from iris.yolox_runtime import postprocess

    # Decoded xywh predictions: same-class overlaps, a different class at the same
    # location, clipping, the native floor, zero-area and a score below the floor.
    output = torch.tensor(
        [
            [
                [8, 10, 8, 12, 0.9, 0.8, 0.1],
                [8, 10, 8, 12, 0.8, 0.8, 0.1],
                [8, 10, 8, 12, 0.8, 0.1, 0.9],
                [1, 1, 8, 12, 0.8, 0.7, 0.1],
                [20, 12, 4, 6, 0.001, 1, 0],
                [1, 1, 0, 3, 0.9, 1, 0],
                [20, 12, 4, 6, 0.0009, 1, 0],
            ]
        ],
        dtype=torch.float32,
    )
    expected = postprocess(output.clone(), (40, 30), 0.5)
    actual = runtime._postprocess_yolox(output.clone(), (40, 30), 0.5, torch, torchvision)
    for key in ("boxes", "scores", "labels"):
        assert torch.equal(actual[key], expected[key])
    assert actual["labels"].tolist() == [1, 2, 1, 1]


def test_prediction_uses_oriented_pixels_and_original_indices_inside_external_autocast():
    torch = pytest.importorskip("torch")
    torchvision = pytest.importorskip("torchvision")
    from PIL import Image

    instance = object.__new__(runtime.Detector)
    instance.detector = detector(minimum=0.5)
    instance.config = instance.detector["config"]
    instance.torch, instance.torchvision, instance.device = torch, torchvision, "cpu"
    checked = []

    def model(inputs):
        assert inputs[0].shape == (3, 8, 5)
        assert inputs[0].dtype == torch.float32
        assert not torch.is_autocast_enabled("cpu")
        checked.append(True)
        return [
            {
                "boxes": torch.tensor([[0, 0, 3, 4], [1, 2, 4, 7]], dtype=torch.float32),
                "labels": torch.tensor([1, 3]),
                "scores": torch.tensor([0.2, 0.9]),
            }
        ]

    instance.model = model
    image = Image.new("RGB", (8, 5), (51, 102, 153))
    image.getexif()[274] = 6
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        result = instance.predict(image)
    assert checked == [True]
    assert result["input_size"] == [5, 8]
    assert result["native_detection_count"] == 2
    assert result["detections"] == [
        {
            "detection_index": 1,
            "label_id": 3,
            "label": "car",
            "score": float(torch.tensor(0.9)),
            "box": [1.0, 2.0, 4.0, 7.0],
        }
    ]
    assert result["timing"]["total_ms"] >= result["timing"]["inference_ms"] >= 0


def test_frozen_batchnorm_restores_original_fasterrcnn_epsilon():
    torch = pytest.importorskip("torch")
    torchvision = pytest.importorskip("torchvision")
    from iris.models import _restore_frozen_batchnorm

    original = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 1),
        torch.nn.BatchNorm2d(4),
        torch.nn.Sequential(torch.nn.BatchNorm2d(4)),
    )
    portable = deepcopy(original)
    _restore_frozen_batchnorm(original, torch, torchvision)
    runtime._restore_frozen_batchnorm(portable, torch, torchvision)
    assert type(portable[1]) is torchvision.ops.misc.FrozenBatchNorm2d
    assert portable[1].eps == 1e-5
    assert type(portable[2][0]) is torchvision.ops.misc.FrozenBatchNorm2d
    assert set(portable.state_dict()) == set(original.state_dict())
    pixels = torch.ones(1, 3, 8, 9)
    assert torch.equal(portable(pixels), original(pixels))
