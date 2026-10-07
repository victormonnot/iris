"""Local detector recipes and execution identities, with no model loading or downloads."""

import builtins
import hashlib
import json
from copy import deepcopy

import pytest
from test_models_taxonomy import checkpoint_row
from test_training_taxonomy import contract as custom_contract

from iris import models
from iris import temporal_detector as adapter
from iris.prediction_taxonomy import output_contract
from iris.training_architectures import FRCNN, SSDLITE, YOLOX
from iris.yolox_spec import SOURCE_COMMIT

contract = custom_contract


def frozen_config(model_id=SSDLITE, *, device="cpu", inference_mode="full", spec=None):
    """Pure synthetic snapshot for cache/archives tests, without optional packages.

    Execution tests can patch adapter._runtime to return this historical runtime;
    no actual inference claim is made by this fixture.
    """
    spec = spec or models.get_spec(model_id)
    architecture = spec["architecture"]
    filtering, preprocessing = adapter._native_profile(architecture)
    inference = {"mode": inference_mode}
    if inference_mode == "tiled":
        inference.update(
            algorithm="iris-tiling-v1", tiling=adapter.validate_tiling_config(640, 0.2)
        )
    return adapter.validate_detector_config(
        {
            "schema": adapter.SCHEMA,
            "model_id": spec["id"],
            "architecture": architecture,
            "origin": spec["origin"],
            "weight_sha256": "a" * 64,
            "classes": deepcopy(spec["classes"]),
            "class_contract": output_contract(spec),
            "device": device,
            "min_score": 0.001,
            "inference": inference,
            "native_filtering": filtering,
            "preprocessing": preprocessing,
            "output_policy": adapter._output_policy(architecture),
            "runtime": {
                "adapter_revision": adapter.ADAPTER_REVISION,
                "python": "3.12.0",
                "packages": {
                    "torch": "2.10.0+cpu",
                    "torchvision": "0.25.0+cpu",
                    "pillow": "12.0.0",
                    "numpy": "2.0.0",
                    "opencv-python-headless": "4.12.0.88",
                },
                "source_sha256": {
                    name: "b" * 64 for name in adapter._sources(architecture, inference_mode)
                },
            },
        }
    )


def runtime_metadata(config):
    """Complete declared facts for a synthetic detector obeying a frozen recipe."""
    architecture = config["architecture"]
    slots = (
        len(config["class_contract"]["class_mapping"]) + (architecture != YOLOX)
        if config["origin"] == "trained"
        else (80 if architecture == YOLOX else 91)
    )
    metadata = {
        "model_id": config["model_id"],
        "architecture": architecture,
        "weight_sha256": config["weight_sha256"],
        "torch_version": config["runtime"]["packages"]["torch"],
        "torchvision_version": config["runtime"]["packages"]["torchvision"],
        "input_transform": deepcopy(config["preprocessing"]),
        "native_filtering": deepcopy(config["native_filtering"]),
        "head_class_slots": slots,
        "precision": "float32",
        "coordinates": config["output_policy"]["coordinates"],
        "device": "cpu",
        "hardware": "Synthetic CPU",
        "platform": "Synthetic platform",
        "threads": 4,
        "interop_threads": 1,
    }
    mapping = adapter._native_mapping(config)
    if mapping is not None:
        metadata["native_to_coco"] = {int(key): value for key, value in mapping.items()}
    if config["origin"] == "trained":
        metadata.update(deepcopy(config["class_contract"]))
    if architecture == YOLOX:
        metadata.update(
            source_commit=SOURCE_COMMIT,
            background_class=False,
            internal_class_index_base=0,
            native_class_index_base=1,
        )
    if config["device"] == "cuda":
        metadata.update(device="cuda:0", hardware="Synthetic GPU")
        metadata["cuda"] = {
            "runtime": "12.8",
            "cudnn": 91002,
            "index": 0,
            "name": "Synthetic GPU",
            "capability": [8, 9],
            "total_memory": 8 * 1024**3,
            "uuid": "synthetic-gpu",
            "tf32_matmul": False,
            "tf32_cudnn": False,
            "cudnn_benchmark": False,
        }
    return metadata


@pytest.fixture
def local_checkpoint(tmp_path, monkeypatch):
    """Verify real synthetic bytes using the normal official size/hash checks."""
    payload = b"Synthetic fixture only: not executable model parameters"
    digest = hashlib.sha256(payload).hexdigest()
    specs = deepcopy(models._SPECS)
    for spec in specs.values():
        spec.update(download_bytes=len(payload), expected_hash_prefix=digest)
        path = tmp_path / "models" / spec["weight_filename"]
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(payload)
    monkeypatch.setattr(models, "_SPECS", specs)
    versions = frozen_config()["runtime"]["packages"]
    monkeypatch.setattr(adapter.importlib.metadata, "version", lambda name: versions[name])
    return tmp_path, digest


@pytest.mark.parametrize("model_id", [SSDLITE, FRCNN, YOLOX])
def test_prepare_verifies_local_bytes_and_freezes_complete_native_recipe_without_ml(
    local_checkpoint, monkeypatch, model_id
):
    root, digest = local_checkpoint
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert name.split(".")[0] not in {"torch", "torchvision"}
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.setattr(models, "urlopen", lambda *a, **k: pytest.fail("No network"))
    config = adapter.prepare_detector(root, model_id)
    assert config["weight_sha256"] == digest
    assert len(config["classes"]) == 80
    assert config["classes"][2] == {"id": 3, "name": "car"}
    assert config["native_filtering"]["max_detections_per_image"] == 100
    assert config["output_policy"]["complete_above_score_floor"] is False
    assert config["output_policy"]["suppressed_candidates_recoverable"] is False
    assert config["output_policy"]["native_score_comparison"] == (
        "gte" if model_id == YOLOX else "gt"
    )
    assert (
        config["runtime"]["source_sha256"]["models.py"]
        == hashlib.sha256(adapter.Path(models.__file__).read_bytes()).hexdigest()
    )
    assert adapter.prepare_detector(root, model_id) == config
    assert json.loads(json.dumps(config)) == config
    assert adapter.validate_detector_config(config) == config


def test_local_checkpoint_replacement_and_absence_prevent_preparation(local_checkpoint):
    root, _ = local_checkpoint
    path = models.checkpoint_path(root, models.get_spec(SSDLITE))
    path.write_bytes(b"x" * path.stat().st_size)
    with pytest.raises(ValueError, match="SHA-256"):
        adapter.prepare_detector(root, SSDLITE)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        adapter.prepare_detector(root, SSDLITE)


def test_custom_classes_are_frozen_without_coco_inference(local_checkpoint, monkeypatch, contract):
    root, digest = local_checkpoint
    row = checkpoint_row(contract)
    row["weight_sha256"] = digest
    row["path"] = "models/" + models.get_spec(FRCNN)["weight_filename"]
    spec = models._trained_spec(row)
    original = models.get_spec
    monkeypatch.setattr(
        models,
        "get_spec",
        lambda model_id, root=None: spec if model_id == spec["id"] else original(model_id, root),
    )
    config = adapter.prepare_detector(root, spec["id"])
    assert config["classes"][1] == {"id": 2, "name": "helmet"}
    assert config["class_contract"] == contract
    assert config["origin"] == "trained"
    metadata = runtime_metadata(config)
    assert adapter.verify_detector_metadata(config, metadata)["head_class_slots"] == 5
    metadata["class_mapping"]["helmet"] = 9
    with pytest.raises(ValueError, match="class_mapping"):
        adapter.verify_detector_metadata(config, metadata)


def test_tiling_is_frozen_as_a_separate_bounded_recipe(local_checkpoint):
    root, _ = local_checkpoint
    config = adapter.prepare_detector(
        root, YOLOX, inference_mode="tiled", tile_size=512, overlap=0.3, min_score=0.2
    )
    assert config["inference"] == {
        "mode": "tiled",
        "algorithm": "iris-tiling-v1",
        "tiling": {"tile_size": 512, "overlap": 0.3, "merge_iou": 0.5, "max_detections": 300},
    }
    assert "tiling.py" in config["runtime"]["source_sha256"]
    assert config["min_score"] == 0.2
    assert config["native_filtering"]["score_threshold"] == 0.001
    for field, value in (("merge_iou", 0.7), ("max_detections", 1000)):
        changed = deepcopy(config)
        changed["inference"]["tiling"][field] = value
        with pytest.raises(ValueError, match="Tiling settings"):
            adapter.validate_detector_config(changed)


@pytest.mark.parametrize(
    "score", [True, "0.1", 0, 0.0001, 1.01, float("nan"), float("inf"), 10**500]
)
def test_preparation_never_claims_outputs_below_native_floor(local_checkpoint, score):
    with pytest.raises(ValueError, match="min_score"):
        adapter.prepare_detector(local_checkpoint[0], SSDLITE, min_score=score)


@pytest.mark.parametrize(
    "field,value",
    [
        ("device", "cuda:0"),
        ("device", "mps"),
        ("inference_mode", "paired"),
    ],
)
def test_public_detector_choices_are_explicit(local_checkpoint, field, value):
    with pytest.raises(ValueError):
        adapter.prepare_detector(local_checkpoint[0], SSDLITE, **{field: value})


@pytest.mark.parametrize(
    "path,value",
    [
        (("schema",), "future"),
        (("weight_sha256",), "abcd"),
        (("min_score",), True),
        (("native_filtering", "max_detections_per_image"), 1000),
        (("native_filtering", "score_threshold"), 0),
        (("preprocessing", "tensor_range"), [0, 255]),
        (("classes",), [{"id": 1, "name": "person"}]),
        (("runtime", "source_sha256", "models.py"), "relative/path"),
        (("output_policy", "suppressed_candidates_recoverable"), True),
        (("runtime", "packages", "torch"), float("nan")),
    ],
)
def test_frozen_config_rejects_changed_semantics_and_unsupported_values(path, value):
    config = frozen_config()
    node = config
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    with pytest.raises(ValueError):
        adapter.validate_detector_config(config)


def test_historical_config_and_signature_validate_without_runtime_or_files(monkeypatch):
    config = frozen_config(YOLOX)
    metadata = runtime_metadata(config)
    signature = adapter.saved_execution_signature(config, metadata)
    monkeypatch.setattr(adapter, "_runtime", lambda *a: pytest.fail("No local runtime inspection"))
    monkeypatch.setattr(adapter.Path, "read_bytes", lambda *a: pytest.fail("No file access"))
    monkeypatch.setattr(
        adapter.importlib.metadata, "version", lambda *a: pytest.fail("No package inspection")
    )
    assert adapter.validate_detector_config(config) == config
    assert adapter.validate_execution_signature(config, signature) == signature
    assert adapter.saved_execution_signature(config, metadata) == signature
    changed_metadata = deepcopy(metadata)
    changed_metadata["weight_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="weight_sha256"):
        adapter.saved_execution_signature(config, changed_metadata)
    mutated = deepcopy(signature)
    mutated["config_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Saved execution signature"):
        adapter.validate_execution_signature(config, mutated)
    mutated = deepcopy(signature)
    mutated["native_to_coco"]["3"] = 8
    with pytest.raises(ValueError, match="mapping"):
        adapter.validate_execution_signature(config, mutated)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_execution_identity_excludes_instrumentation_but_preserves_hardware_and_threads(
    local_checkpoint, device
):
    root, _ = local_checkpoint
    config = adapter.prepare_detector(root, SSDLITE, device=device)
    metadata = runtime_metadata(config)
    # The ordinary SSD adapter reports a tuple before persistence.
    metadata["input_transform"]["fixed_size"] = (320, 320)
    first = adapter.verify_detector_metadata(config, metadata)
    changed = {**metadata, "load_ms": 239.9, "created_at": "synthetic next attempt"}
    assert adapter.verify_detector_metadata(config, changed) == first
    changed["threads"] = 2
    assert adapter.verify_detector_metadata(config, changed) != first
    assert adapter.validate_execution_signature(config, first) == first
    if device == "cuda":
        assert first["device"] == "cuda:0" and first["cuda"]["uuid"] == "synthetic-gpu"
        changed["cuda"]["tf32_matmul"] = True
        with pytest.raises(ValueError, match="float32 policy"):
            adapter.verify_detector_metadata(config, changed)


@pytest.mark.parametrize(
    "path,value",
    [
        (("weight_sha256",), "0" * 64),
        (("model_id",), "foreign-model"),
        (("head_class_slots",), True),
        (("threads",), True),
        (("precision",), "float16"),
        (("torch_version",), "2.10.0+different"),
        (("device",), "cuda:0"),
        (("input_transform", "max_size"), 640),
        (("native_filtering", "nms_iou_threshold"), 0.7),
    ],
)
def test_loaded_detector_must_match_its_frozen_recipe(local_checkpoint, path, value):
    config = adapter.prepare_detector(local_checkpoint[0], SSDLITE)
    metadata = runtime_metadata(config)
    node = metadata
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    with pytest.raises(ValueError):
        adapter.verify_detector_metadata(config, metadata)


def test_current_runtime_or_source_changes_refuse_execution_but_keep_archives_readable(
    local_checkpoint, monkeypatch
):
    config = adapter.prepare_detector(local_checkpoint[0], YOLOX)
    signature = adapter.verify_detector_metadata(config, runtime_metadata(config))
    historical = deepcopy(config)
    historical["runtime"]["source_sha256"]["yolox_runtime.py"] = "0" * 64
    assert adapter.validate_detector_config(historical) == historical
    with pytest.raises(ValueError, match="Execution runtime"):
        adapter.verify_detector_metadata(historical, runtime_metadata(historical))
    monkeypatch.setattr(adapter.importlib.metadata, "version", lambda name: "9.0.0")
    assert adapter.validate_execution_signature(config, signature) == signature
    with pytest.raises(RuntimeError, match="requires torch"):
        adapter.verify_detector_metadata(config, runtime_metadata(config))


def test_factory_reuses_existing_dispatch_and_preserves_cuda_choice(local_checkpoint, monkeypatch):
    root, _ = local_checkpoint
    config = adapter.prepare_detector(root, YOLOX, device="cuda")
    calls, detector = [], object()

    def factory(root, model_id, *, device):
        calls.append((root, model_id, device))
        return detector

    monkeypatch.setattr(models, "TorchvisionDetector", factory)
    assert adapter.detector_factory(root, config) is detector
    assert calls == [(root, YOLOX, "cuda")]


@pytest.mark.parametrize("model_id", [SSDLITE, FRCNN, YOLOX])
def test_worker_honors_the_actual_native_floor_comparator(model_id):
    from iris.temporal_detection_worker import _raw_prediction

    config = frozen_config(model_id)
    frame = {"width": 80, "height": 60}
    prediction = {
        "input_size": [80, 60],
        "detections": [{"label_id": 1, "label": "person", "box": [2, 3, 20, 40], "score": 0.001}],
        "timing": {
            "preprocess_ms": 1,
            "inference_ms": 2,
            "postprocess_ms": 1,
            "total_ms": 4,
        },
    }
    if model_id == YOLOX:
        _raw_prediction(prediction, frame, config)
    else:
        with pytest.raises(ValueError, match="native threshold"):
            _raw_prediction(prediction, frame, config)
    prediction["detections"] = ["malformed"]
    with pytest.raises(ValueError, match="must be objects"):
        _raw_prediction(prediction, frame, config)
