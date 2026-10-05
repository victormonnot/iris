"""Portable bundle contracts and simulated execution; no detector is constructed."""

import json
import subprocess
import sys
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from iris import export_runner as runner
from iris.model_taxonomy import class_contract
from iris.taxonomies import TAXONOMY


def contract(custom=False):
    taxonomy = deepcopy(TAXONOMY)
    if custom:
        taxonomy.update(
            id="taxonomy-" + "a" * 32,
            version=2,
            parent_id=TAXONOMY["id"],
            created_at="2026-10-05T00:00:00+00:00",
            classes=[
                {"id": "helmet", "name": "Helmet", "definition": "A visible helmet."},
                {
                    "id": "vehicle",
                    "name": "Vehicle",
                    "definition": "A passenger car.",
                    "coco_id": 3,
                },
            ],
        )
    internal = {item["id"]: index for index, item in enumerate(taxonomy["classes"], 1)}
    return class_contract(
        {
            "taxonomy": taxonomy,
            "class_mapping": internal,
            "output_class_mapping": internal if custom else {"person": 1, "car": 3},
        }
    )


def detection(custom=False):
    item = {
        "box": [0.0, 0.0, 20.0, 10.0],
        "score": 0.002,
        "native_label_id": 2,
        "label_id": 2 if custom else 3,
        "label": "vehicle" if custom else "car",
    }
    if custom:
        item["taxonomy_id"] = "taxonomy-" + "a" * 32
    return item


def make_bundle(directory, custom=False):
    directory.mkdir()
    image_dir = directory / "parity" / "images"
    image_dir.mkdir(parents=True)
    frames = []
    for identifier in ("frame1", "frame2"):
        path = image_dir / f"{identifier}.png"
        Image.new("RGB", (20, 10), "red").save(path)
        frames.append(
            {
                "frame_id": identifier,
                "path": f"parity/images/{identifier}.png",
                "sha256": runner.digest_bytes(path.read_bytes()),
                "input_size": [20, 10],
                "detections": [detection(custom)] if identifier == "frame1" else [],
            }
        )
    checkpoint = b"SIMULATED CHECKPOINT; NOT MODEL WEIGHTS"
    (directory / "model.pth").write_bytes(checkpoint)
    (directory / "run.py").write_bytes(Path(runner.__file__).read_bytes())
    (directory / "requirements.txt").write_text(
        "torch==2.10.0\ntorchvision==0.25.0\nPillow==12.3.0\n"
    )
    (directory / "README.md").write_text("Simulated bundle; no model execution.")
    reference = {
        "format": "iris-export-reference-v1",
        "model_id": "trained_fixture",
        "weight_sha256": runner.digest_bytes(checkpoint),
        "frames": frames,
    }
    (directory / "parity" / "reference.json").write_bytes(runner.canonical_bytes(reference))
    files = {}
    for path in directory.rglob("*"):
        if path.is_file():
            content = path.read_bytes()
            files[path.relative_to(directory).as_posix()] = {
                "sha256": runner.digest_bytes(content),
                "size": len(content),
            }
    manifest = {
        "format": "iris-model-export-v1",
        "id": "export_fixture",
        "name": "Simulated export",
        "created_at": "2026-10-05T00:00:00+00:00",
        "model": {
            "id": "trained_fixture",
            "name": "Simulated trained model",
            "architecture": runner.ARCHITECTURE,
            "sha256": runner.digest_bytes(checkpoint),
            "size": len(checkpoint),
            "class_contract": contract(custom),
        },
        "source": {
            "evaluation_id": "evaluation_fixture",
            "evaluation_model_id": "evaluation_model_fixture",
            "dataset_id": "dataset_fixture",
            "dataset_manifest_sha256": "a" * 64,
        },
        "profile": deepcopy(runner.PROFILE),
        "files": files,
        "validation": {"real_execution": "not_run", "reference_kind": "saved_iris_evaluation"},
    }
    (directory / "manifest.json").write_bytes(runner.canonical_bytes(manifest))
    return directory, manifest, reference


@pytest.fixture
def bundle(tmp_path):
    return make_bundle(tmp_path / "bundle")


def environment():
    return {
        "python": "3.12.10",
        "torch": "2.10.0+cpu",
        "torchvision": "0.25.0+cpu",
        "pillow": "12.3.0",
        "platform": "simulated Linux",
        "machine": "x86_64",
        "processor": "simulated processor",
        "cpu_count": 8,
        "threads": 4,
        "interop_threads": 4,
        "device": "cpu",
        "precision": "float32",
        "batch_size": 1,
    }


def measurement(manifest, reference, repeats=2):
    return {
        "format": "iris-export-measurement-v1",
        "manifest_sha256": runner.digest_bytes(runner.canonical_bytes(manifest)),
        "environment": environment(),
        "repeats": repeats,
        "warmup": {"frame_id": reference["frames"][0]["frame_id"], "duration_ms": 12.0},
        "load_ms": 50.0,
        "samples": [
            {
                "frame_id": frame["frame_id"],
                "repeat": repeat,
                "input_size": frame["input_size"],
                "detections": deepcopy(frame["detections"]),
                "timing": {
                    "preprocess_ms": 1.0,
                    "inference_ms": 8.0,
                    "postprocess_ms": 1.0,
                    "total_ms": 10.0,
                },
                "decode_ms": 2.0,
            }
            for repeat in range(1, repeats + 1)
            for frame in reference["frames"]
        ],
        "declaration": "simulation",
    }


def rewrite_reference(manifest, reference):
    content = runner.canonical_bytes(reference)
    manifest["files"]["parity/reference.json"] = {
        "sha256": runner.digest_bytes(content),
        "size": len(content),
    }


def test_bundle_inspection_is_standalone_without_site_packages_or_workspace(bundle, tmp_path):
    directory, manifest, reference = bundle
    assert runner.validate_bundle(directory) == (manifest, reference)
    result = subprocess.run(
        [sys.executable, "-I", "-S", str(directory / "run.py"), "inspect"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    result = json.loads(result.stdout)
    assert result["runtime_loaded"] is False
    assert result["real_execution"] == "not_run"
    assert result["reference_frames"] == 2
    assert result["files_verified"] == 7


@pytest.mark.parametrize("custom", [False, True])
def test_frozen_native_namespaces_and_zero_tolerance_parity(tmp_path, custom):
    _, manifest, reference = make_bundle(tmp_path / "bundle", custom)
    payload = measurement(manifest, reference)
    summary = runner.validate_measurement(manifest, reference, payload)
    assert summary["parity_passed"] is True
    assert summary["execution_verified"] is False
    assert summary["declaration"] == "simulation"
    assert summary["sample_count"] == 4
    assert summary["timing_ms"]["total_ms"] == {"min": 10.0, "median": 10.0, "max": 10.0}
    payload["samples"][0]["detections"][0]["box"][2] -= 1e-12
    summary = runner.validate_measurement(manifest, reference, payload)
    assert summary["parity_passed"] is False
    assert summary["mismatched_samples"] == [{"frame_id": "frame1", "repeat": 1}]


@pytest.mark.parametrize("field", ["score", "missing", "order", "dimensions"])
def test_parity_reports_valid_numerical_count_order_and_dimension_mismatches(bundle, field):
    _, manifest, reference = bundle
    reference["frames"][0]["detections"].append({**detection(), "score": 0.9})
    rewrite_reference(manifest, reference)
    payload = measurement(manifest, reference)
    sample = payload["samples"][0]
    if field == "score":
        sample["detections"][0]["score"] += 1e-12
    elif field == "missing":
        sample["detections"].pop()
    elif field == "order":
        sample["detections"].reverse()
    else:
        sample["input_size"] = [21, 10]
    assert not runner.validate_measurement(manifest, reference, payload)["parity_passed"]


@pytest.mark.parametrize(
    "change",
    [
        lambda p: p.update(manifest_sha256="b" * 64),
        lambda p: p.update(repeats=True),
        lambda p: p.update(repeats=11),
        lambda p: p["samples"].pop(),
        lambda p: p["samples"].reverse(),
        lambda p: p["samples"][0].update(repeat=True),
        lambda p: p["samples"][0].update(frame_id="other"),
        lambda p: p["samples"][0].update(decode_ms=float("nan")),
        lambda p: p["samples"][0]["timing"].update(total_ms=9),
        lambda p: p["samples"][0]["timing"].update(preprocess_ms=True),
        lambda p: p["samples"][0]["timing"].update(postprocess_ms=-1),
        lambda p: p["warmup"].update(frame_id="frame2"),
        lambda p: p.update(load_ms=float("inf")),
        lambda p: p.update(load_ms=10**1000),
        lambda p: p.update(declaration="verified"),
        lambda p: p.update(declaration=[]),
        lambda p: p.update(summary={"parity_passed": True}),
        lambda p: p["environment"].update(torch="2.9.0"),
        lambda p: p["environment"].update(pillow="12.2.0"),
        lambda p: p["environment"].update(python="3.14.0"),
        lambda p: p["environment"].update(device="cuda"),
        lambda p: p["environment"].update(threads=2),
        lambda p: p["environment"].update(batch_size=True),
    ],
)
def test_measurement_rejects_incomplete_forged_or_incompatible_evidence(bundle, change):
    _, manifest, reference = bundle
    payload = measurement(manifest, reference)
    change(payload)
    with pytest.raises(ValueError):
        runner.validate_measurement(manifest, reference, payload)


@pytest.mark.parametrize(
    "change",
    [
        lambda p: p.update(native_label_id=0),
        lambda p: p.update(native_label_id=True),
        lambda p: p.update(native_label_id=2.0),
        lambda p: p.update(label="bicycle"),
        lambda p: p.update(label_id=2),
        lambda p: p.update(score=float("nan")),
        lambda p: p.update(box=[0, 0, 21, 10]),
        lambda p: p.update(box=[0, 0, 0, 10]),
        lambda p: p.update(box=[False, 0, 20, 10]),
        lambda p: p.update(taxonomy_id="iris-objects-v1"),
    ],
)
def test_reference_rejects_wrong_native_geometry_or_legacy_output_schema(bundle, change):
    _, manifest, reference = bundle
    change(reference["frames"][0]["detections"][0])
    with pytest.raises(ValueError):
        runner.validate_reference(manifest, reference)


@pytest.mark.parametrize(
    "change",
    [
        lambda m: m["profile"]["parity"].update(box_atol=0.1),
        lambda m: m["profile"]["builder"].update(weights="DEFAULT"),
        lambda m: m["profile"]["builder"].update(rpn_score_thresh=0),
        lambda m: m["model"].update(architecture="ssdlite320_mobilenet_v3_large"),
        lambda m: m["model"]["class_contract"]["class_mapping"].update(car=True),
        lambda m: m["model"]["class_contract"]["output_class_mapping"].update(car=2),
        lambda m: m["files"].update({"../outside": {"sha256": "a" * 64, "size": 1}}),
        lambda m: m["files"].update({"/absolute": {"sha256": "a" * 64, "size": 1}}),
        lambda m: m["files"]["model.pth"].update(size=1),
        lambda m: m["validation"].update(real_execution="passed"),
    ],
)
def test_manifest_rejects_changed_recipe_class_slots_inventory_and_claims(bundle, change):
    _, manifest, _ = bundle
    change(manifest)
    with pytest.raises(ValueError):
        runner.validate_manifest(manifest)


def test_custom_classes_keep_literal_definitions_and_coco_mapping_is_not_output_namespace(tmp_path):
    _, manifest, reference = make_bundle(tmp_path / "custom", custom=True)
    manifest["model"]["class_contract"]["taxonomy"]["classes"][0]["definition"] = (
        "A helmet.\nInclude small helmets."
    )
    assert runner.validate_reference(manifest, reference)
    reference["frames"][0]["detections"][0]["label_id"] = 3
    with pytest.raises(ValueError, match="mapping"):
        runner.validate_reference(manifest, reference)


@pytest.mark.parametrize(
    "changed", ["model.pth", "run.py", "parity/images/frame1.png", "parity/reference.json"]
)
def test_bundle_refuses_altered_files_before_any_model_execution(bundle, changed):
    directory, _, _ = bundle
    path = directory / changed
    data = bytearray(path.read_bytes())
    data[0] ^= 1
    path.write_bytes(data)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        runner.validate_bundle(directory)


def test_bundle_refuses_symlinks_duplicate_json_and_noncanonical_reference(bundle, tmp_path):
    directory, manifest, _ = bundle
    model = directory / "model.pth"
    target = tmp_path / "outside.pth"
    model.rename(target)
    model.symlink_to(target)
    with pytest.raises(ValueError, match="symbolic"):
        runner.validate_bundle(directory)
    model.unlink()
    target.rename(model)
    (directory / "manifest.json").write_text('{"id":"one","id":"two"}')
    with pytest.raises(ValueError, match="Duplicate"):
        runner.validate_bundle(directory)
    (directory / "manifest.json").write_bytes(runner.canonical_bytes(manifest))
    reference_path = directory / "parity" / "reference.json"
    reference_path.write_bytes(reference_path.read_bytes() + b"\n")
    manifest["files"]["parity/reference.json"] = {
        "sha256": runner.digest_bytes(reference_path.read_bytes()),
        "size": reference_path.stat().st_size,
    }
    (directory / "manifest.json").write_bytes(runner.canonical_bytes(manifest))
    with pytest.raises(ValueError, match="canonical"):
        runner.validate_bundle(directory)


class OutputTensor:
    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.value


@pytest.fixture
def fake_runtime(monkeypatch):
    calls = {"builder": [], "loads": [], "forward": [], "tensors": [], "threads": None}

    class BatchNorm:
        num_features = 3

    class FrozenBatchNorm:
        def __init__(self, num_features, eps):
            self.num_features, self.eps = num_features, eps

    class Backbone:
        def __init__(self):
            self.norm = BatchNorm()

        def named_children(self):
            return [("norm", self.norm)]

    class Model:
        def __init__(self):
            self.backbone = Backbone()

        def load_state_dict(self, value, strict):
            assert value == {"simulated": "weights"} and strict is True
            assert isinstance(self.backbone.norm, FrozenBatchNorm)
            assert self.backbone.norm.eps == 1e-5

        def eval(self):
            return self

        def to(self, device):
            assert device == "cpu"
            return self

        def __call__(self, tensors):
            assert len(tensors) == 1
            calls["forward"].append(tensors)
            return [
                {
                    "boxes": OutputTensor([[0.0, 0.0, 20.0, 10.0]]),
                    "labels": OutputTensor([2]),
                    "scores": OutputTensor([0.002]),
                }
            ]

    def builder(**kwargs):
        calls["builder"].append(kwargs)
        return Model()

    def load(source, *, map_location, weights_only):
        assert map_location == "cpu" and weights_only is True
        calls["loads"].append(source.read())
        return {"simulated": "weights"}

    class InputTensor:
        def to(self, **kwargs):
            assert kwargs == {"device": "cpu", "dtype": "float32"}
            return self

        def __truediv__(self, divisor):
            assert divisor == 255.0
            return self

    def pil_to_tensor(image):
        calls["tensors"].append((image.mode, image.size))
        return InputTensor()

    torch = SimpleNamespace(
        __version__="2.10.0+cpu",
        nn=SimpleNamespace(BatchNorm2d=BatchNorm),
        load=load,
        float32="float32",
        inference_mode=nullcontext,
        set_num_threads=lambda value: calls.update(threads=value),
        get_num_threads=lambda: calls["threads"],
        get_num_interop_threads=lambda: 4,
    )
    torchvision = SimpleNamespace(
        __version__="0.25.0+cpu",
        models=SimpleNamespace(
            detection=SimpleNamespace(fasterrcnn_mobilenet_v3_large_320_fpn=builder)
        ),
        ops=SimpleNamespace(misc=SimpleNamespace(FrozenBatchNorm2d=FrozenBatchNorm)),
        transforms=SimpleNamespace(functional=SimpleNamespace(pil_to_tensor=pil_to_tensor)),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torchvision", torchvision)
    monkeypatch.setattr(
        runner.importlib.metadata, "version", lambda name: runner.PROFILE["runtime"][name]
    )
    monkeypatch.setattr(runner.os, "cpu_count", lambda: 8)
    return calls


@pytest.mark.parametrize("custom", [False, True])
def test_simulated_loader_and_predict_preserve_recipe_head_mapping_orientation(
    tmp_path, fake_runtime, custom
):
    directory, manifest, _ = make_bundle(tmp_path / "bundle", custom)
    detector = runner.Detector(directory, manifest)
    options = fake_runtime["builder"][0]
    assert options == {**runner.PROFILE["builder"], "num_classes": 3}
    assert options["weights"] is None and options["weights_backbone"] is None
    assert options["rpn_score_thresh"] == 0.05
    assert fake_runtime["loads"] == [b"SIMULATED CHECKPOINT; NOT MODEL WEIGHTS"]
    image = Image.new("L", (10, 20))
    image.getexif()[274] = 6
    counter = iter([1.0, 1.001, 1.009, 1.010])
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(runner.time, "perf_counter", lambda: next(counter))
        prediction = detector.predict(image)
    assert fake_runtime["tensors"] == [("RGB", (20, 10))]
    assert prediction["input_size"] == [20, 10]
    assert prediction["detections"] == [detection(custom)]
    assert prediction["timing"] == pytest.approx(
        {"preprocess_ms": 1, "inference_ms": 8, "postprocess_ms": 1, "total_ms": 10}
    )


def test_simulated_execution_measures_one_load_one_warmup_and_each_combination(
    bundle, fake_runtime
):
    directory, manifest, reference = bundle
    payload = runner.measure(directory, manifest, reference, 2)
    assert len(fake_runtime["builder"]) == 1
    assert len(fake_runtime["loads"]) == 1
    assert len(fake_runtime["forward"]) == 5
    assert [(row["frame_id"], row["repeat"]) for row in payload["samples"]] == [
        ("frame1", 1),
        ("frame2", 1),
        ("frame1", 2),
        ("frame2", 2),
    ]
    summary = runner.validate_measurement(manifest, reference, payload)
    assert not summary["parity_passed"]
    assert summary["mismatched_samples"] == [
        {"frame_id": "frame2", "repeat": 1},
        {"frame_id": "frame2", "repeat": 2},
    ]
    # The production runner declares external execution; these test artifacts
    # remain private simulations and are never imported as actual measurements.
    assert payload["declaration"] == "external_execution"
    assert summary["execution_verified"] is False


def test_no_runtime_fallback_and_changed_checkpoint_block_loading(
    bundle, fake_runtime, monkeypatch
):
    directory, manifest, _ = bundle
    monkeypatch.setattr(runner.importlib.metadata, "version", lambda _: "0.0")
    with pytest.raises(RuntimeError, match="Expected torch"):
        runner.Detector(directory, manifest)
    assert not fake_runtime["builder"]
    monkeypatch.setattr(
        runner.importlib.metadata, "version", lambda name: runner.PROFILE["runtime"][name]
    )
    (directory / "model.pth").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="changed before loading"):
        runner.Detector(directory, manifest)
    assert not fake_runtime["loads"]


def test_result_writes_cannot_replace_bundle_or_existing_files(bundle, tmp_path):
    directory, _, _ = bundle
    with pytest.raises(ValueError, match="outside"):
        runner._write_new_json(directory / "new.json", {}, directory)
    output = tmp_path / "result.json"
    runner._write_new_json(output, {"ok": True}, directory)
    with pytest.raises(FileExistsError):
        runner._write_new_json(output, {"ok": False}, directory)
    assert output.read_bytes() == b'{"ok":true}'


def test_cli_compatible_failure_is_readable_without_traceback(bundle, capsys):
    directory, _, _ = bundle
    (directory / "model.pth").unlink()
    assert runner.main(["--bundle", str(directory), "inspect"]) == 2
    assert "Missing or unsafe bundle file" in capsys.readouterr().err


def test_builtin_snapshot_tracks_current_contract_and_rejects_changed_semantics(bundle):
    assert runner._BUILTIN_TAXONOMY == TAXONOMY
    _, manifest, _ = bundle
    manifest["model"]["class_contract"]["taxonomy"]["classes"][0]["definition"] = "Changed meaning"
    with pytest.raises(ValueError, match="supported definition"):
        runner.validate_manifest(manifest)


def test_simulated_cli_saves_mismatch_with_exit_three_and_predicts_to_new_file(
    bundle, fake_runtime, tmp_path, capsys
):
    directory, manifest, reference = bundle
    report = tmp_path / "measurement.json"
    assert (
        runner.main(
            ["--bundle", str(directory), "measure", "--repeats", "1", "--output", str(report)]
        )
        == 3
    )
    assert not json.loads(capsys.readouterr().out)["parity_passed"]
    payload = json.loads(report.read_bytes())
    assert not runner.validate_measurement(manifest, reference, payload)["parity_passed"]
    output = tmp_path / "prediction.json"
    assert (
        runner.main(
            [
                "--bundle",
                str(directory),
                "predict",
                str(directory / "parity/images/frame1.png"),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(output.read_bytes())["detections"] == [detection()]
    assert json.loads(capsys.readouterr().out)["detections"] == 1


def test_existing_output_stops_before_loading_runtime(bundle, fake_runtime, tmp_path, capsys):
    directory, _, _ = bundle
    output = tmp_path / "existing.json"
    output.write_text("unchanged")
    assert runner.main(["--bundle", str(directory), "measure", "--output", str(output)]) == 2
    assert "already exists" in capsys.readouterr().err
    assert not fake_runtime["loads"] and not fake_runtime["builder"]
    assert output.read_text() == "unchanged"
