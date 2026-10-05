"""SSDLite export protocols with synthetic bytes and fake runtimes; no ML execution."""

import json
import subprocess
import sys
from copy import deepcopy

import pytest
from PIL import Image
from test_export_cuda import cuda_environment
from test_export_cuda import cuda_runtime as cuda_runtime
from test_export_runner import OutputTensor, detection, make_bundle, measurement
from test_export_runner import fake_runtime as fake_runtime
from test_model_exports import fixture_workspace, options
from test_model_exports import measurement as exported_measurement

from iris import export_runner as runner
from iris import model_exports as exports
from iris.store import Store
from iris.workspace_archive import create_archive
from iris.workspace_restore import inspect_archive, restore_archive


def ssdlite_bundle(tmp_path, *, target="cpu", source="cpu", custom=False):
    directory, manifest, reference = make_bundle(tmp_path / "bundle", custom=custom)
    manifest["format"] = "iris-model-export-v3"
    manifest["model"]["architecture"] = runner.SSDLITE_ARCHITECTURE
    manifest["profile"] = runner.ssdlite_profile(target)
    manifest["source"]["reference_device"] = source
    (directory / "manifest.json").write_bytes(runner.canonical_bytes(manifest))
    return directory, manifest, reference


def ssdlite_workspace(tmp_path, *, source="cpu", custom=False):
    workspace = fixture_workspace(tmp_path, custom=custom)
    store, model, evaluation, _ = workspace
    store.update(
        "trained_models",
        model["id"],
        {
            "architecture": runner.SSDLITE_ARCHITECTURE,
            "parent_model_id": runner.SSDLITE_ARCHITECTURE,
            "metadata": {
                **model["metadata"],
                "architecture": runner.SSDLITE_ARCHITECTURE,
                "training_device": "cuda:0",
            },
        },
    )
    store.update(
        "training_runs", model["training_id"], {"parent_model_id": runner.SSDLITE_ARCHITECTURE}
    )
    run = store.list("evaluation_models", evaluation_id=evaluation["id"])[0]
    store.update(
        "evaluation_models",
        run["id"],
        {
            "metadata": {
                **run["metadata"],
                "architecture": runner.SSDLITE_ARCHITECTURE,
                "input_transform": deepcopy(exports.SSDLITE_INPUT_TRANSFORM),
                "native_filtering": deepcopy(exports.SSDLITE_NATIVE_FILTERING),
                "device": source,
                "torch_version": "2.10.0+cpu" if source == "cpu" else "2.10.0+cu128",
                "torchvision_version": "0.25.0+cpu" if source == "cpu" else "0.25.0+cu128",
            }
        },
    )
    current = store.get("evaluations", evaluation["id"])
    store.update(
        "evaluations",
        current["id"],
        {"config": {**current["config"], "device": "cpu" if source == "cpu" else "cuda"}},
    )
    return workspace


def fake_ssdlite_builder(calls):
    class Model:
        # A bogus conversion to FrozenBatchNorm would change this fake backbone.
        backbone = object()

        def load_state_dict(self, value, strict):
            assert value == {"simulated": "weights"} and strict is True

        def eval(self):
            return self

        def to(self, device):
            assert device == calls.get("device", "cpu")
            calls["model_device"] = device
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

    sys.modules["torchvision"].models.detection.ssdlite320_mobilenet_v3_large = builder


@pytest.mark.parametrize("target", ["cpu", "cuda"])
@pytest.mark.parametrize("source", ["cpu", "cuda:1"])
def test_ssdlite_bundle_inspects_without_site_packages_or_runtime(tmp_path, target, source):
    directory, manifest, reference = ssdlite_bundle(tmp_path, target=target, source=source)
    assert runner.validate_bundle(directory) == (manifest, reference)
    result = subprocess.run(
        [sys.executable, "-I", "-S", str(directory / "run.py"), "inspect"],
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    assert json.loads(result.stdout)["runtime_loaded"] is False
    assert manifest["profile"]["architecture"] == runner.SSDLITE_ARCHITECTURE


@pytest.mark.parametrize("custom", [False, True])
def test_ssdlite_cpu_runner_uses_its_own_recipe_and_native_output(tmp_path, fake_runtime, custom):
    fake_ssdlite_builder(fake_runtime)
    directory, manifest, _ = ssdlite_bundle(tmp_path, custom=custom)
    detector = runner.Detector(directory, manifest)
    assert fake_runtime["builder"] == [
        {
            "weights": None,
            "weights_backbone": None,
            "num_classes": 3,
            "score_thresh": 0.001,
            "nms_thresh": 0.5,
            "detections_per_img": 100,
            "topk_candidates": 300,
        }
    ]
    assert fake_runtime["loads"] == [b"SIMULATED CHECKPOINT; NOT MODEL WEIGHTS"]
    assert fake_runtime["model_device"] == "cpu"
    assert detector.predict(Image.new("RGB", (20, 10)))["detections"] == [detection(custom)]


def test_ssdlite_cuda_runner_requires_nms_but_not_roi_align(tmp_path, cuda_runtime):
    fake_ssdlite_builder(cuda_runtime)
    sys.modules["torch"]._C._dispatch_has_kernel_for_dispatch_key = lambda name, key: (
        name == "torchvision::nms" and key == "CUDA"
    )
    directory, manifest, _ = ssdlite_bundle(tmp_path, target="cuda")
    assert runner.check_runtime(manifest, "cuda:1")["model_loaded"] is False
    assert not cuda_runtime["builder"] and not cuda_runtime["loads"]
    detector = runner.Detector(directory, manifest, "cuda:1")
    detector.predict(Image.new("RGB", (20, 10)))
    assert cuda_runtime["model_device"] == "cuda:1"
    assert cuda_runtime["synchronize"] == ["cuda:1"] * 5
    runner._environment(detector.environment(), manifest["profile"])
    sys.modules["torch"]._C._dispatch_has_kernel_for_dispatch_key = lambda *_: False
    with pytest.raises(RuntimeError, match="CUDA detection operators"):
        runner.check_runtime(manifest, "cuda:1")


@pytest.mark.parametrize(
    "change",
    [
        lambda m: m["model"].update(architecture=runner.ARCHITECTURE),
        lambda m: m.update(profile=runner.native_profile("cpu")),
        lambda m: m.update(format="iris-model-export-v2"),
        lambda m: m.update(format="iris-model-export-v1"),
        lambda m: m["profile"]["builder"].update(topk_candidates=100),
        lambda m: m["profile"]["builder"].update(nms_thresh=0.55),
        lambda m: m["profile"]["input"].update(image_mean=[0.485, 0.456, 0.406]),
        lambda m: m["profile"]["backbone_normalization"].update(type="FrozenBatchNorm2d"),
        lambda m: m["profile"]["parity"].update(score_atol=1e-6),
        lambda m: m["source"].pop("reference_device"),
    ],
)
def test_ssdlite_manifest_rejects_architecture_recipe_and_protocol_substitutions(tmp_path, change):
    _, manifest, _ = ssdlite_bundle(tmp_path)
    change(manifest)
    with pytest.raises(ValueError):
        runner.validate_manifest(manifest)


@pytest.mark.parametrize("target", ["cpu", "cuda"])
def test_ssdlite_exact_parity_keeps_numeric_mismatches(tmp_path, target):
    _, manifest, reference = ssdlite_bundle(tmp_path, target=target, source="cuda:0")
    payload = measurement(manifest, reference)
    if target == "cuda":
        payload["environment"] = cuda_environment()
    assert runner.validate_measurement(manifest, reference, payload)["parity_passed"]
    payload["samples"][0]["detections"][0]["score"] += 1e-12
    assert not runner.validate_measurement(manifest, reference, payload)["parity_passed"]


@pytest.mark.parametrize("target", ["cpu", "cuda"])
@pytest.mark.parametrize("source", ["cpu", "cuda:1"])
@pytest.mark.parametrize("custom", [False, True])
def test_ssdlite_export_measurement_and_archive_round_trip(
    tmp_path, monkeypatch, target, source, custom
):
    workspace = ssdlite_workspace(tmp_path, source=source, custom=custom)
    store = workspace[0]
    assert exports.candidates(store)["models"][0]["eligible"]
    values = {**options(workspace), "target_device": target}
    preview = exports.preview_export(store, **values)
    assert preview["plan"]["format"] == "iris-model-export-plan-v3"
    row = exports.create_export(
        store,
        **values,
        request_id=preview["request_id"],
        expected_fingerprint=preview["fingerprint"],
    )
    exports.run_export(store, row["id"], lambda *_: None, lambda: False)
    published = exports.export_detail(store, row["id"])
    assert published["manifest"]["format"] == "iris-model-export-v3"
    assert published["manifest"]["profile"] == runner.ssdlite_profile(target)
    assert published["manifest"]["source"]["reference_device"] == source
    payload = exported_measurement(published)
    if target == "cuda":
        payload["environment"] = cuda_environment()
    preview = exports.preview_measurement(store, row["id"], payload)
    saved = exports.save_measurement(store, row["id"], payload, preview["fingerprint"])
    assert saved["summary"]["parity_passed"] is True
    assert saved["summary"]["declaration"] == "simulation"
    resources = exports._resources()
    resources["run.py"] += b"\n# Future runner code must not invalidate frozen bundles.\n"
    monkeypatch.setattr(exports, "_resources", lambda: resources)
    archive = tmp_path / "archive.zip"
    create_archive(store.root, archive)
    inspection = inspect_archive(archive)
    restore_archive(
        archive, tmp_path / "restored", expected_archive_sha256=inspection["archive_sha256"]
    )
    restored = Store(tmp_path / "restored")
    assert restored.list("model_export_measurements") == store.list("model_export_measurements")
    assert (
        exports.download_path(restored, row["id"]).read_bytes()
        == exports.download_path(store, row["id"]).read_bytes()
    )


@pytest.mark.parametrize(
    "field", ["model_metadata", "evaluation_architecture", "transform", "filtering"]
)
def test_source_metadata_must_match_ssdlite_recipe(tmp_path, field):
    workspace = ssdlite_workspace(tmp_path)
    store, model, evaluation, _ = workspace
    run = store.list("evaluation_models", evaluation_id=evaluation["id"])[0]
    if field == "model_metadata":
        current = store.get("trained_models", model["id"])
        store.update(
            "trained_models",
            model["id"],
            {"metadata": {**current["metadata"], "architecture": runner.ARCHITECTURE}},
        )
    else:
        metadata = deepcopy(run["metadata"])
        if field == "evaluation_architecture":
            metadata["architecture"] = runner.ARCHITECTURE
        elif field == "transform":
            metadata["input_transform"] = exports.INPUT_TRANSFORM
        else:
            metadata["native_filtering"] = exports.NATIVE_FILTERING
        store.update("evaluation_models", run["id"], {"metadata": metadata})
    with pytest.raises(ValueError):
        exports.preview_export(store, **options(workspace))
    assert not exports.candidates(store)["models"][0]["eligible"]


def test_historical_faster_rcnn_profile_bytes_remain_unchanged():
    profiles = [runner.PROFILE, runner.native_profile("cpu"), runner.native_profile("cuda")]
    assert [runner.digest_bytes(runner.canonical_bytes(profile)) for profile in profiles] == [
        "901339a0a2de978a104538d2272b095e7cd9420023511814b8f5d7eb71afa87c",
        "76523fe0cf18d7b05d4576b115f077fee2cca6d9da3229ddee0715ae01125568",
        "af18930a78d88e524e5ef057a6297a5f48aea6fced7d58c002f7e68dda5c9c1a",
    ]
