"""CUDA export protocol simulations; never load a detector or touch a real GPU."""

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from test_export_runner import fake_runtime as fake_runtime
from test_export_runner import make_bundle, measurement
from test_model_exports import fixture_workspace, options
from test_model_exports import measurement as exported_measurement

from iris import export_runner as runner
from iris import model_exports as exports
from iris.app import create_app
from iris.store import Store
from iris.workspace_archive import create_archive
from iris.workspace_restore import inspect_archive, restore_archive


def modern_bundle(tmp_path, *, target="cuda", source="cpu"):
    directory, manifest, reference = make_bundle(tmp_path / "bundle")
    manifest["format"] = "iris-model-export-v2"
    manifest["profile"] = runner.native_profile(target)
    manifest["source"]["reference_device"] = source
    (directory / "manifest.json").write_bytes(runner.canonical_bytes(manifest))
    return directory, manifest, reference


@pytest.fixture
def cuda_runtime(fake_runtime):
    torch = sys.modules["torch"]
    fake_runtime.update(device="cuda:1", synchronize=[])
    torch.cuda = SimpleNamespace(
        is_available=lambda: True,
        current_device=lambda: 1,
        device_count=lambda: 2,
        get_device_properties=lambda index: SimpleNamespace(
            name=f"Simulated GPU {index}", total_memory=8 * 1024**3
        ),
        get_device_capability=lambda index: (8, 6),
        get_arch_list=lambda: ["sm_80", "sm_86"],
        synchronize=lambda device: fake_runtime["synchronize"].append(device),
    )
    torch.version = SimpleNamespace(cuda="12.8")
    torch._C = SimpleNamespace(_dispatch_has_kernel_for_dispatch_key=lambda name, key: True)
    sys.modules["torchvision"].extension = SimpleNamespace(_has_ops=lambda: True)
    torch.backends = SimpleNamespace(
        cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
        cudnn=SimpleNamespace(version=lambda: 91002, allow_tf32=True, benchmark=True),
    )
    torch.__version__ = "2.10.0+cu128"
    sys.modules["torchvision"].__version__ = "0.25.0+cu128"
    return fake_runtime


def cuda_environment():
    from test_export_runner import environment

    value = environment()
    value.update(device="cuda:1", torch="2.10.0+cu128", torchvision="0.25.0+cu128")
    value["cuda"] = {
        "runtime": "12.8",
        "cudnn": 91002,
        "index": 1,
        "name": "Synthetic GPU",
        "capability": [8, 6],
        "total_memory": 8 * 1024**3,
        "tf32_matmul": False,
        "tf32_cudnn": False,
        "cudnn_benchmark": False,
    }
    return value


@pytest.mark.parametrize("target,source", [("cuda", "cpu"), ("cuda", "cuda:1"), ("cpu", "cuda:0")])
def test_v2_inspection_is_standalone_and_keeps_reference_device(tmp_path, target, source):
    directory, manifest, reference = modern_bundle(tmp_path, target=target, source=source)
    assert runner.validate_bundle(directory) == (manifest, reference)
    result = subprocess.run(
        [sys.executable, "-I", "-S", str(directory / "run.py"), "inspect"],
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    assert json.loads(result.stdout)["runtime_loaded"] is False
    assert manifest["source"]["reference_device"] == source


def test_cuda_runner_loads_on_cpu_and_transfers_model_and_inputs_to_selected_gpu(
    tmp_path, cuda_runtime
):
    directory, manifest, _ = modern_bundle(tmp_path)
    detector = runner.Detector(directory, manifest, "cuda:1")
    assert len(cuda_runtime["loads"]) == 1
    assert cuda_runtime["synchronize"] == ["cuda:1"]
    result = detector.predict(Image.new("RGB", (20, 10)))
    assert result["input_size"] == [20, 10]
    assert cuda_runtime["synchronize"] == ["cuda:1"] * 5
    environment = detector.environment()
    runner._environment(environment, manifest["profile"])
    assert environment["cuda"]["tf32_matmul"] is False
    assert environment["cuda"]["tf32_cudnn"] is False
    assert environment["cuda"]["cudnn_benchmark"] is False


def test_explicit_runtime_check_does_not_build_load_or_predict(tmp_path, cuda_runtime, capsys):
    directory, _, _ = modern_bundle(tmp_path)
    assert runner.main(["--bundle", str(directory), "check-runtime", "--device", "cuda:1"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["device"] == "cuda:1"
    assert result["dependencies_available"] is True
    assert result["model_loaded"] is result["inference_verified"] is False
    assert not any(cuda_runtime[key] for key in ("builder", "loads", "forward", "synchronize"))


@pytest.mark.parametrize("device", ["cpu", "cuda:2", "cuda:-1", "cuda:01", "mps", "cuda:10000"])
def test_bad_device_never_falls_back_or_constructs_model(tmp_path, cuda_runtime, device):
    directory, manifest, _ = modern_bundle(tmp_path)
    with pytest.raises((RuntimeError, ValueError)):
        runner.Detector(directory, manifest, device)
    assert not cuda_runtime["builder"] and not cuda_runtime["loads"]


def test_cuda_unavailable_has_explicit_error_before_loading(tmp_path, cuda_runtime):
    directory, manifest, _ = modern_bundle(tmp_path)
    sys.modules["torch"].cuda.is_available = lambda: False
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        runner.Detector(directory, manifest)
    assert not cuda_runtime["builder"] and not cuda_runtime["loads"]


def test_rocm_runtime_cannot_be_misidentified_as_nvidia_cuda(tmp_path, cuda_runtime):
    directory, manifest, _ = modern_bundle(tmp_path)
    sys.modules["torch"].version.cuda = None
    with pytest.raises(RuntimeError, match="NVIDIA CUDA build"):
        runner.Detector(directory, manifest)
    assert not cuda_runtime["builder"] and not cuda_runtime["loads"]


@pytest.mark.parametrize("missing", ["extension", "torchvision::nms", "torchvision::roi_align"])
def test_cuda_runtime_check_requires_registered_torchvision_cuda_operators(
    tmp_path, cuda_runtime, missing
):
    _, manifest, _ = modern_bundle(tmp_path)
    if missing == "extension":
        sys.modules["torchvision"].extension._has_ops = lambda: False
    else:
        sys.modules["torch"]._C._dispatch_has_kernel_for_dispatch_key = lambda name, key: (
            name != missing
        )
    with pytest.raises(RuntimeError, match="CUDA detection operators are missing"):
        runner.check_runtime(manifest)
    assert not any(cuda_runtime[key] for key in ("builder", "loads", "forward", "synchronize"))


@pytest.mark.parametrize("architectures", [[], ["sm_75"], ["sm_89"], ["compute_90"], ["unknown"]])
def test_cuda_runtime_check_rejects_unsupported_gpu_architecture(
    tmp_path, cuda_runtime, architectures
):
    _, manifest, _ = modern_bundle(tmp_path)
    sys.modules["torch"].cuda.get_arch_list = lambda: architectures
    with pytest.raises(RuntimeError, match="GPU architecture"):
        runner.check_runtime(manifest)
    assert not any(cuda_runtime[key] for key in ("builder", "loads", "forward", "synchronize"))


@pytest.mark.parametrize("architectures", [["sm_80"], ["sm_86"], ["compute_75"], ["compute_86"]])
def test_cuda_runtime_probe_accepts_compatible_binary_or_ptx_without_inference(
    tmp_path, cuda_runtime, architectures
):
    _, manifest, _ = modern_bundle(tmp_path)
    sys.modules["torch"].cuda.get_arch_list = lambda: architectures
    assert runner.check_runtime(manifest)["dependencies_available"] is True
    assert not any(cuda_runtime[key] for key in ("builder", "loads", "forward", "synchronize"))


def test_cuda_cli_saves_exact_mismatch_with_observed_device(tmp_path, cuda_runtime, capsys):
    directory, manifest, reference = modern_bundle(tmp_path)
    output = tmp_path / "measurement.json"
    code = runner.main(
        [
            "--bundle",
            str(directory),
            "measure",
            "--device",
            "cuda:1",
            "--repeats",
            "1",
            "--output",
            str(output),
        ]
    )
    assert code == 3
    assert json.loads(capsys.readouterr().out)["parity_passed"] is False
    payload = json.loads(output.read_bytes())
    assert payload["environment"]["device"] == "cuda:1"
    assert runner.validate_measurement(manifest, reference, payload)["parity_passed"] is False


def test_gpu_measurement_retains_environment_and_exact_cross_device_mismatch(tmp_path):
    _, manifest, reference = modern_bundle(tmp_path)
    payload = measurement(manifest, reference)
    payload["environment"] = cuda_environment()
    assert runner.validate_measurement(manifest, reference, payload)["parity_passed"]
    payload["samples"][0]["detections"][0]["score"] += 1e-12
    result = runner.validate_measurement(manifest, reference, payload)
    assert result["parity_passed"] is result["execution_verified"] is False
    assert result["mismatched_samples"] == [{"frame_id": "frame1", "repeat": 1}]


@pytest.mark.parametrize(
    "change",
    [
        lambda env: env.pop("cuda"),
        lambda env: env.update(device="cpu"),
        lambda env: env.update(device="cuda"),
        lambda env: env.update(torch="2.10.0+cpu"),
        lambda env: env["cuda"].update(index=0),
        lambda env: env["cuda"].update(index=True),
        lambda env: env["cuda"].update(total_memory=0),
        lambda env: env["cuda"].update(capability=[8, True]),
        lambda env: env["cuda"].update(runtime="unknown"),
        lambda env: env["cuda"].update(tf32_matmul=True),
        lambda env: env["cuda"].update(tf32_cudnn=0),
        lambda env: env["cuda"].update(cudnn_benchmark=True),
    ],
)
def test_inconsistent_cuda_evidence_is_rejected(tmp_path, change):
    _, manifest, reference = modern_bundle(tmp_path)
    payload = measurement(manifest, reference)
    payload["environment"] = cuda_environment()
    change(payload["environment"])
    with pytest.raises(ValueError):
        runner.validate_measurement(manifest, reference, payload)


@pytest.mark.parametrize(
    "source,target", [("cpu", "cpu"), ("cpu", "cuda"), ("cuda:1", "cpu"), ("cuda:1", "cuda")]
)
def test_export_target_independent_of_training_and_reference_device(
    tmp_path, monkeypatch, source, target
):
    workspace = fixture_workspace(tmp_path)
    store, model, evaluation, _ = workspace
    store.update(
        "trained_models", model["id"], {"metadata": {**model["metadata"], "device": "cuda:0"}}
    )
    if source != "cpu":
        row = store.get("evaluations", evaluation["id"])
        store.update("evaluations", row["id"], {"config": {**row["config"], "device": "cuda"}})
        run = store.list("evaluation_models", evaluation_id=row["id"])[0]
        store.update(
            "evaluation_models",
            run["id"],
            {
                "metadata": {
                    **run["metadata"],
                    "device": source,
                    "torch_version": "2.10.0+cu128",
                    "torchvision_version": "0.25.0+cu128",
                }
            },
        )
    values = {**options(workspace), "target_device": target}
    preview = exports.preview_export(store, **values)
    row = exports.create_export(
        store,
        **values,
        request_id=preview["request_id"],
        expected_fingerprint=preview["fingerprint"],
    )
    assert (
        exports.create_export(
            store,
            **values,
            request_id=preview["request_id"],
            expected_fingerprint=preview["fingerprint"],
        )["id"]
        == row["id"]
    )
    exports.run_export(store, row["id"], lambda *_: None, lambda: False)
    published = exports.export_detail(store, row["id"])
    manifest = published["manifest"]
    assert manifest["profile"]["device"] == target
    assert manifest["source"].get("reference_device", "cpu") == source
    assert manifest["format"] == (
        "iris-model-export-v1" if source == target == "cpu" else "iris-model-export-v2"
    )
    payload = exported_measurement(published)
    if target == "cuda":
        payload["environment"] = cuda_environment()
    preview = exports.preview_measurement(store, row["id"], payload)
    saved = exports.save_measurement(store, row["id"], payload, preview["fingerprint"])
    assert saved["summary"]["parity_passed"] is True
    # Future runner resources may differ without invalidating an old frozen bundle.
    resources = exports._resources()
    resources["run.py"] += b"\n# simulated later runner version\n"
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


def test_export_api_binds_device_to_preview_and_rejects_unknown_targets(tmp_path):
    workspace = fixture_workspace(tmp_path)
    store = workspace[0]
    client = TestClient(create_app(store.root, run_jobs=False), base_url="http://127.0.0.1")
    values = {**options(workspace), "target_device": "cuda"}
    preview = client.post("/api/model-exports/preview", json=values)
    assert preview.status_code == 200
    bound = {
        **values,
        "request_id": preview.json()["request_id"],
        "expected_fingerprint": preview.json()["fingerprint"],
    }
    changed = client.post("/api/model-exports", json={**bound, "target_device": "cpu"})
    assert changed.status_code == 409
    created = client.post("/api/model-exports", json=bound)
    assert created.status_code == 202
    assert client.post("/api/model-exports", json=bound).json()["id"] == created.json()["id"]
    assert (
        client.post(
            "/api/model-exports/preview", json={**values, "target_device": "mps"}
        ).status_code
        == 422
    )


def test_cuda_reference_cannot_claim_a_cpu_only_runtime(tmp_path):
    workspace = fixture_workspace(tmp_path)
    store, _, evaluation, _ = workspace
    source = store.get("evaluations", evaluation["id"])
    store.update("evaluations", source["id"], {"config": {**source["config"], "device": "cuda"}})
    run = store.list("evaluation_models", evaluation_id=source["id"])[0]
    store.update(
        "evaluation_models", run["id"], {"metadata": {**run["metadata"], "device": "cuda:0"}}
    )
    with pytest.raises(ValueError, match="native float32 export profile"):
        exports.preview_export(store, **options(workspace), target_device="cpu")
