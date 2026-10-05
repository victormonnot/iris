"""CPU/CUDA routing and lifecycle on synthetic metadata and tiny CPU tensors only."""

import subprocess
from copy import deepcopy
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from test_training_recovery import SimulatedTrainer, execute, fixture_workspace, queue, resumed
from test_training_scopes import structural_trainer

from iris import models, training, training_device
from iris import training_recovery as recovery
from iris.app import create_app
from iris.jobs import JobManager
from iris.store import Store
from iris.workspace_archive import create_archive
from iris.workspace_restore import inspect_archive, restore_archive

structural_trainer = structural_trainer


def cuda_status():
    return {
        "devices": [
            {"id": "cpu", "label": "CPU", "available": True, "reason": ""},
            {
                "id": "cuda:0",
                "label": "Simulated NVIDIA GPU",
                "available": True,
                "reason": "",
                "hardware": "Synthetic CUDA fixture; not GPU evidence",
                "uuid": "simulation-0",
                "capability": [8, 6],
                "memory_bytes": 8 * 1024**3,
            },
        ],
        "runtime": {
            "torch_version": "2.10.0+cu128",
            "torchvision_version": "0.25.0+cu128",
            "cuda_version": "12.8",
        },
    }


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    value = fixture_workspace(tmp_path)
    monkeypatch.setattr(training, "catalog", lambda _root: [value[2]])
    status = cuda_status()
    monkeypatch.setattr(training_device, "available_devices", lambda: deepcopy(status))
    return value, status


@pytest.mark.parametrize(
    "device", ["gpu", "mps", "cuda:-1", "cuda:01", "cuda:1000", "cuda:0\n", None, True]
)
def test_invalid_devices_are_rejected_before_inspection(monkeypatch, device):
    monkeypatch.setattr(training_device, "available_devices", lambda: pytest.fail("No probe"))
    with pytest.raises(ValueError, match="Choose cpu"):
        training_device.resolve_device(device)


def test_cpu_does_not_import_or_probe_cuda(monkeypatch):
    monkeypatch.setattr(training_device, "_probe_cuda", lambda: pytest.fail("No CUDA probe"))
    monkeypatch.setattr(
        training_device,
        "_versions",
        lambda: {"torch_version": "2.10.0+cpu", "torchvision_version": "0.25.0+cpu"},
    )
    assert training_device.resolve_device("cpu") == ("cpu", None)
    result = training_device.available_devices()
    assert result["devices"][0]["available"] is True
    assert result["devices"][1]["available"] is False
    assert "CPU-only" in result["devices"][1]["reason"]


@pytest.mark.parametrize(
    "error", [RuntimeError("driver failed"), subprocess.TimeoutExpired("probe", 20)]
)
def test_failed_probe_reports_gpu_unavailable_without_affecting_cpu(monkeypatch, error):
    monkeypatch.setattr(
        training_device,
        "_versions",
        lambda: {"torch_version": "2.10.0+cu128", "torchvision_version": "0.25.0+cu128"},
    )

    def fail():
        raise error

    monkeypatch.setattr(training_device, "_probe_cuda", fail)
    result = training_device.available_devices()
    assert result["devices"][0]["available"]
    assert not result["devices"][1]["available"]
    assert result["devices"][1]["reason"]


@pytest.mark.parametrize(
    "architectures,available",
    [
        (["sm_86"], True),
        (["compute_80"], True),
        (["sm_80"], True),
        (["sm_75"], False),
        (["sm_90"], False),
    ],
)
def test_gpu_properties_and_kernel_check_do_not_execute_tensors(architectures, available):
    torch = SimpleNamespace(
        version=SimpleNamespace(cuda="12.8"),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            get_arch_list=lambda: architectures,
            device_count=lambda: 1,
            get_device_properties=lambda _: SimpleNamespace(
                name="Simulated GPU", uuid="fake", major=8, minor=6, total_memory=8 * 1024**3
            ),
        ),
        _C=SimpleNamespace(_dispatch_has_kernel_for_dispatch_key=lambda *_: True),
    )
    vision = SimpleNamespace(extension=SimpleNamespace(_has_ops=lambda: True))
    result = training_device._cuda_devices(torch, vision)
    assert result["devices"][0]["available"] is available
    torch._C._dispatch_has_kernel_for_dispatch_key = lambda *_: False
    with pytest.raises(RuntimeError, match="operators"):
        training_device._cuda_devices(torch, vision)


def test_gpu_preview_freezes_device_and_changed_hardware_prevents_create(workspace):
    values, status = workspace
    store, dataset, parent = values
    options = dict(
        name="GPU fixture",
        dataset_id=dataset["id"],
        parent_model_id=parent["id"],
        device="cuda",
        steps=3,
    )
    preview = training.preview_training(store, **options)
    config = preview["config"]
    assert config["device"] == preview["workload"]["device"] == "cuda:0"
    assert config["checkpoint_protocol"] == training_device.CUDA_PROTOCOL
    assert config["deterministic_algorithms"] is False
    assert config["device_identity"]["uuid"] == "simulation-0"
    status["devices"][1]["uuid"] = "replacement"
    with pytest.raises(ValueError, match="changed"):
        training.create_training(
            store,
            JobManager(store),
            **options,
            request_id=preview["request_id"],
            expected_fingerprint=preview["fingerprint"],
        )
    assert not store.list("jobs")


def test_gpu_failure_and_resume_preserve_device_and_source_across_archive(workspace, tmp_path):
    values, status = workspace
    store = values[0]
    source = queue(values, device="cuda:0")

    class StopAfterTen(SimulatedTrainer):
        def step(self, *args):
            if self.steps == 10:
                raise RuntimeError("Injected GPU simulation failure")
            return super().step(*args)

    with pytest.raises(RuntimeError, match="simulation"):
        execute(store, source, trainer_factory=StopAfterTen)
    store.update("jobs", source["job_id"], {"status": "failed"})
    before = deepcopy(store.get("training_runs", source["id"]))
    status["devices"][1]["available"] = False
    status["devices"][1]["reason"] = "Simulated GPU unavailable"
    with pytest.raises(RuntimeError, match="unavailable"):
        recovery.preview_resume(store, source["id"])
    assert len(store.list("jobs")) == 1
    status["devices"][1]["available"] = True
    child = resumed(store, source)
    assert child["config"]["device"] == "cuda:0"
    execute(store, child)
    assert store.get("training_runs", source["id"]) == before
    assert training.training_detail(store, child["id"])["checkpoint"]
    archive = tmp_path / "gpu.iris.zip"
    create_archive(store.root, archive)
    restore_archive(
        archive,
        tmp_path / "restored",
        expected_archive_sha256=inspect_archive(archive)["archive_sha256"],
    )
    restored = Store(tmp_path / "restored")
    assert restored.list("training_checkpoints") == store.list("training_checkpoints")
    assert restored.list("training_runs") == store.list("training_runs")


def test_worker_refuses_replaced_gpu_before_trainer_construction(workspace):
    values, status = workspace
    row = queue(values, device="cuda:0")
    status["devices"][1]["uuid"] = "replacement"
    with pytest.raises(ValueError, match="changed"):
        execute(values[0], row, trainer_factory=lambda *_: pytest.fail("No model load"))
    assert not values[0].list("trained_models")


def test_gpu_constructor_memory_error_is_actionable_and_does_not_publish(workspace):
    torch = pytest.importorskip("torch")
    values, _ = workspace
    row = queue(values, device="cuda:0")

    def fail(*_):
        raise torch.cuda.OutOfMemoryError("Simulated allocation failure, no GPU")

    with pytest.raises(RuntimeError, match="No CPU fallback"):
        execute(values[0], row, trainer_factory=fail)
    assert not values[0].list("trained_models")


def test_device_endpoint_and_api_gpu_plan_are_read_only_and_explicit(workspace):
    values, _ = workspace
    store, dataset, parent = values
    with TestClient(create_app(store.root, run_jobs=False), base_url="http://127.0.0.1") as client:
        devices = client.get("/api/training/devices")
        assert devices.status_code == 200 and devices.json()["devices"][1]["id"] == "cuda:0"
        options = dict(
            name="GPU API fixture",
            dataset_id=dataset["id"],
            parent_model_id=parent["id"],
            device="cuda:0",
            steps=3,
        )
        assert client.post("/api/trainings", json=options).status_code == 422
        preview = client.post("/api/trainings/preview", json=options).json()
        assert not store.list("jobs")
        payload = {
            **options,
            "request_id": preview["request_id"],
            "expected_fingerprint": preview["fingerprint"],
        }
        first = client.post("/api/trainings", json=payload)
        assert first.status_code == 202, first.text
        assert client.post("/api/trainings", json=payload).json()["id"] == first.json()["id"]
        assert len(store.list("jobs")) == 1


def test_step_routes_image_boxes_labels_and_synchronizes_selected_gpu(
    structural_trainer, monkeypatch, tmp_path
):
    make, torch, _ = structural_trainer
    trainer = make()
    real_tensor, real_to = torch.tensor, torch.Tensor.to
    routes, synchronized = [], []
    trainer.device = torch.device("cuda:1")

    def tensor(*args, **kwargs):
        routes.append(str(kwargs.get("device")))
        kwargs["device"] = "cpu"
        return real_tensor(*args, **kwargs)

    def transfer(self, *args, **kwargs):
        if str(kwargs.get("device", "")).startswith("cuda"):
            routes.append(str(kwargs["device"]))
            kwargs["device"] = "cpu"
        return real_to(self, *args, **kwargs)

    monkeypatch.setattr(torch, "tensor", tensor)
    monkeypatch.setattr(torch.Tensor, "to", transfer)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: synchronized.append(str(device)))
    trainer.step(Image.new("RGB", (8, 8)), [{"label": "person", "box": [1, 1, 5, 6]}])
    assert routes == ["cuda:1"] * 3 and synchronized == ["cuda:1"]
    trainer.write_checkpoint(tmp_path / "weights.pth")
    assert all(
        value.device.type == "cpu"
        for value in torch.load(tmp_path / "weights.pth", weights_only=True).values()
    )


def test_inference_cuda_policy_matches_export_without_loading_model():
    selected = []
    torch = SimpleNamespace(
        version=SimpleNamespace(cuda="12.8"),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 2,
            set_device=lambda device: selected.append(device.index),
            get_device_properties=lambda _: SimpleNamespace(
                name="Synthetic GPU", major=8, minor=6, total_memory=123, uuid="fake"
            ),
        ),
        backends=SimpleNamespace(
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
            cudnn=SimpleNamespace(allow_tf32=True, benchmark=True, version=lambda: 90000),
        ),
    )
    metadata = models._configure_cuda(torch, SimpleNamespace(index=1))
    assert selected == [1]
    assert metadata["index"] == 1 and metadata["uuid"] == "fake"
    assert metadata["tf32_matmul"] is metadata["tf32_cudnn"] is metadata["cudnn_benchmark"] is False
    with pytest.raises(RuntimeError, match="does not exist"):
        models._configure_cuda(torch, SimpleNamespace(index=2))
    torch.version.cuda = None
    with pytest.raises(RuntimeError, match="NVIDIA CUDA"):
        models._configure_cuda(torch, SimpleNamespace(index=0))


def test_replacement_prediction_head_moves_to_selected_device_before_optimizer(
    structural_trainer, monkeypatch
):
    make, torch, _ = structural_trainer
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    real_to = torch.Tensor.to
    routes, selected = [], []

    def transfer(self, *args, **kwargs):
        if args and str(args[0]).startswith("cuda"):
            routes.append(str(args[0]))
            args = (torch.device("cpu"), *args[1:])
        return real_to(self, *args, **kwargs)

    monkeypatch.setattr(training, "normalize_device", lambda _: "cuda:1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: selected.append(str(device)))
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda _: None)
    monkeypatch.setattr(torch.Tensor, "to", transfer)
    for owner, key in (
        (torch.backends.cuda.matmul, "allow_tf32"),
        (torch.backends.cudnn, "allow_tf32"),
        (torch.backends.cudnn, "benchmark"),
        (torch.backends.cudnn, "deterministic"),
    ):
        monkeypatch.setattr(owner, key, getattr(owner, key))
    try:
        trainer = make(origin="official")
        assert selected == ["cuda:1"] and routes == ["cuda:1"] * 4
        assert trainer.metadata["training_device"] == "cuda:1"
        assert trainer.metadata["deterministic_algorithms"] is False
        assert trainer.metadata["checkpoint_storage_device"] == "cpu"
        assert trainer.metadata["inference_devices"] == ["cpu", "cuda"]
        assert all(tensor.device.type == "cpu" for tensor in trainer.initial.values())
    finally:
        torch.use_deterministic_algorithms(previous_determinism)


@pytest.mark.parametrize(
    "change",
    [
        {"checkpoint_protocol": recovery.PROTOCOL},
        {"device": "cpu"},
        {"deterministic_algorithms": True},
        {"device_identity": None},
        {"precision": "float16"},
    ],
)
def test_cuda_checkpoint_contract_cannot_silently_change(workspace, change):
    row = queue(workspace[0], device="cuda:0")
    with pytest.raises(ValueError):
        recovery.validate_config({**row["config"], **change})
