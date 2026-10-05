"""Validated CPU/CUDA SGD continuation state, separate from inference weights."""

import hashlib
import importlib.metadata
import json
import math
import os
import pickle
import platform
import random
import stat
import zipfile
from copy import deepcopy
from pathlib import Path

PROTOCOL = "iris-training-state-v1"
CUDA_PROTOCOL = "iris-training-state-cuda-v1"
MAX_STATE_BYTES = 512 * 1024 * 1024
_FIELDS = {
    "protocol",
    "binding",
    "sampler",
    "model_state",
    "optimizer_state",
    "optimizer_contract",
    "module_modes",
    "gradient_modules",
    "torch_rng_state",
}


def _device(trainer):
    torch = trainer.torch
    device = getattr(trainer, "device", None)
    if device is None:
        device = next(trainer.model.parameters()).device
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def runtime_identity(trainer) -> dict:
    """Bind continuation to one concrete runtime and selected execution device.

    CUDA RNG restoration preserves continuation state, but does not promise
    bitwise equivalence for nondeterministic detection operators.
    """
    torch = trainer.torch
    device = _device(trainer)
    if device.type not in {"cpu", "cuda"} or any(
        value.device != device
        or (value.is_floating_point() and value.dtype != torch.float32)
        or value.is_complex()
        for value in (*trainer.model.parameters(), *trainer.model.buffers())
    ):
        raise ValueError("Training continuation requires a CPU float32 or CUDA float32 model")
    identity = {
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "torchvision_version": importlib.metadata.version("torchvision"),
        "machine": platform.machine(),
        "device": str(device),
        "precision": "float32",
        "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        uuid = getattr(properties, "uuid", None)
        identity.update(
            cuda_version=torch.version.cuda,
            cudnn_version=torch.backends.cudnn.version(),
            gpu_name=properties.name,
            gpu_capability=[properties.major, properties.minor],
            gpu_uuid=str(uuid) if uuid is not None else None,
            device_index=device.index,
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            cudnn_deterministic=torch.backends.cudnn.deterministic,
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        )
    return identity


def _protocol(trainer, binding: dict) -> str:
    protocol = binding.get("protocol", PROTOCOL) if type(binding) is dict else None
    device = _device(trainer)
    expected = CUDA_PROTOCOL if device.type == "cuda" else PROTOCOL
    if device.type not in {"cpu", "cuda"} or protocol != expected:
        raise ValueError("Unsupported training state protocol for the selected device")
    return protocol


def _cpu_state(torch, value):
    """Keep the durable archive independent of accelerator tensor locations."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_state(torch, item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_state(torch, item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_state(torch, item) for item in value)
    return value


def _json_identity(value) -> str:
    """Reject opaque objects and compare JSON scalars without bool/int coercion."""

    def check(item):
        if type(item) is dict:
            if any(type(key) is not str for key in item):
                raise ValueError("Training state metadata requires string keys")
            for child in item.values():
                check(child)
        elif type(item) is list:
            for child in item:
                check(child)
        elif item is None or type(item) in {str, bool, int}:
            return
        elif type(item) is float and math.isfinite(item):
            return
        else:
            raise ValueError("Training state metadata must contain finite JSON values")

    check(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _validate_sampler(sampler: dict) -> None:
    if type(sampler) is not dict or set(sampler) != {"random_state", "remaining_order"}:
        raise ValueError("Training state sampler is invalid")
    state, order = sampler["random_state"], sampler["remaining_order"]
    if (
        type(order) is not list
        or any(type(index) is not int or index < 0 for index in order)
        or len(order) != len(set(order))
        or type(state) is not list
        or len(state) != 3
        or type(state[0]) is not int
        or state[0] != 3
        or type(state[1]) is not list
        or len(state[1]) != 625
        or any(type(value) is not int or not 0 <= value <= 0xFFFFFFFF for value in state[1])
        or not 0 <= state[1][-1] <= 624
        or (state[2] is not None and (type(state[2]) is not float or not math.isfinite(state[2])))
    ):
        raise ValueError("Training state sampler is invalid")
    # Validate independently without consuming or replacing the process RNG.
    try:
        random.Random().setstate((state[0], tuple(state[1]), state[2]))
    except (TypeError, ValueError) as exc:
        raise ValueError("Training state sampler is invalid") from exc


def _tensor(trainer, value, expected, label: str) -> None:
    torch = trainer.torch
    if (
        type(value) is not torch.Tensor
        or value.device.type != "cpu"
        or value.layout != torch.strided
        or value.dtype != expected.dtype
        or value.shape != expected.shape
        or not torch.isfinite(value).all().item()
    ):
        raise ValueError(f"Training state has an invalid {label} tensor")


def _optimizer_contract(trainer) -> list[dict]:
    torch = trainer.torch
    if type(trainer.optimizer) is not torch.optim.SGD:
        raise ValueError("Training continuation supports the frozen SGD optimizer only")
    names = {id(parameter): name for name, parameter in trainer.selected_parameters.items()}
    seen = []
    groups = []
    for group in trainer.optimizer.param_groups:
        parameter_ids = [id(parameter) for parameter in group["params"]]
        if any(identifier not in names for identifier in parameter_ids):
            raise ValueError("Training optimizer contains an unselected parameter")
        seen.extend(parameter_ids)
        settings = {key: value for key, value in group.items() if key != "params"}
        _json_identity(settings)
        groups.append(
            {"parameter_names": [names[identifier] for identifier in parameter_ids], **settings}
        )
    if len(seen) != len(set(seen)) or set(seen) != set(names):
        raise ValueError("Training optimizer parameter order or membership is invalid")
    return groups


def _validate_optimizer(trainer, saved: dict, contract: list[dict], *, completed: bool) -> None:
    if _json_identity(contract) != _json_identity(_optimizer_contract(trainer)):
        raise ValueError("Training state optimizer parameter order or settings changed")
    current = trainer.optimizer.state_dict()
    if (
        type(saved) is not dict
        or set(saved) != {"state", "param_groups"}
        or type(saved["state"]) is not dict
        or _json_identity(saved["param_groups"]) != _json_identity(current["param_groups"])
    ):
        raise ValueError("Training state optimizer groups changed")
    parameters = {}
    momentum_ids = set()
    for encoded, live in zip(current["param_groups"], trainer.optimizer.param_groups, strict=True):
        for identifier, parameter in zip(encoded["params"], live["params"], strict=True):
            parameters[identifier] = parameter
            if live["momentum"]:
                momentum_ids.add(identifier)
    if (
        any(type(identifier) is not int for identifier in saved["state"])
        or not set(saved["state"]) <= momentum_ids
        or (completed and set(saved["state"]) != momentum_ids)
    ):
        raise ValueError("Training state optimizer momentum entries are incomplete or invalid")
    for identifier, values in saved["state"].items():
        if type(values) is not dict or set(values) != {"momentum_buffer"}:
            raise ValueError("Training state optimizer momentum entry is invalid")
        _tensor(trainer, values["momentum_buffer"], parameters[identifier], "momentum")


def _validate_payload(trainer, payload: dict, binding: dict, sampler: dict) -> None:
    protocol = _protocol(trainer, binding)
    fields = _FIELDS | {"cuda_rng_state"} if protocol == CUDA_PROTOCOL else _FIELDS
    if type(payload) is not dict or set(payload) != fields or payload["protocol"] != protocol:
        raise ValueError("Unsupported training state protocol")
    if type(binding) is not dict or _json_identity(payload["binding"]) != _json_identity(binding):
        raise ValueError("Training state does not match its frozen run binding")
    if "runtime" in binding and _json_identity(binding["runtime"]) != _json_identity(
        runtime_identity(trainer)
    ):
        raise ValueError("Training state runtime changed")
    _validate_sampler(sampler)
    _validate_sampler(payload["sampler"])
    if _json_identity(payload["sampler"]) != _json_identity(sampler):
        raise ValueError("Training state sampler differs from its frozen record")
    current = trainer.model.state_dict()
    saved = payload["model_state"]
    if type(saved) is not dict or set(saved) != set(current):
        raise ValueError("Training state model keys changed")
    for name, tensor in saved.items():
        _tensor(trainer, tensor, current[name], "model")
    for name, expected in trainer.frozen_initial.items():
        digest = hashlib.sha256(saved[name].detach().contiguous().numpy().tobytes()).hexdigest()
        if digest != expected:
            raise ValueError("Training state changed frozen model weights")
    for name, expected in trainer.initial_buffers.items():
        if name not in saved or not trainer.torch.equal(saved[name], expected.detach().cpu()):
            raise ValueError("Training state changed frozen model buffers")
    _validate_optimizer(
        trainer,
        payload["optimizer_state"],
        payload["optimizer_contract"],
        completed=bool(binding.get("step", 0)),
    )
    modules = dict(trainer.model.named_modules())
    modes = payload["module_modes"]
    if (
        type(modes) is not dict
        or set(modes) != set(modules)
        or any(type(mode) is not bool for mode in modes.values())
    ):
        raise ValueError("Training state model modes are invalid")
    if any(
        modes.get(name) is not False for name in getattr(trainer, "frozen_batchnorm_modules", [])
    ):
        raise ValueError("Training state changed frozen batch normalization modes")
    gradients = payload["gradient_modules"]
    if (
        type(gradients) is not list
        or any(type(name) is not str for name in gradients)
        or gradients != sorted(set(gradients))
        or not set(gradients) <= set(trainer.scope["trainable_modules"])
    ):
        raise ValueError("Training state gradient modules are invalid")
    _tensor(trainer, payload["torch_rng_state"], trainer.torch.get_rng_state(), "CPU RNG")
    try:
        trainer.torch.Generator(device="cpu").set_state(payload["torch_rng_state"])
    except RuntimeError as exc:
        raise ValueError("Training state CPU RNG is invalid") from exc
    if protocol == CUDA_PROTOCOL:
        device = _device(trainer)
        _tensor(
            trainer,
            payload["cuda_rng_state"],
            trainer.torch.cuda.get_rng_state(device),
            "CUDA RNG",
        )
        try:
            # An independent generator checks the state without replacing the
            # selected device's process RNG before the whole payload is valid.
            trainer.torch.Generator(device=device).set_state(payload["cuda_rng_state"])
        except RuntimeError as exc:
            raise ValueError("Training state CUDA RNG is invalid") from exc


def write_state(trainer, path: Path, *, binding: dict, sampler: dict) -> None:
    """Write continuation state without changing modes or the final-weight baseline.

    The caller owns staging, fsync, content hashing and atomic publication.
    """
    protocol = _protocol(trainer, binding)
    payload = {
        "protocol": protocol,
        "binding": deepcopy(binding),
        "sampler": deepcopy(sampler),
        "model_state": _cpu_state(trainer.torch, dict(trainer.model.state_dict())),
        "optimizer_state": _cpu_state(trainer.torch, trainer.optimizer.state_dict()),
        "optimizer_contract": _optimizer_contract(trainer),
        "module_modes": {name: module.training for name, module in trainer.model.named_modules()},
        "gradient_modules": sorted(trainer.gradient_modules),
        "torch_rng_state": trainer.torch.get_rng_state(),
    }
    if protocol == CUDA_PROTOCOL:
        payload["cuda_rng_state"] = trainer.torch.cuda.get_rng_state(_device(trainer))
    _validate_payload(trainer, payload, binding, sampler)
    trainer.torch.save(payload, path)
    if path.stat().st_size > MAX_STATE_BYTES:
        path.unlink(missing_ok=True)
        raise ValueError("Training state exceeds the supported size limit")


def load_state(
    trainer,
    path: Path,
    *,
    binding: dict,
    sampler: dict,
    expected_sha256: str | None = None,
) -> None:
    """Validate against a reconstructed trainer, then restore CPU/CUDA RNG last."""
    if expected_sha256 is not None and (
        type(expected_sha256) is not str
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise ValueError("Training state expected checksum is invalid")
    # torch.save emits stored ZIP members. Bound expanded bytes as well as file
    # bytes before deserialization; a compressed payload is not our protocol.
    try:
        with path.open("rb") as source:
            info = os.fstat(source.fileno())
            if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= MAX_STATE_BYTES:
                raise ValueError("Training state is missing or exceeds the supported size limit")
            if (
                expected_sha256 is not None
                and hashlib.file_digest(source, "sha256").hexdigest() != expected_sha256
            ):
                raise ValueError("Training state checksum changed")
            source.seek(0)
            with zipfile.ZipFile(source) as archive:
                members = archive.infolist()
                if (
                    len(members) > 20000
                    or sum(member.file_size for member in members) > MAX_STATE_BYTES
                    or any(member.compress_type != zipfile.ZIP_STORED for member in members)
                ):
                    raise ValueError("Training state archive exceeds the supported size limit")
            source.seek(0)
            payload = trainer.torch.load(source, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, EOFError, pickle.UnpicklingError, zipfile.BadZipFile) as exc:
        raise ValueError("Training state cannot be read safely") from exc
    _validate_payload(trainer, payload, binding, sampler)
    trainer.model.load_state_dict(payload["model_state"], strict=True)
    # PyTorch restores optimizer tensors onto their corresponding parameter's
    # device. Verify that contract so a resumed CUDA step cannot mix devices.
    trainer.optimizer.load_state_dict(payload["optimizer_state"])
    for parameter, state in trainer.optimizer.state.items():
        if any(value.device != parameter.device for value in state.values()):
            raise ValueError("Training optimizer state did not restore to the model device")
    for name, module in trainer.model.named_modules():
        module.training = payload["module_modes"][name]
    trainer.gradient_modules = set(payload["gradient_modules"])
    trainer.torch.set_rng_state(payload["torch_rng_state"])
    if payload["protocol"] == CUDA_PROTOCOL:
        trainer.torch.cuda.set_rng_state(payload["cuda_rng_state"], device=_device(trainer))
