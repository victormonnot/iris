"""Read-only training device inspection; model loading stays inside job workers."""

from __future__ import annotations

import importlib.metadata
import json
import re
import subprocess
import sys

from iris.models import RUNTIME_VERSIONS

DEVICE_PATTERN = r"cpu|cuda(?::(?:0|[1-9][0-9]{0,2}))?"
CUDA_PROTOCOL = "iris-training-state-cuda-v1"


def normalize_device(device):
    if not isinstance(device, str) or not re.fullmatch(DEVICE_PATTERN, device):
        raise ValueError("Choose cpu, cuda, or cuda:<GPU index>")
    return "cuda:0" if device == "cuda" else device


def device_label(device):
    return "CPU" if device == "cpu" else f"NVIDIA GPU ({device})"


def _versions():
    versions = {}
    for package in RUNTIME_VERSIONS:
        try:
            versions[package + "_version"] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package + "_version"] = None
    return versions


def _cuda_devices(torch, torchvision):
    """Inspect properties and registered kernels without tensor/model execution."""
    if not torch.version.cuda:
        raise RuntimeError("This PyTorch installation is CPU-only. Use a compatible CUDA runtime.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Check the NVIDIA driver and CUDA runtime.")
    if not torchvision.extension._has_ops() or any(
        not torch._C._dispatch_has_kernel_for_dispatch_key(name, "CUDA")
        for name in ("torchvision::nms", "torchvision::roi_align")
    ):
        raise RuntimeError(
            "Torchvision CUDA detection operators are missing; install matching builds."
        )
    architectures = torch.cuda.get_arch_list()
    devices = []
    for index in range(min(torch.cuda.device_count(), 1000)):
        properties = torch.cuda.get_device_properties(index)
        capability = [properties.major, properties.minor]
        code = properties.major * 10 + properties.minor
        supported = any(
            (item.startswith("compute_") and int(item[8:]) <= code)
            or (
                item.startswith("sm_")
                and int(item[3:]) // 10 == properties.major
                and int(item[3:]) <= code
            )
            for item in architectures
            if re.fullmatch(r"(?:sm|compute)_[0-9]+", item)
        )
        devices.append(
            {
                "id": f"cuda:{index}",
                "label": f"NVIDIA GPU {index}: {properties.name}",
                "available": supported,
                "reason": ""
                if supported
                else "This PyTorch build does not support this GPU architecture.",
                "hardware": properties.name,
                "uuid": str(properties.uuid) if getattr(properties, "uuid", None) else None,
                "capability": capability,
                "memory_bytes": properties.total_memory,
            }
        )
    return {"devices": devices, "cuda_version": str(torch.version.cuda)}


def _probe_cuda():
    # Keep CUDA contexts and optional import failures out of the HTTP process.
    result = subprocess.run(
        [sys.executable, "-m", "iris.training_device", "--probe-cuda"],
        capture_output=True,
        timeout=20,
        check=False,
    )
    if result.returncode or len(result.stdout) > 64 * 1024:
        raise RuntimeError("The CUDA runtime probe failed; check the installed NVIDIA runtime.")
    payload = json.loads(result.stdout)
    if payload.get("error"):
        raise RuntimeError(payload["error"])
    if not isinstance(payload.get("devices"), list) or not isinstance(
        payload.get("cuda_version"), str
    ):
        raise RuntimeError("The CUDA runtime probe returned an invalid result")
    return payload


def available_devices():
    runtime = {**_versions(), "cuda_version": None}
    devices = [{"id": "cpu", "label": "CPU", "available": True, "reason": ""}]
    try:
        for package, expected in RUNTIME_VERSIONS.items():
            installed = runtime[package + "_version"]
            if not installed or installed.split("+", 1)[0] != expected:
                raise RuntimeError(
                    f"Training requires {package} {expected}; found {installed or 'not installed'}."
                )
        if runtime["torch_version"].endswith("+cpu"):
            raise RuntimeError(
                "This PyTorch installation is CPU-only. Use a compatible CUDA runtime."
            )
        result = _probe_cuda()
        runtime["cuda_version"] = result["cuda_version"]
        devices.extend(result["devices"])
        if not result["devices"]:
            raise RuntimeError("No NVIDIA CUDA device is visible to the IRIS server.")
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
        devices.append(
            {"id": "cuda:0", "label": "NVIDIA GPU (CUDA)", "available": False, "reason": str(exc)}
        )
    return {
        "devices": devices,
        "runtime": runtime,
        "notes": [
            "These devices belong to the IRIS server, not necessarily the browser computer.",
            "Training and inference devices are independent. No automatic CPU fallback is used.",
            "Device inspection does not load a model or establish available training memory.",
        ],
    }


def resolve_device(device, *, expected=None):
    device = normalize_device(device)
    if device == "cpu":
        return device, None
    status = available_devices()
    selected = next((item for item in status["devices"] if item["id"] == device), None)
    if not selected or not selected["available"]:
        raise RuntimeError(
            selected["reason"] if selected else f"The selected GPU {device} is unavailable."
        )
    identity = {
        "device": device,
        **{key: selected[key] for key in ("hardware", "uuid", "capability", "memory_bytes")},
        **status["runtime"],
    }
    if expected is not None and identity != expected:
        raise ValueError("The selected GPU or runtime changed; prepare a new training preview.")
    return device, identity


if __name__ == "__main__":
    if sys.argv[1:] != ["--probe-cuda"]:
        raise SystemExit("Use --probe-cuda to inspect the installed CUDA runtime")
    try:
        import torch
        import torchvision

        result = _cuda_devices(torch, torchvision)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        result = {"error": str(exc)}
    print(json.dumps(result, allow_nan=False))
