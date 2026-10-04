"""Standalone SAM 3 worker, invoked with an isolated Python and no IRIS imports.

Only this process imports the optional CUDA stack. Its stdout is reserved for a
bounded JSON-lines protocol; runtime diagnostics go to stderr. No model weights
are installed or downloaded by this helper.
"""

from __future__ import annotations

import base64
import ctypes
import hashlib
import importlib
import importlib.metadata
import io
import json
import math
import os
import platform
import re
import signal
import socket
import sys
import time
from pathlib import Path

CODE_REVISION = "2345a4ad109ac29c569da749c91d84f10dc08c40"
CHECKPOINT_SHA256 = "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e"
CHECKPOINT_BYTES = 3450062241
PACKAGES = {"torch": "2.10.0", "torchvision": "0.25.0", "numpy": "1.26.4"}
RAW_PROTOCOL = "iris-sam3-native-boxes-v1"
MAX_REQUEST_BYTES = 24 * 1024 * 1024
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_BYTES = 16 * 1024 * 1024
MAX_IMAGE_PIXELS = 16777216
MAX_PROMPT_TOKENS = 30


def validate_runtime_identity(identity):
    """Validate frozen facts without consulting the current host or ML packages."""
    if not isinstance(identity, dict):
        raise ValueError("SAM runtime identity must be an object")
    packages, cuda = identity.get("packages"), identity.get("cuda")
    python = identity.get("python")
    if (
        not isinstance(python, str)
        or re.fullmatch(r"3\.12\.[0-9]+", python) is None
        or identity.get("isolated") is not True
        or identity.get("code_revision") != CODE_REVISION
        or not isinstance(identity.get("tokenizer_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", identity["tokenizer_sha256"]) is None
        or not isinstance(packages, dict)
        or any(
            not isinstance(packages.get(name), str) or packages[name].split("+", 1)[0] != required
            for name, required in PACKAGES.items()
        )
        or not isinstance(packages.get("sam3"), str)
        or not 1 <= len(packages["sam3"]) <= 128
        or not isinstance(cuda, dict)
        or cuda.get("available") is not True
        or cuda.get("bfloat16") is not True
        or not isinstance(cuda.get("version"), str)
        or re.fullmatch(r"[0-9]+\.[0-9]+(?:\.[0-9]+)?", cuda["version"]) is None
        or tuple(map(int, cuda["version"].split(".")[:2])) < (12, 6)
        or not isinstance(cuda.get("device"), str)
        or not 1 <= len(cuda["device"]) <= 512
        or not isinstance(cuda.get("capability"), list)
        or len(cuda["capability"]) != 2
        or any(type(value) is not int or value < 0 for value in cuda["capability"])
    ):
        raise ValueError("SAM runtime identity differs from the supported frozen profile")
    _finite_json(identity)
    return identity


def _parent_guard(expected):
    if not sys.platform.startswith("linux"):
        raise RuntimeError("The isolated SAM worker currently requires Linux process controls")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        raise RuntimeError("Unable to install the SAM parent-death process guard")
    # Close the race between Popen and prctl if the owning job already died.
    if os.getppid() != expected:
        raise RuntimeError("The owning SAM job process has already stopped")


def _finite_json(data):
    return json.dumps(data, allow_nan=False, ensure_ascii=False, separators=(",", ":")).encode()


def _reject_constant(value):
    raise ValueError(f"Nonfinite JSON value: {value}")


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _parse(raw):
    return json.loads(raw, parse_constant=_reject_constant, object_pairs_hook=_pairs)


def _deny_network(*args, **kwargs):
    raise RuntimeError("The isolated SAM runtime does not permit network connections")


def _offline():
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    socket.create_connection = _deny_network
    socket.socket.connect = _deny_network
    socket.socket.connect_ex = _deny_network


def _package_identity():
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("The isolated SAM profile requires Python 3.12 with NumPy 1.26.4")
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("IRIS_SAM_PYTHON must belong to a separate virtual environment")
    packages = {}
    for name, expected in PACKAGES.items():
        installed = importlib.metadata.version(name)
        if installed.split("+", 1)[0] != expected:
            raise RuntimeError(f"SAM requires {name} {expected}; installed version is {installed}")
        packages[name] = installed
    distribution = importlib.metadata.distribution("sam3")
    direct = _parse(distribution.read_text("direct_url.json") or "{}")
    vcs = direct.get("vcs_info", {})
    if (
        direct.get("url", "").rstrip("/").removesuffix(".git")
        != "https://github.com/facebookresearch/sam3"
        or vcs.get("vcs") != "git"
        or vcs.get("commit_id") != CODE_REVISION
        or direct.get("dir_info", {}).get("editable", False)
    ):
        raise RuntimeError("SAM must be a noneditable install from the pinned official Git commit")
    packages["sam3"] = distribution.version
    # The tokenizer is packaged with this commit, never fetched from a model hub.
    tokenizer = Path(distribution.locate_file("sam3/assets/bpe_simple_vocab_16e6.txt.gz"))
    if tokenizer.is_symlink() or not tokenizer.is_file():
        raise RuntimeError("The pinned SAM tokenizer asset is missing")
    with tokenizer.open("rb") as source:
        tokenizer_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
    return {
        "python": platform.python_version(),
        "isolated": True,
        "packages": packages,
        "code_revision": CODE_REVISION,
        "tokenizer_sha256": tokenizer_sha256,
    }


def _identity():
    identity = _package_identity()
    torch = importlib.import_module("torch")
    importlib.import_module("torchvision")
    if not torch.cuda.is_available():
        raise RuntimeError("SAM 3 requires an available CUDA GPU")
    version = torch.version.cuda
    try:
        compatible = tuple(int(value) for value in version.split(".")[:2]) >= (12, 6)
    except (TypeError, ValueError, AttributeError):
        compatible = False
    if not compatible:
        raise RuntimeError("SAM 3 requires a CUDA runtime version of at least 12.6")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The frozen SAM profile requires CUDA bfloat16 support")
    # SAM's optional checkout has imports beyond its declared install_requires.
    # Importing these modules checks that boundary without constructing a model.
    importlib.import_module("sam3.model_builder")
    importlib.import_module("sam3.model.sam3_image_processor")
    identity["cuda"] = {
        "available": True,
        "version": version,
        "bfloat16": True,
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
    }
    return validate_runtime_identity(identity)


def _prompts(value):
    if not isinstance(value, list) or not 1 <= len(value) <= 100:
        raise ValueError("SAM requires 1–100 frozen class prompts")
    seen = set()
    for item in value:
        if (
            not isinstance(item, dict)
            or set(item) != {"class_id", "text"}
            or not isinstance(item["class_id"], str)
            or re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", item["class_id"]) is None
            or item["class_id"] in seen
            or not isinstance(item["text"], str)
            or not 1 <= len(item["text"]) <= 120
            or item["text"] != item["text"].strip()
            or not item["text"].isprintable()
        ):
            raise ValueError("SAM class prompts must be bounded, distinct and explicit")
        seen.add(item["class_id"])
    return value


def _threshold(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("SAM threshold must be finite and between zero and one")
    return value


def _box_processor(processor_class, torch):
    """Keep the pinned processor's text/image path without allocating masks.

    Grounding and confidence semantics follow Meta's pinned Sam3Processor:
    sam3/model/sam3_image_processor.py, _forward_grounding. The image benchmark
    retains native box coordinates and query indices, without requesting masks.
    """

    class BoxProcessor(processor_class):
        def _forward_grounding(self, state):
            with torch.inference_mode():
                output = self.model.forward_grounding(
                    # The pinned mask-disabled path pops backbone_fpn. Keep the
                    # cached image features intact for subsequent class prompts.
                    backbone_out=dict(state["backbone_out"]),
                    find_input=self.find_stage,
                    geometric_prompt=state["geometric_prompt"],
                    find_target=None,
                )
                boxes = output["pred_boxes"]
                scores = (
                    output["pred_logits"].sigmoid()
                    * output["presence_logit_dec"].sigmoid().unsqueeze(1)
                ).squeeze(-1)
                if (
                    boxes.ndim != 3
                    or boxes.shape[0] != 1
                    or boxes.shape[2] != 4
                    or boxes.shape[1] > 300
                    or scores.shape != boxes.shape[:2]
                    or not torch.isfinite(boxes).all().item()
                    or not torch.isfinite(scores).all().item()
                ):
                    raise ValueError("SAM returned malformed or nonfinite native tensors")
                selected = scores[0] > self.confidence_threshold
                selected_boxes = boxes[0][selected].float()
                centers, sizes = selected_boxes[:, :2], selected_boxes[:, 2:]
                xyxy = torch.cat((centers - sizes / 2, centers + sizes / 2), dim=-1)
                state["boxes"] = xyxy.cpu().tolist()
                state["scores"] = scores[0][selected].float().cpu().tolist()
                state["native_indices"] = selected.nonzero(as_tuple=False).flatten().cpu().tolist()
                return state

    return BoxProcessor


class _Runtime:
    def __init__(self, config, checkpoint_path):
        started = time.perf_counter()
        identity = _identity()
        self.prompts = _prompts(config.get("prompts"))
        self.allow_dynamic_prompts = config.get("allow_dynamic_prompts", False)
        if type(self.allow_dynamic_prompts) is not bool:
            raise ValueError("The SAM dynamic-prompt mode must be explicit")
        settings = config.get("settings", {})
        self.threshold = _threshold(settings.get("threshold"))
        if settings.get("device") != "cuda" or settings.get("precision") != "bfloat16":
            raise ValueError("The frozen SAM profile requires CUDA and bfloat16 autocast")
        path = Path(checkpoint_path)
        if (
            not path.is_absolute()
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != CHECKPOINT_BYTES
        ):
            raise ValueError("The explicit SAM checkpoint is missing or has the wrong size")
        before = path.stat()
        with path.open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != CHECKPOINT_SHA256:
                raise ValueError("The explicit SAM checkpoint has the wrong SHA-256")
        torch = importlib.import_module("torch")
        builder = importlib.import_module("sam3.model_builder")
        processor_module = importlib.import_module("sam3.model.sam3_image_processor")
        self.torch = torch
        self.model = builder.build_sam3_image_model(
            checkpoint_path=str(path),
            device="cuda",
            eval_mode=True,
            load_from_HF=False,
            enable_segmentation=False,
            enable_inst_interactivity=False,
            compile=False,
        )
        after = path.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ValueError("SAM checkpoint changed while loading")
        self._validate_tokens(self.prompts)
        # The upstream builder toggles TF32 globally on Ampere; this frozen profile
        # explicitly disables it while using the declared bfloat16 autocast path.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.processor = _box_processor(processor_module.Sam3Processor, torch)(
            self.model, resolution=1008, device="cuda", confidence_threshold=self.threshold
        )
        torch.cuda.synchronize()
        self.metadata = {
            "runtime_identity": identity,
            "model_load_ms": (time.perf_counter() - started) * 1000,
            "weight_sha256": CHECKPOINT_SHA256,
            "code_revision": CODE_REVISION,
            "precision": "bfloat16",
            "tf32": False,
            "segmentation_enabled": False,
            "masks_retained": False,
            "warmup": "none; the first measured image includes first-forward initialization",
            **({"allow_dynamic_prompts": True} if self.allow_dynamic_prompts else {}),
        }

    def _validate_tokens(self, prompts):
        text_encoder = self.model.backbone.language_backbone
        limit = min(MAX_PROMPT_TOKENS, text_encoder.context_length - 2)
        for prompt in prompts:
            if len(text_encoder.tokenizer.encode(prompt["text"])) > limit:
                raise ValueError(f"SAM class prompt exceeds {limit} tokenizer tokens")

    def set_prompts(self, prompts):
        if not self.allow_dynamic_prompts:
            raise ValueError("This SAM runtime keeps its original frozen prompts")
        checked = _prompts(prompts)
        if [p["class_id"] for p in checked] != [p["class_id"] for p in self.prompts]:
            raise ValueError("Dynamic SAM prompts must preserve the frozen class order")
        self._validate_tokens(checked)
        self.prompts = checked
        return {"prompts": checked}

    def predict(self, request):
        from PIL import Image

        started = time.perf_counter()
        if request.get("prompts") != self.prompts or request.get("threshold") != self.threshold:
            raise ValueError("Prediction prompts or threshold differ from the frozen SAM trial")
        encoded = request.get("png_base64")
        if not isinstance(encoded, str) or len(encoded) > (MAX_IMAGE_BYTES + 2) // 3 * 4:
            raise ValueError("SAM input PNG exceeds the transport bound")
        data = base64.b64decode(encoded, validate=True)
        if len(data) > MAX_IMAGE_BYTES or not data.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("SAM input must be a bounded clean PNG")
        if hashlib.sha256(data).hexdigest() != request.get("image_sha256"):
            raise ValueError("SAM input PNG checksum differs from its request")
        with Image.open(io.BytesIO(data)) as source:
            width, height = source.size
            if width * height > MAX_IMAGE_PIXELS or [width, height] != request.get("image_size"):
                raise ValueError("SAM image dimensions exceed the frozen input bounds")
            image = source.convert("RGB")
        raw = {
            "protocol": RAW_PROTOCOL,
            "image": {"width": width, "height": height},
            "coordinates": {
                "format": "xyxy",
                "space": "normalized",
                "image_size": [width, height],
                "to_original": {"scale": [width, height], "offset": [0, 0]},
            },
            "prompts": [],
            "complete": False,
            "metadata": {**self.metadata, "image_sha256": request["image_sha256"]},
        }
        try:
            with (
                self.torch.inference_mode(),
                self.torch.autocast("cuda", dtype=self.torch.bfloat16),
            ):
                state = self.processor.set_image(image)
                for prompt in self.prompts:
                    row = {**prompt, "boxes": [], "scores": [], "native_indices": [], "error": None}
                    try:
                        self.processor.reset_all_prompts(state)
                        state = self.processor.set_text_prompt(prompt["text"], state)
                        self.torch.cuda.synchronize()
                        row.update(
                            boxes=state["boxes"],
                            scores=state["scores"],
                            native_indices=state["native_indices"],
                        )
                    except Exception as exc:
                        row["error"] = (str(exc) or type(exc).__name__)[:4000]
                        raw["prompts"].append(row)
                        raise
                    raw["prompts"].append(row)
                raw["complete"] = True
        except Exception as exc:
            raw["metadata"]["error"] = (str(exc) or type(exc).__name__)[:4000]
        finally:
            image.close()
        raw["metadata"]["timing"] = {"elapsed_ms": (time.perf_counter() - started) * 1000}
        return raw


def main():
    if len(sys.argv) != 3 or sys.argv[1] != "--parent-pid":
        raise RuntimeError("SAM worker requires its owning process identity")
    _parent_guard(int(sys.argv[2]))
    _offline()
    # Redirect even C-level stdout writes so ML package chatter cannot corrupt IPC.
    channel = os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    runtime = None
    for _ in range(1000):
        line = sys.stdin.buffer.readline(MAX_REQUEST_BYTES + 1)
        if not line:
            break
        identifier = None
        try:
            if len(line) > MAX_REQUEST_BYTES or not line.endswith(b"\n"):
                raise ValueError("SAM request exceeds its JSON-lines bound")
            request = _parse(line)
            if not isinstance(request, dict) or type(request.get("id")) is not int:
                raise ValueError("SAM request must carry an integer request identity")
            identifier = request["id"]
            operation = request.get("op")
            if operation == "probe":
                value = {"ready": True, "status": "ready", "reason": None, "identity": _identity()}
            elif operation == "load" and runtime is None:
                runtime = _Runtime(request["config"], request["checkpoint_path"])
                value = {"metadata": runtime.metadata}
            elif operation == "predict" and runtime is not None:
                value = {"raw": runtime.predict(request)}
            elif operation == "set_prompts" and runtime is not None:
                value = runtime.set_prompts(request.get("prompts"))
            else:
                raise ValueError("SAM worker operation is invalid or the model is already loaded")
            response = _finite_json({"id": identifier, "ok": True, "value": value})
            if len(response) >= MAX_RESPONSE_BYTES:
                raise ValueError("SAM response exceeds the saved raw evidence limit")
        except Exception as exc:
            response = _finite_json(
                {"id": identifier, "ok": False, "error": (str(exc) or type(exc).__name__)[:4000]}
            )
        channel.write(response + b"\n")
        if identifier is None:
            break


if __name__ == "__main__":
    main()
