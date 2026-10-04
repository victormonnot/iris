"""Bounded, offline IPC for a separately installed SAM 3 CUDA environment."""

from __future__ import annotations

import base64
import hashlib
import io
import os
import selectors
import signal
import subprocess
import tempfile
import threading
import time
from copy import deepcopy
from pathlib import Path

from iris import sam_runtime_worker as wire

validate_runtime_identity = wire.validate_runtime_identity

_WORKER_PATH = Path(__file__).with_name("sam_runtime_worker.py")
_PROBE_TIMEOUT = 30
_TIMEOUT = 300
_CACHE_SECONDS = 30
_STATUS_CACHE = {}
_STATUS_LOCK = threading.Lock()


class SamRuntimeError(RuntimeError):
    def __init__(self, message, *, raw_response=None):
        super().__init__(message)
        self.raw_response = raw_response


class SamRuntimeCancelled(SamRuntimeError):
    """The process group was stopped; no further image is attempted."""


def _python():
    value = os.environ.get("IRIS_SAM_PYTHON", "")
    if not value:
        raise SamRuntimeError(
            "Set IRIS_SAM_PYTHON to a separately installed SAM Python environment"
        )
    path = Path(value)
    if not path.is_absolute() or not path.is_file() or not os.access(path, os.X_OK):
        raise SamRuntimeError("IRIS_SAM_PYTHON must be an absolute executable Python path")
    # Do not resolve the interpreter symlink: that would discard venv selection.
    return str(path)


def _environment(directory, executable):
    return {
        "PATH": str(Path(executable).parent) + os.pathsep + os.defpath,
        "LANG": "C.UTF-8",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HOME": str(Path(directory) / "hf"),
        "TORCH_HOME": str(Path(directory) / "torch"),
        "XDG_CACHE_HOME": str(Path(directory) / "cache"),
        "TMPDIR": directory,
    }


class _Process:
    def __init__(self, executable):
        self.directory = tempfile.TemporaryDirectory(prefix="iris-sam-runtime-")
        self.process = None
        self.sequence = 0
        self.stderr = bytearray()
        self.lock = threading.Lock()
        try:
            self.process = subprocess.Popen(
                [executable, "-I", str(_WORKER_PATH), "--parent-pid", str(os.getpid())],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.directory.name,
                env=_environment(self.directory.name, executable),
                start_new_session=True,
                bufsize=0,
            )
            for stream in (self.process.stdin, self.process.stdout):
                os.set_blocking(stream.fileno(), False)
            self.drainer = threading.Thread(target=self._drain, daemon=True)
            self.drainer.start()
        except Exception:
            self.close()
            raise

    def _drain(self):
        try:
            while data := self.process.stderr.read(4096):
                self.stderr.extend(data)
                del self.stderr[:-16384]
        except (OSError, ValueError):
            pass

    def close(self):
        process = self.process
        if process is not None:
            # Kill the group even if its leader exited: a runtime subprocess must
            # not outlive a cancelled or failed trial.
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(process.pid, sig)
                except ProcessLookupError:
                    break
                if sig == signal.SIGTERM:
                    try:
                        process.wait(timeout=0.25)
                    except subprocess.TimeoutExpired:
                        pass
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1)
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream:
                    stream.close()
            if hasattr(self, "drainer"):
                self.drainer.join(timeout=0.25)
            self.process = None
        self.directory.cleanup()

    def exchange(self, payload, *, timeout, cancelled=lambda: False):
        with self.lock:
            if self.process is None:
                raise SamRuntimeError("The isolated SAM process has already stopped")
            self.sequence += 1
            request = wire._finite_json({**payload, "id": self.sequence}) + b"\n"
            if len(request) > wire.MAX_REQUEST_BYTES:
                raise SamRuntimeError("The isolated SAM request exceeds its IPC limit")
            received = bytearray()
            sent = 0
            deadline = time.monotonic() + timeout
            try:
                with selectors.DefaultSelector() as selector:
                    selector.register(self.process.stdin, selectors.EVENT_WRITE)
                    selector.register(self.process.stdout, selectors.EVENT_READ)
                    while True:
                        if cancelled():
                            raise SamRuntimeCancelled(
                                "SAM trial was cancelled; local runtime stopped"
                            )
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise SamRuntimeError("The isolated SAM runtime exceeded its deadline")
                        for key, _ in selector.select(min(0.05, remaining)):
                            if key.fileobj is self.process.stdin:
                                count = os.write(key.fd, request[sent : sent + 65536])
                                sent += count
                                if sent == len(request):
                                    selector.unregister(self.process.stdin)
                            else:
                                part = os.read(key.fd, 65536)
                                if not part:
                                    raise SamRuntimeError(
                                        "The isolated SAM runtime exited without a complete reply"
                                    )
                                received.extend(part)
                                if len(received) > wire.MAX_RESPONSE_BYTES:
                                    raise SamRuntimeError(
                                        "The isolated SAM reply exceeds its IPC limit"
                                    )
                                if b"\n" in received:
                                    line, extra = received.split(b"\n", 1)
                                    if extra:
                                        raise SamRuntimeError(
                                            "Unexpected extra data from the isolated SAM runtime"
                                        )
                                    response = wire._parse(line)
                                    if (
                                        not isinstance(response, dict)
                                        or type(response.get("id")) is not int
                                        or response["id"] != self.sequence
                                        or type(response.get("ok")) is not bool
                                    ):
                                        raise SamRuntimeError(
                                            "The isolated SAM reply identity is invalid"
                                        )
                                    if not response["ok"]:
                                        raise SamRuntimeError(
                                            str(response.get("error") or "SAM runtime failed")[
                                                :4000
                                            ]
                                        )
                                    value = response.get("value")
                                    if not isinstance(value, dict):
                                        raise SamRuntimeError(
                                            "The isolated SAM reply payload is invalid"
                                        )
                                    return value
            except Exception as exc:
                self.close()
                if isinstance(exc, SamRuntimeError):
                    raise
                raise SamRuntimeError(
                    f"Invalid isolated SAM transport: {type(exc).__name__}"
                ) from exc


def runtime_status(config=None, *, force=False):
    """Probe packages, pinned code provenance and CUDA, never model weights."""
    try:
        executable = _python()
    except SamRuntimeError as exc:
        return {"ready": False, "status": "missing_runtime", "reason": str(exc), "identity": None}
    signature = Path(executable).stat()
    key = (executable, signature.st_mtime_ns, signature.st_ino)
    with _STATUS_LOCK:
        previous = _STATUS_CACHE.get(key)
        if not force and previous and time.monotonic() - previous[0] < _CACHE_SECONDS:
            return deepcopy(previous[1])
        process = None
        try:
            process = _Process(executable)
            result = process.exchange({"op": "probe"}, timeout=_PROBE_TIMEOUT)
            if result.get("ready") is not True or not isinstance(result.get("identity"), dict):
                raise SamRuntimeError(
                    "The isolated SAM runtime returned invalid readiness evidence"
                )
            validate_runtime_identity(result["identity"])
        except (SamRuntimeError, OSError, ValueError) as exc:
            result = {
                "ready": False,
                "status": "incompatible_runtime",
                "reason": str(exc),
                "identity": None,
            }
        finally:
            if process:
                process.close()
        _STATUS_CACHE.clear()
        _STATUS_CACHE[key] = (time.monotonic(), deepcopy(result))
        return result


def prepare_image(image):
    """Bound and encode clean local pixels before loading any optional runtime."""
    from PIL import Image

    if image.width * image.height > wire.MAX_IMAGE_PIXELS:
        raise SamRuntimeError("SAM input exceeds the 16777216-pixel bound")
    buffer = io.BytesIO()
    with image.convert("RGB") as rgb, Image.frombytes("RGB", rgb.size, rgb.tobytes()) as clean:
        clean.save(buffer, format="PNG")
    data = buffer.getvalue()
    if len(data) > wire.MAX_IMAGE_BYTES:
        raise SamRuntimeError("SAM clean input PNG exceeds the 16 MiB bound")
    return {
        "png_base64": base64.b64encode(data).decode("ascii"),
        "image_sha256": hashlib.sha256(data).hexdigest(),
        "image_size": [image.width, image.height],
    }


class SamRuntime:
    """One explicitly loaded local model per trial, with no reference-data access."""

    def __init__(self, config, checkpoint_path, *, cancelled=lambda: False):
        self.process = None
        self.config = deepcopy(config)
        self.prompts = deepcopy(wire._prompts(config.get("prompts")))
        self.threshold = wire._threshold(config.get("settings", {}).get("threshold"))
        if cancelled():
            raise SamRuntimeCancelled("SAM trial was cancelled before loading")
        path = Path(checkpoint_path)
        if not path.is_absolute() or path.is_symlink() or not path.is_file():
            raise SamRuntimeError("The explicit SAM checkpoint must be a regular local file")
        self.process = _Process(_python())
        try:
            loaded = self.process.exchange(
                {
                    "op": "load",
                    # Deliberate allowlist: no Store, taxonomy reference boxes,
                    # annotations, filenames, benchmark paths or secrets.
                    "config": {
                        "prompts": self.prompts,
                        "settings": {
                            key: config["settings"].get(key)
                            for key in ("threshold", "device", "precision")
                        },
                    },
                    "checkpoint_path": str(path),
                },
                timeout=_TIMEOUT,
                cancelled=cancelled,
            )
            if not isinstance(loaded.get("metadata"), dict):
                raise SamRuntimeError("SAM model loading returned no provenance")
            self.metadata = loaded["metadata"]
            validate_runtime_identity(self.metadata.get("runtime_identity"))
        except Exception:
            self.close()
            raise

    def predict(self, image, class_prompts, threshold, cancelled=lambda: False):
        if self.process is None:
            raise SamRuntimeError("The SAM runtime is closed")
        if class_prompts != self.prompts or threshold != self.threshold:
            raise SamRuntimeError("SAM prompts or threshold changed after loading")
        if cancelled():
            self.close()
            raise SamRuntimeCancelled("SAM trial was cancelled before the next image")
        prepared = prepare_image(image)
        value = self.process.exchange(
            {
                "op": "predict",
                "prompts": deepcopy(class_prompts),
                "threshold": threshold,
                **prepared,
            },
            timeout=_TIMEOUT,
            cancelled=cancelled,
        )
        raw = value.get("raw")
        if not isinstance(raw, dict) or raw.get("protocol") != wire.RAW_PROTOCOL:
            self.close()
            raise SamRuntimeError("SAM returned an invalid raw output protocol", raw_response=raw)
        if raw.get("complete") is not True:
            metadata = raw.get("metadata")
            error = (
                metadata.get("error") if isinstance(metadata, dict) else None
            ) or "SAM returned only a partial image output"
            self.close()
            raise SamRuntimeError(str(error), raw_response=raw)
        return raw

    def close(self):
        if self.process is not None:
            self.process.close()
            self.process = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
