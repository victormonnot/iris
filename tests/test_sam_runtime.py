"""Real isolated IPC with simulated workers; no SAM installation, GPU or weights."""

import hashlib
import json
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from iris import sam_runtime as runtime
from iris import sam_runtime_worker as worker


def identity():
    return {
        "python": "3.12.10",
        "isolated": True,
        "packages": {**worker.PACKAGES, "torch": "2.10.0+cu128", "sam3": "0.1.0"},
        "code_revision": worker.CODE_REVISION,
        "tokenizer_sha256": "b" * 64,
        "cuda": {
            "available": True,
            "version": "12.8",
            "bfloat16": True,
            "device": "Simulated GPU",
            "capability": [8, 0],
        },
    }


CONFIG = {
    "prompts": [{"class_id": "helmet", "text": "protective helmet"}],
    "settings": {"threshold": 0.5, "device": "cuda", "precision": "bfloat16"},
}

FAKE_WORKER = """
import base64, io, json, os, signal, subprocess, sys, time
from PIL import Image
assert sys.flags.isolated == 1
loaded = 0
for line in sys.stdin.buffer:
    req = json.loads(line)
    if req['op'] == 'probe':
        value = {'ready': True, 'status': 'ready', 'reason': None, 'identity': IDENTITY}
    elif req['op'] == 'load':
        loaded += 1
        if MODE == 'load_hang':
            time.sleep(60)
        if MODE == 'noisy':
            sys.stderr.write('x' * 200000)
            sys.stderr.flush()
        meta = {'runtime_identity': IDENTITY, 'model_load_ms': 1,
                'received_config': req['config'], 'environment': dict(os.environ)}
        if MODE == 'tree':
            child = subprocess.Popen([sys.executable, '-c',
                'import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)'])
            meta['child_pid'] = child.pid
        value = {'metadata': meta}
    else:
        if MODE in ('hang', 'tree'):
            time.sleep(60)
        if MODE == 'eof':
            sys.exit(0)
        if MODE == 'nan':
            print('{' + '"id":%d,"ok":true,"value":{"bad":NaN}' % req['id'] + '}', flush=True)
            continue
        if MODE == 'duplicate':
            data = '"id":%d,"id":%d,"ok":true,"value":{}' % (req['id'],req['id'])
            print('{' + data + '}', flush=True)
            continue
        data = base64.b64decode(req['png_base64'])
        with Image.open(io.BytesIO(data)) as image:
            info = dict(image.info)
        raw = {'protocol': 'iris-sam3-native-boxes-v1', 'complete': MODE != 'partial',
               'metadata': {'loads': loaded, 'image_info': info, 'request': req},
               'prompts': [{'class_id':'helmet','text':'protective helmet',
                            'boxes':[[0,0,0.5,0.5]],'scores':[0.8],
                            'native_indices':[5],'error':None}]}
        if MODE == 'partial':
            raw['metadata']['error'] = 'Second prompt failed'
        if MODE == 'oversize':
            raw['metadata']['huge'] = 'x' * 9000000
        value = {'raw': raw}
    reply = {'id': req['id'] + (1 if MODE == 'identity' and req['op']=='predict' else 0),
             'ok': True, 'value': value}
    print(json.dumps(reply), flush=True)
"""


@pytest.fixture
def simulated(tmp_path, monkeypatch):
    checkpoint = tmp_path / "fixture.pt"
    checkpoint.write_bytes(b"Offline fixture; never passed to an actual model")
    monkeypatch.setenv("IRIS_SAM_PYTHON", sys.executable)
    runtime._STATUS_CACHE.clear()

    def prepare(mode="normal"):
        path = tmp_path / "fixture_worker.py"
        path.write_text(f"MODE={mode!r}\nIDENTITY={identity()!r}\n" + FAKE_WORKER)
        monkeypatch.setattr(runtime, "_WORKER_PATH", path)
        return checkpoint

    return prepare


def test_missing_runtime_never_spawns_a_process(monkeypatch):
    monkeypatch.delenv("IRIS_SAM_PYTHON", raising=False)
    monkeypatch.setattr(
        runtime.subprocess, "Popen", lambda *a, **k: pytest.fail("unexpected process")
    )
    assert runtime.runtime_status()["status"] == "missing_runtime"
    monkeypatch.setenv("IRIS_SAM_PYTHON", "python")
    assert runtime.runtime_status(force=True)["ready"] is False


def test_probe_is_cached_bounded_and_never_supplies_checkpoint(simulated):
    simulated()
    status = runtime.runtime_status(force=True)
    assert status["identity"] == identity() and status["ready"] is True
    status["identity"]["packages"]["torch"] = "mutated"
    assert runtime.runtime_status()["identity"]["packages"]["torch"] == "2.10.0+cu128"


def test_one_load_per_trial_clean_pixels_and_allowlisted_environment(simulated, monkeypatch):
    checkpoint = simulated()
    for key in ("OPENAI_API_KEY", "HF_TOKEN", "HTTPS_PROXY", "PYTHONPATH", "AWS_SECRET_ACCESS_KEY"):
        monkeypatch.setenv(key, "DO-NOT-SEND")
    config = deepcopy(CONFIG)
    config.update(reference={"boxes": ["HUMAN-REFERENCE"]}, notes="PRIVATE-NOTES")
    config["settings"]["reference"] = "EXCLUDED"
    with runtime.SamRuntime(config, checkpoint) as model:
        process = model.process
        temporary = process.directory.name
        assert model.metadata["received_config"] == CONFIG
        environment = model.metadata["environment"]
        assert "DO-NOT-SEND" not in json.dumps(environment)
        assert environment["HF_HUB_OFFLINE"] == "1"
        with Image.new("RGB", (20, 12), "blue") as image:
            image.info["comment"] = "PRIVATE-NOTES"
            for _ in range(2):
                raw = model.predict(image, CONFIG["prompts"], 0.5)
                assert raw["metadata"]["loads"] == 1
                assert raw["metadata"]["image_info"] == {}
                assert set(raw["metadata"]["request"]) == {
                    "op",
                    "id",
                    "prompts",
                    "threshold",
                    "png_base64",
                    "image_sha256",
                    "image_size",
                }
    assert not Path(temporary).exists() and process.process is None
    model.close()


@pytest.mark.parametrize("mode", ["eof", "nan", "duplicate", "identity", "oversize"])
def test_corrupt_or_truncated_ipc_cannot_become_empty_predictions(simulated, mode):
    with runtime.SamRuntime(CONFIG, simulated(mode)) as model, Image.new("RGB", (2, 2)) as image:
        process = model.process
        with pytest.raises(runtime.SamRuntimeError):
            model.predict(image, CONFIG["prompts"], 0.5)
        assert process.process is None


def test_partial_native_output_is_retained_and_runtime_stops(simulated):
    model = runtime.SamRuntime(CONFIG, simulated("partial"))
    with Image.new("RGB", (2, 2)) as image:
        with pytest.raises(runtime.SamRuntimeError, match="Second prompt") as failure:
            model.predict(image, CONFIG["prompts"], 0.5)
    assert failure.value.raw_response["prompts"][0]["native_indices"] == [5]
    assert failure.value.raw_response["complete"] is False
    assert model.process is None


def test_runtime_stderr_is_drained_without_blocking_or_unbounded_retention(simulated):
    with runtime.SamRuntime(CONFIG, simulated("noisy")) as model:
        assert len(model.process.stderr) <= 16384
        assert model.metadata["model_load_ms"] == 1


@pytest.mark.parametrize("when", ["load", "predict"])
def test_cancellation_kills_runtime_during_loading_or_inference(simulated, when):
    path = simulated("load_hang" if when == "load" else "hang")
    started = time.monotonic()

    def cancelled():
        return time.monotonic() - started > 0.15

    if when == "load":
        with pytest.raises(runtime.SamRuntimeCancelled):
            runtime.SamRuntime(CONFIG, path, cancelled=cancelled)
    else:
        with runtime.SamRuntime(CONFIG, path) as model, Image.new("RGB", (2, 2)) as image:
            with pytest.raises(runtime.SamRuntimeCancelled):
                model.predict(image, CONFIG["prompts"], 0.5, cancelled)
            assert model.process.process is None
    assert time.monotonic() - started < 3


def _not_running(pid):
    path = Path(f"/proc/{pid}/status")
    try:
        return "\nState:\tZ" in path.read_text()
    except FileNotFoundError:
        return True


def test_deadline_terminates_the_entire_runtime_process_group(simulated, monkeypatch):
    monkeypatch.setattr(runtime, "_TIMEOUT", 0.4)
    with runtime.SamRuntime(CONFIG, simulated("tree")) as model, Image.new("RGB", (2, 2)) as image:
        child = model.metadata["child_pid"]
        with pytest.raises(runtime.SamRuntimeError, match="deadline"):
            model.predict(image, CONFIG["prompts"], 0.5)
        deadline = time.monotonic() + 2
        while not _not_running(child) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert _not_running(child)


def test_parent_death_stops_standalone_worker_without_importing_models(tmp_path):
    launcher = tmp_path / "launcher.py"
    launcher.write_text(
        "import os,subprocess,sys,time\n"
        "p=subprocess.Popen([sys.executable,'-I',sys.argv[1],'--parent-pid',str(os.getpid())],"
        "stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)\n"
        "print(p.pid,flush=True)\ntime.sleep(60)\n"
    )
    parent = subprocess.Popen(
        [sys.executable, "-I", str(launcher), worker.__file__], stdout=subprocess.PIPE, text=True
    )
    try:
        child = int(parent.stdout.readline())
        parent.kill()
        parent.wait(timeout=2)
        deadline = time.monotonic() + 2
        while not _not_running(child) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert _not_running(child)
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=2)
        parent.stdout.close()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(python="3.11.9"),
        lambda value: value.update(python="3.13.0"),
        lambda value: value.update(isolated=1),
        lambda value: value.update(code_revision="a" * 40),
        lambda value: value.update(tokenizer_sha256="bogus"),
        lambda value: value["packages"].update(numpy="2.0.0"),
        lambda value: value["cuda"].update(version="12.5"),
        lambda value: value["cuda"].update(bfloat16=False),
        lambda value: value["cuda"].update(capability=[True, 0]),
    ],
)
def test_runtime_identity_validation_is_pure_and_strict(monkeypatch, mutation):
    monkeypatch.setattr(worker.importlib, "import_module", lambda *a: pytest.fail("ML import"))
    valid = identity()
    assert runtime.validate_runtime_identity(valid) is valid
    mutation(valid)
    with pytest.raises(ValueError):
        runtime.validate_runtime_identity(valid)


@pytest.mark.parametrize("text", ["", " leading", "trailing ", "line\nbreak", "x" * 121])
def test_worker_prompt_boundary_matches_the_frozen_provider(text):
    with pytest.raises(ValueError, match="prompts"):
        worker._prompts([{"class_id": "helmet", "text": text}])


@pytest.fixture
def package_metadata(tmp_path, monkeypatch):
    tokenizer = tmp_path / "bpe.txt.gz"
    tokenizer.write_bytes(b"Offline tokenizer fixture")
    direct = {
        "url": "https://github.com/facebookresearch/sam3.git",
        "vcs_info": {"vcs": "git", "commit_id": worker.CODE_REVISION},
    }
    distribution = SimpleNamespace(
        version="0.1.0",
        read_text=lambda name: json.dumps(direct),
        locate_file=lambda path: tokenizer,
    )
    # The separate SAM interpreter stays on 3.12 regardless of the app's Python version.
    monkeypatch.setattr(
        worker,
        "sys",
        SimpleNamespace(version_info=(3, 12, 10), prefix="/isolated-fixture", base_prefix="/base"),
    )
    monkeypatch.setattr(worker, "platform", SimpleNamespace(python_version=lambda: "3.12.10"))
    monkeypatch.setattr(worker.importlib.metadata, "version", lambda name: worker.PACKAGES[name])
    monkeypatch.setattr(worker.importlib.metadata, "distribution", lambda name: distribution)
    return direct, tokenizer


def test_installed_code_provenance_and_tokenizer_are_checked_without_weights(package_metadata):
    direct, tokenizer = package_metadata
    actual = worker._package_identity()
    assert actual["python"] == "3.12.10"
    assert actual["tokenizer_sha256"] == hashlib.sha256(tokenizer.read_bytes()).hexdigest()
    direct["dir_info"] = {"editable": True}
    with pytest.raises(RuntimeError, match="noneditable"):
        worker._package_identity()
    direct.pop("dir_info")
    direct["vcs_info"]["commit_id"] = "a" * 40
    with pytest.raises(RuntimeError, match="pinned"):
        worker._package_identity()


def test_prepare_image_preflights_bounds_without_runtime_or_credentials(monkeypatch):
    monkeypatch.setattr(runtime, "_Process", lambda *a: pytest.fail("unexpected child process"))
    oversized = SimpleNamespace(width=4097, height=4096)
    with pytest.raises(runtime.SamRuntimeError, match="pixel"):
        runtime.prepare_image(oversized)
    with Image.new("RGB", (10, 6), "red") as image:
        prepared = runtime.prepare_image(image)
        assert prepared["image_size"] == [10, 6]
        monkeypatch.setattr(worker, "MAX_IMAGE_BYTES", 1)
        with pytest.raises(runtime.SamRuntimeError, match="16 MiB"):
            runtime.prepare_image(image)


def test_probe_catches_incomplete_optional_dependencies_without_model_construction(monkeypatch):
    saved = identity()
    saved.pop("cuda")
    monkeypatch.setattr(worker, "_package_identity", lambda: deepcopy(saved))
    cuda = SimpleNamespace(
        is_available=lambda: True,
        is_bf16_supported=lambda: True,
        get_device_name=lambda index: "Simulated GPU",
        get_device_capability=lambda index: (8, 0),
    )
    imports = []

    def imported(name):
        imports.append(name)
        if name == "torch":
            return SimpleNamespace(cuda=cuda, version=SimpleNamespace(cuda="12.8"))
        if name == "sam3.model_builder":
            raise ModuleNotFoundError("No module named 'einops'")
        return SimpleNamespace()

    monkeypatch.setattr(worker.importlib, "import_module", imported)
    with pytest.raises(ModuleNotFoundError, match="einops"):
        worker._identity()
    assert imports == ["torch", "torchvision", "sam3.model_builder"]
    cuda.is_available = lambda: False
    imports.clear()
    with pytest.raises(RuntimeError, match="CUDA GPU"):
        worker._identity()
    assert imports == ["torch", "torchvision"]


@pytest.fixture
def simulated_model(tmp_path, monkeypatch):
    from contextlib import nullcontext

    checkpoint = tmp_path / "sam3.pt"
    checkpoint.write_bytes(b"offline model fixture")
    monkeypatch.setattr(worker, "CHECKPOINT_BYTES", checkpoint.stat().st_size)
    monkeypatch.setattr(
        worker, "CHECKPOINT_SHA256", hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    )
    monkeypatch.setattr(worker, "_identity", identity)
    calls = []
    tokenizer = SimpleNamespace(encode=lambda text: text.split())
    model = SimpleNamespace(
        backbone=SimpleNamespace(
            language_backbone=SimpleNamespace(
                tokenizer=tokenizer,
                context_length=32,
            )
        ),
    )

    def build(**options):
        calls.append(options)
        return model

    class Processor:
        def __init__(self, model, **kwargs):
            self.model = model

        def set_image(self, image):
            return {"image_size": image.size}

        def reset_all_prompts(self, state):
            pass

        def set_text_prompt(self, text, state):
            if text == "fail":
                raise RuntimeError("Fixture second prompt error")
            return {**state, "boxes": [[0, 0, 1, 1]], "scores": [0.9], "native_indices": [12]}

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(synchronize=lambda: None),
        backends=SimpleNamespace(
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
            cudnn=SimpleNamespace(allow_tf32=True),
        ),
        inference_mode=nullcontext,
        autocast=lambda *a, **k: nullcontext(),
        bfloat16="fixture-bfloat16",
    )
    modules = {
        "torch": fake_torch,
        "sam3.model_builder": SimpleNamespace(build_sam3_image_model=build),
        "sam3.model.sam3_image_processor": SimpleNamespace(Sam3Processor=Processor),
    }
    monkeypatch.setattr(worker.importlib, "import_module", lambda name: modules[name])
    return checkpoint, calls, tokenizer


def test_worker_loads_only_explicit_verified_weights_with_masks_disabled(simulated_model):
    checkpoint, calls, _ = simulated_model
    model = worker._Runtime(CONFIG, str(checkpoint))
    assert len(calls) == 1
    assert calls[0] == {
        "checkpoint_path": str(checkpoint),
        "device": "cuda",
        "eval_mode": True,
        "load_from_HF": False,
        "enable_segmentation": False,
        "enable_inst_interactivity": False,
        "compile": False,
    }
    assert model.metadata["runtime_identity"] == identity()
    assert model.metadata["model_load_ms"] >= 0
    assert model.metadata["masks_retained"] is False


def test_worker_refuses_changed_weights_and_truncated_prompts(simulated_model):
    checkpoint, calls, tokenizer = simulated_model
    original = checkpoint.read_bytes()
    checkpoint.write_bytes(b"x" * len(original))
    with pytest.raises(ValueError, match="SHA-256"):
        worker._Runtime(CONFIG, str(checkpoint))
    assert not calls
    checkpoint.write_bytes(original)
    tokenizer.encode = lambda text: list(range(31))
    with pytest.raises(ValueError, match="30 tokenizer tokens"):
        worker._Runtime(CONFIG, str(checkpoint))


def test_worker_partial_error_preserves_previous_native_prompt_and_never_claims_complete(
    simulated_model,
):
    checkpoint, _, _ = simulated_model
    config = deepcopy(CONFIG)
    config["prompts"].append({"class_id": "person", "text": "fail"})
    model = worker._Runtime(config, str(checkpoint))
    with Image.new("RGB", (6, 4)) as image:
        raw = model.predict(
            {
                **runtime.prepare_image(image),
                "prompts": config["prompts"],
                "threshold": 0.5,
            }
        )
    assert raw["complete"] is False
    assert raw["prompts"][0]["native_indices"] == [12]
    assert raw["prompts"][1]["error"] == "Fixture second prompt error"
    assert raw["coordinates"]["to_original"] == {"scale": [6, 4], "offset": [0, 0]}
    assert raw["metadata"]["timing"]["elapsed_ms"] >= 0


class Tensor:
    """Only the tensor operations used by our adapter; no ML runtime import."""

    def __init__(self, value):
        self.data = np.asarray(value)

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def shape(self):
        return self.data.shape

    def __getitem__(self, index):
        return Tensor(self.data[index.data if isinstance(index, Tensor) else index])

    def __mul__(self, other):
        return Tensor(self.data * other.data)

    def __truediv__(self, other):
        return Tensor(self.data / other)

    def __sub__(self, other):
        return Tensor(self.data - other.data)

    def __add__(self, other):
        return Tensor(self.data + other.data)

    def __gt__(self, other):
        return Tensor(self.data > other)

    def sigmoid(self):
        return Tensor(1 / (1 + np.exp(-self.data)))

    def unsqueeze(self, axis):
        return Tensor(np.expand_dims(self.data, axis))

    def squeeze(self, axis):
        return Tensor(self.data.squeeze(axis))

    def float(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.data.tolist()

    def all(self):
        return Tensor(self.data.all())

    def item(self):
        return self.data.item()

    def nonzero(self, **kwargs):
        return Tensor(np.argwhere(self.data))

    def flatten(self):
        return Tensor(self.data.flatten())


def test_boxes_only_grounding_preserves_native_indices_strict_scores_and_unclipped_boxes():
    from contextlib import nullcontext

    fake_torch = SimpleNamespace(
        inference_mode=nullcontext,
        isfinite=lambda item: Tensor(np.isfinite(item.data)),
        cat=lambda values, dim: Tensor(np.concatenate([item.data for item in values], axis=dim)),
    )
    outputs = {
        "pred_boxes": Tensor([[[0.5, 0.5, 0.2, 0.2], [0, 0.5, 1, 1]]]),
        "pred_logits": Tensor([[[0], [np.log(3)]]]),
        "presence_logit_dec": Tensor([[0]]),
    }
    processor = worker._box_processor(object, fake_torch)()

    def forward_grounding(*, backbone_out, **kwargs):
        assert backbone_out.pop("backbone_fpn") == "cached-image-features"
        return outputs

    processor.model = SimpleNamespace(forward_grounding=forward_grounding)
    processor.find_stage = None
    processor.confidence_threshold = 0.25
    state = {"backbone_out": {"backbone_fpn": "cached-image-features"}, "geometric_prompt": None}
    result = processor._forward_grounding(state)
    assert result["native_indices"] == [1]
    assert result["boxes"] == [[-0.5, 0, 0.5, 1]]
    assert result["scores"] == [0.375]
    assert "masks" not in result
    assert state["backbone_out"]["backbone_fpn"] == "cached-image-features"
    processor.confidence_threshold = 1
    assert processor._forward_grounding(state)["boxes"] == []
    outputs["pred_boxes"] = Tensor([[[float("nan"), 0, 1, 1], [0, 0, 1, 1]]])
    with pytest.raises(ValueError, match="nonfinite"):
        processor._forward_grounding(state)
