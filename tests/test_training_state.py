"""Continuation protocol checks using tiny tensors, never a detector or real data."""

import hashlib
import json
import platform
import random
import zipfile
from copy import deepcopy
from types import SimpleNamespace

import pytest

from iris import training_state
from iris.training import _HeadTrainer


@pytest.fixture
def tiny():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    previous_rng = torch.get_rng_state()
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.frozen = torch.nn.Linear(2, 3)
            self.dropout = torch.nn.Dropout(0.4)
            self.head = torch.nn.Linear(3, 1)
            self.register_buffer("fixed_offset", torch.tensor([0.1, 0.2, 0.3]))

        def forward(self, inputs):
            features = self.frozen(inputs) + self.fixed_offset
            return self.head(self.dropout(features) + torch.rand_like(features) * 0.01)

    def make(*, momentum=0.9):
        torch.manual_seed(713)
        model = TinyModel().train()
        model.frozen.eval()
        selected = dict(model.head.named_parameters(prefix="head"))
        frozen = dict(model.frozen.named_parameters(prefix="frozen"))
        for parameter in frozen.values():
            parameter.requires_grad_(False)
        return SimpleNamespace(
            torch=torch,
            model=model,
            scope={"trainable_modules": ["head"]},
            selected_parameters=selected,
            frozen_parameters=frozen,
            parameters=list(selected.values()),
            optimizer=torch.optim.SGD(
                selected.values(), lr=0.025, momentum=momentum, weight_decay=0.0005
            ),
            initial={name: value.detach().clone() for name, value in selected.items()},
            frozen_initial={
                name: hashlib.sha256(value.detach().numpy().tobytes()).hexdigest()
                for name, value in frozen.items()
            },
            initial_buffers={name: value.clone() for name, value in model.named_buffers()},
            gradient_modules=set(),
        )

    yield make, torch
    torch.set_rng_state(previous_rng)
    torch.use_deterministic_algorithms(previous_determinism)


def step(trainer, value):
    torch = trainer.torch
    trainer.optimizer.zero_grad(set_to_none=True)
    inputs = torch.tensor([[value / 10, 0.5], [0.75, value / 20]], dtype=torch.float32)
    loss = (trainer.model(inputs) - 0.2).square().mean()
    loss.backward()
    trainer.optimizer.step()
    trainer.gradient_modules.add("head")
    return loss.item()


def inputs(trainer, count=2):
    return (
        {
            "protocol": training_state.PROTOCOL,
            "config_sha256": "a" * 64,
            "history_sha256": "b" * 64,
            "parent_weight_sha256": "c" * 64,
            "step": count,
            "runtime": training_state.runtime_identity(trainer),
        },
        {
            "random_state": json.loads(json.dumps(random.Random(27).getstate())),
            "remaining_order": [2, 0],
        },
    )


def saved_state(tiny, tmp_path):
    make, torch = tiny
    trainer = make()
    step(trainer, 1)
    step(trainer, 2)
    binding, sampler = inputs(trainer)
    path = tmp_path / "continuation.pth"
    training_state.write_state(trainer, path, binding=binding, sampler=sampler)
    return trainer, path, binding, sampler, torch.load(path, weights_only=True)


def assert_tree_equal(torch, actual, expected):
    if type(actual) is torch.Tensor:
        assert torch.equal(actual, expected)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_tree_equal(torch, actual[key], expected[key])
    elif isinstance(actual, list | tuple):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            assert_tree_equal(torch, left, right)
    else:
        assert actual == expected


@pytest.mark.parametrize("momentum", [0.0, 0.9])
def test_resume_matches_uninterrupted_rng_momentum_weights_and_final_verification(
    tiny, tmp_path, momentum
):
    make, torch = tiny
    uninterrupted = make(momentum=momentum)
    expected_losses = [step(uninterrupted, value) for value in range(1, 8)]
    expected_rng = torch.get_rng_state().clone()

    partial = make(momentum=momentum)
    actual_losses = [step(partial, value) for value in range(1, 4)]
    binding, sampler = inputs(partial, count=3)
    path = tmp_path / "state.pth"
    training_state.write_state(partial, path, binding=binding, sampler=sampler)
    resumed = make(momentum=momentum)
    originals = (resumed.initial, resumed.frozen_initial, resumed.initial_buffers)
    training_state.load_state(resumed, path, binding=binding, sampler=sampler)
    assert originals[0] is resumed.initial
    assert originals[1] is resumed.frozen_initial
    assert originals[2] is resumed.initial_buffers
    assert resumed.gradient_modules == {"head"}
    actual_losses.extend(step(resumed, value) for value in range(4, 8))
    assert actual_losses == expected_losses
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert_tree_equal(torch, resumed.model.state_dict(), uninterrupted.model.state_dict())
    assert_tree_equal(torch, resumed.optimizer.state_dict(), uninterrupted.optimizer.state_dict())
    verification = _HeadTrainer.write_checkpoint(resumed, tmp_path / "final.pth")
    assert verification["trainable_weights_changed"] is True
    assert verification["changed_trainable_modules"] == ["head"]
    assert verification["frozen_parameters_unchanged"] is True
    assert verification["model_buffers_unchanged"] is True


def test_save_preserves_modes_rng_and_original_verification_baselines(tiny, tmp_path):
    make, torch = tiny
    trainer = make()
    step(trainer, 1)
    trainer.model.dropout.eval()
    modes = {name: module.training for name, module in trainer.model.named_modules()}
    rng = torch.get_rng_state().clone()
    originals = deepcopy((trainer.initial, trainer.frozen_initial, trainer.initial_buffers))
    binding, sampler = inputs(trainer, count=1)
    path = tmp_path / "state.pth"
    training_state.write_state(trainer, path, binding=binding, sampler=sampler)
    assert {name: module.training for name, module in trainer.model.named_modules()} == modes
    assert torch.equal(torch.get_rng_state(), rng)
    assert_tree_equal(
        torch, (trainer.initial, trainer.frozen_initial, trainer.initial_buffers), originals
    )
    saved = torch.load(path, weights_only=True)
    binding["config_sha256"] = "changed"
    sampler["remaining_order"].clear()
    assert saved["binding"]["config_sha256"] == "a" * 64
    assert saved["sampler"]["remaining_order"] == [2, 0]
    restored = make()
    training_state.load_state(restored, path, binding=saved["binding"], sampler=saved["sampler"])
    assert {name: module.training for name, module in restored.model.named_modules()} == modes
    assert torch.equal(torch.get_rng_state(), rng)


CORRUPTIONS = [
    "protocol",
    "extra_field",
    "binding",
    "binding_bool_step",
    "sampler_order",
    "sampler_random",
    "model_missing_key",
    "model_extra_key",
    "model_shape",
    "model_dtype",
    "model_nonfinite",
    "frozen_weight",
    "frozen_buffer",
    "optimizer_names",
    "optimizer_lr",
    "optimizer_contract_lr",
    "optimizer_missing_momentum",
    "optimizer_extra_state",
    "optimizer_parameter_order",
    "momentum_shape",
    "momentum_dtype",
    "momentum_nonfinite",
    "momentum_extra_field",
    "module_mode",
    "module_missing",
    "gradient_unknown",
    "gradient_duplicate",
    "rng_shape",
    "rng_dtype",
    "rng_invalid",
]


def corrupt(payload, kind, torch):
    if kind == "protocol":
        payload["protocol"] = "future-v99"
    elif kind == "extra_field":
        payload["unbound"] = True
    elif kind == "binding":
        payload["binding"]["history_sha256"] = "d" * 64
    elif kind == "binding_bool_step":
        payload["binding"]["step"] = True
    elif kind == "sampler_order":
        payload["sampler"]["remaining_order"].reverse()
    elif kind == "sampler_random":
        payload["sampler"]["random_state"][1][0] += 1
    elif kind == "model_missing_key":
        payload["model_state"].pop("head.bias")
    elif kind == "model_extra_key":
        payload["model_state"]["new.weight"] = torch.zeros(1)
    elif kind == "model_shape":
        payload["model_state"]["head.bias"] = torch.zeros(2)
    elif kind == "model_dtype":
        payload["model_state"]["head.bias"] = payload["model_state"]["head.bias"].double()
    elif kind == "model_nonfinite":
        payload["model_state"]["head.bias"].fill_(float("nan"))
    elif kind == "frozen_weight":
        payload["model_state"]["frozen.bias"].add_(1)
    elif kind == "frozen_buffer":
        payload["model_state"]["fixed_offset"].add_(1)
    elif kind == "optimizer_names":
        payload["optimizer_contract"][0]["parameter_names"].reverse()
    elif kind == "optimizer_lr":
        payload["optimizer_state"]["param_groups"][0]["lr"] *= 2
    elif kind == "optimizer_contract_lr":
        payload["optimizer_contract"][0]["lr"] *= 2
    elif kind == "optimizer_missing_momentum":
        payload["optimizer_state"]["state"].pop(0)
    elif kind == "optimizer_extra_state":
        payload["optimizer_state"]["state"][17] = {}
    elif kind == "optimizer_parameter_order":
        payload["optimizer_state"]["param_groups"][0]["params"].reverse()
    elif kind == "momentum_shape":
        payload["optimizer_state"]["state"][0]["momentum_buffer"] = torch.zeros(1)
    elif kind == "momentum_dtype":
        state = payload["optimizer_state"]["state"][0]
        state["momentum_buffer"] = state["momentum_buffer"].double()
    elif kind == "momentum_nonfinite":
        payload["optimizer_state"]["state"][0]["momentum_buffer"].fill_(float("inf"))
    elif kind == "momentum_extra_field":
        payload["optimizer_state"]["state"][0]["lr"] = 1
    elif kind == "module_mode":
        payload["module_modes"]["head"] = 1
    elif kind == "module_missing":
        payload["module_modes"].pop("head")
    elif kind == "gradient_unknown":
        payload["gradient_modules"] = ["frozen"]
    elif kind == "gradient_duplicate":
        payload["gradient_modules"] = ["head", "head"]
    elif kind == "rng_shape":
        payload["torch_rng_state"] = torch.zeros(1, dtype=torch.uint8)
    elif kind == "rng_dtype":
        payload["torch_rng_state"] = payload["torch_rng_state"].float()
    elif kind == "rng_invalid":
        payload["torch_rng_state"].fill_(255)
    else:
        raise AssertionError(kind)


@pytest.mark.parametrize("kind", CORRUPTIONS)
def test_invalid_state_rejected_before_any_trainer_or_rng_mutation(tiny, tmp_path, kind):
    make, torch = tiny
    _, path, binding, sampler, payload = saved_state(tiny, tmp_path)
    corrupt(payload, kind, torch)
    torch.save(payload, path)
    target = make()
    before_model = deepcopy(target.model.state_dict())
    before_optimizer = deepcopy(target.optimizer.state_dict())
    before_rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="Training|Unsupported"):
        training_state.load_state(target, path, binding=binding, sampler=sampler)
    assert_tree_equal(torch, target.model.state_dict(), before_model)
    assert_tree_equal(torch, target.optimizer.state_dict(), before_optimizer)
    assert target.gradient_modules == set()
    assert torch.equal(torch.get_rng_state(), before_rng)


def test_runtime_identity_uses_full_build_versions_and_execution_settings(tiny, monkeypatch):
    make, torch = tiny
    trainer = make()
    monkeypatch.setattr(training_state.importlib.metadata, "version", lambda _: "0.25.0+test")
    identity = training_state.runtime_identity(trainer)
    assert identity == {
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "torchvision_version": "0.25.0+test",
        "machine": platform.machine(),
        "device": "cpu",
        "precision": "float32",
        "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": True,
    }


def test_changed_runtime_is_rejected_before_loading_state(tiny, tmp_path, monkeypatch):
    make, torch = tiny
    _, path, binding, sampler, _ = saved_state(tiny, tmp_path)
    monkeypatch.setattr(torch, "get_num_threads", lambda: binding["runtime"]["threads"] + 1)
    with pytest.raises(ValueError, match="runtime changed"):
        training_state.load_state(make(), path, binding=binding, sampler=sampler)


def test_non_cpu_float32_runtime_cannot_be_claimed(tiny):
    make, _ = tiny
    trainer = make()
    trainer.model.double()
    with pytest.raises(ValueError, match="CPU float32"):
        training_state.runtime_identity(trainer)


@pytest.mark.parametrize("kind", ["duplicate_order", "negative_order", "bool_order", "rng_cursor"])
def test_invalid_expected_sampler_is_rejected_on_write(tiny, tmp_path, kind):
    make, _ = tiny
    trainer = make()
    step(trainer, 1)
    binding, sampler = inputs(trainer, count=1)
    if kind == "duplicate_order":
        sampler["remaining_order"] = [0, 0]
    elif kind == "negative_order":
        sampler["remaining_order"] = [-1]
    elif kind == "bool_order":
        sampler["remaining_order"] = [True]
    else:
        sampler["random_state"][1][-1] = 625
    with pytest.raises(ValueError, match="sampler"):
        training_state.write_state(
            trainer, tmp_path / "invalid.pth", binding=binding, sampler=sampler
        )
    assert not (tmp_path / "invalid.pth").exists()


@pytest.mark.parametrize("kind", ["frozen", "buffer", "weights", "momentum"])
def test_corrupt_live_state_cannot_be_written(tiny, tmp_path, kind):
    make, _ = tiny
    trainer = make()
    step(trainer, 1)
    binding, sampler = inputs(trainer, count=1)
    if kind == "frozen":
        trainer.model.frozen.bias.detach().add_(1)
    elif kind == "buffer":
        trainer.model.fixed_offset.add_(1)
    elif kind == "weights":
        trainer.model.head.bias.detach().fill_(float("inf"))
    else:
        trainer.optimizer.state[trainer.model.head.bias]["momentum_buffer"].fill_(float("nan"))
    with pytest.raises(ValueError, match="Training state"):
        training_state.write_state(
            trainer, tmp_path / "invalid.pth", binding=binding, sampler=sampler
        )
    assert not (tmp_path / "invalid.pth").exists()


def test_optimizer_names_bind_same_shape_parameters_to_correct_slots(tiny, tmp_path):
    make, torch = tiny
    _, path, binding, sampler, _ = saved_state(tiny, tmp_path)
    target = make()
    target.optimizer = torch.optim.SGD(
        list(reversed(target.parameters)), lr=0.025, momentum=0.9, weight_decay=0.0005
    )
    with pytest.raises(ValueError, match="parameter order"):
        training_state.load_state(target, path, binding=binding, sampler=sampler)


def test_file_limit_checked_before_deserialization(tiny, tmp_path, monkeypatch):
    make, torch = tiny
    _, path, binding, sampler, _ = saved_state(tiny, tmp_path)
    monkeypatch.setattr(training_state, "MAX_STATE_BYTES", path.stat().st_size - 1)
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("Must check size first"))
    with pytest.raises(ValueError, match="size limit"):
        training_state.load_state(make(), path, binding=binding, sampler=sampler)


def test_oversized_written_state_is_removed(tiny, tmp_path, monkeypatch):
    make, _ = tiny
    trainer = make()
    step(trainer, 1)
    binding, sampler = inputs(trainer, count=1)
    path = tmp_path / "too-large.pth"
    monkeypatch.setattr(training_state, "MAX_STATE_BYTES", 1)
    with pytest.raises(ValueError, match="size limit"):
        training_state.write_state(trainer, path, binding=binding, sampler=sampler)
    assert not path.exists()


@pytest.mark.parametrize("kind", ["missing", "empty", "not_zip", "compressed", "unsafe_pickle"])
def test_invalid_container_or_unsafe_pickle_is_rejected(tiny, tmp_path, kind):
    make, torch = tiny
    target = make()
    binding, sampler = inputs(target)
    path = tmp_path / "invalid.pth"
    if kind == "empty":
        path.touch()
    elif kind == "not_zip":
        path.write_bytes(b"not a torch archive")
    elif kind == "compressed":
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("payload", b"x" * 1000)
    elif kind == "unsafe_pickle":
        torch.save(SimpleNamespace(marker="must not deserialize arbitrary classes"), path)
    with pytest.raises(ValueError, match="Training state"):
        training_state.load_state(target, path, binding=binding, sampler=sampler)


def test_torch_load_is_always_cpu_and_weights_only(tiny, tmp_path, monkeypatch):
    make, torch = tiny
    _, path, binding, sampler, _ = saved_state(tiny, tmp_path)
    original = torch.load
    calls = []

    def capture(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "load", capture)
    training_state.load_state(make(), path, binding=binding, sampler=sampler)
    assert calls == [{"map_location": "cpu", "weights_only": True}]


def test_expected_checksum_is_verified_on_the_descriptor_used_for_load(tiny, tmp_path, monkeypatch):
    make, torch = tiny
    source, path, binding, sampler, _ = saved_state(tiny, tmp_path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    replacement = tmp_path / "replacement.pth"
    torch.save({"this": "must not be loaded"}, replacement)
    original = torch.load

    def exchange_path(file, **kwargs):
        replacement.replace(path)
        assert file.fileno() >= 0
        return original(file, **kwargs)

    monkeypatch.setattr(torch, "load", exchange_path)
    target = make()
    training_state.load_state(
        target, path, binding=binding, sampler=sampler, expected_sha256=digest
    )
    assert_tree_equal(torch, target.model.state_dict(), source.model.state_dict())


@pytest.mark.parametrize("digest", ["f" * 64, "wrong", "F" * 64, True])
def test_wrong_or_invalid_checksum_prevents_deserialization(tiny, tmp_path, monkeypatch, digest):
    make, torch = tiny
    _, path, binding, sampler, _ = saved_state(tiny, tmp_path)
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("Must verify checksum first"))
    with pytest.raises(ValueError, match="checksum"):
        training_state.load_state(
            make(), path, binding=binding, sampler=sampler, expected_sha256=digest
        )
