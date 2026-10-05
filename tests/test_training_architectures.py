"""Architecture routing and recovery on synthetic runs and tiny CPU modules only."""

import hashlib
from copy import deepcopy
from types import SimpleNamespace

import pytest
from PIL import Image
from test_ssdlite_training import tiny_model
from test_training_recovery import execute, fixture_workspace, queue, resumed
from test_training_state import assert_tree_equal, inputs
from test_training_taxonomy import contract

from iris import models, training, training_state
from iris import training_recovery as recovery
from iris.ssdlite_training import POLICY
from iris.training_architectures import FRCNN, SSDLITE, capabilities, training_scope

contract = contract


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    store, dataset, _ = fixture_workspace(tmp_path)
    parent = {**models.get_spec(SSDLITE), "status": "ready", "weight_sha256": "b" * 64}
    monkeypatch.setattr(training, "catalog", lambda _root: [parent])
    return store, dataset, parent


@pytest.mark.parametrize("scope", training.TRAINING_SCOPES)
def test_ssdlite_preview_freezes_own_modules_and_explicit_negative_policy(workspace, scope):
    store, dataset, parent = workspace
    preview = training.preview_training(
        store,
        name="SSDLite fixture",
        dataset_id=dataset["id"],
        parent_model_id=parent["id"],
        scope=scope,
        checkpoint_interval=8,
        steps=25,
    )
    assert preview["scope"] == training_scope(scope, SSDLITE)
    assert preview["scope"] != training_scope(scope, FRCNN)
    assert preview["config"]["architecture"] == SSDLITE
    assert preview["config"]["training_adapter"] == POLICY
    assert "three hardest background" in " ".join(preview["notes"])
    assert preview["resume_supported"]
    assert not store.list("training_runs") and not store.list("jobs")
    run = queue(workspace, scope=scope)
    assert recovery.base_config(run["config"]) == preview["config"]


def test_two_catalog_architectures_have_independent_mutable_capability_snapshots():
    for architecture in (SSDLITE, FRCNN):
        spec = models.get_spec(architecture)
        assert spec["training"] and len(spec["training_scopes"]) == 3
        spec["training_scopes"][0]["trainable_modules"].clear()
        assert models.get_spec(architecture)["training_scopes"][0]["trainable_modules"]
    assert capabilities("unknown")["training"] is False


@pytest.mark.parametrize("scope", training.TRAINING_SCOPES)
def test_ssdlite_immutable_resume_publishes_its_own_architecture(workspace, monkeypatch, scope):
    store, _, _ = workspace
    source = queue(workspace, scope=scope)
    execute(store, source, stop=10)
    before = deepcopy(store.get("training_runs", source["id"]))
    child = resumed(store, source)
    execute(store, child)
    assert store.get("training_runs", source["id"]) == before
    finished = training.training_detail(store, child["id"])
    model = store.get("trained_models", finished["checkpoint_id"])
    assert model["architecture"] == model["metadata"]["architecture"] == SSDLITE
    assert model["metadata"]["config"]["training_adapter"] == POLICY
    monkeypatch.setattr(models, "_runtime_problem", lambda: None)
    spec = next(row for row in models.catalog(store.root) if row["id"] == model["id"])
    assert spec["training"] and spec["architecture"] == SSDLITE and spec["status"] == "ready"
    assert spec["training_scopes"] == capabilities(SSDLITE)["training_scopes"]
    assert [row["step"] for row in finished["checkpoints"]] == [24, 25]


@pytest.mark.parametrize(
    "change",
    [
        {"architecture": FRCNN},
        {"architecture": "unknown"},
        {"training_adapter": {**POLICY, "empty_image_hard_negatives": 0}},
        {"training_adapter": None},
        {"scope_version": None},
        {"trainable_modules": ["roi_heads.box_predictor"]},
    ],
)
def test_changed_architecture_or_adapter_rejected_before_optimizer(workspace, change):
    store = workspace[0]
    run = queue(workspace)
    store.update("training_runs", run["id"], {"config": {**run["config"], **change}})
    with pytest.raises(ValueError):
        execute(store, run, trainer_factory=lambda *_: pytest.fail("Must fail before optimizer"))
    assert not store.list("trained_models")


def test_changed_parent_architecture_rejected_even_with_unchanged_weight_hash(workspace):
    store, _, parent = workspace
    run = queue(workspace)
    parent["architecture"] = FRCNN
    with pytest.raises(ValueError, match="architecture"):
        execute(store, run, trainer_factory=lambda *_: pytest.fail("Must fail before optimizer"))


@pytest.fixture
def tiny_trainer(tmp_path, monkeypatch, contract):
    torch = pytest.importorskip("torch")
    vision = pytest.importorskip("torchvision")
    original_rng = torch.get_rng_state()
    original_determinism = torch.are_deterministic_algorithms_enabled()

    def make(scope="prediction_head_only", *, trained=False, alter=None):
        def detector(*_args, **_kwargs):
            model = tiny_model(torch, classes=len(contract["class_mapping"]) + 1 if trained else 91)
            if alter:
                alter(model)
            return SimpleNamespace(
                model=model,
                spec={
                    "origin": "trained" if trained else "official",
                    "architecture": SSDLITE,
                    **(contract if trained else {}),
                },
                metadata={"weight_sha256": "b" * 64},
                functional=vision.transforms.functional,
            )

        monkeypatch.setattr(training, "TorchvisionDetector", detector)
        config = {
            **contract,
            "seed": 13,
            "architecture": SSDLITE,
            "training_adapter": deepcopy(POLICY),
            "scope": scope,
            "scope_version": 1,
            "trainable_modules": training_scope(scope, SSDLITE)["trainable_modules"],
            "parent_weight_sha256": "b" * 64,
            "learning_rate": 0.001,
            "momentum": 0.9,
            "weight_decay": 0.0005,
            "device": "cpu",
        }
        return training._HeadTrainer(tmp_path, "synthetic", config)

    yield make, torch
    torch.set_rng_state(original_rng)
    torch.use_deterministic_algorithms(original_determinism)


@pytest.mark.parametrize("scope", training.TRAINING_SCOPES)
@pytest.mark.parametrize("trained", [False, True])
def test_tiny_ssdlite_scope_freezes_bn_and_publishes_portable_custom_head(
    tiny_trainer,
    tmp_path,
    scope,
    trained,
):
    make, torch = tiny_trainer
    trainer = make(scope, trained=trained)
    assert trainer.metadata["architecture"] == SSDLITE
    assert trainer.metadata["training_loss_policy"] == POLICY
    assert "training_proposal_filtering" not in trainer.metadata
    for name, parameter in trainer.model.named_parameters():
        assert parameter.requires_grad == any(
            training._matches_module(name, prefix) for prefix in trainer.scope["trainable_modules"]
        )
    batchnorms = [
        module for module in trainer.model.modules() if isinstance(module, torch.nn.BatchNorm2d)
    ]
    assert batchnorms and all(not module.training for module in batchnorms)
    trainer.step(Image.new("RGB", (16, 12)), [{"label": "helmet", "box": [1, 1, 8, 9]}])
    assert trainer.model.last_targets[0]["labels"].tolist() == [2]
    trainer.step(Image.new("RGB", (16, 12)), [])
    assert trainer.model.last_targets[0]["boxes"].shape == (0, 4)
    path = tmp_path / "completed.pth"
    result = trainer.write_checkpoint(path)
    assert result["head_weights_changed"] and result["model_buffers_unchanged"]
    assert result["frozen_parameters_unchanged"]
    saved = torch.load(path, map_location="cpu", weights_only=True)
    assert all(value.device.type == "cpu" for value in saved.values())
    rebuilt = tiny_model(torch, classes=len(trainer.class_mapping) + 1)
    rebuilt.load_state_dict(saved, strict=True)
    assert_tree_equal(torch, rebuilt.state_dict(), trainer.model.state_dict())


@pytest.mark.parametrize("scope", training.TRAINING_SCOPES)
def test_tiny_ssdlite_resume_retains_bn_modes_momentum_and_training_policy(
    tiny_trainer, tmp_path, scope
):
    make, torch = tiny_trainer
    first = make(scope)
    image = Image.new("RGB", (16, 12))
    first.step(image, [])
    binding, sampler = inputs(first, count=1)
    path = tmp_path / "state.pth"
    first.write_resume_state(path, binding=binding, sampler=sampler)
    expected = first.step(image, [])
    next_attempt = make(scope)
    next_attempt.load_resume_state(
        path,
        binding=binding,
        sampler=sampler,
        expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    assert all(
        not module.training
        for module in next_attempt.model.modules()
        if isinstance(module, torch.nn.BatchNorm2d)
    )
    assert next_attempt.model._iris_ssdlite_training_policy == POLICY
    assert next_attempt.step(image, []) == expected
    assert_tree_equal(torch, next_attempt.model.state_dict(), first.model.state_dict())
    assert_tree_equal(torch, next_attempt.optimizer.state_dict(), first.optimizer.state_dict())


def test_tiny_ssdlite_recovery_rejects_unfrozen_bn_before_applying_weights(tiny_trainer, tmp_path):
    make, torch = tiny_trainer
    trainer = make()
    trainer.step(Image.new("RGB", (16, 12)), [])
    binding, sampler = inputs(trainer, count=1)
    path = tmp_path / "state.pth"
    trainer.write_resume_state(path, binding=binding, sampler=sampler)
    payload = torch.load(path, weights_only=True)
    payload["module_modes"][trainer.frozen_batchnorm_modules[0]] = True
    torch.save(payload, path)
    target = make()
    before = deepcopy(target.model.state_dict())
    with pytest.raises(ValueError, match="normalization"):
        training_state.load_state(target, path, binding=binding, sampler=sampler)
    assert_tree_equal(torch, target.model.state_dict(), before)


def test_ssdlite_reports_keep_adapter_policy_without_private_fields(workspace):
    from iris.experiments import _training_summary

    store = workspace[0]
    row = queue(workspace)
    execute(store, row)
    finished = store.get("training_runs", row["id"])
    metadata = finished["metadata"]
    metadata["training_loss_policy"] = {**POLICY, "private": "DO_NOT_EXPORT"}
    config = finished["config"]
    config["training_adapter"]["private"] = "DO_NOT_EXPORT"
    store.update("training_runs", row["id"], {"metadata": metadata, "config": config})
    result = _training_summary(store, finished["checkpoint_id"])
    assert result["config"]["training_adapter"] == POLICY
    assert result["metadata"]["training_loss_policy"] == POLICY
    assert "DO_NOT_EXPORT" not in str(result)
