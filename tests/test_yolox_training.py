"""YOLOX preprocessing, class semantics, real native losses and durable CPU recovery.

Only seeded synthetic weights/images are used; no downloads or workspace media.
"""

import hashlib
from copy import deepcopy
from types import SimpleNamespace

import pytest
from PIL import Image
from test_training_recovery import execute, fixture_workspace, queue, resumed
from test_training_state import assert_tree_equal, inputs
from test_training_taxonomy import contract as custom_contract

from iris import models, training
from iris.training_architectures import YOLOX, training_scope
from iris.yolox_spec import POLICY

contract = custom_contract


@pytest.fixture
def runtime():
    torch = pytest.importorskip("torch")
    vision = pytest.importorskip("torchvision")
    original_rng = torch.get_rng_state()
    original_determinism = torch.are_deterministic_algorithms_enabled()
    original_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    yield torch, vision
    torch.set_rng_state(original_rng)
    torch.use_deterministic_algorithms(original_determinism)
    torch.set_num_threads(original_threads)


def test_preprocessing_preserves_bgr_scale_and_top_left_padding(runtime):
    from iris.yolox_runtime import preprocess

    torch = runtime[0]
    tensor, ratio = preprocess(Image.new("RGB", (832, 416), (11, 22, 33)))
    assert tensor.shape == (1, 3, 416, 416) and tensor.dtype == torch.float32
    assert ratio == 0.5
    assert tensor[0, :, 0, 0].tolist() == [33, 22, 11]
    assert torch.all(tensor[:, :, 208:, :] == 114)
    assert tensor[0, :, 207, 415].tolist() == [33, 22, 11]


def test_postprocess_maps_compact_coco_slots_and_does_not_add_background(runtime):
    from iris.yolox_runtime import coco_class_ids, postprocess

    torch = runtime[0]
    # Native compact index2 is car (sparse COCO3), not COCO2 (bicycle).
    prediction = torch.zeros((1, 2, 85))
    prediction[0, :, :4] = torch.tensor([100, 100, 40, 40])
    prediction[0, :, 4] = 0.8
    prediction[0, :, 7] = torch.tensor([0.75, 0.5])
    output = postprocess(prediction, (832, 416), 0.5)
    result = models._serialize_predictions(
        output,
        (832, 416),
        dict(enumerate(coco_class_ids(), start=1)),
    )
    assert len(result) == 1
    assert result[0]["label"] == "car" and result[0]["label_id"] == 3
    assert result[0]["native_label_id"] == 3
    assert result[0]["score"] == pytest.approx(0.6)
    assert result[0]["box"] == [160, 160, 240, 240]


def test_native_export_shape_has_only_foreground_classes(runtime):
    from iris.yolox_runtime import build_model, preprocess

    torch = runtime[0]
    model = build_model(4).eval()
    tensor, _ = preprocess(Image.new("RGB", (64, 32)))
    with torch.inference_mode():
        decoded = model(tensor)
        model.head.decode_in_inference = False
        raw = model(tensor)
    assert raw.shape == decoded.shape == (1, 3549, 9)
    assert torch.isfinite(raw).all() and torch.isfinite(decoded).all()
    assert torch.equal(decoded[:, :, 4:], raw[:, :, 4:])
    assert not torch.equal(decoded[:, :, :4], raw[:, :, :4])


def test_head_copy_uses_sparse_coco_mapping_and_preserves_objectness(runtime, contract):
    from iris.yolox_runtime import build_model, coco_class_ids
    from iris.yolox_training import prepare_yolox_head

    torch = runtime[0]
    model = build_model()
    original = deepcopy(model.state_dict())
    metadata = prepare_yolox_head(model, torch, contract, trained=False)
    assert model.head.num_classes == metadata["head_class_slots"] == 4
    assert metadata["background_class"] is False
    for scale in range(3):
        for label, coco_id in (("vehicle", 3), ("bottle", 44)):
            target = contract["class_mapping"][label] - 1
            source = coco_class_ids().index(coco_id)
            assert torch.equal(
                model.head.cls_preds[scale].weight[target],
                original[f"head.cls_preds.{scale}.weight"][source],
            )
        for prefix in ("obj_preds", "reg_preds"):
            assert torch.equal(
                getattr(model.head, prefix)[scale].weight,
                original[f"head.{prefix}.{scale}.weight"],
            )
    prepared = deepcopy(model.state_dict())
    prepare_yolox_head(model, torch, contract, trained=True)
    assert_tree_equal(torch, prepared, model.state_dict())


@pytest.fixture
def make_trainer(tmp_path, monkeypatch, contract, runtime):
    from iris.yolox_runtime import build_model

    torch, vision = runtime

    def make(scope="prediction_head_only"):
        monkeypatch.setattr(
            training,
            "TorchvisionDetector",
            lambda *_args, **_kwargs: SimpleNamespace(
                model=build_model(),
                spec={"origin": "official", "architecture": YOLOX},
                metadata={"weight_sha256": "a" * 64},
                functional=vision.transforms.functional,
            ),
        )
        return training._HeadTrainer(
            tmp_path,
            "synthetic",
            {
                **contract,
                "architecture": YOLOX,
                "training_adapter": deepcopy(POLICY),
                "scope": scope,
                "scope_version": 1,
                "trainable_modules": training_scope(scope, YOLOX)["trainable_modules"],
                "parent_weight_sha256": "a" * 64,
                "seed": 18,
                "learning_rate": 0.001,
                "momentum": 0.9,
                "weight_decay": 0.0005,
                "device": "cpu",
            },
        )

    return make, torch


@pytest.mark.parametrize("scope", training.TRAINING_SCOPES)
def test_real_native_loss_positive_and_negative_preserves_frozen_parameters(
    make_trainer,
    tmp_path,
    scope,
):
    make, torch = make_trainer
    trainer = make(scope)
    image = Image.new("RGB", (64, 32), (43, 73, 101))
    positive = trainer.step(image, [{"label": "vehicle", "box": [12, 8, 44, 30]}])
    negative = trainer.step(image, [])
    assert positive["losses"]["cls_loss"] > 0 and positive["losses"]["iou_loss"] > 0
    assert negative["losses"]["cls_loss"] == 0 and negative["losses"]["iou_loss"] == 0
    assert negative["loss"] == negative["losses"]["conf_loss"] > 0
    assert trainer.metadata["head_class_slots"] == 4
    assert all(
        not module.training
        for module in trainer.model.modules()
        if isinstance(module, torch.nn.BatchNorm2d)
    )
    assert trainer.metadata["training_loss_policy"] == POLICY
    saved = trainer.write_checkpoint(tmp_path / "weights.pth")
    assert saved["head_weights_changed"]
    assert saved["frozen_parameters_unchanged"] and saved["model_buffers_unchanged"]


def test_native_loss_recovery_restores_exact_cpu_continuation(make_trainer, tmp_path):
    make, torch = make_trainer
    image = Image.new("RGB", (64, 32), (43, 73, 101))
    boxes = [{"label": "vehicle", "box": [12, 8, 44, 30]}]
    first = make()
    first.step(image, boxes)
    binding, sampler = inputs(first, count=1)
    path = tmp_path / "state.pth"
    first.write_resume_state(path, binding=binding, sampler=sampler)
    expected = first.step(image, [])
    child = make()
    child.load_resume_state(
        path,
        binding=binding,
        sampler=sampler,
        expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    assert child.step(image, []) == expected
    assert_tree_equal(torch, first.model.state_dict(), child.model.state_dict())
    assert_tree_equal(torch, first.optimizer.state_dict(), child.optimizer.state_dict())


def test_large_negative_loss_has_bounded_finite_update_and_records_unclipped_norm(
    make_trainer,
    monkeypatch,
):
    from iris import yolox_training

    make, torch = make_trainer
    trainer = make("partial_backbone")
    original = yolox_training.training_losses

    def scaled(*args, **kwargs):
        loss, components = original(*args, **kwargs)
        return loss * 1e7, {key: value * 1e7 for key, value in components.items()}

    monkeypatch.setattr(yolox_training, "training_losses", scaled)
    before = {
        name: parameter.detach().clone() for name, parameter in trainer.selected_parameters.items()
    }
    result = trainer.step(Image.new("RGB", (64, 32), (43, 73, 101)), [])
    assert result["losses"]["cls_loss"] == result["losses"]["iou_loss"] == 0
    assert result["gradient_norm_before_clip"] > 1000000
    assert result["gradient_clipped"] and result["gradient_max_norm"] == 10
    norm = (
        torch.stack([parameter.grad.norm().square() for parameter in trainer.parameters])
        .sum()
        .sqrt()
    )
    assert norm.item() <= 10.0001
    delta = (
        torch.stack(
            [
                (parameter.detach() - before[name]).norm().square()
                for name, parameter in trainer.selected_parameters.items()
            ]
        )
        .sum()
        .sqrt()
    )
    assert delta.item() < 0.011  # lr .001, norm <=10, plus the small recorded weight decay.
    assert all(torch.isfinite(parameter).all() for parameter in trainer.parameters)


def test_nonfinite_gradient_stops_before_optimizer_even_with_clipping(make_trainer):
    make, torch = make_trainer
    trainer = make()
    before = deepcopy(trainer.model.state_dict())
    trainer.parameters[0].register_hook(lambda gradient: gradient * float("nan"))
    with pytest.raises(ValueError, match="nonfinite trainable gradient"):
        trainer.step(Image.new("RGB", (64, 32)), [])
    assert_tree_equal(torch, before, trainer.model.state_dict())
    assert trainer.optimizer.state_dict()["state"] == {}


def test_old_unclipped_recipe_cannot_resume_under_v2():
    old = {key: value for key, value in POLICY.items() if key != "gradient_clipping"}
    old["id"] = "iris-yolox-nano-training-v1"
    with pytest.raises(ValueError, match="adapter changed"):
        training._scope_from_config(
            {
                "architecture": YOLOX,
                "scope": "partial_backbone",
                "scope_version": 1,
                "trainable_modules": training_scope("partial_backbone", YOLOX)["trainable_modules"],
                "training_adapter": old,
            }
        )


def test_job_resume_keeps_yolox_architecture_and_frozen_policy(tmp_path, monkeypatch):
    store, dataset, _ = fixture_workspace(tmp_path)
    parent = {**models.get_spec(YOLOX), "status": "ready", "weight_sha256": "b" * 64}
    monkeypatch.setattr(training, "catalog", lambda _root: [parent])
    workspace = store, dataset, parent
    preview = training.preview_training(
        store,
        name="YOLOX fixture",
        dataset_id=dataset["id"],
        parent_model_id=YOLOX,
    )
    assert preview["config"]["training_adapter"] == POLICY
    assert preview["config"]["architecture"] == YOLOX
    run = queue(workspace)
    execute(store, run, stop=10)
    child = resumed(store, run)
    execute(store, child)
    completed = store.get("training_runs", child["id"])
    model = store.get("trained_models", completed["checkpoint_id"])
    assert model["architecture"] == model["metadata"]["architecture"] == YOLOX
    assert model["metadata"]["config"]["training_adapter"] == POLICY
