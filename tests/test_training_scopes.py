"""Per-run scope contracts and optimizer checks on a tiny structural torch fixture."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
from PIL import Image
from test_training import FixtureTrainer, queue, run
from test_training import workspace as training_workspace

from iris import training

SCOPES = tuple(training.TRAINING_SCOPES)
workspace = training_workspace


def preview(workspace, **kwargs):
    store, dataset, _, parent = workspace
    return training.preview_training(
        store,
        name="Scope fixture",
        dataset_id=dataset["id"],
        parent_model_id=parent["id"],
        **kwargs,
    )


@pytest.mark.parametrize("scope", SCOPES)
def test_preview_matches_frozen_queue_contract_without_loading_model(workspace, monkeypatch, scope):
    store, dataset, _, parent = workspace
    monkeypatch.setattr(
        training,
        "TorchvisionDetector",
        lambda *_args, **_kwargs: pytest.fail("Preview must not load a detector"),
    )
    result = preview(workspace, scope=scope, steps=3)
    assert not store.list("jobs") and not store.list("training_runs")
    assert result["scope"] == training.TRAINING_SCOPES[scope]
    assert result["dataset"] == {
        "id": dataset["id"],
        "name": dataset["name"],
        "train_images": 1,
        "positive_train_images": 1,
        "annotation_count": 1,
    }
    assert result["parent"] == {key: parent[key] for key in ("id", "name", "weight_sha256")}
    assert result["workload"] == {
        "steps": 3,
        "batch_size": 1,
        "image_visits": 3,
        "unique_images_min": 1,
        "full_passes": 3,
        "remainder_images": 0,
        "device": "cpu",
    }
    assert "trainable_parameters" not in result["config"]
    queued = queue(workspace, scope=scope, steps=3)
    assert queued["config"] == result["config"]
    assert queued["config"]["scope_version"] == 1
    assert (
        queued["config"]["trainable_modules"]
        == training.TRAINING_SCOPES[scope]["trainable_modules"]
    )


def test_workload_counts_only_train_images_and_annotations(workspace):
    manifest = workspace[2]
    positive = manifest["frames"][0]
    manifest["frames"].extend(
        [
            {**deepcopy(positive), "frame_id": "extra-positive"},
            {**deepcopy(positive), "frame_id": "extra-negative", "boxes": []},
        ]
    )
    result = preview(workspace, steps=5)
    assert result["dataset"]["train_images"] == 3
    assert result["dataset"]["positive_train_images"] == 2
    assert result["dataset"]["annotation_count"] == 2
    assert result["workload"]["unique_images_min"] == 3
    assert result["workload"]["full_passes"] == 1
    assert result["workload"]["remainder_images"] == 2


@pytest.mark.parametrize("scope", [None, "", "backbone", "all", True, 1, [], {}])
def test_unknown_scope_rejected_by_preview_and_create_before_dataset_access(
    workspace, monkeypatch, scope
):
    monkeypatch.setattr(training, "_manifest", lambda *_: pytest.fail("Validate scope first"))
    with pytest.raises(ValueError, match="Choose prediction"):
        preview(workspace, scope=scope)
    with pytest.raises(ValueError, match="Choose prediction"):
        queue(workspace, scope=scope)
    assert not workspace[0].list("jobs")


@pytest.mark.parametrize("scope", SCOPES)
def test_each_scope_consumes_train_only_and_preserves_loss_semantics(workspace, scope):
    store, _, manifest, _ = workspace
    for frame in manifest["frames"][1:]:
        store.artifact_path(frame["image_path"]).unlink()
    FixtureTrainer.seen = []
    row = queue(workspace, scope=scope, steps=2)
    result = run(store, row)
    assert result["steps_completed"] == 2
    assert FixtureTrainer.seen == [(1, 20, 40)] * 2
    saved = training.training_detail(store, row["id"])
    assert saved["metadata"]["scope"] == scope
    assert saved["metadata"]["scope_version"] == 1
    assert saved["metadata"]["quality_metrics"] is None
    assert saved["metadata"]["validation_consumed"] is False
    assert saved["metadata"]["test_consumed"] is False
    assert (
        saved["metadata"]["trainable_modules"]
        == training.TRAINING_SCOPES[scope]["trainable_modules"]
    )


@pytest.mark.parametrize(
    "change",
    [
        {"scope": "unknown"},
        {"scope": "full_model"},
        {"scope_version": 2},
        {"scope_version": True},
        {"trainable_modules": ["roi_heads"]},
        {"trainable_modules": []},
        {"scope_version": None},
    ],
)
def test_worker_rejects_changed_scope_contract_before_trainer_or_checkpoint(workspace, change):
    store = workspace[0]
    row = queue(workspace)
    config = {**row["config"], **change}
    store.update("training_runs", row["id"], {"config": config})

    def forbidden(*_):
        pytest.fail("Invalid scope must fail before loading the trainer")

    with pytest.raises(ValueError):
        run(store, row, trainer_factory=forbidden)
    assert not store.list("trained_models")
    assert not store.get("training_runs", row["id"])["history"]


@pytest.mark.parametrize("remove_scope", [False, True])
def test_legacy_queued_head_training_remains_supported(workspace, remove_scope):
    store = workspace[0]
    row = queue(workspace)
    config = dict(row["config"])
    config.pop("scope_version")
    config.pop("trainable_modules")
    if remove_scope:
        config.pop("scope")
    store.update("training_runs", row["id"], {"config": config})
    assert run(store, row)["checkpoint_id"]
    metadata = store.get("training_runs", row["id"])["metadata"]
    assert metadata["scope"] == "prediction_head_only"
    assert metadata["scope_version"] == 0


def test_unversioned_deeper_scope_is_rejected(workspace):
    store = workspace[0]
    row = queue(workspace, scope="full_model")
    config = dict(row["config"])
    config.pop("scope_version")
    config.pop("trainable_modules")
    store.update("training_runs", row["id"], {"config": config})
    with pytest.raises(ValueError, match="legacy"):
        run(store, row)
    assert not store.list("trained_models")


def test_scope_catalog_is_not_mutated_by_callers():
    scope = training.training_scope("partial_backbone")
    scope["trainable_modules"].clear()
    assert len(training.training_scope("partial_backbone")["trainable_modules"]) == 7


@pytest.fixture
def structural_trainer(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    torchvision = pytest.importorskip("torchvision")
    from collections import OrderedDict

    class SyntheticRPN(torch.nn.Linear):
        def __init__(self):
            super().__init__(2, 2)
            self.score_thresh = 0.05
            self.nms_thresh = 0.7

        def pre_nms_top_n(self):
            return 2000 if self.training else 150

        def post_nms_top_n(self):
            return 2000 if self.training else 150

    class SyntheticDetector(torch.nn.Module):
        """A tiny layout fixture, with a differentiable scalar using each parameter."""

        def __init__(self, classes=3):
            super().__init__()
            self.backbone = torch.nn.Module()
            self.backbone.body = torch.nn.Sequential(
                OrderedDict(
                    (
                        str(index),
                        torch.nn.Sequential(
                            torch.nn.Linear(2, 2), torchvision.ops.misc.FrozenBatchNorm2d(2)
                        ),
                    )
                    for index in range(17)
                )
            )
            self.backbone.fpn = torch.nn.Linear(2, 2)
            self.rpn = SyntheticRPN()
            self.roi_heads = torch.nn.Module()
            self.roi_heads.box_head = torch.nn.Linear(2, 2)
            self.roi_heads.box_predictor = (
                torchvision.models.detection.faster_rcnn.FastRCNNPredictor(2, classes)
            )
            self.omit_parameter = None
            self.nonfinite_loss = False
            self.zero_loss = False

        def forward(self, images, targets):
            self.last_targets = targets
            loss = sum(
                (parameter - 0.25).square().sum()
                for name, parameter in self.named_parameters()
                if name != self.omit_parameter
            )
            if self.nonfinite_loss:
                loss = loss * float("nan")
            if self.zero_loss:
                loss = loss * 0
            return {"fixture_loss": loss}

    def make(
        scope="prediction_head_only",
        *,
        origin="trained",
        alter=None,
        contract=None,
        parent_contract=None,
    ):
        def detector(*args, **kwargs):
            classes = (
                len((parent_contract or contract)["class_mapping"]) + 1
                if (parent_contract or contract)
                else 3
            )
            model = SyntheticDetector(classes if origin == "trained" else 91)
            if alter:
                alter(model)
            return SimpleNamespace(
                model=model,
                metadata={
                    "weight_sha256": "b" * 64,
                    "runtime": "tiny structural fixture",
                    "native_filtering": {"rpn": {"score_threshold": model.rpn.score_thresh}},
                },
                spec={"origin": origin, **(parent_contract or contract or {})},
                functional=torchvision.transforms.functional,
            )

        monkeypatch.setattr(training, "TorchvisionDetector", detector)
        config = {
            "seed": 0,
            "scope": scope,
            "scope_version": 1,
            "trainable_modules": training.training_scope(scope)["trainable_modules"],
            "parent_weight_sha256": "b" * 64,
            "learning_rate": 0.01,
            "momentum": 0.9,
            "weight_decay": 0.0005,
            **(contract or {}),
        }
        return training._HeadTrainer(tmp_path, "fixture", config)

    return make, torch, torchvision


@pytest.mark.parametrize("scope", SCOPES)
def test_scope_selects_exact_parameter_groups_and_records_actual_changes(
    structural_trainer, tmp_path, scope
):
    make, torch, _ = structural_trainer
    trainer = make(scope)
    before = {key: value.clone() for key, value in trainer.model.state_dict().items()}
    for name, parameter in trainer.model.named_parameters():
        assert parameter.requires_grad == any(
            training._matches_module(name, prefix)
            for prefix in training.TRAINING_SCOPES[scope]["trainable_modules"]
        )
    metadata = trainer.metadata
    assert metadata["trainable_parameters"] == sum(
        p.numel() for p in trainer.model.parameters() if p.requires_grad
    )
    assert (
        metadata["trainable_parameters"] + metadata["frozen_parameters"]
        == metadata["total_parameters"]
    )
    assert metadata["frozen_parameters"] == (
        0 if scope == "full_model" else sum(p.numel() for p in trainer.frozen_parameters.values())
    )
    assert all(
        not dict(trainer.model.named_modules())[name].training
        for name in trainer.frozen_batchnorm_modules
    )
    trainer.step(Image.new("RGB", (8, 8)), [{"label": "person", "box": [1, 1, 5, 6]}])
    facts = trainer.write_checkpoint(tmp_path / f"{scope}.pth")
    after = torch.load(tmp_path / f"{scope}.pth", weights_only=True)
    actual_changed = {name for name, value in before.items() if not torch.equal(value, after[name])}
    assert actual_changed
    assert all(
        any(training._matches_module(name, prefix) for prefix in metadata["trainable_modules"])
        for name in actual_changed
    )
    assert facts["changed_trainable_modules"] == metadata["trainable_modules"]
    assert facts["trainable_module_changes"] == dict.fromkeys(metadata["trainable_modules"], True)
    assert set(facts["modules_with_nonzero_gradients"]) == set(metadata["trainable_modules"])
    assert facts["head_weights_changed"] is True
    assert facts["frozen_parameters_unchanged"] is True
    assert facts["frozen_batchnorm_buffers_unchanged"] is True
    for name, value in trainer.model.named_buffers():
        assert torch.equal(value, before[name])


def test_deeper_scopes_train_early_or_late_visual_features_as_requested(structural_trainer):
    make, _, _ = structural_trainer
    partial = make("partial_backbone")
    assert not partial.model.backbone.body[12][0].weight.requires_grad
    assert partial.model.backbone.body[13][0].weight.requires_grad
    assert partial.model.backbone.body[16][0].weight.requires_grad
    assert partial.model.backbone.fpn.weight.requires_grad
    assert partial.model.rpn.weight.requires_grad
    assert partial.model.roi_heads.box_head.weight.requires_grad
    full = make("full_model")
    assert full.model.backbone.body[0][0].weight.requires_grad


def test_official_initialization_copies_only_background_person_car_rows(structural_trainer):
    make, torch, _ = structural_trainer
    saved = {}

    def capture(model):
        saved.update(
            {
                name: value.detach().clone()
                for name, value in model.roi_heads.box_predictor.named_parameters()
            }
        )

    trainer = make("full_model", origin="official", alter=capture)
    head = trainer.model.roi_heads.box_predictor
    assert torch.equal(head.cls_score.weight, saved["cls_score.weight"][[0, 1, 3]])
    rows = [category * 4 + coord for category in [0, 1, 3] for coord in range(4)]
    assert torch.equal(head.bbox_pred.weight, saved["bbox_pred.weight"][rows])
    assert head.cls_score.out_features == 3 and head.bbox_pred.out_features == 12


def test_trained_parent_keeps_exact_head_initialization(structural_trainer):
    make, torch, _ = structural_trainer
    saved = {}

    def capture(model):
        saved.update(
            {
                name: value.detach().clone()
                for name, value in model.roi_heads.box_predictor.named_parameters()
            }
        )

    trainer = make("partial_backbone", alter=capture)
    for name, value in trainer.model.roi_heads.box_predictor.named_parameters():
        assert torch.equal(value, saved[name])
    assert "Preserved" in trainer.metadata["head_initialization"]


@pytest.mark.parametrize("scope", SCOPES)
def test_missing_selected_gradients_are_rejected(structural_trainer, scope):
    make, _, _ = structural_trainer
    trainer = make(scope)
    trainer.model.omit_parameter = next(iter(trainer.selected_parameters))
    with pytest.raises(ValueError, match="Missing or nonfinite trainable gradient"):
        trainer.step(Image.new("RGB", (8, 8)), [])


@pytest.mark.parametrize("failure", ["loss", "gradient", "parameter"])
def test_nonfinite_values_do_not_create_checkpoint(structural_trainer, tmp_path, failure):
    make, torch, _ = structural_trainer
    trainer = make("full_model")
    if failure == "loss":
        trainer.model.nonfinite_loss = True
    elif failure == "gradient":
        trainer.parameters[0].register_hook(lambda gradient: gradient * float("nan"))
    else:
        original = trainer.optimizer.step

        def invalid_step():
            original()
            with torch.no_grad():
                trainer.parameters[0].fill_(float("inf"))

        trainer.optimizer.step = invalid_step
    with pytest.raises(ValueError, match="[Nn]onfinite"):
        trainer.step(Image.new("RGB", (8, 8)), [])
    assert not list(tmp_path.glob("*.pth"))


def test_zero_gradients_are_valid_but_unchanged_weights_are_not_published(
    structural_trainer, tmp_path
):
    make, _, _ = structural_trainer
    trainer = make("partial_backbone")
    trainer.model.zero_loss = True
    trainer.optimizer.param_groups[0]["weight_decay"] = 0
    assert trainer.step(Image.new("RGB", (8, 8)), [])["loss"] == 0
    assert not trainer.gradient_modules
    with pytest.raises(ValueError, match="No trainable weights changed"):
        trainer.write_checkpoint(tmp_path / "unchanged.pth")
    assert not (tmp_path / "unchanged.pth").exists()


@pytest.mark.parametrize("mutation", ["frozen_parameter", "buffer"])
def test_frozen_state_mutation_prevents_publication(structural_trainer, tmp_path, mutation):
    make, torch, _ = structural_trainer
    trainer = make("partial_backbone")
    trainer.step(Image.new("RGB", (8, 8)), [])
    with torch.no_grad():
        if mutation == "frozen_parameter":
            next(iter(trainer.frozen_parameters.values())).add_(1)
        else:
            next(trainer.model.buffers()).add_(1)
    with pytest.raises(ValueError, match="Frozen model"):
        trainer.write_checkpoint(tmp_path / "invalid.pth")
    assert not (tmp_path / "invalid.pth").exists()


def test_scope_changes_report_groups_that_really_changed(structural_trainer, tmp_path):
    make, torch, _ = structural_trainer
    trainer = make("partial_backbone")
    with torch.no_grad():
        trainer.model.backbone.body[13][0].weight.add_(0.1)
    facts = trainer.write_checkpoint(tmp_path / "one-group.pth")
    assert facts["changed_trainable_modules"] == ["backbone.body.13"]
    assert facts["head_weights_changed"] is False
    assert facts["trainable_module_changes"]["roi_heads"] is False


@pytest.mark.parametrize(
    "mutation", ["missing_prefix", "unexpected_body", "batchnorm", "extra_parameter"]
)
def test_unsupported_model_layout_is_rejected(structural_trainer, mutation):
    make, torch, _ = structural_trainer

    def mutate(model):
        if mutation == "missing_prefix":
            del model.backbone.fpn
        elif mutation == "unexpected_body":
            model.backbone.body.add_module("17", torch.nn.Linear(2, 2))
        elif mutation == "batchnorm":
            model.backbone.body[0][1] = torch.nn.BatchNorm2d(2)
        else:
            model.register_parameter("unknown", torch.nn.Parameter(torch.ones(1)))

    with pytest.raises(ValueError):
        make("full_model" if mutation == "extra_parameter" else "partial_backbone", alter=mutate)


@pytest.mark.parametrize("scope", SCOPES)
def test_training_retains_background_proposals_without_changing_parent_filtering(
    structural_trainer, scope
):
    make, _, _ = structural_trainer
    trainer = make(scope)
    assert trainer.model.rpn.score_thresh == 0.0
    filtering = trainer.metadata["training_proposal_filtering"]
    assert filtering["rpn_score_threshold"] == 0.0
    assert filtering["parent_inference_rpn_score_threshold"] == 0.05
    assert filtering["pre_nms_top_n"] == filtering["post_nms_top_n"] == 2000
    assert filtering["rpn_nms_iou_threshold"] == 0.7
    assert trainer.metadata["native_filtering"]["rpn"]["score_threshold"] == 0.05
    trainer.step(Image.new("RGB", (8, 8)), [])
