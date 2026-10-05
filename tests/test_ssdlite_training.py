"""SSDLite adapter contracts on tiny CPU modules; no detector is constructed."""

import subprocess
import sys
from copy import deepcopy
from types import SimpleNamespace

import pytest

from iris.ssdlite_training import (
    POLICY,
    configure_ssdlite_training,
    prepare_model,
    prepare_ssdlite_head,
)


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


def contract(*, custom=False):
    classes = [{"id": "person", "coco_id": 1}, {"id": "car", "coco_id": 3}]
    if custom:
        classes.append({"id": "parcel"})
    return {
        "class_mapping": {item["id"]: index for index, item in enumerate(classes, 1)},
        "taxonomy": {"classes": classes},
    }


def synthetic_ssdlite(torch, classes=91, dtype=None):
    """Only a six-scale head layout and callable synthetic loss, never an SSD."""
    dtype = dtype or torch.float32

    def head(columns):
        result = torch.nn.Module()
        result.num_columns = columns
        result.module_list = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Sequential(
                        torch.nn.Conv2d(2, 2, 3, padding=1, groups=2, dtype=dtype),
                        torch.nn.BatchNorm2d(2, eps=0.001, momentum=0.03, dtype=dtype),
                        torch.nn.ReLU6(),
                    ),
                    torch.nn.Conv2d(2, 6 * columns, 1, dtype=dtype),
                )
                for _ in range(6)
            ]
        )
        return result

    class SyntheticHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Module()
            self.backbone.features = torch.nn.Sequential(
                torch.nn.Conv2d(2, 2, 1, dtype=dtype),
                torch.nn.Conv2d(2, 2, 1, dtype=dtype),
            )
            self.backbone.extra = torch.nn.ModuleList(
                [torch.nn.Conv2d(2, 2, 1, dtype=dtype) for _ in range(4)]
            )
            self.head = torch.nn.Module()
            self.head.classification_head = head(classes)
            self.head.regression_head = head(4)
            self.anchor_generator = SimpleNamespace(num_anchors_per_location=lambda: [6] * 6)
            self.neg_to_pos_ratio = 3.0
            self.native_calls = []
            self.native_losses = None

        def forward(self, images, targets):
            self.last_targets = targets
            reference = next(self.parameters())
            scalar = sum((parameter - 0.25).square().sum() for parameter in self.parameters())
            # Eval-mode BN accepts batch-one, one-pixel feature maps. Leaving
            # even one BN in training mode must fail this structural fixture.
            for module in self.modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    sample = torch.ones(1, 2, 1, 1, device=reference.device, dtype=reference.dtype)
                    scalar = scalar + module(sample).sum() * 0
            return {"fixture_loss": scalar}

        def compute_loss(self, targets, outputs, anchors, matched):
            self.native_calls.append((targets, outputs, anchors, matched))
            self.native_losses = {
                "classification": outputs["cls_logits"].sum() * 0,
                "bbox_regression": outputs["bbox_regression"].sum() * 0,
            }
            return self.native_losses

    model = SyntheticHead()
    with torch.no_grad():
        for index, block in enumerate(model.head.classification_head.module_list):
            rows = torch.arange(6 * classes, dtype=dtype) + index * 1000
            block[1].weight.copy_(rows[:, None, None, None].expand_as(block[1].weight))
            block[1].bias.copy_(rows + 0.125)
    return model


def tiny_model(torch, classes=91):
    """Shared structural fixture for trainer scopes and recovery integration."""
    return synthetic_ssdlite(torch, classes=classes)


def test_module_import_does_not_load_optional_frameworks():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from iris.ssdlite_training import POLICY; "
            "assert 'torch' not in sys.modules and 'torchvision' not in sys.modules; "
            "assert POLICY['empty_image_hard_negatives'] == 3",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("custom", [False, True])
def test_official_rows_are_copied_for_every_anchor_and_scale(torch, custom):
    model = synthetic_ssdlite(torch)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    metadata = prepare_ssdlite_head(model, torch, contract(custom=custom), trained=False)
    count = 4 if custom else 3
    assert metadata == {"head_class_slots": count}
    assert model.head.classification_head.num_columns == count
    for scale, block in enumerate(model.head.classification_head.module_list):
        old_prefix = f"head.classification_head.module_list.{scale}.1"
        assert block[1].out_channels == 6 * count
        for anchor in range(6):
            for target, source in ((0, 0), (1, 1), (2, 3)):
                for field in ("weight", "bias"):
                    assert torch.equal(
                        getattr(block[1], field)[anchor * count + target],
                        before[f"{old_prefix}.{field}"][anchor * 91 + source],
                    )
    for name, value in model.state_dict().items():
        if name.startswith("head.classification_head.module_list.") and name.split(".")[4] == "1":
            continue
        assert torch.equal(value, before[name]), name


def test_custom_rows_are_seeded_without_changing_dtype_or_device(torch):
    results = []
    for seed in (72, 72, 73):
        model = synthetic_ssdlite(torch, dtype=torch.float64)
        torch.manual_seed(seed)
        prepare_ssdlite_head(model, torch, contract(custom=True), trained=False)
        novel = []
        for block in model.head.classification_head.module_list:
            assert block[1].weight.dtype == torch.float64
            assert block[1].weight.device.type == "cpu"
            for anchor in range(6):
                novel.append(block[1].weight[anchor * 4 + 3].detach())
                assert block[1].bias[anchor * 4 + 3].item() == 0
        results.append(torch.stack(novel))
    assert torch.equal(results[0], results[1])
    assert not torch.equal(results[0], results[2])
    assert 0.01 < results[0].std().item() < 0.06


def test_trained_parent_is_reused_without_consuming_rng(torch):
    model = synthetic_ssdlite(torch, classes=3)
    before = deepcopy(model.state_dict())
    rng = torch.get_rng_state().clone()
    prepare_ssdlite_head(model, torch, contract(), trained=True)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())


@pytest.mark.parametrize(
    "malformation",
    ["scales", "anchors", "regression_columns", "output_rows", "projection", "channels"],
)
def test_unsupported_layout_is_rejected(torch, malformation):
    model = synthetic_ssdlite(torch)
    if malformation == "scales":
        del model.head.classification_head.module_list[-1]
    elif malformation == "anchors":
        model.anchor_generator.num_anchors_per_location = lambda: [4] * 6
    elif malformation == "regression_columns":
        model.head.regression_head.num_columns = 5
    elif malformation == "output_rows":
        model.head.classification_head.module_list[0][1] = torch.nn.Conv2d(2, 91, 1)
    elif malformation == "projection":
        model.head.classification_head.module_list[0][1] = torch.nn.Identity()
    else:
        model.head.classification_head.module_list[0][0][0] = torch.nn.Conv2d(3, 3, 3, groups=3)
    with pytest.raises(ValueError, match="layout"):
        prepare_ssdlite_head(model, torch, contract(), trained=False)


@pytest.mark.parametrize("classes,trained", [(3, False), (91, True), (4, True)])
def test_wrong_class_contract_is_rejected(torch, classes, trained):
    model = synthetic_ssdlite(torch, classes=classes)
    with pytest.raises(ValueError, match="91 COCO|class mapping"):
        prepare_ssdlite_head(model, torch, contract(), trained=trained)


def loss_inputs(torch, *, count=5, positive=False):
    values = [[4.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 3.0, 0.0], [0.0, 5.0, 0.0]]
    logits = torch.tensor([values[:count]], requires_grad=True)
    boxes = torch.randn(1, count, 4, requires_grad=True)
    return (
        [{"boxes": torch.ones(1, 4) if positive else torch.empty(0, 4)}],
        {"cls_logits": logits, "bbox_regression": boxes},
        [torch.zeros(count, 4)],
        [torch.zeros(count, dtype=torch.int64) if positive else torch.full((count,), -1)],
    )


def test_negative_image_learns_only_from_three_hardest_background_anchors(torch):
    model = synthetic_ssdlite(torch, classes=3)
    metadata = configure_ssdlite_training(model, torch)
    assert metadata == {"training_loss_policy": POLICY}
    metadata["training_loss_policy"]["id"] = "caller modification"
    assert model._iris_ssdlite_training_policy == POLICY
    inputs = loss_inputs(torch)
    losses = model.compute_loss(*inputs)
    expected = torch.nn.functional.cross_entropy(
        inputs[1]["cls_logits"][0], torch.zeros(5, dtype=torch.int64), reduction="none"
    )[2:].sum()
    assert torch.equal(losses["classification"], expected)
    assert losses["bbox_regression"] is model.native_losses["bbox_regression"]
    sum(losses.values()).backward()
    gradients = inputs[1]["cls_logits"].grad[0]
    assert torch.count_nonzero(gradients[:2]).item() == 0
    assert torch.all(gradients[2:, 0] < 0).item()
    assert torch.count_nonzero(inputs[1]["bbox_regression"].grad).item() == 0
    assert len(model.native_calls) == 1


def test_fewer_than_three_anchors_remains_bounded(torch):
    model = synthetic_ssdlite(torch, classes=3)
    configure_ssdlite_training(model, torch)
    inputs = loss_inputs(torch, count=2)
    losses = model.compute_loss(*inputs)
    expected = torch.nn.functional.cross_entropy(
        inputs[1]["cls_logits"][0], torch.zeros(2, dtype=torch.int64), reduction="sum"
    )
    assert torch.equal(losses["classification"], expected)


def test_positive_image_uses_exact_native_loss_and_arguments(torch):
    model = synthetic_ssdlite(torch, classes=3)
    configure_ssdlite_training(model, torch)
    inputs = loss_inputs(torch, positive=True)
    losses = model.compute_loss(*inputs)
    assert losses is model.native_losses
    assert all(left is right for left, right in zip(inputs, model.native_calls[0], strict=True))


def test_policy_rejects_batch_change_and_impossible_empty_matches(torch):
    model = synthetic_ssdlite(torch, classes=3)
    configure_ssdlite_training(model, torch)
    inputs = loss_inputs(torch)
    with pytest.raises(ValueError, match="batch size one"):
        model.compute_loss(inputs[0] * 2, *inputs[1:])
    assert model.native_calls == []
    inputs[3][0][0] = 0
    with pytest.raises(ValueError, match="positive anchor"):
        model.compute_loss(*inputs)


def test_policy_rejects_changed_native_ratio_and_double_install(torch):
    model = synthetic_ssdlite(torch, classes=3)
    model.neg_to_pos_ratio = 4.0
    with pytest.raises(ValueError, match="three-to-one"):
        configure_ssdlite_training(model, torch)
    model.neg_to_pos_ratio = 3.0
    configure_ssdlite_training(model, torch)
    with pytest.raises(ValueError, match="already"):
        configure_ssdlite_training(model, torch)


def test_combined_preparation_leaves_batchnorm_buffers_and_modes_for_caller(torch):
    model = synthetic_ssdlite(torch).eval()
    before = deepcopy(dict(model.named_buffers()))
    metadata = prepare_model(model, contract(), trained=False, torch=torch)
    assert metadata["head_class_slots"] == 3
    assert metadata["training_loss_policy"] == POLICY
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            assert module.training is False, name
    assert all(torch.equal(value, before[name]) for name, value in model.named_buffers())
