"""Custom output namespaces, legacy checkpoint metadata and local dynamic-head reload."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
from test_models import fixture_output
from test_training_taxonomy import contract as frozen_contract

from iris import models
from iris.model_taxonomy import class_contract
from iris.taxonomies import TAXONOMY

contract = frozen_contract


def checkpoint_row(contract):
    return {
        "id": "trained-fixture",
        "name": "Synthetic custom checkpoint",
        "architecture": models.TRAINING_ARCHITECTURE,
        "path": "models/custom-fixture.pth",
        "weight_sha256": "a" * 64,
        "training_id": "training-fixture",
        "parent_model_id": models.TRAINING_ARCHITECTURE,
        "metadata": deepcopy(contract),
        "created_at": "2026-10-04T00:00:00+00:00",
    }


def test_custom_prediction_ids_never_use_coco_names_even_when_numbers_collide(contract):
    rows = models._serialize_predictions(
        fixture_output(boxes=[[0, 0, 120, 80], [1, 2, 30, 40]], labels=[2, 4], scores=[0.002, 0.7]),
        (120, 80),
        contract=contract,
    )
    assert rows == [
        {
            "box": [0, 0, 120, 80],
            "label_id": 2,
            "native_label_id": 2,
            "label": "helmet",
            "score": 0.002,
            "taxonomy_id": contract["taxonomy_id"],
        },
        {
            "box": [1, 2, 30, 40],
            "label_id": 4,
            "native_label_id": 4,
            "label": "bottle",
            "score": 0.7,
            "taxonomy_id": contract["taxonomy_id"],
        },
    ]


@pytest.mark.parametrize("native_id", [0, 5, 2.0, True])
def test_custom_predictions_reject_unknown_or_noninteger_head_slots(contract, native_id):
    with pytest.raises(ValueError, match="native class"):
        models._serialize_predictions(
            fixture_output(labels=[native_id]), (120, 80), contract=contract
        )


def test_custom_prediction_slots_above_coco_range_are_supported(contract):
    contract["taxonomy"]["classes"] = [
        {"id": f"object_{index}", "name": f"Object {index}", "definition": f"Target {index}."}
        for index in range(1, 101)
    ]
    contract["class_mapping"] = contract["output_class_mapping"] = {
        item["id"]: index for index, item in enumerate(contract["taxonomy"]["classes"], 1)
    }
    result = models._serialize_predictions(
        fixture_output(labels=[100]), (120, 80), contract=contract
    )
    assert result[0]["label_id"] == 100 and result[0]["label"] == "object_100"


def test_legacy_checkpoint_metadata_remains_coco_compatible_without_rewriting():
    metadata = {"class_mapping": {"person": 1, "car": 2}, "training_scene_groups": ["old"]}
    spec = models._trained_spec(checkpoint_row(metadata))
    assert spec["classes"] == [{"id": 1, "name": "person"}, {"id": 3, "name": "car"}]
    assert spec["native_to_coco"] == {0: 0, 1: 1, 2: 3}
    assert spec["taxonomy"] == TAXONOMY
    assert spec["provenance"] == metadata
    assert "taxonomy" not in metadata
    result = models._serialize_predictions(
        fixture_output(labels=[2]), (120, 80), spec["native_to_coco"]
    )
    assert result[0]["label"] == "car" and result[0]["label_id"] == 3
    assert "taxonomy_id" not in result[0]


@pytest.mark.parametrize(
    "corrupt",
    ["missing_snapshot", "taxonomy_alias", "native_mapping", "output_mapping", "missing_mapping"],
)
def test_incomplete_or_forged_custom_metadata_never_falls_back_to_legacy(contract, corrupt):
    if corrupt == "missing_snapshot":
        contract.pop("taxonomy")
    elif corrupt == "taxonomy_alias":
        contract["taxonomy_id"] = TAXONOMY["id"]
    elif corrupt == "native_mapping":
        contract["class_mapping"]["bottle"] = 44
    elif corrupt == "output_mapping":
        contract["output_class_mapping"]["bottle"] = 44
    else:
        contract.pop("class_mapping")
    with pytest.raises(ValueError, match="class definition"):
        models._trained_spec(checkpoint_row(contract))


def test_catalog_keeps_other_models_visible_when_custom_contract_is_corrupt(
    contract, tmp_path, monkeypatch
):
    contract.pop("taxonomy")
    monkeypatch.setattr(models, "_trained_rows", lambda _root: [checkpoint_row(contract)])
    monkeypatch.setattr(models, "_runtime_problem", lambda: None)
    rows = models.catalog(tmp_path)
    invalid = next(item for item in rows if item["id"] == "trained-fixture")
    assert len(rows) == 3
    assert invalid["status"] == "invalid_weights" and "class definition" in invalid["reason"]
    assert not invalid["inference"] and not invalid["training"] and invalid["classes"] == []
    with pytest.raises(ValueError, match="class definition"):
        models.get_spec("trained-fixture", tmp_path)


def test_dynamic_checkpoint_reloads_strictly_with_frozen_metadata_and_zero_downloads(
    contract, tmp_path, monkeypatch
):
    torch = pytest.importorskip("torch")
    torchvision = pytest.importorskip("torchvision")
    import hashlib

    class TinyDetector(torch.nn.Module):
        def __init__(self, classes):
            super().__init__()
            self.backbone = torch.nn.Identity()
            self.roi_heads = torch.nn.Module()
            self.roi_heads.box_predictor = (
                torchvision.models.detection.faster_rcnn.FastRCNNPredictor(2, classes)
            )
            self.transform = SimpleNamespace(
                image_mean=[0.5] * 3,
                image_std=[0.5] * 3,
                min_size=(320,),
                max_size=640,
                fixed_size=None,
                size_divisible=32,
            )

    path = tmp_path / "models" / "custom-fixture.pth"
    path.parent.mkdir()
    original = TinyDetector(5)
    torch.save(original.state_dict(), path)
    row = checkpoint_row(contract)
    row["weight_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(models, "_trained_rows", lambda _root: [deepcopy(row)])
    monkeypatch.setattr(models, "_runtime_problem", lambda: None)
    monkeypatch.setattr(models, "urlopen", lambda *_a, **_k: pytest.fail("No download allowed"))
    captured = {}

    def builder(**kwargs):
        captured.update(kwargs)
        return TinyDetector(kwargs["num_classes"])

    monkeypatch.setattr(torchvision.models.detection, models.TRAINING_ARCHITECTURE, builder)
    detector = models.TorchvisionDetector(tmp_path, row["id"])
    assert captured["weights"] is captured["weights_backbone"] is None
    assert captured["num_classes"] == 5
    assert class_contract(detector.metadata) == contract
    assert detector.metadata["head_class_slots"] == 5
    assert detector.native_to_coco is None
    for name, value in original.state_dict().items():
        assert torch.equal(value, detector.model.state_dict()[name])
    # A head with the wrong number of slots cannot load even when its file hash
    # and frozen metadata have individually passed validation.
    monkeypatch.setattr(
        torchvision.models.detection,
        models.TRAINING_ARCHITECTURE,
        lambda **_kwargs: TinyDetector(3),
    )
    with pytest.raises(RuntimeError, match="size mismatch"):
        models.TorchvisionDetector(tmp_path, row["id"])
