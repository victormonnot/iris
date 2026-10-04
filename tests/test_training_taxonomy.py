"""Frozen custom training contracts and tiny CPU head initialization, without downloads."""

from copy import deepcopy

import pytest
from PIL import Image
from test_training import FixtureTrainer, queue, run
from test_training import workspace as legacy_workspace
from test_training_scopes import structural_trainer as tiny_trainer

from iris import models
from iris.dataset_manifest import taxonomy_mappings
from iris.model_taxonomy import class_contract, dataset_contract
from iris.store import DEFAULT_PROJECT_ID
from iris.taxonomies import TAXONOMY, publish_taxonomy

workspace = legacy_workspace
structural_trainer = tiny_trainer
CLASSES = [
    {"id": "person", "name": "Worker", "definition": "A person in work clothing."},
    {"id": "helmet", "name": "Helmet", "definition": "A visible protective helmet."},
    {"id": "vehicle", "name": "Passenger car", "definition": "A car.", "coco_id": 3},
    {"id": "bottle", "name": "Bottle", "definition": "A bottle.", "coco_id": 44},
]


@pytest.fixture
def contract():
    taxonomy = {
        "id": "taxonomy-" + "a" * 32,
        "version": 2,
        "parent_id": TAXONOMY["id"],
        "classes": deepcopy(CLASSES),
        "box_format": TAXONOMY["box_format"],
        "review_guidance": TAXONOMY["review_guidance"],
        "created_at": "2026-10-04T00:00:00+00:00",
    }
    internal, output = taxonomy_mappings(taxonomy)
    return class_contract(
        {"taxonomy": taxonomy, "class_mapping": internal, "output_class_mapping": output}
    )


@pytest.fixture
def custom_workspace(workspace):
    store, _, manifest, _ = workspace
    taxonomy = publish_taxonomy(
        store, DEFAULT_PROJECT_ID, expected_taxonomy_id=TAXONOMY["id"], classes=CLASSES
    )
    internal, output = taxonomy_mappings(taxonomy)
    manifest.update(taxonomy=taxonomy, class_mapping=internal, coco_mapping=output)
    manifest["frames"][0]["boxes"][0]["label"] = "helmet"
    manifest["frames"][1]["boxes"][0]["label"] = "vehicle"
    manifest["frames"][2]["boxes"] = []
    manifest["frames"].append(
        {**deepcopy(manifest["frames"][0]), "frame_id": "negative-training", "boxes": []}
    )
    return workspace


def test_generic_run_freezes_contract_uses_negatives_and_reserves_holdouts(
    custom_workspace, monkeypatch
):
    store, _, manifest, _ = custom_workspace
    expected = dataset_contract(manifest)
    for frame in manifest["frames"][1:3]:
        store.artifact_path(frame["image_path"]).unlink()
    seen = []

    class Recorder(FixtureTrainer):
        def step(self, image, boxes):
            seen.append(deepcopy(boxes))
            return super().step(image, boxes)

    queued = queue(custom_workspace, steps=4)
    assert class_contract(queued["config"]) == expected
    # Live project publication has no effect on the immutable queued dataset.
    publish_taxonomy(
        store,
        DEFAULT_PROJECT_ID,
        expected_taxonomy_id=expected["taxonomy_id"],
        classes=[{**CLASSES[0], "definition": "Only a worker with a vest."}, *CLASSES[1:]],
    )
    result = run(store, queued, trainer_factory=Recorder)
    assert sum(not boxes for boxes in seen) == 2
    assert [boxes[0]["label"] for boxes in seen if boxes] == ["helmet", "helmet"]
    saved = store.get("trained_models", result["checkpoint_id"])
    assert class_contract(saved["metadata"]) == expected
    assert saved["metadata"]["training_scene_groups"] == ["train"]
    assert saved["metadata"]["validation_consumed"] is False
    assert saved["metadata"]["test_consumed"] is False
    monkeypatch.setattr(models, "_runtime_problem", lambda: None)
    spec = next(item for item in models.catalog(store.root) if item["id"] == saved["id"])
    assert class_contract(spec) == expected
    assert spec["classes"] == [
        {"id": index, "name": item["id"]} for index, item in enumerate(CLASSES, 1)
    ]
    assert "native_to_coco" not in spec


@pytest.mark.parametrize("mismatch", ["legacy", "version", "definition", "mapping"])
def test_custom_training_rejects_incompatible_trained_parent_before_queue(
    custom_workspace, mismatch
):
    store, _, manifest, parent = custom_workspace
    parent.update(origin="trained", **dataset_contract(manifest))
    if mismatch == "legacy":
        for field in ("taxonomy", "taxonomy_id", "class_mapping", "output_class_mapping"):
            parent.pop(field)
    elif mismatch == "version":
        parent["taxonomy"]["id"] = parent["taxonomy_id"] = "taxonomy-" + "b" * 32
    elif mismatch == "definition":
        parent["taxonomy"]["classes"][0]["definition"] = "Different frozen definition."
    else:
        parent["output_class_mapping"]["helmet"] = 90
    with pytest.raises(ValueError, match="class definitions|class_mapping"):
        queue(custom_workspace)
    assert store.list("training_runs") == store.list("jobs") == []


def test_custom_training_resumes_exact_contract_and_still_checks_parent_holdouts(custom_workspace):
    _, _, manifest, parent = custom_workspace
    parent.update(origin="trained", **dataset_contract(manifest))
    queued = queue(custom_workspace)
    assert class_contract(queued["config"]) == class_contract(parent)
    parent["provenance"] = {"training_frame_hashes": [manifest["frames"][1]["sha256"]]}
    with pytest.raises(ValueError, match="held-out"):
        queue(custom_workspace)


@pytest.mark.parametrize("changed", ["dataset", "parent", "queued_mapping"])
def test_worker_rechecks_frozen_semantics_before_loading_any_weights(custom_workspace, changed):
    store, _, manifest, parent = custom_workspace
    parent.update(origin="trained", **dataset_contract(manifest))
    queued = queue(custom_workspace)
    if changed == "dataset":
        manifest["taxonomy"]["classes"][0]["definition"] = "Changed frozen data."
    elif changed == "parent":
        parent["taxonomy"]["classes"][0]["definition"] = "Changed parent contract."
    else:
        config = queued["config"]
        config["class_mapping"]["helmet"] = 3
        store.update("training_runs", queued["id"], {"config": config})

    def unexpected(*_args, **_kwargs):
        pytest.fail("A changed contract must fail before model loading")

    with pytest.raises(ValueError, match="class definitions|class_mapping"):
        run(store, queued, trainer_factory=unexpected)
    assert store.list("trained_models") == []


def test_legacy_queued_run_without_new_snapshot_fields_remains_readable(workspace):
    store = workspace[0]
    queued = queue(workspace)
    config = queued["config"]
    config.pop("taxonomy")
    config.pop("output_class_mapping")
    store.update("training_runs", queued["id"], {"config": config})
    assert run(store, queued)["checkpoint_id"]


def test_custom_head_copies_only_explicit_coco_rows_and_seeds_every_other_row(
    structural_trainer, contract
):
    make, torch, _ = structural_trainer
    parent_state = {}

    def capture(model):
        parent_state.update(
            {
                name: value.detach().clone()
                for name, value in model.roi_heads.box_predictor.named_parameters()
            }
        )

    trainer = make(origin="official", contract=contract, alter=capture)
    head = trainer.model.roi_heads.box_predictor
    assert head.cls_score.out_features == 5 and head.bbox_pred.out_features == 20
    for target, source in ((0, 0), (3, 3), (4, 44)):
        for name, value in head.named_parameters():
            if name.startswith("cls_score"):
                assert torch.equal(value[target], parent_state[name][source])
            else:
                assert torch.equal(
                    value[target * 4 : (target + 1) * 4],
                    parent_state[name][source * 4 : (source + 1) * 4],
                )
    assert not torch.equal(head.cls_score.weight[1], parent_state["cls_score.weight"][1])
    assert not torch.equal(head.cls_score.weight[2], parent_state["cls_score.weight"][2])
    again = make(origin="official", contract=contract)
    for name, value in head.named_parameters():
        assert torch.equal(
            value, dict(again.model.roi_heads.box_predictor.named_parameters())[name]
        )
    assert trainer.metadata["head_initialization_rows"] == {
        "person": None,
        "helmet": None,
        "vehicle": 3,
        "bottle": 44,
    }


def test_custom_targets_and_negative_images_reach_optimizer_with_native_slots(
    structural_trainer, contract, tmp_path
):
    make, torch, _ = structural_trainer
    trainer = make(origin="official", contract=contract)
    trainer.step(
        Image.new("RGB", (8, 8)),
        [
            {"label": "helmet", "box": [1, 1, 4, 5]},
            {"label": "bottle", "box": [4, 1, 7, 6]},
        ],
    )
    assert trainer.model.last_targets[0]["labels"].tolist() == [2, 4]
    trainer.step(Image.new("RGB", (8, 8)), [])
    target = trainer.model.last_targets[0]
    assert list(target["boxes"].shape) == [0, 4]
    assert target["labels"].dtype == torch.int64 and target["labels"].numel() == 0
    path = tmp_path / "custom-state.pth"
    assert trainer.write_checkpoint(path)["head_weights_changed"] is True
    reloaded = make(origin="trained", contract=contract)
    reloaded.model.load_state_dict(torch.load(path, weights_only=True), strict=True)
    for name, value in trainer.model.state_dict().items():
        assert torch.equal(value, reloaded.model.state_dict()[name])


def test_custom_trained_head_is_preserved_and_contract_is_rechecked_in_engine(
    structural_trainer, contract
):
    make, torch, _ = structural_trainer
    before = {}

    def capture(model):
        before.update(
            {
                name: value.clone()
                for name, value in model.roi_heads.box_predictor.state_dict().items()
            }
        )

    trainer = make(origin="trained", contract=contract, alter=capture)
    for name, value in trainer.model.roi_heads.box_predictor.state_dict().items():
        assert torch.equal(value, before[name])
    changed = deepcopy(contract)
    changed["taxonomy"]["classes"][0]["definition"] = "Different parent semantics."
    with pytest.raises(ValueError, match="different class definitions"):
        make(origin="trained", contract=contract, parent_contract=changed)
