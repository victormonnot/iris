"""Frozen datasets keep arbitrary class versions, review tokens and historic split guarantees."""

import hashlib
import json
from copy import deepcopy

import pytest
from test_datasets import add_frame

from iris.annotations import TAXONOMY, adopt_taxonomy, save_annotation
from iris.dataset_manifest import manifest_mappings
from iris.datasets import (
    DatasetConflict,
    create_dataset,
    dataset_brief,
    dataset_candidates,
    dataset_detail,
    load_manifest,
)
from iris.store import DEFAULT_PROJECT_ID, Store
from iris.taxonomies import publish_taxonomy

CLASSES = [
    {"id": "helmet", "name": "Safety helmet", "definition": "A visible protective helmet."},
    {"id": "vehicle", "name": "Passenger car", "definition": "A passenger car.", "coco_id": 3},
    {"id": "marker", "name": "Road marker", "definition": "A painted road marker.", "coco_id": 90},
]
SPLITS = {"alpha": "train", "bravo": "val", "charlie": "test"}


@pytest.fixture
def workspace(tmp_path):
    store = Store(tmp_path / "workspace")
    taxonomy = publish_taxonomy(
        store, DEFAULT_PROJECT_ID, expected_taxonomy_id=TAXONOMY["id"], classes=CLASSES
    )
    frames = [
        add_frame(
            store,
            tmp_path,
            group=group,
            color=(index + 40, 50, 70),
            boxes=[{"id": "box", "label": label, "box": [2, 3, 20, 30]}] if label else [],
        )
        for index, (group, label) in enumerate(
            zip(SPLITS, ("helmet", "vehicle", None), strict=True)
        )
    ]
    return store, taxonomy, frames


def tokens(store, frames):
    return {
        frame["id"]: store.list("annotation_revisions", frame_id=frame["id"])[-1]["id"]
        for frame in frames
    }


def freeze(workspace, **overrides):
    store, taxonomy, frames = workspace
    return create_dataset(
        store,
        **{
            "name": "Custom reviewed release",
            "frame_ids": [frame["id"] for frame in frames],
            "splits": SPLITS,
            "taxonomy_id": taxonomy["id"],
            "expected_revisions": tokens(store, frames),
            **overrides,
        },
    )


def next_taxonomy(store, current):
    return publish_taxonomy(
        store,
        DEFAULT_PROJECT_ID,
        expected_taxonomy_id=current["id"],
        classes=[
            {**current["classes"][0], "definition": "A helmet worn on a head."},
            *current["classes"][1:],
        ],
    )


def adopt_and_validate(store, frame, old, new):
    latest = store.list("annotation_revisions", frame_id=frame["id"])[-1]
    adopted = adopt_taxonomy(
        store,
        frame["id"],
        expected_revision=latest["revision"],
        expected_taxonomy_id=old["id"],
        target_taxonomy_id=new["id"],
    )
    save_annotation(
        store,
        frame["id"],
        expected_revision=adopted["revision"],
        taxonomy_id=new["id"],
        boxes=adopted["boxes"],
        decisions=adopted["decisions"],
        status="validated",
        reviewer="Synthetic new-version reviewer",
    )


def rewrite(store, dataset, mutate):
    manifest = deepcopy(dataset["manifest"])
    mutate(manifest)
    raw = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    store.artifact_path(dataset["path"]).write_bytes(raw)
    store.update(
        "dataset_versions", dataset["id"], {"manifest_sha256": hashlib.sha256(raw).hexdigest()}
    )
    return raw


def test_full_custom_snapshot_mappings_zero_counts_and_negative_are_frozen(workspace):
    store, taxonomy, frames = workspace
    dataset = freeze(workspace)
    manifest = load_manifest(store, dataset["id"], verify_images=True)
    assert manifest["schema_version"] == 2 and manifest["taxonomy"] == taxonomy
    mapping = {"helmet": 1, "vehicle": 2, "marker": 3}
    assert manifest["class_mapping"] == manifest["coco_mapping"] == mapping
    assert manifest_mappings(manifest) == (taxonomy, mapping, mapping)
    assert dataset["taxonomy_id"] == taxonomy["id"] and dataset["ml_supported"] is False
    assert "Training and evaluation" in dataset["ml_limitation"]
    assert dataset["summary"]["class_counts"] == {"helmet": 1, "vehicle": 1, "marker": 0}
    assert dataset["summary"]["split_class_counts"]["test"] == dict.fromkeys(mapping, 0)
    assert dataset["summary"]["negative_count"] == 1
    for frozen, frame in zip(manifest["frames"], frames, strict=True):
        original = store.list("annotation_revisions", frame_id=frame["id"])[-1]
        assert frozen["annotation"] == original and frozen["boxes"] == original["boxes"]
        assert frozen["annotation_revision_id"] == original["id"]
    assert dataset_brief(store, store.get("dataset_versions", dataset["id"])) == {
        key: value for key, value in dataset.items() if key != "manifest"
    }


def test_class_changes_reannotation_and_removed_source_do_not_rewrite_release(workspace):
    store, original, frames = workspace
    dataset = freeze(workspace)
    raw = store.artifact_path(dataset["path"]).read_bytes()
    current = next_taxonomy(store, original)
    adopt_and_validate(store, frames[0], original, current)
    store.artifact_path(frames[0]["path"]).unlink()
    reopened = Store(store.root)
    assert load_manifest(reopened, dataset["id"], verify_images=True) == dataset["manifest"]
    assert dataset_detail(reopened, dataset["id"])["taxonomy"] == original
    assert store.artifact_path(dataset["path"]).read_bytes() == raw


def test_candidates_default_to_current_and_allow_explicit_historical_version(workspace):
    store, original, frames = workspace
    current = next_taxonomy(store, original)
    adopt_and_validate(store, frames[0], original, current)
    default = dataset_candidates(store)
    historic = dataset_candidates(store, taxonomy_id=original["id"])
    assert default["taxonomy"] == current and historic["taxonomy"] == original
    assert default["excluded"]["different_taxonomy"] == 2
    assert historic["excluded"]["different_taxonomy"] == 1
    assert [group["scene_group"] for group in default["groups"]] == ["alpha"]
    assert [group["scene_group"] for group in historic["groups"]] == ["bravo", "charlie"]
    assert {version["id"] for version in historic["taxonomies"]} == {
        TAXONOMY["id"],
        original["id"],
        current["id"],
    }
    assert (
        default["groups"][0]["frames"][0]["annotation_revision_id"]
        == tokens(store, frames)[frames[0]["id"]]
    )


def test_omitted_taxonomy_infers_selected_historical_version_even_after_project_change(workspace):
    store, original, _ = workspace
    next_taxonomy(store, original)
    result = freeze(workspace, taxonomy_id=None, expected_revisions=None)
    assert result["taxonomy"] == original


@pytest.mark.parametrize("explicit", [True, False])
def test_mixed_versions_cannot_be_frozen_even_when_ids_and_boxes_match(workspace, explicit):
    store, original, frames = workspace
    current = next_taxonomy(store, original)
    adopt_and_validate(store, frames[0], original, current)
    with pytest.raises(ValueError, match="same saved class version"):
        freeze(workspace, taxonomy_id=original["id"] if explicit else None)
    assert store.list("dataset_versions") == []
    assert not list((store.root / "datasets").iterdir())


def test_parent_requires_same_taxonomy_but_independent_release_keeps_reservations(workspace):
    store, original, frames = workspace
    parent = freeze(workspace)
    current = next_taxonomy(store, original)
    for frame in frames:
        adopt_and_validate(store, frame, original, current)
    with pytest.raises(ValueError, match="independent release"):
        freeze(workspace, taxonomy_id=current["id"], parent_id=parent["id"])
    with pytest.raises(ValueError, match="already reserved"):
        freeze(
            workspace,
            taxonomy_id=current["id"],
            splits={"alpha": "val", "bravo": "train", "charlie": "test"},
        )
    result = freeze(workspace, taxonomy_id=current["id"])
    assert result["parent_id"] is None and result["taxonomy"] == current
    assert len(store.list("dataset_versions")) == 2


@pytest.mark.parametrize("status", ["draft", "validated"])
def test_changed_annotations_fail_revision_tokens_before_eligibility_and_copying(
    workspace, status, monkeypatch
):
    from iris import datasets

    store, taxonomy, frames = workspace
    expected = tokens(store, frames)
    save_annotation(
        store,
        frames[-1]["id"],
        expected_revision=1,
        boxes=[],
        decisions={},
        status=status,
        reviewer="New synthetic reviewer",
        taxonomy_id=taxonomy["id"],
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Stale revision tokens must fail before copying any frozen pixels")

    monkeypatch.setattr(datasets, "_load_verified_frame", forbidden)
    with pytest.raises(DatasetConflict, match="Annotations changed"):
        freeze(workspace, expected_revisions=expected)
    assert store.list("dataset_versions") == []
    assert not list((store.root / "datasets").iterdir())


@pytest.mark.parametrize("change", ["missing", "extra", "invalid"])
def test_revision_token_map_requires_the_exact_selected_set(workspace, change):
    store, _, frames = workspace
    expected = tokens(store, frames)
    if change == "missing":
        expected.pop(frames[0]["id"])
    elif change == "extra":
        expected["not-selected"] = "some-revision"
    else:
        expected[frames[0]["id"]] = 1
    with pytest.raises(ValueError, match="exactly match"):
        freeze(workspace, expected_revisions=expected)
    assert store.list("dataset_versions") == []


def test_custom_labels_not_in_frozen_definition_are_rejected(workspace):
    store, _, frames = workspace
    latest = store.list("annotation_revisions", frame_id=frames[0]["id"])[-1]
    damaged = [{**latest["boxes"][0], "label": "unknown"}]
    with store.connect() as connection:
        connection.execute(
            "UPDATE annotation_revisions SET boxes=? WHERE id=?",
            (json.dumps(damaged), latest["id"]),
        )
    with pytest.raises(ValueError, match="unsupported class"):
        freeze(workspace)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda manifest: manifest.update(schema_version=True),
        lambda manifest: manifest.update(class_mapping={"helmet": 1, "vehicle": 3, "marker": 90}),
        lambda manifest: manifest.update(coco_mapping={"helmet": 1, "vehicle": 3, "marker": 90}),
        lambda manifest: manifest["class_mapping"].update(helmet=True),
        lambda manifest: manifest["taxonomy"]["classes"][0].update(coco_id=True),
        lambda manifest: manifest["taxonomy"]["classes"].append(manifest["taxonomy"]["classes"][0]),
        lambda manifest: manifest["taxonomy"].update(version=True),
        lambda manifest: manifest["frames"][0]["annotation"].update(taxonomy_id=TAXONOMY["id"]),
    ],
)
def test_invalid_rehashed_frozen_class_contract_is_rejected(workspace, mutation):
    store, _, _ = workspace
    dataset = freeze(workspace)
    rewrite(store, dataset, mutation)
    with pytest.raises(ValueError):
        load_manifest(store, dataset["id"])


def test_manifest_mappings_does_not_consult_live_project_or_registry(workspace, monkeypatch):
    store, _, _ = workspace
    dataset = freeze(workspace)

    def forbidden(*args, **kwargs):
        pytest.fail("Frozen class resolution must not consult live workspace state")

    monkeypatch.setattr(store, "connect", forbidden)
    taxonomy, class_mapping, coco_mapping = manifest_mappings(dataset["manifest"])
    assert taxonomy == dataset["taxonomy"]
    assert class_mapping == coco_mapping == {"helmet": 1, "vehicle": 2, "marker": 3}


def test_schema1_builtin_releases_keep_original_bytes_and_split_reservations(tmp_path):
    store = Store(tmp_path / "legacy-workspace")
    frames = [
        add_frame(store, tmp_path, group=group, color=(index + 10, 50, 70))
        for index, group in enumerate(("alpha", "bravo"))
    ]
    dataset = create_dataset(
        store,
        name="Historical release",
        frame_ids=[frame["id"] for frame in frames],
        splits={"alpha": "train", "bravo": "val"},
    )

    def legacy(manifest):
        manifest["schema_version"] = 1
        manifest.pop("coco_mapping")
        manifest.pop("project_id")

    raw = rewrite(store, dataset, legacy)
    manifest = load_manifest(Store(store.root), dataset["id"], verify_images=True)
    assert manifest_mappings(manifest) == (
        TAXONOMY,
        {"person": 1, "car": 2},
        {"person": 1, "car": 3},
    )
    assert store.artifact_path(dataset["path"]).read_bytes() == raw
    with pytest.raises(ValueError, match="already reserved"):
        create_dataset(
            store,
            name="Cannot reshuffle legacy pixels",
            frame_ids=[frame["id"] for frame in frames],
            splits={"alpha": "val", "bravo": "train"},
        )
