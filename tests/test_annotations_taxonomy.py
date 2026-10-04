"""Custom class review, explicit adoption and preserved historical annotation versions."""

import hashlib

import pytest
from PIL import Image
from test_annotations import box, prediction, save
from test_annotations import workspace as annotation_workspace

from iris.annotations import (
    TAXONOMY,
    AnnotationConflict,
    add_detector_suggestions,
    adopt_taxonomy,
    get_annotation,
)
from iris.media import import_asset
from iris.store import DEFAULT_PROJECT_ID, Store, new_id, now
from iris.taxonomies import publish_taxonomy

workspace = annotation_workspace
HELMET = {"id": "helmet", "name": "Helmet", "definition": "A visible protective helmet."}
VEHICLE = {
    "id": "vehicle",
    "name": "Passenger vehicle",
    "definition": "A passenger car; use the visible extent.",
    "coco_id": 3,
}


def publish(store, previous, classes):
    return publish_taxonomy(
        store, DEFAULT_PROJECT_ID, expected_taxonomy_id=previous["id"], classes=classes
    )


@pytest.fixture
def custom_workspace(tmp_path):
    store = Store(tmp_path / "workspace")
    session = store.insert(
        "sessions",
        {
            "id": new_id(),
            "name": "Synthetic inspection",
            "scene_group": "fixture",
            "created_at": now(),
        },
    )
    taxonomy = publish(store, TAXONOMY, [HELMET, VEHICLE])
    frames = []
    for index in range(2):
        path = tmp_path / f"custom-{index}.png"
        Image.new("RGB", (80, 60), (index + 10, 30, 70)).save(path)
        asset = import_asset(store, session["id"], path, path.name)
        frames.append(store.list("frames", asset_id=asset["id"])[0])
    return store, taxonomy, frames


def test_custom_manual_annotations_and_negatives_use_pinned_definitions(custom_workspace):
    store, taxonomy, (frame, negative) = custom_workspace
    assert frame["taxonomy_id"] == taxonomy["id"]
    initial = get_annotation(store, frame["id"])
    assert initial["taxonomy"] == initial["current_taxonomy"] == taxonomy
    assert initial["taxonomy_outdated"] is False
    with pytest.raises(ValueError, match="class version"):
        save(store, frame, boxes=[box(label="person")], taxonomy_id=taxonomy["id"])
    reviewed = save(
        store,
        frame,
        boxes=[box(label="helmet")],
        status="validated",
        reviewer="Fixture reviewer",
        taxonomy_id=taxonomy["id"],
    )
    saved_revision = store.list("annotation_revisions", frame_id=frame["id"])[0]
    assert reviewed["taxonomy_id"] == taxonomy["id"]
    assert reviewed["boxes"][0]["label"] == "helmet"
    empty = save(
        store, negative, status="validated", reviewer="Fixture reviewer", taxonomy_id=taxonomy["id"]
    )
    assert empty["status"] == "validated" and empty["boxes"] == []
    updated = publish(
        store, taxonomy, [{**HELMET, "definition": "A helmet worn on a head."}, VEHICLE]
    )
    historical = get_annotation(store, frame["id"])
    assert historical["taxonomy"] == taxonomy
    assert historical["current_taxonomy"] == updated
    assert historical["taxonomy_outdated"] is True
    assert historical["history"][0]["taxonomy"] == taxonomy
    assert store.get("annotation_revisions", saved_revision["id"]) == saved_revision
    with pytest.raises(AnnotationConflict, match="class version changed"):
        save(store, frame, expected_revision=1, boxes=reviewed["boxes"], taxonomy_id=updated["id"])
    assert store.list("annotation_revisions", frame_id=frame["id"]) == [saved_revision]


def test_adoption_keeps_sources_decisions_and_history_but_requires_fresh_validation(
    custom_workspace,
):
    store, taxonomy, (frame, _) = custom_workspace
    source = prediction(store, frame)
    proposal = add_detector_suggestions(
        store, frame["id"], prediction_id=source["id"], threshold=0.1, expected_revision=0
    )["suggestions"][0]
    reviewed = save(
        store,
        frame,
        boxes=[box(suggestion=proposal)],
        decisions={proposal["id"]: "accepted"},
        status="validated",
        reviewer="Fixture reviewer",
        notes="Original review notes",
        taxonomy_id=taxonomy["id"],
    )
    before = store.list("annotation_revisions", frame_id=frame["id"])[0]
    target = publish(
        store, taxonomy, [HELMET, {**VEHICLE, "definition": "A passenger car or SUV."}]
    )
    adopted = adopt_taxonomy(
        store,
        frame["id"],
        expected_revision=1,
        expected_taxonomy_id=taxonomy["id"],
        target_taxonomy_id=target["id"],
    )
    assert adopted["revision"] == 2 and adopted["status"] == "draft"
    assert adopted["reviewer"] == ""
    assert adopted["notes"] == reviewed["notes"]
    assert adopted["boxes"] == reviewed["boxes"]
    assert adopted["decisions"] == reviewed["decisions"]
    assert adopted["taxonomy"] == target and adopted["taxonomy_outdated"] is False
    assert adopted["suggestions"][0]["taxonomy_outdated"] is True
    assert store.get("frames", frame["id"])["taxonomy_id"] == taxonomy["id"]
    assert store.get("annotation_revisions", before["id"]) == before
    assert [item["taxonomy_id"] for item in adopted["history"]] == [target["id"], taxonomy["id"]]
    with pytest.raises(AnnotationConflict):
        adopt_taxonomy(
            store,
            frame["id"],
            expected_revision=1,
            expected_taxonomy_id=taxonomy["id"],
            target_taxonomy_id=target["id"],
        )
    with pytest.raises(AnnotationConflict):
        save(store, frame, expected_revision=2, taxonomy_id=taxonomy["id"])
    with pytest.raises(ValueError, match="reviewer"):
        save(store, frame, expected_revision=2, status="validated", taxonomy_id=target["id"])
    revalidated = save(
        store,
        frame,
        expected_revision=2,
        boxes=adopted["boxes"],
        decisions=adopted["decisions"],
        status="validated",
        reviewer="Second review",
        taxonomy_id=target["id"],
    )
    assert revalidated["status"] == "validated" and revalidated["revision"] == 3


def test_adoption_rejects_obsolete_target_even_if_image_revision_is_unchanged(custom_workspace):
    store, original, (frame, _) = custom_workspace
    obsolete = publish(store, original, [{**HELMET, "name": "Safety helmet"}, VEHICLE])
    current = publish(store, obsolete, [{**HELMET, "name": "Protective helmet"}, VEHICLE])
    with pytest.raises(AnnotationConflict, match="project's current class version changed"):
        adopt_taxonomy(
            store,
            frame["id"],
            expected_revision=0,
            expected_taxonomy_id=original["id"],
            target_taxonomy_id=obsolete["id"],
        )
    assert store.list("annotation_revisions") == []
    assert get_annotation(store, frame["id"])["current_taxonomy"] == current


def test_first_custom_adoption_cannot_silently_drop_legacy_boxes(workspace):
    store, (frame, _) = workspace
    reviewed = save(store, frame, boxes=[box()], status="validated", reviewer="Fixture reviewer")
    target = publish(store, TAXONOMY, [HELMET])
    with pytest.raises(ValueError, match="explicitly remove those boxes"):
        adopt_taxonomy(
            store,
            frame["id"],
            expected_revision=1,
            expected_taxonomy_id=TAXONOMY["id"],
            target_taxonomy_id=target["id"],
        )
    assert get_annotation(store, frame["id"])["boxes"] == reviewed["boxes"]
    save(store, frame, expected_revision=1, boxes=[])
    result = adopt_taxonomy(
        store,
        frame["id"],
        expected_revision=2,
        expected_taxonomy_id=TAXONOMY["id"],
        target_taxonomy_id=target["id"],
    )
    assert result["revision"] == 3 and result["boxes"] == [] and result["status"] == "draft"


def test_detector_mapping_is_explicit_versioned_and_legacy_pending_can_be_rejected(workspace):
    store, (frame, _) = workspace
    source = prediction(store, frame)
    legacy = add_detector_suggestions(
        store, frame["id"], prediction_id=source["id"], expected_revision=0
    )["suggestions"][0]
    assert legacy["id"] == hashlib.sha256(f"detector:{source['id']}:0".encode()).hexdigest()
    target = publish(
        store,
        TAXONOMY,
        [
            HELMET,
            {"id": "worker", "name": "Worker", "definition": "A visible human.", "coco_id": 1},
        ],
    )
    adopt_taxonomy(
        store,
        frame["id"],
        expected_revision=0,
        expected_taxonomy_id=TAXONOMY["id"],
        target_taxonomy_id=target["id"],
    )
    proposed = add_detector_suggestions(
        store, frame["id"], prediction_id=source["id"], threshold=0, expected_revision=1
    )
    assert proposed["prediction_sources"][0]["detection_count"] == 1
    assert len(proposed["suggestions"]) == 2
    old = next(item for item in proposed["suggestions"] if item["id"] == legacy["id"])
    custom = next(item for item in proposed["suggestions"] if item["id"] != legacy["id"])
    assert old["taxonomy_outdated"] is True
    assert custom["label"] == "worker" and custom["taxonomy_outdated"] is False
    assert custom["metadata"]["target_taxonomy"] == target["id"]
    repeated = add_detector_suggestions(
        store, frame["id"], prediction_id=source["id"], expected_revision=1
    )
    assert repeated["suggestions"] == proposed["suggestions"]
    with pytest.raises(ValueError, match="class version"):
        save(
            store,
            frame,
            expected_revision=1,
            boxes=[box(suggestion=old)],
            decisions={old["id"]: "accepted", custom["id"]: "rejected"},
            taxonomy_id=target["id"],
        )
    result = save(
        store,
        frame,
        expected_revision=1,
        boxes=[box(suggestion=custom)],
        decisions={old["id"]: "rejected", custom["id"]: "accepted"},
        status="validated",
        reviewer="Fixture reviewer",
        taxonomy_id=target["id"],
    )
    assert result["status"] == "validated"
    assert result["boxes"][0]["source"]["metadata"]["original_label_id"] == 1


def test_detector_does_not_guess_mapping_from_custom_class_name(workspace):
    store, (frame, _) = workspace
    target = publish(
        store, TAXONOMY, [{"id": "person", "name": "Person", "definition": "A human."}]
    )
    adopt_taxonomy(
        store,
        frame["id"],
        expected_revision=0,
        expected_taxonomy_id=TAXONOMY["id"],
        target_taxonomy_id=target["id"],
    )
    source = prediction(store, frame)
    result = add_detector_suggestions(
        store, frame["id"], prediction_id=source["id"], expected_revision=1
    )
    assert result["suggestions"] == [] and result["prediction_sources"][0]["detection_count"] == 0
