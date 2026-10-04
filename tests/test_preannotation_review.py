"""Synthetic custom detector evidence and manual decisions; no model is executed."""

import pytest
from test_review_queue import detection, saved_comparison
from test_review_queue import workspace as review_workspace

from iris.annotations import adopt_taxonomy, save_annotation
from iris.model_taxonomy import class_contract
from iris.review_queue import review_queue
from iris.store import new_id, now
from iris.taxonomies import TAXONOMY, publish_taxonomy

workspace = review_workspace


def custom_pair(store, session, frames):
    taxonomy = publish_taxonomy(
        store,
        session["project_id"],
        expected_taxonomy_id=TAXONOMY["id"],
        classes=[
            {"id": "helmet", "name": "Helmet", "definition": "Synthetic helmet."},
            {"id": "bottle", "name": "Bottle", "definition": "Synthetic bottle.", "coco_id": 44},
        ],
    )
    for frame in frames:
        adopt_taxonomy(
            store,
            frame["id"],
            expected_revision=0,
            expected_taxonomy_id=TAXONOMY["id"],
            target_taxonomy_id=taxonomy["id"],
        )
    contract = class_contract(
        {
            "taxonomy": taxonomy,
            "taxonomy_id": taxonomy["id"],
            "class_mapping": {"helmet": 1, "bottle": 2},
            "output_class_mapping": {"helmet": 1, "bottle": 2},
        }
    )
    output = detection(category=2, label="bottle", taxonomy_id=taxonomy["id"])
    comparison = saved_comparison(store, session, frames, outputs=[[output], [output]])
    store.update(
        "comparisons",
        comparison["id"],
        {
            "config": {
                **comparison["config"],
                "taxonomy": "model-specific-v1",
                "model_class_contracts": {
                    model_id: contract for model_id in comparison["model_ids"]
                },
            }
        },
    )
    for run in store.list("runs", comparison_id=comparison["id"]):
        store.update(
            "runs", run["id"], {"metadata": {**run["metadata"], "class_contract": contract}}
        )
    return taxonomy, comparison


def test_custom_frozen_outputs_match_without_using_coco_numeric_ids(workspace):
    store, session, frames = workspace
    taxonomy, comparison = custom_pair(store, session, frames)
    before = {table: store.list(table) for table in store.columns}
    queue = review_queue(store, session["id"], comparison_id=comparison["id"])
    assert queue["frames"][0]["signal"]["status"] == "agreement"
    assert queue["frames"][0]["signal"]["counts"] == [1, 1]
    assert queue["frames"][0]["hints"]["possible_omission"] is False
    assert queue["frames"][0]["review_status"] == "draft"
    assert {table: store.list(table) for table in store.columns} == before
    # Project definitions may change; the image and both outputs remain pinned.
    publish_taxonomy(
        store,
        session["project_id"],
        expected_taxonomy_id=taxonomy["id"],
        classes=[
            *taxonomy["classes"],
            {"id": "other", "name": "Other", "definition": "Unrelated."},
        ],
    )
    assert (
        review_queue(store, session["id"], comparison_id=comparison["id"])["frames"][0]["signal"][
            "status"
        ]
        == "agreement"
    )


@pytest.mark.parametrize("damage", ["definition", "label", "namespace", "unknown_id"])
def test_inconsistent_custom_predictions_never_become_agreement(workspace, damage):
    store, session, frames = workspace
    _, comparison = custom_pair(store, session, frames)
    prediction = store.list("predictions", comparison_id=comparison["id"])[0]
    if damage == "definition":
        run = store.get("runs", prediction["run_id"])
        store.update("runs", run["id"], {"metadata": {**run["metadata"], "class_contract": {}}})
    else:
        output = dict(prediction["detections"][0])
        output.update(
            {
                "label": {"label": "helmet"},
                "namespace": {"taxonomy_id": TAXONOMY["id"]},
                "unknown_id": {"label_id": 44},
            }[damage]
        )
        store.update("predictions", prediction["id"], {"detections": [output]})
    frame = next(
        row
        for row in review_queue(store, session["id"], comparison_id=comparison["id"])["frames"]
        if row["id"] == prediction["frame_id"]
    )
    assert frame["signal"]["status"] == "unavailable"
    assert frame["signal"]["disagreement"] is None


def test_pending_hint_decisions_and_empty_outputs_do_not_validate(workspace):
    store, session, frames = workspace
    comparison = saved_comparison(store, session, frames, outputs=[[], []])
    identifiers = []
    for kind, metadata in [
        ("detector", {"score": 0.3}),
        ("multimodal", {"recommendation": "uncertain"}),
        ("detector", {"score": 0.1, "target_taxonomy": "obsolete"}),
        ("detector", {"score": 0.1, "frame_sha256": "wrong"}),
    ]:
        identifiers.append(new_id())
        store.insert(
            "annotation_suggestions",
            {
                "id": identifiers[-1],
                "frame_id": frames[0]["id"],
                "job_id": comparison["job_id"],
                "kind": kind,
                "label": "person",
                "box": [1, 1, 20, 20],
                "metadata": metadata,
                "created_at": now(),
            },
        )
    queue = review_queue(store, session["id"])
    hints = queue["frames"][0]["hints"]
    assert hints["low_confidence_count"] == hints["uncertain_count"] == 1
    assert hints["possible_omission"] is True
    assert hints["prediction_id"]
    assert queue["frames"][1]["review_status"] == "unannotated"
    assert not store.list("annotation_revisions")
    save_annotation(
        store,
        frames[0]["id"],
        expected_revision=0,
        boxes=[],
        decisions={identifier: "rejected" for identifier in identifiers},
    )
    hints = review_queue(store, session["id"])["frames"][0]["hints"]
    assert hints["low_confidence_count"] == hints["uncertain_count"] == 0
    assert hints["possible_omission"] is True
    assert store.list("annotation_revisions")[0]["status"] == "draft"


def test_partial_class_coverage_cannot_claim_empty_target_output(workspace):
    store, session, frames = workspace
    saved_comparison(store, session, frames, outputs=[[], []])
    taxonomy = publish_taxonomy(
        store,
        session["project_id"],
        expected_taxonomy_id=TAXONOMY["id"],
        classes=[
            {"id": "helmet", "name": "Helmet", "definition": "Unmapped helmet."},
            {"id": "bottle", "name": "Bottle", "definition": "Mapped bottle.", "coco_id": 44},
        ],
    )
    adopt_taxonomy(
        store,
        frames[0]["id"],
        expected_revision=0,
        expected_taxonomy_id=TAXONOMY["id"],
        target_taxonomy_id=taxonomy["id"],
    )
    assert review_queue(store, session["id"])["frames"][0]["hints"]["possible_omission"] is False


def test_unknown_legacy_category_cannot_turn_into_empty_agreement(workspace):
    store, session, frames = workspace
    comparison = saved_comparison(store, session, frames, outputs=[[detection(category=999)], []])
    queue = review_queue(store, session["id"], comparison_id=comparison["id"])
    assert all(row["signal"]["status"] == "unavailable" for row in queue["frames"])


@pytest.mark.parametrize("metadata", [[], None, "corrupt"])
def test_unreadable_proposal_metadata_does_not_block_manual_review(workspace, metadata):
    store, session, frames = workspace
    suggestion = store.insert(
        "annotation_suggestions",
        {
            "id": new_id(),
            "frame_id": frames[0]["id"],
            "kind": "detector",
            "label": "person",
            "box": [1, 1, 20, 20],
            "metadata": metadata if metadata is not None else {},
            "created_at": now(),
        },
    )
    if metadata is None:
        with store.connect() as conn:
            conn.execute(
                "UPDATE annotation_suggestions SET metadata='null' WHERE id=?", (suggestion["id"],)
            )
    row = review_queue(store, session["id"])["frames"][0]
    assert row["pending_count"] == 1
    assert row["hints"]["low_confidence_count"] == 0
    assert row["hints"]["unreadable_proposal_count"] == 1


def test_failed_output_placeholders_do_not_claim_absence(workspace):
    store, session, frames = workspace
    comparison = saved_comparison(store, session, frames, outputs=[[], []])
    for prediction in store.list("predictions", comparison_id=comparison["id"]):
        store.update(
            "predictions",
            prediction["id"],
            {
                "metadata": {
                    "preannotation": {
                        "state": "invalid_output",
                        "reason": "Synthetic malformed output",
                    },
                }
            },
        )
    for row in review_queue(store, session["id"], comparison_id=comparison["id"])["frames"]:
        assert row["signal"]["status"] == "unavailable"
        assert row["hints"]["possible_omission"] is False
        assert row["hints"]["prediction_status"] == "unavailable"
