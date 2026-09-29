"""Annotation review and provenance using only local, synthetic image fixtures."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
from PIL import Image

from iris.annotations import (
    TAXONOMY,
    AnnotationConflict,
    add_detector_suggestions,
    get_annotation,
    require_revision,
    save_annotation,
)
from iris.media import import_asset
from iris.store import Store, new_id, now


@pytest.fixture
def workspace(tmp_path):
    store = Store(tmp_path / "workspace")
    session = store.insert(
        "sessions",
        {"id": new_id(), "name": "Synthetic flight", "scene_group": "fixture", "created_at": now()},
    )
    frames = []
    for index in range(2):
        path = tmp_path / f"fixture-{index}.png"
        Image.new("RGB", (80, 60), (40 + index, 60, 80)).save(path)
        asset = import_asset(store, session["id"], path, path.name)
        frames.extend(store.list("frames", asset_id=asset["id"]))
    return store, frames


def box(*, suggestion=None, **overrides):
    return {
        "id": "box-one",
        "label": suggestion["label"] if suggestion else "person",
        "box": suggestion["box"] if suggestion else [2, 4, 30, 45],
        "suggestion_id": suggestion["id"] if suggestion else None,
        **overrides,
    }


def save(store, frame, *, expected_revision=0, boxes=None, decisions=None, **options):
    return save_annotation(
        store,
        frame["id"],
        expected_revision=expected_revision,
        boxes=[] if boxes is None else boxes,
        decisions={} if decisions is None else decisions,
        **options,
    )


def prediction(store, frame, *, detections=None):
    created_at = now()
    job = store.insert(
        "jobs",
        {
            "id": new_id(),
            "kind": "infer",
            "status": "succeeded",
            "params": {"fixture": True},
            "created_at": created_at,
        },
    )
    comparison = store.insert(
        "comparisons",
        {
            "id": new_id(),
            "session_id": frame["session_id"],
            "name": "Synthetic predictions",
            "frame_ids": [frame["id"]],
            "model_ids": ["fixture-detector"],
            "config": {
                "taxonomy": "coco-2017-v1",
                "frame_hashes": {frame["id"]: frame["sha256"]},
            },
            "job_id": job["id"],
            "created_at": created_at,
        },
    )
    run = store.insert(
        "runs",
        {
            "id": new_id(),
            "comparison_id": comparison["id"],
            "model_id": "fixture-detector",
            "metadata": {
                "checkpoint": "synthetic-test-only",
                "checkpoint_sha256": "a" * 64,
                "fixture": True,
            },
            "created_at": created_at,
        },
    )
    return store.insert(
        "predictions",
        {
            "id": new_id(),
            "comparison_id": comparison["id"],
            "run_id": run["id"],
            "frame_id": frame["id"],
            "model_id": run["model_id"],
            "detections": detections
            if detections is not None
            else [
                {
                    "label_id": 1,
                    "label": "deliberately-wrong-label",
                    "box": [1, 2, 30, 40],
                    "score": 0.9,
                },
                {"label_id": 3, "label": "car", "box": [15, 12, 70, 58], "score": 0.4},
                {"label_id": 2, "label": "person", "box": [4, 8, 44, 58], "score": 0.99},
            ],
            "timing": {},
            "input_size": [frame["width"], frame["height"]],
            "created_at": created_at,
        },
    )


def suggestions(store, frame, *, threshold=0.5, expected_revision=0):
    source = prediction(store, frame)
    result = add_detector_suggestions(
        store,
        frame["id"],
        prediction_id=source["id"],
        threshold=threshold,
        expected_revision=expected_revision,
    )
    return source, result["suggestions"]


def test_initial_image_is_unannotated_and_public_provenance_is_available(workspace):
    store, (frame, _) = workspace
    annotation = get_annotation(store, frame["id"])
    assert annotation["revision"] == 0
    assert annotation["status"] == "unannotated"
    assert annotation["boxes"] == annotation["suggestions"] == annotation["history"] == []
    assert annotation["decisions"] == {}
    assert annotation["frame_sha256"] == frame["sha256"]
    assert annotation["frame"]["source_filename"] == "fixture-0.png"
    assert annotation["frame"]["scene_group"] == "fixture"
    assert "path" not in annotation["frame"]
    assert annotation["taxonomy"] == TAXONOMY
    assert {item["id"]: item["coco_id"] for item in TAXONOMY["classes"]} == {"person": 1, "car": 3}
    with pytest.raises(KeyError):
        get_annotation(store, "missing")


def test_manual_draft_validation_and_further_edit_keep_immutable_history(workspace):
    store, (frame, _) = workspace
    first = save(store, frame, boxes=[box()], reviewer="  Reviewer  ")
    before = store.list("annotation_revisions", frame_id=frame["id"])
    assert first["status"] == "draft"
    assert first["reviewer"] == "Reviewer"
    assert first["boxes"][0]["review_state"] == "manual"
    assert first["boxes"][0]["source"] == {"kind": "manual"}
    second = save(
        store,
        frame,
        expected_revision=1,
        boxes=first["boxes"],
        status="validated",
        reviewer="Reviewer",
        notes="Synthetic fixture review",
    )
    assert second["status"] == "validated"
    assert second["revision"] == 2
    third = save(store, frame, expected_revision=2, boxes=[box(box=[3, 4, 30, 45])])
    assert third["status"] == "draft"
    assert [item["revision"] for item in third["history"]] == [3, 2, 1]
    assert store.get("annotation_revisions", before[0]["id"]) == before[0]
    assert get_annotation(Store(store.root), frame["id"]) == third


def test_empty_validation_is_explicit_negative_and_requires_named_review(workspace):
    store, (frame, _) = workspace
    with pytest.raises(ValueError, match="reviewer"):
        save(store, frame, status="validated", reviewer=" ")
    assert get_annotation(store, frame["id"])["status"] == "unannotated"
    validated = save(store, frame, status="validated", reviewer="Fixture reviewer")
    assert validated["status"] == "validated" and validated["boxes"] == []
    assert validated["revision"] == 1
    assert validated["history"][0]["box_count"] == 0


@pytest.mark.parametrize(
    "invalid",
    [
        box(id=""),
        box(id=" " * 2),
        box(id="x" * 129),
        box(id=7),
        box(label="truck"),
        box(label=[]),
        box(box=[1, 2, 3]),
        box(box=[1, 2, 3, 4, 5]),
        box(box=[float("nan"), 2, 3, 4]),
        box(box=[1, 2, float("inf"), 4]),
        box(box=[1, 2, 10**400, 4]),
        box(box=[True, 2, 3, 4]),
        box(box=["1", 2, 3, 4]),
        box(box=[1, 2, 1, 4]),
        box(box=[1, 4, 3, 2]),
        box(box=[-1, 2, 3, 4]),
        box(box=[1, 2, 81, 4]),
        box(box=[1, 2, 3, 61]),
        box(suggestion_id="not-this-frame"),
        box(suggestion_id=[]),
        [],
    ],
)
def test_invalid_boxes_never_create_a_revision(workspace, invalid):
    store, (frame, _) = workspace
    with pytest.raises(ValueError):
        save(store, frame, boxes=[invalid])
    assert store.list("annotation_revisions") == []


def test_duplicate_box_ids_and_unbounded_box_counts_are_rejected(workspace):
    store, (frame, _) = workspace
    for boxes in ([box(), box()], [box(id=str(index)) for index in range(501)]):
        with pytest.raises(ValueError):
            save(store, frame, boxes=boxes)
    assert store.list("annotation_revisions") == []


def test_detector_import_has_explicit_mapping_and_never_changes_human_revision(workspace):
    store, (frame, _) = workspace
    source, proposed = suggestions(store, frame)
    assert len(proposed) == 1
    candidate = proposed[0]
    assert candidate["label"] == "person"
    assert candidate["state"] == "pending"
    assert candidate["metadata"]["original_label"] == "deliberately-wrong-label"
    assert candidate["metadata"]["model_metadata"]["checkpoint_sha256"] == "a" * 64
    assert candidate["metadata"]["frame_sha256"] == frame["sha256"]
    assert candidate["metadata"]["prediction_id"] == source["id"]
    assert candidate["metadata"]["score"] == 0.9
    assert store.list("annotation_revisions") == []
    repeated = add_detector_suggestions(
        store, frame["id"], prediction_id=source["id"], threshold=0.5, expected_revision=0
    )
    assert repeated["suggestions"] == proposed
    assert repeated["revision"] == 0 and repeated["boxes"] == []
    assert repeated["prediction_sources"][0]["detection_count"] == 2
    expanded = add_detector_suggestions(
        store, frame["id"], prediction_id=source["id"], threshold=0.1, expected_revision=0
    )
    assert {item["label"] for item in expanded["suggestions"]} == {"person", "car"}
    assert (
        next(item for item in expanded["suggestions"] if item["id"] == candidate["id"]) == candidate
    )


def test_accept_correct_and_reject_preserve_source_and_review_history(workspace):
    store, (frame, _) = workspace
    _, (candidate,) = suggestions(store, frame)
    accepted = save(
        store,
        frame,
        boxes=[box(suggestion=candidate, source={"kind": "forged"}, review_state="manual")],
        decisions={candidate["id"]: "accepted"},
    )
    recorded = accepted["boxes"][0]
    assert recorded["review_state"] == "accepted"
    assert recorded["source"]["kind"] == "detector"
    assert recorded["source"]["metadata"] == candidate["metadata"]
    corrected = save(
        store,
        frame,
        expected_revision=1,
        boxes=[{**recorded, "label": "car", "box": [2, 3, 31, 41]}],
        decisions={candidate["id"]: "corrected"},
        status="validated",
        reviewer="Fixture reviewer",
    )
    assert corrected["boxes"][0]["review_state"] == "corrected"
    assert corrected["suggestions"][0]["state"] == "corrected"
    rejected = save(
        store,
        frame,
        expected_revision=2,
        boxes=[],
        decisions={candidate["id"]: "rejected"},
    )
    assert rejected["boxes"] == [] and rejected["suggestions"][0]["state"] == "rejected"
    revisions = store.list("annotation_revisions", frame_id=frame["id"])
    assert revisions[0]["boxes"][0]["label"] == "person"
    assert revisions[1]["boxes"][0]["source"] == recorded["source"]


def test_multimodal_suggestion_uses_same_human_review_boundary(workspace):
    store, (frame, _) = workspace
    candidate = store.insert(
        "annotation_suggestions",
        {
            "id": new_id(),
            "frame_id": frame["id"],
            "job_id": None,
            "kind": "multimodal",
            "label": "car",
            "box": [4, 6, 35, 50],
            "metadata": {"provider": "fixture-only", "prompt": "Synthetic test prompt"},
            "created_at": now(),
        },
    )
    with pytest.raises(ValueError, match="pending"):
        save(store, frame, status="validated", reviewer="Fixture reviewer")
    result = save(
        store,
        frame,
        boxes=[box(suggestion=candidate)],
        decisions={candidate["id"]: "accepted"},
        status="validated",
        reviewer="Fixture reviewer",
    )
    assert result["boxes"][0]["source"]["kind"] == "multimodal"
    assert result["boxes"][0]["source"]["metadata"]["prompt"] == "Synthetic test prompt"


def test_review_decisions_cannot_be_forged_dropped_or_disconnected_from_boxes(workspace):
    store, (frame, other) = workspace
    _, (candidate,) = suggestions(store, frame)
    _, (foreign,) = suggestions(store, other)
    invalid_snapshots = [
        ([box(suggestion=candidate)], {}),
        ([box(suggestion=candidate)], {candidate["id"]: "rejected"}),
        ([box(suggestion=candidate)], {candidate["id"]: "corrected"}),
        ([box(suggestion=candidate, label="car")], {candidate["id"]: "accepted"}),
        ([], {candidate["id"]: "accepted"}),
        ([], {candidate["id"]: "corrected"}),
        ([], {candidate["id"]: "pending"}),
        ([], {foreign["id"]: "rejected"}),
        ([box(suggestion=foreign)], {foreign["id"]: "accepted"}),
        (
            [box(suggestion=candidate), box(suggestion=candidate, id="duplicate-origin")],
            {candidate["id"]: "accepted"},
        ),
    ]
    for boxes, decisions in invalid_snapshots:
        with pytest.raises(ValueError):
            save(store, frame, boxes=boxes, decisions=decisions)
    accepted = save(
        store,
        frame,
        boxes=[box(suggestion=candidate)],
        decisions={candidate["id"]: "accepted"},
    )
    with pytest.raises(ValueError, match="previous suggestion decisions"):
        save(store, frame, expected_revision=1)
    with pytest.raises(ValueError, match="origin"):
        save(
            store,
            frame,
            expected_revision=1,
            boxes=[{**accepted["boxes"][0], "suggestion_id": None}],
            decisions={candidate["id"]: "rejected"},
        )
    assert len(store.list("annotation_revisions")) == 1


def test_new_proposals_cannot_rewrite_old_validation(workspace):
    store, (frame, _) = workspace
    validated = save(store, frame, status="validated", reviewer="Fixture reviewer")
    old_revision = store.list("annotation_revisions")[0]
    suggestions(store, frame, expected_revision=1)
    current = get_annotation(store, frame["id"])
    assert current["revision"] == validated["revision"] == 1
    assert current["status"] == "validated" and current["boxes"] == []
    assert current["suggestions"][0]["state"] == "pending"
    assert store.get("annotation_revisions", old_revision["id"]) == old_revision
    with pytest.raises(ValueError, match="pending"):
        save(store, frame, expected_revision=1, status="validated", reviewer="Fixture reviewer")


@pytest.mark.parametrize("original_kind", ["manual", "detector"])
def test_multimodal_rereview_can_replace_unchanged_saved_box(workspace, original_kind):
    store, (frame, _) = workspace
    previous_decisions = {}
    if original_kind == "manual":
        original_box = box()
    else:
        _, (candidate,) = suggestions(store, frame)
        original_box = box(suggestion=candidate)
        previous_decisions[candidate["id"]] = "accepted"
    saved = save(store, frame, boxes=[original_box], decisions=previous_decisions)
    candidate = store.insert(
        "annotation_suggestions",
        {
            "id": new_id(),
            "frame_id": frame["id"],
            "job_id": None,
            "kind": "multimodal",
            "label": "car",
            "box": original_box["box"],
            "metadata": {
                "target_box_id": original_box["id"],
                "base_revision": 1,
                "frame_sha256": frame["sha256"],
                "source": {
                    "kind": "annotation",
                    "revision": 1,
                    "target_box_id": original_box["id"],
                },
            },
            "created_at": now(),
        },
    )
    # A save of unrelated notes after the request does not invalidate its target.
    save(
        store,
        frame,
        expected_revision=1,
        boxes=saved["boxes"],
        decisions=previous_decisions,
        notes="Still reviewing the same fixture",
    )
    decisions = {key: "rejected" for key in previous_decisions}
    reviewed = save(
        store,
        frame,
        expected_revision=2,
        boxes=[box(suggestion=candidate)],
        decisions={**decisions, candidate["id"]: "accepted"},
    )
    assert len(reviewed["boxes"]) == 1
    assert reviewed["boxes"][0]["id"] == original_box["id"]
    assert reviewed["boxes"][0]["label"] == "car"
    assert reviewed["boxes"][0]["source"]["kind"] == "multimodal"
    assert store.list("annotation_revisions")[0]["boxes"] == saved["boxes"]


@pytest.mark.parametrize("change", ["label", "box", "revision", "target_id", "frame_hash"])
def test_multimodal_rereview_cannot_replace_a_changed_or_unrelated_box(workspace, change):
    store, (frame, _) = workspace
    saved = save(store, frame, boxes=[box()])
    metadata = {
        "target_box_id": "box-one",
        "base_revision": 1,
        "frame_sha256": frame["sha256"],
        "source": {"kind": "annotation", "revision": 1, "target_box_id": "box-one"},
    }
    if change == "revision":
        metadata["base_revision"] = 2
    elif change == "target_id":
        metadata["target_box_id"] = "unrelated-box"
    elif change == "frame_hash":
        metadata["frame_sha256"] = "b" * 64
    candidate = store.insert(
        "annotation_suggestions",
        {
            "id": new_id(),
            "frame_id": frame["id"],
            "job_id": None,
            "kind": "multimodal",
            "label": "car",
            "box": [2, 4, 30, 45],
            "metadata": metadata,
            "created_at": now(),
        },
    )
    current_box = saved["boxes"][0]
    if change == "label":
        current_box = {**current_box, "label": "car"}
    elif change == "box":
        current_box = {**current_box, "box": [3, 4, 30, 45]}
    save(store, frame, expected_revision=1, boxes=[current_box])
    with pytest.raises(ValueError):
        save(
            store,
            frame,
            expected_revision=2,
            boxes=[box(suggestion=candidate)],
            decisions={candidate["id"]: "accepted"},
        )
    assert len(store.list("annotation_revisions")) == 2


def test_stale_editor_save_or_import_cannot_overwrite_a_newer_revision(workspace):
    store, (frame, _) = workspace
    source = prediction(store, frame)
    saved = save(store, frame, boxes=[box()])
    with pytest.raises(AnnotationConflict):
        save(store, frame)
    with pytest.raises(AnnotationConflict):
        require_revision(store, frame["id"], 0)
    with pytest.raises(AnnotationConflict):
        add_detector_suggestions(
            store, frame["id"], prediction_id=source["id"], expected_revision=0
        )
    assert get_annotation(store, frame["id"]) == saved
    assert require_revision(store, frame["id"], 1) == saved


def test_concurrent_editors_produce_one_revision_and_one_conflict(workspace):
    store, (frame, _) = workspace
    ready = Barrier(2)

    def write():
        ready.wait(timeout=5)
        try:
            return save(store, frame, boxes=[box()])["revision"]
        except AnnotationConflict:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: write(), range(2)))
    assert set(results) == {1, "conflict"}
    assert len(store.list("annotation_revisions")) == 1


@pytest.mark.parametrize("change", ["pixels", "dimensions", "missing", "metadata_hash"])
def test_changed_image_cannot_be_annotated_or_import_predictions(workspace, change):
    store, (frame, _) = workspace
    source = prediction(store, frame)
    save(store, frame, boxes=[box()])
    path = store.artifact_path(frame["path"])
    if change == "metadata_hash":
        store.update("frames", frame["id"], {"sha256": "b" * 64})
    elif change == "missing":
        path.unlink()
    else:
        Image.new("RGB", (80, 60) if change == "pixels" else (10, 10), "red").save(path)
    with pytest.raises((ValueError, OSError)):
        save(store, frame, expected_revision=1)
    with pytest.raises((ValueError, OSError)):
        add_detector_suggestions(
            store, frame["id"], prediction_id=source["id"], expected_revision=1
        )
    assert len(store.list("annotation_revisions")) == 1
    assert store.list("annotation_suggestions") == []


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), float("inf"), True, "0.5"])
def test_invalid_detector_thresholds_are_rejected(workspace, threshold):
    store, (frame, _) = workspace
    source = prediction(store, frame)
    with pytest.raises(ValueError):
        add_detector_suggestions(
            store, frame["id"], prediction_id=source["id"], threshold=threshold, expected_revision=0
        )
    assert store.list("annotation_suggestions") == []


@pytest.mark.parametrize("revision", [-1, 0.0, True, "0"])
def test_expected_revision_is_a_strict_nonnegative_integer(workspace, revision):
    store, (frame, _) = workspace
    with pytest.raises(ValueError):
        save(store, frame, expected_revision=revision)
    assert store.list("annotation_revisions") == []


def test_detector_source_must_match_image_and_compatible_taxonomy(workspace):
    store, (frame, other) = workspace
    source = prediction(store, frame)
    with pytest.raises(ValueError, match="belong"):
        add_detector_suggestions(
            store, other["id"], prediction_id=source["id"], expected_revision=0
        )
    comparison = store.get("comparisons", source["comparison_id"])
    store.update(
        "comparisons",
        comparison["id"],
        {"config": {**comparison["config"], "taxonomy": "different-class-order"}},
    )
    with pytest.raises(ValueError, match="incompatible"):
        add_detector_suggestions(
            store, frame["id"], prediction_id=source["id"], expected_revision=0
        )
    assert get_annotation(store, frame["id"])["prediction_sources"] == []
    assert store.list("annotation_suggestions") == []


def test_large_detector_output_is_rejected_atomically(workspace):
    store, (frame, _) = workspace
    source = prediction(
        store,
        frame,
        detections=[{"label_id": 1, "label": "person", "box": [1, 2, 3, 4], "score": 0.9}] * 101,
    )
    with pytest.raises(ValueError, match="100"):
        add_detector_suggestions(
            store, frame["id"], prediction_id=source["id"], expected_revision=0
        )
    assert store.list("annotation_suggestions") == []
