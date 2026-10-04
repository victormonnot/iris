"""Review ordering semantics using synthetic images and saved detector fixtures only."""

import json

import pytest
from PIL import Image

from iris.annotations import save_annotation
from iris.datasets import create_dataset
from iris.media import import_asset
from iris.review_queue import PROTOCOL, assess_disagreement, review_queue
from iris.store import Store, new_id, now


def detection(box=None, *, category=1, score=0.9, **extra):
    return {
        "box": [0, 0, 20, 20] if box is None else box,
        "label_id": category,
        "label": "Do not trust this display label",
        "score": score,
        **extra,
    }


def assess(left, right, **options):
    return assess_disagreement(left, right, 80, 60, **options)


def add_session(store, *, group="fixture"):
    return store.insert(
        "sessions",
        {
            "id": new_id(),
            "name": "Synthetic review session",
            "scene_group": group,
            "created_at": now(),
        },
    )


def add_frame(store, session, color, *, selected=True):
    path = store.root / f"fixture-{color}.png"
    Image.new("RGB", (80, 60), (color, 60, 80)).save(path)
    asset = import_asset(store, session["id"], path, path.name)
    (frame,) = store.list("frames", asset_id=asset["id"])
    return store.update("frames", frame["id"], {"selected": selected})


@pytest.fixture
def workspace(tmp_path):
    store = Store(tmp_path / "workspace")
    session = add_session(store)
    frames = [add_frame(store, session, color) for color in (10, 20, 30, 40)]
    return store, session, frames


def saved_comparison(store, session, frames, *, outputs=None):
    job = store.insert(
        "jobs",
        {"id": new_id(), "kind": "infer", "status": "succeeded", "params": {}, "created_at": now()},
    )
    comparison = store.insert(
        "comparisons",
        {
            "id": new_id(),
            "session_id": session["id"],
            "name": "Synthetic detector pair",
            "frame_ids": [frame["id"] for frame in frames],
            "model_ids": ["fixture-left", "fixture-right"],
            "config": {
                "taxonomy": "coco-2017-v1",
                "frame_hashes": {frame["id"]: frame["sha256"] for frame in frames},
            },
            "job_id": job["id"],
            "created_at": now(),
        },
    )
    for side, model_id in enumerate(comparison["model_ids"]):
        run = store.insert(
            "runs",
            {
                "id": new_id(),
                "comparison_id": comparison["id"],
                "model_id": model_id,
                "metadata": {"model_id": model_id, "fixture": True},
                "created_at": now(),
            },
        )
        for frame in frames:
            store.insert(
                "predictions",
                {
                    "id": new_id(),
                    "comparison_id": comparison["id"],
                    "run_id": run["id"],
                    "frame_id": frame["id"],
                    "model_id": model_id,
                    "detections": outputs[side] if outputs is not None else [detection()],
                    "timing": {},
                    "input_size": [80, 60],
                    "created_at": now(),
                },
            )
    return comparison


def save(store, frame, **options):
    return save_annotation(
        store,
        frame["id"],
        **{"expected_revision": 0, "boxes": [], "decisions": {}, **options},
    )


def suggestion(store, frame):
    return store.insert(
        "annotation_suggestions",
        {
            "id": new_id(),
            "frame_id": frame["id"],
            "job_id": None,
            "kind": "imported",
            "label": "person",
            "box": [0, 0, 20, 20],
            "metadata": {"fixture": True},
            "created_at": now(),
        },
    )


def test_identical_boxes_agree_without_becoming_correct_predictions():
    result = assess([detection()], [detection(score=0.7)])
    assert result["status"] == "agreement"
    assert result["disagreement"] == 0
    assert result["counts"] == [1, 1]
    assert result["matched_count"] == 1
    assert result["matches"] == [{"left_index": 0, "right_index": 0, "iou": 1.0}]
    assert result["unmatched_counts"] == [0, 0]
    assert result["class_conflicts"] == 0
    assert not {"accuracy", "confidence", "quality", "validated"} & result.keys()


def test_equal_counts_different_positions_disagree():
    result = assess([detection()], [detection([40, 30, 60, 50])])
    assert result["status"] == "disagreement"
    assert result["counts"] == [1, 1]
    assert result["disagreement"] == 1
    assert result["unmatched_indices"] == [[0], [0]]
    assert result["class_conflicts"] == 0


def test_overlapping_class_conflicts_stay_unmatched_and_never_reuse_a_box():
    result = assess([detection(), detection()], [detection(category=3)])
    assert result["matched_count"] == 0
    assert result["disagreement"] == 1
    assert result["unmatched_counts"] == [2, 1]
    assert result["class_conflicts"] == 1
    assert result["class_conflict_pairs"] == [{"left_index": 0, "right_index": 0, "iou": 1.0}]


def test_duplicate_detections_are_matched_one_to_one():
    result = assess([detection(), detection()], [detection()])
    assert result["matched_count"] == 1
    assert result["disagreement"] == pytest.approx(1 / 3)
    assert result["unmatched_counts"] == [1, 0]
    assert result["unmatched_indices"] == [[1], []]


def test_all_empty_is_unknown_and_one_empty_is_complete_disagreement():
    empty = assess([], [])
    assert empty["status"] == "no_detections"
    assert empty["disagreement"] is None
    assert empty["counts"] == [0, 0]
    assert empty["matched_count"] == empty["class_conflicts"] == 0
    assert empty["unmatched_counts"] == [0, 0]
    assert assess([detection()], [])["disagreement"] == 1
    assert assess([], [detection()])["disagreement"] == 1


def test_filtering_preserves_original_indices_and_uses_canonical_ids():
    left = [
        detection(category=2, native_label_id=1),
        detection(score=0.499),
        detection(category=3, native_label_id=2, score=0.5),
        detection([40, 30, 60, 50]),
    ]
    right = [detection(score=0.2), detection(category=3, native_label_id=2)]
    result = assess(left, right)
    assert result["counts"] == [2, 1]
    assert result["matches"] == [{"left_index": 2, "right_index": 1, "iou": 1}]
    assert result["unmatched_indices"] == [[3], []]
    assert assess([detection(category=2)], [detection(category=2)])["status"] == "no_detections"


def test_iou_cutoff_is_inclusive_and_box_edges_have_no_intersection():
    left = [detection([0, 0, 20, 20])]
    right = [detection([0, 0, 10, 20])]
    assert assess(left, right)["matches"][0]["iou"] == 0.5
    assert assess(left, right, iou_threshold=0.50001)["disagreement"] == 1
    touching = [detection([20, 0, 40, 20])]
    assert assess(left, touching, iou_threshold=0.001)["disagreement"] == 1


def test_augmenting_paths_avoid_greedy_trap_and_reverse_ratio_is_identical():
    # All three eligible edges have IoU 0.5. A greedy tie starting with
    # (left 0, right 0) would strand left 1 despite a complete matching.
    left = [detection([0, 0, 20, 20]), detection([10, 0, 30, 20])]
    right = [detection([10, 0, 20, 20]), detection([0, 0, 10, 20])]
    forward = assess(left, right)
    reverse = assess(right, left)
    assert forward["matches"] == [
        {"left_index": 0, "right_index": 1, "iou": 0.5},
        {"left_index": 1, "right_index": 0, "iou": 0.5},
    ]
    assert forward["disagreement"] == reverse["disagreement"] == 0
    assert forward["matched_count"] == reverse["matched_count"] == 2
    assert assess(left, right) == forward


def test_maximum_cardinality_conflicts_are_separate_from_same_class_matches():
    left = [detection([0, 0, 20, 20]), detection([10, 0, 30, 20])]
    right = [detection([10, 0, 20, 20], category=3), detection([0, 0, 10, 20], category=3)]
    result = assess(left, right)
    assert result["class_conflicts"] == 2
    assert result["matched_count"] == 0
    assert result["unmatched_counts"] == [2, 2]
    assert result["disagreement"] == 1


@pytest.mark.parametrize(
    "threshold", [-1, 1.1, True, "0.5", None, float("nan"), float("inf"), 10**400]
)
def test_invalid_confidence_threshold_rejected(threshold):
    with pytest.raises(ValueError, match="Confidence threshold"):
        assess([], [], confidence_threshold=threshold)


@pytest.mark.parametrize("threshold", [0, -1, 1.1, True, "0.5", None, float("nan"), float("inf")])
def test_invalid_iou_threshold_rejected(threshold):
    with pytest.raises(ValueError, match="IoU threshold"):
        assess([], [], iou_threshold=threshold)


@pytest.mark.parametrize("dimensions", [(0, 60), (80, -1), (True, 60), (80.0, 60)])
def test_invalid_dimensions_rejected(dimensions):
    with pytest.raises(ValueError, match="dimensions"):
        assess_disagreement([], [], *dimensions)


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        {},
        [None],
        [detection()] * 101,
        [detection(category=True)],
        [detection(category=0)],
        [detection(category=12)],
        [detection(category=91)],
        [detection(category="1")],
        [detection(score=True)],
        [detection(score="0.9")],
        [detection(score=float("nan"))],
        [detection(score=float("inf"))],
        [detection(score=10**400)],
        [detection(score=-0.1)],
        [detection(score=1.1)],
        [detection([0, 0, 0, 20])],
        [detection([-1, 0, 20, 20])],
        [detection([0, 0, 81, 20])],
        [detection([0, 0, 20, 61])],
        [detection([0, 0, 20])],
        [detection([0, 0, 20, float("nan")])],
        [detection([0, 0, 20, True])],
        [detection([0, 0, 20, "20"])],
        [detection([0, 0, 20, 10**400])],
        [detection([0, 0, 1e-300, 1e-300])],
    ],
)
def test_malformed_saved_detections_never_become_empty_or_agreement(invalid):
    with pytest.raises(ValueError):
        assess(invalid, [])
    with pytest.raises(ValueError):
        assess([], invalid)


def test_invalid_filtered_out_detections_still_reject_corrupt_record():
    with pytest.raises(ValueError):
        assess([detection([-1, 0, 20, 20], category=2, score=0.01)], [])


def test_hundred_native_detections_have_bounded_deterministic_matching():
    result = assess([detection()] * 100, [detection()] * 100)
    assert result["matched_count"] == 100
    assert result["disagreement"] == 0
    assert len({match["left_index"] for match in result["matches"]}) == 100
    assert len({match["right_index"] for match in result["matches"]}) == 100


def test_selected_source_order_and_empty_session_are_preserved(workspace):
    store, session, frames = workspace
    store.update("frames", frames[1]["id"], {"selected": False})
    result = review_queue(store, session["id"])
    assert [frame["id"] for frame in result["frames"]] == [
        frames[index]["id"] for index in (0, 2, 3)
    ]
    assert result["counts"] == {
        "total": 3,
        "needs_review": 3,
        "unannotated": 3,
        "draft": 0,
        "pending": 0,
        "validated": 0,
    }
    assert result["comparison"] is None
    assert result["taxonomy_id"] == "iris-objects-v1"
    assert result["config"]["protocol"] == PROTOCOL
    assert "Maximum-cardinality" in result["config"]["matching"]
    assert len(result["warnings"]) == 4
    for frame in result["frames"]:
        assert frame["source_filename"].startswith("fixture-")
        assert frame["revision"] == frame["box_count"] == frame["pending_count"] == 0
        assert frame["signal"]["status"] == "unavailable"
        assert frame["signal"]["counts"] is None
        assert frame["reserved_split"] is None
        assert "path" not in frame
    empty_session = add_session(store, group="empty")
    empty = review_queue(store, empty_session["id"])
    assert empty["frames"] == []
    assert set(empty["counts"].values()) == {0}


def test_latest_human_status_includes_drafts_negatives_and_new_pending_proposals(workspace):
    store, session, frames = workspace
    save(store, frames[1], boxes=[{"id": "fixture-box", "label": "person", "box": [0, 0, 20, 20]}])
    save(store, frames[2], status="validated", reviewer="Synthetic reviewer")
    save(store, frames[3], status="validated", reviewer="Synthetic reviewer")
    suggestion(store, frames[3])
    result = review_queue(store, session["id"])
    assert [frame["review_status"] for frame in result["frames"]] == [
        "unannotated",
        "draft",
        "validated",
        "pending",
    ]
    assert result["counts"] == {
        "total": 4,
        "needs_review": 3,
        "unannotated": 1,
        "draft": 1,
        "pending": 1,
        "validated": 1,
    }
    assert result["frames"][1]["box_count"] == 1
    assert result["frames"][2]["box_count"] == 0
    assert result["frames"][3]["annotation_status"] == "validated"
    assert result["frames"][3]["pending_count"] == 1
    save(store, frames[2], expected_revision=1)
    updated = review_queue(store, session["id"])
    assert updated["frames"][2]["review_status"] == "draft"
    assert updated["frames"][2]["revision"] == 2
    assert updated["counts"]["needs_review"] == 4


def test_pending_is_exclusive_and_only_unresolved_suggestions_count(workspace):
    store, session, frames = workspace
    first = suggestion(store, frames[0])
    second = suggestion(store, frames[0])
    save(store, frames[0], decisions={first["id"]: "rejected"})
    result = review_queue(store, session["id"])
    assert result["frames"][0]["review_status"] == "pending"
    assert result["frames"][0]["annotation_status"] == "draft"
    assert result["frames"][0]["pending_count"] == 1
    save(
        store,
        frames[0],
        expected_revision=1,
        decisions={first["id"]: "rejected", second["id"]: "rejected"},
    )
    assert review_queue(store, session["id"])["frames"][0]["review_status"] == "draft"
    suggestion(store, frames[1])
    pending_without_revision = review_queue(store, session["id"])["frames"][1]
    assert pending_without_revision["review_status"] == "pending"
    assert pending_without_revision["annotation_status"] == "unannotated"


def test_comparison_pairs_saved_records_and_does_not_use_human_labels(workspace):
    store, session, frames = workspace
    comparison = saved_comparison(store, session, frames[:2])
    before = review_queue(store, session["id"], comparison_id=comparison["id"])
    assert before["comparison"]["models"] == [
        {"id": "fixture-left", "name": "fixture-left"},
        {"id": "fixture-right", "name": "fixture-right"},
    ]
    assert [frame["signal"]["status"] for frame in before["frames"]] == [
        "agreement",
        "agreement",
        "unavailable",
        "unavailable",
    ]
    for frame in before["frames"][:2]:
        expected = [
            store.list("predictions", frame_id=frame["id"], model_id=model_id)[0]["id"]
            for model_id in comparison["model_ids"]
        ]
        assert frame["signal"]["prediction_ids"] == expected
    save(store, frames[0], boxes=[{"id": "human-box", "label": "car", "box": [40, 30, 70, 50]}])
    save(store, frames[1], status="validated", reviewer="Synthetic reviewer")
    after = review_queue(store, session["id"], comparison_id=comparison["id"])
    assert [frame["signal"] for frame in before["frames"]] == [
        frame["signal"] for frame in after["frames"]
    ]


def test_thresholds_change_only_signal_not_review_progress(workspace):
    store, session, frames = workspace
    comparison = saved_comparison(
        store, session, frames, outputs=[[detection(score=0.8)], [detection(score=0.6)]]
    )
    low = review_queue(
        store, session["id"], comparison_id=comparison["id"], confidence_threshold=0.5
    )
    medium = review_queue(
        store, session["id"], comparison_id=comparison["id"], confidence_threshold=0.7
    )
    high = review_queue(
        store, session["id"], comparison_id=comparison["id"], confidence_threshold=0.9
    )
    assert low["frames"][0]["signal"]["status"] == "agreement"
    assert medium["frames"][0]["signal"]["status"] == "disagreement"
    assert high["frames"][0]["signal"]["status"] == "no_detections"
    assert low["counts"] == medium["counts"] == high["counts"]


@pytest.mark.parametrize("status", ["queued", "running", "failed", "cancelled", "interrupted"])
def test_incomplete_comparison_rejected(workspace, status):
    store, session, frames = workspace
    comparison = saved_comparison(store, session, frames)
    store.update("jobs", comparison["job_id"], {"status": status})
    with pytest.raises(ValueError, match="completed"):
        review_queue(store, session["id"], comparison_id=comparison["id"])


@pytest.mark.parametrize(
    "models",
    [[], ["fixture-left"], ["fixture-left", "fixture-left"], ["a", "b", "c"], [1, "b"], None],
)
def test_comparison_requires_two_distinct_model_ids(workspace, models):
    store, session, frames = workspace
    comparison = saved_comparison(store, session, frames)
    if models is not None:
        store.update("comparisons", comparison["id"], {"model_ids": models})
    else:
        with store.connect() as conn:
            conn.execute("UPDATE comparisons SET model_ids='null' WHERE id=?", (comparison["id"],))
    with pytest.raises(ValueError, match="two distinct"):
        review_queue(store, session["id"], comparison_id=comparison["id"])


def test_missing_cross_session_and_other_taxonomy_comparisons_rejected(workspace):
    store, session, frames = workspace
    with pytest.raises(KeyError):
        review_queue(store, "missing")
    with pytest.raises(KeyError):
        review_queue(store, session["id"], comparison_id="missing")
    comparison = saved_comparison(store, session, frames)
    another = add_session(store, group="another")
    with pytest.raises(ValueError, match="this session"):
        review_queue(store, another["id"], comparison_id=comparison["id"])
    store.update("comparisons", comparison["id"], {"config": {"taxonomy": "unknown"}})
    with pytest.raises(ValueError, match="COCO"):
        review_queue(store, session["id"], comparison_id=comparison["id"])


@pytest.mark.parametrize(
    "damage,reason",
    [
        ("missing_prediction", "missing"),
        ("missing_run", "missing"),
        ("wrong_run", "model run"),
        ("wrong_dimensions", "dimensions"),
        ("wrong_frame_hash", "hash"),
        ("missing_frame_hash", "hash"),
        ("invalid_detections", "cannot be compared"),
        ("invalid_run_metadata", "metadata"),
    ],
)
def test_missing_stale_or_malformed_records_are_unavailable(workspace, damage, reason):
    store, session, frames = workspace
    comparison = saved_comparison(store, session, frames)
    prediction = store.list("predictions", frame_id=frames[0]["id"], model_id="fixture-left")[0]
    if damage in {"missing_prediction", "missing_run"}:
        with store.connect() as conn:
            conn.execute("DELETE FROM predictions WHERE id=?", (prediction["id"],))
            if damage == "missing_run":
                conn.execute("DELETE FROM predictions WHERE run_id=?", (prediction["run_id"],))
                conn.execute("DELETE FROM runs WHERE id=?", (prediction["run_id"],))
    elif damage == "wrong_run":
        other_comparison = saved_comparison(store, session, frames[:1])
        other_run = store.list("runs", comparison_id=other_comparison["id"])[0]
        with store.connect() as conn:
            conn.execute("DELETE FROM predictions WHERE run_id=?", (other_run["id"],))
        store.update("predictions", prediction["id"], {"run_id": other_run["id"]})
    elif damage == "wrong_dimensions":
        store.update("predictions", prediction["id"], {"input_size": [60, 80]})
    elif damage == "wrong_frame_hash":
        store.update("frames", frames[0]["id"], {"sha256": "a" * 64})
    elif damage == "missing_frame_hash":
        store.update("comparisons", comparison["id"], {"config": {"taxonomy": "coco-2017-v1"}})
    elif damage == "invalid_detections":
        store.update("predictions", prediction["id"], {"detections": [detection(score="0.9")]})
    elif damage == "invalid_run_metadata":
        store.update("runs", prediction["run_id"], {"metadata": {"model_id": "another-model"}})
    result = review_queue(store, session["id"], comparison_id=comparison["id"])
    signal = result["frames"][0]["signal"]
    assert signal["status"] == "unavailable"
    assert reason in signal["reason"]
    assert signal["disagreement"] is None
    assert signal["counts"] is None
    assert signal["prediction_ids"] == []


def test_imported_split_reservations_apply_to_groups_and_duplicate_pixels(workspace):
    store, session, frames = workspace
    asset = store.get("assets", frames[0]["asset_id"])
    store.update(
        "assets",
        asset["id"],
        {"metadata": {**asset["metadata"], "dataset_import": {"source_split": "test"}}},
    )
    assert {frame["reserved_split"] for frame in review_queue(store, session["id"])["frames"]} == {
        "test"
    }
    another = add_session(store, group="duplicate pixels")
    add_frame(store, another, 10)
    assert review_queue(store, another["id"])["frames"][0]["reserved_split"] == "test"


def test_project_group_reservations_and_global_pixel_reservations(workspace):
    store, original, frames = workspace
    project = store.insert(
        "projects",
        {
            "id": new_id(),
            "name": "Second synthetic project",
            "description": "Independent review queue",
            "taxonomy_id": "iris-objects-v1",
            "created_at": now(),
        },
    )
    other = add_session(store, group=original["scene_group"])
    store.update("sessions", other["id"], {"project_id": project["id"]})
    other_frames = [add_frame(store, other, color) for color in (70, 80)]
    for frame, split in ((frames[0], "train"), (other_frames[0], "val")):
        store.update(
            "assets", frame["asset_id"], {"metadata": {"dataset_import": {"source_split": split}}}
        )
    assert {frame["reserved_split"] for frame in review_queue(store, original["id"])["frames"]} == {
        "train"
    }
    assert {frame["reserved_split"] for frame in review_queue(store, other["id"])["frames"]} == {
        "val"
    }
    duplicate_session = add_session(store, group="duplicate-in-other-project")
    store.update("sessions", duplicate_session["id"], {"project_id": project["id"]})
    add_frame(store, duplicate_session, 10)
    duplicate_queue = review_queue(store, duplicate_session["id"])
    assert duplicate_queue["frames"][0]["reserved_split"] == "train"


def test_frozen_splits_remain_reserved_after_new_draft(workspace):
    store, session, frames = workspace
    validation = add_session(store, group="validation")
    validation_frame = add_frame(store, validation, 80)
    for frame in [frames[0], validation_frame]:
        save(store, frame, status="validated", reviewer="Synthetic reviewer")
    create_dataset(
        store,
        name="Synthetic release",
        frame_ids=[frames[0]["id"], validation_frame["id"]],
        splits={"fixture": "train", "validation": "val"},
    )
    save(store, frames[0], expected_revision=1)
    result = review_queue(store, session["id"])
    assert result["frames"][0]["review_status"] == "draft"
    assert {frame["reserved_split"] for frame in result["frames"]} == {"train"}
    assert review_queue(store, validation["id"])["frames"][0]["reserved_split"] == "val"


def test_conflicting_imported_splits_are_reported(workspace):
    store, session, frames = workspace
    for frame, split in zip(frames[:2], ["train", "test"], strict=True):
        store.update(
            "assets", frame["asset_id"], {"metadata": {"dataset_import": {"source_split": split}}}
        )
    with pytest.raises(ValueError, match="conflicting"):
        review_queue(store, session["id"])


def test_queue_uses_one_snapshot_even_when_other_connection_changes_selection(
    workspace, monkeypatch
):
    from iris import review_queue as module

    store, session, frames = workspace
    actual = module._reservations

    def concurrent_change(store, conn, project_id):
        store.update("frames", frames[0]["id"], {"selected": False})
        return actual(store, conn, project_id)

    monkeypatch.setattr(module, "_reservations", concurrent_change)
    result = review_queue(store, session["id"])
    assert result["counts"]["total"] == 4
    assert result["frames"][0]["id"] == frames[0]["id"]
    assert store.get("frames", frames[0]["id"])["selected"] is False


def test_queue_does_not_mutate_database_or_open_source_images(workspace, monkeypatch):
    store, session, frames = workspace
    comparison = saved_comparison(store, session, frames)
    suggestion(store, frames[0])
    with store.connect() as conn:
        before = "\n".join(conn.iterdump())

    def forbidden(*args, **kwargs):
        pytest.fail("Review queue must not open pixels or import weights")

    monkeypatch.setattr(Image, "open", forbidden)
    result = review_queue(store, session["id"], comparison_id=comparison["id"])
    json.dumps(result, allow_nan=False)
    with store.connect() as conn:
        after = "\n".join(conn.iterdump())
    assert before == after
    assert review_queue(Store(store.root), session["id"], comparison_id=comparison["id"]) == result
