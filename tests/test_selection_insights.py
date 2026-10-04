"""Selection hints inspect only saved local evidence and never imply model correctness."""

import json
from copy import deepcopy

import pytest
from PIL import Image

from iris import selection_insights as insights
from iris.annotations import adopt_taxonomy, save_annotation
from iris.media import _pixel_hash
from iris.model_taxonomy import class_contract
from iris.projects import create_project
from iris.selection_insights import selection_insights
from iris.store import DEFAULT_PROJECT_ID, Store, new_id, now
from iris.taxonomies import TAXONOMY, publish_taxonomy


@pytest.fixture
def workspace(tmp_path):
    store = Store(tmp_path / "workspace")
    session = add_session(store)
    return store, session


def add_session(store, project_id=DEFAULT_PROJECT_ID):
    return store.insert(
        "sessions",
        {
            "id": new_id(),
            "name": "Selection fixture",
            "scene_group": new_id(),
            "project_id": project_id,
            "created_at": now(),
        },
    )


def add_frame(
    store,
    session,
    *,
    asset=None,
    digest=None,
    dhash="0" * 16,
    timestamp=None,
    index=None,
    taxonomy_id=None,
):
    if asset is None:
        asset = store.insert(
            "assets",
            {
                "id": new_id(),
                "session_id": session["id"],
                "filename": "fixture.png",
                "kind": "image",
                "sha256": new_id() * 2,
                "size_bytes": 100,
                "path": "assets/fixture.png",
                "metadata": {},
                "created_at": now(),
            },
        )
    identifier = new_id()
    path = store.root / "frames" / f"{identifier}.png"
    path.parent.mkdir(exist_ok=True)
    image = Image.new("RGB", (80, 60), tuple(bytes.fromhex(identifier[:6])))
    image.save(path)
    pixel_digest = _pixel_hash(image)
    return store.insert(
        "frames",
        {
            "id": identifier,
            "session_id": session["id"],
            "asset_id": asset["id"],
            "taxonomy_id": taxonomy_id
            or store.get("projects", session["project_id"])["taxonomy_id"],
            "frame_index": index,
            "timestamp_seconds": timestamp,
            "width": 80,
            "height": 60,
            "sha256": digest or pixel_digest,
            "perceptual_hash": dhash,
            "path": f"frames/{identifier}.png",
            "selected": False,
            "created_at": now(),
        },
    )


def save(store, frame, *, boxes=None, revision=0, status="validated", decisions=None):
    return save_annotation(
        store,
        frame["id"],
        expected_revision=revision,
        boxes=boxes or [],
        decisions=decisions or {},
        status=status,
        reviewer="Human fixture reviewer",
    )


def suggestion(store, frame):
    return store.insert(
        "annotation_suggestions",
        {
            "id": new_id(),
            "frame_id": frame["id"],
            "kind": "imported",
            "label": "person",
            "box": [1, 1, 10, 10],
            "metadata": {},
            "created_at": now(),
        },
    )


def detection(*, label="person", category=1, score=0.2, **extra):
    return {"label": label, "label_id": category, "score": score, "box": [1, 1, 10, 10], **extra}


def saved_source(
    store, session, frame, *, detections=None, contract=None, status="succeeded", created_at=None
):
    timestamp = created_at or now()
    job = store.insert(
        "jobs",
        {
            "id": new_id(),
            "kind": "infer",
            "status": status,
            "params": {},
            "created_at": timestamp,
        },
    )
    model = "fixture-model"
    config = {"taxonomy": "coco-2017-v1", "frame_hashes": {frame["id"]: frame["sha256"]}}
    metadata = {"model_id": model}
    if contract:
        config.update(
            taxonomy="model-specific-v1", model_class_contracts={model: deepcopy(contract)}
        )
        metadata["class_contract"] = deepcopy(contract)
    comparison = store.insert(
        "comparisons",
        {
            "id": new_id(),
            "session_id": session["id"],
            "name": "Saved fixture source",
            "frame_ids": [frame["id"]],
            "model_ids": [model],
            "config": config,
            "job_id": job["id"],
            "created_at": timestamp,
        },
    )
    run = store.insert(
        "runs",
        {
            "id": new_id(),
            "comparison_id": comparison["id"],
            "model_id": model,
            "metadata": metadata,
            "created_at": timestamp,
        },
    )
    prediction = store.insert(
        "predictions",
        {
            "id": new_id(),
            "comparison_id": comparison["id"],
            "run_id": run["id"],
            "frame_id": frame["id"],
            "model_id": model,
            "detections": [] if detections is None else detections,
            "input_size": [80, 60],
            "timing": {},
            "created_at": timestamp,
        },
    )
    return prediction, comparison, run


def custom(store, *, classes=None, expected=TAXONOMY["id"]):
    snapshot = publish_taxonomy(
        store,
        DEFAULT_PROJECT_ID,
        expected_taxonomy_id=expected,
        classes=classes
        or [
            {"id": "helmet", "name": "Helmet", "definition": "A protective helmet."},
            {"id": "vehicle", "name": "Vehicle", "definition": "A passenger car.", "coco_id": 3},
        ],
    )
    mapping = {item["id"]: index for index, item in enumerate(snapshot["classes"], 1)}
    return class_contract(
        {
            "taxonomy": snapshot,
            "taxonomy_id": snapshot["id"],
            "class_mapping": mapping,
            "output_class_mapping": mapping,
        }
    )


def rows(store, session):
    result = selection_insights(store, session["id"], session["project_id"])
    return {row["frame_id"]: row for row in result["frames"]}, result


def test_human_validation_is_required_for_positive_and_negative_labels(workspace):
    store, session = workspace
    frames = [add_frame(store, session) for _ in range(5)]
    save(store, frames[1], status="draft")
    save(store, frames[2])
    save(store, frames[3], boxes=[{"id": "p", "label": "person", "box": [1, 1, 10, 10]}])
    save(store, frames[4])
    suggestion(store, frames[4])
    values, result = rows(store, session)
    assert [values[frame["id"]]["review_status"] for frame in frames] == [
        "unannotated",
        "draft",
        "validated",
        "validated",
        "pending_suggestions",
    ]
    assert [values[frame["id"]]["negative"] for frame in frames] == [None, None, True, False, None]
    assert [values[frame["id"]]["positive"] for frame in frames] == [None, None, False, True, None]
    assert values[frames[3]["id"]]["class_counts"] == {"person": 1, "car": 0}
    assert values[frames[4]["id"]]["pending_count"] == 1
    assert result["summary"]["negative_frames"] == result["summary"]["positive_frames"] == 1
    save(store, frames[3], status="draft", revision=1)
    updated, _ = rows(store, session)
    assert updated[frames[3]["id"]]["review_status"] == "draft"
    assert updated[frames[3]["id"]]["negative"] is None
    assert updated[frames[3]["id"]]["box_count"] == 0


def test_resolved_suggestions_do_not_leave_validated_frame_pending(workspace):
    store, session = workspace
    frame = add_frame(store, session)
    proposed = suggestion(store, frame)
    save(store, frame, decisions={proposed["id"]: "rejected"})
    values, _ = rows(store, session)
    assert values[frame["id"]]["review_status"] == "validated"
    assert values[frame["id"]]["negative"] is True


def test_exact_and_similar_hashes_are_inspectable_session_local_hints(workspace):
    store, session = workspace
    first = add_frame(store, session, digest="a" * 64)
    exact = add_frame(store, session, digest="a" * 64)
    near = add_frame(store, session, dhash="000000000000003f")  # Six differing bits.
    far = add_frame(store, session, dhash="f" * 16)
    outside = add_frame(store, add_session(store), digest="a" * 64)
    values, result = rows(store, session)
    assert values[first["id"]]["exact_duplicate_ids"] == [exact["id"]]
    assert values[first["id"]]["similar_frame_ids"] == [near["id"]]
    assert values[near["id"]]["similar_frame_ids"] == [first["id"], exact["id"]]
    assert values[far["id"]]["similar_frame_ids"] == []
    assert outside["id"] not in json.dumps(result)
    assert result["limits"]["dhash_distance"] == 6
    assert result["summary"]["exact_duplicate_frames"] == 2
    assert result["summary"]["similar_frames"] == 3


def test_source_neighbors_follow_capture_time_not_insertion_order(workspace):
    store, session = workspace
    initial = add_frame(store, session, index=30, timestamp=3)
    asset = store.get("assets", initial["asset_id"])
    store.update("assets", asset["id"], {"kind": "video", "filename": "/private/clip.mp4"})
    early = add_frame(store, session, asset=asset, index=10, timestamp=1)
    middle = add_frame(store, session, asset=asset, index=20, timestamp=2)
    other = add_frame(store, session, timestamp=2.1)
    values, _ = rows(store, session)
    assert values[middle["id"]]["neighbor_frame_ids"] == [early["id"], initial["id"]]
    assert values[middle["id"]]["source"] == {
        "asset_id": asset["id"],
        "filename": "clip.mp4",
        "media_kind": "video",
        "frame_index": 20,
        "timestamp_seconds": 2.0,
    }
    assert values[other["id"]]["neighbor_frame_ids"] == []


def test_latest_valid_compatible_source_controls_low_score_and_no_target_hints(workspace):
    store, session = workspace
    frame = add_frame(store, session)
    saved_source(store, session, frame, detections=[detection(score=0.2)])
    latest, _, _ = saved_source(
        store,
        session,
        frame,
        detections=[
            detection(score=0.099),
            detection(score=0.1),
            detection(score=0.499),
            detection(score=0.5),
            detection(label="bicycle", category=2, score=0.2),
        ],
    )
    values, _ = rows(store, session)
    signal = values[frame["id"]]
    assert signal["prediction_source_id"] == latest["id"]
    assert signal["low_confidence_count"] == 2
    assert signal["no_target_predictions"] is False
    assert signal["negative"] is None
    empty, _, _ = saved_source(store, session, frame)
    values, _ = rows(store, session)
    assert values[frame["id"]]["prediction_source_id"] == empty["id"]
    assert values[frame["id"]]["low_confidence_count"] == 0
    assert values[frame["id"]]["no_target_predictions"] is True
    assert values[frame["id"]]["negative"] is None


def test_custom_saved_namespace_is_matched_to_frame_revision_not_current_project(workspace):
    store, session = workspace
    contract = custom(store)
    frame = add_frame(store, session)
    compatible, _, _ = saved_source(
        store,
        session,
        frame,
        contract=contract,
        detections=[
            detection(label="helmet", category=1, taxonomy_id=contract["taxonomy_id"]),
        ],
    )
    classes = deepcopy(contract["taxonomy"]["classes"])
    classes[0]["definition"] = "Changed future definition"
    future = custom(store, classes=classes, expected=contract["taxonomy_id"])
    saved_source(store, session, frame, contract=future)
    values, _ = rows(store, session)
    signal = values[frame["id"]]
    assert signal["taxonomy_id"] == contract["taxonomy_id"] and signal["taxonomy_outdated"]
    assert signal["prediction_source_id"] == compatible["id"]
    assert signal["prediction_sources_inspected"] == 2
    assert signal["low_confidence_count"] == 1
    assert signal["prediction_target_class_ids"] == ["helmet", "vehicle"]
    assert signal["prediction_mapping_complete"] is True
    adopt_taxonomy(
        store,
        frame["id"],
        expected_revision=0,
        expected_taxonomy_id=contract["taxonomy_id"],
        target_taxonomy_id=future["taxonomy_id"],
    )
    values, _ = rows(store, session)
    assert values[frame["id"]]["taxonomy_id"] == future["taxonomy_id"]
    assert values[frame["id"]]["no_target_predictions"] is True


def test_partial_official_mapping_does_not_imply_no_unmapped_custom_targets(workspace):
    store, session = workspace
    custom(store)
    frame = add_frame(store, session)
    saved_source(store, session, frame, detections=[detection()])
    values, _ = rows(store, session)
    signal = values[frame["id"]]
    assert signal["prediction_mapping_complete"] is False
    assert signal["prediction_target_class_ids"] == ["vehicle"]
    assert signal["low_confidence_count"] == 0
    assert signal["no_target_predictions"] is None
    assert signal["negative"] is None


@pytest.mark.parametrize(
    "corruption",
    [
        "score",
        "label",
        "geometry",
        "namespace",
        "input_size",
        "pixels",
        "run",
        "json",
        "oversized",
        "job",
    ],
)
def test_corrupt_sources_never_claim_absence_or_break_gallery(workspace, corruption):
    store, session = workspace
    frame = add_frame(store, session)
    prediction, comparison, run = saved_source(store, session, frame, detections=[detection()])
    if corruption in {"score", "label", "geometry", "namespace"}:
        item = prediction["detections"][0]
        item.update(
            {
                "score": {"score": "bad"},
                "label": {"label": "helmet"},
                "geometry": {"box": [1, 1, 90, 90]},
                "namespace": {"taxonomy_id": "unknown"},
            }[corruption]
        )
        store.update("predictions", prediction["id"], {"detections": [item]})
    elif corruption == "input_size":
        store.update("predictions", prediction["id"], {"input_size": [1, 1]})
    elif corruption == "pixels":
        comparison["config"]["frame_hashes"][frame["id"]] = "f" * 64
        store.update("comparisons", comparison["id"], {"config": comparison["config"]})
    elif corruption == "run":
        store.update("predictions", prediction["id"], {"model_id": "another-model"})
    elif corruption == "json":
        with store.connect() as conn:
            conn.execute(
                "UPDATE predictions SET detections=? WHERE id=?", ("broken", prediction["id"])
            )
    elif corruption == "oversized":
        store.update("predictions", prediction["id"], {"detections": [detection()] * 101})
    else:
        store.update("jobs", comparison["job_id"], {"status": "failed"})
    values, _ = rows(store, session)
    signal = values[frame["id"]]
    assert signal["prediction_signal_status"] == "unavailable"
    assert signal["prediction_source_id"] is None
    assert signal["low_confidence_count"] is signal["no_target_predictions"] is None
    assert signal["negative"] is None


def test_foreign_project_and_cross_session_sources_cannot_leak(workspace):
    store, session = workspace
    frame = add_frame(store, session)
    foreign_project = create_project(store, name="Other project")
    foreign = add_session(store, foreign_project["id"])
    add_frame(store, foreign, digest=frame["sha256"])
    prediction, _, _ = saved_source(store, foreign, frame, detections=[detection()])
    with pytest.raises(KeyError):
        selection_insights(store, foreign["id"])
    with pytest.raises(KeyError):
        selection_insights(store, session["id"], foreign_project["id"])
    values, result = rows(store, session)
    assert prediction["id"] not in json.dumps(result)
    assert values[frame["id"]]["exact_duplicate_ids"] == []
    assert values[frame["id"]]["prediction_signal_status"] == "unavailable"
    assert (
        selection_insights(store, foreign["id"], foreign_project["id"])["project_id"]
        == foreign_project["id"]
    )


def test_limits_cap_frames_similarity_links_and_saved_source_search(workspace, monkeypatch):
    store, session = workspace
    monkeypatch.setattr(insights, "MAX_FRAMES", 4)
    monkeypatch.setattr(insights, "MAX_SIMILARITY_FRAMES", 3)
    monkeypatch.setattr(insights, "MAX_RELATED_IDS", 1)
    monkeypatch.setattr(insights, "MAX_PREDICTION_SOURCES", 2)
    frames = [add_frame(store, session) for _ in range(5)]
    saved_source(store, session, frames[0], detections=[detection()])
    for _ in range(2):
        bad, _, _ = saved_source(store, session, frames[0], detections=[detection(label="wrong")])
    values, result = rows(store, session)
    assert len(values) == 4 and frames[-1]["id"] not in values
    assert result["summary"]["total_frames"] == 5
    assert result["limits"]["frames_truncated"]
    assert result["limits"]["similarity_truncated"]
    assert result["limits"]["similarity_pairs_checked"] == 3
    assert values[frames[0]["id"]]["similar_frame_count"] == 2
    assert len(values[frames[0]["id"]]["similar_frame_ids"]) == 1
    assert values[frames[3]["id"]]["similarity_inspected"] is False
    assert values[frames[0]["id"]]["prediction_sources_truncated"]
    assert values[frames[0]["id"]]["prediction_sources_inspected"] == 2
    assert values[frames[0]["id"]]["prediction_source_id"] is None
    assert len(result["warnings"]) > len(insights.WARNINGS)


def test_inspection_is_read_only_without_original_pixels_or_model_registry(workspace, monkeypatch):
    store, session = workspace
    frame = add_frame(store, session)
    saved_source(store, session, frame, detections=[detection()])
    save(store, frame)
    before = {table: store.list(table) for table in store.columns}
    monkeypatch.setattr(store, "artifact_path", lambda *_: pytest.fail("No artifact reads"))
    monkeypatch.setattr("iris.models.catalog", lambda *_: pytest.fail("No model registry"))
    monkeypatch.setattr(
        "iris.models.TorchvisionDetector", lambda *_: pytest.fail("No model invocation")
    )
    first = selection_insights(store, session["id"])
    assert selection_insights(store, session["id"]) == first
    assert {table: store.list(table) for table in store.columns} == before
    assert not store.get("frames", frame["id"])["selected"]


def test_real_default_bounds_keep_large_session_response_and_pair_work_finite(workspace):
    store, session = workspace
    initial = add_frame(store, session)
    store.update("frames", initial["id"], {"sha256": "0" * 64})
    with store.connect() as conn:
        conn.executemany(
            "INSERT INTO frames (id,session_id,asset_id,taxonomy_id,width,height,sha256,"
            "perceptual_hash,path,created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    f"bounded-{index:04d}",
                    session["id"],
                    initial["asset_id"],
                    TAXONOMY["id"],
                    80,
                    60,
                    f"{index:064x}",
                    "0" * 16,
                    "frames/absent.png",
                    f"9999-{index:04d}",
                )
                for index in range(1, insights.MAX_FRAMES + 1)
            ],
        )
    result = selection_insights(store, session["id"])
    assert len(result["frames"]) == insights.MAX_FRAMES
    assert result["summary"]["total_frames"] == insights.MAX_FRAMES + 1
    inspected = insights.MAX_SIMILARITY_FRAMES
    assert result["limits"]["similarity_pairs_checked"] == inspected * (inspected - 1) // 2
    first = result["frames"][0]
    assert first["similar_frame_count"] == inspected - 1
    assert len(first["similar_frame_ids"]) == insights.MAX_RELATED_IDS
    assert not result["frames"][-1]["similarity_inspected"]
    assert all(len(row["neighbor_frame_ids"]) <= 4 for row in result["frames"])


@pytest.mark.parametrize(
    "field,value", [("frame_sha256", "f" * 64), ("reviewer", " "), ("boxes", "invalid")]
)
def test_corrupt_human_validation_cannot_become_a_negative(workspace, field, value):
    store, session = workspace
    frame = add_frame(store, session)
    annotation = save(store, frame)
    store.update("annotation_revisions", annotation["history"][0]["id"], {field: value})
    values, _ = rows(store, session)
    row = values[frame["id"]]
    assert row["negative"] is row["positive"] is None
    assert row["annotation_valid"] is False
    assert row["review_status"] == "draft"
    assert row["box_count"] is None


def test_corrupt_newest_source_can_fall_back_to_inspectable_older_valid_source(workspace):
    store, session = workspace
    frame = add_frame(store, session)
    earlier, _, _ = saved_source(store, session, frame, detections=[detection()])
    saved_source(store, session, frame, detections=[detection(label="invented")])
    values, _ = rows(store, session)
    signal = values[frame["id"]]
    assert signal["prediction_source_id"] == earlier["id"]
    assert signal["prediction_sources_inspected"] == 2
    assert signal["low_confidence_count"] == 1


@pytest.mark.parametrize("expanded_metadata", [False, True])
def test_legacy_trained_head_cannot_claim_coverage_of_other_coco_classes(
    workspace, expanded_metadata
):
    store, session = workspace
    custom(
        store,
        classes=[
            {"id": "vehicle", "name": "Vehicle", "definition": "A passenger car.", "coco_id": 3},
            {"id": "bottle", "name": "Bottle", "definition": "A bottle.", "coco_id": 44},
        ],
    )
    frame = add_frame(store, session)
    _, _, run = saved_source(store, session, frame)
    metadata = {**run["metadata"], "taxonomy_id": TAXONOMY["id"], "native_to_coco": {1: 1, 2: 3}}
    if expanded_metadata:
        metadata.update(class_contract({"taxonomy_id": TAXONOMY["id"]}))
    store.update("runs", run["id"], {"metadata": metadata})
    values, _ = rows(store, session)
    signal = values[frame["id"]]
    assert signal["prediction_target_class_ids"] == ["vehicle"]
    assert signal["prediction_mapping_complete"] is False
    assert signal["no_target_predictions"] is None
    assert signal["low_confidence_count"] == 0
