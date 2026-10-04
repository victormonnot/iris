"""Dataset release integrity and split isolation using synthetic image fixtures only."""

import hashlib
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
from PIL import Image

from iris.annotations import save_annotation
from iris.datasets import create_dataset, dataset_candidates, dataset_detail, load_manifest
from iris.media import import_asset
from iris.store import Store, new_id, now


def add_frame(store, tmp_path, *, group, color, selected=True, validated=True, boxes=None):
    session = store.insert(
        "sessions",
        {
            "id": new_id(),
            "name": f"Synthetic {group}",
            "scene_group": group,
            "created_at": now(),
        },
    )
    path = tmp_path / f"{new_id()}.png"
    Image.new("RGB", (80, 60), color).save(path)
    asset = import_asset(store, session["id"], path, path.name)
    frame = store.list("frames", asset_id=asset["id"])[0]
    store.update("frames", frame["id"], {"selected": selected})
    if validated:
        save_annotation(
            store,
            frame["id"],
            expected_revision=0,
            boxes=boxes or [],
            decisions={},
            status="validated",
            reviewer="Synthetic fixture reviewer",
            notes="Test-only labels",
        )
    return store.get("frames", frame["id"])


@pytest.fixture
def workspace(tmp_path):
    store = Store(tmp_path / "workspace")
    frames = [
        add_frame(
            store,
            tmp_path,
            group="alpha",
            color=(30, 60, 80),
            boxes=[
                {
                    "id": "person-box",
                    "label": "person",
                    "box": [2, 3, 24, 45],
                }
            ],
        ),
        add_frame(
            store,
            tmp_path,
            group="bravo",
            color=(40, 70, 90),
            boxes=[
                {
                    "id": "car-box",
                    "label": "car",
                    "box": [10, 15, 60, 40],
                }
            ],
        ),
        add_frame(store, tmp_path, group="charlie", color=(50, 80, 100)),
    ]
    return store, frames


def freeze(store, frames, **overrides):
    return create_dataset(
        store,
        **{
            "name": "Synthetic release",
            "frame_ids": [frame["id"] for frame in frames],
            "splits": {"alpha": "train", "bravo": "val", "charlie": "test"},
            **overrides,
        },
    )


def test_snapshot_copies_pixels_labels_negatives_and_complete_provenance(workspace):
    store, frames = workspace
    result = freeze(store, frames)
    manifest = load_manifest(store, result["id"], verify_images=True)
    assert result["manifest"] == manifest
    assert manifest["schema_version"] == 2
    assert manifest["class_mapping"] == {"person": 1, "car": 2}
    assert manifest["coco_mapping"] == {"person": 1, "car": 3}
    assert result["ml_supported"] is True and result["ml_limitation"] is None
    assert manifest["taxonomy"]["id"] == "iris-objects-v1"
    assert result["summary"]["split_counts"] == {"train": 1, "val": 1, "test": 1}
    assert result["summary"]["class_counts"] == {"person": 1, "car": 1}
    assert result["summary"]["negative_count"] == 1
    assert result["summary"]["near_duplicate_cross_split_pairs"] == 3
    frozen = manifest["frames"][0]
    original = frames[0]
    revision = store.list("annotation_revisions", frame_id=original["id"])[0]
    asset = store.get("assets", original["asset_id"])
    assert frozen["annotation"] == revision
    assert frozen["boxes"] == revision["boxes"]
    assert frozen["annotation_revision_id"] == revision["id"]
    assert frozen["sha256"] == original["sha256"]
    assert frozen["source"] == {
        "asset_id": asset["id"],
        "filename": asset["filename"],
        "kind": "image",
        "sha256": asset["sha256"],
        "metadata": asset["metadata"],
        "frame_index": original["frame_index"],
        "timestamp_seconds": original["timestamp_seconds"],
        "extraction": original["extraction"],
    }
    assert store.artifact_path(frozen["image_path"]) != store.artifact_path(original["path"])
    raw = store.artifact_path(result["path"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == result["manifest_sha256"]
    assert not list((store.root / "datasets").glob(".staging-*"))


def test_later_edits_and_source_removal_do_not_mutate_release(workspace):
    store, frames = workspace
    first = freeze(store, frames)
    saved_bytes = store.artifact_path(first["path"]).read_bytes()
    save_annotation(
        store,
        frames[0]["id"],
        expected_revision=1,
        boxes=[],
        decisions={},
        status="validated",
        reviewer="Second synthetic review",
    )
    second = freeze(store, frames, parent_id=first["id"])
    assert second["parent_id"] == first["id"]
    assert second["manifest"]["frames"][0]["revision"] == 2
    assert second["summary"]["negative_count"] == 2
    store.artifact_path(frames[0]["path"]).unlink()
    assert load_manifest(store, first["id"], verify_images=True)["frames"][0]["revision"] == 1
    assert load_manifest(store, second["id"], verify_images=True)["frames"][0]["boxes"] == []
    assert store.artifact_path(first["path"]).read_bytes() == saved_bytes


def test_candidates_exclude_unreviewed_drafts_and_new_pending_proposals(workspace, tmp_path):
    store, frames = workspace
    add_frame(store, tmp_path, group="unselected", color=(10, 20, 30), selected=False)
    add_frame(store, tmp_path, group="unannotated", color=(11, 20, 30), validated=False)
    draft = add_frame(store, tmp_path, group="draft", color=(12, 20, 30))
    save_annotation(store, draft["id"], expected_revision=1, boxes=[], decisions={})
    store.insert(
        "annotation_suggestions",
        {
            "id": new_id(),
            "frame_id": frames[1]["id"],
            "job_id": None,
            "kind": "detector",
            "label": "person",
            "box": [2, 3, 24, 45],
            "metadata": {},
            "created_at": now(),
        },
    )
    candidates = dataset_candidates(store)
    assert [group["scene_group"] for group in candidates["groups"]] == ["alpha", "charlie"]
    assert candidates["excluded"] == {
        "unselected": 1,
        "unannotated": 1,
        "draft": 1,
        "pending_suggestions": 1,
        "different_taxonomy": 0,
    }
    with pytest.raises(ValueError, match="pending_suggestions"):
        freeze(store, frames)
    assert store.list("dataset_versions") == []
    assert list((store.root / "datasets").iterdir()) == []


def test_latest_draft_cannot_fall_back_to_previous_validation(workspace):
    store, frames = workspace
    save_annotation(store, frames[0]["id"], expected_revision=1, boxes=[], decisions={})
    with pytest.raises(ValueError, match="draft"):
        freeze(store, frames)


def test_unselected_or_unannotated_frames_cannot_be_frozen(workspace, tmp_path):
    store, frames = workspace
    store.update("frames", frames[0]["id"], {"selected": False})
    with pytest.raises(ValueError, match="unselected"):
        freeze(store, frames)
    unreviewed = add_frame(store, tmp_path, group="alpha", color=(99, 20, 30), validated=False)
    with pytest.raises(ValueError, match="unannotated"):
        freeze(store, [unreviewed, *frames[1:]])


@pytest.mark.parametrize(
    "splits",
    [
        {"alpha": "train", "bravo": "train", "charlie": "test"},
        {"alpha": "val", "bravo": "val", "charlie": "test"},
    ],
)
def test_train_and_validation_require_separate_nonempty_groups(workspace, splits):
    store, frames = workspace
    with pytest.raises(ValueError, match="nonempty train and val"):
        freeze(store, frames, splits=splits)
    assert store.list("dataset_versions") == []


def test_no_test_split_is_allowed_with_explicit_limitation(workspace):
    store, frames = workspace
    result = freeze(store, frames[:2], splits={"alpha": "train", "bravo": "val"})
    assert result["summary"]["split_counts"]["test"] == 0
    assert any("No test split" in warning for warning in result["summary"]["warnings"])


@pytest.mark.parametrize("duplicate_split", ["train", "val"])
def test_duplicate_pixels_cannot_be_repeated_or_leak_between_splits(
    workspace, tmp_path, duplicate_split
):
    store, frames = workspace
    duplicate = add_frame(store, tmp_path, group="duplicate", color=(30, 60, 80))
    with pytest.raises(ValueError, match="duplicate"):
        freeze(
            store,
            [*frames, duplicate],
            splits={
                "alpha": "train",
                "bravo": "val",
                "charlie": "test",
                "duplicate": duplicate_split,
            },
        )
    assert store.list("dataset_versions") == []


def test_split_assignments_are_reserved_across_independent_versions(workspace):
    store, frames = workspace
    freeze(store, frames)
    candidates = dataset_candidates(store)
    assert {group["scene_group"]: group["reserved_split"] for group in candidates["groups"]} == {
        "alpha": "train",
        "bravo": "val",
        "charlie": "test",
    }
    with pytest.raises(ValueError, match="already reserved"):
        freeze(store, frames, splits={"alpha": "val", "bravo": "train", "charlie": "test"})
    assert len(store.list("dataset_versions")) == 1


def test_renaming_scene_groups_does_not_bypass_pixel_split_reservations(workspace, tmp_path):
    store, frames = workspace
    freeze(store, frames)
    copy = add_frame(store, tmp_path, group="new-name", color=(30, 60, 80))
    with pytest.raises(ValueError, match="pixels are already reserved"):
        freeze(store, [copy, frames[1]], splits={"new-name": "val", "bravo": "val"})
    candidates = dataset_candidates(store)
    group = next(group for group in candidates["groups"] if group["scene_group"] == "new-name")
    assert group["reserved_split"] is None
    assert group["frames"][0]["reserved_split"] == "train"


def test_concurrent_releases_cannot_race_split_reservations(workspace):
    store, frames = workspace
    barrier = Barrier(2)

    def attempt(splits):
        barrier.wait(timeout=5)
        try:
            return freeze(store, frames, splits=splits)
        except ValueError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                attempt,
                [
                    {"alpha": "train", "bravo": "val", "charlie": "test"},
                    {"alpha": "val", "bravo": "train", "charlie": "test"},
                ],
            )
        )
    assert sum(isinstance(result, dict) for result in results) == 1
    assert sum(isinstance(result, ValueError) for result in results) == 1
    assert len(store.list("dataset_versions")) == 1


def test_manifest_and_frozen_image_tampering_is_detected(workspace):
    store, frames = workspace
    result = freeze(store, frames)
    manifest_path = store.artifact_path(result["path"])
    raw = manifest_path.read_bytes()
    manifest_path.write_bytes(raw + b" ")
    with pytest.raises(ValueError, match="manifest no longer matches"):
        dataset_detail(store, result["id"])
    manifest_path.write_bytes(raw)
    copied_image = store.artifact_path(result["manifest"]["frames"][0]["image_path"])
    Image.new("RGB", (80, 60), "black").save(copied_image)
    with pytest.raises(ValueError, match="image no longer matches"):
        load_manifest(store, result["id"], verify_images=True)
    copied_image.unlink()
    with pytest.raises(ValueError, match="image is missing"):
        load_manifest(store, result["id"], verify_images=True)


def test_corrupt_source_pixels_abort_freeze_without_publishing_partial_data(workspace):
    store, frames = workspace
    Image.new("RGB", (80, 60), "black").save(store.artifact_path(frames[1]["path"]))
    with pytest.raises(ValueError, match="recorded hash"):
        freeze(store, frames)
    assert store.list("dataset_versions") == []
    assert list((store.root / "datasets").iterdir()) == []


def test_failed_database_publish_removes_copied_artifacts(workspace):
    store, frames = workspace
    with store.connect() as conn:
        conn.execute(
            "CREATE TRIGGER reject_release BEFORE INSERT ON dataset_versions "
            "BEGIN SELECT RAISE(ABORT,'synthetic publish failure'); END"
        )
    with pytest.raises(sqlite3.IntegrityError, match="synthetic publish failure"):
        freeze(store, frames)
    assert store.list("dataset_versions") == []
    assert list((store.root / "datasets").iterdir()) == []


def test_corrupt_previous_manifest_blocks_new_split_reservations(workspace):
    store, frames = workspace
    result = freeze(store, frames)
    store.artifact_path(result["path"]).write_text("{}")
    with pytest.raises(ValueError, match="manifest no longer matches"):
        freeze(store, frames)
    assert len(store.list("dataset_versions")) == 1


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"name": " "}, "Dataset name"),
        ({"name": "x" * 161}, "Dataset name"),
        ({"frame_ids": []}, "distinct frame"),
        ({"frame_ids": ["a", "a"]}, "distinct frame"),
        ({"frame_ids": ["absent"]}, "does not exist"),
        ({"splits": {"alpha": "training"}}, "train, val or test"),
        ({"splits": {}}, "Assign a split"),
        (
            {"splits": {"alpha": "train", "bravo": "val", "charlie": "test", "extra": "val"}},
            "exactly match",
        ),
        ({"parent_id": "absent"}, "Parent dataset version"),
    ],
)
def test_invalid_requests_do_not_publish_releases(workspace, overrides, message):
    store, frames = workspace
    with pytest.raises(ValueError, match=message):
        freeze(store, frames, **overrides)
    assert store.list("dataset_versions") == []


def test_missing_dataset_is_not_found(workspace):
    store, _ = workspace
    with pytest.raises(KeyError):
        dataset_detail(store, "absent")
    with pytest.raises(KeyError):
        load_manifest(store, "absent")


def test_manifest_image_paths_must_remain_inside_version(workspace):
    store, frames = workspace
    result = freeze(store, frames)
    manifest = result["manifest"]
    manifest["frames"][0]["image_path"] = frames[0]["path"]
    raw = json.dumps(manifest).encode()
    store.artifact_path(result["path"]).write_bytes(raw)
    store.update(
        "dataset_versions",
        result["id"],
        {
            "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        },
    )
    with pytest.raises(ValueError, match="outside its version"):
        load_manifest(store, result["id"])
