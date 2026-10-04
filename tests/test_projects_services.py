"""Project ownership and global exact-pixel split protection on synthetic fixtures."""

import hashlib
import json

import pytest
from test_coco_import import archive, config, image_bytes
from test_datasets import add_frame
from test_evaluation import MODEL_IDS, promote, queue, run
from test_evaluation import workspace as evaluation_fixture

from iris import evaluation
from iris.coco_import import commit_import, preview_import
from iris.datasets import create_dataset, dataset_candidates, load_manifest
from iris.jobs import JobManager
from iris.store import DEFAULT_PROJECT_ID, Store, new_id, now

evaluation_workspace = evaluation_fixture


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "workspace")


def add_project(store):
    return store.insert(
        "projects",
        {
            "id": new_id(),
            "name": "Synthetic second project",
            "description": "Independent local task",
            "taxonomy_id": "iris-objects-v1",
            "created_at": now(),
        },
    )["id"]


def project_frame(store, tmp_path, project_id, **kwargs):
    frame = add_frame(store, tmp_path, **kwargs)
    store.update("sessions", frame["session_id"], {"project_id": project_id})
    return frame


def freeze(store, project_id, frames, **kwargs):
    return create_dataset(
        store,
        project_id=project_id,
        name="Synthetic project release",
        frame_ids=[frame["id"] for frame in frames],
        splits=kwargs.pop("splits", {"alpha": "train", "bravo": "val"}),
        **kwargs,
    )


def pair(store, tmp_path, project_id, *, color=10):
    return [
        project_frame(store, tmp_path, project_id, group=group, color=(color + index, 50, 90))
        for index, group in enumerate(("alpha", "bravo"))
    ]


def test_candidates_and_named_groups_are_independent_between_projects(store, tmp_path):
    other = add_project(store)
    original = pair(store, tmp_path, DEFAULT_PROJECT_ID)
    second = pair(store, tmp_path, other, color=20)
    project_frame(store, tmp_path, other, group="private-draft", color="red", validated=False)
    first_release = freeze(store, DEFAULT_PROJECT_ID, original)
    other_candidates = dataset_candidates(store, other)
    assert {f["id"] for g in other_candidates["groups"] for f in g["frames"]} == {
        frame["id"] for frame in second
    }
    assert all(group["reserved_split"] is None for group in other_candidates["groups"])
    assert other_candidates["excluded"]["unannotated"] == 1
    assert dataset_candidates(store)["excluded"]["unannotated"] == 0
    second_release = freeze(store, other, second, splits={"alpha": "val", "bravo": "train"})
    assert second_release["project_id"] == second_release["manifest"]["project_id"] == other
    assert first_release["project_id"] == DEFAULT_PROJECT_ID
    assert [group["reserved_split"] for group in dataset_candidates(store)["groups"]] == [
        "train",
        "val",
    ]
    assert [group["reserved_split"] for group in dataset_candidates(store, other)["groups"]] == [
        "val",
        "train",
    ]


def test_dataset_rejects_foreign_frames_and_parent_without_publishing(store, tmp_path):
    other = add_project(store)
    original = pair(store, tmp_path, DEFAULT_PROJECT_ID)
    second = pair(store, tmp_path, other, color=20)
    parent = freeze(store, DEFAULT_PROJECT_ID, original)
    with pytest.raises(ValueError, match="different project"):
        freeze(store, other, [second[0], original[1]])
    with pytest.raises(ValueError, match="Parent dataset version belongs"):
        freeze(store, other, second, parent_id=parent["id"])
    assert len(store.list("dataset_versions")) == 1
    assert not list((store.root / "datasets").glob(".staging-*"))


def test_exact_pixels_keep_their_split_across_projects(store, tmp_path):
    other = add_project(store)
    original = pair(store, tmp_path, DEFAULT_PROJECT_ID)
    freeze(store, DEFAULT_PROJECT_ID, original)
    duplicate = project_frame(store, tmp_path, other, group="bravo", color=(10, 50, 90))
    unique = project_frame(store, tmp_path, other, group="alpha", color=(30, 50, 90))
    candidates = dataset_candidates(store, other)
    duplicate_candidate = next(
        frame
        for group in candidates["groups"]
        for frame in group["frames"]
        if frame["id"] == duplicate["id"]
    )
    assert duplicate_candidate["reserved_split"] == "train"
    with pytest.raises(ValueError, match="pixels are already reserved"):
        freeze(store, other, [unique, duplicate])


def test_import_keeps_preview_owner_and_scopes_group_reservations(store, tmp_path):
    other = add_project(store)
    first = preview_import(store, archive(tmp_path), "fixture.zip")
    commit_import(store, first["id"], **config(source_split="train"))
    second = preview_import(
        store,
        archive(
            tmp_path,
            files={
                "images/one.png": image_bytes((90, 50, 70)),
                "images/two.png": image_bytes((91, 50, 70)),
            },
        ),
        "fixture.zip",
        project_id=other,
    )
    assert second["project_id"] == other
    committed = commit_import(store, second["id"], **config(source_split="val"))
    assert store.get("sessions", committed["session_id"])["project_id"] == other
    assert dataset_candidates(store, other)["excluded"]["unannotated"] == 2
    reopened = Store(store.root)
    assert commit_import(reopened, second["id"], **config(source_split="val")) == committed
    duplicated = preview_import(store, archive(tmp_path), "fixture.zip", project_id=other)
    with pytest.raises(ValueError, match="pixel reservation"):
        commit_import(
            store, duplicated["id"], **config(scene_group="new-group", source_split="val")
        )
    assert len(store.list("sessions")) == 2


def test_references_and_stale_selection_checks_are_project_local(evaluation_workspace, tmp_path):
    store = evaluation_workspace[0]
    other = add_project(store)
    first = queue(evaluation_workspace)
    run(store, first)
    first_reference = promote(store, first)
    second_dataset = freeze(store, other, pair(store, tmp_path, other, color=90))
    second = evaluation.create_evaluation(
        store,
        JobManager(store),
        name="Other project evaluation",
        dataset_id=second_dataset["id"],
        model_ids=MODEL_IDS[:1],
    )
    run(store, second)
    second_reference = promote(store, second)
    assert second_reference["metadata"]["project_id"] == other
    with pytest.raises(evaluation.ReferenceConflict):
        promote(store, second, expected_previous_id=first_reference["id"])
    assert evaluation.reference_history(store)["current"] == first_reference
    assert evaluation.reference_history(Store(store.root), other) == {
        "current": second_reference,
        "history": [second_reference],
    }


def test_legacy_default_manifest_remains_readable_without_rewriting(store, tmp_path):
    dataset = freeze(store, DEFAULT_PROJECT_ID, pair(store, tmp_path, DEFAULT_PROJECT_ID))
    manifest = dataset["manifest"]
    manifest.pop("project_id")
    raw = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    path = store.artifact_path(dataset["path"])
    path.write_bytes(raw)
    store.update(
        "dataset_versions", dataset["id"], {"manifest_sha256": hashlib.sha256(raw).hexdigest()}
    )
    assert load_manifest(Store(store.root), dataset["id"], verify_images=True) == manifest
    assert path.read_bytes() == raw
