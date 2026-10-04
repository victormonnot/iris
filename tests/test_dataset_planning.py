"""Read-only partition planning and video identity protections using synthetic fixtures."""

import hashlib
from copy import deepcopy

import pytest
from test_datasets import add_frame, freeze
from test_datasets import workspace as dataset_workspace

from iris import dataset_planning
from iris.annotations import save_annotation
from iris.dataset_planning import MAX_SIMILAR_PAIRS, preview_dataset_plan
from iris.datasets import (
    DatasetConflict,
    _canonical,
    create_dataset,
    dataset_candidates,
    load_manifest,
)
from iris.projects import create_project
from iris.store import DEFAULT_PROJECT_ID, Store
from iris.taxonomies import TAXONOMY, publish_taxonomy

workspace = dataset_workspace
VIDEO = "a" * 64


def video_source(store, frame, digest=VIDEO):
    # Only the stored original-byte provenance matters here; no video decoder or
    # media download is exercised by dataset planning/publication.
    store.update("assets", frame["asset_id"], {"kind": "video", "sha256": digest})


def apply_plan(store, plan, **overrides):
    return create_dataset(
        store,
        **{
            "name": "Reviewed plan",
            "project_id": plan["project_id"],
            "taxonomy_id": plan["taxonomy_id"],
            "frame_ids": plan["frame_ids"],
            "splits": plan["splits"],
            "expected_revisions": plan["expected_revisions"],
            **overrides,
        },
    )


def test_plan_is_repeatable_read_only_and_can_be_explicitly_frozen(workspace):
    store, frames = workspace
    before = {
        table: store.list(table)
        for table in ("frames", "annotation_revisions", "dataset_versions", "jobs")
    }
    first = preview_dataset_plan(store, seed=123)
    assert preview_dataset_plan(store, seed=123) == first
    assert {table: store.list(table) for table in before} == before
    assert not (store.root / "datasets").exists()
    assert first["can_freeze"] is True and first["blockers"] == []
    assert set(first["frame_ids"]) == {frame["id"] for frame in frames}
    assert first["summary"]["frame_count"] == 3
    assert first["summary"]["negative_count"] == 1
    assert first["summary"]["class_counts"] == {"person": 1, "car": 1}
    assert first["summary"]["split_counts"] == {"train": 2, "val": 1, "test": 0}
    assert any("not guaranteed" in warning for warning in first["warnings"])
    released = apply_plan(store, first)
    assert released["manifest"]["splits"] == first["splits"]
    assert released["summary"]["split_counts"] == first["summary"]["split_counts"]


def test_requested_ratios_are_approximate_but_balanced_for_equal_groups(tmp_path):
    store = Store(tmp_path / "workspace")
    for index in range(10):
        add_frame(store, tmp_path, group=f"scene-{index}", color=(index, 25, 35))
    default = preview_dataset_plan(store)
    assert default["summary"]["split_counts"] == {"train": 8, "val": 2, "test": 0}
    balanced = preview_dataset_plan(store, ratios={"train": 0.6, "val": 0.2, "test": 0.2}, seed=19)
    assert balanced["summary"]["split_counts"] == {"train": 6, "val": 2, "test": 2}
    assert balanced["fingerprint"] != default["fingerprint"]
    assert (
        preview_dataset_plan(store, seed=1)["splits"]
        != preview_dataset_plan(store, seed=2)["splits"]
    )


def test_video_copies_across_groups_are_an_indivisible_component(workspace):
    store, frames = workspace
    video_source(store, frames[0])
    video_source(store, frames[1])
    plan = preview_dataset_plan(store)
    assert plan["splits"]["alpha"] == plan["splits"]["bravo"]
    assert plan["splits"]["alpha"] != plan["splits"]["charlie"]
    linked = next(group for group in plan["groups"] if group["scene_group"] == "alpha")
    assert linked["related_groups"] == ["alpha", "bravo"]
    assert plan["summary"]["linked_group_count"] == 2
    assert plan["can_freeze"]
    apply_plan(store, plan)


def test_connected_video_groups_are_transitive(workspace, tmp_path):
    store, frames = workspace
    extra = add_frame(store, tmp_path, group="bravo", color=(90, 15, 60))
    video_source(store, frames[0], "a" * 64)
    video_source(store, frames[1], "a" * 64)
    video_source(store, extra, "b" * 64)
    video_source(store, frames[2], "b" * 64)
    plan = preview_dataset_plan(store)
    assert len(set(plan["splits"].values())) == 1
    assert plan["summary"]["linked_group_count"] == 1
    assert not plan["can_freeze"]
    assert any(blocker["code"] == "empty_required_split" for blocker in plan["blockers"])


def test_exact_duplicates_require_explicit_selection_and_near_pairs_are_bounded(tmp_path):
    store = Store(tmp_path / "workspace")
    original = add_frame(store, tmp_path, group="first", color=(20, 30, 40))
    duplicate = add_frame(store, tmp_path, group="copy", color=(20, 30, 40))
    for index in range(20):
        add_frame(store, tmp_path, group=f"near-{index}", color=(index + 30, 35, 45))
    plan = preview_dataset_plan(store)
    assert not plan["can_freeze"]
    assert set(plan["frame_ids"]) >= {original["id"], duplicate["id"]}
    assert plan["splits"]["first"] == plan["splits"]["copy"]
    assert plan["exact_duplicates"][0]["frame_ids"] == sorted([original["id"], duplicate["id"]])
    assert len(plan["similar_pairs"]) == MAX_SIMILAR_PAIRS
    assert plan["similar_pairs_total"] > MAX_SIMILAR_PAIRS and plan["similar_pairs_truncated"]
    assert plan["similar_pairs"][0]["cross_split"] is True
    assert all(
        len(pair["frame_ids"]) == 2 and pair["distance"] <= 4 for pair in plan["similar_pairs"]
    )
    assert all(
        set(pair["frame_ids"]) != {original["id"], duplicate["id"]}
        for pair in plan["similar_pairs"]
    )
    assert store.get("frames", duplicate["id"])["selected"] is True


def test_group_and_pixel_reservations_override_requested_ratios(workspace, tmp_path):
    store, frames = workspace
    freeze(store, frames)
    clone = add_frame(store, tmp_path, group="renamed", color=(50, 80, 100))
    store.update("frames", frames[2]["id"], {"selected": False})
    plan = preview_dataset_plan(store)
    assert plan["can_freeze"]
    assert plan["splits"] == {"alpha": "train", "bravo": "val", "renamed": "test"}
    assert clone["id"] in plan["frame_ids"]
    assert any("zero ratio" in warning for warning in plan["warnings"])


def test_freeze_rejects_same_original_video_split_without_publishing(workspace):
    store, frames = workspace
    video_source(store, frames[0])
    video_source(store, frames[1])
    with pytest.raises(ValueError, match="same original video"):
        freeze(store, frames)
    assert store.list("dataset_versions") == []
    assert not list((store.root / "datasets").iterdir())


def test_video_reservation_blocks_new_pixels_after_source_removal_and_in_other_project(
    workspace, tmp_path
):
    store, frames = workspace
    video_source(store, frames[0])
    first = freeze(store, frames)
    copied = add_frame(store, tmp_path, group="copy-scene", color=(121, 31, 51))
    video_source(store, copied)
    store.artifact_path(store.get("assets", frames[0]["asset_id"])["path"]).unlink()
    other = create_project(store, name="Other workspace project")
    store.update("sessions", copied["session_id"], {"project_id": other["id"]})
    plan = preview_dataset_plan(store, project_id=other["id"])
    assert plan["splits"]["copy-scene"] == "train"
    with pytest.raises(ValueError, match="Original video is already reserved"):
        create_dataset(
            store,
            name="Cross-project leak",
            project_id=other["id"],
            frame_ids=[copied["id"]],
            splits={"copy-scene": "val"},
        )
    assert load_manifest(store, first["id"])["frames"][0]["source"]["sha256"] == VIDEO
    assert len(store.list("dataset_versions")) == 1


def test_historical_video_conflicts_remain_readable_but_block_new_uses(workspace):
    store, frames = workspace
    historical = freeze(store, frames)
    manifest = deepcopy(historical["manifest"])
    for item in manifest["frames"][:2]:
        item["source"].update(kind="video", sha256=VIDEO)
    raw = _canonical(manifest)
    store.artifact_path(historical["path"]).write_bytes(raw)
    store.update(
        "dataset_versions", historical["id"], {"manifest_sha256": hashlib.sha256(raw).hexdigest()}
    )
    video_source(store, frames[0])
    video_source(store, frames[1])
    assert load_manifest(store, historical["id"]) == manifest
    candidates = dataset_candidates(store)
    assert candidates["video_conflicts"] == [{"sha256": VIDEO, "splits": ["train", "val"]}]
    plan = preview_dataset_plan(store)
    assert not plan["can_freeze"] and "alpha" not in plan["splits"]
    assert any(blocker["code"] == "conflicting_reservations" for blocker in plan["blockers"])
    with pytest.raises(ValueError, match="historical conflict"):
        freeze(store, frames)
    assert store.artifact_path(historical["path"]).read_bytes() == raw


def test_video_conflict_diagnostics_expose_only_current_project_candidates(workspace, tmp_path):
    store, frames = workspace
    historical = freeze(store, frames)
    manifest = deepcopy(historical["manifest"])
    for item in manifest["frames"][:2]:
        item["source"].update(kind="video", sha256=VIDEO)
    raw = _canonical(manifest)
    store.artifact_path(historical["path"]).write_bytes(raw)
    store.update(
        "dataset_versions",
        historical["id"],
        {"manifest_sha256": hashlib.sha256(raw).hexdigest()},
    )
    video_source(store, frames[0])
    video_source(store, frames[1])
    other = create_project(store, name="Unrelated project")
    unrelated = dataset_candidates(store, project_id=other["id"])
    assert unrelated["video_conflicts"] == []
    assert VIDEO not in _canonical(unrelated).decode()
    assert not any("Existing releases" in warning for warning in unrelated["warnings"])

    # Once this project's own eligible media contains the same original bytes,
    # its conflict is relevant, while reservations continue spanning projects.
    copy = add_frame(store, tmp_path, group="own-copy", color=(14, 24, 34))
    video_source(store, copy)
    store.update("sessions", copy["session_id"], {"project_id": other["id"]})
    relevant = dataset_candidates(store, project_id=other["id"])
    assert relevant["video_conflicts"] == [{"sha256": VIDEO, "splits": ["train", "val"]}]
    assert historical["id"] not in _canonical(relevant).decode()
    with pytest.raises(ValueError, match="historical conflict"):
        create_dataset(
            store,
            name="Conflict remains protected",
            project_id=other["id"],
            frame_ids=[copy["id"]],
            splits={"own-copy": "train"},
        )
    store.update("frames", copy["id"], {"selected": False})
    assert dataset_candidates(store, project_id=other["id"])["video_conflicts"] == []


def test_pending_proposals_and_changed_reviews_invalidate_preview_eligibility(workspace):
    store, frames = workspace
    before = preview_dataset_plan(store)
    save_annotation(
        store, frames[0]["id"], expected_revision=1, boxes=[], decisions={}, status="draft"
    )
    after = preview_dataset_plan(store)
    assert before["fingerprint"] != after["fingerprint"]
    assert frames[0]["id"] not in after["frame_ids"]
    with pytest.raises(DatasetConflict, match="Annotations changed"):
        apply_plan(store, before)


def test_plans_keep_saved_custom_versions_separate(workspace, tmp_path):
    store, originals = workspace
    custom = publish_taxonomy(
        store,
        DEFAULT_PROJECT_ID,
        expected_taxonomy_id=TAXONOMY["id"],
        classes=[{"id": "helmet", "name": "Helmet", "definition": "A helmet."}],
    )
    frame = add_frame(
        store,
        tmp_path,
        group="custom",
        color=(77, 88, 99),
        boxes=[{"id": "box", "label": "helmet", "box": [1, 2, 20, 30]}],
    )
    current = preview_dataset_plan(store)
    legacy = preview_dataset_plan(store, taxonomy_id=TAXONOMY["id"])
    assert current["taxonomy_id"] == custom["id"] and current["frame_ids"] == [frame["id"]]
    assert current["summary"]["class_counts"] == {"helmet": 1}
    assert legacy["taxonomy_id"] == TAXONOMY["id"]
    assert set(legacy["frame_ids"]) == {item["id"] for item in originals}


@pytest.mark.parametrize(
    "ratios",
    [
        {},
        {"train": 1, "val": 0, "test": 0},
        {"train": 0.8, "val": 0.3, "test": 0},
        {"train": True, "val": 0, "test": 0},
        {"train": float("nan"), "val": 0.2, "test": 0},
    ],
)
def test_invalid_ratios_rejected_before_workspace_read(ratios):
    with pytest.raises(ValueError, match="Ratios"):
        preview_dataset_plan(None, ratios=ratios)


@pytest.mark.parametrize("seed", [True, -1, 2147483648, 1.0, "1"])
def test_invalid_seed_rejected_before_workspace_read(seed):
    with pytest.raises(ValueError, match="Seed"):
        preview_dataset_plan(None, seed=seed)


def test_frame_limit_is_enforced_without_truncating_selection(workspace, monkeypatch):
    store, _ = workspace
    candidates = dataset_candidates(store)
    frame = candidates["groups"][0]["frames"][0]
    candidates["groups"][0]["frames"] = [{**frame, "id": f"frame-{index}"} for index in range(1001)]
    monkeypatch.setattr(dataset_planning, "dataset_candidates", lambda *_args: candidates)
    with pytest.raises(ValueError, match="at most 1000"):
        preview_dataset_plan(store)
