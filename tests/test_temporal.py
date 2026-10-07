"""Temporal publication, ownership and split isolation with generated media only."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Barrier

import pytest
from temporal_fixtures import video_sequence
from test_datasets import add_frame

from iris.annotations import save_annotation
from iris.datasets import create_dataset
from iris.projects import create_project
from iris.store import Store
from iris.temporal import (
    TemporalConflict,
    create_sequence,
    create_temporal_dataset,
    reference_detail,
    save_reference,
    sequence_detail,
    temporal_dataset_detail,
    validate_temporal_dataset_splits,
    validate_temporal_records,
)


@pytest.fixture
def source(tmp_path):
    store = Store(tmp_path / "workspace")
    asset, frames = video_sequence(store, tmp_path, count=5)
    return store, asset, frames


def freeze(source, **kwargs):
    store, asset, frames = source
    return create_sequence(
        store,
        **{
            "name": "Synthetic take",
            "asset_id": asset["id"],
            "frame_ids": [frame["id"] for frame in frames],
            **kwargs,
        },
    )


def reference(sequence, status="human_reviewed"):
    return {
        "schema": "iris-temporal-reference-v1",
        "sequence_id": sequence["id"],
        "sequence_sha256": sequence["manifest_sha256"],
        "taxonomy_id": sequence["manifest"]["taxonomy"]["id"],
        "identities": [{"id": "object-A", "label": "person"}],
        "frames": [
            {
                "frame_index": frame["frame_index"],
                "coverage": "complete",
                "review": {"status": status, "reviewer": "Synthetic reviewer"},
                "objects": [
                    {
                        "identity_id": "object-A",
                        "label": "person",
                        "box": [1, 2, 30, 50],
                        "visibility": "visible",
                        "certainty": "certain",
                    }
                ],
            }
            for frame in sequence["manifest"]["frames"]
        ],
        "notes": "Synthetic reference only",
    }


def dataset(store, sequence, split="train", **kwargs):
    return create_temporal_dataset(
        store,
        name="Synthetic temporal release",
        entries=[
            {
                "sequence_id": sequence["id"],
                "split": split,
            }
        ],
        **kwargs,
    )


def test_sparse_sequence_explicitly_covers_unknown_gaps_and_preserves_annotations(source):
    store, _, frames = source
    annotation = save_annotation(
        store,
        frames[0]["id"],
        expected_revision=0,
        boxes=[],
        decisions={},
        status="validated",
        reviewer="Original image reviewer",
    )
    before = store.list("annotation_revisions")
    result = freeze(
        source,
        frame_ids=[frames[3]["id"], frames[1]["id"]],
        clip={"start_frame": 0, "end_frame": 4},
    )
    assert [f["frame_index"] for f in result["manifest"]["frames"]] == [1, 3]
    assert result["manifest"]["gaps"] == [
        {"start_frame": index, "end_frame": index, "reason": "unknown"} for index in (0, 2, 4)
    ]
    assert result["manifest"]["clock"]["basis"] == "nominal_fps"
    assert sequence_detail(store, result["id"])["latest_reference"] is None
    assert store.list("annotation_revisions") == before
    assert annotation["revision"] == 1
    assert store.list("temporal_references") == []


def test_provided_and_unknown_clocks_never_relabel_estimated_source_times(source):
    _, _, frames = source
    times = {frame["id"]: index * 0.11 + 2 for index, frame in enumerate(frames)}
    result = freeze(
        source,
        clock={
            "basis": "provided",
            "fps": None,
            "provenance": "User supplied timestamp sidecar; synthetic",
        },
        timestamps=times,
    )
    assert [f["timestamp_seconds"] for f in result["manifest"]["frames"]] == list(times.values())
    result = freeze(
        source, clock={"basis": "unknown", "fps": None, "provenance": "No timestamp source"}
    )
    assert all(f["timestamp_seconds"] is None for f in result["manifest"]["frames"])


@pytest.mark.parametrize(
    "mutation", ["frame", "video", "outside", "duplicate", "timestamp", "clock", "gap"]
)
def test_invalid_or_modified_sources_never_publish_a_sequence(source, mutation):
    store, asset, frames = source
    kwargs = {}
    if mutation == "frame":
        store.artifact_path(frames[0]["path"]).write_bytes(b"corrupted PNG")
    elif mutation == "video":
        store.artifact_path(asset["path"]).write_bytes(b"changed video")
    elif mutation == "outside":
        kwargs["clip"] = {"start_frame": 0, "end_frame": 100}
    elif mutation == "duplicate":
        kwargs["frame_ids"] = [frames[0]["id"]] * 2
    elif mutation == "timestamp":
        kwargs["timestamps"] = {frames[0]["id"]: 123}
    elif mutation == "clock":
        kwargs["clock"] = {"basis": "nominal_fps", "fps": 30, "provenance": "wrong FPS"}
    else:
        kwargs["gaps"] = [{"start_frame": 0, "end_frame": 2, "reason": "skipped"}]
    with pytest.raises((ValueError, OSError)):
        freeze(source, **kwargs)
    assert store.list("temporal_sequences") == []


def test_cross_project_frames_parent_and_reference_cannot_be_mixed(source, tmp_path):
    store, _, frames = source
    other = create_project(store, name="Other")
    asset, other_frames = video_sequence(
        store, tmp_path, group="other", project_id=other["id"], color=80
    )
    parent = freeze((store, asset, other_frames), project_id=other["id"])
    for kwargs in (
        {"frame_ids": [frames[0]["id"], other_frames[0]["id"]]},
        {"parent_id": parent["id"]},
    ):
        with pytest.raises(ValueError):
            freeze(source, **kwargs)
    with pytest.raises(ValueError):
        dataset(store, parent)
    sequence = freeze(source)
    with pytest.raises(ValueError):
        save_reference(store, sequence["id"], payload=reference(parent))


def test_reference_revision_is_atomic_and_keeps_declared_review_provenance(source):
    store, _, _ = source
    sequence = freeze(source)
    assistant = save_reference(
        store, sequence["id"], payload=reference(sequence, "assistant_reviewed")
    )
    assert assistant["summary"]["dense_human_reference"] is False
    payload = reference(sequence)
    gate = Barrier(2)

    def attempt():
        gate.wait()
        try:
            return save_reference(store, sequence["id"], payload=payload, expected_revision=1)
        except TemporalConflict:
            return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: attempt(), range(2)))
    assert sum(row is not None for row in results) == 1
    assert len(store.list("temporal_references")) == 2
    assert reference_detail(store, assistant["id"])["payload"] == assistant["payload"]
    assert sequence_detail(store, sequence["id"])["latest_reference"]["summary"][
        "dense_human_reference"
    ]


def test_dataset_pins_exact_reference_revision_not_latest(source):
    store, _, _ = source
    sequence = freeze(source)
    first = save_reference(store, sequence["id"], payload=reference(sequence, "assistant_reviewed"))
    release = create_temporal_dataset(
        store,
        name="Frozen",
        entries=[
            {
                "sequence_id": sequence["id"],
                "split": "val",
                "reference_id": first["id"],
            }
        ],
    )
    save_reference(store, sequence["id"], payload=reference(sequence), expected_revision=1)
    assert temporal_dataset_detail(store, release["id"])["manifest"] == release["manifest"]
    with store.connect() as conn:
        validate_temporal_records(conn)


@pytest.mark.parametrize("relationship", ["same_video", "same_scene", "same_take", "same_pixels"])
def test_related_sequences_cannot_cross_splits(source, tmp_path, relationship):
    store, asset, frames = source
    sequence = freeze(source, take_group="shared-take")
    dataset(store, sequence, "train")
    if relationship == "same_video":
        other = freeze(source, frame_ids=[frames[-1]["id"]])
    else:
        group = "take-a" if relationship == "same_scene" else "new-scene"
        other_asset, other_frames = video_sequence(
            store, tmp_path, group=group, color=0 if relationship == "same_pixels" else 170
        )
        other = freeze(
            (store, other_asset, other_frames),
            take_group="shared-take" if relationship == "same_take" else "other-take",
        )
    with pytest.raises(ValueError, match="split conflict"):
        dataset(store, other, "test")
    assert len(store.list("temporal_datasets")) == 1


def test_reimported_video_cannot_escape_split_by_project_or_group(source, tmp_path):
    store, asset, _ = source
    dataset(store, freeze(source), "train")
    project = create_project(store, name="New experiment")
    other_asset, frames = video_sequence(
        store, tmp_path, project_id=project["id"], group="elsewhere", color=100
    )
    raw = store.artifact_path(asset["path"]).read_bytes()
    store.artifact_path(other_asset["path"]).write_bytes(raw)
    store.update("assets", other_asset["id"], {"sha256": asset["sha256"], "size_bytes": len(raw)})
    other = freeze((store, other_asset, frames), project_id=project["id"])
    with pytest.raises(ValueError, match="source video"):
        dataset(store, other, "test", project_id=project["id"])


def _image_release(store, frames, tmp_path, split):
    frame = frames[0]
    store.update("frames", frame["id"], {"selected": True})
    if not store.list("annotation_revisions", frame_id=frame["id"]):
        save_annotation(
            store,
            frame["id"],
            expected_revision=0,
            boxes=[],
            decisions={},
            status="validated",
            reviewer="Synthetic reviewer",
        )
    group = store.get("sessions", frame["session_id"])["scene_group"]
    independent = add_frame(store, tmp_path, group="independent", color=(255, 255, 0))
    return create_dataset(
        store,
        name="Image release",
        frame_ids=[frame["id"], independent["id"]],
        splits={group: split, "independent": "val" if split != "val" else "train"},
    )


def test_image_training_video_cannot_become_temporal_test(source, tmp_path):
    store, _, frames = source
    _image_release(store, frames, tmp_path, "train")
    sequence = freeze(source, frame_ids=[frames[-1]["id"]])
    with pytest.raises(ValueError, match="split conflict"):
        dataset(store, sequence, "test")
    dataset(store, sequence, "train")


def test_temporal_training_video_cannot_become_image_test(source, tmp_path):
    store, _, frames = source
    dataset(store, freeze(source, frame_ids=[frames[-1]["id"]]), "train")
    with pytest.raises(ValueError):
        _image_release(store, frames, tmp_path, "test")
    assert store.list("dataset_versions") == []


def test_sequence_versions_keep_original_snapshots_and_group(source):
    store, _, frames = source
    parent = freeze(source)
    child = freeze(source, parent_id=parent["id"], frame_ids=[frames[1]["id"]])
    assert sequence_detail(store, parent["id"])["manifest"] == parent["manifest"]
    assert child["manifest"]["parent_id"] == parent["id"]
    with pytest.raises(ValueError):
        freeze(source, parent_id=parent["id"], take_group="different-take")


def test_wrong_reference_owner_does_not_publish_dataset(source, tmp_path):
    store, _, _ = source
    first = freeze(source)
    second = freeze(source, name="Second view")
    ref = save_reference(store, first["id"], payload=reference(first))
    with pytest.raises(ValueError):
        create_temporal_dataset(
            store,
            name="Wrong reference",
            entries=[
                {
                    "sequence_id": second["id"],
                    "reference_id": ref["id"],
                    "split": "train",
                }
            ],
        )
    assert store.list("temporal_datasets") == []


def test_sequence_tampering_is_detected_before_reference_publication(source):
    store, _, _ = source
    sequence = freeze(source)
    with store.connect() as conn:
        conn.execute(
            "UPDATE temporal_sequences SET manifest_sha256=? WHERE id=?", ("0" * 64, sequence["id"])
        )
    with pytest.raises(ValueError, match="checksum"):
        save_reference(store, sequence["id"], payload=reference(sequence))
    assert store.list("temporal_references") == []


def test_unknown_identity_is_not_an_empty_negative_reference(source):
    store, _, _ = source
    sequence = freeze(source)
    payload = reference(sequence)
    payload["frames"] = payload["frames"][:1]
    payload["frames"][0]["coverage"] = "partial"
    payload["frames"][0]["objects"][0].update(identity_id=None, certainty="uncertain")
    original = deepcopy(payload)
    saved = save_reference(store, sequence["id"], payload=payload)
    assert saved["summary"]["dense_human_reference"] is False
    assert saved["summary"]["omitted_frames"] == 4
    assert saved["summary"]["uncertain_objects"] == 1
    assert payload == original


@pytest.mark.parametrize("publication", ["reference", "dataset"])
@pytest.mark.parametrize(
    "changed", ["frame_corrupt", "frame_missing", "video_corrupt", "video_missing"]
)
def test_changed_media_after_sequence_freeze_blocks_new_publication(source, publication, changed):
    store, asset, frames = source
    sequence = freeze(source)
    path = store.artifact_path(frames[0]["path"] if changed.startswith("frame") else asset["path"])
    if changed.endswith("missing"):
        path.unlink()
    else:
        path.write_bytes(b"changed after freezing")
    with pytest.raises(ValueError):
        if publication == "reference":
            save_reference(store, sequence["id"], payload=reference(sequence))
        else:
            dataset(store, sequence)
    assert store.list("temporal_references") == []
    assert store.list("temporal_datasets") == []


def test_archive_cross_checks_do_not_reinterpret_unrelated_legacy_video_conflicts(source):
    store, asset, _ = source
    dataset(store, freeze(source), "train")
    manifests = [
        {
            "project_id": "default",
            "frames": [
                {
                    "scene_group": f"legacy-{index}",
                    "sha256": str(index) * 64,
                    "split": split,
                    "source": {"kind": "video", "sha256": "f" * 64},
                }
                for index, split in enumerate(("train", "val"))
            ],
        }
    ]
    with store.connect() as conn:
        validate_temporal_dataset_splits(conn, manifests)
        manifests[0]["frames"][1]["source"]["sha256"] = asset["sha256"]
        with pytest.raises(ValueError, match="source split"):
            validate_temporal_dataset_splits(conn, manifests)


def test_nominal_clock_rejects_integer_overflow_without_publishing(source):
    with pytest.raises(ValueError, match="FPS"):
        freeze(
            source, clock={"basis": "nominal_fps", "fps": 10**400, "provenance": "Invalid input"}
        )
    assert source[0].list("temporal_sequences") == []
