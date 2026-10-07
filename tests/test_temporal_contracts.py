"""Temporal evidence uses synthetic frames; no tracker or quality claim is exercised."""

from copy import deepcopy

import pytest

from iris import temporal_contracts as contracts
from iris.taxonomies import TAXONOMY


def sequence(*, sparse=False, basis="nominal_fps"):
    indices = [0, 2] if sparse else [0, 1, 2]
    return {
        "schema": contracts.SEQUENCE_SCHEMA,
        "id": "sequence-1",
        "project_id": "project-1",
        "name": "Synthetic sequence",
        "parent_id": None,
        "asset": {
            "id": "video-1",
            "sha256": "a" * 64,
            "session_id": "session-1",
            "scene_group": "take-one",
        },
        "take_group": "take-one",
        "taxonomy": deepcopy(TAXONOMY),
        "clock": {
            "basis": basis,
            "fps": 10 if basis == "nominal_fps" else None,
            "provenance": "Synthetic declared clock, not a camera recording",
        },
        "clip": {"start_frame": 0, "end_frame": 2},
        "frames": [
            {
                "frame_id": f"frame-{index}",
                "frame_index": index,
                "timestamp_seconds": None if basis == "unknown" else index / 10,
                "width": 100,
                "height": 80,
                "sha256": f"{index:064x}",
                "file_sha256": f"{index + 100:064x}",
            }
            for index in indices
        ],
        "gaps": [{"start_frame": 1, "end_frame": 1, "reason": "skipped"}] if sparse else [],
    }


def reference(manifest, *, status="human_reviewed", coverage="complete"):
    return {
        "schema": contracts.REFERENCE_SCHEMA,
        "sequence_id": manifest["id"],
        "sequence_sha256": contracts.sequence_hash(manifest),
        "taxonomy_id": manifest["taxonomy"]["id"],
        "identities": [{"id": "person-A", "label": "person"}],
        "frames": [
            {
                "frame_index": frame["frame_index"],
                "coverage": coverage,
                "review": {"status": status, "reviewer": "Fixture reviewer"},
                "objects": [
                    {
                        "identity_id": "person-A",
                        "label": "person",
                        "box": [0, 10, 20, 80],
                        "visibility": "visible",
                        "certainty": "certain",
                    }
                ],
            }
            for frame in manifest["frames"]
        ],
        "notes": "Synthetic evidence only",
    }


def set_path(value, path, replacement):
    for component in path[:-1]:
        value = value[component]
    value[path[-1]] = replacement


def test_sequence_round_trip_canonical_hash_and_copy_preserve_source_identity():
    original = sequence()
    result = contracts.validate_sequence_manifest(original)
    assert result == original and result is not original
    assert type(result["clock"]["fps"]) is float
    assert type(result["frames"][0]["timestamp_seconds"]) is float
    reordered = dict(reversed(list(original.items())))
    assert contracts.sequence_hash(original) == contracts.sequence_hash(reordered)
    result["taxonomy"]["classes"][0]["name"] = "Changed"
    result["frames"][0]["width"] = 1
    result["asset"]["id"] = "changed"
    assert original["taxonomy"] == TAXONOMY
    assert original["frames"][0]["width"] == 100
    assert original["asset"]["id"] == "video-1"


@pytest.mark.parametrize("basis", ["provided", "unknown"])
def test_external_and_unknown_clocks_do_not_invent_capture_times(basis):
    manifest = sequence(basis=basis)
    if basis == "provided":
        manifest["frames"][1]["timestamp_seconds"] = 0.07
        manifest["frames"][2]["timestamp_seconds"] = 0.19
    assert contracts.validate_sequence_manifest(manifest) == manifest


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("schema",), "iris-temporal-sequence-v2"),
        (("id",), ""),
        (("id",), "a" * 129),
        (("project_id",), True),
        (("parent_id",), "sequence-1"),
        (("asset", "sha256"), "A" * 64),
        (("asset", "scene_group"), " "),
        (("asset", "scene_group"), "x" * 161),
        (("take_group",), ""),
        (("take_group",), "x" * 161),
        (("name",), "x" * 161),
        (("clock", "basis"), "capture_pts"),
        (("clock", "fps"), True),
        (("clock", "fps"), 0),
        (("clock", "fps"), float("nan")),
        (("clock", "fps"), 10**400),
        (("clock", "provenance"), ""),
        (("clip", "start_frame"), True),
        (("clip", "end_frame"), -1),
        (("clip", "end_frame"), contracts.MAX_CLIP_FRAMES),
        (("frames", 1, "frame_id"), "frame-0"),
        (("frames", 1, "frame_index"), 0),
        (("frames", 0, "frame_index"), -1),
        (("frames", 0, "width"), False),
        (("frames", 0, "height"), 0),
        (("frames", 0, "sha256"), "not-a-hash"),
        (("frames", 0, "file_sha256"), "not-a-file-hash"),
        (("frames", 0, "timestamp_seconds"), True),
        (("frames", 0, "timestamp_seconds"), -0.1),
        (("frames", 0, "timestamp_seconds"), float("inf")),
        (("frames", 1, "timestamp_seconds"), 0.11),
        (("frames", 1, "timestamp_seconds"), None),
    ],
)
def test_sequence_rejects_invalid_timing_provenance_and_frame_fields(path, replacement):
    manifest = sequence()
    set_path(manifest, path, replacement)
    with pytest.raises(ValueError):
        contracts.validate_sequence_manifest(manifest)


@pytest.mark.parametrize("basis", ["provided", "unknown"])
def test_non_nominal_clocks_do_not_mix_with_nominal_fps(basis):
    manifest = sequence(basis=basis)
    manifest["clock"]["fps"] = 10
    with pytest.raises(ValueError, match="null FPS"):
        contracts.validate_sequence_manifest(manifest)


def test_unknown_clock_cannot_claim_a_timestamp_and_provided_times_must_increase():
    manifest = sequence(basis="unknown")
    manifest["frames"][0]["timestamp_seconds"] = 0
    with pytest.raises(ValueError, match="null timestamps"):
        contracts.validate_sequence_manifest(manifest)
    manifest = sequence(basis="provided")
    manifest["frames"][1]["timestamp_seconds"] = 0
    with pytest.raises(ValueError, match="strictly increasing"):
        contracts.validate_sequence_manifest(manifest)


def test_gaps_are_explicit_and_cover_every_unavailable_frame():
    manifest = sequence(sparse=True)
    assert contracts.validate_sequence_manifest(manifest) == manifest
    manifest["gaps"] = []
    with pytest.raises(ValueError, match="cover the clip"):
        contracts.validate_sequence_manifest(manifest)
    manifest["gaps"] = [{"start_frame": 1, "end_frame": 2, "reason": "unknown"}]
    with pytest.raises(ValueError, match="overlap"):
        contracts.validate_sequence_manifest(manifest)


def test_boundary_gaps_and_large_clip_use_intervals_instead_of_fabricated_frames():
    manifest = sequence(sparse=True, basis="unknown")
    manifest["clip"] = {"start_frame": 0, "end_frame": contracts.MAX_CLIP_FRAMES - 1}
    manifest["frames"] = [manifest["frames"][1]]
    manifest["gaps"] = [
        {"start_frame": 0, "end_frame": 1, "reason": "unknown"},
        {"start_frame": 3, "end_frame": contracts.MAX_CLIP_FRAMES - 1, "reason": "unavailable"},
    ]
    assert len(contracts.validate_sequence_manifest(manifest)["frames"]) == 1
    manifest["gaps"].reverse()
    with pytest.raises(ValueError, match="ordered"):
        contracts.validate_sequence_manifest(manifest)


def test_reference_round_trip_copies_boxes_and_preserves_exact_source_binding():
    manifest = sequence()
    original = reference(manifest)
    result = contracts.validate_reference(original, manifest)
    assert result == original and type(result["frames"][0]["objects"][0]["box"][0]) is float
    result["frames"][0]["objects"][0]["box"][0] = 10
    result["frames"][0]["review"]["reviewer"] = "Changed"
    assert original["frames"][0]["objects"][0]["box"][0] == 0
    assert original["frames"][0]["review"]["reviewer"] == "Fixture reviewer"
    manifest["asset"]["sha256"] = "b" * 64
    with pytest.raises(ValueError, match="frozen manifest"):
        contracts.validate_reference(original, manifest)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("schema",), "iris-temporal-reference-v2"),
        (("sequence_id",), "another-sequence"),
        (("sequence_sha256",), "0" * 64),
        (("taxonomy_id",), "other-classes"),
        (("notes",), "x" * 4001),
        (("identities", 0, "label"), "truck"),
        (("identities", 0, "id"), ""),
        (("frames", 0, "frame_index"), True),
        (("frames", 0, "frame_index"), 99),
        (("frames", 0, "review", "status"), "validated"),
        (("frames", 0, "review", "reviewer"), " "),
        (("frames", 0, "coverage"), "unreviewed"),
        (("frames", 0, "review", "status"), "unreviewed"),
        (("frames", 0, "objects", 0, "identity_id"), "unregistered-person"),
        (("frames", 0, "objects", 0, "label"), "car"),
        (("frames", 0, "objects", 0, "box"), None),
        (("frames", 0, "objects", 0, "box"), [0, 10, 101, 80]),
        (("frames", 0, "objects", 0, "box"), [0, 10, 0, 80]),
        (("frames", 0, "objects", 0, "box"), [True, 10, 20, 80]),
        (("frames", 0, "objects", 0, "box"), [0, 10, float("nan"), 80]),
        (("frames", 0, "objects", 0, "visibility"), "out_of_view"),
        (("frames", 0, "objects", 0, "visibility"), "unknown"),
        (("frames", 0, "objects", 0, "certainty"), "uncertain"),
        (("frames", 0, "objects", 0, "identity_id"), None),
    ],
)
def test_reference_rejects_inconsistent_identity_review_and_observations(path, replacement):
    manifest = sequence()
    payload = reference(manifest)
    set_path(payload, path, replacement)
    with pytest.raises(ValueError):
        contracts.validate_reference(payload, manifest)


@pytest.mark.parametrize("visibility", ["occluded", "out_of_view", "unknown"])
def test_nonvisible_objects_can_record_continuity_without_invented_coordinates(visibility):
    manifest = sequence()
    payload = reference(manifest, coverage="partial")
    obj = payload["frames"][0]["objects"][0]
    obj.update(visibility=visibility, box=None)
    if visibility == "unknown":
        obj.update(certainty="uncertain")
    assert contracts.validate_reference(payload, manifest)["frames"][0]["objects"][0] == obj


def test_unresolved_identity_is_explicit_and_cannot_become_complete_evidence():
    manifest = sequence()
    payload = reference(manifest, coverage="partial")
    payload["frames"][0]["objects"][0].update(identity_id=None, certainty="uncertain")
    summary = contracts.reference_summary(payload, manifest)
    assert summary["uncertain_objects"] == 1
    assert summary["partial_frames"] == 3
    assert not summary["dense_human_reference"]


def test_duplicate_identity_in_frame_and_duplicate_or_gap_references_are_rejected():
    manifest = sequence(sparse=True)
    payload = reference(manifest)
    payload["frames"][0]["objects"] *= 2
    with pytest.raises(ValueError, match="once per reference frame"):
        contracts.validate_reference(payload, manifest)
    payload = reference(manifest)
    payload["frames"][1]["frame_index"] = 1
    with pytest.raises(ValueError, match="available"):
        contracts.validate_reference(payload, manifest)
    payload["frames"][1]["frame_index"] = 0
    with pytest.raises(ValueError, match="unique"):
        contracts.validate_reference(payload, manifest)
    payload = reference(manifest)
    payload["identities"] *= 2
    with pytest.raises(ValueError, match="identity IDs must be unique"):
        contracts.validate_reference(payload, manifest)


def test_sparse_assistant_and_partial_evidence_never_become_dense_human_ground_truth():
    dense = sequence()
    assert contracts.reference_summary(reference(dense), dense)["dense_human_reference"]
    sparse = sequence(sparse=True)
    summary = contracts.reference_summary(reference(sparse), sparse)
    assert summary["human_complete_frames"] == 2
    assert not summary["dense_human_reference"]
    assistant = reference(dense, status="assistant_reviewed")
    summary = contracts.reference_summary(assistant, dense)
    assert summary["assistant_reviewed_frames"] == 3
    assert summary["human_reviewed_frames"] == 0
    assert not summary["dense_human_reference"]
    partial = reference(dense, coverage="partial")
    assert not contracts.reference_summary(partial, dense)["dense_human_reference"]


def test_missing_frames_are_unknown_but_explicit_reviewed_empty_frames_are_negatives():
    manifest = sequence()
    payload = reference(manifest)
    payload["frames"] = [payload["frames"][0]]
    payload["frames"][0]["objects"] = []
    summary = contracts.reference_summary(payload, manifest)
    assert summary["annotated_frames"] == summary["human_complete_frames"] == 1
    assert summary["omitted_frames"] == summary["unreviewed_frames"] == 2
    assert summary["object_count"] == 0
    assert not summary["dense_human_reference"]
    payload["frames"][0].update(
        coverage="unreviewed", review={"status": "unreviewed", "reviewer": ""}
    )
    summary = contracts.reference_summary(payload, manifest)
    assert summary["annotated_frames"] == summary["human_complete_frames"] == 0
    assert summary["unreviewed_frames"] == 3


@pytest.mark.parametrize("where", ["sequence", "frame", "identity", "reference_object", "review"])
def test_tracking_ids_and_unknown_fields_cannot_be_smuggled_into_reference_contracts(where):
    manifest = sequence()
    payload = reference(manifest)
    targets = {
        "sequence": manifest,
        "frame": manifest["frames"][0],
        "identity": payload["identities"][0],
        "reference_object": payload["frames"][0]["objects"][0],
        "review": payload["frames"][0]["review"],
    }
    targets[where]["tracker_id"] = 123
    with pytest.raises(ValueError, match="supported fields"):
        contracts.validate_reference(payload, manifest)


def test_non_builtin_classes_work_without_person_specific_identity_assumptions():
    manifest = sequence()
    manifest["taxonomy"] = {
        "id": "taxonomy-" + "a" * 32,
        "version": 2,
        "parent_id": TAXONOMY["id"],
        "classes": [{"id": "forklift", "name": "Forklift", "definition": "Fixture class"}],
        "box_format": TAXONOMY["box_format"],
        "review_guidance": TAXONOMY["review_guidance"],
        "created_at": "2026-10-07T12:00:00Z",
    }
    payload = reference(manifest)
    payload["identities"] = [{"id": "machine-A", "label": "forklift"}]
    for frame in payload["frames"]:
        frame["objects"][0].update(identity_id="machine-A", label="forklift")
    assert contracts.validate_reference(payload, manifest) == payload
    assert contracts.reference_summary(payload, manifest)["dense_human_reference"]


def test_collection_limits_bound_imports_before_tracker_or_ml_work(monkeypatch):
    manifest = sequence()
    monkeypatch.setattr(contracts, "MAX_SEQUENCE_FRAMES", 2)
    with pytest.raises(ValueError, match="Available frames"):
        contracts.validate_sequence_manifest(manifest)
    monkeypatch.setattr(contracts, "MAX_SEQUENCE_FRAMES", 10_000)
    payload = reference(manifest)
    monkeypatch.setattr(contracts, "MAX_REFERENCE_OBJECTS", 2)
    with pytest.raises(ValueError, match="at most 2 objects"):
        contracts.validate_reference(payload, manifest)


def test_notes_preserve_multiline_text_while_identifiers_reject_control_characters():
    manifest = sequence()
    payload = reference(manifest)
    payload["notes"] = "First observation.\n\tSecond observation."
    assert contracts.validate_reference(payload, manifest)["notes"] == payload["notes"]
    manifest["id"] = "sequence\n1"
    with pytest.raises(ValueError, match="bounded text"):
        contracts.validate_sequence_manifest(manifest)
    manifest["id"] = "sequence\ud800"
    with pytest.raises(ValueError, match="UTF-8"):
        contracts.validate_sequence_manifest(manifest)


def test_negative_zero_does_not_change_canonical_sequence_identity():
    manifest = sequence()
    equivalent = deepcopy(manifest)
    equivalent["frames"][0]["timestamp_seconds"] = -0.0
    assert contracts.sequence_hash(equivalent) == contracts.sequence_hash(manifest)


def test_frame_file_digest_is_mandatory_and_distinct_from_its_pixel_digest():
    manifest = sequence()
    original_hash = contracts.sequence_hash(manifest)
    manifest["frames"][0]["file_sha256"] = "f" * 64
    assert contracts.sequence_hash(manifest) != original_hash
    assert manifest["frames"][0]["sha256"] == "0" * 64
    del manifest["frames"][0]["file_sha256"]
    with pytest.raises(ValueError, match="supported fields"):
        contracts.validate_sequence_manifest(manifest)
