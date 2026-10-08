"""Independent small tracking examples with fully explicit reference evidence."""

import itertools
import json
import math
import random
import subprocess
import sys
from copy import deepcopy

import pytest
from test_temporal_contracts import sequence

from iris import tracking_metrics as metrics
from iris.temporal import _digest
from iris.temporal_contracts import REFERENCE_SCHEMA, sequence_hash, validate_reference

BOX = [0, 0, 10, 10]
FAR = [70, 60, 80, 70]


def gt(identity="A", box=BOX, *, label="person", visibility="visible", certainty="certain"):
    return {
        "identity_id": identity,
        "label": label,
        "box": deepcopy(box),
        "visibility": visibility,
        "certainty": certainty,
    }


def observed(track_id=17, box=BOX, *, label_id=1, confirmed=True):
    return {
        "track_id": track_id,
        "label_id": label_id,
        "box": deepcopy(box),
        "estimated_box": deepcopy(FAR),
        "confirmed": confirmed,
    }


def example(objects, observations, *, indices=None):
    """Minimal already-checked T4 data and a real validated T1 reference payload."""
    assert len(objects) == len(observations)
    indices = indices or list(range(len(objects)))
    manifest = sequence()
    frame_template = manifest["frames"][0]
    manifest["frames"] = [
        {
            **frame_template,
            "frame_index": index,
            "frame_id": f"frame-{index}",
            "timestamp_seconds": index / 10,
            "sha256": f"{index:064x}",
            "file_sha256": f"{index + 100:064x}",
        }
        for index in indices
    ]
    manifest["clip"] = {"start_frame": indices[0], "end_frame": indices[-1]}
    manifest["gaps"] = [
        {"start_frame": first + 1, "end_frame": second - 1, "reason": "skipped"}
        for first, second in zip(indices, indices[1:], strict=False)
        if second != first + 1
    ]
    identities = {
        obj["identity_id"]: obj["label"]
        for frame in objects
        for obj in frame
        if obj["identity_id"] is not None
    }
    payload = {
        "schema": REFERENCE_SCHEMA,
        "sequence_id": manifest["id"],
        "sequence_sha256": sequence_hash(manifest),
        "taxonomy_id": manifest["taxonomy"]["id"],
        "identities": [{"id": identity, "label": label} for identity, label in identities.items()],
        "frames": [
            {
                "frame_index": index,
                "coverage": "complete",
                "review": {"status": "human_reviewed", "reviewer": "Synthetic fixture author"},
                "objects": deepcopy(frame),
            }
            for index, frame in zip(indices, objects, strict=True)
        ],
        "notes": "Synthetic fixture; not human evidence about a real video",
    }
    payload = validate_reference(payload, manifest)
    reference = {
        "id": "reference-1",
        "revision": 1,
        "payload": payload,
        "payload_sha256": _digest(payload),
    }
    frames = [
        {
            "frame_index": index,
            "observations": deepcopy(frame),
            "predictions": [],
            "unassigned": [],
        }
        for index, frame in zip(indices, observations, strict=True)
    ]
    comparison = {
        "id": "comparison-1",
        "report": {
            "sequence": manifest,
            "lanes": [
                {
                    "name": name,
                    "report": {
                        "profile": {"class_ids": [1, 3]},
                        "profile_sha256": str(index) * 64,
                        "cache": {"fingerprint": "a" * 64, "result_sha256": "b" * 64},
                        "passes": [{"frames": deepcopy(frames), "semantic_sha256": "c" * 64}],
                    },
                }
                for index, name in enumerate(("ByteTrack", "BoT-SORT"), 1)
            ],
        },
    }
    return comparison, reference


def evaluate(pair, **options):
    return metrics.evaluate_quality(
        *pair, class_mapping=options.pop("class_mapping", {"1": "person", "3": "car"}), **options
    )


def first_lane(pair, **options):
    return evaluate(pair, **options)["lanes"][0]


def test_assignment_matches_exhaustive_small_rectangular_oracle_and_ties_are_stable():
    rng = random.Random(3912)
    for rows in range(1, 6):
        for columns in range(1, 6):
            for _ in range(3):
                matrix = [[rng.randrange(7) for _ in range(columns)] for _ in range(rows)]
                result = metrics._maximum_assignment(matrix)
                if rows <= columns:
                    optimum = max(
                        sum(matrix[row][column] for row, column in enumerate(permutation))
                        for permutation in itertools.permutations(range(columns), rows)
                    )
                else:
                    optimum = max(
                        sum(matrix[row][column] for column, row in enumerate(permutation))
                        for permutation in itertools.permutations(range(rows), columns)
                    )
                assert sum(matrix[row][column] for row, column in result) == optimum
                assert len({row for row, _ in result}) == min(rows, columns)
                assert len({column for _, column in result}) == min(rows, columns)
                assert metrics._maximum_assignment(matrix) == result
    assert metrics._maximum_assignment([[1, 1], [1, 1]]) == [(0, 0), (1, 1)]
    assert metrics._maximum_assignment([]) == []
    assert metrics._maximum_assignment([[]]) == []
    with pytest.raises(ValueError, match="rectangular"):
        metrics._maximum_assignment([[1, 2], [1]])


def test_perfect_measured_boxes_ignore_estimates_and_arbitrary_starting_id():
    pair = example([[gt()]] * 3, [[observed(900)]] * 3)
    before = deepcopy(pair)
    report = evaluate(pair)
    lane = report["lanes"][0]
    assert lane["counts"]["true_positives"] == 3
    assert lane["counts"]["identity_switches"] == lane["counts"]["fragments"] == 0
    assert lane["precision"] == lane["recall"] == 1
    assert lane["identity"] == {
        "available": True,
        "reason": None,
        "idtp": 3,
        "idfp": 0,
        "idfn": 0,
        "idf1": 1,
    }
    assert report["coverage"]["dense"] is True
    assert report["coverage"]["evaluated_transitions"] == 2
    assert pair == before  # Inputs remain frozen and hashes are not replaced.
    assert json.dumps(evaluate(pair), sort_keys=True) == json.dumps(report, sort_keys=True)


@pytest.mark.parametrize("last_id, switches, idtp, idf1", [(1, 0, 2, 0.8), (2, 1, 1, 0.4)])
def test_miss_and_recovery_distinguish_fragment_from_identity_switch(last_id, switches, idtp, idf1):
    lane = first_lane(example([[gt()]] * 3, [[observed(1)], [], [observed(last_id)]]))
    assert lane["counts"]["false_negatives"] == 1
    assert lane["counts"]["fragments"] == 1
    assert lane["counts"]["identity_switches"] == switches
    assert lane["identity"]["idtp"] == idtp
    assert lane["identity"]["idf1"] == pytest.approx(idf1)
    assert lane["frames"][1]["false_negatives"] == ["A"]
    assert {event["kind"] for event in lane["frames"][2]["events"]} == (
        {"fragment", "identity_switch"} if switches else {"fragment"}
    )


def test_id_split_with_no_missed_boxes_reduces_idf1_but_is_not_a_fragment():
    lane = first_lane(example([[gt()]] * 4, [[observed(i)] for i in (1, 1, 2, 2)]))
    assert lane["precision"] == lane["recall"] == 1
    assert lane["identity"]["idf1"] == 0.5
    assert lane["counts"]["identity_switches"] == 1
    assert lane["counts"]["fragments"] == 0


def test_initial_or_trailing_misses_are_not_fragments():
    lane = first_lane(example([[gt()]] * 4, [[], [observed()], [observed()], []]))
    assert lane["counts"]["false_negatives"] == 2
    assert lane["counts"]["fragments"] == 0


def test_two_reference_identities_swap_track_ids_and_transfer_without_extra_fn_penalty():
    objects = [[gt("A"), gt("B", FAR)]] * 2
    predictions = [[observed(1), observed(2, FAR)], [observed(2), observed(1, FAR)]]
    lane = first_lane(example(objects, predictions))
    assert lane["counts"]["identity_switches"] == 2
    assert lane["counts"]["identity_transfers"] == 2
    assert lane["counts"]["false_negatives"] == lane["counts"]["false_positives"] == 0
    assert lane["counts"]["fragments"] == 0
    assert lane["identity"]["idf1"] == 0.5


def test_global_identity_assignment_uses_all_overlaps_not_only_local_matches():
    # Local deterministic matching uses tracker1 twice. The global optimum is
    # B->1, A->2, which has two identity TPs rather than the naive single TP.
    pair = example([[gt("B")], [gt("A")]], [[observed(1)], [observed(1), observed(2)]])
    lane = first_lane(pair)
    assert [frame["matches"][0]["track_id"] for frame in lane["frames"]] == [1, 1]
    assert lane["identity"] == {
        "available": True,
        "reason": None,
        "idtp": 2,
        "idfp": 1,
        "idfn": 0,
        "idf1": 0.8,
    }


def test_previous_frame_correspondence_precedes_better_iou_in_ambiguous_overlap():
    near = [2, 0, 12, 10]
    pair = example(
        [[gt("A"), gt("B", FAR)], [gt("A"), gt("B", near)]],
        [[observed(1), observed(2, FAR)], [observed(1, near), observed(2)]],
    )
    lane = first_lane(pair)
    assert lane["counts"]["identity_switches"] == 0
    assert [
        (match["reference_identity"], match["track_id"]) for match in lane["frames"][1]["matches"]
    ] == [("A", 1), ("B", 2)]
    assert lane["frames"][1]["matches"][0]["iou"] == pytest.approx(2 / 3)


def test_empty_tracker_update_clears_previous_frame_priority_but_preserves_switch_memory():
    near = [2, 0, 12, 10]
    pair = example(
        [[gt("A"), gt("B", FAR)], [gt("A"), gt("B", near)], [gt("A"), gt("B", near)]],
        [[observed(1), observed(2, FAR)], [], [observed(1, near), observed(2)]],
    )
    lane = first_lane(pair)
    assert lane["counts"]["identity_switches"] == 2
    assert lane["counts"]["fragments"] == 2


@pytest.mark.parametrize("kind", ["missing", "assistant", "partial", "unreviewed", "unlocalized"])
def test_excluded_reference_frames_never_count_negative_or_bridge_events(kind):
    pair = example([[gt()]] * 3, [[observed(1)], [observed(99, FAR)], [observed(2)]])
    payload = pair[1]["payload"]
    middle = payload["frames"][1]
    if kind == "missing":
        payload["frames"].pop(1)
    elif kind == "assistant":
        middle["review"]["status"] = "assistant_reviewed"
    elif kind == "partial":
        middle["coverage"] = "partial"
    elif kind == "unreviewed":
        middle["coverage"] = "unreviewed"
        middle["review"] = {"status": "unreviewed", "reviewer": ""}
    else:
        middle["objects"] = [gt(visibility="occluded", box=None)]
    report = evaluate(pair)
    lane = report["lanes"][0]
    assert report["coverage"]["evaluated_frame_indices"] == [0, 2]
    assert report["coverage"]["evaluated_transitions"] == 0
    assert lane["counts"]["false_positives"] == lane["counts"]["false_negatives"] == 0
    assert lane["counts"]["identity_switches"] == lane["counts"]["fragments"] == 0
    assert lane["identity"]["idf1"] is None
    assert lane["identity"]["reason"] == "incomplete_reference_coverage"
    assert lane["frames"][1]["evaluated"] is False
    assert lane["frames"][1]["false_positives"] == []


def test_source_gaps_break_events_without_inventing_reference_or_negative_frames():
    report = evaluate(example([[gt()], [gt()]], [[observed(1)], [observed(2)]], indices=[0, 2]))
    assert report["coverage"]["source_frames"] == 3
    assert report["coverage"]["available_frames"] == 2
    assert report["coverage"]["excluded_frames"] == 0
    assert report["coverage"]["evaluated_transitions"] == 0
    lane = report["lanes"][0]
    assert lane["counts"]["identity_switches"] == 0
    assert lane["identity"]["idf1"] is None


@pytest.mark.parametrize("middle", [[], [gt(box=None, visibility="out_of_view")]])
def test_reference_absence_resets_local_events_but_global_idf1_keeps_identity(middle):
    lane = first_lane(example([[gt()], middle, [gt()]], [[observed(1)], [], [observed(2)]]))
    assert lane["counts"]["identity_switches"] == lane["counts"]["fragments"] == 0
    assert lane["identity"]["idf1"] == 0.5


def test_explicit_negative_counts_false_positive_and_empty_denominators_are_not_perfect():
    lane = first_lane(example([[], []], [[observed()], []]))
    assert lane["counts"]["false_positives"] == 1
    assert lane["precision"] == 0
    assert lane["recall"] is None
    assert lane["identity"]["reason"] == "no_ground_truth_identity_detections"
    empty = first_lane(example([[]], [[]]))
    assert empty["precision"] is empty["recall"] is empty["identity"]["idf1"] is None


def test_known_reference_without_tracker_outputs_has_zero_recall_and_zero_idf1():
    lane = first_lane(example([[gt()]], [[]]))
    assert lane["precision"] is None
    assert lane["recall"] == 0
    assert lane["identity"]["idf1"] == 0
    assert lane["identity"]["idfn"] == 1


def test_all_unreviewed_report_is_unknown_not_a_good_empty_score():
    pair = example([[gt()]], [[observed()]])
    frame = pair[1]["payload"]["frames"][0]
    frame["coverage"] = "unreviewed"
    frame["review"] = {"status": "unreviewed", "reviewer": ""}
    lane = first_lane(pair)
    assert lane["counts"]["true_positives"] == lane["counts"]["false_negatives"] == 0
    assert lane["precision"] is lane["recall"] is None
    assert lane["identity"]["reason"] == "no_evaluated_frames"


def test_boxed_occlusion_is_localizable_and_boxless_other_class_does_not_poison_scope():
    pair = example(
        [[gt(visibility="occluded"), gt("C", box=None, label="car", visibility="occluded")]],
        [[observed(), observed(2, FAR, label_id=3)]],
    )
    report = evaluate(pair, class_mapping={"1": "person", "3": None})
    assert report["coverage"]["dense"] is True
    assert report["lanes"][0]["counts"]["true_positives"] == 1
    assert report["lanes"][0]["counts"]["false_positives"] == 0


def test_unconfirmed_predictions_and_unassigned_boxes_do_not_earn_matches():
    pair = example([[gt()]], [[observed(confirmed=False)]])
    for lane in pair[0]["report"]["lanes"]:
        frame = lane["report"]["passes"][0]["frames"][0]
        frame["predictions"] = [observed(2)]
        frame["unassigned"] = [{"label_id": 1, "box": BOX}]
    lane = first_lane(pair)
    assert lane["counts"]["observations"] == 0
    assert lane["counts"]["false_negatives"] == 1
    assert lane["counts"]["excluded_predictions"] == 1
    assert lane["counts"]["excluded_unconfirmed"] == 1
    assert lane["counts"]["excluded_unassigned"] == 1


def test_wrong_class_overlap_remains_fp_fn_with_one_to_one_diagnostic():
    pair = example([[gt()]], [[observed(1, label_id=3), observed(2, label_id=3)]])
    lane = first_lane(pair)
    assert lane["counts"]["class_confusions"] == 1
    assert lane["counts"]["false_positives"] == 2
    assert lane["counts"]["false_negatives"] == 1
    assert lane["identity"]["idf1"] == 0
    assert lane["frames"][0]["class_confusions"] == [
        {
            "reference_identity": "A",
            "track_id": 1,
            "reference_label": "person",
            "observed_label": "car",
            "iou": 1,
        }
    ]


def test_iou_threshold_is_inclusive_and_uses_measured_coordinates():
    pair = example([[gt()]], [[observed(box=[0, 0, 20, 10])]])
    assert first_lane(pair)["counts"]["true_positives"] == 1
    assert first_lane(pair, iou_threshold=math.nextafter(0.5, 1))["counts"]["true_positives"] == 0


@pytest.mark.parametrize("threshold", [True, None, "0.5", 0, -1, 1.01, math.inf, math.nan, 10**400])
def test_invalid_threshold_is_rejected_without_coercion(threshold):
    with pytest.raises(ValueError, match="IoU threshold"):
        evaluate(example([[gt()]], [[observed()]]), iou_threshold=threshold)


@pytest.mark.parametrize(
    "mapping",
    [
        None,
        {},
        {1: "person", 3: "car"},
        {"01": "person", "3": "car"},
        {"1": "Person", "3": "car"},
        {"1": [], "3": "car"},
        {"1": None, "3": None},
    ],
)
def test_mapping_requires_explicit_exact_frozen_taxonomy_scope(mapping):
    with pytest.raises(ValueError):
        evaluate(example([[gt()]], [[observed()]]), class_mapping=mapping)


def test_source_frame_coverage_and_reference_mismatch_are_rejected():
    pair = example([[gt()]] * 2, [[observed()]] * 2)
    pair[0]["report"]["lanes"][1]["report"]["passes"][0]["frames"].pop()
    with pytest.raises(ValueError, match="complete frozen"):
        evaluate(pair)
    pair = example([[gt()]], [[observed()]])
    pair[1]["payload"]["sequence_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="sequence"):
        evaluate(pair)


@pytest.mark.parametrize(
    "limit, value, error",
    [
        ("MAX_FRAMES", 1, "source frames"),
        ("MAX_FRAME_OBJECTS", 1, "boxes per"),
        ("MAX_IDENTITY_COUNT", 1, "identities per lane"),
        ("MAX_ASSIGNMENT_WORK", 1, "matching workload"),
    ],
)
def test_caps_fail_explicitly_instead_of_truncating_data(monkeypatch, limit, value, error):
    pair = example([[gt()]] * 2, [[observed()]] * 2)
    monkeypatch.setattr(metrics, limit, value)
    with pytest.raises(ValueError, match=error):
        evaluate(pair)


def test_engine_import_does_not_require_optional_tracking_or_ml_dependencies():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from iris.tracking_metrics import quality_status; quality_status(); "
            "assert not {'torch', 'scipy', 'lap', 'cython_bbox', 'ultralytics'} & set(sys.modules)",
        ],
        check=True,
    )
