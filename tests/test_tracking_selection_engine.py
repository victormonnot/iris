"""Controlled selected-object failures and recoveries, independent of native trackers."""

import json
import math
import subprocess
import sys
from copy import deepcopy

import pytest
from test_tracking_metrics import example, gt

from iris.temporal_contracts import validate_sequence_manifest
from iris.tracking_contracts import FRAME_SCHEMA, make_profile, profile_hash
from iris.tracking_selection_contracts import (
    DEFAULT_POLICY,
    canonicalize_request,
    digest,
    selection_status,
)
from iris.tracking_selection_engine import run_selection, validate_report

BOX = [10, 10, 20, 30]
NEAR = [12, 10, 22, 30]
FAR = [70, 50, 80, 70]


def observed(track_id=17, box=BOX, *, detection_index=0, score=0.9, confirmed=True, label_id=1):
    return {
        "track_id": track_id,
        "detection_index": detection_index,
        "label_id": label_id,
        "label": "person" if label_id == 1 else "car",
        "score": score,
        "box": list(box),
        "estimated_box": list(FAR),
        "confirmed": confirmed,
    }


def bundle(observations, objects=None, *, indices=None, evaluation=True, unknown_clock=False):
    objects = objects if objects is not None else [[gt("A", BOX)] for _ in observations]
    comparison, reference = example(objects, [[] for _ in observations], indices=indices)
    manifest = comparison["report"]["sequence"]
    if unknown_clock:
        manifest["clock"] = {
            "basis": "unknown",
            "fps": None,
            "provenance": "Synthetic unknown clock",
        }
        for frame in manifest["frames"]:
            frame["timestamp_seconds"] = None
        from iris.temporal_contracts import sequence_hash

        reference["payload"]["sequence_sha256"] = sequence_hash(manifest)
        reference["payload_sha256"] = digest(reference["payload"])
    manifest = validate_sequence_manifest(manifest)
    profile = make_profile("bytetrack", class_ids=[1])
    frames = [
        {
            "schema": FRAME_SCHEMA,
            "sequence_id": manifest["id"],
            "frame_id": source["frame_id"],
            "frame_index": source["frame_index"],
            "timestamp_seconds": source["timestamp_seconds"],
            "input_size": [source["width"], source["height"]],
            "update_index": index,
            "observations": deepcopy(rows),
            "predictions": [],
            "unassigned": [],
            "gmc": {"method": "none", "status": "disabled", "matrix": None, "downscale": 2},
            "timing": {"gmc_ms": 0, "association_ms": 1, "total_ms": 1},
        }
        for index, (source, rows) in enumerate(
            zip(manifest["frames"], observations, strict=True), 1
        )
    ]
    source = {
        "kind": "comparison",
        "job_id": "comparison-1",
        "sequence_id": manifest["id"],
        "profile_sha256": profile_hash(profile),
    }
    request = {
        "name": "Controlled selection",
        "source": source,
        "selection": {"frame_id": frames[0]["frame_id"], "detection_index": 0},
        "release_frame_id": None,
        "policy": deepcopy(DEFAULT_POLICY),
        "max_seconds": 60,
        "evaluation": {
            "reference_id": reference["id"],
            "identity_id": "A",
            "class_mapping": {"1": "person"},
            "iou_threshold": 0.5,
        }
        if evaluation
        else None,
    }
    return {
        "request": request,
        "sequence": {"manifest": manifest},
        "replay": {
            "sequence": manifest,
            "profile": profile,
            "profile_sha256": profile_hash(profile),
            "passes": [{"frames": frames}],
            "repeatability": {"status": "not_checked", "passes": 1, "semantic_sha256": ["a" * 64]},
        },
        "reference": reference if evaluation else None,
        "source_binding": {"source": source},
        "fingerprint": "f" * 64,
    }


def lane(value, index=1):
    return run_selection(value)["lanes"][index]


def states(value, index=1):
    return [row["state"] for row in lane(value, index)["frames"]]


def update_reference(value):
    value["reference"]["payload_sha256"] = digest(value["reference"]["payload"])


def test_changed_id_recovers_same_selected_object_without_counting_id_as_identity():
    value = bundle([[observed()], [], [observed(33)], [observed(33)]])
    original = deepcopy(value)
    report = run_selection(value)
    baseline, guarded = report["lanes"]
    assert [row["state"] for row in guarded["frames"]] == [
        "observed",
        "lost",
        "recovering",
        "recovered",
    ]
    assert [row["state"] for row in baseline["frames"]] == ["observed", "lost", "lost", "lost"]
    assert guarded["summary"]["track_id_changes"] == 1
    assert {row["logical_object_id"] for row in guarded["frames"]} == {"selection-1"}
    assert guarded["quality"]["counts"]["wrong_other_identity"] == 0
    assert guarded["quality"]["recoveries"] == {
        "total": 1,
        "correct": 1,
        "wrong": 0,
        "unavailable": 0,
        "events": [{"frame_index": 3, "outcome": "correct"}],
    }
    assert guarded["quality"]["durations"]["unobserved_seconds"] == pytest.approx(0.2)
    assert value == original
    assert validate_report(value, json.loads(json.dumps(report))) == report


def test_same_id_after_one_missing_frame_also_requires_fresh_confirmation():
    value = bundle([[observed()], [], [observed()], [observed()], [observed()]])
    assert states(value) == ["observed", "lost", "recovering", "recovered", "observed"]
    assert states(value, 0) == ["observed", "lost", "recovered", "observed", "observed"]
    assert lane(value)["summary"]["track_id_changes"] == 0


def test_direct_id_change_is_not_a_physical_identity_error():
    value = bundle([[observed()], [observed(99)], [observed(99)]])
    assert states(value) == ["observed", "recovering", "recovered"]
    assert lane(value)["quality"]["recoveries"]["correct"] == 1


def test_unchanged_id_cannot_bypass_geometry_or_competition():
    value = bundle(
        [[observed()], [observed(box=FAR)]], [[gt("A", BOX)], [gt("A", BOX), gt("B", FAR)]]
    )
    assert states(value) == ["observed", "lost"]
    assert lane(value, 0)["quality"]["counts"]["wrong_other_identity"] == 1
    assert "insufficient_overlap" in lane(value)["frames"][1]["candidates"][0]["reasons"]
    competition = bundle(
        [
            [observed()],
            [observed(), observed(21, NEAR, detection_index=1)],
            [observed()],
            [observed()],
        ]
    )
    assert states(competition) == ["observed", "ambiguous", "recovering", "recovered"]
    assert lane(competition)["frames"][1]["selected"] is None


def test_lookalike_counterexample_remains_an_explicit_wrong_recovery():
    # A leaves; B alone occupies the same geometry. Pure geometry cannot know this.
    absent = gt("A", None, visibility="out_of_view")
    value = bundle(
        [[observed()], [observed(99)], [observed(99)]],
        [[gt("A", BOX)], [absent, gt("B", BOX)], [absent, gt("B", BOX)]],
    )
    guarded = lane(value)
    assert states(value) == ["observed", "recovering", "recovered"]
    assert guarded["quality"]["counts"]["wrong_other_identity"] == 1
    assert guarded["quality"]["counts"]["abstained_absent"] == 1
    assert guarded["quality"]["recoveries"]["wrong"] == 1
    assert "lookalike" in run_selection(value)["limitations"][0]


def test_same_id_lookalike_counterexample_is_not_hidden_by_policy():
    value = bundle(
        [[observed()], [observed()]],
        [[gt("A", BOX)], [gt("A", None, visibility="out_of_view"), gt("B", BOX)]],
    )
    assert states(value) == ["observed", "observed"]
    assert lane(value)["quality"]["counts"]["wrong_other_identity"] == 1


def test_prediction_and_unassigned_boxes_never_refresh_selection():
    value = bundle([[observed()], [], [], []])
    for index, frame in enumerate(value["replay"]["passes"][0]["frames"][1:], 1):
        frame["predictions"] = [
            {
                "track_id": 17,
                "label_id": 1,
                "label": "person",
                "box": BOX,
                "confirmed": True,
                "last_observed_frame_id": "frame-0",
                "last_observed_frame_index": 0,
                "last_observed_timestamp_seconds": 0.0,
                "last_observed_update_index": 1,
                "age_updates": index,
                "age_seconds": index / 10,
            }
        ]
        frame["unassigned"] = [
            {
                "detection_index": 2,
                "label_id": 1,
                "label": "person",
                "score": 0.2,
                "box": BOX,
                "reason": "unmatched_low_confidence",
            }
        ]
    value["request"]["policy"]["max_lost_updates"] = 2
    assert states(value) == ["observed", "lost", "lost", "expired"]
    assert lane(value)["frames"][-1]["last_observed"]["frame_id"] == "frame-0"


def test_pending_candidates_never_extend_timeout_or_chain_through_geometry_drift():
    value = bundle([[observed()], [], [observed(99)], [observed(99)], [observed()]])
    value["request"]["policy"]["max_lost_updates"] = 2
    assert states(value) == ["observed", "lost", "recovering", "expired", "expired"]
    value = bundle([[observed()], [observed(99)], [observed(100)], [observed(99)], [observed(99)]])
    assert states(value) == ["observed", "recovering", "recovering", "recovering", "recovered"]


def test_source_gap_clears_confirmation_and_withholds_recovery_episode_claim():
    value = bundle(
        [[observed()], [observed(99)], [observed(99)], [observed(99)]], indices=[0, 1, 3, 4]
    )
    assert states(value) == ["observed", "recovering", "recovering", "recovered"]
    quality = lane(value)["quality"]
    assert quality["counts"]["correct_target"] == 2
    assert quality["recoveries"]["unavailable"] == 1
    assert quality["durations"]["supported_intervals"] == 2
    assert quality["durations"]["excluded_intervals"] == 1
    assert lane(value)["frames"][2]["source_gap"] is True


def test_unknown_clock_uses_updates_without_inventing_seconds():
    value = bundle([[observed()], [], [], [observed()]], unknown_clock=True)
    value["request"]["policy"]["max_lost_updates"] = 2
    value["request"]["policy"]["max_lost_seconds"] = 0.001
    assert states(value) == ["observed", "lost", "lost", "expired"]
    assert all(row["age"]["seconds"] is None for row in lane(value)["frames"])
    assert lane(value)["quality"]["durations"]["evaluated_seconds"] is None


def test_seconds_timeout_and_explicit_release_are_terminal():
    value = bundle([[observed()], [], [observed()], [observed()], [observed()]])
    value["request"]["policy"]["max_lost_seconds"] = 0.15
    value["request"]["release_frame_id"] = "frame-3"
    result = lane(value)
    assert [row["state"] for row in result["frames"]] == [
        "observed",
        "lost",
        "expired",
        "released",
        "released",
    ]
    assert result["summary"]["active_frames"] == 3
    assert result["frames"][3]["event"] == "released"
    assert result["quality"]["coverage"]["active_frames"] == 3


def test_later_selection_and_null_seconds_limit_are_explicit():
    value = bundle([[observed()], [observed()], [observed()]])
    value["request"]["selection"]["frame_id"] = "frame-1"
    value["request"]["policy"]["max_lost_seconds"] = None
    assert states(value) == ["idle", "observed", "observed"]
    assert lane(value)["frames"][0]["logical_object_id"] is None


@pytest.mark.parametrize("kind", ["assistant", "partial", "missing", "occluded"])
def test_unknown_reference_intervals_are_not_negative_or_recovery_evidence(kind):
    value = bundle([[observed()], [], [observed(99)], [observed(99)]])
    frame = value["reference"]["payload"]["frames"][1]
    if kind == "assistant":
        frame["review"]["status"] = "assistant_reviewed"
    elif kind == "partial":
        frame["coverage"] = "partial"
    elif kind == "missing":
        value["reference"]["payload"]["frames"].pop(1)
    else:
        frame["objects"][0].update(visibility="occluded", box=None)
    update_reference(value)
    quality = lane(value)["quality"]
    assert quality["coverage"]["evaluated_frames"] == 3
    assert quality["counts"]["abstained_visible"] == 1
    assert quality["durations"]["supported_intervals"] == 1
    assert quality["recoveries"]["correct"] == 0
    assert quality["recoveries"]["unavailable"] == 1


def test_reference_overlap_is_ambiguous_not_a_track_identity_claim():
    value = bundle([[observed()], [observed()]], [[gt("A", BOX)], [gt("A", BOX), gt("B", NEAR)]])
    quality = lane(value)["quality"]
    assert quality["counts"]["reference_ambiguous"] == 1
    assert quality["coverage"]["evaluated_frames"] == 1
    assert quality["durations"]["evaluated_seconds"] is None


def test_false_positive_and_known_absence_are_kept_separate():
    value = bundle(
        [[observed()], [observed()], []],
        [
            [gt("A", BOX)],
            [gt("A", None, visibility="out_of_view")],
            [gt("A", None, visibility="out_of_view")],
        ],
    )
    quality = lane(value)["quality"]
    assert quality["counts"]["unmatched_selected_box"] == 1
    assert quality["counts"]["abstained_absent"] == 1
    assert quality["counts"]["wrong_other_identity"] == 0
    assert quality["rates"]["target_agreement"] == 1


def test_reference_annotations_cannot_change_policy_output():
    first = bundle([[observed()], [observed(99)], [observed(99)]])
    second = deepcopy(first)
    for frame in second["reference"]["payload"]["frames"][1:]:
        frame["objects"][0]["box"] = FAR
    update_reference(second)
    a, b = run_selection(first), run_selection(second)
    assert [item["frames"] for item in a["lanes"]] == [item["frames"] for item in b["lanes"]]
    assert a["lanes"][1]["quality"] != b["lanes"][1]["quality"]


def test_missing_reference_and_semantic_mismatch_explicitly_withhold_quality():
    value = bundle([[observed()], [observed()]], evaluation=False)
    quality = lane(value)["quality"]
    assert quality["status"] == "unavailable" and quality["reason"] == "no_reference_selected"
    assert all(rate is None for rate in quality["rates"].values())
    value = bundle([[observed()], [observed(99)], [observed(99)]])
    value["replay"]["repeatability"]["status"] = "observed_mismatch"
    quality = lane(value)["quality"]
    assert quality["reason"] == "source_replay_semantic_mismatch"
    assert quality["recoveries"]["unavailable"] == 1
    assert quality["coverage"]["evaluated_frames"] == 0


@pytest.mark.parametrize(
    "mutation",
    [
        "unconfirmed",
        "score",
        "wrong_index",
        "missing_frame",
        "same_release",
        "early_release",
        "initial_wrong_gt",
        "initial_ambiguous",
        "mapping",
    ],
)
def test_invalid_initial_intent_or_assessment_is_rejected(mutation):
    value = bundle([[observed()], [observed()], [observed()]])
    frames = value["replay"]["passes"][0]["frames"]
    if mutation == "unconfirmed":
        frames[0]["observations"][0]["confirmed"] = False
    elif mutation == "score":
        frames[0]["observations"][0]["score"] = 0.1
    elif mutation == "wrong_index":
        value["request"]["selection"]["detection_index"] = 7
    elif mutation == "missing_frame":
        value["request"]["selection"]["frame_id"] = "missing"
    elif mutation == "same_release":
        value["request"]["release_frame_id"] = "frame-0"
    elif mutation == "early_release":
        value["request"]["selection"]["frame_id"] = "frame-2"
        value["request"]["release_frame_id"] = "frame-1"
    elif mutation == "initial_wrong_gt":
        value["reference"]["payload"]["frames"][0]["objects"][0]["box"] = FAR
        update_reference(value)
    elif mutation == "initial_ambiguous":
        value["reference"]["payload"]["identities"].append({"id": "B", "label": "person"})
        value["reference"]["payload"]["frames"][0]["objects"].append(gt("B", NEAR))
        update_reference(value)
    else:
        value["request"]["evaluation"]["class_mapping"] = {"1": "car"}
    with pytest.raises(ValueError):
        run_selection(value)


@pytest.mark.parametrize(
    ("key", "bad"),
    [
        ("min_score", True),
        ("min_score", math.nan),
        ("min_iou", -0.1),
        ("max_center_distance", math.inf),
        ("max_area_ratio", 0.5),
        ("max_lost_seconds", 0),
        ("max_lost_seconds", 10**1000),
        ("max_lost_updates", True),
        ("max_lost_updates", 1001),
        ("recovery_confirmation_updates", 1),
        ("recovery_confirmation_updates", 2.0),
    ],
)
def test_policy_rejects_nonfinite_bool_or_out_of_bounds_values(key, bad):
    value = bundle([[observed()]])
    value["request"]["policy"][key] = bad
    with pytest.raises(ValueError):
        canonicalize_request(value["request"])


def test_strict_request_rejects_unknown_fields_invalid_utf8_and_source_kinds():
    value = bundle([[observed()]])
    for mutation in (
        {**value["request"], "surprise": 1},
        {**value["request"], "name": "bad\ud800"},
        {**value["request"], "source": {**value["request"]["source"], "kind": "training"}},
    ):
        with pytest.raises(ValueError):
            canonicalize_request(mutation)
    status = selection_status()
    status["default_policy"]["min_score"] = 0
    assert selection_status()["default_policy"]["min_score"] == 0.3


def test_rehashed_report_edits_are_recomputed_and_cancel_size_bounds_are_effective(monkeypatch):
    value = bundle([[observed()], [observed()]])
    report = run_selection(value)
    report["lanes"][1]["frames"][1]["selected"]["track_id"] = 999
    with pytest.raises(ValueError, match="frozen evidence"):
        validate_report(value, report)
    calls = []

    def cancelled():
        calls.append(1)
        if len(calls) == 4:
            raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        run_selection(value, checkpoint=cancelled)
    from iris import tracking_selection_contracts

    monkeypatch.setattr(tracking_selection_contracts, "MAX_REPORT_BYTES", 100)
    with pytest.raises(ValueError, match="bounded size"):
        run_selection(value)


def test_engine_imports_without_tracker_or_ml_libraries():
    code = """
import builtins
original = builtins.__import__
def checked(name, *args, **kwargs):
    if name.split('.')[0] in {'torch','torchvision','numpy','cv2','scipy','lap'}:
        raise AssertionError(name)
    return original(name, *args, **kwargs)
builtins.__import__ = checked
from iris.tracking_selection_engine import run_selection
from iris.tracking_selection_contracts import selection_status
from iris.tracking_selection_metrics import prepare_evaluation
assert selection_status()['limits']['max_frames'] == 500
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_decimal_timestamp_roundoff_does_not_expire_at_inclusive_limit():
    value = bundle([[observed()], [observed()], [], [], [observed()], [observed()]])
    value["request"]["selection"]["frame_id"] = "frame-1"
    value["request"]["policy"]["max_lost_seconds"] = 0.3
    # 0.4 - 0.1 is 0.30000000000000004 in binary floating point.
    assert states(value, 0) == ["idle", "observed", "lost", "lost", "recovered", "observed"]
    assert states(value) == ["idle", "observed", "lost", "lost", "recovering", "expired"]


def test_low_or_unconfirmed_candidates_cannot_confirm_recovery():
    value = bundle(
        [
            [observed()],
            [observed(score=0.2)],
            [observed()],
            [observed(confirmed=False)],
            [observed()],
            [observed()],
        ]
    )
    assert states(value) == ["observed", "lost", "recovering", "lost", "recovering", "recovered"]
    assert "low_score" in lane(value)["frames"][1]["candidates"][0]["reasons"]
    assert "unconfirmed" in lane(value)["frames"][3]["candidates"][0]["reasons"]


def test_selection_is_generic_across_native_classes_and_uses_explicit_taxonomy_mapping():
    value = bundle(
        [[observed(label_id=3)], [observed(label_id=3), observed(42, detection_index=1)]],
        [[gt("A", BOX, label="car")], [gt("A", BOX, label="car")]],
    )
    profile = make_profile("bytetrack", class_ids=[1, 3])
    value["replay"]["profile"] = profile
    value["replay"]["profile_sha256"] = profile_hash(profile)
    value["request"]["source"]["profile_sha256"] = profile_hash(profile)
    value["request"]["evaluation"]["class_mapping"] = {"1": None, "3": "car"}
    selected = lane(value)
    assert [row["state"] for row in selected["frames"]] == ["observed", "observed"]
    assert selected["quality"]["rates"]["target_agreement"] == 1
    assert "different_class" in selected["frames"][1]["candidates"][1]["reasons"]


def test_pending_candidate_geometry_must_be_coherent_as_well_as_with_anchor():
    # Both candidates overlap the anchor, but alternate between its opposite sides.
    value = bundle(
        [
            [observed()],
            [observed(99, [1, 10, 11, 30])],
            [observed(99, [19, 10, 29, 30])],
            [observed(99, [19, 10, 29, 30])],
        ],
        evaluation=False,
    )
    assert states(value) == ["observed", "recovering", "recovering", "recovered"]
    assert lane(value)["frames"][2]["pending"]["observations"] == 1
