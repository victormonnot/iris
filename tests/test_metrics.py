"""Analytical detection fixtures verify the metric protocol, not flight accuracy."""

import copy
import json

import pytest

from iris.metrics import evaluate_predictions, get_protocol


def box(label="person", coordinates=None):
    return {"label": label, "box": coordinates or [0, 0, 20, 20]}


def frame(identifier="frame", boxes=None):
    return {"frame_id": identifier, "width": 100, "height": 100, "boxes": boxes or []}


def detection(label="person", score=0.9, coordinates=None):
    return {
        **box(label, coordinates),
        "label_id": {"person": 1, "car": 3, "bicycle": 2}[label],
        "score": score,
    }


def row(identifier="frame", detections=None):
    return {"frame_id": identifier, "detections": detections or []}


def score_one(boxes=None, detections=None, **kwargs):
    return evaluate_predictions([frame(boxes=boxes)], [row(detections=detections)], **kwargs)


def test_perfect_two_classes_reports_official_coco_ap_and_micro_rates(capsys):
    result = score_one(
        [box(), box("car", [40, 40, 60, 60])],
        [detection(), detection("car", coordinates=[40, 40, 60, 60])],
    )
    assert capsys.readouterr().out == ""
    summary = result["summary"]
    for metric in ("map", "map50", "map75", "precision", "recall"):
        assert summary[metric] == pytest.approx(1)
    assert (summary["tp"], summary["fp"], summary["fn"]) == (2, 0, 0)
    assert summary["evaluated_classes"] == ["person", "car"]
    assert all(item["support"] == item["predictions"] == 1 for item in result["per_class"])
    assert result["frames"][0]["matches"] == [
        {"label": "person", "ground_truth_index": 0, "detection_index": 0, "iou": 1},
        {"label": "car", "ground_truth_index": 1, "detection_index": 1, "iou": 1},
    ]
    protocol = result["protocol"]
    assert protocol["id"] == "coco-bbox-iris-v1"
    assert protocol["engine_version"] == "2.0.11"
    assert protocol["max_dets"] == [1, 10, 100]
    assert protocol["ap_iou_thresholds"] == [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    assert protocol["ap_recall_thresholds"] == [index / 100 for index in range(101)]
    json.dumps(result, allow_nan=False)


def test_missed_class_contributes_zero_ap_when_it_has_ground_truth():
    result = score_one([box(), box("car")], [detection()])
    assert result["summary"]["map"] == pytest.approx(0.5)
    assert result["summary"]["recall"] == 0.5
    assert result["per_class"][1]["ap"] == 0
    assert result["per_class"][1]["precision"] is None
    assert result["frames"][0]["false_negatives"] == [1]


def test_duplicate_detections_are_false_positives_and_score_ties_are_stable():
    result = score_one([box()], [detection(), detection()])
    assert result["summary"]["precision"] == 0.5
    assert result["summary"]["map"] == pytest.approx(1)
    assert result["frames"][0]["matches"][0]["detection_index"] == 0
    assert result["frames"][0]["false_positives"] == [1]
    assert result["per_class"][1]["ap"] is None
    assert result["summary"]["evaluated_classes"] == ["person"]


def test_high_confidence_false_positive_on_negative_image_reduces_ap():
    frames = [frame("negative"), frame("positive", [box()])]
    predictions = [row("negative", [detection(score=0.9)]), row("positive", [detection(score=0.8)])]
    result = evaluate_predictions(frames, predictions, confidence_threshold=0.85)
    assert result["summary"]["map"] == pytest.approx(0.5)
    assert result["summary"]["precision"] == result["summary"]["recall"] == 0
    assert (result["summary"]["tp"], result["summary"]["fp"], result["summary"]["fn"]) == (0, 1, 1)
    assert result["frames"][0]["false_positives"] == [0]
    assert result["frames"][1]["false_negatives"] == [0]


def test_ap_includes_detections_below_operating_point_threshold():
    result = score_one([box()], [detection(score=0.01)], confidence_threshold=0.99)
    assert result["summary"]["map"] == pytest.approx(1)
    assert result["summary"]["prediction_count"] == 0
    assert result["summary"]["project_prediction_count_before_threshold"] == 1
    assert result["summary"]["precision"] is None
    assert result["summary"]["recall"] == 0


def test_recall_interpolation_uses_101_points_including_zero_and_half_recall():
    result = evaluate_predictions(
        [frame("first", [box()]), frame("second", [box()])],
        [row("first", [detection()]), row("second")],
    )
    assert result["summary"]["map"] == pytest.approx(51 / 101)
    assert result["summary"]["precision"] == 1
    assert result["summary"]["recall"] == 0.5


def test_wrong_class_is_both_false_positive_and_missed_ground_truth():
    result = score_one([box()], [detection("car")])
    assert result["summary"]["map"] == 0
    assert (result["summary"]["tp"], result["summary"]["fp"], result["summary"]["fn"]) == (0, 1, 1)
    assert result["per_class"][1]["ap"] is None
    assert result["per_class"][1]["precision"] == 0
    assert result["per_class"][1]["recall"] is None


@pytest.mark.parametrize("detections,precision,fp", [([], None, 0), ([detection()], 0, 1)])
def test_negative_only_dataset_has_undefined_ap_and_recall(detections, precision, fp):
    result = score_one([], detections)
    assert all(result["summary"][metric] is None for metric in ("map", "map50", "map75", "recall"))
    assert result["summary"]["precision"] == precision
    assert result["summary"]["fp"] == fp
    assert result["summary"]["evaluated_classes"] == []
    assert result["warnings"]
    json.dumps(result, allow_nan=False)


def test_empty_outputs_on_positive_image_score_zero_ap_and_recall():
    result = score_one([box()], [])
    assert result["summary"]["map"] == result["summary"]["recall"] == 0
    assert result["summary"]["precision"] is None
    assert result["frames"][0]["false_negatives"] == [0]


def test_unrelated_valid_coco_classes_are_reported_and_indices_preserved():
    result = score_one([box()], [detection("bicycle"), detection()])
    assert result["summary"]["map"] == pytest.approx(1)
    assert result["summary"]["ignored_prediction_count"] == 1
    assert result["summary"]["native_prediction_count"] == 2
    assert result["summary"]["prediction_count"] == 1
    assert result["frames"][0]["matches"][0]["detection_index"] == 1
    assert "Ignored 1" in result["warnings"][0]


def test_equal_scores_across_images_use_manifest_order_not_prediction_row_order():
    frames = [frame("negative"), frame("positive", [box()])]
    predictions = [row("positive", [detection()]), row("negative", [detection()])]
    assert evaluate_predictions(frames, predictions)["summary"]["map"] == pytest.approx(0.5)
    reversed_result = evaluate_predictions(list(reversed(frames)), predictions)
    assert reversed_result["summary"]["map"] == pytest.approx(1)


def test_equal_iou_ties_match_last_unmatched_ground_truth_like_coco():
    result = score_one([box(), box()], [detection()])
    assert result["frames"][0]["matches"][0]["ground_truth_index"] == 1
    assert result["frames"][0]["false_negatives"] == [0]
    assert result["summary"]["map"] == pytest.approx(51 / 101)


@pytest.mark.parametrize(
    "width,expected_ap,expected_ap50,expected_ap75,expected_tp",
    [(10, 0.1, 1, 0, 1), (15, 0.6, 1, 1, 1), (9.99, 0, 0, 0, 0)],
)
def test_iou_and_confidence_boundaries_are_inclusive(
    width, expected_ap, expected_ap50, expected_ap75, expected_tp
):
    result = score_one(
        [box(coordinates=[0, 0, 20, 10])],
        [detection(score=0.5, coordinates=[0, 0, width, 10])],
    )
    assert result["summary"]["map"] == pytest.approx(expected_ap)
    assert result["summary"]["map50"] == pytest.approx(expected_ap50)
    assert result["summary"]["map75"] == pytest.approx(expected_ap75)
    assert result["summary"]["tp"] == expected_tp


def test_operating_point_iou_threshold_is_independent_from_coco_ap_sweep():
    detections = [detection(coordinates=[0, 0, 15, 20])]
    result = score_one([box()], detections, iou_threshold=0.8)
    assert result["summary"]["map"] == pytest.approx(0.6)
    assert result["summary"]["tp"] == 0
    assert result["summary"]["fp"] == result["summary"]["fn"] == 1
    assert score_one([box()], detections, iou_threshold=0.75)["summary"]["tp"] == 1


def test_metrics_do_not_mutate_source_artifacts():
    frames, predictions = [frame(boxes=[box()])], [row(detections=[detection()])]
    before = copy.deepcopy((frames, predictions))
    evaluate_predictions(frames, predictions)
    assert (frames, predictions) == before


@pytest.mark.parametrize(
    "frames,predictions",
    [
        ([], []),
        ([frame()], []),
        ([frame()], [row("extra")]),
        ([frame()], [row(), row()]),
        ([frame(), frame()], [row()]),
        ([frame()], [{"frame_id": "frame"}]),
        ([{**frame(), "width": True}], [row()]),
        ([{**frame(), "height": 0}], [row()]),
        ([{**frame(), "boxes": None}], [row()]),
        ([frame(boxes=[box("bicycle")])], [row()]),
        ([frame(boxes=[box([])])], [row()]),
        ([frame(boxes=[{**box(), "iscrowd": 1}])], [row()]),
        ([frame(boxes=[{**box(), "ignore": 1}])], [row()]),
        ([frame()], [row(detections=[detection()] * 101)]),
    ],
)
def test_invalid_frame_sets_and_unsupported_protocols_fail_closed(frames, predictions):
    with pytest.raises(ValueError):
        evaluate_predictions(frames, predictions)


@pytest.mark.parametrize(
    "changes",
    [
        {"label_id": 2, "label": "car"},
        {"label_id": 1, "label": "car"},
        {"label_id": 0, "label": "__background__"},
        {"label_id": 12, "label": "N/A"},
        {"label_id": 999, "label": "unknown"},
        {"label_id": True},
        {"label_id": 1.0},
        {"score": float("nan")},
        {"score": float("inf")},
        {"score": -0.1},
        {"score": 1.1},
        {"score": True},
        {"score": 10**1000},
        {"box": [0, 0, 0, 20]},
        {"box": [-1, 0, 20, 20]},
        {"box": [0, 0, 101, 20]},
        {"box": [0, 0, 20, float("nan")]},
        {"box": [0, 0, True, 20]},
        {"box": [0, 0, 20]},
    ],
)
def test_malformed_native_detections_are_rejected(changes):
    with pytest.raises(ValueError):
        score_one([box()], [{**detection(), **changes}])


def test_invalid_ignored_coco_detection_is_still_rejected():
    with pytest.raises(ValueError):
        score_one([], [detection("bicycle", score=float("nan"))])


@pytest.mark.parametrize(
    "options",
    [
        {"confidence_threshold": -1},
        {"confidence_threshold": 1.1},
        {"confidence_threshold": float("nan")},
        {"confidence_threshold": True},
        {"iou_threshold": 0},
        {"iou_threshold": 1.1},
        {"iou_threshold": float("inf")},
    ],
)
def test_invalid_operating_point_is_rejected(options):
    with pytest.raises(ValueError):
        get_protocol(**options)
