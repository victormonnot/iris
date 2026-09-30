"""Merged outputs can exceed 100 boxes without changing COCO's AP limit."""

from copy import deepcopy

import pytest
from test_metrics import box, detection, frame, row

from iris.metrics import evaluate_predictions, get_protocol


def test_operating_point_counts_all_merged_boxes_but_coco_ap_retains_maxdets_100():
    boxes = [
        [10 * (i % 12), 10 * (i // 12), 10 * (i % 12) + 5, 10 * (i // 12) + 5] for i in range(120)
    ]
    truth = [
        {
            "frame_id": "dense",
            "width": 120,
            "height": 100,
            "boxes": [box(coordinates=value) for value in boxes],
        }
    ]
    predicted = [
        row(
            "dense",
            [detection(coordinates=value, score=0.99 - i / 1000) for i, value in enumerate(boxes)],
        )
    ]
    metrics = evaluate_predictions(truth, predicted, max_detections=300)
    assert metrics["summary"]["tp"] == 120
    assert metrics["summary"]["fp"] == metrics["summary"]["fn"] == 0
    assert metrics["summary"]["recall"] == metrics["summary"]["precision"] == 1
    assert metrics["summary"]["map"] == pytest.approx(84 / 101)
    assert metrics["summary"]["native_prediction_count"] == 120
    assert metrics["protocol"]["max_dets"] == [1, 10, 100]
    assert metrics["protocol"]["max_saved_detections_per_image"] == 300
    assert metrics["protocol"]["native_max_detections_per_call"] == 100
    assert metrics["protocol"]["id"] == "coco-bbox-iris-v2"
    assert "native_max_detections_per_image" not in metrics["protocol"]


def test_final_original_image_outputs_are_scored_without_mutating_raw_tiles():
    original = row(detections=[detection(), detection("car", coordinates=[40, 40, 60, 60])])
    original["metadata"] = {"tiles": [{"detections": [{"score": "raw diagnostic only"}]}]}
    before = deepcopy(original)
    truth = [frame(boxes=[box(), box("car", [40, 40, 60, 60])])]
    legacy = evaluate_predictions(truth, [original])
    tiled = evaluate_predictions(truth, [original], max_detections=300)
    assert original == before
    assert legacy.pop("protocol") == get_protocol()
    assert tiled.pop("protocol") == get_protocol(max_detections=300)
    assert legacy == tiled


@pytest.mark.parametrize("limit,count", [(100, 101), (300, 301)])
def test_pipeline_limit_is_validated_before_scoring(limit, count):
    with pytest.raises(ValueError, match=f"at most {limit}"):
        evaluate_predictions(
            [frame()], [row(detections=[detection()] * count)], max_detections=limit
        )


@pytest.mark.parametrize("limit", [True, 0, 99, 101, 300.0, 301, None, "300"])
def test_protocol_rejects_unsupported_output_limits(limit):
    with pytest.raises(ValueError, match="100 or 300"):
        get_protocol(max_detections=limit)
