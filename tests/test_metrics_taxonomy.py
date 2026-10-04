"""Generic COCO metrics use frozen semantic IDs rather than native detector numbers."""

from copy import deepcopy

import pytest

from iris.metrics import evaluate_predictions, get_protocol
from iris.taxonomies import TAXONOMY

TAXONOMY_CUSTOM = {
    "id": "taxonomy-" + "1" * 32,
    "version": 2,
    "parent_id": TAXONOMY["id"],
    "box_format": TAXONOMY["box_format"],
    "review_guidance": TAXONOMY["review_guidance"],
    "created_at": "2026-10-04T12:00:00+00:00",
    "classes": [
        {"id": "helmet", "name": "Helmet", "definition": "A protective helmet."},
        {"id": "car", "name": "Vehicle", "definition": "Custom passenger vehicle.", "coco_id": 3},
        {"id": "marker", "name": "Marker", "definition": "An unmapped marker."},
    ],
}


def calculate(boxes, detections, **kwargs):
    return evaluate_predictions(
        [{"frame_id": "frame", "width": 40, "height": 30, "boxes": boxes}],
        [{"frame_id": "frame", "detections": detections}],
        taxonomy=deepcopy(TAXONOMY_CUSTOM),
        **kwargs,
    )


def test_non_coco_class_and_native_coco_number_are_distinct_with_absent_classes():
    target = {"label": "car", "box": [1, 1, 20, 20]}
    result = calculate([target], [{**target, "label_id": 2, "score": 0.99}])
    assert result["summary"]["map"] == pytest.approx(1)
    assert result["summary"]["evaluated_classes"] == ["car"]
    assert [row["label"] for row in result["per_class"]] == ["helmet", "car", "marker"]
    assert result["per_class"][0]["ap"] is None
    assert result["per_class"][1]["support"] == 1
    assert result["per_class"][2]["ap"] is None
    assert result["protocol"]["classes"] == {"helmet": 1, "car": 2, "marker": 3}
    assert result["protocol"]["taxonomy"] == TAXONOMY_CUSTOM
    with pytest.raises(ValueError, match="frozen taxonomy mapping"):
        calculate([target], [{**target, "label_id": 3, "score": 0.99}])


def test_custom_negative_only_split_keeps_every_class_and_undefined_ap():
    result = calculate(
        [], [{"label": "helmet", "label_id": 1, "box": [1, 1, 20, 20], "score": 0.9}]
    )
    assert result["summary"]["map"] is None
    assert result["summary"]["recall"] is None
    assert result["summary"]["precision"] == 0
    assert result["summary"]["fp"] == 1
    assert len(result["per_class"]) == 3
    assert all(row["ap"] is None and row["support"] == 0 for row in result["per_class"])


def test_custom_protocol_keeps_coco_ap_cap_but_records_saved_output_limit():
    normal = get_protocol(taxonomy=TAXONOMY_CUSTOM)
    tiled = get_protocol(taxonomy=TAXONOMY_CUSTOM, max_detections=300)
    assert normal["id"] == tiled["id"] == "coco-bbox-iris-v3"
    assert normal["max_saved_detections_per_image"] == 100
    assert tiled["max_saved_detections_per_image"] == 300
    assert normal["max_dets"] == tiled["max_dets"] == [1, 10, 100]
    assert "two project classes" not in normal["operating_point"]["aggregation"]


@pytest.mark.parametrize(
    "mutation",
    [
        {"label_id": 3},
        {"label_id": True},
        {"label": "person"},
        {"taxonomy_id": "iris-objects-v1"},
        {"ignored": True},
    ],
)
def test_custom_namespace_cannot_silently_accept_native_or_unrelated_labels(mutation):
    detection = {"label": "car", "label_id": 2, "box": [1, 1, 20, 20], "score": 0.9, **mutation}
    with pytest.raises(ValueError):
        calculate([], [detection])
