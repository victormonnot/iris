"""Original-pixel tiling and merging, using explicitly synthetic detectors."""

from copy import deepcopy

import pytest
from PIL import Image

from iris.tiling import (
    MAX_DETECTIONS_PER_TILE,
    TiledInferenceCancelled,
    tile_boxes,
    tiled_predict,
    validate_tiling_config,
)

TIMING = {"preprocess_ms": 1.25, "inference_ms": 2.5, "postprocess_ms": 0.75, "total_ms": 4.5}


def detection(box=(0, 0, 12, 10), *, score=0.8, label=1, **extra):
    return {"box": list(box), "score": score, "label_id": label, "label": "fixture", **extra}


def prediction(size=(128, 128), detections=None):
    return {
        "input_size": list(size),
        "detections": detections if detections is not None else [detection()],
        "timing": TIMING.copy(),
    }


class FixtureDetector:
    def __init__(self, outputs=None):
        self.outputs = outputs
        self.calls = []

    def predict(self, image):
        self.calls.append({"size": image.size, "pixel": image.getpixel((0, 0))})
        if self.outputs is None:
            return prediction(image.size, [])
        return deepcopy(self.outputs[len(self.calls) - 1])


def test_defaults_are_explicit_and_normalized():
    assert validate_tiling_config() == {
        "tile_size": 640,
        "overlap": 0.2,
        "merge_iou": 0.5,
        "max_detections": 300,
    }
    assert validate_tiling_config(overlap=0)["overlap"] == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tile_size": True},
        {"tile_size": 127},
        {"tile_size": 2049},
        {"tile_size": 640.0},
        {"tile_size": "640"},
        {"overlap": True},
        {"overlap": -0.01},
        {"overlap": 0.5001},
        {"overlap": float("nan")},
        {"overlap": float("inf")},
        {"overlap": "0.2"},
        {"overlap": 10**400},
        {"merge_iou": False},
        {"merge_iou": 0.09},
        {"merge_iou": 0.91},
        {"merge_iou": float("nan")},
        {"max_detections": True},
        {"max_detections": 0},
        {"max_detections": 1001},
        {"max_detections": 1.5},
    ],
)
def test_invalid_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        validate_tiling_config(**kwargs)


@pytest.mark.parametrize("size", [(1, 1), (32, 24), (640, 640), (200, 640)])
def test_small_image_is_one_unchanged_crop(size):
    assert tile_boxes(*size, validate_tiling_config()) == [[0, 0, *size]]


def test_stride_and_last_edge_are_deterministic():
    assert tile_boxes(1000, 800, validate_tiling_config()) == [
        [0, 0, 640, 640],
        [360, 0, 1000, 640],
        [0, 160, 640, 800],
        [360, 160, 1000, 800],
    ]
    assert tile_boxes(641, 640, validate_tiling_config()) == [
        [0, 0, 640, 640],
        [1, 0, 641, 640],
    ]
    assert tile_boxes(300, 128, validate_tiling_config(128, 0.2)) == [
        [0, 0, 128, 128],
        [102, 0, 230, 128],
        [172, 0, 300, 128],
    ]


@pytest.mark.parametrize("width,height,overlap", [(321, 233, 0), (321, 233, 0.2), (255, 255, 0.5)])
def test_tiles_cover_every_original_pixel_without_padding(width, height, overlap):
    boxes = tile_boxes(width, height, validate_tiling_config(128, overlap))
    covered = set()
    for x1, y1, x2, y2 in boxes:
        assert 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height
        assert x2 - x1 == 128 and y2 - y1 == 128
        covered.update((x, y) for y in range(y1, y2) for x in range(x1, x2))
    assert len(covered) == width * height
    assert len({tuple(box) for box in boxes}) == len(boxes)


def test_maximum_tiles_is_checked_before_building_a_huge_plan():
    config = validate_tiling_config(128, 0)
    assert len(tile_boxes(1024, 1024, config)) == 64
    with pytest.raises(ValueError, match="65 tiles"):
        tile_boxes(128 * 65, 128, config)
    with pytest.raises(ValueError, match="limit is 64"):
        tile_boxes(10**100, 10**100, config)


@pytest.mark.parametrize("width,height", [(0, 2), (2, -1), (True, 3), (3, 2.5), ("3", 2)])
def test_invalid_image_dimensions(width, height):
    with pytest.raises(ValueError, match="dimensions"):
        tile_boxes(width, height, validate_tiling_config())


@pytest.mark.parametrize("config", [None, [], {"resize": True}])
def test_invalid_config_is_not_silently_ignored(config):
    with pytest.raises(ValueError):
        tile_boxes(128, 128, config)


def test_detector_receives_original_crop_pixels_and_progress():
    image = Image.new("RGB", (192, 128))
    image.putpixel((0, 0), (12, 23, 34))
    image.putpixel((64, 0), (45, 56, 67))
    detector = FixtureDetector()
    events = []
    result = tiled_predict(
        detector,
        image,
        validate_tiling_config(128, 0.5),
        progress=lambda done, total: events.append((done, total)),
    )
    assert detector.calls == [
        {"size": (128, 128), "pixel": (12, 23, 34)},
        {"size": (128, 128), "pixel": (45, 56, 67)},
    ]
    assert events == [(1, 2), (2, 2)]
    assert result["input_size"] == [192, 128]
    assert result["detections"] == []
    assert result["timing"]["tile_count"] == result["timing"]["forward_passes"] == 2
    assert result["timing"]["preprocess_ms"] == 2.5
    assert result["timing"]["inference_ms"] == 5
    assert result["timing"]["postprocess_ms"] == 1.5
    assert result["timing"]["crop_ms"] >= 0
    assert result["timing"]["merge_ms"] >= 0
    assert result["timing"]["total_ms"] >= (
        result["timing"]["crop_ms"] + result["timing"]["merge_ms"]
    )


def overlapping_predictions():
    return [
        prediction(
            detections=[
                detection((70, 20, 90, 40), score=0.8, native_label_id=1),
                detection((70, 20, 90, 40), score=0.7, label=3, native_label_id=2),
                detection((0, 0, 10, 10), score=0.2),
            ]
        ),
        prediction(
            detections=[
                detection((6, 20, 26, 40), score=0.9, native_label_id=1, source="second"),
                detection((6, 20, 26, 40), score=0.6, label=3, native_label_id=2),
                detection((40, 50, 50, 60), score=0.4),
            ]
        ),
    ]


def test_original_coordinates_and_class_aware_duplicate_merge_keep_native_labels():
    outputs = overlapping_predictions()
    original = deepcopy(outputs)
    result = tiled_predict(
        FixtureDetector(outputs), Image.new("RGB", (192, 128)), validate_tiling_config(128, 0.5)
    )
    assert [item["box"] for item in result["detections"]] == [
        [70, 20, 90, 40],
        [70, 20, 90, 40],
        [104, 50, 114, 60],
        [0, 0, 10, 10],
    ]
    person, car = result["detections"][:2]
    assert person["tile_index"] == 1 and person["native_label_id"] == 1
    assert person["source"] == "second"
    assert car["tile_index"] == 0 and car["label_id"] == 3 and car["native_label_id"] == 2
    assert outputs == original
    assert result["timing"]["raw_detection_count"] == 6
    assert result["timing"]["merged_detection_count"] == 4
    assert result["timing"]["kept_detection_count"] == 4
    assert result["timing"]["truncated_detection_count"] == 0
    assert result["metadata"]["tiles"] == [
        {**original[0], "tile_index": 0, "box": [0, 0, 128, 128]},
        {**original[1], "tile_index": 1, "box": [64, 0, 192, 128]},
    ]
    assert result["metadata"]["tiles"][1]["detections"][0]["box"] == [6, 20, 26, 40]


def test_progress_callback_time_is_excluded_from_wall_measurement(monkeypatch):
    from iris import tiling

    clock = [100.0]
    monkeypatch.setattr(tiling.time, "perf_counter", lambda: clock[0])

    class TimedDetector(FixtureDetector):
        def predict(self, image):
            clock[0] += 0.05
            return super().predict(image)

    def progress(_done, _total):
        clock[0] += 10

    result = tiled_predict(
        TimedDetector(),
        Image.new("RGB", (192, 128)),
        validate_tiling_config(128, 0.5),
        progress=progress,
    )
    assert result["timing"]["total_ms"] == pytest.approx(100)
    assert clock[0] == pytest.approx(120.1)


def test_global_limit_is_applied_after_class_aware_merging():
    result = tiled_predict(
        FixtureDetector(overlapping_predictions()),
        Image.new("RGB", (192, 128)),
        validate_tiling_config(128, 0.5, max_detections=2),
    )
    assert [item["score"] for item in result["detections"]] == [0.9, 0.7]
    assert result["timing"]["merged_detection_count"] == 4
    assert result["timing"]["kept_detection_count"] == 2
    assert result["timing"]["truncated_detection_count"] == 2


def test_score_ties_preserve_tile_then_detector_order():
    outputs = [
        prediction(
            detections=[
                detection((70, 20, 90, 40), source="first"),
                detection((70, 20, 90, 40), source="same-tile-duplicate"),
                detection((70, 20, 90, 40), label=3),
            ]
        ),
        prediction(detections=[detection((6, 20, 26, 40), source="second-tile")]),
    ]
    result = tiled_predict(
        FixtureDetector(outputs), Image.new("RGB", (192, 128)), validate_tiling_config(128, 0.5)
    )
    assert len(result["detections"]) == 2
    assert result["detections"][0]["source"] == "first"
    assert [item["label_id"] for item in result["detections"]] == [1, 3]


@pytest.mark.parametrize("threshold,count", [(0.5, 2), (0.499, 1)])
def test_nms_suppresses_strictly_above_iou_threshold(threshold, count):
    output = prediction(
        detections=[
            detection((0, 0, 30, 10), score=0.9),
            detection((10, 0, 40, 10)),
        ]
    )
    result = tiled_predict(
        FixtureDetector([output]),
        Image.new("RGB", (128, 128)),
        validate_tiling_config(128, merge_iou=threshold),
    )
    assert len(result["detections"]) == count


@pytest.mark.parametrize(
    "field,value",
    [
        ("box", [0, 0, float("nan"), 10]),
        ("box", [False, 0, 12, 10]),
        ("box", [0, 0, 0, 10]),
        ("box", [-1, 0, 12, 10]),
        ("box", [0, 0, 129, 10]),
        ("box", [0, 0, 12, 129]),
        ("box", [0, 0, 12]),
        ("box", "0,0,12,10"),
        ("score", True),
        ("score", "0.8"),
        ("score", float("inf")),
        ("score", -0.1),
        ("score", 1.1),
        ("label_id", True),
        ("label_id", 0),
        ("label_id", 1.5),
        ("label_id", "1"),
        ("native_label_id", True),
        ("native_label_id", -1),
    ],
)
def test_invalid_tile_detection_is_rejected_before_mapping(field, value):
    output = prediction()
    output["detections"][0][field] = value
    with pytest.raises(ValueError):
        tiled_predict(
            FixtureDetector([output]), Image.new("RGB", (128, 128)), validate_tiling_config(128)
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("input_size", [128, 127]),
        ("input_size", [128.0, 128]),
        ("input_size", None),
        ("detections", {}),
        ("detections", [None]),
        ("detections", [detection()] * 1001),
        ("timing", {}),
        ("timing", {**TIMING, "total_ms": float("nan")}),
        ("timing", {**TIMING, "inference_ms": -1}),
        ("timing", {**TIMING, "preprocess_ms": True}),
    ],
)
def test_invalid_prediction_metadata_fails_the_whole_image(field, value):
    output = prediction()
    output[field] = value
    with pytest.raises(ValueError):
        tiled_predict(
            FixtureDetector([output]), Image.new("RGB", (128, 128)), validate_tiling_config(128)
        )


def test_per_tile_detection_limit_is_explicit():
    assert MAX_DETECTIONS_PER_TILE == 1000


def test_small_image_keeps_its_original_dimensions():
    output = prediction((32, 24), [detection((0, 0, 32, 24))])
    result = tiled_predict(
        FixtureDetector([output]), Image.new("RGB", (32, 24)), validate_tiling_config()
    )
    assert result["detections"][0]["box"] == [0, 0, 32, 24]
    assert result["timing"]["tile_count"] == 1


def test_cancellation_before_first_forward_does_not_call_detector():
    detector = FixtureDetector()
    with pytest.raises(TiledInferenceCancelled):
        tiled_predict(
            detector,
            Image.new("RGB", (192, 128)),
            validate_tiling_config(128, 0.5),
            cancelled=lambda: True,
        )
    assert detector.calls == []


def test_cancellation_after_a_forward_discards_the_incomplete_image():
    detector = FixtureDetector()
    progress = []
    with pytest.raises(TiledInferenceCancelled):
        tiled_predict(
            detector,
            Image.new("RGB", (192, 128)),
            validate_tiling_config(128, 0.5),
            cancelled=lambda: bool(detector.calls),
            progress=lambda done, total: progress.append((done, total)),
        )
    assert len(detector.calls) == 1
    assert progress == []


@pytest.mark.parametrize("width", [128, 192])
def test_cancellation_from_progress_is_observed_before_next_tile_or_merge(width):
    stop = []
    detector = FixtureDetector()
    with pytest.raises(TiledInferenceCancelled):
        tiled_predict(
            detector,
            Image.new("RGB", (width, 128)),
            validate_tiling_config(128, 0.5),
            cancelled=lambda: bool(stop),
            progress=lambda _done, _total: stop.append(True),
        )
    assert len(detector.calls) == 1


def test_cancellation_during_merge_is_observed():
    checks = []

    def cancelled():
        checks.append(True)
        return len(checks) == 5

    output = prediction(detections=[detection(), detection((20, 20, 30, 30))])
    with pytest.raises(TiledInferenceCancelled):
        tiled_predict(
            FixtureDetector([output]),
            Image.new("RGB", (128, 128)),
            validate_tiling_config(128),
            cancelled=cancelled,
        )


def test_detector_error_is_not_returned_as_an_empty_prediction():
    class BrokenDetector:
        def predict(self, image):
            raise RuntimeError("Synthetic detector failure")

    with pytest.raises(RuntimeError, match="Synthetic detector failure"):
        tiled_predict(BrokenDetector(), Image.new("RGB", (128, 128)), validate_tiling_config(128))
