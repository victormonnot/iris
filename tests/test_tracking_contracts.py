"""Tracker contracts preserve native measurements and honest prediction state."""

import json
import subprocess
import sys
from copy import deepcopy

import pytest

from iris import tracking_contracts as contracts


def source_frame():
    return {
        "frame_id": "frame-18",
        "frame_index": 18,
        "timestamp_seconds": 1.8,
        "input_size": [640, 480],
        "detections": [
            {
                "detection_index": 1,
                "label_id": 1,
                "label": "person",
                "score": 0.9,
                "box": [10, 20, 30, 70],
            },
            {
                "detection_index": 7,
                "label_id": 3,
                "label": "car",
                "score": 0.05,
                "box": [100, 100, 200, 200],
            },
        ],
    }


def tracking_frame(source=None):
    source = source or source_frame()
    return {
        "schema": contracts.FRAME_SCHEMA,
        "sequence_id": "sequence-a",
        **{key: deepcopy(value) for key, value in source.items() if key != "detections"},
        "update_index": 5,
        "observations": [
            {
                **deepcopy(source["detections"][0]),
                "track_id": 1,
                "confirmed": True,
                "estimated_box": [11, 20, 31, 70],
            }
        ],
        "predictions": [
            {
                "track_id": 4,
                "label_id": 1,
                "label": "person",
                "box": [-4, 10, 15, 50],
                "confirmed": True,
                "last_observed_frame_id": "frame-10",
                "last_observed_frame_index": 10,
                "last_observed_timestamp_seconds": 1.0,
                "last_observed_update_index": 3,
                "age_updates": 2,
                "age_seconds": 0.8,
            }
        ],
        "unassigned": [{**deepcopy(source["detections"][1]), "reason": "below_low_threshold"}],
        "gmc": {"method": "none", "status": "disabled", "matrix": None, "downscale": 2},
        "timing": {"gmc_ms": 0, "association_ms": 1.1, "total_ms": 1.3},
    }


def set_path(value, path, replacement):
    keys = path.split(".")
    for key in keys[:-1]:
        value = value[int(key)] if isinstance(value, list) else value[key]
    if isinstance(value, list):
        value[int(keys[-1])] = replacement
    else:
        value[keys[-1]] = replacement


@pytest.fixture
def profile():
    return contracts.make_profile("bytetrack", class_ids=[1, 3])


def test_contracts_import_without_loading_tracking_image_or_ml_runtimes():
    script = (
        "import sys; import iris.tracking_contracts; "
        "assert not {'cv2', 'torch', 'numpy', 'scipy', 'lap', 'ultralytics'} & set(sys.modules)"
    )
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)


@pytest.mark.parametrize("algorithm,gmc", [("bytetrack", "none"), ("botsort", "sparseOptFlow")])
def test_default_profiles_are_explicit_finite_json_and_detached(algorithm, gmc):
    classes = [1, 3]
    profile = contracts.make_profile(algorithm, class_ids=classes)
    assert set(profile) == contracts.PROFILE_FIELDS
    assert profile["gmc_method"] == gmc
    assert profile["with_reid"] is False
    assert profile["time_policy"] == "one_update_per_available_frame"
    assert profile["gmc_downscale"] == 2
    assert profile["new_track_threshold"] == 0.6
    classes.append(8)
    assert profile["class_ids"] == [1, 3]
    assert json.loads(json.dumps(profile, allow_nan=False)) == profile


def test_bytetrack_birth_uses_native_sum_and_normalizes_equivalent_decimal():
    computed = contracts.make_profile("bytetrack", class_ids=[1], high_threshold=0.7)
    explicit = contracts.make_profile(
        "bytetrack", class_ids=[1], high_threshold=0.7, new_track_threshold=0.8
    )
    assert computed["new_track_threshold"] == 0.7 + 0.1
    assert contracts.profile_hash(computed) == contracts.profile_hash(explicit)


def test_profile_hash_ignores_key_order_but_binds_execution_choices(profile):
    assert contracts.profile_hash(profile) == contracts.profile_hash(
        dict(reversed(profile.items()))
    )
    changed = {**profile, "buffer_updates": 31}
    assert contracts.profile_hash(changed) != contracts.profile_hash(profile)
    assert len(contracts.profile_hash(profile)) == 64


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "future"),
        ("algorithm", "sort"),
        ("algorithm", ["bytetrack"]),
        ("class_ids", []),
        ("class_ids", [3, 1]),
        ("class_ids", [1, 1]),
        ("class_ids", [0]),
        ("class_ids", [True]),
        ("class_ids", [1.0]),
        ("class_ids", list(range(1, 102))),
        ("class_ids", (1, 3)),
        ("high_threshold", True),
        ("high_threshold", 0.1),
        ("high_threshold", 0.91),
        ("high_threshold", float("nan")),
        ("high_threshold", float("inf")),
        ("high_threshold", "0.5"),
        ("low_threshold", 0.05),
        ("new_track_threshold", 0.65),
        ("match_threshold", -0.01),
        ("match_threshold", 1.01),
        ("match_threshold", False),
        ("buffer_updates", -1),
        ("buffer_updates", 10001),
        ("buffer_updates", 1.0),
        ("buffer_updates", True),
        ("fuse_score", 1),
        ("gmc_method", "sparseOptFlow"),
        ("gmc_downscale", 1),
        ("gmc_downscale", 2.0),
        ("seed", -1),
        ("seed", 2**31),
        ("seed", True),
        ("opencv_threads", 0),
        ("opencv_threads", 33),
        ("opencv_threads", 1.0),
        ("with_reid", True),
        ("with_reid", 0),
        ("time_policy", "source_frame_delta"),
        ("unknown", 1),
    ],
)
def test_profile_rejects_unsupported_or_ambiguous_execution(profile, field, value):
    profile[field] = value
    with pytest.raises(ValueError):
        contracts.validate_profile(profile)
    with pytest.raises(ValueError):
        contracts.profile_hash(profile)


@pytest.mark.parametrize("value", [None, [], {}, "bytetrack"])
def test_profile_requires_complete_object(value):
    with pytest.raises(ValueError):
        contracts.validate_profile(value)


def test_factory_rejects_unknown_options_and_invalid_defaults():
    with pytest.raises(ValueError, match="Unknown"):
        contracts.make_profile("bytetrack", class_ids=[1], frame_rate=30)
    with pytest.raises(ValueError):
        contracts.make_profile("bytetrack", class_ids=[1], high_threshold="0.7")
    with pytest.raises(ValueError):
        contracts.make_profile("unsupported", class_ids=[1])


@pytest.mark.parametrize(
    "low,high,birth",
    [(0, 0.1, 0.1), (0.1, 0.7, 0.9), (0.5, 1, 1)],
)
def test_botsort_thresholds_are_independent_in_native_order(low, high, birth):
    profile = contracts.make_profile(
        "botsort",
        class_ids=[1],
        low_threshold=low,
        high_threshold=high,
        new_track_threshold=birth,
    )
    assert profile["low_threshold"] == low
    assert profile["high_threshold"] == high
    assert profile["new_track_threshold"] == birth


@pytest.mark.parametrize(
    "changes",
    [
        {"low_threshold": 0.5},
        {"low_threshold": 0.6},
        {"new_track_threshold": 0.49},
        {"high_threshold": 0},
        {"gmc_method": "orb"},
        {"gmc_method": ["none"]},
    ],
)
def test_botsort_rejects_threshold_inversion_or_unimplemented_gmc(changes):
    with pytest.raises(ValueError):
        contracts.make_profile("botsort", class_ids=[1], **changes)


def test_profile_bounds_accept_native_endpoints():
    profile = contracts.make_profile(
        "bytetrack",
        class_ids=[1],
        high_threshold=0.9,
        match_threshold=1,
        buffer_updates=10000,
        seed=2**31 - 1,
        opencv_threads=32,
        fuse_score=False,
    )
    assert profile["new_track_threshold"] == 1
    zero_buffer = contracts.make_profile("botsort", class_ids=[1], buffer_updates=0)
    assert zero_buffer["buffer_updates"] == 0


def test_input_accepts_cached_metadata_but_keeps_native_facts_only(profile):
    source = source_frame()
    cached = {
        **source,
        **{key: "opaque metadata verified by cache service" for key in contracts.SOURCE_METADATA},
    }
    checked = contracts.validate_update_input(cached, profile)
    assert checked == source
    checked["detections"][0]["box"][0] = 15
    assert source["detections"][0]["box"][0] == 10


@pytest.mark.parametrize(
    "path,value",
    [
        ("frame_id", ""),
        ("frame_id", "  "),
        ("frame_id", "\ud800"),
        ("frame_index", -1),
        ("frame_index", True),
        ("frame_index", 1.0),
        ("timestamp_seconds", False),
        ("timestamp_seconds", -1),
        ("timestamp_seconds", float("nan")),
        ("timestamp_seconds", 10**1000),
        ("input_size", [640]),
        ("input_size", [640, 480, 3]),
        ("input_size", [True, 480]),
        ("input_size", [640, 0]),
        ("input_size", [640.0, 480]),
        ("detections", None),
        ("detections.0.detection_index", True),
        ("detections.0.detection_index", -1),
        ("detections.0.label_id", 2),
        ("detections.0.label_id", True),
        ("detections.0.label", ""),
        ("detections.0.score", True),
        ("detections.0.score", -0.1),
        ("detections.0.score", 1.01),
        ("detections.0.score", float("inf")),
        ("detections.0.box", [10, 20, 10, 70]),
        ("detections.0.box", [10, 20, 30, 20]),
        ("detections.0.box", [-1, 20, 30, 70]),
        ("detections.0.box", [10, 20, 641, 70]),
        ("detections.0.box", [10, 20, 30, 481]),
        ("detections.0.box", [True, 20, 30, 70]),
        ("detections.0.box", [10, 20, 30, float("nan")]),
        ("detections.0.box", [10, 20, 30]),
        ("extra", 1),
        ("detections.0.extra", 1),
    ],
)
def test_input_rejects_invalid_native_measurements(profile, path, value):
    source = source_frame()
    set_path(source, path, value)
    with pytest.raises(ValueError):
        contracts.validate_update_input(source, profile)


def test_empty_unknown_clock_input_is_a_real_update(profile):
    source = {**source_frame(), "detections": [], "timestamp_seconds": None}
    assert contracts.validate_update_input(source, profile) == source


@pytest.mark.parametrize("mode", ["duplicate", "unsorted", "labels", "too_many"])
def test_input_rejects_duplicate_indices_misordered_or_inconsistent_class_labels(profile, mode):
    source = source_frame()
    if mode == "duplicate":
        source["detections"][1]["detection_index"] = 1
    elif mode == "unsorted":
        source["detections"].reverse()
    elif mode == "labels":
        source["detections"][1]["label_id"] = 1
    else:
        detection = source["detections"][0]
        source["detections"] = [{**detection, "detection_index": i} for i in range(301)]
    with pytest.raises(ValueError):
        contracts.validate_update_input(source, profile)


def test_output_preserves_observations_separately_from_predictions_and_detaches(profile):
    source, result = source_frame(), tracking_frame()
    checked = contracts.validate_tracking_frame(result, source, profile)
    assert checked == result
    assert checked["observations"][0]["box"] != checked["observations"][0]["estimated_box"]
    assert "score" not in checked["predictions"][0]
    assert "detection_index" not in checked["predictions"][0]
    assert checked["predictions"][0]["box"][0] == -4
    checked["observations"][0]["box"][0] = 12
    assert result["observations"][0]["box"][0] == 10
    assert source["detections"][0]["box"][0] == 10


@pytest.mark.parametrize(
    "path,value",
    [
        ("schema", "future"),
        ("sequence_id", ""),
        ("frame_id", "another-frame"),
        ("frame_index", True),
        ("frame_index", 19),
        ("timestamp_seconds", None),
        ("timestamp_seconds", float("nan")),
        ("input_size", [640, 481]),
        ("input_size", [640.0, 480]),
        ("update_index", True),
        ("update_index", 0),
        ("observations", {}),
        ("predictions", None),
        ("unassigned", {}),
        ("observations.0.score", 0.8),
        ("observations.0.score", True),
        ("observations.0.box", [11, 20, 30, 70]),
        ("observations.0.detection_index", 0),
        ("observations.0.track_id", 0),
        ("observations.0.track_id", True),
        ("observations.0.confirmed", 1),
        ("observations.0.estimated_box", [0, 0, 0, 10]),
        ("observations.0.estimated_box", [0, 0, float("inf"), 10]),
        ("unassigned.0.score", 0.1),
        ("unassigned.0.reason", "missing"),
        ("unassigned.0.reason", ["below_low_threshold"]),
        ("predictions.0.track_id", 0),
        ("predictions.0.label_id", 2),
        ("predictions.0.label", "car"),
        ("predictions.0.confirmed", 1),
        ("predictions.0.box", [0, 0, 0, 10]),
        ("predictions.0.box", [0, 0, float("nan"), 10]),
        ("predictions.0.score", 0.8),
        ("predictions.0.detection_index", 2),
        ("predictions.0.last_observed_frame_id", "frame-18"),
        ("predictions.0.last_observed_frame_index", 18),
        ("predictions.0.last_observed_frame_index", True),
        ("predictions.0.last_observed_update_index", 5),
        ("predictions.0.last_observed_update_index", 0),
        ("predictions.0.last_observed_timestamp_seconds", 2),
        ("predictions.0.age_updates", 0),
        ("predictions.0.age_updates", True),
        ("predictions.0.age_updates", 8),
        ("predictions.0.age_seconds", 1),
        ("predictions.0.age_seconds", None),
        ("predictions.0.age_seconds", True),
        ("gmc.method", "sparseOptFlow"),
        ("gmc.status", "estimated"),
        ("gmc.downscale", 2.0),
        ("gmc.matrix", [[1, 0, 0], [0, 1, 0]]),
        ("timing.gmc_ms", -1),
        ("timing.association_ms", True),
        ("timing.total_ms", float("inf")),
        ("timing.extra", 1),
        ("extra", 1),
    ],
)
def test_output_rejects_fabricated_measurements_predictions_or_provenance(profile, path, value):
    result = tracking_frame()
    set_path(result, path, value)
    with pytest.raises(ValueError):
        contracts.validate_tracking_frame(result, source_frame(), profile)


@pytest.mark.parametrize(
    "case", ["missing", "duplicate", "cross_bucket", "observation_id", "prediction_id"]
)
def test_every_detection_and_track_has_exactly_one_disposition(profile, case):
    result = tracking_frame()
    if case == "missing":
        result["unassigned"] = []
    elif case == "duplicate":
        result["unassigned"].append(deepcopy(result["unassigned"][0]))
    elif case == "cross_bucket":
        result["unassigned"].append(
            {**source_frame()["detections"][0], "reason": "native_suppressed"}
        )
    elif case == "observation_id":
        result["predictions"][0]["track_id"] = result["observations"][0]["track_id"]
    else:
        result["predictions"].append(deepcopy(result["predictions"][0]))
    with pytest.raises(ValueError):
        contracts.validate_tracking_frame(result, source_frame(), profile)


@pytest.mark.parametrize("reason", sorted(contracts.UNASSIGNED_REASONS))
def test_unassigned_reasons_remain_explicit_without_editing_detector_output(profile, reason):
    result = tracking_frame()
    result["unassigned"][0]["reason"] = reason
    checked = contracts.validate_tracking_frame(result, source_frame(), profile)
    assert checked["unassigned"][0]["reason"] == reason


@pytest.mark.parametrize("clock", ["unknown_current", "unknown_previous", "all_unknown"])
def test_unknown_prediction_age_is_never_invented_from_update_count(profile, clock):
    source = source_frame()
    result = tracking_frame()
    if clock in {"unknown_current", "all_unknown"}:
        source["timestamp_seconds"] = result["timestamp_seconds"] = None
    if clock in {"unknown_previous", "all_unknown"}:
        result["predictions"][0]["last_observed_timestamp_seconds"] = None
    result["predictions"][0]["age_seconds"] = None
    assert contracts.validate_tracking_frame(result, source, profile) == result
    result["predictions"][0]["age_seconds"] = 0.2
    with pytest.raises(ValueError, match="unknown"):
        contracts.validate_tracking_frame(result, source, profile)


def test_empty_detection_frame_can_have_only_predictions(profile):
    source = {**source_frame(), "detections": []}
    result = tracking_frame()
    result["observations"] = []
    result["unassigned"] = []
    assert contracts.validate_tracking_frame(result, source, profile) == result


@pytest.mark.parametrize("status", ["initialized", "estimated", "identity_insufficient_matches"])
def test_botsort_gmc_result_records_real_affine_and_status(status):
    profile = contracts.make_profile("botsort", class_ids=[1, 3])
    result = tracking_frame()
    result["gmc"] = {
        "method": "sparseOptFlow",
        "status": status,
        "matrix": [[1, 0, -5], [0, 1, 2]] if status == "estimated" else [[1, 0, 0], [0, 1, 0]],
        "downscale": 2,
    }
    assert contracts.validate_tracking_frame(result, source_frame(), profile) == result


@pytest.mark.parametrize(
    "changes",
    [
        {"matrix": None},
        {"matrix": [[1, 0, 0]]},
        {"matrix": [[1, 0], [0, 1]]},
        {"matrix": [[True, 0, 0], [0, 1, 0]]},
        {"matrix": [[1, 0, float("nan")], [0, 1, 0]]},
        {"status": "disabled"},
        {"status": []},
        {"status": "identity_insufficient_matches", "matrix": [[1, 0, 4], [0, 1, 0]]},
        {"status": "initialized", "matrix": [[1, 0, 4], [0, 1, 0]]},
    ],
)
def test_botsort_rejects_missing_nonfinite_or_mislabelled_camera_motion(changes):
    profile = contracts.make_profile("botsort", class_ids=[1, 3])
    result = tracking_frame()
    result["gmc"] = {
        "method": "sparseOptFlow",
        "status": "estimated",
        "matrix": [[1, 0, 0], [0, 1, 0]],
        "downscale": 2,
        **changes,
    }
    with pytest.raises(ValueError):
        contracts.validate_tracking_frame(result, source_frame(), profile)


def test_semantic_comparison_excludes_only_wall_clock_durations(profile):
    first = contracts.validate_tracking_frame(tracking_frame(), source_frame(), profile)
    second = deepcopy(first)
    second["timing"] = {"gmc_ms": 0, "association_ms": 2.7, "total_ms": 5.1}
    assert first != second
    assert contracts.semantic_frame(first) == contracts.semantic_frame(second)
    second["predictions"][0]["box"][0] -= 0.1
    assert contracts.semantic_frame(first) != contracts.semantic_frame(second)
    semantic = contracts.semantic_frame(first)
    semantic["observations"].clear()
    assert first["observations"]
    assert set(first) - set(semantic) == {"timing"}
