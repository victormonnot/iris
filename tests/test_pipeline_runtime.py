"""Controlled portable execution: sequencing, state isolation and native parity."""

from copy import deepcopy

import pytest
from PIL import Image

from iris import pipeline_bundle_contracts as bundles
from iris import pipeline_runtime as runtime
from iris.pipeline_selection import SelectionState
from iris.tracking_contracts import FRAME_SCHEMA, make_profile, semantic_frame
from iris.tracking_selection_contracts import DEFAULT_POLICY, digest

BOX = [10, 10, 20, 30]


def detection(index=2, label_id=1):
    return {
        "detection_index": index,
        "label_id": label_id,
        "label": "person" if label_id == 1 else "car",
        "score": 0.9,
        "box": list(BOX),
    }


def observation(index=2, track_id=12, **overrides):
    return {
        **detection(index),
        "track_id": track_id,
        "confirmed": True,
        "estimated_box": [11, 10, 21, 30],
        **overrides,
    }


def selection_frame(index, observations=None, *, source_index=None, timestamp=True):
    return {
        "frame_id": f"f{index}",
        "frame_index": index if source_index is None else source_index,
        "timestamp_seconds": index / 10 if timestamp else None,
        "input_size": [100, 100],
        "update_index": index + 1,
        "observations": [observation()] if observations is None else observations,
        "predictions": [],
        "unassigned": [],
    }


class FakeDetector:
    def __init__(self, directory, manifest, device=None):
        self.calls = 0
        self.rows = [detection(0, 3), detection(2, 1)]
        self.fail = False
        self.changed = False
        self.metadata = {"kind": "controlled", "device": "cpu"}

    def verify_runtime(self):
        if self.changed:
            raise RuntimeError("Detector runtime changed")

    def predict(self, image):
        self.calls += 1
        if self.fail:
            raise RuntimeError("Detector failure")
        return {
            "input_size": list(image.size),
            "detections": deepcopy(self.rows),
            "native_detection_count": 3,
            "timing": {"preprocess_ms": 0, "inference_ms": 0, "postprocess_ms": 0, "total_ms": 0},
        }


class FakeTracker:
    def __init__(self, profile):
        self.profile = deepcopy(profile)
        self.metadata = {"kind": "controlled", "algorithm": profile["algorithm"]}
        self.calls = 0
        self.images = []
        self.inputs = []
        self.fail = False
        self.changed = False
        self.confirmed = True

    def verify_runtime(self):
        if self.changed:
            raise RuntimeError("Tracker runtime changed")

    def reset(self, sequence_id):
        self.sequence_id = sequence_id
        self.calls = 0
        self.images = []
        self.inputs = []

    def update(self, source, image=None):
        self.calls += 1
        if self.fail:
            raise RuntimeError("Native failure after mutation")
        self.images.append(image)
        self.inputs.append(deepcopy(source))
        gmc = {
            "method": self.profile["gmc_method"],
            "downscale": 2,
            "status": "disabled",
            "matrix": None,
        }
        if gmc["method"] != "none":
            gmc.update(status="initialized", matrix=[[1, 0, 0], [0, 1, 0]])
        return {
            "schema": FRAME_SCHEMA,
            "sequence_id": self.sequence_id,
            **{key: value for key, value in source.items() if key != "detections"},
            "update_index": self.calls,
            "observations": [
                {
                    **row,
                    "track_id": row["detection_index"] + 10,
                    "confirmed": self.confirmed,
                    "estimated_box": [11, 10, 21, 30],
                }
                for row in source["detections"]
            ],
            "predictions": [],
            "unassigned": [],
            "gmc": gmc,
            "timing": {"gmc_ms": 0, "association_ms": 0, "total_ms": 0},
        }


def manifest(*, selection=True, algorithm="bytetrack"):
    policy = deepcopy(DEFAULT_POLICY)
    return {
        "format": "iris-pipeline-bundle-v2",
        "tracker": {"profile": make_profile(algorithm, class_ids=[1])},
        "detector": {
            "config": {"min_score": 0.1},
            "output_mapping": {
                "entries": [{"output_id": 1, "label": "person"}, {"output_id": 3, "label": "car"}]
            },
        },
        "selection": {
            "algorithm": "guarded_geometry",
            "policy": policy,
            "policy_sha256": digest(policy),
        }
        if selection
        else None,
        "files": {"iris_bundle/pipeline_selection.py": {"sha256": "b" * 64, "size": 10}},
    }


@pytest.fixture
def build(monkeypatch, tmp_path):
    def factory(*, selection=True, algorithm="bytetrack", tracker_factory=FakeTracker):
        value = manifest(selection=selection, algorithm=algorithm)

        def checked(directory, expected_manifest=None):
            if expected_manifest is not None and expected_manifest != value:
                raise ValueError("Manifest changed")
            return {
                "manifest": deepcopy(value),
                "manifest_sha256": digest(value),
                "files_verified": 1,
            }

        monkeypatch.setattr(bundles, "validate_directory", checked, raising=False)
        return runtime.Pipeline(
            tmp_path, detector_factory=FakeDetector, tracker_factory=tracker_factory
        )

    return factory


def update(pipe, index=0, **overrides):
    return pipe.update(
        overrides.pop("image", Image.new("RGB", (100, 100), (10, 20, 30))),
        **{
            "frame_id": f"f{index}",
            "frame_index": index,
            "timestamp_seconds": index / 10,
            **overrides,
        },
    )


def test_full_detector_classes_preserved_and_tracker_indices_never_renumbered(build):
    pipe = build()
    pipe.reset("one")
    output = update(pipe, select_detection_index=2)
    assert [row["label_id"] for row in output["detector"]["detections"]] == [3, 1]
    assert [row["detection_index"] for row in output["tracking"]["observations"]] == [2]
    assert output["selection"]["selected"]["box"] == BOX
    assert output["selection"]["selected"]["estimated_box"] != BOX
    assert output["selection"]["logical_object_id"] == "selection-1"
    output["selection"]["last_observed"]["observation"]["box"][0] = 99
    output["detector"]["detections"][0]["box"][0] = 99
    output2 = update(pipe, 1)
    assert output2["selection"]["selected"]["box"] == BOX
    assert output2["selection"]["state"] == "observed"
    assert pipe.metadata["independent_quality"] == "not_qualified"


def test_selection_is_explicit_release_terminal_and_new_selection_has_new_local_number(build):
    pipe = build()
    pipe.reset("one")
    assert update(pipe)["selection"]["state"] == "idle"
    assert (
        update(pipe, 1, select_detection_index=2)["selection"]["logical_object_id"] == "selection-1"
    )
    assert update(pipe, 2, release=True)["selection"]["state"] == "released"
    assert update(pipe, 3)["selection"]["selected"] is None
    selected = update(pipe, 4, select_detection_index=2)
    assert selected["selection"]["logical_object_id"] == "selection-2"
    pipe.reset("two")
    restarted = update(pipe, select_detection_index=2)
    assert restarted["selection"]["logical_object_id"] == "selection-1"
    assert restarted["update_index"] == restarted["tracking"]["update_index"] == 1


@pytest.mark.parametrize(
    "bad",
    [
        {"frame_index": True},
        {"frame_index": -1},
        {"frame_id": ""},
        {"frame_id": "bad\x00"},
        {"timestamp_seconds": float("nan")},
        {"timestamp_seconds": None},
        {"timestamp_seconds": -0.1},
        {"timestamp_seconds": True},
        {"release": 1},
        {"select_detection_index": True},
        {"select_detection_index": 2, "release": True},
        {"image": Image.new("L", (100, 100))},
    ],
)
def test_bad_input_poison_requires_reset_even_before_detector_mutation(build, bad):
    pipe = build()
    pipe.reset("one")
    with pytest.raises(ValueError):
        update(pipe, **bad)
    assert pipe._detector.calls == 0
    with pytest.raises(runtime.PipelineError, match="reset"):
        update(pipe)
    pipe.reset("restarted")
    assert update(pipe)["update_index"] == 1


@pytest.mark.parametrize(
    "bad",
    [
        {"frame_id": "f0"},
        {"frame_index": 0},
        {"timestamp_seconds": 0},
        {"image": Image.new("RGB", (101, 100))},
    ],
)
def test_order_and_dimensions_checked_before_detector(build, bad):
    pipe = build()
    pipe.reset("one")
    update(pipe)
    with pytest.raises(ValueError):
        update(pipe, 1, **bad)
    assert pipe._detector.calls == 1
    with pytest.raises(runtime.PipelineError):
        update(pipe, 2)


def test_unknown_clock_and_nominal_clock_remain_explicit_and_resettable(build):
    pipe = build()
    pipe.reset("unknown", clock_kind="unknown")
    selected = update(pipe, timestamp_seconds=None, select_detection_index=2)
    assert selected["selection"]["age"]["seconds"] is None
    assert pipe.metadata["clock_kind"] == "unknown"
    with pytest.raises(ValueError, match="null"):
        update(pipe, 1)
    pipe.reset("nominal", clock_kind="nominal_fps")
    assert update(pipe)["timestamp_seconds"] == 0
    assert pipe.metadata["clock_kind"] == "nominal_fps"


@pytest.mark.parametrize("component", ["_detector", "_tracker"])
def test_component_failure_after_mutation_never_reuses_partial_state(build, component):
    pipe = build()
    pipe.reset("one")
    getattr(pipe, component).fail = True
    with pytest.raises(RuntimeError, match="failure|Failure"):
        update(pipe)
    getattr(pipe, component).fail = False
    with pytest.raises(runtime.PipelineError, match="reset"):
        update(pipe, 1)
    pipe.reset("two")
    assert update(pipe)["tracking"]["update_index"] == 1


@pytest.mark.parametrize("selected", [0, 99])
def test_selection_cannot_use_untracked_class_or_missing_observation(build, selected):
    pipe = build()
    pipe.reset("one")
    with pytest.raises(ValueError, match="confirmed measured"):
        update(pipe, select_detection_index=selected)
    with pytest.raises(runtime.PipelineError):
        update(pipe, 1)


def test_selection_cannot_use_unconfirmed_observation_or_replace_active_selection(build):
    pipe = build()
    pipe.reset("one")
    pipe._tracker.confirmed = False
    with pytest.raises(ValueError, match="confirmed measured"):
        update(pipe, select_detection_index=2)
    pipe._tracker.confirmed = True
    pipe.reset("two")
    update(pipe, select_detection_index=2)
    with pytest.raises(ValueError, match="Release"):
        update(pipe, 1, select_detection_index=2)


def test_missing_selection_settings_rejects_events_and_emits_null(build):
    pipe = build(selection=False)
    pipe.reset("one")
    assert update(pipe)["selection"] is None
    with pytest.raises(ValueError, match="settings"):
        update(pipe, 1, select_detection_index=2)


def test_limit_stops_before_inference_and_explicit_reset_restores_capacity(build, monkeypatch):
    monkeypatch.setattr(runtime, "MAX_UPDATES", 2)
    pipe = build()
    pipe.reset("one")
    update(pipe)
    update(pipe, 1)
    with pytest.raises(runtime.PipelineError, match="limit"):
        update(pipe, 2)
    assert pipe._detector.calls == pipe._tracker.calls == 2
    pipe.reset("two")
    assert update(pipe)["update_index"] == 1


def test_gmc_receives_original_contiguous_bgr_image(build):
    pipe = build(algorithm="botsort")
    pipe.reset("one")
    output = update(pipe)
    pixels = pipe._tracker.images[0]
    assert pixels.shape == (100, 100, 3)
    assert pixels[0, 0].tolist() == [30, 20, 10]
    assert pixels.dtype.name == "uint8" and pixels.flags.c_contiguous
    assert output["tracking"]["gmc"]["method"] == "sparseOptFlow"


def test_exif_coordinates_are_rejected_until_oriented_before_pipeline(build):
    from PIL import ImageOps

    pipe = build()
    pipe.reset("one")
    image = Image.new("RGB", (100, 100))
    image.getexif()[274] = 6
    with pytest.raises(ValueError, match="orientation"):
        update(pipe, image=image)
    pipe.reset("two")
    assert update(pipe, image=ImageOps.exif_transpose(image))["input_size"] == [100, 100]


def test_full_detector_output_is_validated_even_for_unused_class(build):
    pipe = build()
    pipe.reset("one")
    pipe._detector.rows[0]["box"] = [5, 5, 4, 4]
    with pytest.raises(ValueError):
        update(pipe)
    assert pipe._tracker.calls == 0


def test_runtime_mutation_poison_cannot_be_cleared_without_fixing_runtime(build):
    pipe = build()
    pipe.reset("one")
    pipe._detector.changed = True
    with pytest.raises(RuntimeError, match="changed"):
        pipe.verify_runtime()
    with pytest.raises(RuntimeError, match="changed"):
        pipe.reset("two")
    pipe._detector.changed = False
    pipe.reset("two")
    assert update(pipe)["update_index"] == 1
    pipe.close()
    with pytest.raises(runtime.PipelineError, match="closed"):
        pipe.reset("three")


def test_streaming_selection_gap_resets_confirmation_and_pending_does_not_refresh_memory():
    machine = SelectionState({**DEFAULT_POLICY, "max_lost_updates": 4, "max_lost_seconds": None})
    assert machine.update(selection_frame(0), select_detection_index=2)["state"] == "observed"
    candidate = [observation(track_id=33)]
    assert machine.update(selection_frame(1, candidate))["state"] == "recovering"
    gap = machine.update(selection_frame(2, candidate, source_index=5))
    assert gap["source_gap"] and gap["pending"]["observations"] == 1
    assert gap["last_observed"]["update_index"] == 1
    assert machine.update(selection_frame(3, candidate, source_index=6))["state"] == "recovered"
    lost = machine.update(selection_frame(4, [], source_index=7))
    assert lost["last_observed"]["observation"]["track_id"] == 33


def test_expired_selection_never_revives_without_explicit_event():
    machine = SelectionState({**DEFAULT_POLICY, "max_lost_updates": 1, "max_lost_seconds": None})
    machine.update(selection_frame(0), select_detection_index=2)
    assert machine.update(selection_frame(1, []))["state"] == "lost"
    assert machine.update(selection_frame(2))["state"] == "expired"
    assert machine.update(selection_frame(3))["selected"] is None
    fresh = machine.update(selection_frame(4), select_detection_index=2)
    assert fresh["state"] == "observed" and fresh["logical_object_id"] == "selection-2"


def test_streaming_selection_retains_no_frame_history_or_previous_detection_list():
    machine = SelectionState(DEFAULT_POLICY)
    machine.update(selection_frame(0), select_detection_index=2)
    for index in range(1, 1000):
        machine.update(selection_frame(index))
    assert machine._previous == {"frame_index": 999}
    assert machine._last["update_index"] == 1000
    assert machine._pending is None
    assert not any(isinstance(value, list) for value in vars(machine).values())


@pytest.mark.parametrize("algorithm", ["bytetrack", "botsort"])
def test_pipeline_native_adapter_semantics_match_direct_fresh_adapter(build, algorithm):
    from iris.tracking import make_tracker, tracking_status

    if not tracking_status()["available"]:
        pytest.skip("Native tracking dependencies are optional")
    pipe = build(algorithm=algorithm, tracker_factory=make_tracker)
    direct = make_tracker(pipe.manifest["tracker"]["profile"])
    pipe.reset("native")
    direct.reset("native")
    import numpy as np

    image = Image.fromarray(
        np.random.default_rng(12).integers(0, 256, (100, 100, 3), dtype=np.uint8)
    )
    for index in range(6):
        # A source gap is one real update; absence remains a prediction, not a detection.
        source_index = index if index < 3 else index + 4
        pipe._detector.rows = [detection(2)] if index != 2 else []
        actual = update(pipe, source_index, image=image)
        expected = direct.update(
            {
                "frame_id": f"f{source_index}",
                "frame_index": source_index,
                "timestamp_seconds": source_index / 10,
                "input_size": [100, 100],
                "detections": deepcopy(pipe._detector.rows),
            },
            image=np.ascontiguousarray(np.asarray(image)[:, :, ::-1])
            if algorithm == "botsort"
            else None,
        )
        assert semantic_frame(actual["tracking"]) == semantic_frame(expected)
    pipe.reset("reset")
    direct.reset("reset")
    actual = update(pipe, image=image)
    expected = direct.update(
        {
            "frame_id": "f0",
            "frame_index": 0,
            "timestamp_seconds": 0,
            "input_size": [100, 100],
            "detections": [detection(2)],
        },
        image=np.ascontiguousarray(np.asarray(image)[:, :, ::-1])
        if algorithm == "botsort"
        else None,
    )
    assert semantic_frame(actual["tracking"]) == semantic_frame(expected)


def test_interruption_after_native_mutation_also_poisons_until_reset(build):
    pipe = build()
    pipe.reset("one")
    original = pipe._tracker.update

    def interrupted(*args, **kwargs):
        original(*args, **kwargs)
        raise KeyboardInterrupt("Interrupted during native update")

    pipe._tracker.update = interrupted
    with pytest.raises(KeyboardInterrupt):
        update(pipe)
    pipe._tracker.update = original
    with pytest.raises(runtime.PipelineError, match="reset"):
        update(pipe, 1)
    pipe.reset("two")
    assert update(pipe)["update_index"] == 1
