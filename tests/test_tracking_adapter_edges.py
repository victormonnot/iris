"""Independent lifecycle and isolation checks against the installed native trackers."""

import json
import subprocess
import sys
from copy import deepcopy

import cv2
import numpy as np
import pytest

from iris.tracking import TrackingError, make_tracker
from iris.tracking_contracts import make_profile, semantic_frame


def frame(index, detections=(), *, timestamp=None):
    return {
        "frame_id": f"source-{index}",
        "frame_index": index,
        "timestamp_seconds": timestamp,
        "input_size": [100, 80],
        "detections": list(detections),
    }


def detection(index=7, label=1, score=0.9):
    return {
        "detection_index": index,
        "label_id": label,
        "label": str(label),
        "score": score,
        "box": [10, 10, 30, 40],
    }


@pytest.fixture
def native_runtime():
    pytest.importorskip("scipy", reason="optional tracking runtime not installed")
    pytest.importorskip("lap", reason="optional tracking runtime not installed")
    pytest.importorskip("cython_bbox", reason="optional tracking runtime not installed")


@pytest.mark.parametrize("algorithm", ["bytetrack", "botsort"])
def test_interleaved_adapters_preserve_external_counter_and_independent_semantics(
    native_runtime, algorithm
):
    profile = make_profile(algorithm, class_ids=[1, 3], gmc_method="none")
    first, second = make_tracker(profile), make_tracker(profile)
    original_count = first._base._count
    first._base._count = 4321
    try:
        first.reset("shared-sequence")
        second.reset("shared-sequence")
        assert first._base._count == 4321
        for source in [
            frame(0, [detection(3, 1), detection(8, 3)]),
            frame(500, [detection(11, 3)]),
            frame(501, [detection(6, 1), detection(20, 3)]),
        ]:
            baseline = first.update(source)
            assert first._base._count == 4321
            replay = second.update(source)
            assert first._base._count == 4321
            assert semantic_frame(baseline) == semantic_frame(replay)
        assert {row["track_id"] for row in baseline["observations"]} == {1, 2}
    finally:
        first._base._count = original_count


@pytest.mark.parametrize("algorithm", ["bytetrack", "botsort"])
def test_late_first_class_observation_does_not_get_first_update_confirmation(
    native_runtime, algorithm
):
    tracker = make_tracker(make_profile(algorithm, class_ids=[1, 3], gmc_method="none"))
    tracker.reset("late-class")
    tracker.update(frame(0, [detection(0, 1)]))
    later = tracker.update(frame(1, [detection(11, 3)]))
    if algorithm == "bytetrack":
        assert later["observations"] == []
        assert later["unassigned"][0]["reason"] == "native_unconfirmed"
    else:
        assert len(later["observations"]) == 1
        assert later["observations"][0]["confirmed"] is False
    confirmed = tracker.update(frame(2, [detection(17, 3)]))
    assert confirmed["observations"][0]["confirmed"] is True
    assert confirmed["observations"][0]["track_id"] == 2
    assert confirmed["observations"][0]["detection_index"] == 17


def test_actual_gmc_is_shared_once_even_on_empty_frames(native_runtime, monkeypatch):
    tracker = make_tracker(make_profile("botsort", class_ids=[1, 3]))
    tracker.reset("camera")
    original = tracker._gmc.apply
    images_seen = []

    def counted(image, detections=None):
        images_seen.append(image)
        return original(image, detections)

    monkeypatch.setattr(tracker._gmc, "apply", counted)
    texture = np.random.default_rng(17).integers(0, 256, (80, 100, 3), dtype=np.uint8)
    first = tracker.update(frame(0, [detection(3, 1), detection(8, 3)]), image=texture)
    empty = tracker.update(frame(200), image=texture)
    assert len(images_seen) == 2
    assert first["gmc"]["status"] == "initialized"
    assert empty["gmc"]["status"] == "estimated"
    assert empty["observations"] == []
    assert {row["label_id"] for row in empty["predictions"]} == {1, 3}
    assert {row["age_updates"] for row in empty["predictions"]} == {1}
    np.testing.assert_allclose(empty["gmc"]["matrix"], np.eye(2, 3), atol=1e-6)


@pytest.mark.parametrize("failure_stage", ["gmc", "association"])
def test_native_failure_restores_shared_state_and_poison_requires_reset(
    native_runtime, monkeypatch, failure_stage
):
    tracker = make_tracker(make_profile("botsort", class_ids=[1]))
    tracker.reset("failure")
    texture = np.random.default_rng(8).integers(0, 256, (80, 100, 3), dtype=np.uint8)
    tracker.update(frame(0, [detection()]), image=texture)
    previous_count, previous_threads = tracker._base._count, cv2.getNumThreads()
    tracker._base._count = 9876
    cv2.setNumThreads(3)

    def fail(*args, **kwargs):
        raise ValueError("intentional native failure")

    try:
        if failure_stage == "gmc":
            monkeypatch.setattr(tracker._gmc, "apply", fail)
        else:
            monkeypatch.setattr(tracker._native[1], "update", fail)
        with pytest.raises(TrackingError, match="failed on frame 1"):
            tracker.update(frame(1), image=texture)
        assert tracker._base._count == 9876
        assert cv2.getNumThreads() == 3
        with pytest.raises(TrackingError, match="reset"):
            tracker.update(frame(2), image=texture)
        tracker.reset("fresh-sequence")
        fresh = tracker.update(frame(0, [detection()]), image=texture)
        assert fresh["observations"][0]["track_id"] == 1
        assert fresh["gmc"]["status"] == "initialized"
        assert tracker._base._count == 9876
        assert cv2.getNumThreads() == 3
    finally:
        tracker._base._count = previous_count
        cv2.setNumThreads(previous_threads)


def test_invalid_input_can_be_corrected_without_changing_live_state(native_runtime):
    tracker = make_tracker(make_profile("bytetrack", class_ids=[1]))
    tracker.reset("input")
    tracker.update(frame(0, [detection()], timestamp=0.1))
    valid = frame(2, [detection()], timestamp=0.2)
    wrong_name = deepcopy(valid)
    wrong_name["detections"][0]["label"] = "changed"
    for invalid in [wrong_name, frame(2, [detection()], timestamp=None), frame(0)]:
        with pytest.raises(ValueError):
            tracker.update(invalid)
    result = tracker.update(valid)
    assert result["update_index"] == 2
    assert result["observations"][0]["track_id"] == 1


def test_base_imports_and_cli_status_survive_absent_optional_tracking_and_ml():
    script = """
import builtins
import importlib.metadata
import json
import sys

original_import = builtins.__import__
blocked = {'scipy', 'lap', 'cython_bbox', 'torch', 'torchvision'}
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in blocked:
        raise AssertionError('Optional runtime imported by a base IRIS path: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded
original_version = importlib.metadata.version
def absent_version(name):
    if name in blocked:
        raise importlib.metadata.PackageNotFoundError(name)
    return original_version(name)
importlib.metadata.version = absent_version

import iris.tracking
import iris.tracking_replay
import iris.cli
from iris.tracking_contracts import make_profile
assert iris.tracking.tracking_status()['available'] is False
try:
    iris.tracking.make_tracker(make_profile('bytetrack', class_ids=[1]))
except ImportError as exc:
    assert 'optional tracking extra' in str(exc)
else:
    raise AssertionError('An absent optional runtime must not construct a tracker')
sys.argv = ['iris', 'tracking', 'status']
iris.cli.main()
assert not any(name.split('.')[0] in blocked for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["available"] is False
