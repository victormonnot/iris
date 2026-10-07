"""Real pinned native algorithms: measured provenance, weak matches and temporal state."""

from copy import deepcopy

import numpy as np
import pytest

from iris.tracking import TrackingError, make_tracker, tracking_status
from iris.tracking_contracts import make_profile, semantic_frame


@pytest.fixture(autouse=True)
def optional_tracking_dependencies():
    if not tracking_status()["available"]:
        pytest.skip("Install the optional tracking extra for real native adapter tests")


@pytest.fixture(params=["bytetrack", "botsort"])
def adapter(request):
    tracker = make_tracker(make_profile(request.param, class_ids=[1, 3], gmc_method="none"))
    tracker.reset("sequence-a")
    return tracker


def detection(index=7, *, score=0.9, label=1, box=None):
    return {
        "detection_index": index,
        "label_id": label,
        "label": "person" if label == 1 else "car",
        "score": score,
        "box": [10, 10, 40, 70] if box is None else box,
    }


def frame(index, detections, *, at=None):
    return {
        "frame_id": f"frame-{index}",
        "frame_index": index,
        "timestamp_seconds": index / 10 if at is None else at,
        "input_size": [160, 128],
        "detections": detections,
    }


def test_real_low_score_second_pass_keeps_identity_and_exact_input_evidence(adapter):
    initial = frame(0, [detection()])
    weak = frame(1, [detection(index=93, score=0.2, box=[12, 10, 42, 70])])
    originals = deepcopy([initial, weak])
    a, b = adapter.update(initial), adapter.update(weak)
    assert a["observations"][0]["track_id"] == b["observations"][0]["track_id"]
    observed = b["observations"][0]
    assert {key: observed[key] for key in weak["detections"][0]} == weak["detections"][0]
    assert observed["estimated_box"] != observed["box"]
    assert [initial, weak] == originals
    assert not b["unassigned"] and not b["predictions"]


def test_one_empty_update_predicts_without_fabricating_detection_then_recovers_same_id(adapter):
    a = adapter.update(frame(0, [detection()]))
    lost = adapter.update(frame(1, []))
    back = adapter.update(frame(2, [detection(index=3)]))
    identity = a["observations"][0]["track_id"]
    assert lost["observations"] == lost["unassigned"] == []
    assert len(lost["predictions"]) == 1
    predicted = lost["predictions"][0]
    assert predicted["track_id"] == identity
    assert predicted["last_observed_frame_id"] == "frame-0"
    assert predicted["age_updates"] == 1
    assert predicted["age_seconds"] == 0.1
    assert "score" not in predicted and "detection_index" not in predicted
    assert back["observations"][0]["track_id"] == identity
    assert back["predictions"] == []


def test_overlapping_objects_of_different_classes_never_share_ids_or_measurements(adapter):
    a = adapter.update(frame(0, [detection(2), detection(5, label=3)]))
    ids = {row["label_id"]: row["track_id"] for row in a["observations"]}
    assert len(set(ids.values())) == 2
    b = adapter.update(frame(1, [detection(0, label=3)]))
    assert b["observations"][0]["track_id"] == ids[3]
    assert b["predictions"][0]["track_id"] == ids[1]
    assert b["predictions"][0]["label"] == "person"


@pytest.mark.parametrize(
    ("score", "reason"),
    [
        (0.001, "below_low_threshold"),
        (0.1, "below_low_threshold"),
        (0.2, "unmatched_low_confidence"),
        (0.5, "strict_high_boundary"),
        (0.55, "below_birth_threshold"),
    ],
)
def test_native_ignored_candidates_remain_explicit_without_invented_track_ids(
    adapter, score, reason
):
    original = detection(99, score=score)
    result = adapter.update(frame(0, [original]))
    assert result["observations"] == result["predictions"] == []
    assert result["unassigned"] == [{**original, "reason": reason}]
    assert "track_id" not in result["unassigned"][0]


def test_source_gap_is_one_update_not_hundreds_of_invented_empty_observations(adapter):
    adapter.update(frame(0, [detection()]))
    result = adapter.update(frame(1000, [], at=120))
    assert result["update_index"] == 2
    assert result["predictions"][0]["age_updates"] == 1
    assert result["predictions"][0]["age_seconds"] == 120


def test_unknown_clock_stays_unknown_in_predicted_ages(adapter):
    a, b = frame(0, [detection()]), frame(1, [])
    a["timestamp_seconds"] = b["timestamp_seconds"] = None
    adapter.update(a)
    output = adapter.update(b)
    assert output["timestamp_seconds"] is None
    assert output["predictions"][0]["age_seconds"] is None


@pytest.mark.parametrize("algorithm", ["bytetrack", "botsort"])
def test_buffer_uses_updates_and_preserves_native_expiry_order(algorithm):
    tracker = make_tracker(
        make_profile(algorithm, class_ids=[1], buffer_updates=1, gmc_method="none")
    )
    tracker.reset("sequence")
    first = tracker.update(frame(0, [detection()]))
    tracker.update(frame(1, []))
    # Association happens before native lost-track cleanup, so this is still a match.
    late = tracker.update(frame(2, [detection()]))
    assert late["observations"][0]["track_id"] == first["observations"][0]["track_id"]
    tracker.update(frame(3, []))
    expired = tracker.update(frame(4, []))
    assert not expired["predictions"]
    tracker.update(frame(5, []))
    reborn = tracker.update(frame(6, [detection()]))
    if algorithm == "bytetrack":
        assert reborn["unassigned"][0]["reason"] == "native_unconfirmed"
        reborn = tracker.update(frame(7, [detection()]))
    assert reborn["observations"][0]["track_id"] != first["observations"][0]["track_id"]


def test_reset_reproduces_sequence_semantics_and_does_not_retain_lost_objects(adapter):
    inputs = [frame(0, [detection()]), frame(1, []), frame(3, [detection(score=0.2)])]
    first = [semantic_frame(adapter.update(value)) for value in inputs]
    adapter.reset("sequence-a")
    second = [semantic_frame(adapter.update(value)) for value in inputs]
    assert second == first
    adapter.reset("sequence-b")
    assert adapter.update(frame(0, []))["predictions"] == []


@pytest.mark.parametrize(
    "change", ["backwards", "same_timestamp", "new_dimensions", "class_name", "clock"]
)
def test_invalid_sequence_update_does_not_mutate_state_and_valid_retry_succeeds(adapter, change):
    adapter.update(frame(0, [detection()]))
    invalid = frame(1, [detection()])
    if change == "backwards":
        invalid["frame_index"] = 0
    elif change == "same_timestamp":
        invalid["timestamp_seconds"] = 0
    elif change == "new_dimensions":
        invalid["input_size"] = [161, 128]
    elif change == "class_name":
        invalid["detections"][0]["label"] = "changed"
    else:
        invalid["timestamp_seconds"] = None
    with pytest.raises(ValueError):
        adapter.update(invalid)
    assert adapter.update(frame(1, [detection()]))["update_index"] == 2


def test_real_sparse_camera_estimate_and_frame_pixels_are_preserved():
    import cv2

    tracker = make_tracker(make_profile("botsort", class_ids=[1, 3]))
    tracker.reset("camera-motion")
    image = np.random.default_rng(17).integers(0, 256, (128, 160, 3), dtype=np.uint8)
    shifted = cv2.warpAffine(image, np.float32([[1, 0, 4], [0, 1, 2]]), (160, 128))
    before = image.copy(), shifted.copy()
    initial = tracker.update(frame(0, []), image=image)
    moving = tracker.update(frame(1, []), image=shifted)
    assert initial["gmc"]["status"] == "initialized"
    assert moving["gmc"]["status"] == "estimated"
    assert moving["gmc"]["matrix"][0][2] == pytest.approx(4, abs=0.7)
    assert moving["gmc"]["matrix"][1][2] == pytest.approx(2, abs=0.7)
    assert np.array_equal(image, before[0]) and np.array_equal(shifted, before[1])
    assert moving["observations"] == moving["predictions"] == []


def test_native_gmc_failure_is_explicit_requires_reset_and_never_silently_disables_camera_motion():
    tracker = make_tracker(make_profile("botsort", class_ids=[1]))
    tracker.reset("featureless")
    image = np.zeros((128, 160, 3), dtype=np.uint8)
    tracker.update(frame(0, [detection()]), image=image)
    with pytest.raises(TrackingError, match="failed on frame 1") as error:
        tracker.update(frame(1, [detection()]), image=image)
    assert error.value.__cause__ is not None
    with pytest.raises(TrackingError, match="reset"):
        tracker.update(frame(2, [detection()]), image=image)
    tracker.reset("fresh")
    assert tracker.update(frame(0, []), image=image)["gmc"]["status"] == "initialized"


def test_replay_runtime_cannot_silently_change_after_construction(adapter, monkeypatch):
    import iris.tracking as tracking

    expected = adapter.metadata["provenance"]
    monkeypatch.setattr(
        tracking, "_provenance", lambda _: {**expected, "manifest_sha256": "changed"}
    )
    with pytest.raises(TrackingError, match="runtime changed"):
        adapter.verify_runtime()
    with pytest.raises(TrackingError):
        adapter.reset("new-sequence")


def test_replay_thread_environment_is_part_of_runtime_identity(adapter, monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "31")
    with pytest.raises(TrackingError, match="runtime changed"):
        adapter.verify_runtime()


def test_nonadjacent_source_identity_reuse_is_rejected_before_state_changes(adapter):
    adapter.update(frame(0, [detection()]))
    adapter.update(frame(1, []))
    repeated = frame(2, [detection()])
    repeated["frame_id"] = "frame-0"
    with pytest.raises(ValueError, match="unique"):
        adapter.update(repeated)
    assert adapter.update(frame(2, [detection()]))["update_index"] == 3
    adapter.reset("another-sequence")
    assert adapter.update(frame(0, [detection()]))["update_index"] == 1
