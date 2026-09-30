"""Sampling plans and local previews use generated videos, never flight data."""

import base64
import io
import math
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from iris.app import create_app
from iris.media import extract_frames, import_asset, preview_extraction
from iris.store import Store, new_id, now
from iris.video_sampling import plan_extraction


def metadata(count=100, fps=10.0):
    return {"frame_count": count, "fps": fps, "duration_seconds": count / fps}


def indices(plan):
    return [position["frame_index"] for position in plan["positions"]]


def test_uniform_samples_full_frame_range_with_bounded_integer_spacing():
    plan = plan_extraction(metadata(), {"sampling_mode": "uniform", "max_frames": 5})
    assert indices(plan) == [0, 25, 49, 74, 99]
    assert plan["first_timestamp_seconds"] == 0
    assert plan["last_timestamp_seconds"] == 9.9
    assert plan["planned_count"] == plan["max_frames"] == 5
    assert plan["algorithm"] == "iris-video-sampling-v1"
    assert plan["interval_seconds"] is None
    assert plan["truncated"] is False


def test_uniform_ignores_interval_and_uses_middle_frame_when_budget_is_one():
    config = {"sampling_mode": "uniform", "max_frames": 1}
    first = plan_extraction(metadata(), {**config, "interval_seconds": 0.001})
    second = plan_extraction(metadata(), {**config, "interval_seconds": 999})
    assert first == second
    assert indices(first) == [49]


def test_uniform_never_duplicates_when_budget_exceeds_available_frames():
    plan = plan_extraction(metadata(3), {"sampling_mode": "uniform", "max_frames": 500})
    assert indices(plan) == [0, 1, 2]


def test_uniform_range_is_half_open_and_clamped_to_video_duration():
    config = {"sampling_mode": "uniform", "start_seconds": 0.31, "end_seconds": 1.01}
    plan = plan_extraction(metadata(), config)
    assert indices(plan) == list(range(4, 11))
    assert all(0.31 <= row["timestamp_seconds"] < 1.01 for row in plan["positions"])
    clamped = plan_extraction(metadata(), {**config, "end_seconds": 1000})
    assert clamped["end_seconds"] == clamped["duration_seconds"] == 10
    assert indices(clamped)[-1] == 99


@pytest.mark.parametrize("fps", [29.97, 30000 / 1001, 23.976, 1 / 3, 1000.0])
def test_uniform_fractional_fps_boundaries_use_actual_nominal_timestamps(fps):
    lower, upper = 37 / fps, 74 / fps
    exact = plan_extraction(
        metadata(100, fps),
        {"sampling_mode": "uniform", "start_seconds": lower, "end_seconds": upper},
    )
    assert indices(exact) == list(range(37, 74))
    inside = plan_extraction(
        metadata(100, fps),
        {
            "sampling_mode": "uniform",
            "start_seconds": math.nextafter(lower, math.inf),
            "end_seconds": math.nextafter(upper, -math.inf),
        },
    )
    assert indices(inside) == list(range(38, 74))


def test_uniform_plan_is_bounded_for_enormous_video_metadata():
    count = 10**16
    plan = plan_extraction(metadata(count, 1000), {"sampling_mode": "uniform", "max_frames": 500})
    assert len(plan["positions"]) == 500
    assert len(set(indices(plan))) == 500
    assert indices(plan) == sorted(indices(plan))
    assert plan["first_timestamp_seconds"] == 0
    assert plan["last_timestamp_seconds"] < plan["end_seconds"]
    assert indices(plan)[-1] >= count - 3


def test_legacy_interval_preserves_prefix_floor_and_limit():
    config = {"start_seconds": 0.51, "end_seconds": 1.8, "interval_seconds": 0.5, "max_frames": 2}
    plan = plan_extraction(metadata(8, 4), config)
    assert indices(plan) == [2, 4]
    assert plan["sampling_mode"] == "interval"
    assert plan["truncated"] is True
    assert plan["first_timestamp_seconds"] == 0.5  # Legacy floor is intentional.
    assert plan == plan_extraction(metadata(8, 4), {**config, "sampling_mode": "interval"})


def test_legacy_subframe_interval_is_clamped_and_end_is_exclusive():
    plan = plan_extraction(metadata(8, 4), {"interval_seconds": 0.001, "max_frames": 500})
    assert indices(plan) == list(range(8))
    assert plan["interval_seconds"] == 0.25
    assert plan["truncated"] is False


@pytest.mark.parametrize(
    "config",
    [
        {"sampling_mode": "unknown"},
        {"sampling_mode": None},
        {"sampling_mode": ["uniform"]},
        {"max_frames": True},
        {"max_frames": 0},
        {"max_frames": 501},
        {"max_frames": 1.5},
        {"start_seconds": float("nan")},
        {"start_seconds": -1},
        {"start_seconds": True},
        {"start_seconds": 10},
        {"start_seconds": 1, "end_seconds": 1},
        {"end_seconds": float("inf")},
        {"interval_seconds": 0},
        {"interval_seconds": None},
        {"interval_seconds": True},
        {"dedup_hamming": -1},
        {"dedup_hamming": 17},
        {"dedup_hamming": False},
    ],
)
def test_invalid_plan_settings_are_rejected(config):
    with pytest.raises(ValueError):
        plan_extraction(metadata(), config)


@pytest.mark.parametrize(
    "update",
    [
        {"fps": 0},
        {"fps": float("nan")},
        {"fps": float("inf")},
        {"fps": True},
        {"fps": 1e-320},
        {"fps": None},
        {"duration_seconds": 0},
        {"duration_seconds": float("inf")},
        {"frame_count": True},
        {"frame_count": 1.5},
        {"frame_count": 0},
        {"frame_count": "100"},
    ],
)
def test_corrupt_metadata_is_rejected_before_planning(update):
    with pytest.raises(ValueError):
        plan_extraction({**metadata(), **update}, {"sampling_mode": "uniform"})


def test_empty_uniform_range_rejects_frames_outside_chosen_times():
    with pytest.raises(ValueError, match="no sampleable"):
        plan_extraction(
            metadata(8, 4),
            {"sampling_mode": "uniform", "start_seconds": 0.01, "end_seconds": 0.2},
        )


@pytest.fixture
def video(tmp_path):
    store = Store(tmp_path / "workspace")
    session = store.insert(
        "sessions",
        {"id": new_id(), "name": "Generated video", "scene_group": "fixture", "created_at": now()},
    )
    source = tmp_path / "synthetic.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 4.0, (480, 320))
    assert writer.isOpened(), "The local MJPEG encoder is required for synthetic fixtures"
    try:
        for index in range(32):
            # Distinct, compressible solid colors test exact frame identity.
            writer.write(np.full((320, 480, 3), (index * 7, 30, 150), dtype=np.uint8))
    finally:
        writer.release()
    asset = import_asset(store, session["id"], source, source.name)
    return store, asset


def workspace_snapshot(store):
    return {
        "files": {str(path.relative_to(store.root)) for path in store.root.rglob("*")},
        "assets": store.list("assets"),
        "frames": store.list("frames"),
        "jobs": store.list("jobs"),
    }


def test_preview_plan_does_not_decode_or_mutate_workspace(video, monkeypatch):
    store, asset = video
    before = workspace_snapshot(store)

    def forbidden(*args, **kwargs):
        raise AssertionError("Planning must not decode images")

    monkeypatch.setattr(cv2, "VideoCapture", forbidden)
    result = preview_extraction(store, asset["id"], {"sampling_mode": "uniform", "max_frames": 5})
    assert indices(result) == [0, 8, 15, 23, 31]
    assert result["source_sha256"] == asset["sha256"]
    assert result["asset_id"] == asset["id"]
    assert "thumbnails" not in result
    assert workspace_snapshot(store) == before


def test_real_preview_thumbnails_are_bounded_and_show_exact_planned_endpoints(video):
    store, asset = video
    before = workspace_snapshot(store)
    result = preview_extraction(
        store, asset["id"], {"sampling_mode": "uniform", "max_frames": 25}, include_images=True
    )
    assert len(result["positions"]) == 25
    assert len(result["thumbnails"]) == 12
    assert result["thumbnails"][0]["frame_index"] == 0
    assert result["thumbnails"][-1]["frame_index"] == 31
    by_index = {position["frame_index"]: position for position in result["positions"]}
    for thumbnail in result["thumbnails"]:
        assert (
            thumbnail["timestamp_seconds"]
            == by_index[thumbnail["frame_index"]]["timestamp_seconds"]
        )
        prefix, encoded = thumbnail["image_data_url"].split(",", 1)
        assert prefix == "data:image/jpeg;base64"
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
            assert image.format == "JPEG"
            assert image.mode == "RGB"
            assert image.size == (324, 216)
            red, green, blue = image.getpixel((100, 100))
            assert abs(red - 150) <= 5
            assert abs(green - 30) <= 5
            assert abs(blue - thumbnail["frame_index"] * 7) <= 5
    assert workspace_snapshot(store) == before


@pytest.mark.parametrize("operation", ["preview", "extract"])
def test_changed_video_is_rejected_before_any_decoder_is_opened(video, monkeypatch, operation):
    store, asset = video
    source = store.artifact_path(asset["path"])
    with source.open("ab") as output:
        output.write(b"source changed after import")

    def forbidden(*args, **kwargs):
        raise AssertionError("Changed media must not reach a decoder")

    monkeypatch.setattr(cv2, "VideoCapture", forbidden)
    with pytest.raises(ValueError, match="changed since import"):
        if operation == "preview":
            preview_extraction(store, asset["id"], {}, include_images=True)
        else:
            extract_frames(store, asset["id"], {}, lambda *_: None, lambda: False)
    assert store.list("frames") == []


def test_preview_refuses_source_path_outside_workspace(video, tmp_path):
    store, asset = video
    private = tmp_path / "private.avi"
    private.write_bytes(b"not media")
    linked = store.root / "linked.avi"
    linked.symlink_to(private)
    store.update("assets", asset["id"], {"path": "linked.avi"})
    with pytest.raises(ValueError, match="outside"):
        preview_extraction(store, asset["id"], {}, include_images=True)


def test_uniform_real_extraction_matches_preview_and_preserves_existing_selections(video):
    store, asset = video
    config = {"sampling_mode": "uniform", "max_frames": 5, "job_id": "fixture-job"}
    preview = preview_extraction(store, asset["id"], config)
    first = extract_frames(store, asset["id"], config, lambda *_: None, lambda: False)
    assert first["created"] == first["sampled"] == 5
    assert first["plan"]["positions"] == preview["positions"]
    frames = sorted(store.list("frames"), key=lambda frame: frame["frame_index"])
    assert [frame["frame_index"] for frame in frames] == indices(preview)
    assert all(
        frame["extraction"]["sampling_algorithm"] == preview["algorithm"] for frame in frames
    )
    assert all(frame["extraction"]["sampling_plan"]["planned_count"] == 5 for frame in frames)
    assert all(frame["selected"] is False for frame in frames)
    store.update("frames", frames[2]["id"], {"selected": True})
    before = store.list("frames")
    repeated = extract_frames(store, asset["id"], config, lambda *_: None, lambda: False)
    assert repeated["created"] == 0
    assert repeated["skipped_existing"] == 5
    assert store.list("frames") == before


def test_uniform_cancellation_can_resume_the_identical_plan(video):
    store, asset = video
    config = {"sampling_mode": "uniform", "max_frames": 7}
    updates = []
    partial = extract_frames(
        store, asset["id"], config, lambda *args: updates.append(args), lambda: len(updates) == 2
    )
    assert partial["cancelled"] is True
    assert partial["created"] == 2
    complete = extract_frames(store, asset["id"], config, lambda *_: None, lambda: False)
    assert complete["plan"] == partial["plan"]
    assert complete["skipped_existing"] == 2
    assert complete["created"] == 5
    assert len(store.list("frames")) == 7


def test_cancellation_during_source_hashing_opens_no_decoder(video, monkeypatch):
    store, asset = video
    checks = 0

    def cancelled():
        nonlocal checks
        checks += 1
        return checks >= 3

    def forbidden(*args, **kwargs):
        raise AssertionError("Cancelled source hashing must not open the decoder")

    monkeypatch.setattr(cv2, "VideoCapture", forbidden)
    result = extract_frames(store, asset["id"], {}, lambda *_: None, cancelled)
    assert result["cancelled"] is True
    assert result["sampled"] == result["created"] == 0
    assert store.list("frames") == []


def test_cancellation_after_decode_does_not_save_the_pending_frame(video, monkeypatch):
    store, asset = video
    original_capture = cv2.VideoCapture
    decoded = False

    class CancellingCapture:
        def __init__(self, path):
            self.capture = original_capture(path)

        def isOpened(self):
            return self.capture.isOpened()

        def set(self, key, value):
            return self.capture.set(key, value)

        def read(self):
            nonlocal decoded
            result = self.capture.read()
            decoded = True
            return result

        def release(self):
            self.capture.release()

    monkeypatch.setattr(cv2, "VideoCapture", CancellingCapture)
    result = extract_frames(store, asset["id"], {}, lambda *_: None, lambda: decoded)
    assert result["cancelled"] is True
    assert result["sampled"] == 1
    assert result["created"] == 0
    assert store.list("frames") == []
    assert not list((store.root / "frames").glob("*.png"))


@pytest.mark.parametrize("failure", ["seek to", "decode"])
def test_inaccurate_timing_preview_reports_recovery_without_artifacts(video, monkeypatch, failure):
    store, asset = video
    source = store.artifact_path(asset["path"])
    original_bytes = source.read_bytes()
    # Model an estimated container frame count that exceeds the decodable frames.
    store.update("assets", asset["id"], {"metadata": {**asset["metadata"], **metadata(64, 4)}})
    original_capture = cv2.VideoCapture
    released = []

    class EstimatedTimingCapture:
        def __init__(self, path):
            self.capture = original_capture(path)
            self.frame_index = 0

        def isOpened(self):
            return self.capture.isOpened()

        def set(self, key, value):
            self.frame_index = value
            if value >= 32:
                return failure != "seek to"
            return self.capture.set(key, value)

        def read(self):
            if self.frame_index >= 32:
                return False, None
            return self.capture.read()

        def release(self):
            released.append(True)
            self.capture.release()

    monkeypatch.setattr(cv2, "VideoCapture", EstimatedTimingCapture)
    with TestClient(create_app(store.root, run_jobs=False), base_url="http://127.0.0.1") as client:
        before = workspace_snapshot(store)
        response = client.post(
            f"/api/assets/{asset['id']}/extract/preview-images",
            json={"sampling_mode": "uniform", "max_frames": 3},
        )
        assert response.status_code == 422
        assert f"Cannot {failure} video frame 63" in response.json()["detail"]
        assert "timing metadata may be inaccurate" in response.json()["detail"]
        assert "try a shorter range or a constant-frame-rate copy" in response.json()["detail"]
        assert workspace_snapshot(store) == before
    assert released == [True]
    assert source.read_bytes() == original_bytes


def test_real_worker_keeps_completed_frames_when_estimated_endpoint_cannot_decode(video):
    store, asset = video
    source = store.artifact_path(asset["path"])
    original_bytes = source.read_bytes()
    store.update("assets", asset["id"], {"metadata": {**asset["metadata"], **metadata(64, 4)}})
    with TestClient(create_app(store.root), base_url="http://127.0.0.1") as client:
        response = client.post(
            f"/api/assets/{asset['id']}/extract",
            json={"sampling_mode": "uniform", "max_frames": 3},
        )
        assert response.status_code == 202
        job_id = response.json()["id"]
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            job = store.get("jobs", job_id)
            if job["status"] not in {"queued", "running"}:
                break
            time.sleep(0.05)
        else:
            pytest.fail("Fixture worker did not finish before the deadline")
        assert job["status"] == "failed", job
        assert "video frame 63" in job["error"]
        assert "timing metadata may be inaccurate" in job["error"]
        assert "try a shorter range or a constant-frame-rate copy" in job["error"]
        frames = sorted(store.list("frames"), key=lambda frame: frame["frame_index"])
        assert [frame["frame_index"] for frame in frames] == [0, 31]
        assert all(store.artifact_path(frame["path"]).is_file() for frame in frames)
        assert all(frame["extraction"]["job_id"] == job_id for frame in frames)
        assert client.get(f"/api/jobs/{job_id}/log").status_code == 200
    assert source.read_bytes() == original_bytes
