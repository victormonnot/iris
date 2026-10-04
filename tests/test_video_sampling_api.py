"""Video sampling previews and real local extraction using generated media."""

import base64
import hashlib
import io
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import iris.app as app_module
from iris.app import create_app

BASE_URL = "http://127.0.0.1"


@pytest.fixture
def video_bytes(tmp_path):
    path = tmp_path / "sampling.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 6, (48, 32))
    assert writer.isOpened(), "Synthetic video tests require the local MJPEG encoder"
    try:
        for index in range(24):
            writer.write(np.full((32, 48, 3), (index * 9, 40, 120), dtype=np.uint8))
    finally:
        writer.release()
    return path.read_bytes()


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path / "workspace", run_jobs=False), base_url=BASE_URL) as api:
        yield api


def _session(client, name="Sampling fixture"):
    response = client.post("/api/sessions", json={"name": name, "scene_group": name})
    assert response.status_code == 201, response.text
    return response.json()


def _upload(client, session_id, content, filename="sampling.avi"):
    response = client.post(
        f"/api/sessions/{session_id}/assets",
        files={"file": (filename, content, "application/octet-stream")},
    )
    assert response.status_code == 201, response.text
    return response.json()


def _video(client, video_bytes):
    flight = _session(client)
    asset = _upload(client, flight["id"], video_bytes)
    return flight, asset


def _snapshot(store):
    with store.connect() as conn:
        database = tuple(conn.iterdump())
    artifacts = {
        str(path.relative_to(store.root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in store.root.rglob("*")
        if path.is_file() and path != store.db_path
    }
    return database, artifacts


def _wait_for_job(client, job_id):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        job = next(job for job in client.get("/api/jobs").json() if job["id"] == job_id)
        if job["status"] not in {"queued", "running"}:
            return job
        time.sleep(0.05)
    pytest.fail(f"Worker did not finish within 15 seconds: {job}")


def test_uniform_preview_spans_the_video_and_changes_nothing(client, video_bytes):
    flight, asset = _video(client, video_bytes)
    store = client.app.state.store
    before = _snapshot(store)
    response = client.post(
        f"/api/assets/{asset['id']}/extract/preview",
        json={"sampling_mode": "uniform", "max_frames": 3, "end_seconds": 100},
    )
    assert response.status_code == 200, response.text
    plan = response.json()
    assert plan["algorithm"] == "iris-video-sampling-v1"
    assert plan["sampling_mode"] == "uniform"
    assert plan["asset_id"] == asset["id"]
    assert plan["source_sha256"] == asset["sha256"]
    assert plan["fps"] == 6
    assert plan["duration_seconds"] == plan["end_seconds"] == 4
    assert plan["start_seconds"] == 0
    assert plan["timestamp_basis"] == "frame_index / nominal_fps"
    assert plan["max_frames"] == plan["planned_count"] == 3
    assert plan["interval_seconds"] is None
    assert plan["truncated"] is False
    positions = plan["positions"]
    assert positions[0] == {"frame_index": 0, "timestamp_seconds": 0}
    assert positions[-1] == {"frame_index": 23, "timestamp_seconds": 23 / 6}
    assert positions[1]["frame_index"] in {11, 12}
    assert plan["first_timestamp_seconds"] == positions[0]["timestamp_seconds"]
    assert plan["last_timestamp_seconds"] == positions[-1]["timestamp_seconds"]
    assert "path" not in plan
    assert "thumbnails" not in plan or plan["thumbnails"] == []
    assert client.get(f"/api/sessions/{flight['id']}/frames").json() == []
    assert _snapshot(store) == before


def test_uniform_preview_respects_a_trimmed_range_and_single_frame_budget(client, video_bytes):
    _, asset = _video(client, video_bytes)
    endpoint = f"/api/assets/{asset['id']}/extract/preview"
    response = client.post(
        endpoint,
        json={"sampling_mode": "uniform", "start_seconds": 0.51, "end_seconds": 1.51},
    )
    assert response.status_code == 200, response.text
    positions = response.json()["positions"]
    assert [position["frame_index"] for position in positions] == list(range(4, 10))
    assert all(0.51 <= position["timestamp_seconds"] < 1.51 for position in positions)
    midpoint = client.post(endpoint, json={"sampling_mode": "uniform", "max_frames": 1})
    assert midpoint.status_code == 200, midpoint.text
    assert midpoint.json()["planned_count"] == 1
    assert midpoint.json()["positions"][0]["frame_index"] in {11, 12}


def test_interval_preview_exposes_prefix_truncation_and_preserves_default_queue_config(
    client, video_bytes
):
    _, asset = _video(client, video_bytes)
    endpoint = f"/api/assets/{asset['id']}/extract"
    params = {"interval_seconds": 0.5, "max_frames": 3}
    preview = client.post(endpoint + "/preview", json=params)
    assert preview.status_code == 200, preview.text
    plan = preview.json()
    assert plan["sampling_mode"] == "interval"
    assert plan["interval_seconds"] == 0.5
    assert plan["truncated"] is True
    assert [position["frame_index"] for position in plan["positions"]] == [0, 3, 6]
    queued = client.post(endpoint, json=params)
    assert queued.status_code == 202, queued.text
    saved = queued.json()["params"]
    assert saved["asset_id"] == asset["id"]
    assert saved["config"] == {
        "interval_seconds": 0.5,
        "start_seconds": 0,
        "end_seconds": None,
        "max_frames": 3,
        "dedup_hamming": None,
    }
    assert saved["extraction_contract"]["plan"] == plan
    assert saved["extraction_contract"]["source_sha256"] == asset["sha256"]


@pytest.mark.parametrize("suffix", ["", "/preview", "/preview-images"])
@pytest.mark.parametrize(
    "payload",
    [
        {"sampling_mode": "unknown"},
        {"sampling_mode": True},
        {"sampling_mode": "uniform", "max_frames": "3"},
        {"sampling_mode": "uniform", "max_frames": 0},
        {"sampling_mode": "uniform", "start_seconds": 4},
        {"sampling_mode": "uniform", "start_seconds": 2, "end_seconds": 1},
        {"sampling_mode": "uniform", "unexpected": True},
    ],
)
def test_invalid_sampling_requests_do_not_mutate_workspace(client, video_bytes, suffix, payload):
    _, asset = _video(client, video_bytes)
    before = _snapshot(client.app.state.store)
    response = client.post(f"/api/assets/{asset['id']}/extract{suffix}", json=payload)
    assert response.status_code == 422, response.text
    assert _snapshot(client.app.state.store) == before


@pytest.mark.parametrize("suffix", ["/preview", "/preview-images"])
def test_previews_reject_missing_assets_and_still_images(client, suffix):
    assert client.post(f"/api/assets/missing/extract{suffix}", json={}).status_code == 404
    flight = _session(client)
    output = io.BytesIO()
    Image.new("RGB", (48, 32), "blue").save(output, format="PNG")
    asset = _upload(client, flight["id"], output.getvalue(), "still.png")
    before = _snapshot(client.app.state.store)
    response = client.post(f"/api/assets/{asset['id']}/extract{suffix}", json={})
    assert response.status_code == 422, response.text
    assert _snapshot(client.app.state.store) == before


def test_thumbnail_preview_is_bounded_and_matches_the_plan_without_extracting(client, video_bytes):
    flight, asset = _video(client, video_bytes)
    store = client.app.state.store
    before = _snapshot(store)
    endpoint = f"/api/assets/{asset['id']}/extract"
    params = {"sampling_mode": "uniform", "max_frames": 24}
    plan = client.post(endpoint + "/preview", json=params).json()
    response = client.post(endpoint + "/preview-images", json=params)
    assert response.status_code == 200, response.text
    preview = response.json()
    for key, value in plan.items():
        if key != "thumbnails":
            assert preview[key] == value
    thumbnails = preview["thumbnails"]
    assert len(thumbnails) == 12
    indices = [thumbnail["frame_index"] for thumbnail in thumbnails]
    assert indices == sorted(set(indices))
    assert (indices[0], indices[-1]) == (0, 23)
    for thumbnail in thumbnails:
        position = {
            "frame_index": thumbnail["frame_index"],
            "timestamp_seconds": thumbnail["timestamp_seconds"],
        }
        assert position in plan["positions"]
        prefix, encoded = thumbnail["image_data_url"].split(",", 1)
        assert prefix == "data:image/jpeg;base64"
        with Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))) as image:
            assert image.format == "JPEG"
            assert 0 < image.width <= 48 and 0 < image.height <= 32
            expected_blue = thumbnail["frame_index"] * 9
            assert abs(image.convert("RGB").getpixel((0, 0))[2] - expected_blue) < 12
    assert client.get(f"/api/sessions/{flight['id']}/frames").json() == []
    assert _snapshot(store) == before


@pytest.mark.parametrize("tamper", ["changed", "container", "missing", "outside", "symlink"])
def test_thumbnail_preview_rejects_changed_or_unsafe_sources_without_mutation(
    client, video_bytes, tmp_path, tamper
):
    _, asset = _video(client, video_bytes)
    store = client.app.state.store
    record = store.get("assets", asset["id"])
    source = store.artifact_path(record["path"])
    if tamper == "changed":
        source.write_bytes(source.read_bytes() + b"changed fixture")
    elif tamper == "container":
        source.write_bytes(b"not a video container")
        store.update(
            "assets", asset["id"], {"sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
        )
    elif tamper == "missing":
        source.unlink()
    else:
        outside = tmp_path / "outside.avi"
        outside.write_bytes(video_bytes)
        if tamper == "outside":
            store.update("assets", asset["id"], {"path": str(outside)})
        else:
            source.unlink()
            source.symlink_to(outside)
    before = _snapshot(store)
    endpoint = f"/api/assets/{asset['id']}/extract"
    # Planning uses imported metadata, while image decoding verifies the source.
    assert client.post(endpoint + "/preview", json={"sampling_mode": "uniform"}).status_code == 200
    response = client.post(endpoint + "/preview-images", json={"sampling_mode": "uniform"})
    assert response.status_code == 422, response.text
    assert _snapshot(store) == before


def test_image_preview_serializes_decoding_but_metadata_preview_remains_available(
    client, video_bytes, monkeypatch
):
    _, first_asset = _video(client, video_bytes)
    other_flight = _session(client, "Another video")
    second_asset = _upload(client, other_flight["id"], video_bytes)
    entered = threading.Event()
    release = threading.Event()
    original = app_module.preview_extraction

    def gated_preview(*args, **kwargs):
        if kwargs.get("include_images"):
            entered.set()
            assert release.wait(10), "Test did not release image preview decoding"
        return original(*args, **kwargs)

    monkeypatch.setattr(app_module, "preview_extraction", gated_preview)
    before = _snapshot(client.app.state.store)
    first_endpoint = f"/api/assets/{first_asset['id']}/extract"
    second_endpoint = f"/api/assets/{second_asset['id']}/extract"
    with ThreadPoolExecutor(max_workers=1) as executor:
        request = executor.submit(client.post, first_endpoint + "/preview-images", json={})
        try:
            assert entered.wait(5), "The image preview did not enter decoding"
            assert client.post(second_endpoint + "/preview-images", json={}).status_code == 409
            assert client.post(second_endpoint + "/preview", json={}).status_code == 200
        finally:
            release.set()
        response = request.result(timeout=10)
    assert response.status_code == 200, response.text
    assert client.post(second_endpoint + "/preview-images", json={}).status_code == 200
    assert _snapshot(client.app.state.store) == before


def test_failed_thumbnail_preview_releases_decoding_lock(client, video_bytes):
    _, asset = _video(client, video_bytes)
    store = client.app.state.store
    source = store.artifact_path(store.get("assets", asset["id"])["path"])
    endpoint = f"/api/assets/{asset['id']}/extract/preview-images"
    source.write_bytes(source.read_bytes() + b"changed fixture")
    assert client.post(endpoint, json={"sampling_mode": "uniform"}).status_code == 422
    source.write_bytes(video_bytes)
    response = client.post(endpoint, json={"sampling_mode": "uniform"})
    assert response.status_code == 200, response.text
    assert response.json()["thumbnails"]
    assert client.get("/api/jobs").json() == []


def test_real_uniform_worker_matches_preview_and_preserves_selection_across_restart(
    tmp_path, video_bytes
):
    data_dir = tmp_path / "workspace"
    params = {"sampling_mode": "uniform", "max_frames": 3}
    with TestClient(create_app(data_dir), base_url=BASE_URL) as client:
        flight, asset = _video(client, video_bytes)
        endpoint = f"/api/assets/{asset['id']}/extract"
        preview = client.post(endpoint + "/preview", json=params)
        assert preview.status_code == 200, preview.text
        plan = preview.json()
        queued = client.post(endpoint, json=params)
        assert queued.status_code == 202, queued.text
        completed = _wait_for_job(client, queued.json()["id"])
        assert completed["status"] == "succeeded", completed
        assert completed["result"]["plan"] == plan
        assert completed["result"]["sampled"] == completed["result"]["created"] == 3
        frames_url = f"/api/sessions/{flight['id']}/frames"
        frames = client.get(frames_url).json()
        observed = sorted((frame["frame_index"], frame["timestamp_seconds"]) for frame in frames)
        assert observed == [
            (position["frame_index"], position["timestamp_seconds"])
            for position in plan["positions"]
        ]
        assert not any(frame["selected"] for frame in frames)
        for frame in frames:
            assert frame["extraction"]["sampling_mode"] == "uniform"
            assert frame["extraction"]["sampling_algorithm"] == "iris-video-sampling-v1"
            assert frame["extraction"]["job_id"] == completed["id"]
            assert frame["extraction"]["sampling_plan"]["planned_count"] == 3
        selected = client.patch(f"/api/frames/{frames[0]['id']}", json={"selected": True})
        assert selected.status_code == 200
        preserved = client.get(frames_url).json()
        assert client.post(endpoint + "/preview-images", json=params).status_code == 200
        assert client.get(frames_url).json() == preserved
        repeated = client.post(endpoint, json=params)
        assert repeated.status_code == 202, repeated.text
        second = _wait_for_job(client, repeated.json()["id"])
        assert second["status"] == "succeeded", second
        assert second["result"]["created"] == 0
        assert second["result"]["skipped_existing"] == 3
        assert client.get(frames_url).json() == preserved
        jobs = client.get("/api/jobs").json()

    with TestClient(create_app(data_dir), base_url=BASE_URL) as reopened:
        assert reopened.get(frames_url).json() == preserved
        assert reopened.get("/api/jobs").json() == jobs
        assert reopened.post(endpoint + "/preview", json=params).json() == plan
