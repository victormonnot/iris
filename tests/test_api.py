"""Local HTTP workflows with generated fixtures; no models or flight data."""

import io
import threading
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from iris.app import create_app
from iris.store import Store

BASE_URL = "http://127.0.0.1"


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path / "workspace", run_jobs=False), base_url=BASE_URL) as api:
        yield api


@pytest.fixture
def image_bytes():
    output = io.BytesIO()
    Image.new("RGB", (32, 24), (30, 70, 110)).save(output, format="PNG")
    return output.getvalue()


@pytest.fixture
def video_bytes(tmp_path):
    path = tmp_path / "generated.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 4, (48, 32))
    assert writer.isOpened(), "Synthetic video tests require the local MJPEG encoder"
    try:
        for index in range(8):
            writer.write(np.full((32, 48, 3), (index * 25, 40, 120), dtype=np.uint8))
    finally:
        writer.release()
    return path.read_bytes()


def session(client, name="Fixture session", scene_group="fixture-scene"):
    response = client.post("/api/sessions", json={"name": name, "scene_group": scene_group})
    assert response.status_code == 201, response.text
    return response.json()


def upload(client, session_id, content, filename="fixture.png"):
    response = client.post(
        f"/api/sessions/{session_id}/assets",
        files={"file": (filename, content, "application/octet-stream")},
    )
    assert response.status_code == 201, response.text
    return response.json()


def wait_for_job(client, job_id):
    deadline = time.monotonic() + 15
    current = None
    while time.monotonic() < deadline:
        response = client.get("/api/jobs")
        assert response.status_code == 200, response.text
        current = next(job for job in response.json() if job["id"] == job_id)
        if current["status"] not in {"queued", "running"}:
            return current
        time.sleep(0.05)
    pytest.fail(f"Worker did not finish within 15 seconds: {current}")


def test_session_names_are_normalized_and_group_is_required(client):
    created = session(client, name="  New flight  ", scene_group="  field-west  ")
    assert created["name"] == "New flight"
    assert created["scene_group"] == "field-west"
    assert client.get(f"/api/sessions/{created['id']}").json() == created
    for payload in (
        {"name": " ", "scene_group": "field"},
        {"name": "Flight", "scene_group": "\t"},
        {"name": "Flight"},
        {"name": "Flight", "scene_group": "field", "unexpected": True},
    ):
        response = client.post("/api/sessions", json=payload)
        assert response.status_code == 422, response.text
    assert client.get("/api/sessions").json() == [created]


def test_selection_original_and_frame_provenance_survive_restart(tmp_path, image_bytes):
    data_dir = tmp_path / "workspace"
    with TestClient(create_app(data_dir, run_jobs=False), base_url=BASE_URL) as client:
        flight = session(client)
        asset = upload(client, flight["id"], image_bytes, "../../private/original.png")
        assert asset["filename"] == "original.png"
        assert "path" not in asset
        (frame,) = client.get(f"/api/sessions/{flight['id']}/frames").json()
        assert frame["asset_id"] == asset["id"]
        assert frame["session_id"] == flight["id"]
        assert frame["timestamp_seconds"] is None
        assert (frame["width"], frame["height"]) == (32, 24)
        assert "path" not in frame
        assert frame["selected"] is False
        response = client.patch(f"/api/frames/{frame['id']}", json={"selected": True})
        assert response.status_code == 200
        assert response.json()["selected"] is True
        assert (
            client.patch(f"/api/frames/{frame['id']}", json={"selected": "false"}).status_code
            == 422
        )

    with TestClient(create_app(data_dir, run_jobs=False), base_url=BASE_URL) as reopened:
        (persisted,) = reopened.get(f"/api/sessions/{flight['id']}/frames").json()
        assert persisted["id"] == frame["id"]
        assert persisted["selected"] is True
        assert persisted["sha256"] == frame["sha256"]
        original = reopened.get(f"/api/assets/{asset['id']}/media")
        assert original.status_code == 200
        assert original.content == image_bytes
        normalized = reopened.get(f"/api/frames/{frame['id']}/image")
        assert normalized.status_code == 200
        assert normalized.headers["content-type"] == "image/png"
        with Image.open(io.BytesIO(normalized.content)) as image:
            assert image.size == (32, 24)


def test_duplicate_upload_is_idempotent_and_warns_across_sessions(client, image_bytes):
    first = session(client, name="Flight A")
    second = session(client, name="Flight B")
    original = upload(client, first["id"], image_bytes)
    repeated = upload(client, first["id"], image_bytes, "renamed.png")
    assert repeated["id"] == original["id"]
    assert len(client.get(f"/api/sessions/{first['id']}/frames").json()) == 1
    other = upload(client, second["id"], image_bytes)
    assert other["id"] != original["id"]
    for flight, asset in ((first, original), (second, other)):
        (frame,) = client.get(f"/api/sessions/{flight['id']}/frames").json()
        assert frame["asset_id"] == asset["id"]
        assert frame["session_id"] == flight["id"]
        assert frame["duplicate_count"] == 1


def test_invalid_uploads_and_missing_resources_do_not_leave_data(client):
    flight = session(client)
    for content in (b"", b"this is not a video"):
        response = client.post(
            f"/api/sessions/{flight['id']}/assets",
            files={"file": ("pretend.mp4", content, "video/mp4")},
        )
        assert response.status_code == 422, response.text
    assert client.get(f"/api/sessions/{flight['id']}/assets").json() == []
    assert client.get(f"/api/sessions/{flight['id']}/frames").json() == []
    assert list((client.app.state.store.root / "uploads").iterdir()) == []
    assert client.get("/api/sessions/missing/frames").status_code == 404
    assert client.post("/api/assets/missing/extract", json={}).status_code == 404
    assert client.post("/api/jobs/missing/cancel").status_code == 404
    assert client.get("/api/frames/missing/image").status_code == 404


def test_artifact_paths_cannot_escape_workspace_even_through_symlinks(tmp_path, client):
    private = tmp_path / "private.txt"
    private.write_text("private fixture outside workspace", encoding="utf-8")
    store = client.app.state.store
    (store.root / "linked.txt").symlink_to(private)
    for path in ("../private.txt", str(private), "linked.txt"):
        with pytest.raises(ValueError, match="outside"):
            store.artifact_path(path)
    assert client.get("/private.txt").status_code == 404
    assert client.get("/iris.sqlite3").status_code == 404
    assert client.get("/static/%2e%2e/%2e%2e/private.txt").status_code == 404


def test_queue_rejects_overlapping_extraction_and_allows_retry_after_cancel(client, video_bytes):
    flight = session(client)
    asset = upload(client, flight["id"], video_bytes, "generated.avi")
    endpoint = f"/api/assets/{asset['id']}/extract"
    first = client.post(endpoint, json={"max_frames": 3})
    assert first.status_code == 202
    assert first.json()["status"] == "queued"
    assert client.post(endpoint, json={"max_frames": 2}).status_code == 409
    cancelled = client.post(f"/api/jobs/{first.json()['id']}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    assert cancelled.json()["cancel_requested"] is True
    assert cancelled.json()["finished_at"]
    assert client.get(f"/api/sessions/{flight['id']}/frames").json() == []
    retry = client.post(endpoint, json={"max_frames": 2})
    assert retry.status_code == 202
    assert retry.json()["id"] != first.json()["id"]


@pytest.mark.parametrize(
    "params",
    [
        {"interval_seconds": 0},
        {"start_seconds": -1},
        {"start_seconds": 2},
        {"start_seconds": 1, "end_seconds": 0.5},
        {"max_frames": 501},
        {"max_frames": True},
        {"interval_seconds": True},
        {"dedup_hamming": False},
        {"dedup_hamming": 17},
    ],
)
def test_invalid_extraction_requests_never_enter_queue(client, video_bytes, params):
    flight = session(client)
    asset = upload(client, flight["id"], video_bytes, "generated.avi")
    response = client.post(f"/api/assets/{asset['id']}/extract", json=params)
    assert response.status_code == 422, response.text
    assert client.get("/api/jobs").json() == []


def test_still_image_cannot_be_submitted_for_video_extraction(client, image_bytes):
    flight = session(client)
    asset = upload(client, flight["id"], image_bytes)
    assert client.post(f"/api/assets/{asset['id']}/extract", json={}).status_code == 422
    assert client.get("/api/jobs").json() == []


def test_real_subprocess_extracts_frames_and_persists_job_provenance(tmp_path, video_bytes):
    data_dir = tmp_path / "workspace"
    params = {
        "start_seconds": 0.5,
        "end_seconds": 1.8,
        "interval_seconds": 0.5,
        "max_frames": 2,
        "dedup_hamming": None,
    }
    with TestClient(create_app(data_dir), base_url=BASE_URL) as client:
        flight = session(client)
        asset = upload(client, flight["id"], video_bytes, "generated.avi")
        queued = client.post(f"/api/assets/{asset['id']}/extract", json=params)
        assert queued.status_code == 202
        completed = wait_for_job(client, queued.json()["id"])
        assert completed["status"] == "succeeded", completed
        assert completed["params"] == {"asset_id": asset["id"], "config": params}
        assert completed["result"]["created"] == completed["result"]["sampled"] == 2
        assert completed["progress"] == 1
        assert completed["started_at"] and completed["finished_at"]
        frames = client.get(f"/api/sessions/{flight['id']}/frames").json()
        assert sorted(frame["timestamp_seconds"] for frame in frames) == [0.5, 1.0]
        assert {frame["id"] for frame in frames} == set(completed["result"]["frame_ids"])
        assert all(frame["asset_id"] == asset["id"] for frame in frames)
        assert all(frame["extraction"] == {**params, "job_id": completed["id"]} for frame in frames)
        log = client.get(f"/api/jobs/{completed['id']}/log")
        assert log.status_code == 200 and log.text.strip()

    with TestClient(create_app(data_dir), base_url=BASE_URL) as reopened:
        assert reopened.get("/api/jobs").json() == [completed]
        assert reopened.get(f"/api/sessions/{flight['id']}/frames").json() == frames
        assert reopened.get(f"/api/jobs/{completed['id']}/log").text == log.text


def test_missing_source_fails_real_job_with_a_saved_error_and_log(tmp_path, video_bytes):
    data_dir = tmp_path / "workspace"
    with TestClient(create_app(data_dir), base_url=BASE_URL) as client:
        flight = session(client)
        asset = upload(client, flight["id"], video_bytes, "generated.avi")
        store = client.app.state.store
        source = store.get("assets", asset["id"])
        store.artifact_path(source["path"]).unlink()
        queued = client.post(f"/api/assets/{asset['id']}/extract", json={})
        assert queued.status_code == 202
        failed = wait_for_job(client, queued.json()["id"])
        assert failed["status"] == "failed", failed
        assert failed["error"] and failed["finished_at"]
        assert client.get(f"/api/jobs/{failed['id']}/log").status_code == 200
        assert client.get(f"/api/sessions/{flight['id']}/frames").json() == []
    assert Store(data_dir).get("jobs", failed["id"])["error"] == failed["error"]


def test_restart_marks_unfinished_queue_as_interrupted(tmp_path, video_bytes):
    data_dir = tmp_path / "workspace"
    with TestClient(create_app(data_dir, run_jobs=False), base_url=BASE_URL) as client:
        flight = session(client)
        asset = upload(client, flight["id"], video_bytes, "generated.avi")
        queued = client.post(f"/api/assets/{asset['id']}/extract", json={}).json()
    with TestClient(create_app(data_dir), base_url=BASE_URL) as reopened:
        (interrupted,) = reopened.get("/api/jobs").json()
        assert interrupted["id"] == queued["id"]
        assert interrupted["params"] == queued["params"]
        assert interrupted["status"] == "interrupted"
        assert interrupted["finished_at"]
        assert reopened.get(f"/api/sessions/{flight['id']}/frames").json() == []


def test_second_server_cannot_interrupt_active_workspace_and_lock_is_released(
    tmp_path, video_bytes, monkeypatch
):
    data_dir = tmp_path / "workspace"
    app = create_app(data_dir)
    entered_worker = threading.Event()
    release_worker = threading.Event()
    execute = app.state.jobs._execute

    def gated_execute(job):
        # Hold the scheduling boundary so the second startup is guaranteed to
        # overlap an active job, then execute the normal real subprocess.
        entered_worker.set()
        assert release_worker.wait(10), "Test did not release the worker"
        execute(job)

    monkeypatch.setattr(app.state.jobs, "_execute", gated_execute)
    with TestClient(app, base_url=BASE_URL) as first:
        flight = session(first)
        asset = upload(first, flight["id"], video_bytes, "generated.avi")
        queued = first.post(f"/api/assets/{asset['id']}/extract", json={}).json()
        try:
            assert entered_worker.wait(5), "Worker did not claim the fixture job"
            (before,) = first.get("/api/jobs").json()
            assert before["status"] == "running"
            with pytest.raises(RuntimeError, match="already open"):
                with TestClient(create_app(data_dir), base_url=BASE_URL):
                    pytest.fail("A second server acquired the active workspace")
            assert first.get("/api/jobs").json() == [before]
        finally:
            release_worker.set()
        completed = wait_for_job(first, queued["id"])
        assert completed["status"] == "succeeded", completed

    with TestClient(create_app(data_dir), base_url=BASE_URL) as reopened:
        assert reopened.get("/api/jobs").json() == [completed]
        assert reopened.get(f"/api/sessions/{flight['id']}").status_code == 200


def test_running_extraction_can_be_cancelled_without_losing_completed_frames(tmp_path):
    path = tmp_path / "cancellation.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 20, (160, 120))
    assert writer.isOpened(), "Synthetic video tests require the local MJPEG encoder"
    # A bounded, few-megabyte fixture with distinct frames leaves time to cancel
    # after real processing begins without any artificially slowed worker.
    rng = np.random.default_rng(2026)
    try:
        for _ in range(240):
            writer.write(rng.integers(0, 256, (120, 160, 3), dtype=np.uint8))
    finally:
        writer.release()

    data_dir = tmp_path / "workspace"
    with TestClient(create_app(data_dir), base_url=BASE_URL) as client:
        flight = session(client)
        asset = upload(client, flight["id"], path.read_bytes(), path.name)
        queued = client.post(
            f"/api/assets/{asset['id']}/extract",
            json={"interval_seconds": 0.05, "max_frames": 240},
        ).json()
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            (current,) = client.get("/api/jobs").json()
            if current["status"] == "running" and current["progress"] > 0:
                break
            assert current["status"] in {"queued", "running"}, current
            time.sleep(0.005)
        else:
            pytest.fail("Worker did not produce progress before the cancellation deadline")
        before = client.get(f"/api/sessions/{flight['id']}/frames").json()
        assert before
        response = client.post(f"/api/jobs/{queued['id']}/cancel")
        assert response.status_code == 200
        assert response.json()["cancel_requested"] is True
        cancelled = wait_for_job(client, queued["id"])
        assert cancelled["status"] == "cancelled", cancelled
        assert cancelled["error"] is None
        assert cancelled["progress"] < 1
        frames = client.get(f"/api/sessions/{flight['id']}/frames").json()
        assert {frame["id"] for frame in before} <= {frame["id"] for frame in frames}
        assert 0 < len(frames) < 240
        assert cancelled["result"]["created"] == len(frames)

    with TestClient(create_app(data_dir), base_url=BASE_URL) as reopened:
        assert reopened.get("/api/jobs").json() == [cancelled]
        assert reopened.get(f"/api/sessions/{flight['id']}/frames").json() == frames
        for frame in frames:
            assert reopened.get(f"/api/frames/{frame['id']}/image").status_code == 200


@pytest.mark.parametrize(
    "headers",
    [
        {"Origin": "https://outside.example"},
        {"Origin": "http://127.0.0.1:9999"},
        {"Origin": "null"},
        {"Origin": "http://["},
        {"Sec-Fetch-Site": "cross-site"},
    ],
)
def test_browser_cross_origin_mutations_are_rejected(client, headers):
    response = client.post(
        "/api/sessions", json={"name": "Flight", "scene_group": "field"}, headers=headers
    )
    assert response.status_code == 403, response.text
    assert client.get("/api/sessions").json() == []


def test_same_origin_mutation_succeeds_but_unknown_host_is_rejected(client):
    response = client.post(
        "/api/sessions",
        json={"name": "Flight", "scene_group": "field"},
        headers={"Origin": BASE_URL, "Sec-Fetch-Site": "same-origin"},
    )
    assert response.status_code == 201
    assert client.get("/api/sessions", headers={"Host": "outside.example"}).status_code == 400
    response = client.get("/api/system")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
    assert response.json()["capabilities"]["inference"] is False
