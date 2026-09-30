"""HTTP and subprocess integration with synthetic media and a loopback VLM fixture."""

import base64
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from iris import remote_provider
from iris.app import create_app
from iris.assistance_provider import DEFAULT_MODEL

BASE = "http://127.0.0.1"


@pytest.fixture
def model_server(monkeypatch):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/api/version":
                self.respond({"version": "fixture-1"})
            else:
                self.respond({"models": [{"name": DEFAULT_MODEL, "digest": "sha256:" + "a" * 64}]})

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/api/show":
                self.respond({"capabilities": ["vision"]})
            else:
                assert self.path == "/api/chat"
                calls.append(payload)
                self.respond(
                    {
                        "model": DEFAULT_MODEL,
                        "done": True,
                        "done_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "passages": [
                                        {
                                            "start_sample_id": "s2",
                                            "end_sample_id": "s2",
                                            "reason": "Synthetic response fixture only.",
                                            "uncertainty": "high",
                                        }
                                    ],
                                    "summary": "Protocol fixture; no real model inference.",
                                }
                            ),
                        },
                    }
                )

        def respond(self, value):
            content = json.dumps(value).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("IRIS_OLLAMA_URL", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("IRIS_OLLAMA_MODEL", DEFAULT_MODEL)
    try:
        yield calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "sample.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 4, (96, 64))
    assert writer.isOpened()
    try:
        for index in range(24):
            writer.write(np.full((64, 96, 3), (index * 10, 40, 100), dtype=np.uint8))
    finally:
        writer.release()
    return path.read_bytes()


def imported(client, video):
    session = client.post(
        "/api/sessions", json={"name": "Video review fixture", "scene_group": "synthetic"}
    ).json()
    response = client.post(
        f"/api/sessions/{session['id']}/assets", files={"file": ("sample.avi", video)}
    )
    assert response.status_code == 201, response.text
    return session, response.json()


def preview(client, asset, **fields):
    response = client.post(
        f"/api/assets/{asset['id']}/video-reviews/preview",
        json={"sample_count": 3, **fields},
    )
    assert response.status_code == 201, response.text
    return response.json()


def wait_job(client, job_id):
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        current = next(job for job in client.get("/api/jobs").json() if job["id"] == job_id)
        if current["status"] not in {"queued", "running"}:
            return current
        time.sleep(0.05)
    pytest.fail(f"Worker did not complete: {current}")


def test_real_worker_pipeline_requires_preview_then_human_extraction(tmp_path, video, model_server):
    workspace = tmp_path / "workspace"
    with TestClient(create_app(workspace), base_url=BASE) as client:
        session, asset = imported(client, video)
        record = preview(client, asset)
        assert record["status"] == "preview" and record["job"] is None
        assert [item["id"] for item in record["images"]] == ["s1", "s2", "s3"]
        images = []
        for image in record["images"]:
            assert "path" not in image
            response = client.get(image["url"])
            assert response.headers["content-type"] == "image/jpeg"
            assert response.headers["cache-control"] == "no-store"
            with Image.open(io.BytesIO(response.content)) as decoded:
                assert max(decoded.size) <= 512
            images.append(response.content)
        assert model_server == []
        assert client.get("/api/jobs").json() == []
        frames_url = f"/api/sessions/{session['id']}/frames"
        assert client.get(frames_url).json() == []
        endpoint = f"/api/video-reviews/{record['id']}"
        response = client.post(endpoint + "/run", json={})
        assert response.status_code == 202, response.text
        completed = wait_job(client, response.json()["id"])
        assert completed["status"] == "succeeded", completed
        review = client.get(endpoint).json()
        assert review["status"] == "succeeded"
        assert len(review["result"]["passages"]) == 1
        passage = review["result"]["passages"][0]
        assert passage["start_frame_index"] == record["images"][1]["frame_index"]
        assert passage["start_seconds"] == record["images"][1]["timestamp_seconds"]
        assert passage["end_seconds"] > passage["start_seconds"]
        assert len(model_server) == 1
        assert [
            base64.b64decode(value) for value in model_server[0]["messages"][1]["images"]
        ] == images
        assert client.get(frames_url).json() == []
        assert client.app.state.store.list("annotation_revisions") == []
        assert client.post(endpoint + "/run", json={}).status_code in {409, 422}
        assert len(client.get("/api/jobs").json()) == 1
        choice = {
            "passage_ids": [passage["id"]],
            "frames_per_passage": 4,
            "context_seconds": 0.5,
            "coverage_frames": 3,
        }
        planned = client.post(endpoint + "/extract/preview", json=choice)
        assert planned.status_code == 200, planned.text
        plan = planned.json()
        assert plan["positions"][0]["frame_index"] == 0
        assert plan["positions"][-1]["frame_index"] == 23
        assert len(client.get("/api/jobs").json()) == 1
        assert (
            client.post(endpoint + "/extract", json={**choice, "passage_ids": []}).status_code
            == 422
        )
        queued = client.post(endpoint + "/extract", json=choice)
        assert queued.status_code == 202, queued.text
        extracted = wait_job(client, queued.json()["id"])
        assert extracted["status"] == "succeeded", extracted
        frames = client.get(frames_url).json()
        assert sorted(frame["frame_index"] for frame in frames) == [
            item["frame_index"] for item in plan["positions"]
        ]
        assert all(not frame["selected"] for frame in frames)
        assert all(frame["extraction"]["video_review_id"] == record["id"] for frame in frames)
        assert len(model_server) == 1
    with TestClient(create_app(workspace), base_url=BASE) as client:
        assert client.get(endpoint).json() == review
        assert client.get(frames_url).json() == frames
        assert (
            client.get(f"/api/assets/{asset['id']}/video-reviews").json()[0]["id"] == record["id"]
        )
        assert len(model_server) == 1


@pytest.mark.parametrize(
    "fields",
    [
        {"provider": "arbitrary"},
        {"model": True},
        {"sample_count": True},
        {"sample_count": 1},
        {"sample_count": 13},
        {"start_seconds": -1},
        {"start_seconds": 4, "end_seconds": 3},
        {"instructions": "x" * 2001},
        {"allow_external": True},
    ],
)
def test_invalid_previews_never_decode_or_queue(tmp_path, video, fields):
    with TestClient(create_app(tmp_path / "workspace", run_jobs=False), base_url=BASE) as client:
        _, asset = imported(client, video)
        response = client.post(f"/api/assets/{asset['id']}/video-reviews/preview", json=fields)
        assert response.status_code == 422, response.text
        assert client.get("/api/jobs").json() == []
        assert client.get(f"/api/assets/{asset['id']}/video-reviews").json() == []


def test_hosted_preview_needs_exact_consent_and_budget_without_remote_calls(
    tmp_path, video, monkeypatch
):
    monkeypatch.setenv(remote_provider.KEY_ENV, "fixture-only-not-a-real-key")
    monkeypatch.setenv(
        remote_provider.ENDPOINT_ENV,
        "https://fixture.eu-central-1.maas.aliyuncs.com/compatible-mode/v1",
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("HTTP preview and queueing must never call hosted inference")

    monkeypatch.setattr(remote_provider, "_request", forbidden)
    with TestClient(create_app(tmp_path / "workspace", run_jobs=False), base_url=BASE) as client:
        _, asset = imported(client, video)
        record = preview(client, asset, provider="alibaba", model=remote_provider.DEFAULT_MODEL)
        endpoint = f"/api/video-reviews/{record['id']}/run"
        budget = record["config"]["estimated_cost"]["upper_bound_usd"]
        for approval in ({}, {"allow_external": True}, {"allow_external": True, "max_cost_usd": 0}):
            response = client.post(endpoint, json=approval)
            assert response.status_code == 422, response.text
            assert client.get("/api/jobs").json() == []
        approved = client.post(endpoint, json={"allow_external": True, "max_cost_usd": budget})
        assert approved.status_code == 202, approved.text
        assert approved.json()["kind"] == "video_review"
        assert client.post(
            endpoint, json={"allow_external": True, "max_cost_usd": budget}
        ).status_code in {409, 422}


def test_preview_images_cannot_be_changed_before_model_request(tmp_path, video, model_server):
    with TestClient(create_app(tmp_path / "workspace", run_jobs=False), base_url=BASE) as client:
        _, asset = imported(client, video)
        record = preview(client, asset)
        store = client.app.state.store
        saved = store.get("video_reviews", record["id"])
        store.artifact_path(saved["images"][0]["path"]).write_bytes(b"modified")
        assert client.get(record["images"][0]["url"]).status_code == 422
        response = client.post(f"/api/video-reviews/{record['id']}/run", json={})
        assert response.status_code == 422, response.text
        assert client.get("/api/jobs").json() == []
        assert model_server == []


def test_preview_expiry_and_missing_records_are_explicit(tmp_path, video, model_server):
    with TestClient(create_app(tmp_path / "workspace", run_jobs=False), base_url=BASE) as client:
        _, asset = imported(client, video)
        record = preview(client, asset)
        client.app.state.store.update(
            "video_reviews", record["id"], {"expires_at": "2000-01-01T00:00:00+00:00"}
        )
        endpoint = f"/api/video-reviews/{record['id']}"
        assert client.get(endpoint).json()["status"] == "expired"
        assert client.post(endpoint + "/run", json={}).status_code in {409, 422}
        assert client.get(endpoint + "/images/999").status_code == 404
        assert client.get("/api/video-reviews/missing").status_code == 404
        assert client.post("/api/video-reviews/missing/run", json={}).status_code == 404
        assert client.post("/api/assets/missing/video-reviews/preview", json={}).status_code == 404
        assert client.get("/api/jobs").json() == []
        assert model_server == []
