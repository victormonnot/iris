"""Remote-consent integration using synthetic images and a non-network reviewer."""

import io
import socket
import threading
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from iris.annotations import get_annotation, save_annotation
from iris.app import create_app
from iris.assistance import run_assistance

API_ENDPOINT = "https://123456.eu-central-1.maas.aliyuncs.com/compatible-mode/v1"
API_KEY = "fixture-key-never-valid"
MODEL = "qwen3-vl-32b-instruct"
BOX = {"id": "fixture-box", "label": "person", "box": [2, 3, 30, 35]}


@pytest.fixture
def remote_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("IRIS_DASHSCOPE_API_KEY", API_KEY)
    monkeypatch.setenv("IRIS_DASHSCOPE_BASE_URL", API_ENDPOINT)
    # An invalid local endpoint prevents a status probe from contacting a real daemon.
    monkeypatch.setenv("IRIS_OLLAMA_URL", "http://203.0.113.1")

    def no_network(*args, **kwargs):
        pytest.fail("Preview and queue operations must not open a network connection")

    monkeypatch.setattr(socket, "create_connection", no_network)
    app = create_app(tmp_path / "workspace", run_jobs=False)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        session = client.post(
            "/api/sessions", json={"name": "API fixture", "scene_group": "synthetic"}
        ).json()
        image = io.BytesIO()
        Image.new("RGB", (80, 60), (25, 35, 45)).save(image, format="PNG")
        imported = client.post(
            f"/api/sessions/{session['id']}/assets",
            files={"file": ("synthetic.png", image.getvalue(), "image/png")},
        )
        assert imported.status_code == 201, imported.text
        (frame,) = client.get(f"/api/sessions/{session['id']}/frames").json()
        save_annotation(
            app.state.store, frame["id"], expected_revision=0, boxes=[BOX], decisions={}
        )
        yield client, app.state.store, app.state.jobs, frame


def payload(**changes):
    return {
        "expected_revision": 1,
        "prediction_id": None,
        "threshold": 0.5,
        "instructions": "Synthetic test only",
        "provider": "alibaba",
        "model": MODEL,
        **changes,
    }


def preview(workspace, **changes):
    client, _, _, frame = workspace
    response = client.post(f"/api/frames/{frame['id']}/assist/preview", json=payload(**changes))
    assert response.status_code == 201, response.text
    return response.json()


def consent(quote, **changes):
    return payload(
        preview_id=quote["id"],
        allow_external=True,
        max_cost_usd=quote["cost"]["upper_bound_usd"],
        **changes,
    )


def queue(workspace, request):
    client, _, _, frame = workspace
    return client.post(f"/api/frames/{frame['id']}/assist", json=request)


def test_catalog_and_preview_are_local_and_show_exact_transmission_images(remote_workspace):
    client, store, _, frame = remote_workspace
    response = client.get("/api/annotation-providers")
    assert response.status_code == 200, response.text
    assert API_KEY not in response.text
    assert MODEL in response.text and "qwen3-vl-235b-a22b-instruct" in response.text
    quote = preview(remote_workspace)
    assert quote["frame_id"] == frame["id"]
    assert quote["provider"] == "alibaba" and quote["model"] == MODEL
    assert quote["endpoint"] == API_ENDPOINT
    assert quote["candidate_count"] == 1
    assert quote["cost"]["upper_bound_usd"] > 0
    assert len(quote["images"]) == 2
    sizes = []
    for picture in quote["images"]:
        assert picture["url"].startswith("/api/")
        image_response = client.get(picture["url"])
        assert image_response.status_code == 200, image_response.text
        with Image.open(io.BytesIO(image_response.content)) as image:
            sizes.append(image.size)
    assert sizes == [(80, 60), (28, 32)]
    assert store.get("assistance_previews", quote["id"])["job_id"] is None
    assert store.list("jobs") == []
    assert store.list("assistance_records") == []
    assert API_KEY.encode() not in store.db_path.read_bytes()


@pytest.mark.parametrize("missing", ["preview_id", "allow_external", "max_cost_usd"])
def test_remote_queue_requires_preview_consent_and_budget(remote_workspace, missing):
    quote = preview(remote_workspace)
    request = consent(quote)
    del request[missing]
    response = queue(remote_workspace, request)
    assert response.status_code in {409, 422}, response.text
    _, store, _, _ = remote_workspace
    assert store.list("jobs") == []
    assert store.get("assistance_previews", quote["id"])["job_id"] is None


@pytest.mark.parametrize(
    "changes",
    [
        {"allow_external": False},
        {"max_cost_usd": 0},
        {"max_cost_usd": -1},
        {"preview_id": "unknown-preview"},
        {"threshold": 0.7},
        {"instructions": "Different request"},
        {"model": "qwen3-vl-235b-a22b-instruct"},
    ],
)
def test_confirmation_rejects_mismatched_or_insufficient_consent(remote_workspace, changes):
    quote = preview(remote_workspace)
    request = consent(quote)
    request.update(changes)
    response = queue(remote_workspace, request)
    assert response.status_code in {404, 409, 422}, response.text
    _, store, _, _ = remote_workspace
    assert store.list("jobs") == []
    assert store.get("assistance_previews", quote["id"])["job_id"] is None


def test_expired_preview_cannot_be_queued(remote_workspace):
    _, store, _, _ = remote_workspace
    quote = preview(remote_workspace)
    store.update("assistance_previews", quote["id"], {"expires_at": "2000-01-01T00:00:00+00:00"})
    response = queue(remote_workspace, consent(quote))
    assert response.status_code in {409, 422}, response.text
    assert store.list("jobs") == []


def test_revision_change_requires_another_preview(remote_workspace):
    _, store, _, frame = remote_workspace
    quote = preview(remote_workspace)
    save_annotation(
        store, frame["id"], expected_revision=1, boxes=[{**BOX, "label": "car"}], decisions={}
    )
    assert queue(remote_workspace, consent(quote)).status_code == 409
    refreshed_request = consent(quote, expected_revision=2)
    response = queue(remote_workspace, refreshed_request)
    assert response.status_code in {409, 422}, response.text
    assert store.list("jobs") == []


def test_provider_endpoint_change_requires_another_preview(remote_workspace, monkeypatch):
    _, store, _, _ = remote_workspace
    quote = preview(remote_workspace)
    monkeypatch.setenv(
        "IRIS_DASHSCOPE_BASE_URL",
        "https://654321.eu-central-1.maas.aliyuncs.com/compatible-mode/v1",
    )
    response = queue(remote_workspace, consent(quote))
    assert response.status_code in {409, 422}, response.text
    assert store.list("jobs") == []


def test_pricing_change_requires_another_preview(remote_workspace, monkeypatch):
    from iris import remote_provider

    _, store, _, _ = remote_workspace
    quote = preview(remote_workspace)
    monkeypatch.setitem(remote_provider.MODELS[MODEL], "input_usd_per_million", "0.32")
    response = queue(remote_workspace, consent(quote))
    assert response.status_code in {409, 422}, response.text
    assert store.list("jobs") == []


def test_changed_source_pixels_block_consent(remote_workspace):
    _, store, _, frame = remote_workspace
    quote = preview(remote_workspace)
    record = store.get("frames", frame["id"])
    Image.new("RGB", (80, 60), "red").save(store.artifact_path(record["path"]))
    response = queue(remote_workspace, consent(quote))
    assert response.status_code in {409, 422}, response.text
    assert store.list("jobs") == []


@pytest.mark.parametrize("after_queue", [False, True])
def test_changed_preview_pixels_cannot_be_sent(remote_workspace, after_queue):
    _, store, _, _ = remote_workspace
    quote = preview(remote_workspace)
    if after_queue:
        response = queue(remote_workspace, consent(quote))
        assert response.status_code == 202, response.text
        record_id = response.json()["params"]["assistance_id"]
    stored = store.get("assistance_previews", quote["id"])
    path = store.artifact_path(stored["images"][0]["path"])
    original = path.read_bytes()
    path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    if not after_queue:
        response = queue(remote_workspace, consent(quote))
        assert response.status_code in {409, 422}, response.text
        assert store.list("jobs") == []
        return

    def never_construct(**kwargs):
        pytest.fail("Changed preview pixels must be rejected before constructing a reviewer")

    with pytest.raises(ValueError, match="[Pp]review.*changed"):
        run_assistance(
            store, record_id, lambda *_: None, lambda: False, reviewer_factory=never_construct
        )
    assert store.list("annotation_suggestions") == []


def test_confirmed_preview_is_consumed_once_even_if_job_is_cancelled(remote_workspace):
    _, store, jobs, _ = remote_workspace
    quote = preview(remote_workspace)
    response = queue(remote_workspace, consent(quote))
    assert response.status_code == 202, response.text
    job = response.json()
    assert store.get("assistance_previews", quote["id"])["job_id"] == job["id"]
    jobs.cancel(job["id"])
    repeated = queue(remote_workspace, consent(quote))
    assert repeated.status_code in {409, 422}, repeated.text
    assert len(store.list("jobs")) == 1


def test_concurrent_confirmations_queue_only_one_request(remote_workspace):
    _, store, _, _ = remote_workspace
    quote = preview(remote_workspace)
    ready = threading.Barrier(2)

    def submit():
        ready.wait(timeout=5)
        return queue(remote_workspace, consent(quote))

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(submit)
        second = executor.submit(submit)
        responses = [first.result(timeout=10), second.result(timeout=10)]
    assert sorted(response.status_code for response in responses) in ([202, 409], [202, 422])
    (job,) = store.list("jobs")
    assert len(store.list("assistance_records")) == 1
    assert store.get("assistance_previews", quote["id"])["job_id"] == job["id"]


@pytest.mark.parametrize("previous_status", ["queued", "running"])
def test_restart_interrupts_approved_requests_without_replaying_them(
    remote_workspace, monkeypatch, previous_status
):
    _, store, jobs, _ = remote_workspace
    quote = preview(remote_workspace)
    response = queue(remote_workspace, consent(quote))
    assert response.status_code == 202, response.text
    job = response.json()
    store.update("jobs", job["id"], {"status": previous_status})
    dispatched = threading.Event()
    monkeypatch.setattr(jobs, "_execute", lambda job: dispatched.set())
    jobs.start()
    try:
        assert store.get("jobs", job["id"])["status"] == "interrupted"
        assert not dispatched.wait(0.3)
        repeated = queue(remote_workspace, consent(quote))
        assert repeated.status_code in {409, 422}, repeated.text
        assert len(store.list("jobs")) == 1
        assert store.get("assistance_previews", quote["id"])["job_id"] == job["id"]
    finally:
        jobs.close()


def test_worker_receives_exact_preview_bytes_and_never_validates_annotations(remote_workspace):
    client, store, _, frame = remote_workspace
    quote = preview(remote_workspace)
    displayed = [client.get(item["url"]).content for item in quote["images"]]
    original = deepcopy(get_annotation(store, frame["id"]))
    response = queue(remote_workspace, consent(quote))
    assert response.status_code == 202, response.text
    record_id = response.json()["params"]["assistance_id"]

    class PreviewReviewer:
        def __init__(self, config, expected_images):
            assert expected_images == displayed
            assert config["endpoint"] == API_ENDPOINT
            assert config["model"] == MODEL
            self.metadata = {"provider": "alibaba", "model": MODEL, "fixture": True}

        def review(self, image, candidates, instructions=""):
            assert image.size == (80, 60)
            assert instructions == payload()["instructions"]
            return {
                "reviews": [
                    {
                        "candidate_id": candidate["id"],
                        "label": "uncertain",
                        "reason": "Fixture response; no model inference",
                    }
                    for candidate in candidates
                ],
                "scene_notes": "Synthetic fixture only",
                "prompt": "Fixture prompt",
                "metadata": self.metadata,
                "raw_response": {"fixture": True},
            }

    result = run_assistance(
        store, record_id, lambda *_: None, lambda: False, reviewer_factory=PreviewReviewer
    )
    assert result["suggestions_created"] == 1
    after = get_annotation(store, frame["id"])
    assert after["revision"] == original["revision"]
    assert after["boxes"] == original["boxes"]
    assert after["status"] == "draft"
    assert after["suggestions"][0]["state"] == "pending"
    assert API_KEY.encode() not in store.db_path.read_bytes()


def test_cancelled_remote_request_does_not_construct_a_reviewer(remote_workspace):
    _, store, jobs, _ = remote_workspace
    quote = preview(remote_workspace)
    response = queue(remote_workspace, consent(quote))
    assert response.status_code == 202, response.text
    job = response.json()
    jobs.cancel(job["id"])

    def never_construct(**kwargs):
        pytest.fail("A cancelled request must not construct a remote reviewer")

    result = run_assistance(
        store,
        job["params"]["assistance_id"],
        lambda *_: None,
        lambda: True,
        reviewer_factory=never_construct,
    )
    assert result["cancelled"]
    assert store.list("annotation_suggestions") == []


@pytest.mark.parametrize("change", ["missing_consent", "insufficient_budget", "changed_price"])
def test_worker_checks_consent_and_price_again_before_provider_creation(
    remote_workspace, monkeypatch, change
):
    from iris import remote_provider

    _, store, _, _ = remote_workspace
    quote = preview(remote_workspace)
    response = queue(remote_workspace, consent(quote))
    assert response.status_code == 202, response.text
    record_id = response.json()["params"]["assistance_id"]
    record = store.get("assistance_records", record_id)
    if change == "changed_price":
        monkeypatch.setitem(remote_provider.MODELS[MODEL], "input_usd_per_million", "0.32")
    else:
        config = deepcopy(record["config"])
        if change == "missing_consent":
            del config["consent"]
        else:
            config["consent"]["max_cost_usd"] = 0
        store.update("assistance_records", record_id, {"config": config})

    def never_construct(**kwargs):
        pytest.fail("Invalid consent or pricing must be rejected before provider creation")

    with pytest.raises(ValueError, match="approved|pricing|budget"):
        run_assistance(
            store, record_id, lambda *_: None, lambda: False, reviewer_factory=never_construct
        )
    assert store.list("annotation_suggestions") == []


def test_explicit_local_model_selection_needs_no_external_consent(remote_workspace, monkeypatch):
    _, store, _, _ = remote_workspace
    calls = []

    def local_status(config):
        calls.append(config)
        return {
            "provider": "ollama",
            "endpoint": "http://127.0.0.1:11434",
            "model": config["model"],
            "model_digest": "a" * 64,
            "status": "ready",
        }

    monkeypatch.setattr("iris.assistance.provider_status", local_status)
    response = queue(
        remote_workspace, payload(provider="ollama", model="fixture-local-vision:latest")
    )
    assert response.status_code == 202, response.text
    record = store.get("assistance_records", response.json()["params"]["assistance_id"])
    assert len(calls) == 1 and calls[0]["model"] == "fixture-local-vision:latest"
    assert record["config"]["provider"]["model"] == "fixture-local-vision:latest"
    assert "consent" not in record["config"]
    assert store.list("assistance_previews") == []
