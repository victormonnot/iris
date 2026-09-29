"""Synthetic HTTP fixtures only: these tests make no claims about VLM quality."""

import base64
import io
import json
import os
import socket
import threading
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from PIL import Image

from iris import assistance_provider as provider

DIGEST = "a" * 64
CANDIDATES = [{"id": "candidate_1", "label": "person", "box": [1, 2, 41, 62]}]
RESULT = {
    "reviews": [
        {
            "candidate_id": "candidate_1",
            "label": "uncertain",
            "reason": "Synthetic fixture: no real visual interpretation.",
        }
    ],
    "scene_notes": "Simulated response to verify the protocol.",
}


@pytest.fixture
def server():
    state = {
        "requests": [],
        "tags": {"models": [{"name": provider.DEFAULT_MODEL, "digest": DIGEST}]},
        "show": {"capabilities": ["completion", "vision"]},
        "chat": {
            "model": provider.DEFAULT_MODEL,
            "done": True,
            "done_reason": "stop",
            "message": {"role": "assistant", "content": json.dumps(RESULT)},
        },
        "overrides": {},
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.handle_request()

        def do_POST(self):
            self.handle_request()

        def handle_request(self):
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length)) if length else None
            state["requests"].append((self.command, self.path, body))
            response = {
                "/api/version": {"version": "fixture-0.34.4"},
                "/api/tags": state["tags"],
                "/api/show": state["show"],
                "/api/chat": state["chat"],
            }.get(self.path, {"error": "Unsupported fixture endpoint"})
            status, body = state["overrides"].get(self.path, (200, response))
            if callable(body):
                body = body()
            encoded = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            if status == 302:
                self.send_header("Location", "http://203.0.113.10/never-contact")
            self.end_headers()
            self.wfile.write(encoded)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    state["config"] = {
        "endpoint": f"http://127.0.0.1:{httpd.server_port}",
        "model": provider.DEFAULT_MODEL,
    }
    yield state
    httpd.shutdown()
    httpd.server_close()
    thread.join()


def test_ready_probes_only_metadata_and_ignores_proxy_environment(server, monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://203.0.113.10:9999")
    monkeypatch.setenv("ALL_PROXY", "http://203.0.113.10:9999")
    monkeypatch.setenv("IRIS_OLLAMA_URL", server["config"]["endpoint"])
    status = provider.provider_status()
    assert status["status"] == "ready"
    assert status["model_digest"] == DIGEST
    assert status["version"] == "fixture-0.34.4"
    assert [(method, path) for method, path, _ in server["requests"]] == [
        ("GET", "/api/version"),
        ("GET", "/api/tags"),
        ("POST", "/api/show"),
        ("GET", "/api/tags"),
    ]
    assert all(
        body is None or body == {"model": provider.DEFAULT_MODEL}
        for _, _, body in server["requests"]
    )


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://127.0.0.1:11434",
        "http://example.com",
        "http://192.168.1.4:11434",
        "http://127.0.0.1.evil.example",
        "http://2130706433",
        "http://0x7f000001",
        "http://user:password@localhost:11434",
        "http://localhost:11434/api",
        "http://localhost:11434?x=1",
        "http://localhost:11434#fragment",
        "http://localhost:11434?",
        "http://localhost:11434#",
        "http://[::]:11434",
        "http://127.0.0.1:0",
        "http://127.0.0.1:99999",
        " http://localhost:11434",
    ],
)
def test_remote_or_ambiguous_endpoint_rejected_without_request(monkeypatch, endpoint):
    monkeypatch.setattr(provider, "_request", lambda *a, **kw: pytest.fail("Network request"))
    assert provider.provider_status({"endpoint": endpoint})["status"] == "invalid_config"


@pytest.mark.parametrize(
    "model",
    [
        "qwen3-vl:235b-cloud",
        "harmless-CLOUD:latest",
        "registry.example/model:latest",
        "../weights",
        "https://example.com/model",
        "x\nInjected",
        None,
    ],
)
def test_remote_or_ambiguous_model_rejected_without_request(monkeypatch, model):
    monkeypatch.setattr(provider, "_request", lambda *a, **kw: pytest.fail("Network request"))
    assert provider.provider_status({"model": model})["status"] == "invalid_config"


def test_localhost_resolution_is_checked_and_returns_literal(monkeypatch):
    rows = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 11434))]
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **kw: rows)
    assert provider._address("http://localhost:11434") == ("127.0.0.1", 11434)
    rows.append((socket.AF_INET, socket.SOCK_STREAM, 6, "", ("203.0.113.10", 11434)))
    with pytest.raises(ValueError, match="loopback"):
        provider._address("http://localhost:11434")
    assert provider._address("http://[::1]:11434") == ("::1", 11434)


def test_missing_model_does_not_pull_or_generate(server):
    server["tags"]["models"] = []
    assert provider.provider_status(server["config"])["status"] == "missing_model"
    assert len(server["requests"]) == 2


@pytest.mark.parametrize(
    "location,field",
    [
        ("tags", "remote_host"),
        ("tags", "remote_model"),
        ("show", "remote_host"),
        ("show", "remote_model"),
    ],
)
def test_renamed_cloud_alias_rejected(server, location, field):
    row = server["tags"]["models"][0] if location == "tags" else server["show"]
    row[field] = "remote-cloud-alias"
    assert provider.provider_status(server["config"])["status"] == "unsupported_model"
    assert all(path != "/api/chat" for _, path, _ in server["requests"])


def test_model_without_vision_is_unsupported(server):
    server["show"]["capabilities"] = ["completion"]
    assert provider.provider_status(server["config"])["status"] == "unsupported_model"


def test_missing_or_changing_digest_is_unavailable(server):
    server["tags"]["models"][0]["digest"] = "short-prefix"
    assert provider.provider_status(server["config"])["status"] == "unavailable"
    server["tags"]["models"][0]["digest"] = DIGEST
    calls = []

    def tags():
        calls.append(1)
        return {
            "models": [
                {"name": provider.DEFAULT_MODEL, "digest": DIGEST if len(calls) == 1 else "b" * 64}
            ]
        }

    server["overrides"]["/api/tags"] = (200, tags)
    assert provider.provider_status(server["config"])["status"] == "unavailable"


def test_redirect_not_followed_and_bad_or_oversized_json_unavailable(server):
    for status, body in [
        (302, b"redirect"),
        (200, b"[]"),
        (200, b"not JSON"),
        (200, b"x" * (provider.MAX_RESPONSE_BYTES + 1)),
    ]:
        server["requests"].clear()
        server["overrides"]["/api/version"] = (status, body)
        assert provider.provider_status(server["config"])["status"] == "unavailable"
        assert len(server["requests"]) == 1


def test_review_encodes_bounded_images_and_keeps_geometry_out_of_response(server):
    image = Image.new("RGB", (2200, 1300), "navy")
    candidates = deepcopy(CANDIDATES)
    candidates[0]["box"] = [0.1, 2, 1700.1, 1200]
    original = deepcopy(candidates)
    reviewer = provider.OllamaReviewer(server["config"])
    result = reviewer.review(image, candidates, "A sign may contain text.")
    assert result["reviews"] == RESULT["reviews"]
    assert result["metadata"]["model_digest"] == DIGEST
    assert result["raw_response"] == server["chat"]
    assert result["metadata"]["prompt_version"] == provider.PROMPT_VERSION
    assert result["metadata"]["settings"] == provider.SETTINGS
    assert candidates == original
    request = [body for _, path, body in server["requests"] if path == "/api/chat"]
    assert len(request) == 1
    request = request[0]
    assert request["stream"] is False and request["keep_alive"] == 0
    assert request["format"]["additionalProperties"] is False
    assert request["messages"][0] == {"role": "system", "content": provider.SYSTEM_PROMPT}
    message = request["messages"][1]
    assert "candidate_1" in message["content"]
    assert "1700.1" not in message["content"]
    assert len(message["images"]) == 2
    sizes = [Image.open(io.BytesIO(base64.b64decode(value))).size for value in message["images"]]
    assert max(sizes[0]) == 1024 and max(sizes[1]) == 320


@pytest.mark.parametrize(
    "change",
    [
        "unknown_id",
        "duplicate_id",
        "missing_id",
        "invented_geometry",
        "unknown_label",
        "long_reason",
        "padded_reason",
        "empty_reason",
        "long_notes",
        "extra_field",
        "array_response",
        "duplicate_json_key",
        "markdown",
        "truncated",
        "different_model",
        "incomplete",
    ],
)
def test_invalid_model_output_rejected_and_raw_response_preserved(server, change):
    result = deepcopy(RESULT)
    if change == "unknown_id":
        result["reviews"][0]["candidate_id"] = "invented"
    elif change == "duplicate_id":
        result["reviews"].append(deepcopy(result["reviews"][0]))
    elif change == "missing_id":
        result["reviews"] = []
    elif change == "invented_geometry":
        result["reviews"][0]["box"] = [1, 2, 3, 4]
    elif change == "unknown_label":
        result["reviews"][0]["label"] = "truck"
    elif change == "long_reason":
        result["reviews"][0]["reason"] = "a" * 501
    elif change == "padded_reason":
        result["reviews"][0]["reason"] = "a" + " " * 500
    elif change == "empty_reason":
        result["reviews"][0]["reason"] = " "
    elif change == "long_notes":
        result["scene_notes"] = "a" * 2001
    elif change == "extra_field":
        result["validated"] = True
    elif change == "array_response":
        result = []
    server["chat"]["message"]["content"] = json.dumps(result)
    if change == "duplicate_json_key":
        server["chat"]["message"]["content"] = '{"reviews":[],"reviews":[],"scene_notes":""}'
    elif change == "markdown":
        server["chat"]["message"]["content"] = "```json\n" + json.dumps(result) + "\n```"
    elif change == "truncated":
        server["chat"]["done_reason"] = "length"
    elif change == "different_model":
        server["chat"]["model"] = "other:latest"
    elif change == "incomplete":
        server["chat"]["done"] = False
    with pytest.raises(provider.ProviderResponseError) as exc:
        provider.OllamaReviewer(server["config"]).review(Image.new("RGB", (100, 100)), CANDIDATES)
    assert exc.value.raw_response == server["chat"]
    assert exc.value.metadata["model_digest"] == DIGEST
    assert json.loads(exc.value.prompt)[0] == {"role": "system", "content": provider.SYSTEM_PROMPT}


def test_http_failure_preserves_bounded_raw_output(server):
    server["overrides"]["/api/chat"] = (500, {"error": "fixture server failure"})
    with pytest.raises(provider.ProviderResponseError) as exc:
        provider.OllamaReviewer(server["config"]).review(Image.new("RGB", (100, 100)), CANDIDATES)
    assert exc.value.raw_response["http_status"] == 500
    assert "fixture server failure" in exc.value.raw_response["body"]
    assert exc.value.metadata["model_digest"] == DIGEST
    assert exc.value.prompt


def test_changed_model_before_review_sends_no_images(server):
    reviewer = provider.OllamaReviewer(server["config"])
    server["tags"]["models"][0]["digest"] = "b" * 64
    with pytest.raises(provider.ProviderResponseError, match="changed"):
        reviewer.review(Image.new("RGB", (100, 100)), CANDIDATES)
    assert all(path != "/api/chat" for _, path, _ in server["requests"])


def test_changed_model_after_review_rejects_result_but_retains_it(server):
    def chat():
        server["tags"]["models"][0]["digest"] = "b" * 64
        return server["chat"]

    server["overrides"]["/api/chat"] = (200, chat)
    with pytest.raises(provider.ProviderResponseError, match="changed") as exc:
        provider.OllamaReviewer(server["config"]).review(Image.new("RGB", (100, 100)), CANDIDATES)
    assert exc.value.raw_response == server["chat"]


@pytest.mark.parametrize(
    "candidates",
    [
        [],
        CANDIDATES * 9,
        CANDIDATES * 2,
        [{"id": "c", "label": "truck", "box": [1, 2, 3, 4]}],
        [{"id": "c", "label": "car", "box": [1, 2, 3, 1000]}],
        [{"id": "c", "label": "car", "box": [1, 2, float("nan"), 4]}],
        [{"id": "c", "label": "car", "box": [1, 2, True, 4]}],
        [{"id": "c", "label": "car", "box": [3, 2, 1, 4]}],
        [{"id": "injected\ntext", "label": "car", "box": [1, 2, 3, 4]}],
    ],
)
def test_invalid_candidates_rejected_before_image_request(server, candidates):
    reviewer = provider.OllamaReviewer(server["config"])
    server["requests"].clear()
    with pytest.raises(ValueError):
        reviewer.review(Image.new("RGB", (100, 100)), candidates)
    assert server["requests"] == []


def test_unavailable_service_is_not_replaced_with_a_fixture_provider(monkeypatch):
    def unavailable(*args, **kwargs):
        raise provider.ProviderResponseError("Connection refused fixture")

    monkeypatch.setattr(provider, "_request", unavailable)
    status = provider.provider_status()
    assert status["provider"] == "ollama" and status["status"] == "unavailable"
    with pytest.raises(provider.ProviderResponseError):
        provider.OllamaReviewer()


@pytest.mark.skipif(
    os.environ.get("IRIS_TEST_OLLAMA") != "1",
    reason="Set IRIS_TEST_OLLAMA=1 after explicitly provisioning a local Ollama vision model",
)
def test_real_ollama_review_of_synthetic_image():
    """Opt-in protocol smoke check; synthetic input cannot establish annotation accuracy."""
    status = provider.provider_status()
    assert status["status"] == "ready", status["reason"]
    reviewer = provider.OllamaReviewer()
    image = Image.new("RGB", (192, 128), (25, 40, 55))
    image.paste((210, 165, 80), (40, 30, 100, 95))
    candidates = [{"id": "synthetic-1", "label": "person", "box": [40, 30, 100, 95]}]
    original = deepcopy(candidates)
    result = reviewer.review(
        image,
        candidates,
        instructions="Synthetic protocol test. Do not assume a target object exists.",
    )
    assert len(result["reviews"]) == 1
    review = result["reviews"][0]
    assert set(review) == {"candidate_id", "label", "reason"}
    assert review["candidate_id"] == "synthetic-1"
    assert review["label"] in {"person", "car", "none", "uncertain"}
    assert 1 <= len(review["reason"]) <= 500
    assert isinstance(result["scene_notes"], str)
    assert result["metadata"]["model_digest"] == status["model_digest"]
    assert result["metadata"]["version"] == status["version"]
    assert result["metadata"]["model"] == status["model"]
    assert result["metadata"]["prompt_version"] == provider.PROMPT_VERSION
    assert result["metadata"]["local_only"] is True
    assert result["raw_response"]["done"] is True
    assert json.loads(result["raw_response"]["message"]["content"])["reviews"] == result["reviews"]
    assert json.loads(result["prompt"])[0]["role"] == "system"
    assert candidates == original
