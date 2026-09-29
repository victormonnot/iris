"""Offline protocol fixtures; no hosted inference, credentials or image upload."""

import base64
import io
import json
import traceback
from copy import deepcopy

import pytest
from PIL import Image

from iris import remote_provider as provider
from iris.assistance_provider import ProviderResponseError

ENDPOINT = "https://llm-fixture.eu-central-1.maas.aliyuncs.com/compatible-mode/v1"
KEY = "sk-offline-test-only"
CONFIG = {"provider": "alibaba", "endpoint": ENDPOINT, "model": provider.DEFAULT_MODEL}
CANDIDATES = [{"id": "candidate_1", "label": "person", "box": [1.5, 2.1, 41.2, 62.9]}]
REVIEW = {
    "reviews": [
        {"candidate_id": "candidate_1", "label": "uncertain", "reason": "Synthetic fixture."}
    ],
    "scene_notes": "Protocol fixture; no visual-quality claim.",
}


@pytest.fixture
def transport(monkeypatch):
    monkeypatch.setenv(provider.KEY_ENV, KEY)
    monkeypatch.setenv(provider.ENDPOINT_ENV, ENDPOINT)
    state = {
        "connections": [],
        "requests": [],
        "status": 200,
        "response": {
            "id": "chatcmpl-fixture",
            "model": provider.DEFAULT_MODEL,
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(REVIEW)}}],
            "usage": {"prompt_tokens": 2500, "completion_tokens": 200, "total_tokens": 2700},
        },
        "error": None,
        "closed": 0,
        "timeouts": [],
    }

    class Socket:
        def settimeout(self, seconds):
            state["timeouts"].append(seconds)

    class Response:
        def __init__(self):
            self.status = state["status"]
            self.fp = None
            data = state["response"]
            self.body = io.BytesIO(data if isinstance(data, bytes) else json.dumps(data).encode())

        def read1(self, size):
            return self.body.read(size)

    class Connection:
        def __init__(self, host, port, timeout):
            state["connections"].append((host, port, timeout))
            self.sock = Socket()

        def request(self, method, path, *, body, headers):
            state["requests"].append((method, path, json.loads(body), headers))
            if state["error"]:
                raise state["error"]

        def getresponse(self):
            return Response()

        def close(self):
            state["closed"] += 1

    monkeypatch.setattr(provider.http.client, "HTTPSConnection", Connection)
    return state


def run_review(**kwargs):
    return provider.AlibabaReviewer(CONFIG, **kwargs).review(
        Image.new("RGB", (2400, 1200), "blue"), CANDIDATES
    )


def test_catalog_status_and_construction_are_offline(transport):
    catalog = provider.catalog()
    assert len(catalog) == 2
    assert {row["model"] for row in catalog} == set(provider.MODELS)
    assert all(row["status"] == "ready" for row in catalog)
    assert all(row["connection_verified"] is False for row in catalog)
    assert all(row["deployment_scope"] == "Global" for row in catalog)
    reviewer = provider.AlibabaReviewer(CONFIG)
    assert reviewer.metadata["model_identity"].endswith("digest is unavailable.")
    assert transport["connections"] == []
    assert KEY not in json.dumps(catalog)
    assert KEY not in json.dumps(reviewer.metadata)


def test_missing_endpoint_or_key_is_clear_without_network(transport, monkeypatch):
    monkeypatch.delenv(provider.ENDPOINT_ENV)
    assert provider.provider_status()["status"] == "missing_config"
    assert provider.ENDPOINT_ENV in provider.provider_status()["reason"]
    monkeypatch.setenv(provider.ENDPOINT_ENV, ENDPOINT)
    monkeypatch.delenv(provider.KEY_ENV)
    assert provider.provider_status()["status"] == "missing_key"
    assert transport["connections"] == []
    with pytest.raises(ProviderResponseError, match=provider.KEY_ENV):
        provider.AlibabaReviewer(CONFIG)


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://llm-fixture.eu-central-1.maas.aliyuncs.com/compatible-mode/v1",
        "https://evil.example/compatible-mode/v1",
        "https://llm-fixture.eu-central-1.maas.aliyuncs.com.evil.example/compatible-mode/v1",
        "https://llm-fixture.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
        "https://llm-fixture.eu-central-1.maas.aliyuncs.com:443/compatible-mode/v1",
        "https://key@llm-fixture.eu-central-1.maas.aliyuncs.com/compatible-mode/v1",
        ENDPOINT + "?key=secret",
        ENDPOINT + "#fragment",
        ENDPOINT + "/",
        "https://a.b.eu-central-1.maas.aliyuncs.com/compatible-mode/v1",
        "https://127.0.0.1/compatible-mode/v1",
    ],
)
def test_endpoint_is_constrained_to_frankfurt_workspace(transport, endpoint):
    status = provider.provider_status({**CONFIG, "endpoint": endpoint})
    assert status["status"] == "invalid_config"
    assert status["endpoint"] is None
    with pytest.raises(ValueError):
        provider.AlibabaReviewer({**CONFIG, "endpoint": endpoint})
    assert transport["connections"] == []


@pytest.mark.parametrize("extra", [{"api_key": KEY}, {"provider": "other"}, {"model": "other"}])
def test_config_cannot_override_secret_provider_or_model(transport, extra):
    assert provider.provider_status({**CONFIG, **extra})["status"] == "invalid_config"
    with pytest.raises(ValueError):
        provider.AlibabaReviewer({**CONFIG, **extra})
    assert transport["connections"] == []


@pytest.mark.parametrize("key", ["bad\nkey", "bad key", "clé", "k" * 513])
def test_invalid_keys_cannot_inject_headers(transport, monkeypatch, key):
    monkeypatch.setenv(provider.KEY_ENV, key)
    status = provider.provider_status(CONFIG)
    assert status["status"] == "missing_key"
    assert key not in status["reason"]
    assert transport["connections"] == []


def test_price_bound_uses_full_documented_input_and_configured_output():
    small = provider.conservative_estimate({"model": "qwen3-vl-32b-instruct"})
    large = provider.conservative_estimate({"model": "qwen3-vl-235b-a22b-instruct"})
    assert small["upper_bound_usd"] == 0.0212992
    assert large["upper_bound_usd"] == 0.038204416
    assert small["input_token_bound"] == 129024
    assert small["output_token_bound"] == 1024
    assert small["currency"] == "USD"
    assert small["price_checked_at"] == "2026-09-29"
    assert small["pricing_source"].endswith("qwen3-vl-32b-instruct")


def test_single_request_uses_direct_https_json_object_and_metadata_free_jpegs(
    transport, monkeypatch
):
    monkeypatch.setenv("HTTPS_PROXY", "https://proxy.invalid")
    monkeypatch.setenv("ALL_PROXY", "https://proxy.invalid")
    result = run_review()
    assert result["reviews"] == REVIEW["reviews"]
    assert transport["connections"] == [
        ("llm-fixture.eu-central-1.maas.aliyuncs.com", 443, provider.REVIEW_TIMEOUT)
    ]
    assert len(transport["requests"]) == 1
    method, path, payload, headers = transport["requests"][0]
    assert (method, path) == ("POST", "/compatible-mode/v1/chat/completions")
    assert headers["Authorization"] == "Bearer " + KEY
    assert payload["model"] == provider.DEFAULT_MODEL
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["max_tokens"] == 1024
    assert payload["stream"] is False
    content = payload["messages"][1]["content"]
    assert content[0]["type"] == "text"
    assert "Required JSON schema" in content[0]["text"]
    assert len(content) == 3
    for item, expected_size in zip(content[1:], [(1024, 512), (41, 61)], strict=True):
        data = base64.b64decode(item["image_url"]["url"].split(",", 1)[1])
        with Image.open(io.BytesIO(data)) as image:
            assert image.format == "JPEG"
            assert image.size == expected_size
            assert not image.getexif()
    assert "base64" not in result["prompt"]
    assert result["metadata"]["usage_cost_usd"] == 0.000528
    assert result["metadata"]["usage_exceeds_bound"] is False
    assert KEY not in json.dumps(result)
    assert transport["closed"] == 1


def test_exact_approved_image_bytes_are_sent(transport):
    image = Image.new("RGB", (2400, 1200), "blue")
    approved = provider._images(image, CANDIDATES)
    run_review(expected_images=approved)
    payload = transport["requests"][0][2]
    actual = [
        base64.b64decode(part["image_url"]["url"].split(",", 1)[1])
        for part in payload["messages"][1]["content"][1:]
    ]
    assert actual == approved


@pytest.mark.parametrize("approved", [[], [b"invalid JPEG"], [b"x", b"y"]])
def test_changed_preview_never_sends(transport, approved):
    with pytest.raises(ValueError, match="approved preview"):
        run_review(expected_images=approved)
    assert transport["connections"] == []


@pytest.mark.parametrize("status", [301, 302, 307, 401, 429, 500])
def test_failed_http_calls_never_follow_redirect_or_retry(transport, status):
    transport["status"] = status
    transport["response"] = {"error": {"message": f"Fixture echoed {KEY}", "api_key": KEY}}
    with pytest.raises(ProviderResponseError) as error:
        run_review()
    assert str(status) in str(error.value)
    assert KEY not in json.dumps(error.value.raw_response)
    assert "[redacted]" in json.dumps(error.value.raw_response)
    assert len(transport["requests"]) == 1
    assert transport["closed"] == 1


def test_transport_failure_has_no_retry_and_scrubs_secret(transport):
    transport["error"] = OSError("Fixture error " + KEY)
    with pytest.raises(ProviderResponseError) as error:
        run_review()
    assert KEY not in str(error.value)
    assert KEY not in "".join(traceback.format_exception(error.value))
    assert len(transport["requests"]) == 1
    assert transport["closed"] == 1


def test_duplicate_json_key_cannot_leak_credentials_through_exception_chain(transport):
    transport["response"] = (f'{{"{KEY}":1,"{KEY}":2}}').encode()
    with pytest.raises(ProviderResponseError) as error:
        run_review()
    assert KEY not in json.dumps(error.value.raw_response)
    assert KEY not in "".join(traceback.format_exception(error.value))
    assert KEY not in json.dumps(error.value.metadata)


def test_only_an_actual_request_can_mark_connection_verified(transport):
    reviewer = provider.AlibabaReviewer(CONFIG)
    assert reviewer.metadata["connection_verified"] is False
    result = reviewer.review(Image.new("RGB", (100, 100)), CANDIDATES)
    assert result["metadata"]["connection_verified"] is True
    assert provider.provider_status(CONFIG)["connection_verified"] is False


@pytest.mark.parametrize("body", [b"not JSON", b'{"model":"a","model":"b"}', b'{"a":NaN}'])
def test_invalid_raw_json_is_preserved_without_retry(transport, body):
    transport["response"] = body
    with pytest.raises(ProviderResponseError) as error:
        run_review()
    assert error.value.raw_response["body"] == body.decode()
    assert error.value.prompt
    assert len(transport["requests"]) == 1


def test_response_size_is_bounded(transport, monkeypatch):
    monkeypatch.setattr(provider, "MAX_RESPONSE_BYTES", 64)
    transport["response"] = b"x" * 65
    with pytest.raises(ProviderResponseError, match="1 MiB") as error:
        run_review()
    assert len(error.value.raw_response["body"]) == 64
    assert error.value.raw_response["truncated"] is True


@pytest.mark.parametrize(
    "change",
    [
        {"model": "unexpected-model"},
        {"choices": []},
        {"choices": [{"finish_reason": "length", "message": {"content": "{}"}}]},
        {"choices": [{"finish_reason": "stop", "message": {"content": "not JSON"}}]},
        {"choices": [{"finish_reason": "stop", "message": {"content": "{}"}}]},
    ],
)
def test_unusable_generations_keep_raw_response_usage_and_do_not_retry(transport, change):
    transport["response"].update(change)
    with pytest.raises(ProviderResponseError) as error:
        run_review()
    assert error.value.raw_response == transport["response"]
    assert error.value.metadata["usage_cost_usd"] == 0.000528
    assert error.value.prompt
    assert len(transport["requests"]) == 1


@pytest.mark.parametrize(
    "change",
    [
        {"candidate_id": "unknown"},
        {"label": "truck"},
        {"reason": ""},
        {"box": [1, 2, 3, 4]},
    ],
)
def test_local_schema_validation_rejects_invalid_model_reviews(transport, change):
    result = deepcopy(REVIEW)
    result["reviews"][0].update(change)
    transport["response"]["choices"][0]["message"]["content"] = json.dumps(result)
    with pytest.raises(ProviderResponseError):
        run_review()
    assert len(transport["requests"]) == 1


def test_response_echoes_of_credentials_and_pixels_are_scrubbed(transport):
    transport["response"]["debug"] = {
        "authorization": "Bearer " + KEY,
        "content": KEY + " data:image/jpeg;base64,AAAAaaaa====",
    }
    result = run_review()
    assert KEY not in json.dumps(result)
    assert "AAAAaaaa" not in json.dumps(result)


def test_invalid_candidates_are_rejected_before_network(transport):
    reviewer = provider.AlibabaReviewer(CONFIG)
    with pytest.raises(ValueError, match="inside"):
        reviewer.review(Image.new("RGB", (20, 20)), CANDIDATES)
    assert transport["connections"] == []


def test_usage_without_reliable_token_counts_has_no_invented_cost(transport):
    transport["response"]["usage"] = {"prompt_tokens": "2500", "completion_tokens": 200}
    result = run_review()
    assert result["metadata"]["usage_cost_usd"] is None
    assert result["metadata"]["usage"] == transport["response"]["usage"]
