"""OpenAI Responses fixtures only: no provider calls, real secrets or image upload."""

import base64
import hashlib
import io
import json
import traceback
from copy import deepcopy

import pytest
from PIL import Image

from iris import multimodal_provider as provider
from iris.taxonomies import TAXONOMY

KEY = "sk-offline-fixture-only"


def test_failed_second_image_does_not_inherit_first_receipt(transport):
    adapter = provider.OpenAIPreannotator(provider.freeze_config(TAXONOMY))
    adapter.propose(Image.new("RGB", (64, 64)))
    assert adapter.metadata["http_status"] == 200
    transport["error"] = OSError("Synthetic connection failure")
    with pytest.raises(provider.ProviderResponseError) as raised:
        adapter.propose(Image.new("RGB", (64, 64), "red"))
    metadata = raised.value.metadata
    assert metadata.get("http_status") is None
    assert metadata.get("http_request_id") is None
    assert metadata["request_id"] is None
    assert metadata["returned_model"] is None
    assert metadata["usage"] is None
    assert metadata["usage_cost_usd"] is None
    assert metadata["response_received"] is False


PROPOSALS = {
    "coordinate_space": "normalized",
    "proposals": [
        {
            "label": "person",
            "box": [0.1, 0.2, 0.5, 0.6],
            "score": None,
            "uncertain": False,
            "reason": "Synthetic protocol fixture.",
        }
    ],
}


@pytest.fixture
def transport(monkeypatch):
    monkeypatch.setenv(provider.KEY_ENV, KEY)
    monkeypatch.delenv(provider.FALLBACK_KEY_ENV, raising=False)
    state = {
        "requests": [],
        "connections": [],
        "closed": 0,
        "error": None,
        "status": 200,
        "remaining": 0,
        "response": {
            "id": "resp-fixture",
            "model": provider.MODEL,
            "status": "completed",
            "error": None,
            "incomplete_details": None,
            "output": [
                {
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": json.dumps(PROPOSALS)}],
                }
            ],
            "usage": {
                "input_tokens": 2000,
                "output_tokens": 100,
                "input_tokens_details": {"cached_tokens": 100, "cache_write_tokens": 200},
                "output_tokens_details": {"reasoning_tokens": 80},
                "total_tokens": 2100,
            },
        },
    }

    class Socket:
        def settimeout(self, value):
            assert value > 0

    class Response:
        fp = None

        def __init__(self):
            self.status = state["status"]
            self.length = state["remaining"]
            raw = state["response"]
            self.body = io.BytesIO(raw if isinstance(raw, bytes) else json.dumps(raw).encode())

        def getheader(self, name):
            assert name == "x-request-id"
            return "http-request-fixture"

        def read1(self, limit):
            return self.body.read(limit)

    class Connection:
        def __init__(self, host, port, timeout):
            state["connections"].append((host, port, timeout))
            self.sock = Socket()

        def request(self, method, path, *, body, headers):
            state["requests"].append((method, path, body, headers))
            if state["error"]:
                raise state["error"]

        def getresponse(self):
            return Response()

        def close(self):
            state["closed"] += 1

    monkeypatch.setattr(provider.http.client, "HTTPSConnection", Connection)
    return state


def configuration(**changes):
    return provider.freeze_config(TAXONOMY, **changes)


def image():
    return Image.new("RGB", (2400, 1200), "blue")


def run(config=None):
    return provider.OpenAIPreannotator(config or configuration()).propose(image())


def test_status_config_constructor_and_preparation_are_offline(transport):
    config = configuration()
    assert provider.provider_status(config)["status"] == "ready"
    assert provider.provider_status(config)["connection_verified"] is False
    provider.OpenAIPreannotator(config)
    prepared = provider.prepare_request(image(), config)
    assert transport["connections"] == []
    assert KEY not in json.dumps(config)
    assert KEY not in prepared["prompt"]
    assert provider.validate_frozen_config(config) == config


def test_config_and_preview_work_without_key_and_status_does_not_echo_secrets(
    transport, monkeypatch
):
    monkeypatch.delenv(provider.KEY_ENV)
    config = configuration()
    assert provider.provider_status()["status"] == "missing_key"
    assert provider.prepare_request(image(), config)["request_sha256"]
    with pytest.raises(provider.ProviderResponseError) as caught:
        provider.OpenAIPreannotator(config)
    assert caught.value.response_received is False
    monkeypatch.setenv(provider.FALLBACK_KEY_ENV, KEY)
    assert provider.provider_status()["status"] == "ready"
    monkeypatch.setenv(provider.KEY_ENV, "bad key\n")
    assert provider.provider_status()["status"] == "missing_key"
    assert "bad key" not in json.dumps(provider.provider_status())
    assert transport["connections"] == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("model", "gpt-6-luna"),
        ("reasoning_effort", "none"),
        ("max_output_tokens", True),
        ("max_output_tokens", 8193),
        ("image_long_edge", 1600),
        ("image_long_edge", True),
    ],
)
def test_frozen_option_limits_reject_unsupported_settings(transport, field, value):
    with pytest.raises(ValueError):
        configuration(**{field: value})
    assert transport["connections"] == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("endpoint", "https://other.example/v1/responses"),
        ("api_key", KEY),
        ("system_prompt", "Use reference boxes"),
        ("model", "other"),
    ],
)
def test_frozen_contract_cannot_replace_endpoint_prompt_or_insert_secret(transport, field, value):
    config = configuration()
    config[field] = value
    with pytest.raises(ValueError):
        provider.validate_frozen_config(config)
    assert transport["connections"] == []


def test_store_false_is_a_boolean_and_no_tools_retry_or_reference_inputs(transport):
    config = configuration(reasoning_effort="max", image_long_edge=2048)
    invalid = deepcopy(config)
    invalid["settings"]["store"] = 0
    with pytest.raises(ValueError):
        provider.validate_config(invalid)
    result = run(config)
    assert len(transport["requests"]) == 1 and transport["closed"] == 1
    method, path, body, headers = transport["requests"][0]
    assert (method, path) == ("POST", "/v1/responses")
    assert transport["connections"][0][:2] == ("api.openai.com", 443)
    assert headers["Authorization"] == "Bearer " + KEY
    payload = json.loads(body)
    assert payload["store"] is False and payload["stream"] is False
    assert payload["tools"] == [] and payload["tool_choice"] == "none"
    assert "previous_response_id" not in payload and "conversation" not in payload
    assert payload["text"]["format"]["strict"] is True
    assert len(payload["input"]) == 1 and len(payload["input"][0]["content"]) == 2
    text, pixels = payload["input"][0]["content"]
    assert set(json.loads(text["text"])) == {"classes", "image_size", "coordinate_space"}
    assert pixels["detail"] == "original"
    assert result["result"]["proposals"][0]["box"] == [240, 240, 1200, 720]
    assert result["result"]["proposals"][0]["score"] is None
    assert result["metadata"]["request_id"] == "resp-fixture"
    assert result["metadata"]["http_request_id"] == "http-request-fixture"


def test_preparation_freezes_actual_clean_upload_pixels_transform_and_request_hash(transport):
    original = image()
    original.info["annotation_notes"] = "SECRET HUMAN REFERENCE"
    prepared = provider.prepare_request(original, configuration())
    repeated = provider.prepare_request(original, configuration())
    assert prepared == repeated
    descriptor = prepared["image"]
    assert (descriptor["width"], descriptor["height"]) == (2400, 1200)
    assert (descriptor["sent_width"], descriptor["sent_height"]) == (1536, 768)
    assert descriptor["transform"]["scale"] == [1.5625, 1.5625]
    assert hashlib.sha256(prepared["image_bytes"]).hexdigest() == descriptor["sha256"]
    pixels = prepared["payload"]["input"][0]["content"][1]["image_url"]
    assert base64.b64decode(pixels.split(",")[1]) == prepared["image_bytes"]
    with Image.open(io.BytesIO(prepared["image_bytes"])) as sent:
        assert not sent.info and sent.size == (1536, 768)
    assert "SECRET HUMAN REFERENCE" not in str(prepared)
    provider.OpenAIPreannotator(configuration()).propose(
        original, expected_image_sha256=descriptor["sha256"]
    )
    assert hashlib.sha256(transport["requests"][0][2]).hexdigest() == prepared["request_sha256"]


def test_changed_approved_bytes_prevent_dispatch(transport):
    with pytest.raises(ValueError, match="approved"):
        provider.OpenAIPreannotator(configuration()).propose(
            image(), expected_image_sha256="0" * 64
        )
    assert transport["requests"] == []


def test_estimate_is_explicitly_a_planning_allowance_and_usage_is_not_double_counted(transport):
    prepared = provider.prepare_request(image(), configuration())
    estimate = prepared["estimate"]
    assert estimate["image_tokens_estimate"] == 1384
    assert estimate["guaranteed_billing_cap"] is False
    assert estimate["upper_bound_usd"] > 0.2048
    result = run()
    # 1700 uncached +100 cached +200 cache writes; 80 reasoning are part of100output.
    assert result["metadata"]["usage_cost_usd"] == pytest.approx(0.0246)
    assert result["metadata"]["usage"]["output_tokens_details"]["reasoning_tokens"] == 80
    assert "not an invoice" in result["metadata"]["usage_cost_basis"]


@pytest.mark.parametrize(
    "usage",
    [
        None,
        {},
        {"input_tokens": True, "output_tokens": 1},
        {"input_tokens": 5, "output_tokens": 1, "input_tokens_details": {"cached_tokens": 6}},
    ],
)
def test_missing_or_invalid_usage_is_unknown_not_zero(transport, usage):
    transport["response"]["usage"] = usage
    assert run()["metadata"]["usage_cost_usd"] is None


def test_dispatch_and_receipt_callbacks_surround_http_before_geometry_validation(transport):
    events = []
    adapter = provider.OpenAIPreannotator(configuration())
    adapter.before_dispatch = lambda: events.append(("before", len(transport["requests"])))
    transport["response"]["output"][0]["content"][0]["text"] = json.dumps(
        {
            **PROPOSALS,
            "proposals": [{**PROPOSALS["proposals"][0], "box": [-1, 0, 1, 1]}],
        }
    )
    adapter.after_response = lambda raw, metadata: events.append(
        ("received", len(transport["requests"]), raw, metadata)
    )
    with pytest.raises(provider.ProviderResponseError) as caught:
        adapter.propose(image())
    assert events[0] == ("before", 0)
    assert events[1][:2] == ("received", 1)
    assert events[1][3]["usage_cost_usd"] is not None
    assert caught.value.response_received is True
    assert caught.value.raw_response == transport["response"]


@pytest.mark.parametrize("when", ["before", "after"])
def test_callback_errors_propagate_without_network_reclassification(transport, when):
    adapter = provider.OpenAIPreannotator(configuration())
    failure = ValueError("Dispatch ledger CAS lost")

    def reject(*args):
        raise failure

    setattr(adapter, "before_dispatch" if when == "before" else "after_response", reject)
    with pytest.raises(ValueError) as caught:
        adapter.propose(image())
    assert caught.value is failure
    assert len(transport["requests"]) == (0 if when == "before" else 1)
    assert transport["closed"] == 1


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown_class",
        "numeric_score",
        "inverted",
        "extra",
        "unknown_model",
        "truncated",
        "refusal",
        "tool",
    ],
)
def test_unusable_responses_keep_raw_and_usage_without_false_empty_predictions(transport, mutation):
    raw, parsed = transport["response"], deepcopy(PROPOSALS)
    if mutation == "unknown_model":
        raw["model"] = "gpt-6-luna"
    elif mutation == "truncated":
        raw.update(status="incomplete", incomplete_details={"reason": "max_output_tokens"})
    elif mutation == "refusal":
        raw["output"][0]["content"] = [{"type": "refusal", "refusal": "Fixture refusal"}]
    elif mutation == "tool":
        raw["output"].append({"type": "function_call"})
    else:
        if mutation == "unknown_class":
            parsed["proposals"][0]["label"] = "invented"
        elif mutation == "numeric_score":
            parsed["proposals"][0]["score"] = 0.9
        elif mutation == "inverted":
            parsed["proposals"][0]["box"] = [0.8, 0.5, 0.2, 0.3]
        else:
            parsed["reference"] = "Should not be accepted"
        raw["output"][0]["content"][0]["text"] = json.dumps(parsed)
    with pytest.raises(provider.ProviderResponseError) as caught:
        run()
    assert caught.value.raw_response == raw and caught.value.response_received
    assert caught.value.metadata["usage"] == raw["usage"]
    assert len(transport["requests"]) == 1


@pytest.mark.parametrize(
    "body,status,complete",
    [
        (b"null", 200, True),
        (b"invalid JSON", 200, True),
        (b"{}", 429, True),
        (b"{}", 302, True),
        (b"partial", 200, False),
    ],
)
def test_http_failures_and_partial_eof_have_explicit_receipts_and_never_retry(
    transport, body, status, complete
):
    transport.update(response=body, status=status, remaining=0 if complete else 20)
    adapter = provider.OpenAIPreannotator(configuration())
    receipts = []
    adapter.after_response = lambda raw, meta: receipts.append((raw, meta))
    with pytest.raises(provider.ProviderResponseError) as caught:
        adapter.propose(image())
    assert caught.value.response_received is complete
    assert bool(receipts) is complete
    assert caught.value.raw_response is not None
    assert len(transport["requests"]) == 1 and transport["closed"] == 1


def test_timeout_evidence_redacts_key_and_image_data_without_receipt(transport):
    transport["error"] = TimeoutError(f"timeout with {KEY} data:image/png;base64,SEVMTE8=")
    receipts = []
    adapter = provider.OpenAIPreannotator(configuration())
    adapter.after_response = lambda *args: receipts.append(args)
    with pytest.raises(provider.ProviderResponseError) as caught:
        adapter.propose(image())
    error = caught.value
    assert error.response_received is False and not receipts
    serial = str(error) + json.dumps(error.raw_response) + json.dumps(error.metadata)
    serial += "".join(traceback.format_exception(error))
    assert KEY not in serial and "SEVMTE8=" not in serial
    assert len(transport["requests"]) == 1


def test_empty_valid_result_is_explicit_and_resolved_snapshot_recorded(transport):
    transport["response"]["model"] = "gpt-6-astra-2026-10-01"
    transport["response"]["output"][0]["content"][0]["text"] = json.dumps(
        {
            "coordinate_space": "normalized",
            "proposals": [],
        }
    )
    result = run()
    assert result["result"]["proposals"] == []
    assert result["metadata"]["returned_model"] == "gpt-6-astra-2026-10-01"


def test_response_limit_preserves_bounded_evidence_and_no_complete_receipt(transport):
    transport["response"] = b"x" * (provider.MAX_RESPONSE_BYTES + 1)
    adapter = provider.OpenAIPreannotator(configuration())
    receipts = []
    adapter.after_response = lambda *args: receipts.append(args)
    with pytest.raises(provider.ProviderResponseError) as caught:
        adapter.propose(image())
    assert caught.value.response_received is False and not receipts
    assert caught.value.raw_response["truncated"] is True
    assert len(caught.value.raw_response["body"]) == provider.MAX_RESPONSE_BYTES
    assert len(transport["requests"]) == 1


def test_duplicate_json_keys_and_missing_response_identity_are_errors(transport):
    raw = transport["response"]
    raw["output"][0]["content"][0]["text"] = (
        '{"coordinate_space":"normalized","proposals":[],"proposals":[]}'
    )
    with pytest.raises(provider.ProviderResponseError, match="Duplicate"):
        run()
    raw["output"][0]["content"][0]["text"] = json.dumps(PROPOSALS)
    del raw["id"]
    with pytest.raises(provider.ProviderResponseError, match="identity"):
        run()


def test_pure_validator_rejects_malformed_reasoning_without_type_error(transport):
    config = configuration()
    config["settings"]["reasoning"]["effort"] = []
    with pytest.raises(ValueError):
        provider.validate_frozen_config(config)


def test_reported_long_context_tier_and_cache_writes_use_frozen_rates(transport):
    transport["response"]["usage"] = {
        "input_tokens": 300_000,
        "output_tokens": 2000,
        "input_tokens_details": {"cached_tokens": 50_000, "cache_write_tokens": 10_000},
    }
    # (240k*10 +50k*1 +10k*12.5)*2 +2000*50*1.5, all per million.
    assert run()["metadata"]["usage_cost_usd"] == pytest.approx(5.3)
