"""Offline cached DINO-X -> Astra review fixtures, never real API calls or keys."""

import base64
import hashlib
import io
import json
from copy import deepcopy

import pytest
from PIL import Image
from test_multimodal_provider import KEY
from test_multimodal_provider import transport as transport

from iris import dinox_provider as dinox
from iris import dinox_review_provider as provider
from iris import multimodal_provider as api
from iris.preannotation_contracts import normalize_output
from iris.taxonomies import TAXONOMY


def configuration():
    return provider.freeze_config(TAXONOMY)


def picture(size=(200, 100)):
    image = Image.new("RGB", size, "blue")
    image.info["comment"] = "PRIVATE-HUMAN-REFERENCE"
    image.filename = "PRIVATE-FILENAME.png"
    return image


def candidates(*, empty=False, count=2, size=(200, 100), clipped=False):
    objects = [
        {
            "category": "person" if index % 2 == 0 else "car",
            "bbox": [-2 if clipped else 2, 3, 30, 60],
            "score": 0.8,
        }
        for index in range(0 if empty else count)
    ]
    return dinox.normalize({"objects": objects}, dinox.freeze_config(TAXONOMY), *size)


def decisions(source, action="accept"):
    return [
        {
            "id": row["id"],
            "action": action,
            "label": row["label"],
            "reason": "Synthetic visual evidence.",
            "uncertain": False,
        }
        for row in source["proposals"]
    ]


def response(value):
    return {
        "id": "resp-review-fixture",
        "model": "gpt-6-astra",
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": json.dumps(value)}],
            }
        ],
        "usage": {
            "input_tokens": 2000,
            "output_tokens": 100,
            "input_tokens_details": {"cached_tokens": 100, "cache_write_tokens": 200},
            "output_tokens_details": {"reasoning_tokens": 40},
        },
    }


def prepared():
    return provider.prepare_request(picture(), configuration(), candidates())


def call(adapter, request, *, before=lambda: None, after=lambda *_: None):
    return adapter.request(request, before_dispatch=before, after_response=after)


def test_offline_reference_free_preparation_and_exact_known_dynamic_estimate(
    transport, monkeypatch
):
    monkeypatch.delenv(api.KEY_ENV)
    monkeypatch.delenv(api.FALLBACK_KEY_ENV, raising=False)
    config, source = configuration(), candidates()
    provider.DinoXReviewer(config)
    assert provider.validate_frozen_config(config) == config
    assert config["protocol"] != "iris-combined-preannotation-v1"
    assert config["max_external_calls_per_image"] == 1
    assert config["native_scores_sent"] is False
    first = provider.prepare_request(picture(), config, source)
    assert first == provider.prepare_request(picture(), config, source)
    safe = provider.safe_request(first, config)
    assert provider.validate_input(safe, config, source) == safe
    assert provider.reconstruct_input(config, safe["image"], source) == {
        key: value for key, value in safe.items() if key != "request_sha256"
    }
    assert all(
        word not in json.dumps(safe)
        for word in ["PRIVATE-HUMAN", "PRIVATE-FILENAME", KEY, "native_score", "planning_prompts"]
    )
    payload = first["payload"]
    body = json.loads(payload["input"][0]["content"][0]["text"])
    assert set(body) == {"classes", "image_size", "coordinate_space", "candidates"}
    assert body["classes"] == [
        {key: row[key] for key in ("id", "name", "definition")} for row in TAXONOMY["classes"]
    ]
    assert body["candidates"][0] == {
        "id": "dinox-0",
        "label": "person",
        "box": [0.01, 0.03, 0.15, 0.6],
    }
    assert payload["store"] is False and payload["tools"] == []
    assert payload["model"] == api.MODEL and payload["max_output_tokens"] == 2048
    data = base64.b64decode(payload["input"][0]["content"][1]["image_url"].split(",")[1])
    with Image.open(io.BytesIO(data)) as image:
        assert image.info == {} and image.size == (200, 100)
    assert safe["request_sha256"] == hashlib.sha256(api._request_bytes(payload)).hexdigest()
    assert safe["estimate"]["output_token_limit"] == 2048
    assert safe["estimate"]["text_byte_allowance"] == len(safe["prompt"].encode())
    assert safe["estimate"]["guaranteed_billing_cap"] is False
    assert not transport["connections"]


@pytest.mark.parametrize(
    "field",
    [
        "provider",
        "model",
        "candidate_model",
        "instructions",
        "output_schema",
        "geometry_policy",
        "openai_config",
    ],
)
def test_frozen_protocol_cannot_be_redefined(field):
    config = configuration()
    config[field] = "changed"
    with pytest.raises(ValueError):
        provider.validate_frozen_config(config)


def test_preserves_native_clipping_uncertain_relabel_and_all_rejections():
    source = candidates(clipped=True)
    snapshot = deepcopy(source)
    chosen = decisions(source)
    chosen[0].update(action="relabel", label="car", uncertain=True)
    chosen[1]["action"] = "reject"
    result = provider.normalize_response(
        response({"decisions": list(reversed(chosen))}), configuration(), source
    )
    assert source == snapshot
    assert result["decisions"] == chosen
    kept = result["proposals"][0]
    assert kept["id"] == source["proposals"][0]["id"]
    assert kept["box"] == source["proposals"][0]["box"] == [0, 3, 30, 60]
    assert kept["score"] is None and kept["label"] == "car" and kept["uncertain"] is True
    assert kept["source"]["dinox"] == source["proposals"][0]["source"]
    assert kept["source"]["dinox_geometry"] == source["proposals"][0]["geometry"]
    assert kept["source"]["native_score"] == 0.8
    assert kept["source"]["native_label"] == "person"
    assert "sam" not in kept["source"]
    result = provider.normalize_response(
        response({"decisions": decisions(source, "reject")}), configuration(), source
    )
    assert result["proposals"] == [] and len(result["decisions"]) == 2
    assert any("missed DINO-X objects cannot be recovered" in text for text in result["warnings"])


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "duplicate",
        "new_id",
        "geometry",
        "score",
        "unknown_class",
        "same_relabel",
        "changed_accept",
        "changed_reject",
        "uncertain",
        "empty_reason",
        "long_reason",
        "null",
        "new_root",
        "nonobject",
    ],
)
def test_decisions_cannot_silently_omit_invent_or_modify_candidates(mutation):
    source = candidates()
    value = {"decisions": decisions(source)}
    row = value["decisions"][0]
    if mutation == "missing":
        value["decisions"].pop()
    elif mutation == "duplicate":
        value["decisions"][1] = deepcopy(row)
    elif mutation == "new_id":
        row["id"] = "dinox-99"
    elif mutation == "geometry":
        row["box"] = [0, 0, 1, 1]
    elif mutation == "score":
        row["score"] = 0.9
    elif mutation == "unknown_class":
        row.update(action="relabel", label="unknown")
    elif mutation == "same_relabel":
        row["action"] = "relabel"
    elif mutation == "changed_accept":
        row["label"] = "car"
    elif mutation == "changed_reject":
        row.update(action="reject", label="car")
    elif mutation == "uncertain":
        row["uncertain"] = 1
    elif mutation == "empty_reason":
        row["reason"] = " "
    elif mutation == "long_reason":
        row["reason"] = " " * 2001 + "x"
    elif mutation == "null":
        value["decisions"] = None
    elif mutation == "new_root":
        value["extra_boxes"] = []
    else:
        value["decisions"][0] = []
    with pytest.raises(ValueError):
        provider.normalize_response(response(value), configuration(), source)


def test_empty_candidates_still_prepare_one_explicit_review(transport):
    source = candidates(empty=True)
    request = provider.prepare_request(picture(), configuration(), source)
    assert json.loads(request["payload"]["input"][0]["content"][0]["text"])["candidates"] == []
    transport["response"] = response({"decisions": []})
    result = call(provider.DinoXReviewer(configuration()), request)
    normalized = provider.normalize_response(result["raw_response"], configuration(), source)
    assert normalized["proposals"] == [] and normalized["decisions"] == []
    assert len(transport["requests"]) == 1
    with pytest.raises(ValueError):
        provider.normalize_response(
            response({"decisions": decisions(candidates())}), configuration(), source
        )


@pytest.mark.parametrize(
    "field",
    [
        "provider",
        "model",
        "native_coordinates",
        "native_box",
        "native_index",
        "clipped",
        "score",
        "id",
    ],
)
def test_canonical_but_false_native_provenance_is_rejected(field):
    source = candidates()
    native = source["raw_output"]["proposals"][0]
    if field == "score":
        native["score"] = None
    elif field == "id":
        native["id"] = "invented"
    elif field == "native_box":
        native["source"][field][0] += 1
    elif field == "native_index":
        native["source"][field] = True
    elif field == "clipped":
        native["source"][field] = True
    else:
        native["source"][field] = "other"
    source["proposals"] = normalize_output(source["raw_output"], TAXONOMY, width=200, height=100)[
        "proposals"
    ]
    with pytest.raises(ValueError):
        provider.prepare_request(picture(), configuration(), source)


def test_native_mutation_wrong_image_and_candidate_bound_are_rejected():
    source = candidates()
    source["proposals"][0]["box"][0] += 1
    with pytest.raises(ValueError):
        provider.prepare_request(picture(), configuration(), source)
    with pytest.raises(ValueError, match="different dimensions"):
        provider.prepare_request(picture((300, 100)), configuration(), candidates())
    with pytest.raises(ValueError):
        provider.prepare_request(picture(), configuration(), candidates(count=101))
    assert (
        len(
            provider.prepare_request(picture(), configuration(), candidates(count=100))[
                "dinox_normalized"
            ]["proposals"]
        )
        == 100
    )
    assert provider._number(10**1000) is False


@pytest.mark.parametrize(
    "field",
    [
        "prompt",
        "settings",
        "input_sha256",
        "request_sha256",
        "source_sha256",
        "estimate",
        "image",
        "protocol",
    ],
)
def test_saved_request_tampering_cannot_dispatch(transport, field):
    request = prepared()
    request[field] = "changed"
    with pytest.raises(ValueError):
        call(
            provider.DinoXReviewer(configuration()),
            request,
            before=lambda: pytest.fail("No dispatch expected"),
        )
    assert not transport["connections"]


@pytest.mark.parametrize(
    "change", ["payload", "pixels", "source", "missing_source", "native_scores"]
)
def test_actual_payload_pixels_and_source_are_bound_before_dispatch(transport, change):
    request = prepared()
    if change == "payload":
        request["payload"]["input"][0]["content"][0]["text"] = "hidden reference"
    elif change == "pixels":
        request["image_bytes"] += b"x"
    elif change == "source":
        request["dinox_normalized"]["filtered_count"] += 1
    elif change == "missing_source":
        request["dinox_normalized"] = None
    else:
        prompt = json.loads(request["prompt"])
        body = json.loads(prompt["input_text"])
        body["candidates"][0]["native_score"] = 0.8
        prompt["input_text"] = json.dumps(body)
        request["prompt"] = json.dumps(prompt)
    with pytest.raises(ValueError):
        call(
            provider.DinoXReviewer(configuration()),
            request,
            before=lambda: pytest.fail("No dispatch expected"),
        )
    assert not transport["connections"]


@pytest.mark.parametrize(
    "mutation", ["model", "incomplete", "refusal", "tool", "multiple", "duplicates", "malformed"]
)
def test_invalid_generations_are_receipted_before_normalization_and_never_retried(
    transport, mutation
):
    raw = response({"decisions": decisions(candidates())})
    if mutation == "model":
        raw["model"] = "other-model"
    elif mutation == "incomplete":
        raw["status"] = "incomplete"
    elif mutation == "refusal":
        raw["output"][0]["content"] = [{"type": "refusal", "refusal": "No"}]
    elif mutation == "tool":
        raw["output"].append({"type": "web_search_call"})
    elif mutation == "multiple":
        raw["output"].append(deepcopy(raw["output"][0]))
    else:
        raw["output"][0]["content"][0]["text"] = (
            '{"decisions":[],"decisions":[]}' if mutation == "duplicates" else "{"
        )
    transport["response"] = raw
    receipts = []
    result = call(
        provider.DinoXReviewer(configuration()),
        prepared(),
        after=lambda *args: receipts.append(args),
    )
    assert receipts[0][0] == raw
    with pytest.raises(ValueError):
        provider.normalize_response(result["raw_response"], configuration(), candidates())
    assert len(receipts) == 1 and len(transport["requests"]) == 1


def test_durable_receipt_precedes_parsing_and_reports_usage_without_double_counting_reasoning(
    transport,
):
    transport["response"] = response({"decisions": decisions(candidates())})
    events = []
    result = call(
        provider.DinoXReviewer(configuration()),
        prepared(),
        before=lambda: events.append("before"),
        after=lambda raw, meta: events.append(("after", raw, meta)),
    )
    assert events[0] == "before" and events[1][0] == "after"
    assert events[1][1] == transport["response"]
    assert result["metadata"]["usage_cost_usd"] == pytest.approx(0.0246)
    assert result["metadata"]["response_received"] is True
    assert result["metadata"]["source_sha256"] == prepared()["source_sha256"]
    assert len(transport["requests"]) == 1


@pytest.mark.parametrize("when", ["before", "after", "error_receipt"])
def test_callback_failures_remain_durable_conflicts_without_retries(transport, when):
    class Conflict(RuntimeError):
        pass

    def fail(*args):
        raise Conflict("durable state changed")

    transport["response"] = response({"decisions": decisions(candidates())})
    if when == "error_receipt":
        transport["status"] = 429
    with pytest.raises(Conflict):
        call(
            provider.DinoXReviewer(configuration()),
            prepared(),
            before=fail if when == "before" else lambda: None,
            after=fail if when != "before" else lambda *_: None,
        )
    assert len(transport["requests"]) == (0 if when == "before" else 1)


@pytest.mark.parametrize("kind", ["http", "malformed", "null", "incomplete", "network"])
def test_transport_errors_preserve_evidence_redact_key_and_do_not_retry(transport, kind):
    transport["response"] = {"error": {"message": KEY}}
    if kind == "http":
        transport["status"] = 429
    elif kind == "malformed":
        transport["response"] = (KEY + " not JSON").encode()
    elif kind == "null":
        transport["response"] = b"null"
    elif kind == "incomplete":
        transport["remaining"] = 12
    else:
        transport["error"] = OSError("network failure " + KEY)
    receipts = []
    with pytest.raises(provider.ProviderResponseError) as caught:
        call(
            provider.DinoXReviewer(configuration()),
            prepared(),
            after=lambda *args: receipts.append(args),
        )
    assert len(transport["requests"]) == 1
    assert caught.value.response_received is (kind not in {"incomplete", "network"})
    assert len(receipts) == (0 if kind in {"incomplete", "network"} else 1)
    assert KEY not in str(caught.value)
    assert KEY not in repr(caught.value.raw_response)
    assert KEY not in repr(caught.value.metadata)
    assert KEY not in repr(receipts)


def test_reused_adapter_failure_cannot_inherit_previous_receipt_fields(transport):
    adapter = provider.DinoXReviewer(configuration())
    transport["response"] = response({"decisions": decisions(candidates())})
    success = call(adapter, prepared())
    assert success["metadata"]["http_status"] == 200
    assert success["metadata"]["usage_cost_basis"]
    transport["error"] = OSError("second request failed before response")
    with pytest.raises(provider.ProviderResponseError) as caught:
        call(adapter, prepared())
    metadata = caught.value.metadata
    assert metadata["response_received"] is False
    assert metadata["http_status"] is None and metadata["http_request_id"] is None
    assert metadata["usage"] is None and metadata["usage_cost_usd"] is None
    assert metadata["request_id"] is None and metadata["returned_model"] is None
    assert "usage_cost_basis" not in metadata and "output_limit_exceeded" not in metadata
    assert len(transport["requests"]) == 2


def test_durable_dispatch_callback_prevents_replay(transport):
    dispatched = False

    def mark_once():
        nonlocal dispatched
        if dispatched:
            raise RuntimeError("already dispatched; explicit recovery is required")
        dispatched = True

    transport["response"] = response({"decisions": decisions(candidates())})
    adapter = provider.DinoXReviewer(configuration())
    call(adapter, prepared(), before=mark_once)
    with pytest.raises(RuntimeError, match="already dispatched"):
        call(adapter, prepared(), before=mark_once)
    assert len(transport["requests"]) == 1
