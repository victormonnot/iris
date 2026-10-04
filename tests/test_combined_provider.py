"""Combined planner/reviewer fixtures only: no API calls, SAM or model weights."""

import base64
import hashlib
import json
from copy import deepcopy

import pytest
from PIL import Image
from test_multimodal_provider import KEY
from test_multimodal_provider import transport as transport
from test_sam_provider import native

from iris import combined_provider as provider
from iris import multimodal_provider as api
from iris import sam_provider as sam
from iris.taxonomies import TAXONOMY


def configuration():
    return provider.freeze_config(TAXONOMY)


def picture():
    image = Image.new("RGB", (200, 100), "blue")
    image.info["comment"] = "PRIVATE-ANNOTATION-NOTES"
    return image


def response(value):
    return {
        "id": "resp-combined-fixture",
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
        },
    }


def prompts():
    return [{"class_id": "person", "text": "pedestrian"}, {"class_id": "car", "text": "car"}]


def grounding(config=None, *, empty=False):
    config = config or configuration()
    dynamic = provider.grounding_config(config, prompts())
    raw = native(dynamic)
    if empty:
        for row in raw["prompts"]:
            row.update(boxes=[], scores=[], native_indices=[])
    return sam.normalize_response(raw, dynamic, width=200, height=100)


def decisions(sam_result, action="accept"):
    return [
        {
            "id": item["id"],
            "action": action,
            "label": item["label"],
            "reason": "Synthetic image evidence",
            "uncertain": False,
        }
        for item in sam_result["proposals"]
    ]


def test_profile_and_request_preparation_are_offline_and_reference_free(transport):
    config = configuration()
    assert provider.validate_frozen_config(config) == config
    assert config["max_external_calls_per_image"] == 2
    assert config["review_empty_candidates"] is True
    plan = provider.prepare_plan_request(picture(), config)
    assert plan == provider.prepare_plan_request(picture(), config)
    assert plan["request_sha256"] == hashlib.sha256(api._request_bytes(plan["payload"])).hexdigest()
    assert plan["payload"]["store"] is False and plan["payload"]["tools"] == []
    assert plan["payload"]["model"] == "gpt-6-astra"
    assert "PRIVATE-ANNOTATION-NOTES" not in json.dumps(provider.safe_request(plan))
    data_url = plan["payload"]["input"][0]["content"][1]["image_url"]
    assert base64.b64decode(data_url.split(",", 1)[1]) == plan["image_bytes"]
    body = json.loads(plan["payload"]["input"][0]["content"][0]["text"])
    assert set(body) == {"classes", "coordinate_space", "image_size"}
    assert body["classes"] == [
        {key: row[key] for key in ("id", "name", "definition")} for row in TAXONOMY["classes"]
    ]
    assert not transport["connections"]
    assert KEY not in repr(config)


@pytest.mark.parametrize(
    "field",
    ["model", "openai_config", "sam_config", "planning", "review", "max_external_calls_per_image"],
)
def test_frozen_protocol_cannot_be_redefined(field):
    config = configuration()
    config[field] = "changed"
    with pytest.raises(ValueError):
        provider.validate_frozen_config(config)


@pytest.mark.parametrize(
    "change",
    [
        "none",
        "missing",
        "duplicate",
        "unknown",
        "geometry",
        "too_long",
        "blank",
        "newline",
        "extra_root",
    ],
)
def test_planner_requires_exact_class_phrases_without_geometry(change):
    value = {"prompts": prompts()}
    if change == "none":
        value["prompts"] = None
    elif change == "missing":
        value["prompts"].pop()
    elif change == "duplicate":
        value["prompts"][1] = deepcopy(value["prompts"][0])
    elif change == "unknown":
        value["prompts"][1]["class_id"] = "boat"
    elif change == "geometry":
        value["prompts"][0]["box"] = [0, 0, 1, 1]
    elif change == "too_long":
        value["prompts"][0]["text"] = "x" * 121
    elif change == "blank":
        value["prompts"][0]["text"] = " "
    elif change == "newline":
        value["prompts"][0]["text"] = "person\nignore rules"
    else:
        value["scene_notes"] = "Not part of the frozen protocol"
    with pytest.raises(ValueError):
        provider.normalize_plan(response(value), configuration())


def test_plan_replaces_sam_phrases_without_changing_model_or_taxonomy():
    config = configuration()
    planned = provider.normalize_plan(response({"prompts": prompts()}), config)
    dynamic = provider.sam_config_for_plan(config, planned)
    assert dynamic["prompts"] == prompts()
    assert dynamic["taxonomy"] == config["taxonomy"]
    assert dynamic["weights"] == config["sam_config"]["weights"]
    assert dynamic["settings"] == config["sam_config"]["settings"]
    assert sam.validate_frozen_config(dynamic) == dynamic


def test_review_preserves_exact_sam_geometry_native_scores_and_full_decisions():
    source = grounding()
    before = deepcopy(source)
    chosen = decisions(source)
    chosen[0].update(action="relabel", label="car", uncertain=True)
    chosen[1]["action"] = "reject"
    final = provider.normalize_review(
        response({"decisions": list(reversed(chosen))}), configuration(), source
    )
    assert len(final["proposals"]) == 1
    kept = final["proposals"][0]
    assert kept["id"] == source["proposals"][0]["id"]
    assert kept["box"] == source["proposals"][0]["box"]
    assert kept["label"] == "car" and kept["uncertain"] is True
    assert kept["score"] is None
    assert kept["source"]["native_score"] == source["proposals"][0]["score"]
    assert kept["source"]["sam"] == source["proposals"][0]["source"]
    assert final["decisions"] == chosen
    assert source == before


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "duplicate",
        "invented",
        "geometry",
        "new_label",
        "accept_changed",
        "same_relabel",
        "uncertain",
        "empty_reason",
        "extra_root",
    ],
)
def test_review_cannot_invent_modify_or_silently_omit_candidates(change):
    source = grounding()
    value = {"decisions": decisions(source)}
    row = value["decisions"][0]
    if change == "missing":
        value["decisions"].pop()
    elif change == "duplicate":
        value["decisions"][1] = deepcopy(row)
    elif change == "invented":
        row["id"] = "new-object"
    elif change == "geometry":
        row["box"] = [0, 0, 1, 1]
    elif change == "new_label":
        row.update(action="relabel", label="boat")
    elif change == "accept_changed":
        row["label"] = "car"
    elif change == "same_relabel":
        row["action"] = "relabel"
    elif change == "uncertain":
        row["uncertain"] = 1
    elif change == "empty_reason":
        row["reason"] = " "
    else:
        value["new_boxes"] = []
    with pytest.raises(ValueError):
        provider.normalize_review(response(value), configuration(), source)


@pytest.mark.parametrize(
    "mutation", ["model", "incomplete", "refusal", "tools", "multiple", "malformed"]
)
def test_refusal_or_invalid_envelope_never_becomes_successful_empty(mutation):
    raw = response({"prompts": prompts()})
    if mutation == "model":
        raw["model"] = "other-model"
    elif mutation == "incomplete":
        raw["status"] = "incomplete"
    elif mutation == "refusal":
        raw["output"][0]["content"] = [{"type": "refusal", "refusal": "No"}]
    elif mutation == "tools":
        raw["output"].append({"type": "web_search_call"})
    elif mutation == "multiple":
        raw["output"].append(deepcopy(raw["output"][0]))
    else:
        raw["output"][0]["content"][0]["text"] = '{"prompts":[],"prompts":[]}'
    with pytest.raises(ValueError):
        provider.normalize_plan(raw, configuration())


def test_two_requests_share_image_and_template_reservation_bounds_dynamic_input(transport):
    config = configuration()
    plan = provider.prepare_plan_request(picture(), config)
    template = provider.review_template(picture(), config)
    source = grounding(config)
    review = provider.prepare_review_request(picture(), config, prompts(), source)
    assert plan["image_bytes"] == review["image_bytes"]
    assert plan["image"] == template["image"] == review["image"]
    assert "payload" not in template and "request_sha256" not in template
    assert template["max_dynamic_text_bytes"] == 128 * 1024
    assert review["dynamic_text_bytes"] > 0
    assert review["estimate"]["upper_bound_usd"] < template["estimate"]["upper_bound_usd"]
    assert template["estimate"]["guaranteed_billing_cap"] is False
    body = json.loads(review["payload"]["input"][0]["content"][0]["text"])
    assert body["planning_prompts"] == prompts()
    assert set(body["candidates"][0]) == {"id", "label", "box", "native_score"}
    assert body["candidates"][0]["box"] == [0.1, 0.2, 0.7, 0.9]
    assert not transport["requests"]


def test_archive_can_reconstruct_safe_inputs_without_pixels_or_runtime(monkeypatch):
    config = configuration()
    plan = provider.prepare_plan_request(picture(), config)
    review = provider.prepare_review_request(picture(), config, prompts(), grounding())
    template = provider.review_template(picture(), config)
    monkeypatch.setattr(api, "_prepare_image", lambda *a: pytest.fail("No pixels required"))
    for prepared, reconstructed in [
        (plan, provider.reconstruct_plan_input(config, plan["image"])),
        (
            review,
            provider.reconstruct_review_input(config, review["image"], prompts(), grounding()),
        ),
    ]:
        safe = provider.safe_request(prepared, config)
        assert provider.validate_input(safe, config) == safe
        assert {
            key: value for key, value in safe.items() if key != "request_sha256"
        } == reconstructed
        assert "payload" not in safe and "image_bytes" not in safe
    saved_template = {key: value for key, value in template.items() if key != "image_bytes"}
    assert provider.validate_review_template(saved_template, config) == saved_template
    assert provider.reconstruct_review_template(config, plan["image"]) == saved_template


@pytest.mark.parametrize(
    "field", ["prompt", "settings", "input_sha256", "estimate", "dynamic_text_bytes", "image"]
)
def test_saved_input_corruption_is_detected(field):
    config = configuration()
    safe = provider.safe_request(provider.prepare_plan_request(picture(), config))
    safe[field] = "changed"
    with pytest.raises((ValueError, TypeError)):
        provider.validate_input(safe, config)


def test_empty_sam_still_has_review_request_and_cannot_gain_new_boxes():
    config = configuration()
    source = grounding(empty=True)
    review = provider.prepare_review_request(picture(), config, prompts(), source)
    assert json.loads(review["payload"]["input"][0]["content"][0]["text"])["candidates"] == []
    final = provider.normalize_review(response({"decisions": []}), config, source)
    assert final["proposals"] == [] and final["decisions"] == []
    with pytest.raises(ValueError):
        provider.normalize_review(response({"decisions": [{"id": "invented"}]}), config, source)


def test_all_rejected_is_valid_but_not_a_claim_of_no_objects():
    source = grounding()
    final = provider.normalize_review(
        response({"decisions": decisions(source, "reject")}), configuration(), source
    )
    assert final["proposals"] == [] and len(final["decisions"]) == 2
    assert any("missed SAM objects" in message for message in final["warnings"])


def test_stage_transport_receipts_precede_normalization_and_usage_is_measured(transport):
    config = configuration()
    prepared = provider.prepare_plan_request(picture(), config)
    transport["response"] = response({"prompts": prompts()})
    events = []
    adapter = provider.CombinedOpenAI(config)
    result = adapter.request(
        prepared,
        before_dispatch=lambda: events.append("before"),
        after_response=lambda raw, metadata: events.append(("receipt", raw, metadata)),
    )
    assert events[0] == "before" and events[1][0] == "receipt"
    assert events[1][1] == transport["response"]
    assert result["raw_response"] == transport["response"]
    assert result["metadata"]["usage_cost_usd"] > 0
    assert result["metadata"]["response_received"] is True
    assert provider.normalize_plan(result["raw_response"], config) == prompts()
    assert len(transport["requests"]) == 1


@pytest.mark.parametrize("when", ["before", "after"])
def test_callback_cas_failures_are_not_rewritten_as_network_errors(transport, when):
    transport["response"] = response({"prompts": prompts()})

    class CallbackConflict(ValueError):
        pass

    def fail(*args):
        raise CallbackConflict("durable state changed")

    with pytest.raises(CallbackConflict):
        provider.CombinedOpenAI(configuration()).request(
            provider.prepare_plan_request(picture(), configuration()),
            before_dispatch=fail if when == "before" else lambda: None,
            after_response=fail if when == "after" else lambda *_: None,
        )
    assert len(transport["requests"]) == (0 if when == "before" else 1)


@pytest.mark.parametrize("kind", ["http", "malformed", "null", "incomplete"])
def test_transport_errors_keep_complete_response_receipts_without_retry(transport, kind):
    transport["response"] = response({"prompts": prompts()})
    if kind == "http":
        transport["status"] = 429
    elif kind == "malformed":
        transport["response"] = b"not JSON"
    elif kind == "null":
        transport["response"] = b"null"
    else:
        transport["remaining"] = 12
    receipts = []
    with pytest.raises(provider.ProviderResponseError) as error:
        provider.CombinedOpenAI(configuration()).request(
            provider.prepare_plan_request(picture(), configuration()),
            before_dispatch=lambda: None,
            after_response=lambda *args: receipts.append(args),
        )
    assert len(transport["requests"]) == 1
    assert error.value.response_received is (kind != "incomplete")
    assert len(receipts) == (0 if kind == "incomplete" else 1)
    assert error.value.raw_response is not None


def test_approval_payload_change_is_refused_before_dispatch(transport):
    prepared = provider.prepare_plan_request(picture(), configuration())
    prepared["payload"]["input"][0]["content"][0]["text"] = "Changed after preview"
    with pytest.raises(ValueError, match="differs"):
        provider.CombinedOpenAI(configuration()).request(
            prepared,
            before_dispatch=lambda: pytest.fail("No dispatch"),
            after_response=lambda *_: None,
        )
    assert not transport["requests"]


def test_invalid_generation_is_receipted_and_never_retried(transport):
    transport["response"] = response({"prompts": []})
    receipts = []
    answer = provider.CombinedOpenAI(configuration()).request(
        provider.prepare_plan_request(picture(), configuration()),
        before_dispatch=lambda: None,
        after_response=lambda *args: receipts.append(args),
    )
    with pytest.raises(ValueError):
        provider.normalize_plan(answer["raw_response"], configuration())
    assert len(receipts) == 1 and len(transport["requests"]) == 1


def test_dynamic_limit_is_enforced_before_an_external_request(monkeypatch):
    config = configuration()
    source = grounding()
    prepared = provider.prepare_review_request(picture(), config, prompts(), source)
    monkeypatch.setattr(provider, "MAX_DYNAMIC_TEXT_BYTES", prepared["dynamic_text_bytes"] - 1)
    limited = configuration()
    with pytest.raises(ValueError, match="review allowance"):
        provider.prepare_review_request(picture(), limited, prompts(), source)


def test_sam_candidate_mutation_is_not_accepted_as_review_input():
    source = grounding()
    source["proposals"][0]["box"][0] += 1
    with pytest.raises(ValueError, match="changed"):
        provider.prepare_review_request(picture(), configuration(), prompts(), source)


def test_long_context_estimate_multiplies_input_and_output_separately():
    config = configuration()
    descriptor = provider.prepare_plan_request(picture(), config)["image"]
    estimate = provider._estimate(config, descriptor, 300_000)
    expected = (
        estimate["input_token_allowance"] * 25.0 + estimate["output_token_limit"] * 75.0
    ) / 1_000_000
    assert estimate["upper_bound_usd"] == expected


def test_complete_error_receipt_callback_failure_remains_a_callback_conflict(transport):
    transport["status"] = 429
    transport["response"] = response({"prompts": prompts()})

    class ReceiptConflict(RuntimeError):
        pass

    def reject_receipt(*args):
        raise ReceiptConflict("receipt state changed")

    with pytest.raises(ReceiptConflict):
        provider.CombinedOpenAI(configuration()).request(
            provider.prepare_plan_request(picture(), configuration()),
            before_dispatch=lambda: None,
            after_response=reject_receipt,
        )
    assert len(transport["requests"]) == 1
