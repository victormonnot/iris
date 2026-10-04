"""Combined durable stages against real pure DTOs, with no model or HTTP calls."""

import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from decimal import ROUND_CEILING, Decimal

import pytest
from PIL import Image
from test_sam_runtime import identity

from iris import benchmark_combined_dispatch as dispatch
from iris import combined_provider as provider
from iris.benchmark import PROTOCOL, _digest
from iris.benchmark_combined import COMBINED_SCORING, work_plan
from iris.job_dispatch import DispatchConflict
from iris.media import import_asset
from iris.sam_provider import normalize_response
from iris.store import Store, _encode, new_id, now
from iris.taxonomies import TAXONOMY


def response(payload, identifier="response-fixture"):
    return {
        "id": identifier,
        "model": "gpt-6-astra",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": json.dumps(payload)}],
            }
        ],
    }


def make_trial(store, benchmark, frames, taxonomy, *, role="tuning"):
    settings = provider.freeze_config(taxonomy)
    candidate = {
        "protocol": PROTOCOL,
        "approach": "combined",
        "model_id": provider.MODEL,
        "model_name": "Offline combined fixture",
        "provider_config": settings,
        "reference_manifest_sha256": benchmark["manifest_sha256"],
        "scoring": COMBINED_SCORING,
    }
    config = store.insert(
        "benchmark_configs",
        {
            "id": new_id(),
            "benchmark_id": benchmark["id"],
            "name": "Combined fixture",
            "approach": "combined",
            "config": candidate,
            "fingerprint": _digest(candidate),
            "created_at": now(),
        },
    )
    requests, images = [], {}
    for frame in frames:
        path = frame.get("image_path") or store.get("frames", frame["frame_id"])["path"]
        with Image.open(store.artifact_path(path)) as source:
            image = source.convert("RGB")
        images[frame["frame_id"]] = image
        planning = provider.safe_request(provider.prepare_plan_request(image, settings))
        template = provider.reconstruct_review_template(settings, planning["image"])
        requests.append(
            {
                "frame_id": frame["frame_id"],
                "planning": {"input": planning, "estimate": planning["estimate"]},
                "review": {"template": template, "estimate": template["estimate"]},
            }
        )
    ceiling = sum(
        int(
            (Decimal(str(item[name]["estimate"]["upper_bound_usd"])) * 1_000_000).to_integral_value(
                rounding=ROUND_CEILING
            )
        )
        for item in requests
        for name in ("planning", "review")
    )
    plan = {
        "protocol": dispatch.PLAN_PROTOCOL,
        "provider": "openai",
        "model": "gpt-6-astra",
        "runtime_identity": identity(),
        "requests": requests,
        "estimate": {
            "currency": "USD",
            "upper_bound_usd": ceiling / 1_000_000,
            "estimated_ceiling_microusd": ceiling,
        },
        "approval": {
            "allow_external": True,
            "fingerprint": "e" * 64,
            "approved_at": now(),
            "budget_usd": ceiling / 1_000_000,
            "budget_microusd": ceiling,
            "estimated_ceiling_microusd": ceiling,
        },
    }
    trial = {
        "id": new_id(),
        "benchmark_id": benchmark["id"],
        "config_id": config["id"],
        "split": role,
        "job_id": new_id(),
        "created_at": now(),
        "config": {
            "protocol": PROTOCOL,
            "fingerprint": "e" * 64,
            "candidate_config": candidate,
            "source_config_fingerprint": config["fingerprint"],
            "external_plan": plan,
            "benchmark_manifest_sha256": benchmark["manifest_sha256"],
            "role": role,
            "frame_ids": [frame["frame_id"] for frame in frames],
            "work": work_plan(candidate, len(frames)),
            "warnings": [],
        },
    }
    with store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "INSERT INTO jobs(id,kind,status,params,created_at) VALUES(?,?,?,?,?)",
            (
                trial["job_id"],
                "benchmark",
                "queued",
                json.dumps({"trial_id": trial["id"]}),
                now(),
            ),
        )
        encoded = _encode(trial)
        conn.execute(
            f"INSERT INTO benchmark_trials({','.join(encoded)}) "
            f"VALUES({','.join('?' for _ in encoded)})",
            tuple(encoded.values()),
        )
        outputs = dispatch.initialize_outputs(conn, trial, frames, plan)
    return {
        "store": store,
        "trial": trial,
        "outputs": outputs,
        "frames": frames,
        "provider": settings,
        "images": images,
    }


@pytest.fixture
def combined(tmp_path):
    store = Store(tmp_path / "workspace")
    session = store.insert(
        "sessions",
        {
            "id": new_id(),
            "name": "Generated fixture",
            "scene_group": "fixture",
            "created_at": now(),
        },
    )
    frames = []
    for index in range(2):
        path = tmp_path / f"{index}.png"
        Image.new("RGB", (40, 30), (index * 100, 20, 30)).save(path)
        asset = import_asset(store, session["id"], path, path.name)
        frame = store.list("frames", asset_id=asset["id"])[0]
        frames.append(
            {"frame_id": frame["id"], "width": 40, "height": 30, "sha256": frame["sha256"]}
        )
    benchmark = store.insert(
        "benchmarks",
        {
            "id": new_id(),
            "project_id": "default",
            "name": "Private fixture",
            "path": "fixture",
            "manifest_sha256": "a" * 64,
            "summary": {},
            "created_at": now(),
        },
    )
    case = make_trial(store, benchmark, frames, TAXONOMY)
    yield case
    for image in case["images"].values():
        image.close()


def native(case, index=0):
    frame = case["frames"][index]
    width, height = frame["width"], frame["height"]
    return {
        "protocol": "iris-sam3-native-boxes-v1",
        "complete": True,
        "image": {"width": width, "height": height},
        "coordinates": {
            "format": "xyxy",
            "space": "normalized",
            "image_size": [width, height],
            "to_original": {"scale": [width, height], "offset": [0, 0]},
        },
        "prompts": [
            {
                **prompt,
                "boxes": [[0.1, 0.1, 0.5, 0.5]] if number == 0 else [],
                "scores": [0.9] if number == 0 else [],
                "native_indices": [5] if number == 0 else [],
                "error": None,
            }
            for number, prompt in enumerate(case["provider"]["sam_config"]["prompts"])
        ],
        "metadata": {"runtime_identity": identity(), "timing": {"elapsed_ms": 10}},
    }


def advance(case, attempt, *, index=0, through="review"):
    store, identifier = case["store"], case["outputs"][index]
    config = case["provider"]
    dispatch.begin_stage(store, identifier, attempt, "planning")
    plan_raw = response({"prompts": config["sam_config"]["prompts"]}, "planning-1")
    dispatch.save_stage_response(
        store,
        identifier,
        attempt,
        "planning",
        plan_raw,
        {"request_id": "planning-1", "usage_cost_usd": 0.001},
    )
    planning = provider.normalize_plan(plan_raw, config)
    dispatch.complete_stage(store, identifier, attempt, "planning", planning)
    if through == "planning":
        return planning
    dispatch.begin_stage(store, identifier, attempt, "grounding")
    ground_raw = native(case, index)
    dispatch.save_stage_response(store, identifier, attempt, "grounding", ground_raw)
    frame = case["frames"][index]
    grounded = normalize_response(
        ground_raw,
        provider.grounding_config(config, planning),
        width=frame["width"],
        height=frame["height"],
    )
    dispatch.complete_stage(store, identifier, attempt, "grounding", grounded)
    if through == "grounding":
        return planning, grounded
    request = provider.safe_request(
        provider.prepare_review_request(
            case["images"][frame["frame_id"]],
            config,
            planning,
            grounded,
        )
    )
    dispatch.begin_stage(store, identifier, attempt, "review", request=request)
    if through == "review_started":
        return request
    review_raw = response(
        {
            "decisions": [
                {
                    "id": item["id"],
                    "action": "accept",
                    "label": item["label"],
                    "reason": "Fixture candidate retained",
                    "uncertain": False,
                }
                for item in grounded["proposals"]
            ]
        },
        "review-1",
    )
    dispatch.save_stage_response(
        store,
        identifier,
        attempt,
        "review",
        review_raw,
        {"request_id": "review-1", "usage_cost_usd": 0.002},
    )
    result = provider.normalize_review(review_raw, config, grounded)
    dispatch.complete_stage(store, identifier, attempt, "review", result)
    if through == "review":
        return result
    return dispatch.publish_output(
        store, identifier, attempt, result, {"timing": {"elapsed_ms": 30}}
    )


def test_two_known_unsent_requests_per_image_and_no_invented_zero_cost(combined):
    summary = dispatch.dispatch_summary(combined["store"], combined["trial"]["id"])
    assert summary["counts"]["not_started"] == 4
    assert {row["stage"] for row in summary["outputs"]} == {"planning", "review"}
    assert summary["usage_cost_usd"] is None and summary["reserved_microusd"] == 0
    assert len(combined["store"].list("benchmark_outputs")) == 2


def test_claim_and_stage_admission_are_atomic_once(combined):
    store, trial, identifier = combined["store"], combined["trial"], combined["outputs"][0]

    def claim():
        try:
            return dispatch.claim_trial(store, trial["id"])
        except DispatchConflict:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: claim(), range(2)))
    (attempt,) = [result for result in results if result]

    def begin():
        try:
            dispatch.begin_stage(store, identifier, attempt, "planning")
            return True
        except DispatchConflict:
            return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sum(pool.map(lambda _: begin(), range(2))) == 1
    saved = store.get("benchmark_outputs", identifier)["metadata"]["pipeline"]["stages"]["planning"]
    assert saved["budget"]["reserved_microusd"] == saved["budget"]["ceiling_microusd"] > 0
    assert saved["state"] == "dispatching"


def test_complete_pipeline_replays_raw_provenance_and_records_two_costs(combined):
    store, trial, identifier = combined["store"], combined["trial"], combined["outputs"][0]
    attempt = dispatch.claim_trial(store, trial["id"])
    result = advance(combined, attempt, through="publish")
    assert result["result"]["proposals"][0]["score"] is None
    assert set(result["raw_response"]) == {"planning", "grounding", "review"}
    repeated = dispatch.save_stage_response(
        store, identifier, attempt, "review", result["raw_response"]["review"]
    )
    assert repeated["metadata"]["state"] == "ready"
    assert repeated["result"] == result["result"]
    dispatch.record_image_timing(store, identifier, attempt, 123.5)
    assert store.get("benchmark_outputs", identifier)["metadata"]["timing"]["elapsed_ms"] == 123.5
    summary = dispatch.dispatch_summary(store, trial["id"])
    assert summary["usage_cost_usd"] == 0.003
    assert summary["counts"] == {
        "dispatching": 0,
        "not_started": 2,
        "outcome_unknown": 0,
        "response_received": 2,
    }
    assert all(row["stage"] != "grounding" for row in summary["outputs"])
    with pytest.raises(DispatchConflict):
        dispatch.begin_stage(store, identifier, attempt, "planning")


def test_review_unknown_keeps_raw_planning_grounding_and_both_reservations(combined):
    store, trial, identifier = combined["store"], combined["trial"], combined["outputs"][0]
    attempt = dispatch.claim_trial(store, trial["id"])
    advance(combined, attempt, through="review_started")
    partial = {"http_status": 200, "body": "partial", "truncated": True}
    output = dispatch.fail_stage(store, identifier, attempt, "review", "Socket lost", raw=partial)
    assert output["raw_response"]["planning"]["id"] == "planning-1"
    assert output["raw_response"]["grounding"]["complete"] is True
    assert output["raw_response"]["review"] == partial
    summary = dispatch.dispatch_summary(store, trial["id"])
    assert summary["usage_cost_usd"] is None and summary["known_usage_cost_usd"] == 0.001
    assert summary["unknown_outcome_count"] == 1 and summary["usage_missing_count"] == 1
    stages = output["metadata"]["pipeline"]["stages"]
    assert summary["reserved_microusd"] == sum(
        item["budget"]["ceiling_microusd"] for item in stages.values()
    )
    with pytest.raises(DispatchConflict):
        dispatch.begin_stage(store, combined["outputs"][1], attempt, "planning")


@pytest.mark.parametrize("stage", ["planning", "grounding", "review"])
def test_cancel_between_completed_stages_preserves_success_without_advancing(combined, stage):
    store, trial, identifier = combined["store"], combined["trial"], combined["outputs"][0]
    attempt = dispatch.claim_trial(store, trial["id"])
    advance(combined, attempt, through=stage)
    store.update("jobs", trial["job_id"], {"cancel_requested": True})
    row = dispatch.fail_stage(store, identifier, attempt, stage, "Stopped after completed stage")
    assert row["metadata"]["pipeline"]["stages"][stage]["result"] is not None
    assert row["metadata"]["pipeline"]["stages"][stage]["error"] is None
    assert row["result"] is None and row["metadata"]["state"] == "cancelled"
    dispatch.validate_output_row(row, trial)
    dispatch.record_image_timing(store, identifier, attempt, 25)


def test_stage_order_and_raw_before_normalization_are_required(combined):
    store, identifier = combined["store"], combined["outputs"][0]
    attempt = dispatch.claim_trial(store, combined["trial"]["id"])
    with pytest.raises(DispatchConflict):
        dispatch.begin_stage(store, identifier, attempt, "grounding")
    with pytest.raises(DispatchConflict):
        dispatch.complete_stage(store, identifier, attempt, "planning", [])
    dispatch.begin_stage(store, identifier, attempt, "planning")
    dispatch.save_stage_response(store, identifier, attempt, "planning", {"invalid": "raw"})
    with pytest.raises(ValueError, match="raw stage evidence"):
        dispatch.complete_stage(store, identifier, attempt, "planning", [])
    with pytest.raises(DispatchConflict):
        dispatch.begin_stage(store, identifier, attempt, "grounding")


def test_unknown_can_receive_late_response_but_cannot_restart_or_publish(combined):
    store, trial, identifier = combined["store"], combined["trial"], combined["outputs"][0]
    attempt = dispatch.claim_trial(store, trial["id"])
    dispatch.begin_stage(store, identifier, attempt, "planning")
    store.update("jobs", trial["job_id"], {"status": "interrupted"})
    dispatch.recover_combined_dispatches(store)
    before = store.get("benchmark_outputs", identifier)
    dispatch.recover_combined_dispatches(store)
    assert store.get("benchmark_outputs", identifier) == before
    raw = response({"prompts": combined["provider"]["sam_config"]["prompts"]})
    saved = dispatch.save_stage_response(store, identifier, attempt, "planning", raw)
    assert saved["raw_response"]["planning"] == raw
    assert saved["metadata"]["pipeline"]["stages"]["planning"]["state"] == "response_received"
    with pytest.raises(DispatchConflict):
        dispatch.complete_stage(
            store,
            identifier,
            attempt,
            "planning",
            provider.normalize_plan(raw, combined["provider"]),
        )
    with pytest.raises(DispatchConflict):
        dispatch.claim_trial(store, trial["id"])


def test_local_failure_is_not_external_unknown_and_review_stays_unsent(combined):
    store, trial, identifier = combined["store"], combined["trial"], combined["outputs"][0]
    attempt = dispatch.claim_trial(store, trial["id"])
    advance(combined, attempt, through="planning")
    dispatch.begin_stage(store, identifier, attempt, "grounding")
    raw = native(combined)
    raw["complete"] = False
    output = dispatch.fail_stage(store, identifier, attempt, "grounding", RuntimeError(), raw=raw)
    assert output["error"] == "RuntimeError"
    assert output["raw_response"]["grounding"] == raw
    summary = dispatch.dispatch_summary(store, trial["id"])
    assert summary["unknown_outcome_count"] == 0 and summary["counts"]["not_started"] == 3


def test_fabricated_final_proposal_or_dynamic_review_request_cannot_be_saved(combined):
    store, trial, identifier = combined["store"], combined["trial"], combined["outputs"][0]
    attempt = dispatch.claim_trial(store, trial["id"])
    planning, grounded = advance(combined, attempt, through="grounding")
    prepared = provider.safe_request(
        provider.prepare_review_request(
            combined["images"][combined["frames"][0]["frame_id"]],
            combined["provider"],
            planning,
            grounded,
        )
    )
    altered = deepcopy(prepared)
    altered["input_sha256"] = "f" * 64
    with pytest.raises(ValueError):
        dispatch.begin_stage(store, identifier, attempt, "review", request=altered)
    assert (
        store.get("benchmark_outputs", identifier)["metadata"]["pipeline"]["stages"]["review"][
            "state"
        ]
        == "not_started"
    )
    dispatch.begin_stage(store, identifier, attempt, "review", request=prepared)
    review_raw = response(
        {
            "decisions": [
                {
                    "id": grounded["proposals"][0]["id"],
                    "label": grounded["proposals"][0]["label"],
                    "action": "accept",
                    "uncertain": False,
                    "reason": "Fixture",
                }
            ]
        }
    )
    dispatch.save_stage_response(store, identifier, attempt, "review", review_raw)
    result = provider.normalize_review(review_raw, combined["provider"], grounded)
    fabricated = deepcopy(result)
    fabricated["proposals"][0]["box"] = [0, 0, 1, 1]
    with pytest.raises(ValueError, match="differs"):
        dispatch.publish_output(store, identifier, attempt, fabricated)
    assert store.get("benchmark_outputs", identifier)["result"] is None
    dispatch.publish_output(store, identifier, attempt, result)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda plan: plan["approval"].update(allow_external=False),
        lambda plan: plan["approval"].update(fingerprint="f" * 64),
        lambda plan: plan["approval"].update(budget_microusd=True),
        lambda plan: plan["requests"][0]["planning"]["input"].pop("request_sha256"),
        lambda plan: plan["requests"][0]["review"]["template"].update(
            max_dynamic_text_bytes=1000000
        ),
        lambda plan: plan["runtime_identity"]["packages"].update(torch="2.11.0"),
    ],
)
def test_corrupt_consent_template_or_runtime_blocks_claim(combined, mutation):
    store, trial = combined["store"], deepcopy(combined["trial"])
    mutation(trial["config"]["external_plan"])
    store.update("benchmark_trials", trial["id"], {"config": trial["config"]})
    with pytest.raises(ValueError):
        dispatch.claim_trial(store, trial["id"])
    assert store.get("jobs", trial["job_id"])["result"] is None
