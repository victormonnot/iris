"""Durable three-stage benchmark journal, with two external sends per image.

The planning and review allowances are reserved separately before transport.
Grounding is local. Complete raw responses survive cancellation, while advancing
or publishing a pipeline always requires its original active job claim.
"""

from __future__ import annotations

import math
from copy import deepcopy
from decimal import ROUND_FLOOR, Decimal

from iris.benchmark_dispatch import _active, _digest, _json, _micros, _write
from iris.job_dispatch import DispatchConflict
from iris.store import Store, _decode, new_id, now

PLAN_PROTOCOL = "iris-combined-trial-v1"
DISPATCH_PROTOCOL = "iris-combined-dispatch-v1"
STAGES = ("planning", "grounding", "review")
EXTERNAL_STAGES = ("planning", "review")
EXTERNAL_STATES = {"not_started", "dispatching", "response_received", "outcome_unknown"}
LOCAL_STATES = {"not_started", "running", "response_received", "interrupted"}
_PROTECTED = {"pipeline", "state", "frame_sha256", "config_fingerprint", "reference_withheld"}
_RAW_LIMIT = 16 * 1024 * 1024
_RESULT_LIMIT = 2 * 1024 * 1024


def _provider(frozen):
    from iris.combined_provider import validate_frozen_config

    config = frozen.get("candidate_config", {})
    provider = config.get("provider_config")
    if config.get("approach") != "combined":
        raise ValueError("Combined dispatch requires a frozen combined candidate")
    validate_frozen_config(provider)
    return provider


def _plan(frozen):
    from iris.combined_provider import reconstruct_plan_input, validate_input
    from iris.sam_runtime import validate_runtime_identity

    provider = _provider(frozen)
    plan = frozen.get("external_plan")
    if (
        not isinstance(plan, dict)
        or plan.get("protocol") != PLAN_PROTOCOL
        or plan.get("provider") != "openai"
        or plan.get("model") != "gpt-6-astra"
        or not isinstance(plan.get("requests"), list)
        or not 1 <= len(plan["requests"]) <= 25
    ):
        raise ValueError("Unsupported combined benchmark external plan")
    validate_runtime_identity(plan.get("runtime_identity"))
    _json(plan)
    seen, ceiling = set(), 0
    for item in plan["requests"]:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("frame_id"), str)
            or not item["frame_id"]
            or item["frame_id"] in seen
            or not isinstance(item.get("planning"), dict)
            or not isinstance(item.get("review"), dict)
        ):
            raise ValueError(
                "Combined requests require distinct image IDs and both external stages"
            )
        seen.add(item["frame_id"])
        prepared = item["planning"].get("input")
        validate_input(prepared, provider, stage="planning")
        if not _digest(prepared.get("request_sha256")):
            raise ValueError("Planning requires the checksum of its actual prepared POST")
        if {key: value for key, value in prepared.items() if key != "request_sha256"} != (
            reconstruct_plan_input(provider, prepared["image"])
        ):
            raise ValueError("Combined planning input differs from its frozen configuration")
        _validate_template(item["review"].get("template"), provider, prepared["image"])
        for stage, evidence in (("planning", prepared), ("review", item["review"]["template"])):
            estimate = item[stage].get("estimate")
            if (
                not isinstance(estimate, dict)
                or estimate.get("currency") != "USD"
                or estimate != evidence.get("estimate")
            ):
                raise ValueError("Combined planning allowances differ from their frozen inputs")
            ceiling += _micros(estimate.get("upper_bound_usd"))
    approval, estimate = plan.get("approval"), plan.get("estimate")
    if (
        not isinstance(approval, dict)
        or approval.get("allow_external") is not True
        or not _digest(frozen.get("fingerprint"))
        or approval.get("fingerprint") != frozen["fingerprint"]
        or not isinstance(approval.get("approved_at"), str)
        or not approval["approved_at"]
        or type(approval.get("estimated_ceiling_microusd")) is not int
        or approval["estimated_ceiling_microusd"] != ceiling
        or type(approval.get("budget_microusd")) is not int
        or approval["budget_microusd"] < ceiling
        or _micros(approval.get("budget_usd"), ROUND_FLOOR) != approval["budget_microusd"]
        or not isinstance(estimate, dict)
        or estimate.get("currency") != "USD"
        or type(estimate.get("estimated_ceiling_microusd")) is not int
        or estimate["estimated_ceiling_microusd"] != ceiling
        or _micros(estimate.get("upper_bound_usd")) != ceiling
    ):
        raise ValueError("Combined consent or budget differs from the two per-image allowances")
    expected_work = {
        "image_count": len(seen),
        "request_count": 2 * len(seen),
        "image_encodings": len(seen),
        "prompt_evaluations": len(seen) * len(provider["taxonomy"]["classes"]),
        "max_iterations": 1,
        "warmup_passes": 0,
    }
    if frozen.get("work") != expected_work or any(
        type(frozen["work"][key]) is not int for key in expected_work
    ):
        raise ValueError("Combined trial work differs from the bounded single-iteration protocol")
    return plan


def _validate_template(template, provider, image):
    from iris.combined_provider import reconstruct_review_template

    if template != reconstruct_review_template(provider, image):
        raise ValueError("The frozen combined review template or allowance changed")


def validate_external_trial(frozen, config, frames):
    """Pure archival checks: no credentials, installed runtimes or image reads."""
    plan = _plan(frozen)
    if (
        frozen.get("candidate_config") != config
        or frozen.get("frame_ids") != [frame["frame_id"] for frame in frames]
        or frozen["frame_ids"] != [request["frame_id"] for request in plan["requests"]]
    ):
        raise ValueError("Combined image order differs from its frozen manifest partition")
    for frame, request in zip(frames, plan["requests"], strict=True):
        image = request["planning"]["input"]["image"]
        for actual, expected in (
            (image["source_pixel_sha256"], frame.get("sha256", image["source_pixel_sha256"])),
            (image["width"], frame.get("width", image["width"])),
            (image["height"], frame.get("height", image["height"])),
        ):
            if actual != expected:
                raise ValueError(
                    "Combined outgoing image provenance differs from its reference pixels"
                )
    return plan


def _stage_template(stage, request, plan):
    external = stage in EXTERNAL_STAGES
    return {
        "state": "not_started",
        "external": external,
        "provider": "openai" if external else "sam3",
        "model": plan["model"] if external else "sam3",
        "attempt_id": None,
        "attempted_at": None,
        "response_received_at": None,
        "completed_at": None,
        "request": deepcopy(request["planning"]["input"]) if stage == "planning" else None,
        "budget": {
            "currency": "USD",
            "ceiling_microusd": _micros(request[stage]["estimate"]["upper_bound_usd"])
            if external
            else 0,
            "reserved_microusd": 0,
        },
        "result": None,
        "error": None,
        "metadata": {},
    }


def initialize_outputs(conn, trial, frames, external_plan):
    if trial["config"].get("external_plan") != external_plan:
        raise ValueError("Combined initialization must use the trial's saved external plan")
    validate_external_trial(trial["config"], trial["config"]["candidate_config"], frames)
    identifiers = []
    for request in external_plan["requests"]:
        identifier = new_id()
        metadata = {
            "state": "not_started",
            "reference_withheld": True,
            "frame_sha256": request["planning"]["input"]["image"]["source_pixel_sha256"],
            "config_fingerprint": trial["config"]["source_config_fingerprint"],
            "pipeline": {
                "protocol": DISPATCH_PROTOCOL,
                "stages": {
                    stage: _stage_template(stage, request, external_plan) for stage in STAGES
                },
            },
            "timing": {},
        }
        conn.execute(
            "INSERT INTO benchmark_outputs (id,trial_id,frame_id,raw_response,metadata,created_at) "
            "VALUES (?,?,?,?,?,?)",
            (
                identifier,
                trial["id"],
                request["frame_id"],
                _json(dict.fromkeys(STAGES)),
                _json(metadata),
                now(),
            ),
        )
        identifiers.append(identifier)
    return identifiers


def _grounding_input(stages, request, plan):
    image = request["planning"]["input"]["image"]
    return {
        "prompts": deepcopy(stages["planning"]["result"]),
        "runtime_identity": deepcopy(plan["runtime_identity"]),
        "image": {key: image[key] for key in ("source_pixel_sha256", "width", "height")},
    }


def _normalized(stage, raw, provider, stages, image):
    from iris.combined_provider import grounding_config, normalize_plan, normalize_review
    from iris.sam_provider import normalize_response

    if stage == "planning":
        return normalize_plan(raw, provider)
    if stage == "grounding":
        return normalize_response(
            raw,
            grounding_config(provider, stages["planning"]["result"]),
            width=image["width"],
            height=image["height"],
        )
    return normalize_review(raw, provider, stages["grounding"]["result"])


def _validate_timing(metadata):
    timing = metadata.get("timing", {})
    if not isinstance(timing, dict):
        raise ValueError("Combined measured timing must be an object")
    value = timing.get("elapsed_ms")
    if value is not None:
        try:
            valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError("Combined measured time must be finite nonnegative milliseconds")


def validate_output_row(row, trial, *, validated_plan=None):
    from iris.combined_provider import reconstruct_review_input, validate_input

    frozen = trial["config"]
    plan = _plan(frozen) if validated_plan is None else validated_plan
    provider = frozen["candidate_config"]["provider_config"]
    request = next((item for item in plan["requests"] if item["frame_id"] == row["frame_id"]), None)
    metadata, raw = row.get("metadata"), row.get("raw_response")
    pipeline = metadata.get("pipeline") if isinstance(metadata, dict) else None
    if (
        request is None
        or row.get("trial_id") != trial["id"]
        or not isinstance(pipeline, dict)
        or pipeline.get("protocol") != DISPATCH_PROTOCOL
        or not isinstance(pipeline.get("stages"), dict)
        or set(pipeline["stages"]) != set(STAGES)
        or not isinstance(raw, dict)
        or set(raw) != set(STAGES)
        or metadata.get("reference_withheld") is not True
        or metadata.get("frame_sha256")
        != request["planning"]["input"]["image"]["source_pixel_sha256"]
        or metadata.get("config_fingerprint") != frozen["source_config_fingerprint"]
    ):
        raise ValueError("Combined output has invalid pipeline provenance")
    stages = pipeline["stages"]
    _validate_timing(metadata)
    image = request["planning"]["input"]["image"]
    previous, attempt = None, None
    for name in STAGES:
        stage = stages[name]
        expected = _stage_template(name, request, plan)
        if (
            not isinstance(stage, dict)
            or any(stage.get(key) != expected[key] for key in ("external", "provider", "model"))
            or type(stage.get("external")) is not bool
            or stage.get("state") not in (EXTERNAL_STATES if stage["external"] else LOCAL_STATES)
            or not isinstance(stage.get("budget"), dict)
            or stage["budget"].get("currency") != "USD"
            or type(stage["budget"].get("ceiling_microusd")) is not int
            or stage["budget"]["ceiling_microusd"] != expected["budget"]["ceiling_microusd"]
            or type(stage["budget"].get("reserved_microusd")) is not int
            or stage["budget"]["reserved_microusd"]
            not in {0, expected["budget"]["ceiling_microusd"]}
            or not isinstance(stage.get("metadata"), dict)
            or (stage.get("error") is not None and not isinstance(stage["error"], str))
        ):
            raise ValueError("Combined stage receipt or reservation is invalid")
        state = stage["state"]
        _validate_timing(stage["metadata"])
        owner = stage.get("attempt_id")
        if owner is not None and (not isinstance(owner, str) or not owner):
            raise ValueError("Combined stage attempt identity is invalid")
        if owner is not None:
            if attempt is not None and owner != attempt:
                raise ValueError("Combined stages have different trial attempt owners")
            attempt = owner
        if state == "not_started":
            if (
                stage["budget"]["reserved_microusd"] != 0
                or stage.get("attempted_at") is not None
                or stage.get("response_received_at") is not None
                or raw[name] is not None
                or stage.get("result") is not None
                or stage.get("completed_at") is not None
            ):
                raise ValueError("Unsent combined stages cannot contain responses or reservations")
        else:
            if (
                not owner
                or not isinstance(stage.get("attempted_at"), str)
                or not stage["attempted_at"]
                or stage["budget"]["reserved_microusd"] != expected["budget"]["ceiling_microusd"]
                or (
                    previous is not None
                    and (
                        previous.get("result") is None
                        or previous.get("error") is not None
                        or not previous.get("completed_at")
                    )
                )
            ):
                raise ValueError("Combined stage began without its prerequisite or durable claim")
        if state == "response_received":
            if (
                raw[name] is None
                or not isinstance(stage.get("response_received_at"), str)
                or not stage["response_received_at"]
            ):
                raise ValueError("Combined response receipt needs its saved raw evidence")
        elif stage.get("response_received_at") is not None or stage.get("result") is not None:
            raise ValueError("Incomplete combined stages cannot publish normalized results")
        if name == "planning":
            if stage.get("request") != expected["request"]:
                raise ValueError("Combined planning request differs from its approved input")
        elif state != "not_started":
            if name == "grounding":
                if stage.get("request") != _grounding_input(stages, request, plan):
                    raise ValueError("SAM grounding request differs from saved planning prompts")
            else:
                sent = stage.get("request")
                validate_input(sent, provider, stage="review")
                if not _digest(sent.get("request_sha256")):
                    raise ValueError("Review requires the checksum of its actual prepared POST")
                expected_input = reconstruct_review_input(
                    provider, image, stages["planning"]["result"], stages["grounding"]["result"]
                )
                if {
                    key: value for key, value in sent.items() if key != "request_sha256"
                } != expected_input:
                    raise ValueError(
                        "Combined reviewer input differs from saved candidate evidence"
                    )
                if (
                    _micros(sent["estimate"]["upper_bound_usd"])
                    > stage["budget"]["ceiling_microusd"]
                ):
                    raise ValueError("The dynamic review request exceeds its approved allowance")
        elif stage.get("request") is not None:
            raise ValueError("An unstarted dynamic stage cannot claim an outgoing request")
        if raw[name] is not None:
            _json(raw[name], limit=_RAW_LIMIT if stage["external"] else _RESULT_LIMIT)
        if stage.get("result") is not None:
            if (
                stage.get("error") is not None
                or not isinstance(stage.get("completed_at"), str)
                or not stage["completed_at"]
            ):
                raise ValueError("A completed stage must be successful and timestamped")
            try:
                expected_result = _normalized(name, raw[name], provider, stages, image)
            except Exception as exc:
                raise ValueError(
                    "Combined normalized result has invalid raw stage evidence"
                ) from exc
            if stage["result"] != expected_result:
                raise ValueError("Combined normalized result differs from its saved raw evidence")
            if (
                name == "grounding"
                and raw[name].get("metadata", {}).get("runtime_identity")
                != plan["runtime_identity"]
            ):
                raise ValueError("SAM grounding runtime identity differs from the approved plan")
        elif stage.get("completed_at") is not None:
            raise ValueError("An incomplete stage cannot claim normalization completion")
        previous = stage
    if row.get("result") is not None:
        if (
            metadata.get("state") != "ready"
            or row.get("error") is not None
            or stages["review"].get("result") != row["result"]
        ):
            raise ValueError("Combined final proposals differ from the completed review")
    elif metadata.get("state") == "ready":
        raise ValueError("A ready combined pipeline requires final saved proposals")
    _json(metadata)
    return pipeline


def _trial(conn, trial_id):
    trial = _decode(
        conn.execute("SELECT * FROM benchmark_trials WHERE id=?", (trial_id,)).fetchone()
    )
    if trial is None:
        raise ValueError("Combined benchmark trial not found")
    job = _decode(conn.execute("SELECT * FROM jobs WHERE id=?", (trial["job_id"],)).fetchone())
    config = _decode(
        conn.execute("SELECT * FROM benchmark_configs WHERE id=?", (trial["config_id"],)).fetchone()
    )
    if (
        job is None
        or job["kind"] != "benchmark"
        or job["params"].get("trial_id") != trial_id
        or config is None
        or config["benchmark_id"] != trial["benchmark_id"]
        or trial["config"].get("source_config_fingerprint") != config["fingerprint"]
    ):
        raise DispatchConflict("Combined benchmark job or configuration ownership changed")
    validate_external_trial(
        trial["config"],
        config["config"],
        [{"frame_id": identifier} for identifier in trial["config"].get("frame_ids", [])],
    )
    return trial, job


def _output(conn, output_id, attempt_id):
    row = _decode(
        conn.execute("SELECT * FROM benchmark_outputs WHERE id=?", (output_id,)).fetchone()
    )
    if row is None:
        raise ValueError("Combined benchmark output not found")
    trial, job = _trial(conn, row["trial_id"])
    if (
        not isinstance(job.get("result"), dict)
        or job["result"].get("benchmark_attempt_id") != attempt_id
    ):
        raise DispatchConflict("This attempt does not own the combined benchmark trial")
    pipeline = validate_output_row(row, trial, validated_plan=trial["config"]["external_plan"])
    if any(
        stage["attempt_id"] is not None and stage["attempt_id"] != attempt_id
        for stage in pipeline["stages"].values()
    ):
        raise DispatchConflict("Saved combined stages belong to a different job attempt")
    return row, trial, job


def _stage(row, name):
    if name not in STAGES:
        raise ValueError("Choose planning, grounding or review")
    return row["metadata"]["pipeline"]["stages"][name]


def _metadata(stage, incoming):
    if incoming is not None:
        if not isinstance(incoming, dict):
            raise ValueError("Stage metadata must be an object")
        _json(incoming)
        stage["metadata"].update(deepcopy(incoming))


def claim_trial(store: Store, trial_id: str):
    with store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        trial, job = _trial(conn, trial_id)
        result = job["result"] or {}
        if not _active(job) or not isinstance(result, dict) or result.get("benchmark_attempt_id"):
            raise DispatchConflict("This combined trial was already claimed or stopped")
        rows = [
            _decode(raw)
            for raw in conn.execute("SELECT * FROM benchmark_outputs WHERE trial_id=?", (trial_id,))
        ]
        if {row["frame_id"] for row in rows} != set(trial["config"]["frame_ids"]):
            raise DispatchConflict("Combined stage placeholders are incomplete")
        for row in rows:
            pipeline = validate_output_row(
                row, trial, validated_plan=trial["config"]["external_plan"]
            )
            if row["error"] is not None or any(
                stage["state"] != "not_started" or stage["attempt_id"]
                for stage in pipeline["stages"].values()
            ):
                raise DispatchConflict("A combined trial image was already attempted")
        attempt = new_id()
        result["benchmark_attempt_id"] = attempt
        conn.execute("UPDATE jobs SET result=? WHERE id=?", (_json(result), job["id"]))
    return attempt


def begin_stage(store, output_id, attempt_id, stage, *, request=None):
    with store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row, trial, job = _output(conn, output_id, attempt_id)
        target = _stage(row, stage)
        if (
            not _active(job)
            or row["error"] is not None
            or target["state"] != "not_started"
            or target["attempt_id"]
        ):
            raise DispatchConflict("This combined stage was already attempted or stopped")
        plan = trial["config"]["external_plan"]
        item = next(value for value in plan["requests"] if value["frame_id"] == row["frame_id"])
        stages = row["metadata"]["pipeline"]["stages"]
        if stage != "planning":
            previous = stages[STAGES[STAGES.index(stage) - 1]]
            if previous["result"] is None or previous["error"] is not None:
                raise DispatchConflict("Complete the preceding combined stage before advancing")
        if stage == "planning":
            if request is not None and request != target["request"]:
                raise ValueError("Planning transport input differs from its approved request")
        elif stage == "grounding":
            expected = _grounding_input(stages, item, plan)
            if request is not None and request != expected:
                raise ValueError("Grounding input differs from the saved planning stage")
            target["request"] = expected
        else:
            if request is None:
                raise ValueError("The exact dynamic review request must be saved before transport")
            target["request"] = deepcopy(request)
        reserved = 0
        for raw in conn.execute("SELECT * FROM benchmark_outputs WHERE trial_id=?", (trial["id"],)):
            other = _decode(raw)
            pipeline = validate_output_row(other, trial, validated_plan=plan)
            if other["error"] is not None:
                raise DispatchConflict("A failed combined image prevents further trial stages")
            reserved += sum(
                value["budget"]["reserved_microusd"] for value in pipeline["stages"].values()
            )
        amount = target["budget"]["ceiling_microusd"]
        if reserved + amount > plan["approval"]["budget_microusd"]:
            raise DispatchConflict("The approved combined planning budget is exhausted")
        target["budget"]["reserved_microusd"] = amount
        target.update(
            state="dispatching" if target["external"] else "running",
            attempt_id=attempt_id,
            attempted_at=now(),
        )
        row["metadata"]["state"] = stage
        validate_output_row(row, trial, validated_plan=plan)
        _write(conn, row)
    return row


def _save_response(row, attempt_id, stage, raw, metadata):
    target = _stage(row, stage)
    if target["attempt_id"] != attempt_id or target["state"] == "not_started":
        raise DispatchConflict("No started stage belongs to this combined attempt")
    if raw is None:
        raise ValueError("Complete stage responses require raw evidence")
    _json(raw, limit=_RAW_LIMIT if target["external"] else _RESULT_LIMIT)
    if target["state"] == "response_received" and row["raw_response"][stage] != raw:
        raise DispatchConflict("A saved combined stage response cannot be replaced")
    _metadata(target, metadata)
    row["raw_response"][stage] = deepcopy(raw)
    target["state"] = "response_received"
    target["response_received_at"] = target["response_received_at"] or now()
    if row["result"] is None and row["error"] is None:
        row["metadata"]["state"] = "raw_saved"


def save_stage_response(store, output_id, attempt_id, stage, raw, metadata=None):
    with store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row, _, _ = _output(conn, output_id, attempt_id)
        _save_response(row, attempt_id, stage, raw, metadata)
        _write(conn, row)
    return row


def complete_stage(store, output_id, attempt_id, stage, result, metadata=None):
    if stage not in STAGES:
        raise ValueError("Complete planning, grounding or review")
    _json(result, limit=_RESULT_LIMIT)
    with store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row, trial, job = _output(conn, output_id, attempt_id)
        target = _stage(row, stage)
        if (
            not _active(job)
            or row["error"] is not None
            or target["state"] != "response_received"
            or target["attempt_id"] != attempt_id
        ):
            raise DispatchConflict(
                "This combined stage cannot complete after failure or cancellation"
            )
        if target["result"] is not None:
            if target["result"] == result:
                return row
            raise DispatchConflict("Completed combined stage results are immutable")
        _metadata(target, metadata)
        target.update(result=deepcopy(result), completed_at=now())
        validate_output_row(row, trial, validated_plan=trial["config"]["external_plan"])
        row["metadata"]["state"] = stage + "_complete"
        _write(conn, row)
    return row


def fail_stage(
    store, output_id, attempt_id, stage, error, raw=None, metadata=None, response_received=False
):
    with store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row, _, job = _output(conn, output_id, attempt_id)
        target = _stage(row, stage)
        if row["result"] is not None:
            raise DispatchConflict("Published combined results cannot be replaced by an error")
        if target["result"] is not None:
            # Cancellation between a completed stage and the next admission must
            # preserve that completed evidence while stopping the whole image.
            if raw is not None and raw != row["raw_response"][stage]:
                raise DispatchConflict("Completed stage evidence cannot be replaced")
            row["error"] = (str(error) or type(error).__name__)[:4000]
            row["metadata"]["state"] = "failed" if _active(job) else "cancelled"
            _write(conn, row)
            return row
        if response_received:
            _save_response(
                row,
                attempt_id,
                stage,
                raw if raw is not None else row["raw_response"][stage],
                metadata,
            )
        else:
            _metadata(target, metadata)
            if target["state"] == "not_started":
                if raw is not None:
                    raise ValueError("An unstarted combined stage cannot contain a response")
                target["attempt_id"] = attempt_id
            else:
                if target["attempt_id"] != attempt_id:
                    raise DispatchConflict("This attempt does not own the combined stage")
                if target["state"] in {"dispatching", "running"}:
                    target["state"] = "outcome_unknown" if target["external"] else "interrupted"
                if raw is not None and target["state"] != "response_received":
                    _json(raw, limit=_RAW_LIMIT if target["external"] else _RESULT_LIMIT)
                    row["raw_response"][stage] = deepcopy(raw)
        message = (str(error) or type(error).__name__)[:4000]
        target["error"] = message
        row["error"] = message
        row["metadata"]["state"] = "failed"
        _write(conn, row)
    return row


def publish_output(store, output_id, attempt_id, result, metadata=None):
    _json(result, limit=_RESULT_LIMIT)
    with store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row, trial, job = _output(conn, output_id, attempt_id)
        review = _stage(row, "review")
        if review["state"] != "response_received" or review["attempt_id"] != attempt_id:
            raise DispatchConflict("Final proposals require a saved complete review response")
        if row["result"] is not None:
            if row["result"] == result:
                return row
            raise DispatchConflict("Published combined proposals are immutable")
        if metadata is not None:
            if not isinstance(metadata, dict) or any(
                key in {"pipeline", "state"} or metadata[key] != row["metadata"].get(key)
                for key in metadata.keys() & _PROTECTED
            ):
                raise ValueError("Publication metadata cannot replace the combined stage journal")
            _json(metadata)
            row["metadata"].update(deepcopy(metadata))
        if not _active(job):
            row["error"] = "Trial stopped; stage responses saved without publishing proposals"
            row["metadata"]["state"] = "cancelled"
        elif row["error"] is not None or review["error"] is not None:
            raise DispatchConflict("A failed combined image cannot publish proposals")
        else:
            if review["result"] is not None and review["result"] != result:
                raise DispatchConflict("Completed review results are immutable")
            review.update(result=deepcopy(result), completed_at=review["completed_at"] or now())
            row.update(result=deepcopy(result))
            row["metadata"]["state"] = "ready"
            validate_output_row(row, trial, validated_plan=trial["config"]["external_plan"])
        _write(conn, row)
    return row


def record_image_timing(store, output_id, attempt_id, elapsed_ms):
    if type(elapsed_ms) not in (int, float) or not math.isfinite(elapsed_ms) or elapsed_ms < 0:
        raise ValueError("Combined image elapsed time must be finite nonnegative milliseconds")
    with store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row, _, _ = _output(conn, output_id, attempt_id)
        row["metadata"].setdefault("timing", {})["elapsed_ms"] = float(elapsed_ms)
        _write(conn, row)
    return row


def recover_combined_dispatches(store):
    with store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        for raw in conn.execute(
            "SELECT o.* FROM benchmark_outputs o JOIN benchmark_trials t ON t.id=o.trial_id "
            "JOIN jobs j ON j.id=t.job_id WHERE j.status NOT IN ('queued','running') "
            "AND json_extract(t.config,'$.external_plan.protocol')=?",
            (PLAN_PROTOCOL,),
        ).fetchall():
            row = _decode(raw)
            trial, _ = _trial(conn, row["trial_id"])
            pipeline = validate_output_row(
                row, trial, validated_plan=trial["config"]["external_plan"]
            )
            changed = False
            for stage in pipeline["stages"].values():
                if stage["state"] in {"dispatching", "running"}:
                    stage["state"] = "outcome_unknown" if stage["external"] else "interrupted"
                    stage["error"] = "Server stopped during this stage; no automatic retry"
                    row["error"] = stage["error"]
                    row["metadata"]["state"] = "failed"
                    changed = True
            if changed:
                _write(conn, row)


def dispatch_summary(store, trial_id):
    with store.connect() as conn:
        trial = _decode(
            conn.execute("SELECT * FROM benchmark_trials WHERE id=?", (trial_id,)).fetchone()
        )
        if (
            trial is None
            or trial["config"].get("external_plan", {}).get("protocol") != PLAN_PROTOCOL
        ):
            return None
        trial, job = _trial(conn, trial_id)
        plan = trial["config"]["external_plan"]
        rows = [
            _decode(raw)
            for raw in conn.execute(
                "SELECT * FROM benchmark_outputs WHERE trial_id=? ORDER BY created_at,id",
                (trial_id,),
            )
        ]
    counts = dict.fromkeys(sorted(EXTERNAL_STATES), 0)
    outputs, costs, missing, reserved = [], [], 0, 0
    for row in rows:
        pipeline = validate_output_row(row, trial, validated_plan=plan)
        for name in EXTERNAL_STAGES:
            stage = pipeline["stages"][name]
            state = stage["state"]
            if state == "dispatching" and not _active(job):
                state = "outcome_unknown"
            counts[state] += 1
            reserved += stage["budget"]["reserved_microusd"]
            cost = stage["metadata"].get("usage_cost_usd")
            known = (
                type(cost) in (int, float)
                and math.isfinite(cost)
                and cost >= 0
                and state == "response_received"
            )
            if state != "not_started":
                if known:
                    costs.append(Decimal(str(cost)))
                else:
                    missing += 1
            outputs.append(
                {
                    "output_id": row["id"],
                    "frame_id": row["frame_id"],
                    "stage": name,
                    "state": state,
                    "reserved_microusd": stage["budget"]["reserved_microusd"],
                    "request_id": stage["metadata"].get("request_id"),
                    "usage": stage["metadata"].get("usage"),
                    "usage_cost_usd": cost if known else None,
                }
            )
    state = next(
        (name for name in ("outcome_unknown", "dispatching", "response_received") if counts[name]),
        "not_started",
    )
    total = float(sum(costs, Decimal(0))) if costs else None
    approval = plan["approval"]
    return {
        "protocol": DISPATCH_PROTOCOL,
        "provider": "openai",
        "model": plan["model"],
        "external": True,
        "state": state,
        "counts": counts,
        "outputs": outputs,
        "budget_microusd": approval["budget_microusd"],
        "reserved_microusd": reserved,
        "remaining_planned_microusd": max(0, approval["budget_microusd"] - reserved),
        "estimated_ceiling_microusd": approval["estimated_ceiling_microusd"],
        "usage_cost_usd": total if not missing else None,
        "known_usage_cost_usd": total,
        "usage_missing_count": missing,
        "unknown_outcome_count": counts["outcome_unknown"],
        "message": "At most two external requests per image; reservations are planning allowances, "
        "not billing caps. Missing usage and unknown outcomes are never treated as zero cost.",
    }
