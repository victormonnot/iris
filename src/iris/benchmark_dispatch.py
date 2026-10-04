"""One-attempt, per-image external benchmark receipts and planning reservations.

The ledger is committed before transport and kept independently from model output
validity. Reservations are conservative planning allowances, never provider invoices.
No helper in this module performs a network request or retries a generation.
"""

from __future__ import annotations

import json
import math
import re
from copy import deepcopy
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal

from iris.job_dispatch import DispatchConflict
from iris.store import Store, _decode, _encode, new_id, now

PLAN_PROTOCOL = "iris-multimodal-trial-v1"
DISPATCH_PROTOCOL = "iris-benchmark-dispatch-v1"
STATES = {"not_started", "dispatching", "response_received", "outcome_unknown"}
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_PROTECTED = {"dispatch", "budget", "input", "state"}
_MAX_PLAN_BYTES = 16 * 1024 * 1024
_MAX_RESPONSE_BYTES = 2 * 1024 * 1024
# A bounded HTTP body can grow when preserved inside a JSON error envelope:
# control characters need six bytes each, plus the envelope and formatting.
_MAX_RAW_JSON_BYTES = 16 * 1024 * 1024


def _json(value, *, limit=_MAX_PLAN_BYTES):
    try:
        encoded = json.dumps(value, allow_nan=False, ensure_ascii=False)
    except (ValueError, TypeError, RecursionError) as exc:
        raise ValueError("Benchmark evidence must contain finite JSON") from exc
    if len(encoded.encode()) > limit:
        raise ValueError("Benchmark evidence exceeds its saved size limit")
    return encoded


def _micros(value, rounding=ROUND_CEILING):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError("A positive finite USD planning allowance is required")
    return int((Decimal(str(value)) * 1_000_000).to_integral_value(rounding=rounding))


def _digest(value):
    return isinstance(value, str) and _SHA.fullmatch(value) is not None


def _plan(frozen):
    if not isinstance(frozen, dict):
        raise ValueError("The external benchmark trial must have a frozen plan")
    plan = frozen.get("external_plan")
    if (
        not isinstance(plan, dict)
        or plan.get("protocol") != PLAN_PROTOCOL
        or plan.get("provider") != "openai"
        or not isinstance(plan.get("model"), str)
        or not 1 <= len(plan["model"]) <= 128
        or not isinstance(plan.get("requests"), list)
        or not 1 <= len(plan["requests"]) <= 25
    ):
        raise ValueError("The external benchmark plan has an unsupported format")
    _json(plan)
    seen, ceiling = set(), 0
    for request in plan["requests"]:
        if not isinstance(request, dict):
            raise ValueError("Every external request needs its frozen image and estimate")
        identifier, prepared, estimate = (
            request.get("frame_id"),
            request.get("input"),
            request.get("estimate"),
        )
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError("External request frame IDs must be distinct")
        seen.add(identifier)
        if (
            not isinstance(prepared, dict)
            or set(prepared) != {"image", "prompt", "request_sha256"}
            or not _digest(prepared.get("request_sha256"))
            or not isinstance(prepared.get("prompt"), str)
            or not prepared["prompt"].strip()
            or not isinstance(prepared.get("image"), dict)
        ):
            raise ValueError("External requests must save only image identity, prompt and checksum")
        image = prepared["image"]
        if not _digest(image.get("sha256")) or any(
            type(image.get(key)) is not int or image[key] <= 0 for key in ("width", "height")
        ):
            raise ValueError("The transmitted image needs a checksum and positive dimensions")
        if not isinstance(image.get("encoding"), (str, dict)) or not image["encoding"]:
            raise ValueError("The transmitted image encoding must be recorded")
        for key in ("original_width", "original_height", "sent_width", "sent_height", "bytes"):
            if key in image and (type(image[key]) is not int or image[key] <= 0):
                raise ValueError("Transmitted image sizes must be positive integers")
        if "source_pixel_sha256" in image and not _digest(image["source_pixel_sha256"]):
            raise ValueError("The transmitted image source checksum is invalid")
        if not isinstance(estimate, dict) or estimate.get("currency") != "USD":
            raise ValueError("External request estimates must use USD")
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
        raise ValueError("The approved benchmark budget differs from its per-image allowances")
    return plan


def validate_external_trial(frozen: dict, config: dict, frames: list[dict]) -> dict:
    """Validate a saved external plan without consulting credentials or current prices.

    ``config`` is the candidate configuration payload, not its database wrapper.
    ``frames`` is the ordered manifest partition selected by the trial.
    """
    plan = _plan(frozen)
    provider = config.get("provider_config") if isinstance(config, dict) else None
    if (
        not isinstance(provider, dict)
        or config.get("approach") != "multimodal"
        or provider.get("provider") != plan["provider"]
        or provider.get("model") != plan["model"]
        or frozen.get("candidate_config") != config
        or frozen.get("frame_ids") != [frame["frame_id"] for frame in frames]
        or [request["frame_id"] for request in plan["requests"]] != frozen["frame_ids"]
    ):
        raise ValueError("External benchmark requests differ from the frozen trial inputs")
    return plan


def initialize_outputs(conn, trial: dict, frames: list[dict], external_plan: dict):
    """Create all known-unsent receipts inside the trial/job creation transaction."""
    if trial["config"].get("external_plan") != external_plan:
        raise ValueError("The external plan must be stored in the trial before initialization")
    validate_external_trial(trial["config"], trial["config"]["candidate_config"], frames)
    identifiers = []
    for request in external_plan["requests"]:
        metadata = {
            "state": "not_started",
            "dispatch": {
                "protocol": DISPATCH_PROTOCOL,
                "provider": external_plan["provider"],
                "model": external_plan["model"],
                "external": True,
                "state": "not_started",
                "attempt_id": None,
                "attempted_at": None,
                "response_received_at": None,
            },
            "budget": {
                "currency": "USD",
                "ceiling_microusd": _micros(request["estimate"]["upper_bound_usd"]),
                "reserved_microusd": 0,
            },
            "input": deepcopy(request["input"]),
            "timing": {},
        }
        identifier = new_id()
        conn.execute(
            "INSERT INTO benchmark_outputs (id,trial_id,frame_id,metadata,created_at) "
            "VALUES (?,?,?,?,?)",
            (identifier, trial["id"], request["frame_id"], _json(metadata), now()),
        )
        identifiers.append(identifier)
    return identifiers


def validate_output_row(row: dict, trial: dict, *, validated_plan=None) -> dict:
    """Check the per-image journal against its immutable request and approved budget."""
    plan = _plan(trial["config"]) if validated_plan is None else validated_plan
    request = next((item for item in plan["requests"] if item["frame_id"] == row["frame_id"]), None)
    metadata = row.get("metadata")
    dispatch = metadata.get("dispatch") if isinstance(metadata, dict) else None
    budget = metadata.get("budget") if isinstance(metadata, dict) else None
    if (
        request is None
        or row.get("trial_id") != trial["id"]
        or not isinstance(dispatch, dict)
        or dispatch.get("protocol") != DISPATCH_PROTOCOL
        or dispatch.get("provider") != plan["provider"]
        or dispatch.get("model") != plan["model"]
        or dispatch.get("external") is not True
        or dispatch.get("state") not in STATES
        or metadata.get("input") != request["input"]
        or not isinstance(budget, dict)
        or budget.get("currency") != "USD"
        or type(budget.get("ceiling_microusd")) is not int
        or budget["ceiling_microusd"] != _micros(request["estimate"]["upper_bound_usd"])
        or type(budget.get("reserved_microusd")) is not int
        or budget["reserved_microusd"] not in {0, budget["ceiling_microusd"]}
    ):
        raise ValueError("External benchmark receipt differs from its frozen request")
    state = dispatch["state"]
    attempt = dispatch.get("attempt_id")
    if attempt is not None and (not isinstance(attempt, str) or not attempt):
        raise ValueError("External benchmark attempt identity is invalid")
    if state == "not_started":
        if (
            budget["reserved_microusd"] != 0
            or dispatch.get("attempted_at") is not None
            or dispatch.get("response_received_at") is not None
            or row.get("raw_response") is not None
            or row.get("result") is not None
        ):
            raise ValueError("An unsent benchmark image cannot have a reservation or response")
    elif (
        not attempt
        or not isinstance(dispatch.get("attempted_at"), str)
        or not dispatch["attempted_at"]
        or budget["reserved_microusd"] != budget["ceiling_microusd"]
    ):
        raise ValueError("An attempted benchmark request needs its durable reservation")
    if state == "response_received":
        if (
            row.get("raw_response") is None
            or not isinstance(dispatch.get("response_received_at"), str)
            or not dispatch["response_received_at"]
        ):
            raise ValueError("A complete benchmark response needs saved raw evidence")
    elif dispatch.get("response_received_at") is not None or row.get("result") is not None:
        raise ValueError("Unknown or unsent requests cannot have validated proposals")
    _json(metadata)
    return dispatch


def _trial(conn, trial_id):
    trial = _decode(
        conn.execute("SELECT * FROM benchmark_trials WHERE id=?", (trial_id,)).fetchone()
    )
    if trial is None:
        raise ValueError("External benchmark trial not found")
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
        raise DispatchConflict("External benchmark job or configuration ownership changed")
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
        raise ValueError("External benchmark output not found")
    trial, job = _trial(conn, row["trial_id"])
    if (
        not isinstance(job.get("result"), dict)
        or job["result"].get("benchmark_attempt_id") != attempt_id
    ):
        raise DispatchConflict("This external benchmark attempt does not own the trial")
    validate_output_row(row, trial)
    return row, trial, job


def _write(conn, row):
    encoded = _encode({key: row[key] for key in ("metadata", "raw_response", "result", "error")})
    conn.execute(
        "UPDATE benchmark_outputs SET metadata=?,raw_response=?,result=?,error=? WHERE id=?",
        (*(encoded[key] for key in ("metadata", "raw_response", "result", "error")), row["id"]),
    )


def _metadata(row, incoming):
    if incoming is None:
        return
    if not isinstance(incoming, dict) or incoming.keys() & _PROTECTED:
        raise ValueError("Provider metadata cannot replace the benchmark dispatch journal")
    _json(incoming)
    row["metadata"].update(deepcopy(incoming))


def _active(job):
    return job["status"] in {"queued", "running"} and not job["cancel_requested"]


def claim_trial(store: Store, trial_id: str) -> str:
    with store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        trial, job = _trial(conn, trial_id)
        result = job["result"] or {}
        if not _active(job) or not isinstance(result, dict) or result.get("benchmark_attempt_id"):
            raise DispatchConflict("This external benchmark trial was already attempted or stopped")
        outputs = [
            _decode(row)
            for row in conn.execute("SELECT * FROM benchmark_outputs WHERE trial_id=?", (trial_id,))
        ]
        if {row["frame_id"] for row in outputs} != set(trial["config"]["frame_ids"]):
            raise DispatchConflict("External benchmark request receipts are incomplete")
        for row in outputs:
            dispatch = validate_output_row(
                row, trial, validated_plan=trial["config"]["external_plan"]
            )
            if (
                dispatch["state"] != "not_started"
                or dispatch.get("attempt_id")
                or row["error"] is not None
            ):
                raise DispatchConflict("An external benchmark image was already attempted")
        attempt = new_id()
        result["benchmark_attempt_id"] = attempt
        conn.execute("UPDATE jobs SET result=? WHERE id=?", (_json(result), job["id"]))
    return attempt


def begin_dispatch(store: Store, output_id: str, attempt_id: str):
    """Reserve one allowance and mark the attempt before entering HTTP transport."""
    with store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row, trial, job = _output(conn, output_id, attempt_id)
        dispatch, budget = row["metadata"]["dispatch"], row["metadata"]["budget"]
        if (
            not _active(job)
            or dispatch["state"] != "not_started"
            or dispatch.get("attempt_id")
            or row["error"] is not None
        ):
            raise DispatchConflict("This benchmark image was already attempted or stopped")
        reserved = 0
        for raw in conn.execute("SELECT * FROM benchmark_outputs WHERE trial_id=?", (trial["id"],)):
            other = _decode(raw)
            validate_output_row(other, trial, validated_plan=trial["config"]["external_plan"])
            reserved += other["metadata"]["budget"]["reserved_microusd"]
        cap = trial["config"]["external_plan"]["approval"]["budget_microusd"]
        if reserved + budget["ceiling_microusd"] > cap:
            raise DispatchConflict("The approved benchmark planning budget is exhausted")
        budget["reserved_microusd"] = budget["ceiling_microusd"]
        dispatch.update(state="dispatching", attempt_id=attempt_id, attempted_at=now())
        row["metadata"]["state"] = "dispatching"
        _write(conn, row)
    return row


def _owned_response(row, attempt_id):
    dispatch = row["metadata"]["dispatch"]
    if dispatch.get("attempt_id") != attempt_id or dispatch["state"] == "not_started":
        raise DispatchConflict("No dispatched request belongs to this benchmark attempt")
    return dispatch


def _response(row, attempt_id, raw, metadata):
    dispatch = _owned_response(row, attempt_id)
    if raw is None:
        raise ValueError("A complete response needs raw evidence, including invalid JSON envelopes")
    _json(raw, limit=_MAX_RAW_JSON_BYTES)
    if dispatch["state"] == "response_received" and row["raw_response"] != raw:
        raise DispatchConflict("A saved provider response cannot be replaced")
    _metadata(row, metadata)
    row["raw_response"] = deepcopy(raw)
    dispatch.update(state="response_received")
    if dispatch.get("response_received_at") is None:
        dispatch["response_received_at"] = now()
    request_id = row["metadata"].get("request_id")
    if isinstance(request_id, str) and request_id:
        dispatch["request_id"] = request_id
    if row["result"] is None and row["error"] is None:
        row["metadata"]["state"] = "response_received"


def save_response(store: Store, output_id: str, attempt_id: str, raw, metadata):
    """Keep a complete response and receipt together, including a late response."""
    with store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row, _, _ = _output(conn, output_id, attempt_id)
        _response(row, attempt_id, raw, metadata)
        _write(conn, row)
    return row


def fail_output(
    store: Store,
    output_id: str,
    attempt_id: str,
    error,
    raw=None,
    metadata=None,
    response_received=False,
):
    with store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row, _, _ = _output(conn, output_id, attempt_id)
        if row["result"] is not None:
            raise DispatchConflict("Published benchmark proposals cannot be replaced by an error")
        message = str(error) or type(error).__name__
        if response_received:
            _response(row, attempt_id, raw if raw is not None else row["raw_response"], metadata)
        else:
            _metadata(row, metadata)
            dispatch = row["metadata"]["dispatch"]
            if dispatch["state"] == "not_started":
                dispatch["attempt_id"] = attempt_id
                if raw is not None:
                    raise ValueError("An unsent request cannot contain a provider response")
            else:
                _owned_response(row, attempt_id)
                if dispatch["state"] == "dispatching":
                    dispatch["state"] = "outcome_unknown"
                if raw is not None and dispatch["state"] != "response_received":
                    _json(raw, limit=_MAX_RAW_JSON_BYTES)
                    row["raw_response"] = deepcopy(raw)
        row["error"] = message[:4000]
        row["metadata"]["state"] = "failed"
        _write(conn, row)
    return row


def publish_output(store: Store, output_id: str, attempt_id: str, result, metadata):
    """Publish validated proposals only while the owning trial is still active."""
    if not isinstance(result, dict) or not isinstance(result.get("proposals"), list):
        raise ValueError("Validated benchmark proposals must be an object containing a list")
    _json(result, limit=_MAX_RESPONSE_BYTES)
    with store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row, _, job = _output(conn, output_id, attempt_id)
        dispatch = _owned_response(row, attempt_id)
        if dispatch["state"] != "response_received":
            raise DispatchConflict("Benchmark proposals require a saved complete provider response")
        if row["result"] is not None:
            if row["result"] == result:
                return row
            raise DispatchConflict("Published benchmark proposals are immutable")
        _metadata(row, metadata)
        if not _active(job):
            row["error"] = (
                "Trial stopped; the provider response was saved without publishing proposals"
            )
            row["metadata"]["state"] = "cancelled"
        elif row["error"] is not None:
            raise DispatchConflict("An invalid benchmark response cannot publish proposals")
        else:
            row["result"] = deepcopy(result)
            row["metadata"]["state"] = "ready"
        _write(conn, row)
    return row


def recover_benchmark_dispatches(store: Store):
    """Classify interrupted sends conservatively; never send, release, or resume them."""
    with store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        rows = conn.execute(
            "SELECT o.* FROM benchmark_outputs o JOIN benchmark_trials t ON t.id=o.trial_id "
            "JOIN jobs j ON j.id=t.job_id WHERE j.status NOT IN ('queued','running') "
            "AND json_extract(t.config,'$.external_plan.protocol')=?",
            (PLAN_PROTOCOL,),
        ).fetchall()
        for raw in rows:
            row = _decode(raw)
            trial, _ = _trial(conn, row["trial_id"])
            dispatch = validate_output_row(row, trial)
            if dispatch["state"] == "dispatching":
                dispatch["state"] = "outcome_unknown"
                row["metadata"]["state"] = "failed"
                row["error"] = (
                    "Server stopped after dispatch; provider outcome and possible charge "
                    "are unknown"
                )
                _write(conn, row)


def dispatch_summary(store: Store, trial_id: str) -> dict | None:
    with store.connect() as conn:
        trial = _decode(
            conn.execute("SELECT * FROM benchmark_trials WHERE id=?", (trial_id,)).fetchone()
        )
        if trial is None or not isinstance(trial["config"].get("external_plan"), dict):
            return None
        trial, job = _trial(conn, trial_id)
        plan = _plan(trial["config"])
        rows = [
            _decode(row)
            for row in conn.execute(
                "SELECT * FROM benchmark_outputs WHERE trial_id=? ORDER BY created_at,id",
                (trial_id,),
            )
        ]
    counts = dict.fromkeys(sorted(STATES), 0)
    outputs, costs, missing, reserved = [], [], 0, 0
    for row in rows:
        dispatch = validate_output_row(row, trial, validated_plan=plan)
        state = dispatch["state"]
        if state == "dispatching" and not _active(job):
            state = "outcome_unknown"
        counts[state] += 1
        amount = row["metadata"]["budget"]["reserved_microusd"]
        reserved += amount
        cost = row["metadata"].get("usage_cost_usd")
        known = type(cost) in (int, float) and math.isfinite(cost) and cost >= 0
        if state != "not_started":
            if state == "response_received" and known:
                costs.append(Decimal(str(cost)))
            else:
                missing += 1
        outputs.append(
            {
                "output_id": row["id"],
                "frame_id": row["frame_id"],
                "state": state,
                "reserved_microusd": amount,
                "request_id": dispatch.get("request_id"),
                "usage": row["metadata"].get("usage"),
                "usage_cost_usd": cost if state == "response_received" and known else None,
            }
        )
    state = (
        "outcome_unknown"
        if counts["outcome_unknown"]
        else "dispatching"
        if counts["dispatching"]
        else "response_received"
        if counts["response_received"]
        else "not_started"
    )
    total = float(sum(costs, Decimal(0))) if costs else None
    approval = plan["approval"]
    return {
        "protocol": DISPATCH_PROTOCOL,
        "provider": plan["provider"],
        "model": plan["model"],
        "external": True,
        "state": state,
        "counts": counts,
        "budget_microusd": approval["budget_microusd"],
        "reserved_microusd": reserved,
        "remaining_planned_microusd": max(0, approval["budget_microusd"] - reserved),
        "estimated_ceiling_microusd": approval["estimated_ceiling_microusd"],
        "usage_cost_usd": total if not missing else None,
        "known_usage_cost_usd": total,
        "usage_missing_count": missing,
        "unknown_outcome_count": counts["outcome_unknown"],
        "outputs": outputs,
        "message": "Reservations are planning allowances, not guaranteed billing caps. "
        "Reported usage at saved prices is not an invoice; unknown outcomes are not zero cost.",
    }
