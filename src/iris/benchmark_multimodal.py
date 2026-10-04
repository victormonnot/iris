"""Explicit external benchmark plans, consent receipts and reference-free execution.

All previews stay local. Only the executor passes prepared pixels and the frozen
provider configuration to the adapter, after durable dispatch admission.
"""

import base64
import hashlib
import hmac
import math
import secrets
import time
from datetime import UTC, datetime
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal

from iris.benchmark import (
    PROTOCOL,
    SCORING,
    BenchmarkConflict,
    _digest,
    load_benchmark_manifest,
    open_benchmark_image,
)
from iris.store import now

EXTERNAL_PROTOCOL = "iris-multimodal-trial-v1"
PREVIEW_LIFETIME_SECONDS = 600
_PREVIEW_SECRET = secrets.token_bytes(32)
MULTIMODAL_SCORING = {
    **SCORING,
    "scope": "All multimodal proposed boxes; confidence unavailable, no score threshold or AP",
}


def provider_catalog():
    from iris.multimodal_provider import provider_status

    return {
        "multimodal": provider_status(),
        "multimodal_settings": {
            "defaults": {
                "image_long_edge": 1536,
                "reasoning_effort": "low",
                "max_output_tokens": 4096,
            },
            "image_long_edges": [512, 1024, 1536, 2048],
            "reasoning_efforts": ["low", "medium", "high", "xhigh", "max"],
            "max_output_tokens": {"min": 1024, "max": 8192},
            "detail": "original",
        },
    }


def preview_config(store, benchmark_id, *, model_id, multimodal=None):
    from iris.multimodal_provider import freeze_config, provider_status

    benchmark = store.get("benchmarks", benchmark_id)
    if benchmark is None:
        raise KeyError(benchmark_id)
    if benchmark["status"] != "tuning":
        raise BenchmarkConflict("This benchmark's configurations are already locked")
    settings = {} if multimodal is None else multimodal
    if not isinstance(settings, dict) or set(settings) - {
        "image_long_edge",
        "reasoning_effort",
        "max_output_tokens",
    }:
        raise ValueError("Unknown multimodal configuration setting")
    manifest = load_benchmark_manifest(store, benchmark_id)
    provider = freeze_config(manifest["taxonomy"], model=model_id, **settings)
    config = {
        "protocol": PROTOCOL,
        "approach": "multimodal",
        "model_id": model_id,
        "model_name": "GPT-6 Astra · multimodal alone",
        "provider_config": provider,
        "reference_manifest_sha256": benchmark["manifest_sha256"],
        "scoring": MULTIMODAL_SCORING,
    }
    return {
        "benchmark_id": benchmark_id,
        "config": config,
        "fingerprint": _digest(config),
        "work": {
            role: {
                "image_count": sum(frame["role"] == role for frame in manifest["frames"]),
                "request_count": sum(frame["role"] == role for frame in manifest["frames"]),
            }
            for role in ("tuning", "evaluation")
        },
        "provider_status": provider_status(provider),
        "warnings": [
            "Configuration preparation is local; it does not verify account access or send images.",
            "Multimodal box localization may be inaccurate or incomplete. "
            "Human review is required.",
            "This model name is an API alias; "
            "record the returned model identity for each response.",
            "Reference labels and correction decisions are excluded from the candidate request.",
        ],
    }


def validate_config(config, manifest):
    from iris.multimodal_provider import validate_frozen_config

    provider = config.get("provider_config")
    validate_frozen_config(provider)
    if (
        provider["taxonomy"] != manifest["taxonomy"]
        or provider["model"] != config.get("model_id")
        or config.get("scoring") != MULTIMODAL_SCORING
    ):
        raise ValueError("Frozen multimodal classes, model or scoring protocol changed")
    return config


def _microusd(value, rounding=ROUND_CEILING):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("Costs must be finite nonnegative USD amounts")
    return int((Decimal(str(value)) * 1_000_000).to_integral_value(rounding=rounding))


def prepared_input(prepared):
    """An allowlist of inspectable evidence: no encoded image payload or credentials."""
    return {
        "image": prepared["image"],
        "prompt": prepared["prompt"],
        "request_sha256": prepared["request_sha256"],
    }


def prepare_plan(store, config, frames):
    from iris.multimodal_provider import prepare_request, provider_status

    settings = config["config"]["provider_config"]
    requests = []
    total_microusd = 0
    for frame in frames:
        with open_benchmark_image(store, frame) as image:
            prepared = prepare_request(image, settings)
        total_microusd += _microusd(prepared["estimate"]["upper_bound_usd"])
        requests.append(
            {
                "frame_id": frame["frame_id"],
                "input": prepared_input(prepared),
                "estimate": prepared["estimate"],
                "image_url": (
                    f"/api/benchmark-configs/{config['id']}/frames/{frame['frame_id']}/input-image"
                ),
            }
        )
    return {
        "protocol": EXTERNAL_PROTOCOL,
        "provider": "openai",
        "model": settings["model"],
        "requests": requests,
        "estimate": {
            "currency": "USD",
            "upper_bound_usd": total_microusd / 1_000_000,
            "estimated_ceiling_microusd": total_microusd,
            "estimate_label": "Conservative planning estimate for all approved requests",
            "basis": "Sum of per-image planning estimates, rounded up to micro-USD. "
            "An admission budget, not a guaranteed provider billing cap or invoice.",
        },
        "provider_status": provider_status(settings),
    }


def _signed_preview(fingerprint):
    expires = int(time.time()) + PREVIEW_LIFETIME_SECONDS
    body = f"{fingerprint}:{expires}".encode()
    signature = hmac.new(_PREVIEW_SECRET, body, hashlib.sha256).hexdigest()
    return base64.urlsafe_b64encode(body + b":" + signature.encode()).decode(), expires


def attach_preview(preview, plan):
    # Configuration presence is not model input; adding a key does not change
    # which pixels, definitions, prompt or prices the preview described.
    stable = {key: value for key, value in plan.items() if key != "provider_status"}
    preview["fingerprint"] = _digest({"trial": preview["fingerprint"], "external": stable})
    token, expires = _signed_preview(preview["fingerprint"])
    ready = plan["provider_status"]["status"] == "ready"
    return {
        **preview,
        "warnings": [
            *preview["warnings"],
            "The trial stops at the first failed request. Remaining images stay unsent; "
            "no automatic retry is performed.",
        ],
        "external_plan": plan,
        "preview_token": token,
        "expires_at": datetime.fromtimestamp(expires, UTC).isoformat(),
        "launch_allowed": ready,
        "launch_reason": plan["provider_status"]["reason"],
    }


def approve_plan(preview, *, approve_external, max_cost_usd, preview_token):
    if approve_external is not True:
        raise ValueError("Approve sending the listed images and class definitions to OpenAI")
    if not isinstance(preview_token, str) or len(preview_token) > 512:
        raise BenchmarkConflict("An unexpired external preview receipt is required")
    try:
        body, signature = base64.urlsafe_b64decode(preview_token.encode()).rsplit(b":", 1)
        fingerprint, expires = body.decode().split(":")
        expected = hmac.new(_PREVIEW_SECRET, body, hashlib.sha256).hexdigest()
        valid = hmac.compare_digest(signature.decode(), expected)
        valid = valid and fingerprint == preview["fingerprint"] and time.time() <= int(expires)
    except (ValueError, UnicodeError):
        valid = False
    if not valid:
        raise BenchmarkConflict("External preview expired or changed; inspect a fresh preview")
    if not preview["launch_allowed"]:
        raise ValueError(preview["launch_reason"])
    budget = _microusd(max_cost_usd, ROUND_FLOOR)
    plan = preview["external_plan"]
    ceiling = plan["estimate"]["estimated_ceiling_microusd"]
    if budget < ceiling or budget > 1000_000_000:
        raise ValueError("The USD planning budget must cover the preview and be at most 1000")
    return {
        **plan,
        "approval": {
            "approved_at": now(),
            "budget_usd": max_cost_usd,
            "budget_microusd": budget,
            "estimated_ceiling_microusd": ceiling,
            "fingerprint": preview["fingerprint"],
            "allow_external": True,
        },
    }


def input_image(store, config_id, frame_id):
    from iris.benchmark import benchmark_frame, validate_benchmark_config
    from iris.multimodal_provider import prepare_request

    config = store.get("benchmark_configs", config_id)
    if config is None or config["approach"] != "multimodal":
        raise KeyError(config_id)
    benchmark = store.get("benchmarks", config["benchmark_id"])
    manifest = load_benchmark_manifest(store, benchmark["id"])
    validate_benchmark_config(config, benchmark, manifest)
    frame = benchmark_frame(store, benchmark["id"], frame_id)
    with open_benchmark_image(store, frame) as image:
        prepared = prepare_request(image, config["config"]["provider_config"])
    return prepared["image_bytes"]


def run_trial(store, trial_id, progress, cancelled):
    """Execute each approved request at most once, stopping at the first failure."""
    from iris.benchmark import score_benchmark_outputs
    from iris.benchmark_dispatch import (
        begin_dispatch,
        claim_trial,
        fail_output,
        publish_output,
        save_response,
        validate_external_trial,
    )
    from iris.benchmark_runs import _context, _persist_result
    from iris.multimodal_provider import OpenAIPreannotator, ProviderResponseError
    from iris.preannotation_contracts import normalize_output

    trial = store.get("benchmark_trials", trial_id)
    if trial is None:
        raise KeyError(trial_id)
    benchmark, config, manifest = _context(
        store, trial["benchmark_id"], trial["config_id"], trial["split"]
    )
    frozen = trial["config"]
    frames = [frame for frame in manifest["frames"] if frame["role"] == trial["split"]]
    if (
        frozen.get("protocol") != PROTOCOL
        or frozen.get("source_config_fingerprint") != config["fingerprint"]
        or frozen.get("benchmark_manifest_sha256") != benchmark["manifest_sha256"]
        or frozen.get("candidate_config") != config["config"]
        or frozen.get("role") != trial["split"]
        or frozen.get("work") != {"image_count": len(frames), "request_count": len(frames)}
    ):
        raise ValueError("Frozen multimodal benchmark inputs changed")
    plan = validate_external_trial(frozen, config["config"], frames)
    prepared = prepare_plan(store, config, frames)
    if prepared["requests"] != plan["requests"] or prepared["estimate"] != plan["estimate"]:
        raise ValueError("Outgoing requests differ from the approved image and prompt preview")
    result = {
        "trial_id": trial_id,
        "frames_total": len(frames),
        "outputs_created": 0,
        "frames_ready": 0,
        "frames_issues": 0,
        "cancelled": False,
        "quality": None,
    }
    if cancelled():
        return {**result, "cancelled": True}
    attempt = result["benchmark_attempt_id"] = claim_trial(store, trial_id)
    _persist_result(store, trial, result)
    # The provider receives only this frozen allowlist, then copied pixels. It
    # never receives the store, manifest, reference boxes or human corrections.
    adapter = OpenAIPreannotator(config["config"]["provider_config"])
    outputs = {row["frame_id"]: row for row in store.list("benchmark_outputs", trial_id=trial_id)}
    for index, (frame, request) in enumerate(zip(frames, plan["requests"], strict=True)):
        if cancelled():
            result["cancelled"] = True
            break
        output_id = outputs[frame["frame_id"]]["id"]

        def before_dispatch(output_id=output_id):
            if cancelled():
                raise BenchmarkConflict("Trial stopped before external dispatch")
            begin_dispatch(store, output_id, attempt)

        def after_response(raw, metadata, output_id=output_id):
            save_response(
                store,
                output_id,
                attempt,
                raw,
                {**metadata, "timing": {"elapsed_ms": metadata.get("elapsed_ms")}},
            )

        adapter.before_dispatch = before_dispatch
        adapter.after_response = after_response
        progress(index / len(frames), f"Sending approved OpenAI image {index + 1} / {len(frames)}")
        try:
            with open_benchmark_image(store, frame) as image:
                response = adapter.propose(
                    image, expected_image_sha256=request["input"]["image"]["sha256"]
                )
            normalized = response["result"]
            if normalized != normalize_output(
                normalized["raw_output"],
                manifest["taxonomy"],
                width=frame["width"],
                height=frame["height"],
            ):
                raise ValueError("Provider normalization differs from the frozen output contract")
            metadata = {
                **response["metadata"],
                "timing": {"elapsed_ms": response["metadata"]["elapsed_ms"]},
            }
            if cancelled():
                row = fail_output(
                    store, output_id, attempt, "Trial stopped after receiving the response"
                )
                result["cancelled"] = True
            else:
                row = publish_output(store, output_id, attempt, normalized, metadata)
        except ProviderResponseError as exc:
            row = fail_output(
                store,
                output_id,
                attempt,
                str(exc),
                raw=exc.raw_response,
                metadata={**exc.metadata, "timing": {"elapsed_ms": exc.metadata.get("elapsed_ms")}},
                response_received=exc.response_received,
            )
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            row = fail_output(store, output_id, attempt, str(exc) or type(exc).__name__)
        result["outputs_created"] += 1
        result["frames_issues" if row["error"] is not None else "frames_ready"] += 1
        result["cancelled"] = (
            result["cancelled"] or cancelled() or row["metadata"]["state"] == "cancelled"
        )
        _persist_result(store, trial, result)
        progress(
            result["outputs_created"] / len(frames),
            f"Saved {result['outputs_created']} / {len(frames)} OpenAI image attempts",
        )
        if row["error"] is not None or result["cancelled"]:
            break
    result["quality"] = {
        **score_benchmark_outputs(
            manifest, trial["split"], store.list("benchmark_outputs", trial_id=trial_id)
        ),
        "protocol": MULTIMODAL_SCORING,
    }
    _persist_result(store, trial, result)
    return result
