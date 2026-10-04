"""Local SAM image benchmark orchestration; references stay outside the adapter."""

import json
import math
import time

from iris.benchmark import (
    PROTOCOL,
    SCORING,
    BenchmarkConflict,
    _digest,
    load_benchmark_manifest,
    open_benchmark_image,
    score_benchmark_outputs,
)
from iris.store import _decode, new_id, now

LOCAL_PROTOCOL = "iris-sam-trial-v1"
SAM_SCORING = {
    **SCORING,
    "scope": "Native SAM boxes above the frozen SAM score threshold; no mask metrics or AP",
}


def provider_catalog(store):
    from iris.sam_provider import provider_status

    return {
        "segmentation": provider_status(store.root),
        "segmentation_settings": {
            "defaults": {"threshold": 0.5, "device": "cuda"},
            "threshold": {"min": 0, "max": 1},
            "prompt_limits": {
                "min_length": 1,
                "max_length": 120,
                "max_tokens": 30,
                "max_classes": 100,
            },
            "devices": ["cuda"],
        },
    }


def work_plan(config, image_count):
    count = len(config["provider_config"]["prompts"])
    return {
        "image_count": image_count,
        "class_count": count,
        "image_encodings": image_count,
        "prompt_evaluations": image_count * count,
        "warmup_passes": 0,
    }


def preview_config(store, benchmark_id, *, model_id, segmentation=None):
    from iris.sam_provider import freeze_config, provider_status

    benchmark = store.get("benchmarks", benchmark_id)
    if benchmark is None:
        raise KeyError(benchmark_id)
    if benchmark["status"] != "tuning":
        raise BenchmarkConflict("This benchmark's configurations are already locked")
    settings = {} if segmentation is None else segmentation
    if not isinstance(settings, dict) or set(settings) - {"class_prompts", "threshold", "device"}:
        raise ValueError("Unknown SAM configuration setting")
    manifest = load_benchmark_manifest(store, benchmark_id)
    config = {
        "protocol": PROTOCOL,
        "approach": "segmentation",
        "model_id": model_id,
        "model_name": "SAM 3 · local text-to-box",
        "provider_config": freeze_config(manifest["taxonomy"], model=model_id, **settings),
        "reference_manifest_sha256": benchmark["manifest_sha256"],
        "scoring": SAM_SCORING,
    }
    return {
        "benchmark_id": benchmark_id,
        "config": config,
        "fingerprint": _digest(config),
        "work": {
            role: work_plan(config, sum(frame["role"] == role for frame in manifest["frames"]))
            for role in ("tuning", "evaluation")
        },
        "provider_status": provider_status(store.root, config["provider_config"]),
        "warnings": [
            "Preparation is local. Saving this profile does not install or load SAM weights.",
            "SAM uses the frozen short phrase for each class, not the full class definition.",
            "Only native boxes are produced; masks are disabled. SAM scores are not calibrated "
            "or comparable to scores from other providers.",
            "All image and prompt evaluations must succeed before quality is reported. "
            "No reference boxes or human corrections enter SAM.",
        ],
    }


def validate_config(config, manifest):
    from iris.sam_provider import validate_frozen_config

    provider = config.get("provider_config")
    validate_frozen_config(provider)
    if (
        provider["taxonomy"] != manifest["taxonomy"]
        or provider["model"] != config.get("model_id")
        or config.get("scoring") != SAM_SCORING
    ):
        raise ValueError("Frozen SAM classes, model or scoring protocol changed")
    return config


def attach_preview(store, preview, config):
    from iris.sam_provider import provider_status

    status = provider_status(store.root, config["provider_config"], force=True)
    plan = {
        "protocol": LOCAL_PROTOCOL,
        "runtime_identity": status.get("runtime", {}).get("identity"),
        "weight_sha256": config["provider_config"]["weights"]["sha256"],
        "work": preview["work"],
    }
    return {
        **preview,
        "fingerprint": _digest({"trial": preview["fingerprint"], "local_plan": plan}),
        "local_plan": plan,
        "provider_status": status,
        "launch_allowed": status["status"] == "ready",
        "launch_reason": status["reason"],
        "warnings": [
            *preview["warnings"],
            "Local SAM only: no hosted API, automatic download or CPU fallback. "
            "Model loading is recorded separately; image times include the first cold prediction.",
        ],
    }


def validate_trial(frozen, config, frames):
    from iris.sam_runtime import validate_runtime_identity

    plan = frozen.get("local_plan")
    if (
        not isinstance(plan, dict)
        or plan.get("protocol") != LOCAL_PROTOCOL
        or not isinstance(plan.get("runtime_identity"), dict)
        or not plan["runtime_identity"]
        or plan.get("weight_sha256") != config["provider_config"]["weights"]["sha256"]
        or plan.get("work") != work_plan(config, len(frames))
        or frozen.get("work") != plan["work"]
    ):
        raise ValueError("Frozen SAM trial runtime, weights or work changed")
    validate_runtime_identity(plan["runtime_identity"])
    return plan


def validate_saved_output(output, *, config, frame, plan, attempt):
    """Validate archived local evidence without loading or probing a model."""
    from iris.sam_provider import ProviderResponseError, normalize_response

    metadata = output.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("SAM output has no provenance")
    elapsed = metadata.get("timing", {}).get("elapsed_ms")
    state = metadata.get("state")
    if (
        not isinstance(attempt, str)
        or not attempt
        or metadata.get("attempt_id") != attempt
        or metadata.get("reference_withheld") is not True
        or metadata.get("frame_sha256") != frame["sha256"]
        or metadata.get("config_fingerprint") != config["fingerprint"]
        or metadata.get("runtime_identity") != plan["runtime_identity"]
        or metadata.get("provider", {}).get("runtime_identity") != plan["runtime_identity"]
        or type(elapsed) not in (float, int)
        or not math.isfinite(elapsed)
        or elapsed < 0
        or state not in {"raw_saved", "ready", "failed", "cancelled"}
    ):
        raise ValueError("SAM output provenance or timing is inconsistent")
    raw = output.get("raw_response")
    if len(json.dumps(raw, allow_nan=False).encode()) > 2 * 1024 * 1024:
        raise ValueError("SAM raw evidence exceeds its bound")
    if state == "ready":
        try:
            normalized = normalize_response(
                raw,
                config["config"]["provider_config"],
                width=frame["width"],
                height=frame["height"],
            )
        except ProviderResponseError as exc:
            raise ValueError("SAM published output has invalid raw evidence") from exc
        if output.get("error") is not None or output.get("result") != normalized:
            raise ValueError("SAM published proposals differ from their native evidence")
    elif output.get("result") is not None or (state != "raw_saved" and not output.get("error")):
        raise ValueError("Incomplete SAM evidence cannot contain published proposals")


def run_trial(store, trial_id, progress, cancelled):
    from iris.benchmark_runs import _context, _persist_result
    from iris.sam_provider import (
        ProviderResponseError as SamProviderError,
    )
    from iris.sam_provider import (
        Sam3Preannotator,
        normalize_response,
        provider_status,
    )
    from iris.sam_runtime import prepare_image

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
        or frozen.get("frame_ids") != [frame["frame_id"] for frame in frames]
        or store.list("benchmark_outputs", trial_id=trial_id)
    ):
        raise BenchmarkConflict("SAM trial inputs changed or this trial was already attempted")
    plan = validate_trial(frozen, config["config"], frames)
    settings = config["config"]["provider_config"]
    status = provider_status(store.root, settings, force=True)
    if status["status"] != "ready":
        raise ValueError(status["reason"])
    if status["runtime"]["identity"] != plan["runtime_identity"]:
        raise BenchmarkConflict("SAM runtime changed since approval; prepare a new explicit trial")
    for frame in frames:
        with open_benchmark_image(store, frame) as image:
            prepare_image(image)
    result = {
        "trial_id": trial_id,
        "frames_total": len(frames),
        "outputs_created": 0,
        "frames_ready": 0,
        "frames_issues": 0,
        "cancelled": False,
        "quality": None,
        "model_load_ms": None,
    }
    if cancelled():
        return {**result, "cancelled": True}
    with store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        job = _decode(conn.execute("SELECT * FROM jobs WHERE id=?", (trial["job_id"],)).fetchone())
        if (
            job is None
            or job["kind"] != "benchmark"
            or job["params"].get("trial_id") != trial_id
            or job["status"] not in {"queued", "running"}
            or job["cancel_requested"]
            or (job["result"] or {}).get("benchmark_attempt_id")
        ):
            raise BenchmarkConflict("SAM trial was already claimed or stopped")
        result["benchmark_attempt_id"] = new_id()
        conn.execute("UPDATE jobs SET result=? WHERE id=?", (json.dumps(result), job["id"]))
    started = time.perf_counter()
    progress(0, "Loading the pinned local SAM image model; reference boxes are withheld")
    adapter = Sam3Preannotator(store.root, settings, cancelled=cancelled)
    result["model_load_ms"] = (time.perf_counter() - started) * 1000
    _persist_result(store, trial, result)
    try:
        if adapter.metadata.get("runtime_identity") != plan["runtime_identity"]:
            raise BenchmarkConflict("Loaded SAM runtime differs from the approved runtime")
        for index, frame in enumerate(frames):
            if cancelled():
                result["cancelled"] = True
                break
            raw, error, normalized = None, None, None
            started = time.perf_counter()
            try:
                with open_benchmark_image(store, frame) as image:
                    raw = adapter.predict(image, cancelled=cancelled)
            except SamProviderError as exc:
                raw, error = exc.raw_response, str(exc)
            except (ValueError, RuntimeError, OSError) as exc:
                error = str(exc) or type(exc).__name__
            try:
                if len(json.dumps(raw, allow_nan=False).encode()) > 2 * 1024 * 1024:
                    raise ValueError("Raw SAM output exceeds the 2 MiB evidence limit")
            except (ValueError, TypeError, RecursionError):
                raw, error = {"unserializable_output": True}, "Invalid or oversized SAM evidence"
            output = store.insert(
                "benchmark_outputs",
                {
                    "id": new_id(),
                    "trial_id": trial_id,
                    "frame_id": frame["frame_id"],
                    "raw_response": raw,
                    "metadata": {
                        "state": "raw_saved",
                        "attempt_id": result["benchmark_attempt_id"],
                        "reference_withheld": True,
                        "frame_sha256": frame["sha256"],
                        "config_fingerprint": config["fingerprint"],
                        "provider": adapter.metadata,
                        "runtime_identity": plan["runtime_identity"],
                        "timing": {"elapsed_ms": (time.perf_counter() - started) * 1000},
                    },
                    "created_at": now(),
                },
            )
            if error is None:
                try:
                    normalized = normalize_response(
                        raw, settings, width=frame["width"], height=frame["height"]
                    )
                except (ValueError, TypeError, KeyError, SamProviderError) as exc:
                    error = str(exc) or type(exc).__name__
            with store.connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                job = _decode(
                    conn.execute("SELECT * FROM jobs WHERE id=?", (trial["job_id"],)).fetchone()
                )
                stopped = (
                    job["status"] not in {"queued", "running"}
                    or job["cancel_requested"]
                    or (job["result"] or {}).get("benchmark_attempt_id")
                    != result["benchmark_attempt_id"]
                    or cancelled()
                )
                if stopped:
                    result["cancelled"] = True
                    normalized, error = None, "SAM trial stopped; raw evidence retained"
                metadata = {
                    **output["metadata"],
                    "state": "cancelled" if stopped else "ready" if error is None else "failed",
                }
                conn.execute(
                    "UPDATE benchmark_outputs SET result=?,error=?,metadata=? WHERE id=?",
                    (
                        json.dumps(normalized, allow_nan=False) if error is None else None,
                        error,
                        json.dumps(metadata, allow_nan=False),
                        output["id"],
                    ),
                )
            result["outputs_created"] += 1
            result["frames_issues" if error is not None else "frames_ready"] += 1
            _persist_result(store, trial, result)
            progress(
                (index + 1) / len(frames), f"Saved {index + 1} / {len(frames)} SAM image outputs"
            )
            if error is not None or result["cancelled"]:
                break
        result["quality"] = {
            **score_benchmark_outputs(
                manifest, trial["split"], store.list("benchmark_outputs", trial_id=trial_id)
            ),
            "protocol": SAM_SCORING,
        }
        _persist_result(store, trial, result)
        return result
    finally:
        adapter.close()
