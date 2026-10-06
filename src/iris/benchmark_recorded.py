"""Offline admission of submitted provider evidence for independent human review.

Import jobs verify internal consistency; they do not authenticate historical API
calls, execute models, spend provider credits or create human review intervals.
"""

import json
import math
import time
from copy import deepcopy
from datetime import datetime

from iris import dinox_provider as dinox
from iris import dinox_review_provider as review
from iris import multimodal_provider as multimodal
from iris.benchmark import PROTOCOL, SCORING, BenchmarkConflict, _digest
from iris.store import _decode, new_id, now

BUNDLE_PROTOCOL = "iris-benchmark-recorded-bundle-v1"
IMPORT_PROTOCOL = "iris-benchmark-recorded-import-v1"
MAX_BUNDLE_BYTES = 16 * 1024 * 1024
LATENCY_SCOPE = (
    "Per-image saved-output validation, normalization and hashing; excludes preflight "
    "image verification and database writes; not provider inference or full import duration"
)
WARNING = (
    "Submitted saved evidence is checked for internal consistency, not authenticated "
    "provider history. This import makes no external request. Source costs and timings "
    "are historical declarations, separate from local import and human review."
)


def _canonical(value):
    try:
        return json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode()
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ValueError("Recorded evidence must be finite JSON") from exc


def _number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value) and value >= 0
    except OverflowError:
        return False


def _settings(settings, taxonomy):
    if not isinstance(settings, dict):
        raise ValueError("Recorded evidence settings are required")
    settings = {key: value for key, value in settings.items() if value is not None}
    transform = settings.get("transform")
    fields = {"dinox_config", "transform"}
    if transform == "threshold":
        fields.add("threshold")
    elif transform == "review":
        fields.add("review_config")
    elif transform != "identity":
        raise ValueError("Choose identity, threshold or review for recorded proposals")
    if set(settings) != fields:
        raise ValueError("Recorded transform settings are inconsistent")
    source = dinox.validate_frozen_config(settings["dinox_config"])
    if source["taxonomy"] != taxonomy:
        raise ValueError("Recorded detector classes differ from the frozen reference")
    if transform == "threshold" and (
        not _number(settings["threshold"])
        or not source["settings"]["bbox_threshold"] <= settings["threshold"] <= 1
    ):
        raise ValueError("Recorded threshold must retain a subset of source detections")
    if transform == "review":
        reviewed = review.validate_frozen_config(settings["review_config"])
        if reviewed["taxonomy"] != taxonomy:
            raise ValueError("Recorded review classes differ from the frozen reference")
    return deepcopy(settings)


def _model(settings):
    return dinox.MODEL + ("+gpt-6-astra" if settings["transform"] == "review" else "")


def work_plan(count):
    return {"image_count": count, "import_count": count, "request_count": 0}


def preview_config(store, benchmark_id, *, model_id, recorded):
    from iris.benchmark import load_benchmark_manifest

    benchmark = store.get("benchmarks", benchmark_id)
    manifest = load_benchmark_manifest(store, benchmark_id)
    settings = _settings(recorded, manifest["taxonomy"])
    if model_id != _model(settings):
        raise ValueError("Recorded model identity differs from its frozen source")
    config = {
        "protocol": PROTOCOL,
        "approach": "recorded_proposals",
        "model_id": model_id,
        "model_name": "Saved " + model_id + " · " + settings["transform"],
        "recorded": settings,
        "execution": "offline_import",
        "provenance": "submitted_saved_evidence",
        "scoring": SCORING,
        "reference_manifest_sha256": benchmark["manifest_sha256"],
    }
    return {
        "benchmark_id": benchmark_id,
        "config": config,
        "fingerprint": _digest(config),
        "work": {
            role: work_plan(sum(f["role"] == role for f in manifest["frames"]))
            for role in ("tuning", "evaluation")
        },
        "warnings": [WARNING],
    }


def validate_config(config, manifest):
    settings = _settings(config.get("recorded"), manifest["taxonomy"])
    if (
        set(config)
        != {
            "protocol",
            "approach",
            "model_id",
            "model_name",
            "recorded",
            "execution",
            "provenance",
            "scoring",
            "reference_manifest_sha256",
        }
        or config["recorded"] != settings
        or config["model_id"] != _model(settings)
        or config["model_name"] != "Saved " + _model(settings) + " · " + settings["transform"]
        or config["execution"] != "offline_import"
        or config["provenance"] != "submitted_saved_evidence"
        or config["scoring"] != SCORING
    ):
        raise ValueError("Recorded configuration provenance is inconsistent")
    return config


def _receipt(value, *, reviewer=False):
    fields = {"status", "recorded_at", "elapsed_ms"}
    fields |= (
        {"response_id", "request_sha256", "usage_cost_usd"}
        if reviewer
        else {"task_id", "estimated_cost_cny"}
    )
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("A complete bounded source receipt is required")
    identifier = value["response_id" if reviewer else "task_id"]
    cost = value["usage_cost_usd" if reviewer else "estimated_cost_cny"]
    if (
        not isinstance(identifier, str)
        or not 1 <= len(identifier) <= 256
        or not identifier.isprintable()
        or value["status"] != ("completed" if reviewer else "succeeded")
        or value["elapsed_ms"] is not None
        and not _number(value["elapsed_ms"])
        or cost is not None
        and not _number(cost)
    ):
        raise ValueError("Source receipt must describe a completed outcome")
    try:
        timestamp = datetime.fromisoformat(value["recorded_at"])
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise ValueError("Source receipt timestamp needs a timezone") from exc
    return value


def _derive(entry, config):
    try:
        return _derive_checked(entry, config)
    except (KeyError, TypeError, AttributeError, OverflowError, RecursionError) as exc:
        raise ValueError("Recorded frame evidence is malformed") from exc


def _derive_checked(entry, config):
    settings = config["recorded"]
    fields = {"frame_id", "image_file_sha256", "source_pixel_sha256", "width", "height", "dinox"}
    if settings["transform"] == "review":
        fields.add("review")
    if not isinstance(entry, dict) or set(entry) != fields:
        raise ValueError("Recorded frame evidence has unexpected or missing fields")
    source = entry["dinox"]
    if not isinstance(source, dict) or set(source) != {"raw_result", "receipt"}:
        raise ValueError("Recorded DINO-X evidence is incomplete")
    _receipt(source["receipt"])
    native = dinox.normalize(
        source["raw_result"], settings["dinox_config"], entry["width"], entry["height"]
    )
    if settings["transform"] == "identity":
        return native
    if settings["transform"] == "threshold":
        result = deepcopy(native)
        result["proposals"] = [
            p for p in native["proposals"] if p["score"] >= settings["threshold"]
        ]
        result["recorded_transform"] = {
            "threshold": settings["threshold"],
            "source_count": len(native["proposals"]),
        }
        kept_ids = {item["id"] for item in result["proposals"]}
        result["raw_output"]["proposals"] = [
            item for item in result["raw_output"]["proposals"] if item["id"] in kept_ids
        ]
        result["filtered_count"] += len(native["proposals"]) - len(result["proposals"])
        return result
    evidence = entry["review"]
    if not isinstance(evidence, dict) or set(evidence) != {"input", "raw_response", "receipt"}:
        raise ValueError("Recorded review evidence is incomplete")
    receipt = _receipt(evidence["receipt"], reviewer=True)
    saved_input = review.validate_input(evidence["input"], settings["review_config"], native)
    image = saved_input["image"]
    if (
        image["source_pixel_sha256"] != entry["source_pixel_sha256"]
        or image["width"] != entry["width"]
        or image["height"] != entry["height"]
        or receipt["request_sha256"] != saved_input["request_sha256"]
        or receipt["response_id"] != evidence["raw_response"].get("id")
        or receipt["usage_cost_usd"]
        != multimodal._usage_metadata(
            evidence["raw_response"], settings["review_config"]["openai_config"]
        )["usage_cost_usd"]
    ):
        raise ValueError("Recorded review receipt, usage cost or image binding changed")
    return review.normalize_response(evidence["raw_response"], settings["review_config"], native)


def validate_bundle(bundle, config, frames):
    if len(_canonical(bundle)) > MAX_BUNDLE_BYTES:
        raise ValueError("Recorded evidence bundle exceeds 16 MiB")
    if (
        not isinstance(bundle, dict)
        or set(bundle) != {"protocol", "frames"}
        or bundle["protocol"] != BUNDLE_PROTOCOL
        or not isinstance(bundle["frames"], list)
        or len(bundle["frames"]) != len(frames)
    ):
        raise ValueError("Recorded bundle must cover the exact frozen image role")
    tasks, responses = set(), set()
    for entry, frame in zip(bundle["frames"], frames, strict=True):
        if not isinstance(entry, dict) or any(
            entry.get(key) != expected
            for key, expected in (
                ("frame_id", frame["frame_id"]),
                ("image_file_sha256", frame["image_file_sha256"]),
                ("source_pixel_sha256", frame["sha256"]),
                ("width", frame["width"]),
                ("height", frame["height"]),
            )
        ):
            raise ValueError("Recorded frame identity or frozen image hash changed")
        _derive(entry, config)
        task = entry["dinox"]["receipt"]["task_id"]
        if task in tasks:
            raise ValueError("A detector source task cannot describe two different frames")
        tasks.add(task)
        if "review" in entry:
            response = entry["review"]["receipt"]["response_id"]
            if response in responses:
                raise ValueError("A review response cannot describe two different frames")
            responses.add(response)
    return bundle


def _verify_pixels(store, frames, bundle, config):
    from iris.benchmark import open_benchmark_image

    for frame, entry in zip(frames, bundle["frames"], strict=True):
        with open_benchmark_image(store, frame) as image:
            if "review" in entry:
                native = dinox.normalize(
                    entry["dinox"]["raw_result"],
                    config["recorded"]["dinox_config"],
                    frame["width"],
                    frame["height"],
                )
                prepared = review.prepare_request(
                    image, config["recorded"]["review_config"], native
                )
                if review.safe_request(prepared) != entry["review"]["input"]:
                    raise ValueError("Saved review input differs from the actual frozen pixels")


def preview_import(store, benchmark_id, *, config_id, role, bundle):
    from iris.benchmark_runs import _context

    benchmark, config, manifest = _context(store, benchmark_id, config_id, role)
    if config["approach"] != "recorded_proposals":
        raise ValueError("Saved evidence import requires a recorded proposals configuration")
    frames = [f for f in manifest["frames"] if f["role"] == role]
    validate_bundle(bundle, config["config"], frames)
    _verify_pixels(store, frames, bundle, config["config"])
    identity = {
        "benchmark_id": benchmark_id,
        "config_id": config_id,
        "role": role,
        "manifest_sha256": benchmark["manifest_sha256"],
        "config_fingerprint": config["fingerprint"],
        "phase": benchmark["status"],
        "recorded_bundle_sha256": _digest(bundle),
    }
    return {
        **identity,
        "fingerprint": _digest(identity),
        "frame_ids": [f["frame_id"] for f in frames],
        "work": work_plan(len(frames)),
        "reference_withheld": True,
        "warnings": [WARNING],
    }


def validate_trial(frozen, config, frames):
    bundle = frozen.get("recorded_bundle")
    if (
        frozen.get("candidate_config") != config
        or frozen.get("protocol") != PROTOCOL
        or frozen.get("frame_ids") != [frame["frame_id"] for frame in frames]
        or frozen.get("recorded_bundle_sha256") != _digest(bundle)
        or frozen.get("work") != work_plan(len(frames))
        or frozen.get("warnings") != [WARNING]
        or "external_plan" in frozen
        or "local_plan" in frozen
    ):
        raise ValueError("Recorded import provenance or work plan changed")
    return validate_bundle(bundle, config, frames)


def create_import(store, jobs, benchmark_id, *, config_id, role, bundle, expected_fingerprint):
    from iris.benchmark_runs import benchmark_trial_detail

    with jobs.guard, store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        existing = _decode(
            conn.execute(
                "SELECT * FROM benchmark_trials WHERE benchmark_id=? AND config_id=? AND split=? "
                "AND json_extract(config,'$.recorded_bundle_sha256')=? "
                "ORDER BY created_at,id LIMIT 1",
                (benchmark_id, config_id, role, _digest(bundle)),
            ).fetchone()
        )
        if existing:
            if existing["config"]["fingerprint"] != expected_fingerprint:
                raise BenchmarkConflict("Recorded import fingerprint changed")
            identifier = existing["id"]
        else:
            preview = preview_import(
                store, benchmark_id, config_id=config_id, role=role, bundle=bundle
            )
            if preview["fingerprint"] != expected_fingerprint:
                raise BenchmarkConflict("Recorded import inputs changed; preview again")
            config = store.get("benchmark_configs", config_id)
            frozen = {
                "protocol": PROTOCOL,
                "fingerprint": preview["fingerprint"],
                "source_config_fingerprint": config["fingerprint"],
                "benchmark_manifest_sha256": preview["manifest_sha256"],
                "frame_ids": preview["frame_ids"],
                "role": role,
                "candidate_config": config["config"],
                "work": preview["work"],
                "warnings": preview["warnings"],
                "recorded_bundle": deepcopy(bundle),
                "recorded_bundle_sha256": preview["recorded_bundle_sha256"],
            }
            identifier, job_id, created = new_id(), new_id(), now()
            conn.execute(
                "INSERT INTO jobs (id,kind,status,params,message,created_at) VALUES (?,?,?,?,?,?)",
                (
                    job_id,
                    "benchmark",
                    "queued",
                    json.dumps({"trial_id": identifier, "operation": "import_recorded_proposals"}),
                    "Waiting for local validation of submitted saved evidence; no API calls",
                    created,
                ),
            )
            conn.execute(
                "INSERT INTO benchmark_trials "
                "(id,benchmark_id,config_id,split,config,job_id,created_at) VALUES (?,?,?,?,?,?,?)",
                (identifier, benchmark_id, config_id, role, json.dumps(frozen), job_id, created),
            )
    return benchmark_trial_detail(store, identifier)


def _source(entry):
    return {key: deepcopy(entry[key]["receipt"]) for key in ("dinox", "review") if key in entry}


def validate_output_row(output, trial, *, validated_bundle=None, attempt=None):
    frozen = trial["config"]
    bundle = validated_bundle if validated_bundle is not None else frozen["recorded_bundle"]
    entry = next((f for f in bundle["frames"] if f["frame_id"] == output["frame_id"]), None)
    metadata = output.get("metadata") or {}
    imported = metadata.get("recorded", {})
    timing = metadata.get("timing", {})
    if (
        output["trial_id"] != trial["id"]
        or entry is None
        or output.get("error") is not None
        or output.get("raw_response") != entry
        or output.get("result") != _derive(entry, frozen["candidate_config"])
        or set(metadata) != {"state", "recorded", "timing", "source"}
        or metadata["state"] != "ready"
        or set(imported) != {"protocol", "bundle_sha256", "frame_evidence_sha256", "attempt_id"}
        or imported.get("protocol") != IMPORT_PROTOCOL
        or imported.get("bundle_sha256") != frozen["recorded_bundle_sha256"]
        or imported.get("frame_evidence_sha256") != _digest(entry)
        or not isinstance(imported.get("attempt_id"), str)
        or not imported["attempt_id"]
        or attempt is not None
        and imported["attempt_id"] != attempt
        or set(timing) != {"elapsed_ms", "scope"}
        or not _number(timing.get("elapsed_ms"))
        or timing.get("scope") != "local_import_validation"
        or metadata.get("source") != _source(entry)
    ):
        raise ValueError("Recorded output differs from its immutable source evidence")
    return imported


def run_trial(store, trial_id, progress, cancelled):
    from iris.benchmark import score_benchmark_outputs
    from iris.benchmark_runs import _context, _persist_result

    trial = store.get("benchmark_trials", trial_id)
    benchmark, config, manifest = _context(
        store, trial["benchmark_id"], trial["config_id"], trial["split"]
    )
    frames = [f for f in manifest["frames"] if f["role"] == trial["split"]]
    bundle = validate_trial(trial["config"], config["config"], frames)
    if (
        trial["config"]["source_config_fingerprint"] != config["fingerprint"]
        or trial["config"]["benchmark_manifest_sha256"] != benchmark["manifest_sha256"]
    ):
        raise ValueError("Recorded import configuration ownership changed")
    _verify_pixels(store, frames, bundle, config["config"])
    result = {
        "trial_id": trial_id,
        "frames_total": len(frames),
        "outputs_created": 0,
        "frames_ready": 0,
        "frames_issues": 0,
        "cancelled": False,
        "quality": None,
        "operation": "import_recorded_proposals",
    }
    with store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        job = _decode(conn.execute("SELECT * FROM jobs WHERE id=?", (trial["job_id"],)).fetchone())
        if (
            not job
            or job["kind"] != "benchmark"
            or job["params"] != {"trial_id": trial_id, "operation": "import_recorded_proposals"}
            or job["status"] not in {"queued", "running"}
            or (job["result"] or {}).get("benchmark_attempt_id")
            or conn.execute(
                "SELECT 1 FROM benchmark_outputs WHERE trial_id=?", (trial_id,)
            ).fetchone()
        ):
            raise BenchmarkConflict("Recorded import was already claimed or is no longer active")
        if job["cancel_requested"] or cancelled():
            return {**result, "cancelled": True}
        result["benchmark_attempt_id"] = new_id()
        conn.execute("UPDATE jobs SET result=? WHERE id=?", (json.dumps(result), trial["job_id"]))
    for index, entry in enumerate(bundle["frames"]):
        if cancelled():
            result["cancelled"] = True
            break
        started = time.perf_counter()
        normalized = _derive(entry, config["config"])
        metadata = {
            "state": "ready",
            "source": _source(entry),
            "recorded": {
                "protocol": IMPORT_PROTOCOL,
                "bundle_sha256": _digest(bundle),
                "frame_evidence_sha256": _digest(entry),
                "attempt_id": result["benchmark_attempt_id"],
            },
            "timing": {
                "elapsed_ms": (time.perf_counter() - started) * 1000,
                "scope": "local_import_validation",
            },
        }
        with store.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            current = conn.execute(
                "SELECT status,cancel_requested FROM jobs WHERE id=?", (trial["job_id"],)
            ).fetchone()
            if current["status"] not in {"queued", "running"} or current["cancel_requested"]:
                result["cancelled"] = True
                break
            conn.execute(
                "INSERT INTO benchmark_outputs "
                "(id,trial_id,frame_id,raw_response,result,metadata,created_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (
                    new_id(),
                    trial_id,
                    entry["frame_id"],
                    json.dumps(entry),
                    json.dumps(normalized),
                    json.dumps(metadata),
                    now(),
                ),
            )
        result["outputs_created"] += 1
        result["frames_ready"] += 1
        _persist_result(store, trial, result)
        progress((index + 1) / len(frames), "Validated saved evidence locally; no provider request")
    result["quality"] = score_benchmark_outputs(
        manifest, trial["split"], store.list("benchmark_outputs", trial_id=trial_id)
    )
    _persist_result(store, trial, result)
    return result
