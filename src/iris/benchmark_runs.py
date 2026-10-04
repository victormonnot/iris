"""Bounded benchmark trials: adapters see images and classes, never reference boxes."""

import json
import math
import time
from copy import deepcopy

from iris.benchmark import (
    PROTOCOL,
    ROLES,
    BenchmarkConflict,
    _digest,
    load_benchmark_manifest,
    open_benchmark_image,
    score_benchmark_outputs,
    validate_benchmark_config,
)
from iris.inference import _validate_prediction, _work_plan
from iris.model_taxonomy import class_contract
from iris.models import TorchvisionDetector, catalog
from iris.preannotation_contracts import build_contract, normalize_candidates
from iris.store import Store, _decode, new_id, now
from iris.tiling import TiledInferenceCancelled, tile_boxes, tiled_predict


def _context(store, benchmark_id, config_id, role, *, check_phase=True):
    benchmark, config = (
        store.get("benchmarks", benchmark_id),
        store.get("benchmark_configs", config_id),
    )
    if benchmark is None or config is None or config["benchmark_id"] != benchmark_id:
        raise KeyError(config_id)
    if role not in ROLES:
        raise ValueError("Choose the tuning or evaluation role")
    if check_phase and (
        (role == "tuning" and benchmark["status"] != "tuning")
        or (role == "evaluation" and benchmark["status"] != "locked")
    ):
        raise BenchmarkConflict("Tuning runs precede the explicit lock; evaluation runs follow it")
    manifest = load_benchmark_manifest(store, benchmark_id)
    validate_benchmark_config(config, benchmark, manifest)
    return benchmark, config, manifest


def _ready_config(store, config, manifest):
    frozen = config["config"]
    model = next(
        (model for model in catalog(store.root) if model["id"] == frozen["model_id"]), None
    )
    if (
        model is None
        or model["status"] != "ready"
        or model.get("weight_sha256") != frozen["weight_sha256"]
    ):
        raise ValueError("The frozen detector checkpoint is no longer installed with the same hash")
    if (
        build_contract(store, frozen["model_id"], manifest["taxonomy"])
        != frozen["proposal_contract"]
    ):
        raise ValueError("The frozen detector class definitions changed")


def preview_benchmark_trial(store: Store, benchmark_id: str, *, config_id: str, role: str):
    benchmark, config, manifest = _context(store, benchmark_id, config_id, role)
    _ready_config(store, config, manifest)
    frames = [frame for frame in manifest["frames"] if frame["role"] == role]
    for frame in frames:
        with open_benchmark_image(store, frame):
            pass
    frozen = config["config"]
    work = _work_plan(
        [{**frame, "id": frame["frame_id"]} for frame in frames],
        [{"model_id": frozen["model_id"], "variant": frozen["inference"]["mode"]}],
        frozen["inference"],
    )
    with store.connect() as conn:
        history = [
            _decode(row)
            for row in conn.execute(
                "SELECT t.*,j.status AS job_status FROM benchmark_trials t "
                "JOIN jobs j ON j.id=t.job_id "
                "WHERE t.benchmark_id=? AND t.config_id=? AND t.split=? ORDER BY t.created_at,t.id",
                (benchmark_id, config_id, role),
            )
        ]
    if any(row["job_status"] in {"queued", "running"} for row in history):
        raise BenchmarkConflict("This configuration already has an active trial for this role")
    inputs = {
        "benchmark_id": benchmark_id,
        "config_id": config_id,
        "role": role,
        "manifest_sha256": benchmark["manifest_sha256"],
        "config_fingerprint": config["fingerprint"],
        "phase": benchmark["status"],
        "previous_trials": [row["id"] for row in history],
    }
    return {
        **inputs,
        "fingerprint": _digest(inputs),
        "frame_ids": [frame["frame_id"] for frame in frames],
        "work": work,
        "reference_withheld": True,
        "warnings": [
            "Candidate adapters receive only image pixels, class definitions and frozen settings.",
            *(
                [
                    "This evaluation role was already used; this is a recorded repeat, "
                    "not a fresh held-out set."
                ]
                if role == "evaluation" and history
                else []
            ),
        ],
    }


def create_benchmark_trial(
    store: Store, jobs, benchmark_id: str, *, config_id: str, role: str, expected_fingerprint: str
):
    with jobs.guard, store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        existing = _decode(
            conn.execute(
                "SELECT * FROM benchmark_trials WHERE benchmark_id=? AND config_id=? AND split=? "
                "AND json_extract(config,'$.fingerprint')=? ORDER BY created_at,id LIMIT 1",
                (benchmark_id, config_id, role, expected_fingerprint),
            ).fetchone()
        )
        if existing:
            identifier = existing["id"]
        else:
            preview = preview_benchmark_trial(store, benchmark_id, config_id=config_id, role=role)
            if preview["fingerprint"] != expected_fingerprint:
                raise BenchmarkConflict(
                    "Benchmark trial inputs changed; preview the explicit launch again"
                )
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
            }
            identifier, job_id, created = new_id(), new_id(), now()
            conn.execute(
                "INSERT INTO jobs (id,kind,status,params,message,created_at) VALUES (?,?,?,?,?,?)",
                (
                    job_id,
                    "benchmark",
                    "queued",
                    json.dumps({"trial_id": identifier}),
                    "Waiting for a local benchmark trial; human reference withheld",
                    created,
                ),
            )
            conn.execute(
                "INSERT INTO benchmark_trials "
                "(id,benchmark_id,config_id,split,config,job_id,created_at) VALUES (?,?,?,?,?,?,?)",
                (identifier, benchmark_id, config_id, role, json.dumps(frozen), job_id, created),
            )
    return benchmark_trial_detail(store, identifier)


def _persist_result(store, trial, result):
    with store.connect() as conn:
        conn.execute(
            "UPDATE jobs SET result=? WHERE id=? AND status IN ('queued','running')",
            (json.dumps(result, allow_nan=False), trial["job_id"]),
        )


def run_benchmark_trial(store: Store, trial_id: str, progress, cancelled, detector_factory=None):
    trial = store.get("benchmark_trials", trial_id)
    if trial is None:
        raise KeyError(trial_id)
    job = store.get("jobs", trial["job_id"])
    if (
        job is None
        or job["kind"] != "benchmark"
        or job["params"].get("trial_id") != trial_id
        or job["status"] not in {"queued", "running"}
    ):
        raise ValueError("Benchmark trial job provenance or active state is inconsistent")
    if store.list("benchmark_outputs", trial_id=trial_id):
        raise ValueError("A benchmark trial is immutable; launch a new explicit trial")
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
    ):
        raise ValueError("Frozen benchmark trial inputs changed")
    settings = config["config"]
    work = _work_plan(
        [{**frame, "id": frame["frame_id"]} for frame in frames],
        [{"model_id": settings["model_id"], "variant": settings["inference"]["mode"]}],
        settings["inference"],
    )
    if work != frozen.get("work"):
        raise ValueError("Frozen benchmark inference work changed")
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
    _ready_config(store, config, manifest)
    with store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        current = _decode(
            conn.execute("SELECT * FROM jobs WHERE id=?", (trial["job_id"],)).fetchone()
        )
        if (
            current["status"] not in {"queued", "running"}
            or current["cancel_requested"]
            or (current["result"] or {}).get("benchmark_attempt_id")
        ):
            raise BenchmarkConflict("This benchmark trial was already claimed or stopped")
        result["benchmark_attempt_id"] = new_id()
        conn.execute("UPDATE jobs SET result=? WHERE id=?", (json.dumps(result), trial["job_id"]))
    progress(0, "Loading the frozen local detector; no reference boxes are sent")
    # Deliberate allowlist: the detector receives no manifest, annotation, reference
    # box, correction, notes, session or scorer argument. Pixel loading and scoring
    # stay on opposite sides of this adapter boundary.
    detector = (detector_factory or TorchvisionDetector)(
        store.root, settings["model_id"], device=settings["device"]
    )
    try:
        if detector.metadata.get("weight_sha256") != settings["weight_sha256"]:
            raise ValueError("Loaded benchmark detector weights differ from the frozen checkpoint")
        source_contract = settings["proposal_contract"]["source_contract"]
        if (
            source_contract["taxonomy_id"] != "coco-2017-v1"
            and class_contract(detector.metadata) != source_contract
        ):
            raise ValueError("Loaded benchmark detector classes differ from the frozen checkpoint")
        for index, frame in enumerate(frames):
            if cancelled():
                raise TiledInferenceCancelled()
            raw, normalized, error, timing = None, None, None, {}
            started = time.perf_counter()
            try:
                with open_benchmark_image(store, frame) as image:
                    decoded = time.perf_counter()
                    decode_ms = (decoded - started) * 1000
                    if index == 0:
                        if settings["inference"]["mode"] == "tiled":
                            first = tile_boxes(
                                image.width, image.height, settings["inference"]["tiling"]
                            )[0]
                            with image.crop(tuple(first)) as tile:
                                detector.warmup(tile)
                        else:
                            detector.warmup(image)
                        started = time.perf_counter()
                    if cancelled():
                        raise TiledInferenceCancelled()
                    if settings["inference"]["mode"] == "tiled":
                        raw = tiled_predict(
                            detector, image, settings["inference"]["tiling"], cancelled=cancelled
                        )
                    else:
                        raw = detector.predict(image)
                elapsed = (time.perf_counter() - started) * 1000
                if index == 0:
                    elapsed += decode_ms
                # The adapter's complete output is persisted before normalization.
                # Invalid JSON is preserved as an explicitly non-replayable summary.
                if len(json.dumps(raw, allow_nan=False).encode()) > 2 * 1024 * 1024:
                    raise ValueError("Raw detector output exceeds the 2 MiB evidence limit")
                timing = {
                    **(raw.get("timing", {}) if isinstance(raw, dict) else {}),
                    "decode_ms": decode_ms,
                    "elapsed_ms": elapsed,
                }
                output = store.insert(
                    "benchmark_outputs",
                    {
                        "id": new_id(),
                        "trial_id": trial_id,
                        "frame_id": frame["frame_id"],
                        "raw_response": deepcopy(raw),
                        "metadata": {
                            "state": "raw_saved",
                            "frame_sha256": frame["sha256"],
                            "reference_withheld": True,
                            "config_fingerprint": config["fingerprint"],
                            "detector": detector.metadata,
                            "timing": timing,
                        },
                        "created_at": now(),
                    },
                )
                try:
                    _validate_prediction(raw, frame)
                    normalized = normalize_candidates(
                        raw["detections"],
                        settings["proposal_contract"],
                        settings["threshold"],
                        width=frame["width"],
                        height=frame["height"],
                    )
                    if len(normalized["proposals"]) > 100:
                        raise ValueError(
                            "More than 100 proposals; the frozen configuration exceeds "
                            "the review limit"
                        )
                except (ValueError, TypeError, KeyError, AttributeError) as exc:
                    error = str(exc) or type(exc).__name__
                with store.connect() as conn:
                    conn.execute("BEGIN IMMEDIATE")
                    current = _decode(
                        conn.execute("SELECT * FROM jobs WHERE id=?", (trial["job_id"],)).fetchone()
                    )
                    stopped = (
                        current["status"] not in {"queued", "running"}
                        or current["cancel_requested"]
                        or (current["result"] or {}).get("benchmark_attempt_id")
                        != result["benchmark_attempt_id"]
                        or cancelled()
                    )
                    if stopped:
                        result["cancelled"] = True
                        normalized, error = (
                            None,
                            "Trial stopped after saving the raw detector output",
                        )
                    conn.execute(
                        "UPDATE benchmark_outputs SET result=?,error=?,metadata=? WHERE id=?",
                        (
                            json.dumps(normalized, allow_nan=False) if error is None else None,
                            error,
                            json.dumps(
                                {
                                    **output["metadata"],
                                    "state": "cancelled"
                                    if stopped
                                    else "ready"
                                    if error is None
                                    else "invalid_output",
                                },
                                allow_nan=False,
                            ),
                            output["id"],
                        ),
                    )
            except TiledInferenceCancelled:
                raise
            except Exception as exc:
                error = str(exc) or type(exc).__name__
                # A failure before an adapter returns a complete output is not an
                # empty prediction. Keep the error and previous complete images.
                exists = store.list(
                    "benchmark_outputs", trial_id=trial_id, frame_id=frame["frame_id"]
                )
                if exists:
                    store.update("benchmark_outputs", exists[0]["id"], {"error": error})
                else:
                    safe = None
                    if raw is not None:
                        try:
                            if len(json.dumps(raw, allow_nan=False).encode()) > 2 * 1024 * 1024:
                                raise ValueError("Oversized raw output")
                            safe = raw
                        except (ValueError, TypeError, RecursionError):
                            safe = {"non_json_output_summary": repr(raw)[:4000]}
                    store.insert(
                        "benchmark_outputs",
                        {
                            "id": new_id(),
                            "trial_id": trial_id,
                            "frame_id": frame["frame_id"],
                            "raw_response": safe,
                            "error": error,
                            "metadata": {
                                "state": "failed",
                                "reference_withheld": True,
                                "frame_sha256": frame["sha256"],
                                "config_fingerprint": config["fingerprint"],
                                "timing": {"elapsed_ms": (time.perf_counter() - started) * 1000},
                            },
                            "created_at": now(),
                        },
                    )
            result["outputs_created"] += 1
            result["frames_issues" if error is not None else "frames_ready"] += 1
            _persist_result(store, trial, result)
            progress(
                result["outputs_created"] / len(frames),
                f"Saved {result['outputs_created']} / {len(frames)} benchmark image outputs",
            )
            if result["cancelled"]:
                raise TiledInferenceCancelled()
        outputs = store.list("benchmark_outputs", trial_id=trial_id)
        result["quality"] = score_benchmark_outputs(manifest, trial["split"], outputs)
        _persist_result(store, trial, result)
    except TiledInferenceCancelled:
        result["cancelled"] = True
        _persist_result(store, trial, result)
    finally:
        del detector
    return result


def benchmark_trial_detail(store: Store, trial_id: str, *, include_outputs: bool = True):
    from iris.benchmark_corrections import correction_summaries

    trial = store.get("benchmark_trials", trial_id)
    if trial is None:
        raise KeyError(trial_id)
    benchmark, config, manifest = _context(
        store, trial["benchmark_id"], trial["config_id"], trial["split"], check_phase=False
    )
    outputs = store.list("benchmark_outputs", trial_id=trial_id)
    rows = {row["frame_id"]: row for row in outputs}
    job = store.get("jobs", trial["job_id"])
    saved_quality = (job.get("result") or {}).get("quality")
    timings = [row.get("metadata", {}).get("timing", {}).get("elapsed_ms") for row in outputs]
    timings = [
        value
        for value in timings
        if type(value) in {int, float} and math.isfinite(value) and value >= 0
    ]
    detail = {
        **trial,
        "job": job,
        "config_name": config["name"],
        "quality": saved_quality
        or {
            "complete": False,
            "metrics": None,
            "reason": "Trial has no complete saved quality measurement",
        },
        "frames": [
            {
                "frame_id": frame["frame_id"],
                "width": frame["width"],
                "height": frame["height"],
                "source_filename": frame["source"]["filename"],
                "output_id": rows.get(frame["frame_id"], {}).get("id"),
                "state": rows.get(frame["frame_id"], {})
                .get("metadata", {})
                .get("state", job["status"]),
                "error": rows.get(frame["frame_id"], {}).get("error"),
                "proposal_count": len(
                    (rows.get(frame["frame_id"], {}).get("result") or {}).get("proposals", [])
                ),
            }
            for frame in manifest["frames"]
            if frame["role"] == trial["split"]
        ],
        "counts": {
            "total": len(trial["config"]["frame_ids"]),
            "outputs": len(outputs),
            "ready": sum(row["result"] is not None and row["error"] is None for row in outputs),
            "issues": sum(row["error"] is not None for row in outputs),
        },
        "latency": {
            "measured_count": len(timings),
            "planned_count": len(trial["config"]["frame_ids"]),
            "total_ms": sum(timings) if timings else None,
            "mean_ms": sum(timings) / len(timings) if timings else None,
            "includes": "image decode and local inference; successful outputs exclude warmup",
            "note": "Saved image attempts, including failures; missing measurements are not zero.",
        },
    }
    if include_outputs:
        detail["outputs"] = outputs
    detail["corrections"] = correction_summaries(store, trial_id)
    return detail
