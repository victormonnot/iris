"""Offline comparisons of every saved attempt on one frozen reference role."""

import hashlib
import json
import math
import re
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime

from iris.benchmark import (
    PROTOCOL as BENCHMARK_PROTOCOL,
)
from iris.benchmark import (
    ROLES,
    SCORING,
    WARNINGS,
    load_benchmark_manifest,
    score_benchmark_outputs,
    score_proposals,
    validate_benchmark_config,
)
from iris.store import _decode

PROTOCOL = "iris-benchmark-comparison-v1"
MAX_TRIALS = 100
APPROACHES = ("multimodal", "segmentation", "combined")
ACTIVE = {"queued", "running"}
CORRECTION_KEYS = (
    "id",
    "revision",
    "status",
    "boxes",
    "decisions",
    "timing",
    "reviewer",
    "created_at",
)
METRICS = ("tp", "fp", "fn", "class_conflicts", "precision", "recall", "matched_iou_mean")
LATENCY_SCOPES = {
    "multimodal": "OpenAI request and response round trip, including provider processing",
    "segmentation": "image decode and local SAM prediction; includes cold first prediction; "
    "excludes separately recorded model loading; no warmup",
    "combined": "end-to-end planning, local grounding and review; includes first model "
    "loading and cold prediction; no warmup",
    "local_detector": "image decode and local inference; successful outputs exclude warmup",
    "recorded_proposals": "per-image saved-output validation, normalization and hashing; "
    "excludes preflight image verification and database writes; "
    "not provider inference or full import duration",
}


def _canonical(value):
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode()


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest() if value is not None else None


def _number(value):
    return value if type(value) in (int, float) and math.isfinite(value) and value >= 0 else None


def _timestamp(value):
    try:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("Comparison source timestamps require a timezone")
        return parsed
    except (TypeError, ValueError) as exc:
        raise ValueError("Comparison source timestamp is invalid") from exc


class _ReadView:
    """Reuse the caller's transaction, including dispatch validators' nested reads."""

    def __init__(self, connection, store=None):
        self.connection = connection
        self.store = store

    @contextmanager
    def connect(self):
        yield self.connection

    def get(self, table, identifier):
        if table != "benchmarks":
            raise ValueError("Unexpected comparison lookup")
        return _decode(
            self.connection.execute("SELECT * FROM benchmarks WHERE id=?", (identifier,)).fetchone()
        )

    def artifact_path(self, path):
        return self.store.artifact_path(path)


def _proposals(output):
    result = output.get("result")
    if result is None:
        return None
    if not isinstance(result, dict) or not isinstance(result.get("proposals"), list):
        raise ValueError("A benchmark output has invalid normalized proposals")
    return [
        {key: deepcopy(item.get(key)) for key in ("id", "label", "box", "score")}
        for item in result["proposals"]
    ]


def _quality(manifest, role, outputs, status, scoring):
    value = score_benchmark_outputs(manifest, role, outputs)
    value["protocol"] = deepcopy(scoring)
    if status != "succeeded":
        value.update(
            complete=False,
            metrics=None,
            reason="Headline quality requires a succeeded trial and one valid output "
            "for every role image.",
        )
    return value


def _corrections(frames):
    rows = [frame["correction"] for frame in frames if frame["correction"] is not None]
    reviewed = [row for row in rows if row["status"] == "reviewed"]
    timed = [row for row in reviewed if _number(row["timing"].get("elapsed_ms")) is not None]
    return {
        "reviewed_count": len(reviewed),
        "planned_count": len(frames),
        "timed_count": len(timed),
        "fully_timed_count": sum(row["timing"].get("fully_timed") is True for row in timed),
        "recorded_review_ms": sum(row["timing"]["elapsed_ms"] for row in timed) if timed else None,
        "complete": bool(frames) and len(reviewed) == len(frames),
        "changes": {
            key: sum(row["timing"].get("changes", {}).get(key, 0) for row in reviewed)
            for key in ("accepted", "corrected", "rejected", "added")
        },
        "reviewers": sorted({row["reviewer"] for row in reviewed}),
        "note": "Only the latest saved revision counts. Completed reviews and measured intervals "
        "are distinct; missing time is not zero. Fully timed does not prove continuous attention.",
    }


def _cost(view, trial, approach):
    summary = None
    if approach == "multimodal":
        from iris.benchmark_dispatch import dispatch_summary

        summary = dispatch_summary(view, trial["id"])
    elif approach == "combined":
        from iris.benchmark_combined_dispatch import dispatch_summary

        summary = dispatch_summary(view, trial["id"])
    fields = (
        "usage_cost_usd",
        "known_usage_cost_usd",
        "usage_missing_count",
        "unknown_outcome_count",
        "budget_microusd",
        "reserved_microusd",
        "estimated_ceiling_microusd",
    )
    result = {
        "external": summary is not None,
        **{key: summary[key] if summary else None for key in fields},
        "request_counts": summary["counts"] if summary else None,
        "note": summary["message"] if summary else "Local monetary cost is unmeasured, not zero.",
    }
    if approach == "recorded_proposals":
        frames = trial["config"]["recorded_bundle"]["frames"]
        detector = [frame["dinox"]["receipt"]["estimated_cost_cny"] for frame in frames]
        reviewer = [
            frame["review"]["receipt"]["usage_cost_usd"] for frame in frames if "review" in frame
        ]
        result.update(
            recorded=True,
            source_costs={
                "dinox_estimate_cny": sum(detector)
                if all(v is not None for v in detector)
                else None,
                "review_usage_cost_usd": sum(reviewer)
                if reviewer and all(v is not None for v in reviewer)
                else None,
                "detector_receipt_count": len(detector),
                "review_receipt_count": len(reviewer),
                "missing_value_count": sum(v is None for v in detector + reviewer),
            },
            note="No provider request was made by this import. Source costs are submitted "
            "historical estimates, not new charges or authenticated invoices. Shared source "
            "calls across configurations must not be summed as separate spending. "
            "Local import execution cost is unmeasured.",
        )
    return result


def _identity(trial, outputs, config):
    models = set()
    for output in outputs:
        raw = output.get("raw_response")
        if not isinstance(raw, dict):
            continue
        responses = (
            [raw]
            if config["approach"] == "multimodal"
            else [raw.get(stage) for stage in ("planning", "review")]
            if config["approach"] == "combined"
            else [raw.get("review", {}).get("raw_response")]
            if config["approach"] == "recorded_proposals"
            else []
        )
        for response in responses:
            model = response.get("model") if isinstance(response, dict) else None
            if isinstance(model, str) and re.fullmatch(
                r"gpt-6-astra(?:-\d{4}-\d{2}-\d{2})?", model
            ):
                models.add(model)
    plan = trial["config"].get("local_plan") or trial["config"].get("external_plan") or {}
    runtime = plan.get("runtime_identity")
    if runtime is not None:
        from iris.sam_runtime import validate_runtime_identity

        validate_runtime_identity(runtime)
    return {
        "requested_model": config["config"]["model_id"],
        "returned_models": sorted(models),
        "runtime": deepcopy(runtime),
    }


def _validate_trial(trial, job, config, benchmark, frames):
    frozen = trial["config"]
    if (
        trial["config_id"] != config["id"]
        or trial["benchmark_id"] != benchmark["id"]
        or job is None
        or job["kind"] != "benchmark"
        or job["params"].get("trial_id") != trial["id"]
        or frozen.get("protocol") != BENCHMARK_PROTOCOL
        or frozen.get("source_config_fingerprint") != config["fingerprint"]
        or frozen.get("benchmark_manifest_sha256") != benchmark["manifest_sha256"]
        or frozen.get("candidate_config") != config["config"]
        or frozen.get("role") != trial["split"]
        or frozen.get("frame_ids") != [frame["frame_id"] for frame in frames]
    ):
        raise ValueError("Frozen comparison trial provenance is inconsistent")


def _trial_snapshot(view, trial, job, config, benchmark, manifest, outputs, corrections):
    reference = [frame for frame in manifest["frames"] if frame["role"] == trial["split"]]
    _validate_trial(trial, job, config, benchmark, reference)
    by_frame = {row["frame_id"]: row for row in outputs}
    if len(by_frame) != len(outputs) or not set(by_frame) <= {
        frame["frame_id"] for frame in reference
    }:
        raise ValueError("Comparison outputs differ from the frozen role")
    if config["approach"] == "segmentation":
        from iris.benchmark_segmentation import validate_saved_output, validate_trial

        plan = validate_trial(trial["config"], config["config"], reference)
        for output in outputs:
            frame = next(frame for frame in reference if frame["frame_id"] == output["frame_id"])
            validate_saved_output(
                output,
                config=config,
                frame=frame,
                plan=plan,
                attempt=(job["result"] or {}).get("benchmark_attempt_id"),
            )
    if config["approach"] == "recorded_proposals":
        from iris.benchmark_recorded import validate_output_row, validate_trial

        job_result = {} if job["result"] is None else job["result"]
        if (
            job["params"] != {"trial_id": trial["id"], "operation": "import_recorded_proposals"}
            or not isinstance(job_result, dict)
            or job_result
            and (
                job_result.get("trial_id") != trial["id"]
                or job_result.get("operation") != "import_recorded_proposals"
            )
        ):
            raise ValueError("Recorded comparison import job ownership is inconsistent")
        attempt = job_result.get("benchmark_attempt_id")
        if outputs and (not isinstance(attempt, str) or not attempt):
            raise ValueError("Recorded comparison outputs have no owning import attempt")
        bundle = validate_trial(trial["config"], config["config"], reference)
        for output in outputs:
            validate_output_row(
                output,
                trial,
                validated_bundle=bundle,
                attempt=attempt,
            )
    frames = []
    for frame in reference:
        row = by_frame.get(frame["frame_id"])
        proposals = _proposals(row) if row is not None else None
        usable = row is not None and proposals is not None and row["error"] is None
        correction = corrections.get(row["id"]) if row else None
        frames.append(
            {
                "frame_id": frame["frame_id"],
                "output_id": row["id"] if row else None,
                "state": row["metadata"].get(
                    "state", "ready" if usable else "failed" if row["error"] else "pending"
                )
                if row
                else "missing",
                "error": row["error"] if row else None,
                "proposals": proposals,
                "quality": score_proposals(frame, proposals, manifest["taxonomy"])
                if usable
                else None,
                "correction": {key: deepcopy(correction[key]) for key in CORRECTION_KEYS}
                if correction
                else None,
                "elapsed_ms": _number(row["metadata"].get("timing", {}).get("elapsed_ms"))
                if row
                else None,
                "raw_response_sha256": _digest(row["raw_response"]) if row else None,
                "result_sha256": _digest(row["result"]) if row else None,
            }
        )
    measured = [frame["elapsed_ms"] for frame in frames if frame["elapsed_ms"] is not None]
    ready = sum(frame["quality"] is not None for frame in frames)
    failed = sum(frame["error"] is not None for frame in frames)
    approach = config["approach"]
    return {
        "id": trial["id"],
        "job_id": job["id"],
        "status": job["status"],
        "error": job["error"],
        "created_at": trial["created_at"],
        "finished_at": job["finished_at"],
        "coverage": {
            "planned": len(frames),
            "ready": ready,
            "failed": failed,
            "missing": len(frames) - ready - failed,
        },
        "quality": _quality(
            manifest, trial["split"], outputs, job["status"], config["config"]["scoring"]
        ),
        "latency": {
            "measured_count": len(measured),
            "planned_count": len(frames),
            "total_ms": sum(measured) if measured else None,
            "mean_ms": sum(measured) / len(measured) if measured else None,
            "model_load_ms": _number((job["result"] or {}).get("model_load_ms"))
            if approach in {"segmentation", "combined"}
            else None,
            "includes": LATENCY_SCOPES[approach],
            "model_load_included": approach == "combined",
            "note": "Saved image attempts include failures. Missing measurements are not zero. "
            "Timing scopes differ between approaches.",
        },
        "cost": _cost(view, trial, approach),
        "corrections": _corrections(frames),
        "identity": _identity(trial, outputs, config),
        "frames": frames,
    }


def _repeatability(trials, *, recorded=False):
    complete = [trial for trial in trials if trial["quality"]["complete"]]
    geometries = {
        _digest(
            [
                [
                    {key: proposal[key] for key in ("label", "box")}
                    for proposal in frame["proposals"]
                ]
                for frame in trial["frames"]
            ]
        )
        for trial in complete
    }
    metrics = {}
    for key in METRICS:
        values = [
            trial["quality"]["metrics"]["summary"][key]
            for trial in complete
            if trial["quality"]["metrics"]["summary"][key] is not None
        ]
        metrics[key] = {
            "count": len(values),
            "min": min(values) if values else None,
            "mean": sum(values) / len(values) if values else None,
            "max": max(values) if values else None,
        }
    measured = len(complete) >= 2 and not recorded
    return {
        "trial_count": len(trials),
        "complete_count": len(complete),
        "incomplete_count": len(trials) - len(complete),
        "measured": measured,
        "identical_geometry": len(geometries) == 1 if measured else None,
        "distinct_geometry_count": len(geometries) if complete else None,
        "metrics": metrics,
        "mixed_runtime_identity": len({_digest(trial["identity"]["runtime"]) for trial in complete})
        > 1,
        "mixed_returned_models": len(
            {model for trial in complete for model in trial["identity"]["returned_models"]}
        )
        > 1,
        "note": "Repeated imports do not establish repeated provider execution. "
        "Geometry and metric ranges describe submitted evidence only."
        if recorded
        else "Descriptive repeats on the same images, not independent samples "
        "or confidence intervals. "
        "Geometry equality preserves proposal order and ignores IDs, scores and explanations. "
        "One complete trial cannot measure repeatability; model aliases do not identify weights.",
    }


def _coverage(configs):
    trials = [trial for config in configs for trial in config["trials"]]
    approaches = sorted(
        {
            config["approach"]
            for config in configs
            if any(trial["quality"]["complete"] for trial in config["trials"])
        }
    )
    return {
        "config_count": len(configs),
        "trial_count": len(trials),
        "complete_trial_count": sum(trial["quality"]["complete"] for trial in trials),
        "active_trial_count": sum(trial["status"] in ACTIVE for trial in trials),
        "approaches": approaches,
        "missing_approaches": [approach for approach in APPROACHES if approach not in approaches],
    }


def build_comparison(store, benchmark_id, *, role, connection=None):
    """Read one coherent SQLite snapshot; never probe or invoke a provider."""
    if role not in ROLES:
        raise ValueError("Choose the tuning or evaluation role")
    if connection is None:
        with store.connect() as conn:
            conn.execute("BEGIN")
            return build_comparison(store, benchmark_id, role=role, connection=conn)
    if not connection.in_transaction:
        raise ValueError("Comparison reads require a transaction")
    view = _ReadView(connection, store)
    benchmark = view.get("benchmarks", benchmark_id)
    if benchmark is None:
        raise KeyError(benchmark_id)
    manifest = load_benchmark_manifest(view, benchmark_id)
    trial_rows = [
        _decode(row)
        for row in connection.execute(
            "SELECT * FROM benchmark_trials WHERE benchmark_id=? AND split=? "
            "ORDER BY created_at,id LIMIT ?",
            (benchmark_id, role, MAX_TRIALS + 1),
        )
    ]
    if len(trial_rows) > MAX_TRIALS:
        raise ValueError(
            "A comparison supports at most 100 trials in one role; no trials were omitted"
        )
    reference = [frame for frame in manifest["frames"] if frame["role"] == role]
    configs = []
    for raw in connection.execute(
        "SELECT * FROM benchmark_configs WHERE benchmark_id=? ORDER BY created_at,id",
        (benchmark_id,),
    ):
        config = _decode(raw)
        validate_benchmark_config(config, benchmark, manifest)
        trials = []
        for trial in trial_rows:
            if trial["config_id"] != config["id"]:
                continue
            job = _decode(
                connection.execute("SELECT * FROM jobs WHERE id=?", (trial["job_id"],)).fetchone()
            )
            outputs = [
                _decode(row)
                for row in connection.execute(
                    "SELECT * FROM benchmark_outputs WHERE trial_id=? ORDER BY created_at,id",
                    (trial["id"],),
                )
            ]
            corrections = {}
            for output in outputs:
                saved = _decode(
                    connection.execute(
                        "SELECT * FROM benchmark_corrections WHERE output_id=? "
                        "ORDER BY revision DESC LIMIT 1",
                        (output["id"],),
                    ).fetchone()
                )
                if saved:
                    corrections[output["id"]] = saved
            trials.append(
                _trial_snapshot(view, trial, job, config, benchmark, manifest, outputs, corrections)
            )
        configs.append(
            {
                "id": config["id"],
                "name": config["name"],
                "approach": config["approach"],
                "model_id": config["config"]["model_id"],
                "fingerprint": config["fingerprint"],
                "config": deepcopy(config["config"]),
                "trials": trials,
                "repeatability": _repeatability(
                    trials, recorded=config["approach"] == "recorded_proposals"
                ),
            }
        )
    return {
        "protocol": PROTOCOL,
        "benchmark": {
            **{
                key: benchmark[key]
                for key in ("id", "name", "status", "manifest_sha256", "locked_at")
            },
            "taxonomy": deepcopy(manifest["taxonomy"]),
            "independence": deepcopy(manifest["reference"]),
        },
        "role": role,
        "reference": {
            "frames": [
                {
                    **{
                        key: deepcopy(frame[key])
                        for key in (
                            "frame_id",
                            "scene_group",
                            "width",
                            "height",
                            "image_file_sha256",
                            "sha256",
                            "boxes",
                        )
                    },
                    "source_filename": frame["source"]["filename"],
                    "image_url": f"/api/benchmarks/{benchmark_id}/frames/{frame['frame_id']}/image",
                }
                for frame in reference
            ],
            "image_count": len(reference),
            "scene_count": len({frame["scene_group"] for frame in reference}),
            "object_count": sum(len(frame["boxes"]) for frame in reference),
        },
        "scoring": deepcopy(SCORING),
        "configs": configs,
        "coverage": _coverage(configs),
        "warnings": [
            *WARNINGS,
            "All configurations and role attempts are included; successful repeats are "
            "descriptive, not independent measurements.",
            "No automatic ranking: quality, correction time, processing latency and cost "
            "measure different things.",
            "Provider confidence scores are not comparable; this report contains no AP.",
            "Only succeeded complete trials contribute headline quality. Failed or missing "
            "images are never empty predictions.",
            "A saved or simulated output does not by itself establish performance "
            "on real deployment data.",
        ],
    }


def validate_comparison_snapshot(snapshot):
    """Validate derived geometry and aggregates without loading a runtime or current state."""
    try:
        _canonical(snapshot)
        if (
            set(snapshot)
            != {
                "protocol",
                "benchmark",
                "role",
                "reference",
                "scoring",
                "configs",
                "coverage",
                "warnings",
            }
            or snapshot["protocol"] != PROTOCOL
            or snapshot["role"] not in ROLES
            or snapshot["scoring"] != SCORING
        ):
            raise ValueError("Unsupported comparison snapshot")
        reference = snapshot["reference"]
        frames = reference["frames"]
        if set(reference) != {"frames", "image_count", "scene_count", "object_count"} or set(
            snapshot["benchmark"]
        ) != {"id", "name", "status", "manifest_sha256", "locked_at", "taxonomy", "independence"}:
            raise ValueError("Comparison envelope changed")
        for frame in frames:
            if (
                set(frame)
                != {
                    "frame_id",
                    "scene_group",
                    "width",
                    "height",
                    "image_file_sha256",
                    "sha256",
                    "boxes",
                    "source_filename",
                    "image_url",
                }
                or frame["image_url"]
                != f"/api/benchmarks/{snapshot['benchmark']['id']}/frames/{frame['frame_id']}/image"
            ):
                raise ValueError("Comparison reference projection changed")
        manifest = {
            "taxonomy": snapshot["benchmark"]["taxonomy"],
            "frames": [{**frame, "role": snapshot["role"]} for frame in frames],
        }
        if (
            not frames
            or len({frame["frame_id"] for frame in frames}) != len(frames)
            or reference["image_count"] != len(frames)
            or reference["scene_count"] != len({frame["scene_group"] for frame in frames})
            or reference["object_count"] != sum(len(frame["boxes"]) for frame in frames)
        ):
            raise ValueError("Comparison reference counts changed")
        if len({config["id"] for config in snapshot["configs"]}) != len(snapshot["configs"]):
            raise ValueError("Duplicate comparison configuration")
        trial_ids = set()
        for config in snapshot["configs"]:
            if set(config) != {
                "id",
                "name",
                "approach",
                "model_id",
                "fingerprint",
                "config",
                "trials",
                "repeatability",
            }:
                raise ValueError("Comparison configuration projection changed")
            if (
                _digest(config["config"]) != config["fingerprint"]
                or config["config"]["approach"] != config["approach"]
                or config["config"]["model_id"] != config["model_id"]
            ):
                raise ValueError("Comparison configuration fingerprint changed")
            for trial in config["trials"]:
                if trial["id"] in trial_ids:
                    raise ValueError("Duplicate comparison trial")
                trial_ids.add(trial["id"])
                if [frame["frame_id"] for frame in trial["frames"]] != [
                    frame["frame_id"] for frame in frames
                ]:
                    raise ValueError("Comparison trial image coverage changed")
                outputs = []
                for expected, frame in zip(frames, trial["frames"], strict=True):
                    usable = (
                        frame["proposals"] is not None
                        and frame["error"] is None
                        and frame["output_id"] is not None
                    )
                    quality = (
                        score_proposals(expected, frame["proposals"], manifest["taxonomy"])
                        if usable
                        else None
                    )
                    if quality != frame["quality"]:
                        raise ValueError("Comparison frame quality changed")
                    if frame["output_id"] is not None:
                        outputs.append(
                            {
                                "frame_id": frame["frame_id"],
                                "error": frame["error"],
                                "result": {"proposals": frame["proposals"]}
                                if frame["proposals"] is not None
                                else None,
                            }
                        )
                if trial["quality"] != _quality(
                    manifest,
                    snapshot["role"],
                    outputs,
                    trial["status"],
                    config["config"]["scoring"],
                ) or trial["corrections"] != _corrections(trial["frames"]):
                    raise ValueError("Comparison quality or correction aggregation changed")
                ready = sum(frame["quality"] is not None for frame in trial["frames"])
                failed = sum(frame["error"] is not None for frame in trial["frames"])
                if trial["coverage"] != {
                    "planned": len(frames),
                    "ready": ready,
                    "failed": failed,
                    "missing": len(frames) - ready - failed,
                }:
                    raise ValueError("Comparison trial coverage changed")
            if config["repeatability"] != _repeatability(
                config["trials"], recorded=config["approach"] == "recorded_proposals"
            ):
                raise ValueError("Comparison repetition summary changed")
        if len(trial_ids) > MAX_TRIALS or snapshot["coverage"] != _coverage(snapshot["configs"]):
            raise ValueError("Comparison coverage changed")
        return snapshot
    except (KeyError, TypeError, AttributeError, IndexError, OverflowError) as exc:
        raise ValueError("Invalid comparison snapshot") from exc


def validate_snapshot_sources(connection, snapshot, *, manifest=None, store=None, captured_at=None):
    """Validate saved sources by ID, preserving historical correction revisions."""
    validate_comparison_snapshot(snapshot)
    benchmark = _decode(
        connection.execute(
            "SELECT * FROM benchmarks WHERE id=?", (snapshot["benchmark"]["id"],)
        ).fetchone()
    )
    if (
        benchmark is None
        or benchmark["manifest_sha256"] != snapshot["benchmark"]["manifest_sha256"]
    ):
        raise ValueError("Comparison reference identity changed")
    if captured_at is not None:
        cutoff = _timestamp(captured_at)
        expected_configs = [
            row["id"]
            for row in connection.execute(
                "SELECT id,created_at FROM benchmark_configs WHERE benchmark_id=? "
                "ORDER BY created_at,id",
                (benchmark["id"],),
            )
            if _timestamp(row["created_at"]) <= cutoff
        ]
        if expected_configs != [row["id"] for row in snapshot["configs"]]:
            raise ValueError("Comparison omits or reorders configurations present at capture")
        for config in snapshot["configs"]:
            expected_trials = [
                row["id"]
                for row in connection.execute(
                    "SELECT id,created_at FROM benchmark_trials WHERE benchmark_id=? "
                    "AND config_id=? AND split=? ORDER BY created_at,id",
                    (benchmark["id"], config["id"], snapshot["role"]),
                )
                if _timestamp(row["created_at"]) <= cutoff
            ]
            if expected_trials != [row["id"] for row in config["trials"]]:
                raise ValueError("Comparison omits or reorders trials present at capture")
    original = manifest
    manifest = {
        "taxonomy": snapshot["benchmark"]["taxonomy"],
        "frames": [
            {**frame, "role": snapshot["role"]} for frame in snapshot["reference"]["frames"]
        ],
    }
    view = _ReadView(connection, store)
    if original is None and store is not None:
        original = load_benchmark_manifest(view, benchmark["id"])
    if original is not None:
        if (
            original["taxonomy"] != manifest["taxonomy"]
            or original["reference"] != snapshot["benchmark"]["independence"]
        ):
            raise ValueError("Comparison reference definitions changed")
        source_frames = [frame for frame in original["frames"] if frame["role"] == snapshot["role"]]
        if [frame["frame_id"] for frame in source_frames] != [
            frame["frame_id"] for frame in manifest["frames"]
        ]:
            raise ValueError("Comparison reference coverage changed")
        for source, frame in zip(source_frames, manifest["frames"], strict=True):
            if (
                any(
                    source[key] != frame[key]
                    for key in (
                        "frame_id",
                        "scene_group",
                        "width",
                        "height",
                        "image_file_sha256",
                        "sha256",
                        "boxes",
                    )
                )
                or frame["source_filename"] != source["source"]["filename"]
            ):
                raise ValueError("Comparison reference geometry or identity changed")
    for recorded in snapshot["configs"]:
        config = _decode(
            connection.execute(
                "SELECT * FROM benchmark_configs WHERE id=?", (recorded["id"],)
            ).fetchone()
        )
        if (
            config is None
            or config["benchmark_id"] != benchmark["id"]
            or any(
                config[key] != recorded[key]
                for key in ("name", "approach", "fingerprint", "config")
            )
        ):
            raise ValueError("Comparison configuration source changed")
        for saved in recorded["trials"]:
            trial = _decode(
                connection.execute(
                    "SELECT * FROM benchmark_trials WHERE id=?", (saved["id"],)
                ).fetchone()
            )
            if (
                trial is None
                or trial["job_id"] != saved["job_id"]
                or trial["split"] != snapshot["role"]
            ):
                raise ValueError("Comparison trial source changed")
            job = _decode(
                connection.execute("SELECT * FROM jobs WHERE id=?", (trial["job_id"],)).fetchone()
            )
            outputs = [
                _decode(row)
                for row in connection.execute(
                    "SELECT * FROM benchmark_outputs WHERE trial_id=? ORDER BY created_at,id",
                    (trial["id"],),
                )
            ]
            corrections = {}
            for frame in saved["frames"]:
                correction = frame["correction"]
                if correction is not None:
                    row = _decode(
                        connection.execute(
                            "SELECT * FROM benchmark_corrections WHERE id=? "
                            "AND output_id=? AND revision=?",
                            (correction["id"], frame["output_id"], correction["revision"]),
                        ).fetchone()
                    )
                    if row is None:
                        raise ValueError("Comparison correction source is missing")
                    corrections[frame["output_id"]] = row
            if (
                _trial_snapshot(view, trial, job, config, benchmark, manifest, outputs, corrections)
                != saved
            ):
                raise ValueError("Comparison trial or correction evidence changed")
    return snapshot
