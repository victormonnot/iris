"""Portable, bounded cost evidence: validate declarations without running ML."""

import hashlib
import json
import math
import statistics
from copy import deepcopy
from datetime import datetime

from iris.temporal_contracts import sequence_hash, validate_sequence_manifest
from iris.temporal_detector import validate_detector_config, validate_execution_signature
from iris.tracking_contracts import profile_hash, validate_profile

REPORT_SCHEMA = "iris-tracking-cost-v1"
MAX_REPORT_BYTES = 24 * 1024**2
MAX_FRAMES = 500
MAX_REPEATS = 5
TIMING_FIELDS = (
    "pipeline_ms",
    "verify_decode_ms",
    "detector_call_ms",
    "filter_ms",
    "tracking_image_ms",
    "tracker_call_ms",
    "detector_preprocess_ms",
    "detector_inference_ms",
    "detector_postprocess_ms",
    "detector_crop_ms",
    "detector_merge_ms",
    "tracker_gmc_ms",
    "tracker_association_ms",
    "tracker_adapter_ms",
)
OUTER_TIMINGS = (
    "verify_decode_ms",
    "detector_call_ms",
    "filter_ms",
    "tracking_image_ms",
    "tracker_call_ms",
)
PROTOCOL = {
    "name": "iris-saved-frame-cost-v1",
    "pipeline": (
        "Verified saved PNG decode, fresh detection, filtering and tracker update; batch one"
    ),
    "timing": "Monotonic wall clock; outer calls include adapters; "
    "detector CUDA stage boundaries synchronized",
    "nested_stages": "Detector and tracker substages are nested within outer calls; "
    "do not add them twice",
    "loop_wall": "Repetition wall time additionally includes sampling, "
    "bookkeeping and progress callbacks",
    "scheduling": "Single virtual worker; latest available arrival at or before finish wins; "
    "drain latest pending; no sleep",
    "arrival": "Original source frame index offset divided by explicitly assumed cadence FPS",
    "simulated_output_fps": "Completed-output intervals: sum(n-1) divided by "
    "sum(last finish minus first finish); undefined without an interval",
    "tracker_clock": "One native update per processed frame; "
    "no synthetic updates for dropped or absent source frames",
    "warmup": "One first-frame pipeline before measurement; "
    "tracker reset before each measured repetition",
    "rss": "Linux process RSS sampled at frame boundaries; "
    "highwater is separate process-lifetime RSS including startup",
    "cuda_memory": "Selected device PyTorch allocated and reserved peaks reset after warmup "
    "per repetition; includes resident model and allocator baseline",
    "percentile": "p95 nearest rank: sorted sample at ceil(0.95*n)-1; "
    "median uses arithmetic midpoint",
    "excluded": "Camera acquisition, transport, video demux, UI, "
    "report publication and progress callbacks",
}
LIMITATIONS = [
    "This is a saved-frame pipeline measurement, not a real camera or flight measurement.",
    "Cadence is a virtual latest-frame simulation with explicitly assumed arrivals, "
    "not measured capture timing.",
    "Missing source frames and available frames dropped by the simulation remain distinct.",
    "No cached detector timing is added to infer complete pipeline performance; "
    "the detector executes afresh.",
    "No quality score is inferred from runtime or dropped frames; "
    "the new observations can differ from the old comparison.",
    "Boundary RSS sampling can miss short peaks. Process lifetime highwater includes imports, "
    "setup and warmup; it is not a per-stage allocation.",
    "CUDA figures describe this process's PyTorch allocator, "
    "not total device memory or other applications.",
    "Operating system caches, thermal state and concurrent workloads affect timings; "
    "small samples do not establish deployment capacity.",
    "Imported execution and hardware remain producer declarations; "
    "checksums and structural validation are not execution authentication.",
]


def _json(value):
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
    except (ValueError, TypeError, OverflowError, UnicodeError, RecursionError) as exc:
        raise ValueError("Cost evidence must be finite UTF-8 JSON") from exc


def digest(value):
    return hashlib.sha256(_json(value)).hexdigest()


def _object(value, keys, description):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"{description} must contain exactly its documented fields")


def _integer(value, description, *, maximum=2**53 - 1, minimum=0):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{description} must be an integer in its documented range")
    return value


def _number(value, description, *, maximum=1e12, minimum=0):
    try:
        valid = type(value) in (float, int) and math.isfinite(value) and minimum <= value <= maximum
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{description} must be a finite number in its documented range")
    return float(value)


def _text(value, description, *, maximum=2000):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f"{description} must be nonempty bounded text")
    return value


def _hash(value, description):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"{description} must be a lowercase SHA-256")


def _same(value, expected, description):
    if _json(value) != _json(expected):
        raise ValueError(f"{description} does not match the frozen evidence")


def validate_cost_request(config):
    _object(config, {"lane_index", "device", "repeats", "policy", "cadence_fps"}, "Cost request")
    _integer(config["lane_index"], "Lane index", maximum=1)
    _integer(config["repeats"], "Repetitions", minimum=1, maximum=MAX_REPEATS)
    if config["device"] not in ("cpu", "cuda"):
        raise ValueError("Cost device must be cpu or cuda; no automatic fallback")
    if config["policy"] not in ("offline_all", "simulated_latest"):
        raise ValueError("Cost policy must be offline_all or simulated_latest")
    result = deepcopy(config)
    if config["policy"] == "offline_all":
        if config["cadence_fps"] is not None:
            raise ValueError("Offline processing must not declare a simulated cadence")
    else:
        result["cadence_fps"] = _number(
            config["cadence_fps"], "Simulated cadence", minimum=0.1, maximum=240
        )
    return result


def detector_recipe(config):
    return {
        key: deepcopy(value) for key, value in config.items() if key not in {"device", "runtime"}
    }


def source_binding(comparison, lane_index):
    report = comparison["report"]
    sequence = validate_sequence_manifest(report["sequence"])
    if not 1 <= len(sequence["frames"]) <= MAX_FRAMES:
        raise ValueError(f"Cost measurement accepts 1–{MAX_FRAMES} available source frames")
    lane = report["lanes"][lane_index]
    cache = lane["report"]["cache"]
    return {
        "comparison_id": comparison["id"],
        "sequence_id": sequence["id"],
        "sequence_sha256": sequence_hash(sequence),
        "cache_id": cache["id"],
        "cache_fingerprint": cache["fingerprint"],
        "cache_result_sha256": cache["result_sha256"],
        "lane_index": lane_index,
        "lane_name": lane["name"],
        "profile_sha256": profile_hash(lane["report"]["profile"]),
        "detector_recipe_sha256": digest(detector_recipe(cache["config"]["detector"])),
    }


def next_frame(indices, position, virtual_ms, request):
    """Return selected available position, skipped available indices and virtual start."""
    if request["policy"] == "offline_all":
        return position, [], None
    arrival = (indices[position] - indices[0]) * 1000 / request["cadence_fps"]
    start = max(virtual_ms, arrival)
    selected = position
    while selected + 1 < len(indices):
        following = (indices[selected + 1] - indices[0]) * 1000 / request["cadence_fps"]
        if following > start:
            break
        selected += 1
    return selected, indices[position:selected], start


def frame_schedule(frame_index, first_index, start_ms, service_ms, request):
    if request["policy"] == "offline_all":
        return None
    arrival = (frame_index - first_index) * 1000 / request["cadence_fps"]
    finish = start_ms + service_ms
    return {
        "arrival_ms": arrival,
        "start_ms": start_ms,
        "finish_ms": finish,
        "queue_delay_ms": start_ms - arrival,
        "latency_ms": finish - arrival,
    }


def distribution(values):
    ordered = sorted(values)
    if not ordered:
        return None
    total = sum(values)
    return {
        "count": len(values),
        "min": ordered[0],
        "median": statistics.median(ordered),
        "p95": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "max": ordered[-1],
        "mean": total / len(values),
        "total": total,
    }


def _maximum(values):
    known = [value for value in values if value is not None]
    return max(known) if known else None


def summarize(report):
    passes = report["passes"]
    frames = [frame for run in passes for frame in run["frames"]]
    pipeline_ms = sum(frame["timing"]["pipeline_ms"] for frame in frames)
    simulated = report["request"]["policy"] == "simulated_latest"
    finish_ms = (
        sum(
            run["frames"][-1]["schedule"]["finish_ms"] - run["frames"][0]["schedule"]["finish_ms"]
            for run in passes
        )
        if simulated
        else 0
    )
    output_intervals = sum(len(run["frames"]) - 1 for run in passes)
    memory = [run["memory"] for run in passes]
    cuda = [item["cuda"] for item in memory if item["cuda"] is not None]
    manifest = report["sequence"]
    available = len(manifest["frames"])
    return {
        "sample_count": len(frames),
        "repeats": len(passes),
        "available_frames_per_pass": available,
        "processed_frames": len(frames),
        "dropped_frames": sum(len(run["dropped_frame_indices"]) for run in passes),
        "source_gap_frames_per_pass": manifest["clip"]["end_frame"]
        - manifest["clip"]["start_frame"]
        + 1
        - available,
        "stages_ms": {
            key: distribution([frame["timing"][key] for frame in frames]) for key in TIMING_FIELDS
        },
        "service_fps": len(frames) * 1000 / pipeline_ms if pipeline_ms else None,
        "simulated_latency_ms": distribution([frame["schedule"]["latency_ms"] for frame in frames])
        if simulated
        else None,
        "simulated_output_fps": output_intervals * 1000 / finish_ms if finish_ms else None,
        "wall_ms": sum(run["wall_ms"] for run in passes),
        "memory": {
            "rss_sampled_peak_bytes": _maximum(item["rss_sampled_peak_bytes"] for item in memory),
            "process_lifetime_peak_bytes": _maximum(
                item["process_lifetime_peak_bytes"] for item in memory
            ),
            "cuda_allocated_peak_bytes": _maximum(item["allocated_peak_bytes"] for item in cuda),
            "cuda_reserved_peak_bytes": _maximum(item["reserved_peak_bytes"] for item in cuda),
        },
    }


def _validate_execution(execution, detector, sequence, profile):
    _object(
        execution,
        {
            "started_at",
            "host",
            "detector_signature",
            "tracker_metadata",
            "tracker_metadata_sha256",
            "pipeline_sources",
            "setup_ms",
            "warmup",
        },
        "Cost execution",
    )
    _text(execution["started_at"], "Execution start", maximum=100)
    try:
        parsed = datetime.fromisoformat(execution["started_at"])
        if parsed.tzinfo is None:
            raise ValueError("Missing timezone")
    except ValueError as exc:
        raise ValueError("Execution start must have an explicit timezone") from exc
    host = execution["host"]
    _object(
        host,
        {"cpu", "platform", "machine", "python", "logical_cpus", "affinity_cpus"},
        "Execution host",
    )
    for key in ("cpu", "platform", "machine", "python"):
        _text(host[key], f"Host {key}")
    if host["logical_cpus"] is not None:
        _integer(host["logical_cpus"], "Logical CPUs", minimum=1, maximum=65536)
    affinity = host["affinity_cpus"]
    if affinity is not None:
        if not isinstance(affinity, list) or not 1 <= len(affinity) <= 65536:
            raise ValueError("CPU affinity must be a bounded list or unknown")
        for cpu in affinity:
            _integer(cpu, "Affinity CPU", maximum=65535)
        if affinity != sorted(set(affinity)):
            raise ValueError("CPU affinity IDs must be sorted and distinct")
    validate_execution_signature(detector, execution["detector_signature"])
    metadata = execution["tracker_metadata"]
    if (
        not isinstance(metadata, dict)
        or metadata.get("algorithm") != profile["algorithm"]
        or metadata.get("schema") != "iris-tracker-runtime-v1"
    ):
        raise ValueError("Tracker execution metadata must match the selected algorithm")
    policy = metadata.get("execution_policy")
    if not isinstance(policy, dict):
        raise ValueError("Tracker execution policy must be an object")
    for key, expected in {
        "device": "cpu",
        "learned_reid": False,
        "opencv_threads": profile["opencv_threads"],
        "seed": profile["seed"],
        "native_buffer_unit": "available_frame_updates",
        "native_time_step": 1,
        "skipped_source_frames": "no_synthetic_updates",
    }.items():
        _same(policy.get(key), expected, f"Tracker execution policy {key}")
    _same(execution["tracker_metadata_sha256"], digest(metadata), "Tracker metadata hash")
    _object(
        execution["pipeline_sources"],
        {"tracking_cost_runtime.py", "tracking_cost_contracts.py"},
        "Cost pipeline sources",
    )
    for value in execution["pipeline_sources"].values():
        _hash(value, "Cost pipeline source hash")
    _object(
        execution["setup_ms"],
        {"detector_load_ms", "tracker_setup_ms", "warmup_ms"},
        "Cost setup timings",
    )
    for value in execution["setup_ms"].values():
        _number(value, "Setup duration")
    _same(
        execution["warmup"],
        {"frame_id": sequence["frames"][0]["frame_id"], "passes": 1, "tracker_reset": True},
        "Cost warmup",
    )


def _validate_memory(memory, device, signature):
    rss = {
        "rss_start_bytes",
        "rss_end_bytes",
        "rss_sampled_peak_bytes",
        "process_lifetime_peak_bytes",
    }
    _object(memory, rss | {"cuda"}, "Cost memory")
    for key in rss:
        if memory[key] is not None:
            _integer(memory[key], key)
    sampled = memory["rss_sampled_peak_bytes"]
    if sampled is None and any(
        memory[key] is not None for key in ("rss_start_bytes", "rss_end_bytes")
    ):
        raise ValueError("Known RSS boundary samples require a sampled peak")
    if sampled is not None and any(
        value is not None and value > sampled
        for value in (memory["rss_start_bytes"], memory["rss_end_bytes"])
    ):
        raise ValueError("Sampled RSS peak cannot be below a recorded boundary sample")
    cuda = memory["cuda"]
    if device == "cpu":
        if cuda is not None:
            raise ValueError("CPU measurement cannot claim CUDA memory")
        return
    _object(
        cuda,
        {
            "device",
            "allocated_start_bytes",
            "reserved_start_bytes",
            "allocated_peak_bytes",
            "reserved_peak_bytes",
        },
        "CUDA allocator memory",
    )
    if cuda["device"] != signature["device"]:
        raise ValueError("CUDA memory must use the actual selected detector device")
    for key in set(cuda) - {"device"}:
        _integer(cuda[key], key)
    if (
        cuda["allocated_peak_bytes"] < cuda["allocated_start_bytes"]
        or cuda["reserved_peak_bytes"] < cuda["reserved_start_bytes"]
        or cuda["reserved_peak_bytes"] < cuda["allocated_peak_bytes"]
        or cuda["reserved_start_bytes"] < cuda["allocated_start_bytes"]
    ):
        raise ValueError("CUDA allocator peaks and baselines are inconsistent")


def validate_cost_report(report, comparison):
    """Check complete source binding and recompute schedule/stats without optional runtimes."""
    if len(_json(report)) > MAX_REPORT_BYTES:
        raise ValueError("Cost report exceeds the portable JSON size limit")
    _object(
        report,
        {
            "schema",
            "complete",
            "request",
            "source",
            "sequence",
            "profile",
            "detector_config",
            "execution",
            "passes",
            "summary",
            "protocol",
            "limitations",
        },
        "Cost report",
    )
    if report["schema"] != REPORT_SCHEMA or report["complete"] is not True:
        raise ValueError("Cost report must be a complete supported schema")
    request = validate_cost_request(report["request"])
    _same(report["request"], request, "Canonical cost request")
    source = source_binding(comparison, request["lane_index"])
    _same(report["source"], source, "Cost source binding")
    sequence = validate_sequence_manifest(report["sequence"])
    _same(sequence, comparison["report"]["sequence"], "Frozen cost sequence")
    lane = comparison["report"]["lanes"][request["lane_index"]]["report"]
    profile = validate_profile(report["profile"])
    _same(profile, lane["profile"], "Frozen tracker profile")
    detector = validate_detector_config(report["detector_config"])
    _same(
        detector_recipe(detector),
        detector_recipe(lane["cache"]["config"]["detector"]),
        "Frozen detector recipe",
    )
    if detector["device"] != request["device"]:
        raise ValueError("Detector device differs from the explicit cost request")
    _validate_execution(report["execution"], detector, sequence, profile)
    _same(report["protocol"], PROTOCOL, "Cost protocol")
    _same(report["limitations"], LIMITATIONS, "Cost limitations")
    passes = report["passes"]
    if not isinstance(passes, list) or len(passes) != request["repeats"]:
        raise ValueError("Cost report must contain every requested repetition")
    sources = sequence["frames"]
    indices = [frame["frame_index"] for frame in sources]
    for pass_index, run in enumerate(passes):
        _object(
            run,
            {"pass_index", "frames", "dropped_frame_indices", "wall_ms", "memory"},
            "Cost repetition",
        )
        _integer(run["pass_index"], "Pass index", maximum=MAX_REPEATS - 1)
        if run["pass_index"] != pass_index:
            raise ValueError("Cost repetition order is invalid")
        frames = run["frames"]
        if not isinstance(frames, list) or not 1 <= len(frames) <= len(sources):
            raise ValueError("Cost repetition must contain a bounded nonempty frame list")
        _number(run["wall_ms"], "Repetition wall duration")
        _validate_memory(
            run["memory"], request["device"], report["execution"]["detector_signature"]
        )
        position, virtual_ms, dropped = 0, 0.0, []
        for frame in frames:
            if position >= len(indices):
                raise ValueError("Cost report processes more frames than its frozen source")
            selected, omitted, start_ms = next_frame(indices, position, virtual_ms, request)
            dropped.extend(omitted)
            frozen = sources[selected]
            _object(
                frame,
                {
                    "frame_id",
                    "frame_index",
                    "timestamp_seconds",
                    "input_size",
                    "timing",
                    "schedule",
                    "work",
                    "outputs_sha256",
                },
                "Cost frame sample",
            )
            _integer(frame["frame_index"], "Source frame index")
            _same(
                {key: frame[key] for key in ("frame_id", "frame_index", "timestamp_seconds")},
                {key: frozen[key] for key in ("frame_id", "frame_index", "timestamp_seconds")},
                "Measured source frame",
            )
            _same(
                frame["input_size"],
                [frozen["width"], frozen["height"]],
                "Measured source dimensions",
            )
            _hash(frame["outputs_sha256"], "Fresh output semantic hash")
            timing = frame["timing"]
            _object(timing, TIMING_FIELDS, "Cost frame timings")
            for value in timing.values():
                _number(value, "Frame stage duration")
            tolerance = 1e-5
            if sum(timing[key] for key in OUTER_TIMINGS) > timing["pipeline_ms"] + tolerance:
                raise ValueError("Outer stage durations exceed the same-frame pipeline duration")
            detector_nested = sum(
                timing[key]
                for key in (
                    "detector_preprocess_ms",
                    "detector_inference_ms",
                    "detector_postprocess_ms",
                    "detector_crop_ms",
                    "detector_merge_ms",
                )
            )
            if detector_nested > timing["detector_call_ms"] + tolerance:
                raise ValueError("Nested detector stages exceed the detector call duration")
            if (
                timing["tracker_gmc_ms"] + timing["tracker_association_ms"]
                > timing["tracker_adapter_ms"] + tolerance
                or timing["tracker_adapter_ms"] > timing["tracker_call_ms"] + tolerance
            ):
                raise ValueError("Nested tracker stages exceed the tracker call duration")
            schedule = frame_schedule(
                frozen["frame_index"], indices[0], start_ms, timing["pipeline_ms"], request
            )
            _same(frame["schedule"], schedule, "Simulated latest-frame schedule")
            if schedule is not None:
                virtual_ms = schedule["finish_ms"]
            work = frame["work"]
            _object(
                work,
                {
                    "detection_count",
                    "observation_count",
                    "prediction_count",
                    "unassigned_count",
                    "forward_passes",
                    "tile_count",
                },
                "Frame work counts",
            )
            for key, value in work.items():
                _integer(value, key, maximum=150000)
            if (
                work["detection_count"] != work["observation_count"] + work["unassigned_count"]
                or work["detection_count"] > 300
                or work["forward_passes"] < 1
            ):
                raise ValueError("Frame detection dispositions or detector work are inconsistent")
            if detector["inference"]["mode"] == "full" and (
                work["forward_passes"] != 1 or work["tile_count"] != 0
            ):
                raise ValueError("Full-image detection must contain exactly one forward pass")
            if detector["inference"]["mode"] == "full" and work["detection_count"] > 100:
                raise ValueError("Full-image detection exceeds its frozen output cap")
            if detector["inference"]["mode"] == "tiled":
                from iris.tiling import tile_boxes

                tile_count = len(
                    tile_boxes(frozen["width"], frozen["height"], detector["inference"]["tiling"])
                )
                if work["tile_count"] != tile_count or work["forward_passes"] != tile_count:
                    raise ValueError("Tiled detector work must match its frozen crop geometry")
            if profile["gmc_method"] == "none" and timing["tracking_image_ms"] != 0:
                raise ValueError("Disabled GMC must not claim image-conversion work")
            position = selected + 1
        if position != len(sources):
            raise ValueError("Cost report omits pending source frames or the last pending arrival")
        _same(run["dropped_frame_indices"], dropped, "Simulated dropped available frames")
        if sum(frame["timing"]["pipeline_ms"] for frame in frames) > run["wall_ms"] + 1e-5:
            raise ValueError("Frame pipeline durations exceed the repetition wall duration")
    _same(report["summary"], summarize(report), "Cost summary")
    return deepcopy(report)


def cost_status():
    return {
        "schema": REPORT_SCHEMA,
        "protocol": deepcopy(PROTOCOL),
        "limits": {
            "max_frames": MAX_FRAMES,
            "max_repeats": MAX_REPEATS,
            "max_report_bytes": MAX_REPORT_BYTES,
            "min_cadence_fps": 0.1,
            "max_cadence_fps": 240,
        },
    }
