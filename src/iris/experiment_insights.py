"""Bounded report summaries derived solely from already validated saved evidence."""

from __future__ import annotations

import math
from pathlib import PurePosixPath

PROTOCOL = "iris-experiment-insights-v1"
MAX_FRAMES = 1000
MAX_SUGGESTIONS = 6
_DELTA_FIELDS = (
    "map",
    "map50",
    "map75",
    "precision",
    "recall",
    "f1",
    "tp",
    "fp",
    "fn",
    "ground_truth_count",
    "prediction_count",
    "frame_count",
)
_RUNTIME_FIELDS = (
    "device",
    "hardware",
    "platform",
    "torch_version",
    "torchvision_version",
    "precision",
    "threads",
    "interop_threads",
)
_REASONS = {
    "mixed": "Improvements and regressions on the same image at the saved operating point",
    "regressed": "New misses or more false positives at the saved operating point",
    "improved": "Recovered objects or fewer false positives at the saved operating point",
    "negative": "False positives on a reviewed image with no labeled objects",
    "single": "Saved misses or false positives in the evaluated pipeline",
}


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _quality_delta(lanes):
    if len(lanes) != 2:
        return None
    baseline, candidate = (lane["metrics"]["summary"] for lane in lanes)
    return {
        key: candidate[key] - baseline[key]
        if _finite(baseline.get(key)) and _finite(candidate.get(key))
        else None
        for key in _DELTA_FIELDS
    }


def _change(counts):
    changes = counts["changes"]
    if changes is None:
        return "single"
    regressed = changes["new_misses"] > 0 or changes["fp_delta"] > 0
    improved = changes["recovered"] > 0 or changes["fp_delta"] < 0
    return (
        "mixed"
        if improved and regressed
        else "regressed"
        if regressed
        else "improved"
        if improved
        else "unchanged"
    )


def _scenes(frames, aggregate):
    groups = {}
    for frame in frames:
        group = groups.setdefault(
            frame["scene_group"],
            {
                "scene_group": frame["scene_group"],
                "frame_count": 0,
                "negative_frame_count": 0,
                "counts": {},
            },
        )
        group["frame_count"] += 1
        group["negative_frame_count"] += frame["counts"][aggregate]["ground_truth_count"] == 0
        for label, counts in frame["counts"].items():
            total = group["counts"].setdefault(
                label,
                {
                    "ground_truth_count": 0,
                    "frame_count": 0,
                    "runs": {
                        run_id: {key: 0 for key in ("tp", "fp", "fn", "error_frames")}
                        for run_id in counts["runs"]
                    },
                    "changes": dict.fromkeys(("new_misses", "recovered", "fp_delta"), 0)
                    if counts["changes"] is not None
                    else None,
                },
            )
            total["ground_truth_count"] += counts["ground_truth_count"]
            total["frame_count"] += 1
            for run_id, run in counts["runs"].items():
                for key in ("tp", "fp", "fn"):
                    total["runs"][run_id][key] += run[key]
                total["runs"][run_id]["error_frames"] += bool(run["fp"] or run["fn"])
            if total["changes"] is not None:
                for key in total["changes"]:
                    total["changes"][key] += counts["changes"][key]
    return [groups[key] for key in sorted(groups)]


def _suggestions(frames, changes, aggregate):
    buckets = {key: [] for key in _REASONS}
    for position, frame in enumerate(frames):
        counts = frame["counts"][aggregate]
        status = changes[frame["frame_id"]]
        errors = sum(run["fp"] + run["fn"] for run in counts["runs"].values())
        severity = (
            sum(counts["changes"][key] for key in ("new_misses", "recovered"))
            + abs(counts["changes"]["fp_delta"])
            if counts["changes"] is not None
            else errors
        )
        item = (severity, position, frame)
        if status in buckets and (status != "single" or errors):
            buckets[status].append(item)
        if not counts["ground_truth_count"] and any(run["fp"] for run in counts["runs"].values()):
            buckets["negative"].append(item)
    for rows in buckets.values():
        rows.sort(key=lambda row: (-row[0], row[1]))
    selected, chosen, scenes = [], set(), set()
    while len(selected) < MAX_SUGGESTIONS:
        added = False
        for category, rows in buckets.items():
            available = [item for item in rows if item[2]["frame_id"] not in chosen]
            if not available:
                continue
            diverse = [item for item in available if item[2]["scene_group"] not in scenes]
            frame = (diverse or available)[0][2]
            selected.append({"frame_id": frame["frame_id"], "reason": _REASONS[category]})
            chosen.add(frame["frame_id"])
            scenes.add(frame["scene_group"])
            added = True
            if len(selected) == MAX_SUGGESTIONS:
                break
        if not added:
            break
    return selected


def _sampling(frames):
    videos, still, unknown = {}, 0, 0
    for frame in frames:
        source = frame["source"]
        if source.get("kind") != "video":
            still += source.get("kind") == "image"
            unknown += source.get("kind") != "image"
            continue
        identity = next(
            (
                source[key]
                for key in ("asset_id", "id", "sha256")
                if isinstance(source.get(key), str) and source[key]
            ),
            f"unknown:{frame['frame_id']}",
        )
        filename = source.get("filename")
        filename = PurePosixPath(filename.replace("\\", "/")).name if filename else "Unknown"
        video = videos.setdefault(
            identity,
            {
                "source_id": identity,
                "filename": filename,
                "frame_count": 0,
                "timestamps_available": 0,
                "first_timestamp_seconds": None,
                "last_timestamp_seconds": None,
                "timestamps_approximate": True,
            },
        )
        video["frame_count"] += 1
        stamp = source.get("timestamp_seconds")
        if _finite(stamp) and stamp >= 0:
            video["timestamps_available"] += 1
            for key, operator in (
                ("first_timestamp_seconds", min),
                ("last_timestamp_seconds", max),
            ):
                video[key] = stamp if video[key] is None else operator(stamp, video[key])
    return {
        "continuous_inference": False,
        "video_sources": list(videos.values()),
        "still_image_count": still,
        "unknown_source_count": unknown,
        "warning": (
            "Results cover only the frozen evaluated images. Video samples are not continuous "
            "video inference or tracking; timestamp bounds do not establish interval coverage. "
            "Recorded video timestamps are approximate (frame index / nominal FPS)."
        ),
    }


def _timing(lanes, frame_count):
    reasons = []
    if len(lanes) != 2:
        reasons.append("A timing comparison requires two evaluated pipelines.")
    for lane in lanes:
        timing = lane.get("timing", {})
        if (
            timing.get("frame_count") != frame_count
            or timing.get("measured_frame_count") != frame_count
            or not _finite(timing.get("mean_total_ms"))
            or timing["mean_total_ms"] < 0
        ):
            reasons.append("Complete timing evidence is unavailable for every frozen image.")
        runtime = lane.get("runtime", {})
        if any(runtime.get(key) in (None, "") for key in _RUNTIME_FIELDS):
            reasons.append("Recorded hardware and runtime settings are incomplete.")
        protocol = runtime.get("timing_protocol", {})
        expected = (
            "torchvision-tiled-v1" if lane.get("variant") == "tiled" else "torchvision-forward-v1"
        )
        if (
            protocol.get("version") != expected
            or protocol.get("batch_size") != 1
            or protocol.get("warmup_in_timings") is not False
            or not protocol.get("total_ms")
            or not protocol.get("decode_ms")
        ):
            reasons.append("The complete recorded IRIS timing protocol is unavailable.")
        if str(runtime.get("device", "")).startswith("cuda"):
            cuda = runtime.get("cuda", {})
            if (
                not isinstance(cuda, dict)
                or not {
                    "runtime",
                    "cudnn",
                    "index",
                    "name",
                    "capability",
                    "total_memory",
                    "tf32_matmul",
                    "tf32_cudnn",
                    "cudnn_benchmark",
                }
                <= cuda.keys()
            ):
                reasons.append("Recorded CUDA runtime and device settings are incomplete.")
    if len(lanes) == 2:
        first, second = (lane.get("runtime", {}) for lane in lanes)
        if any(first.get(key) != second.get(key) for key in (*_RUNTIME_FIELDS, "platform", "cuda")):
            reasons.append("The recorded hardware or runtime settings differ between pipelines.")
        left, right = (runtime.get("timing_protocol", {}) for runtime in (first, second))
        if lanes[0].get("variant") == lanes[1].get("variant"):
            if left != right:
                reasons.append("The recorded timing protocols differ between pipelines.")
        elif any(
            left.get(key) != right.get(key)
            for key in ("batch_size", "warmup_in_timings", "decode_ms")
        ):
            reasons.append("The recorded whole-pipeline timing boundaries differ.")
    return {
        "comparable": not reasons,
        "reasons": list(dict.fromkeys(reasons)),
        "scope": (
            "Saved IRIS whole-pipeline time on the same frozen images, including decoding "
            "and verification; tiled runs also include crops, all passes and merging. "
            "Weight loading, warmup and database writes are excluded. This is not target "
            "deployment performance or a controlled hardware benchmark."
        ),
    }


def build_insights(snapshot, available_examples, detail):
    """Aggregate validated operating-point evidence; do not recompute AP or matching."""
    if not 1 <= len(available_examples) <= MAX_FRAMES:
        raise ValueError("Report insights require 1–1000 evaluated frames")
    frame_ids = [frame["frame_id"] for frame in available_examples]
    if len(set(frame_ids)) != len(frame_ids) or frame_ids != [
        frame["frame_id"] for frame in detail["frames"]
    ]:
        raise ValueError("Report insight frames differ from the frozen evaluation")
    lanes = snapshot["lanes"]
    if not 1 <= len(lanes) <= 2:
        raise ValueError("Report insights require one or two evaluated pipelines")
    aggregate = snapshot["error_analysis"].get("aggregate_filter", "all")
    changes = {
        frame["frame_id"]: _change(frame["counts"][aggregate]) for frame in available_examples
    }
    return {
        "protocol": PROTOCOL,
        "quality_delta": _quality_delta(lanes),
        "scenes": _scenes(available_examples, aggregate),
        "frame_changes": changes,
        "suggested_examples": _suggestions(available_examples, changes, aggregate),
        "sampling": _sampling(detail["frames"]),
        "timing": _timing(lanes, len(available_examples)),
    }
