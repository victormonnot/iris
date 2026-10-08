"""Pure, bounded profile-study requests and reproducible result summaries.

Development and validation are tuning inputs. Reserved test sequences are never
read by the runner, and repeated replays never multiply reference observations.
"""

import json
import math
from copy import deepcopy

from iris.temporal import _digest
from iris.tracking_contracts import make_profile, profile_hash, validate_profile
from iris.tracking_cost_contracts import detector_recipe, distribution
from iris.tracking_metrics import evaluate_quality

REPORT_SCHEMA = "iris-tracking-study-v1"
MAX_CANDIDATES = 7
MAX_SOURCES = 4
MAX_FRAMES = 500
MAX_UPDATES = 20_000
MAX_SECONDS = 600
MAX_REPORT_BYTES = 48 * 1024**2
LIMITATIONS = [
    "Development and validation both inform tuning; neither is an independent test.",
    "Reserved test entries are metadata only: no tracker replay or reference scoring.",
    "Repeated replays measure timing and observed repeatability, not additional labeled data.",
    "Human review does not establish an independent reference; "
    "assisted seeds may share tracker outputs.",
    "Only human-complete geometrically evaluable frames count; "
    "missing or uncertain frames are not negatives.",
    "Buffer lengths count available-frame updates, not seconds. "
    "Source gaps receive no synthetic updates.",
    "Timings describe cached-input tracker replays on this host, including cold first updates. "
    "They exclude detector inference, complete-pipeline latency, capture cadence and memory.",
    "Timing differences are descriptive observations without confidence intervals "
    "or a speed guarantee.",
    "No profile is applied automatically. "
    "A separate target-device pipeline measurement remains necessary.",
]
REQUEST_FIELDS = {
    "name",
    "dataset_id",
    "sources",
    "baseline",
    "candidates",
    "class_mapping",
    "iou_threshold",
    "repeats",
    "max_updates",
    "max_seconds",
}


def _object(value, fields, name):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ValueError(f"{name} must contain exactly its documented fields")


def _text(value, name, maximum=160):
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > maximum:
        raise ValueError(f"{name} must be a nonempty bounded string")
    try:
        value.encode("utf-8")
    except UnicodeError as exc:
        raise ValueError(f"{name} must be valid UTF-8") from exc
    return value.strip()


def _integer(value, name, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer between {minimum} and {maximum}")
    return value


def _number(value, name, minimum, maximum):
    try:
        valid = type(value) in (int, float) and math.isfinite(value) and minimum <= value <= maximum
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{name} must be a finite number in its documented range")
    return float(value)


def canonicalize_request(payload):
    _object(payload, REQUEST_FIELDS, "Tracking study request")
    result = deepcopy(payload)
    result["name"] = _text(payload["name"], "Study name")
    result["dataset_id"] = _text(payload["dataset_id"], "Dataset ID")
    sources = payload["sources"]
    if not isinstance(sources, list) or not 1 <= len(sources) <= MAX_SOURCES:
        raise ValueError(f"Select 1–{MAX_SOURCES} development/validation sequences")
    checked_sources = []
    for source in sources:
        _object(source, {"sequence_id", "comparison_id"}, "Study source")
        checked_sources.append({key: _text(value, key) for key, value in source.items()})
    if len({item["sequence_id"] for item in checked_sources}) != len(checked_sources):
        raise ValueError("Study source sequences must be distinct")
    result["sources"] = sorted(checked_sources, key=lambda item: item["sequence_id"])
    candidates = payload["candidates"]
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= MAX_CANDIDATES:
        raise ValueError(f"Choose 1–{MAX_CANDIDATES} explicit candidate profiles")
    profiles = []
    for item in [payload["baseline"], *candidates]:
        _object(item, {"name", "profile"}, "Named tracker profile")
        profiles.append(
            {
                "name": _text(item["name"], "Profile name", 80),
                "profile": validate_profile(item["profile"]),
            }
        )
    if len({profile_hash(item["profile"]) for item in profiles}) != len(profiles):
        raise ValueError("Baseline and candidate profiles must have distinct canonical hashes")
    if len({item["name"] for item in profiles}) != len(profiles):
        raise ValueError("Baseline and candidate names must be distinct")
    classes = profiles[0]["profile"]["class_ids"]
    if any(item["profile"]["class_ids"] != classes for item in profiles):
        raise ValueError("All study profiles must use the same native classes")
    for item in profiles[1:]:
        if any(
            item["profile"][key] != profiles[0]["profile"][key]
            for key in ("seed", "opencv_threads", "gmc_downscale", "time_policy", "with_reid")
        ):
            raise ValueError("Study candidates must preserve the baseline execution controls")
    result["baseline"], result["candidates"] = profiles[0], profiles[1:]
    mapping = payload["class_mapping"]
    if not isinstance(mapping, dict) or set(mapping) != {str(value) for value in classes}:
        raise ValueError("Map every study native class to a taxonomy ID or null")
    result["class_mapping"] = {
        key: None if value is None else _text(value, "Taxonomy class ID")
        for key, value in sorted(mapping.items(), key=lambda item: int(item[0]))
    }
    if not any(value is not None for value in result["class_mapping"].values()):
        raise ValueError("The study requires at least one included class")
    result["iou_threshold"] = _number(payload["iou_threshold"], "IoU threshold", 1e-12, 1)
    result["repeats"] = _integer(payload["repeats"], "Replay repetitions", 1, 3)
    result["max_updates"] = _integer(payload["max_updates"], "Update budget", 1, MAX_UPDATES)
    result["max_seconds"] = _number(payload["max_seconds"], "Time budget", 1, MAX_SECONDS)
    return result


def suggestions(baseline_profile):
    baseline = validate_profile(baseline_profile)
    output, seen = [], {profile_hash(baseline)}

    def add(name, **changes):
        profile = {**deepcopy(baseline), **changes}
        if profile["algorithm"] == "bytetrack":
            profile["new_track_threshold"] = profile["high_threshold"] + 0.1
        else:
            profile["new_track_threshold"] = max(
                profile["new_track_threshold"], profile["high_threshold"]
            )
        profile = validate_profile(profile)
        digest = profile_hash(profile)
        if digest not in seen:
            output.append({"name": name, "profile": profile})
            seen.add(digest)

    low = max(
        (baseline["low_threshold"] + baseline["high_threshold"]) / 2,
        baseline["high_threshold"] - 0.15,
    )
    high = min(
        0.9 if baseline["algorithm"] == "bytetrack" else 1, baseline["high_threshold"] + 0.15
    )
    if baseline["low_threshold"] < low < baseline["high_threshold"]:
        add("Lower confidence threshold", high_threshold=low)
    add("Higher confidence threshold", high_threshold=round(high, 12))
    add(
        "Longer lost-track buffer",
        buffer_updates=min(10000, max(60, baseline["buffer_updates"] * 2)),
    )
    add("Shorter lost-track buffer", buffer_updates=max(0, baseline["buffer_updates"] // 3))
    add(
        "Stricter association",
        match_threshold=max(0, round(baseline["match_threshold"] - 0.15, 12)),
    )
    if baseline["algorithm"] == "botsort":
        method = "none" if baseline["gmc_method"] != "none" else "sparseOptFlow"
        add("Camera compensation " + ("off" if method == "none" else "on"), gmc_method=method)
    else:
        alternate = make_profile(
            "botsort",
            class_ids=baseline["class_ids"],
            gmc_method="none",
            seed=baseline["seed"],
            opencv_threads=baseline["opencv_threads"],
        )
        if profile_hash(alternate) not in seen:
            output.append({"name": "BoT-SORT without camera compensation", "profile": alternate})
            seen.add(profile_hash(alternate))
    if len(output) < MAX_CANDIDATES:
        alternate = make_profile(
            "botsort" if baseline["algorithm"] == "bytetrack" else "bytetrack",
            class_ids=baseline["class_ids"],
            seed=baseline["seed"],
            opencv_threads=baseline["opencv_threads"],
        )
        if profile_hash(alternate) not in seen:
            output.append(
                {
                    "name": "BoT-SORT with camera compensation"
                    if alternate["algorithm"] == "botsort"
                    else "ByteTrack alternative",
                    "profile": alternate,
                }
            )
    return {"candidates": output[:MAX_CANDIDATES]}


def budget_for(request, sources):
    counts = [len(source["sequence"]["manifest"]["frames"]) for source in sources]
    if len(counts) != len(request["sources"]) or any(
        not 1 <= count <= MAX_FRAMES for count in counts
    ):
        raise ValueError(f"Study sources require 1–{MAX_FRAMES} available frames each")
    required = (1 + len(request["candidates"])) * sum(counts) * request["repeats"]
    if required > request["max_updates"]:
        raise ValueError(
            f"Study needs {required} updates, above its explicit "
            f"{request['max_updates']} update budget"
        )
    return {
        "profiles": 1 + len(request["candidates"]),
        "sequences": len(sources),
        "frames_per_pass": sum(counts),
        "repeats": request["repeats"],
        "required_updates": required,
        "max_updates": request["max_updates"],
        "max_seconds": request["max_seconds"],
    }


def source_bindings(bundle):
    return [
        {
            "entry": deepcopy(source["entry"]),
            "comparison_id": source["comparison"]["id"],
            "cache_id": source["cache"]["id"],
            "cache_fingerprint": source["cache"]["fingerprint"],
            "result_sha256": source["comparison"]["report"]["lanes"][0]["report"]["cache"][
                "result_sha256"
            ],
            "reference_revision": source["reference"]["revision"],
            "detector_recipe_sha256": _digest(
                detector_recipe(source["cache"]["config"]["detector"])
            ),
        }
        for source in bundle["sources"]
    ]


def _timing(replays):
    passes = [item for replay in replays for item in replay["passes"]]
    frames = [frame for item in passes for frame in item["frames"]]
    return {
        "tracker_ms": distribution([frame["timing"]["total_ms"] for frame in frames]),
        "gmc_ms": distribution([frame["timing"]["gmc_ms"] for frame in frames]),
        "association_ms": distribution([frame["timing"]["association_ms"] for frame in frames]),
        "image_read_ms": distribution(
            [row["image_read_ms"] for item in passes for row in item["image_reads"]]
        ),
        "setup_ms": distribution([item["timing"]["adapter_setup_ms"] for item in passes]),
        "replay_wall_ms": distribution([item["timing"]["replay_ms"] for item in passes]),
    }


def _repeatability(replays):
    statuses = {replay["repeatability"]["status"] for replay in replays}
    return (
        "observed_mismatch"
        if "observed_mismatch" in statuses
        else "not_checked"
        if "not_checked" in statuses
        else "observed_match"
    )


def _aggregate(rows, replays, profile_index, named):
    counts = {key: sum(row["counts"][key] for row in rows) for key in rows[0]["counts"]}
    eligible = all(row["identity"]["available"] for row in rows)
    identity = {"available": eligible, "idtp": None, "idfp": None, "idfn": None, "idf1": None}
    if eligible:
        for key in ("idtp", "idfp", "idfn"):
            identity[key] = sum(row["identity"][key] for row in rows)
        denominator = 2 * identity["idtp"] + identity["idfp"] + identity["idfn"]
        identity["idf1"] = 2 * identity["idtp"] / denominator if denominator else None
    return {
        "profile_index": profile_index,
        "name": named["name"],
        "profile_sha256": profile_hash(named["profile"]),
        "counts": counts,
        "precision": counts["true_positives"] / counts["observations"]
        if counts["observations"]
        else None,
        "recall": counts["true_positives"] / counts["ground_truth"]
        if counts["ground_truth"]
        else None,
        "identity": identity,
        "timing": _timing(replays),
        "repeatability": _repeatability(replays),
    }


def _comparison_status(baseline, candidate, evaluated_frames):
    if "observed_mismatch" in (baseline["repeatability"], candidate["repeatability"]):
        return "unstable"
    if not evaluated_frames:
        return "insufficient"
    keys = (
        "false_positives",
        "false_negatives",
        "identity_switches",
        "fragments",
        "identity_transfers",
        "class_confusions",
    )
    changes = [candidate["counts"][key] - baseline["counts"][key] for key in keys]
    if baseline["identity"]["available"] and candidate["identity"]["available"]:
        changes.extend(
            candidate["identity"][key] - baseline["identity"][key] for key in ("idfp", "idfn")
        )
    better, worse = any(value < 0 for value in changes), any(value > 0 for value in changes)
    return (
        "tradeoff"
        if better and worse
        else "gain"
        if better
        else "regression"
        if worse
        else "no_gain"
    )


def summarize(bundle, runs, checkpoint=None):
    request = bundle["request"]
    named_profiles = [request["baseline"], *request["candidates"]]
    source_results = []
    for source, run in zip(bundle["sources"], runs, strict=True):
        rows, coverage = [], None
        for index, named in enumerate(named_profiles):
            if checkpoint is not None:
                checkpoint()
            comparison = {
                "id": source["comparison"]["id"],
                "report": {
                    "sequence": source["sequence"]["manifest"],
                    "lanes": [
                        {"name": request["baseline"]["name"], "report": run["replays"][0]},
                        {"name": named["name"], "report": run["replays"][index]},
                    ],
                },
            }
            quality = evaluate_quality(
                comparison,
                source["reference"],
                class_mapping=request["class_mapping"],
                iou_threshold=request["iou_threshold"],
            )
            if checkpoint is not None:
                checkpoint()
            coverage = quality["coverage"]
            rows.append(
                {key: value for key, value in quality["lanes"][1].items() if key != "frames"}
            )
        source_results.append(
            {
                "sequence_id": source["sequence"]["id"],
                "split": source["entry"]["split"],
                "reference_id": source["reference"]["id"],
                "coverage": coverage,
                "profiles": rows,
            }
        )
    splits = {}
    for split in ("train", "val"):
        indices = [
            index
            for index, source in enumerate(bundle["sources"])
            if source["entry"]["split"] == split
        ]
        if not indices:
            splits[split] = None
            continue
        splits[split] = {
            "sequence_count": len(indices),
            "evaluated_frames": sum(
                source_results[index]["coverage"]["evaluated_frames"] for index in indices
            ),
            "available_frames": sum(
                source_results[index]["coverage"]["available_frames"] for index in indices
            ),
            "profiles": [
                _aggregate(
                    [source_results[index]["profiles"][profile_index] for index in indices],
                    [runs[index]["replays"][profile_index] for index in indices],
                    profile_index,
                    named,
                )
                for profile_index, named in enumerate(named_profiles)
            ],
        }
    comparisons = []
    for index, named in enumerate(named_profiles):
        statuses, deltas = {}, {}
        for split, aggregate in splits.items():
            if aggregate is None:
                statuses[split], deltas[split] = None, None
                continue
            baseline, candidate = aggregate["profiles"][0], aggregate["profiles"][index]
            statuses[split] = (
                "baseline"
                if index == 0
                else _comparison_status(
                    baseline,
                    candidate,
                    aggregate["evaluated_frames"]
                    if all(
                        source["coverage"]["evaluated_frames"] > 0
                        for source in source_results
                        if source["split"] == split
                    )
                    else 0,
                )
            )
            deltas[split] = {
                "tracker_median_ms": candidate["timing"]["tracker_ms"]["median"]
                - baseline["timing"]["tracker_ms"]["median"],
                "tracker_p95_ms": candidate["timing"]["tracker_ms"]["p95"]
                - baseline["timing"]["tracker_ms"]["p95"],
            }
        comparisons.append(
            {
                "profile_index": index,
                "name": named["name"],
                "role": "baseline" if index == 0 else "candidate",
                "by_split": statuses,
                "cost_delta_by_split": deltas,
            }
        )
    has_validation = splits["val"] is not None
    any_gain = any(
        item["by_split"]["val"] == "gain"
        and item["by_split"]["train"] not in ("regression", "tradeoff", "unstable", "insufficient")
        for item in comparisons[1:]
    )
    return {
        "source_results": source_results,
        "splits": splits,
        "comparisons": comparisons,
        "decision": {
            "applied": False,
            "retained_profile_sha256": profile_hash(request["baseline"]["profile"]),
            "reason": "development_only"
            if not has_validation
            else "manual_review_required"
            if any_gain
            else "no_verified_gain",
        },
        "limitations": deepcopy(LIMITATIONS),
    }


def validate_report(bundle, report, checkpoint=None):
    from iris.tracking_study_replay import validate_replay

    _object(
        report,
        {"schema", "complete", "request", "fingerprint", "dataset", "sources", "runs", "summary"},
        "Tracking study report",
    )
    if len(json.dumps(report, allow_nan=False).encode("utf-8")) > MAX_REPORT_BYTES:
        raise ValueError("Tracking study report exceeds its portable size budget")
    request = canonicalize_request(bundle["request"])
    if (
        report["schema"] != REPORT_SCHEMA
        or report["complete"] is not True
        or _digest(report["request"]) != _digest(request)
        or report["fingerprint"] != bundle["fingerprint"]
    ):
        raise ValueError("Tracking study report does not match its frozen request")
    expected_dataset = {
        "id": bundle["dataset"]["id"],
        "manifest_sha256": bundle["dataset"]["manifest_sha256"],
        "reserved_test_entries": [
            deepcopy(entry)
            for entry in bundle["dataset"]["manifest"]["entries"]
            if entry["split"] == "test"
        ],
    }
    if _digest(report["dataset"]) != _digest(expected_dataset) or _digest(
        report["sources"]
    ) != _digest(source_bindings(bundle)):
        raise ValueError("Tracking study source or reserved-test bindings changed")
    runs, profiles = report["runs"], [request["baseline"], *request["candidates"]]
    if not isinstance(runs, list) or len(runs) != len(bundle["sources"]):
        raise ValueError("Tracking study must include every requested source")
    runtime_by_algorithm, common_execution = {}, None
    for source, run in zip(bundle["sources"], runs, strict=True):
        _object(run, {"sequence_id", "split", "replays"}, "Tracking study source run")
        if (
            run["sequence_id"] != source["sequence"]["id"]
            or run["split"] != source["entry"]["split"]
            or run["split"] not in ("train", "val")
            or not isinstance(run["replays"], list)
            or len(run["replays"]) != len(profiles)
        ):
            raise ValueError("Tracking study run changed its source, role or profile count")
        for named, replay in zip(profiles, run["replays"], strict=True):
            if checkpoint is not None:
                checkpoint()
            validate_replay(replay, source, named["profile"], request["repeats"])
            # Runtime policy contains profile-controlled threads/seed, which may
            # differ between profiles. Identical profiles must retain metadata.
            key = profile_hash(named["profile"])
            runtime = replay["passes"][0]["runtime_sha256"]
            if key in runtime_by_algorithm and runtime_by_algorithm[key] != runtime:
                raise ValueError("Identical profiles changed execution runtime across sources")
            runtime_by_algorithm[key] = runtime
            metadata = replay["passes"][0]["metadata"]
            execution = {key: metadata[key] for key in ("python", "platform", "packages")}
            execution["adapter_sha256"] = metadata["provenance"]["adapter_sha256"]
            execution["manifest_sha256"] = metadata["provenance"]["manifest_sha256"]
            execution["policy"] = {
                key: value
                for key, value in metadata["execution_policy"].items()
                if key != "unconfirmed_returned"
            }
            if common_execution is not None and _digest(execution) != _digest(common_execution):
                raise ValueError("Study profiles changed their shared host or execution runtime")
            common_execution = execution
    if _digest(report["summary"]) != _digest(summarize(bundle, runs, checkpoint=checkpoint)):
        raise ValueError(
            "Tracking study quality, timing or conclusions do not match saved evidence"
        )
    return report


def study_status():
    return {
        "schema": REPORT_SCHEMA,
        "limits": {
            "max_candidates": MAX_CANDIDATES,
            "max_sources": MAX_SOURCES,
            "max_frames": MAX_FRAMES,
            "max_repeats": 3,
            "max_updates": MAX_UPDATES,
            "max_seconds": MAX_SECONDS,
            "max_report_bytes": MAX_REPORT_BYTES,
        },
        "limitations": deepcopy(LIMITATIONS),
        "budget_policy": "Complete-only; cooperative deadline checked between work units, "
        "not a hard native-call timeout.",
        "cost_scope": "Cached-input tracker only; native adapter time, image reads, "
        "setup and replay wall measured separately.",
    }
