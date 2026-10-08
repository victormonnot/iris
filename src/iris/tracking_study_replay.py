"""Validate saved custom-profile replays against frozen detector payloads.

This is separate from T4's fixed two-default-profile protocol. Validation does
not import trackers or require the software/hardware that produced a report.
"""

import re

from iris.temporal import _digest
from iris.tracking_contracts import profile_hash, semantic_frame, validate_tracking_frame
from iris.tracking_study_contracts import _number, _object, _text

REPLAY_FIELDS = {
    "schema",
    "complete",
    "cache",
    "sequence",
    "profile",
    "profile_sha256",
    "input_filter",
    "source_payloads",
    "detector_executions",
    "excluded_class_ids",
    "source_images_required",
    "passes",
    "repeatability",
    "timing_scope",
    "limitations",
}

EXECUTION_POLICY = {
    "device": "cpu",
    "learned_reid": False,
    "class_association": "independent_native_instance_per_class",
    "native_frame_rate_argument": 30,
    "native_buffer_unit": "available_frame_updates",
    "native_time_step": 1,
    "skipped_source_frames": "no_synthetic_updates",
    "native_high_and_low_comparison": "strict_greater_than",
    "native_second_pass_match_threshold": 0.5,
    "native_unconfirmed_match_threshold": 0.7,
    "native_duplicate_iou_distance": 0.15,
    "native_expiry_order": "association_before_lost_cleanup_not_a_hard_ttl",
    "native_iou_geometry": "cython_bbox_inclusive_pixel_overlap",
    "gmc_scope": "one_sequence_estimate_shared_across_classes",
    "gmc_failure": "raise_and_require_reset_no_silent_fallback",
    "gmc_insufficient_matches": "native_identity_transform_reported",
    "opencv_rng": "seed_plus_update_index_before_each_update_no_rng_restore_api",
    "repeatability": "measure_semantic_hashes_on_repeated_fresh_replays",
}


def _hash(value, name, length=64):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{" + str(length) + "}", value) is None:
        raise ValueError(f"{name} must be a lowercase hexadecimal digest")


def _provenance(value, algorithm):
    _object(
        value,
        {
            "repository",
            "commit",
            "license",
            "files",
            "license_sha256",
            "adaptations_sha256",
            "manifest_sha256",
            "adapter_sha256",
        },
        "Tracker source provenance",
    )
    for key in ("repository", "license"):
        _text(value[key], f"Tracker provenance {key}", 1000)
    _hash(value["commit"], "Tracker source commit", 40)
    for key in ("license_sha256", "adaptations_sha256", "manifest_sha256"):
        _hash(value[key], key)
    _object(
        value["adapter_sha256"],
        {"tracking.py", "tracking_contracts.py"},
        "Tracker adapter source hashes",
    )
    for digest in value["adapter_sha256"].values():
        _hash(digest, "Adapter source hash")
    names = {"__init__.py", "basetrack.py", "kalman_filter.py", "matching.py"}
    names |= {"byte_tracker.py"} if algorithm == "bytetrack" else {"bot_sort.py", "gmc.py"}
    _object(value["files"], names, "Vendored tracker source files")
    for name, item in value["files"].items():
        _object(item, {"path", "upstream_sha256", "vendored_sha256"}, "Vendored source provenance")
        _hash(item["vendored_sha256"], "Vendored source hash")
        if name == "__init__.py":
            if item["path"] is not None or item["upstream_sha256"] is not None:
                raise ValueError("Local tracker package initializer has no upstream source")
        else:
            _text(item["path"], "Upstream source path", 1000)
            _hash(item["upstream_sha256"], "Upstream source hash")


def _metadata(metadata, profile):
    _object(
        metadata,
        {"schema", "algorithm", "provenance", "python", "platform", "packages", "execution_policy"},
        "Study tracker runtime metadata",
    )
    if (
        not isinstance(metadata, dict)
        or metadata.get("schema") != "iris-tracker-runtime-v1"
        or metadata.get("algorithm") != profile["algorithm"]
    ):
        raise ValueError("Study tracker metadata must declare its runtime schema and algorithm")
    for key in ("python", "platform"):
        _text(metadata[key], f"Tracker runtime {key}", 2000)
    _object(
        metadata["packages"],
        {"scipy", "lap", "cython_bbox", "numpy", "opencv-python-headless"},
        "Tracker runtime packages",
    )
    for version in metadata["packages"].values():
        _text(version, "Historical package version", 100)
    _provenance(metadata["provenance"], profile["algorithm"])
    policy = metadata.get("execution_policy")
    expected_policy = {
        **EXECUTION_POLICY,
        "opencv_threads": profile["opencv_threads"],
        "seed": profile["seed"],
        "unconfirmed_returned": profile["algorithm"] == "botsort",
    }
    _object(policy, set(expected_policy) | {"blas_environment"}, "Tracker execution policy")
    for key, expected in expected_policy.items():
        if type(policy.get(key)) is not type(expected) or policy[key] != expected:
            raise ValueError(f"Study tracker execution policy changed: {key}")
    environment = policy["blas_environment"]
    _object(
        environment,
        {"OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"},
        "Tracker BLAS environment",
    )
    for value in environment.values():
        if value is not None and (not isinstance(value, str) or len(value) > 2000):
            raise ValueError("Historical BLAS environment must be text or null")


def validate_replay(replay, source, profile, repeats):
    _object(replay, REPLAY_FIELDS, "Study tracker replay")
    cache, sequence, frames = source["cache"], source["sequence"], source["frames"]
    template = source["comparison"]["report"]["lanes"][0]["report"]
    known = {entry["id"] for entry in cache["config"]["detector"]["classes"]}
    payloads = [
        {
            "frame_id": frame["frame_id"],
            "payload_sha256": frame["payload_sha256"],
            "producer_job_id": frame["job_id"],
            "execution_signature_sha256": frame["payload"]["execution_signature_sha256"],
        }
        for frame in frames
    ]
    executions = {
        attempt["result"]["execution"]["signature_sha256"]: attempt["result"]["execution"][
            "signature"
        ]
        for attempt in source["attempts"]
        if attempt["result"]["execution"] is not None
    }
    expected = {
        "schema": "iris-tracking-replay-v1",
        "complete": True,
        "cache": template["cache"],
        "sequence": sequence["manifest"],
        "profile": profile,
        "profile_sha256": profile_hash(profile),
        "input_filter": {
            "min_score": cache["config"]["detector"]["min_score"],
            "class_ids": profile["class_ids"],
        },
        "source_payloads": payloads,
        "detector_executions": executions,
        "excluded_class_ids": sorted(known - set(profile["class_ids"])),
        "source_images_required": profile["gmc_method"] != "none",
        "timing_scope": template["timing_scope"],
        "limitations": template["limitations"],
    }
    if (
        replay["complete"] is not True
        or type(replay["source_images_required"]) is not bool
        or _digest({key: replay[key] for key in expected}) != _digest(expected)
    ):
        raise ValueError("Study replay changed its frozen profile, input or source provenance")
    if cache["config"]["detector"]["min_score"] > profile["low_threshold"]:
        raise ValueError("Study replay profile requests detections below the saved cache floor")
    passes = replay["passes"]
    if not isinstance(passes, list) or len(passes) != repeats:
        raise ValueError("Study replay repetition count changed")
    runtime_hash, semantic_hashes = None, []
    for pass_index, replay_pass in enumerate(passes):
        _object(
            replay_pass,
            {
                "pass_index",
                "semantic_sha256",
                "frames",
                "metadata",
                "runtime_sha256",
                "image_reads",
                "timing",
            },
            "Study replay pass",
        )
        if type(replay_pass["pass_index"]) is not int or replay_pass["pass_index"] != pass_index:
            raise ValueError("Study replay pass order changed")
        _metadata(replay_pass["metadata"], profile)
        current_runtime = _digest(replay_pass["metadata"])
        if (
            replay_pass["runtime_sha256"] != current_runtime
            or runtime_hash is not None
            and current_runtime != runtime_hash
        ):
            raise ValueError("Study replay runtime metadata changed between repetitions")
        runtime_hash = current_runtime
        results, reads = replay_pass["frames"], replay_pass["image_reads"]
        if (
            not isinstance(results, list)
            or len(results) != len(frames)
            or not isinstance(reads, list)
            or len(reads) != len(frames)
        ):
            raise ValueError("Study replay must preserve every available source frame")
        _object(
            replay_pass["timing"],
            {"adapter_setup_ms", "replay_ms", "source_image_read_ms"},
            "Study replay timing",
        )
        for value in replay_pass["timing"].values():
            _number(value, "Replay duration", 0, 1e12)
        last_observed = {}
        for position, (result, frame, read) in enumerate(zip(results, frames, reads, strict=True)):
            _object(read, {"frame_id", "image_read_ms"}, "Study image-read timing")
            _number(read["image_read_ms"], "Source-image duration", 0, 1e12)
            if (
                read["frame_id"] != frame["frame_id"]
                or profile["gmc_method"] == "none"
                and read["image_read_ms"] != 0
            ):
                raise ValueError("Study source-image timing does not match its profile or frame")
            filtered = {
                **frame["payload"],
                "detections": [
                    row
                    for row in frame["payload"]["detections"]
                    if row["label_id"] in profile["class_ids"]
                ],
            }
            validate_tracking_frame(result, filtered, profile)
            timing = result["timing"]
            if timing["gmc_ms"] + timing["association_ms"] > timing["total_ms"] + 1e-6:
                raise ValueError("Nested tracker stage times exceed the measured adapter total")
            if result["sequence_id"] != sequence["id"] or result["update_index"] != position + 1:
                raise ValueError("Study replay changed sequence reset or frame-update order")
            for observation in result["observations"]:
                previous = last_observed.get(observation["track_id"])
                if previous is not None and any(
                    previous[key] != observation[key] for key in ("label_id", "label")
                ):
                    raise ValueError("Study tracking identity changed native class")
                last_observed[observation["track_id"]] = {
                    "last_observed_frame_id": result["frame_id"],
                    "last_observed_frame_index": result["frame_index"],
                    "last_observed_timestamp_seconds": result["timestamp_seconds"],
                    "last_observed_update_index": result["update_index"],
                    **{key: observation[key] for key in ("label_id", "label", "confirmed")},
                }
            for prediction in result["predictions"]:
                previous = last_observed.get(prediction["track_id"])
                if previous is None or any(
                    prediction[key] != value for key, value in previous.items()
                ):
                    raise ValueError(
                        "Study prediction does not point to its last measured identity"
                    )
        if replay_pass["timing"]["source_image_read_ms"] != sum(
            read["image_read_ms"] for read in reads
        ):
            raise ValueError("Study source-image timing total is inconsistent")
        if (
            replay_pass["timing"]["source_image_read_ms"]
            > replay_pass["timing"]["replay_ms"] + 1e-6
        ):
            raise ValueError("Source-image reads exceed the measured replay loop duration")
        digest = _digest(
            {
                "profile_sha256": replay["profile_sha256"],
                "cache_fingerprint": cache["fingerprint"],
                "sequence_sha256": cache["config"]["sequence_sha256"],
                "frames": [semantic_frame(result) for result in results],
            }
        )
        if replay_pass["semantic_sha256"] != digest:
            raise ValueError("Study replay semantic checksum changed")
        semantic_hashes.append(digest)
    expected_repeatability = {
        "status": "not_checked"
        if repeats == 1
        else "observed_match"
        if len(set(semantic_hashes)) == 1
        else "observed_mismatch",
        "passes": repeats,
        "semantic_sha256": semantic_hashes,
        "scope": "Observed outputs for these saved inputs and this runtime; excludes timings",
    }
    if _digest(replay["repeatability"]) != _digest(expected_repeatability):
        raise ValueError("Study replay repeatability claim changed")
    return replay
