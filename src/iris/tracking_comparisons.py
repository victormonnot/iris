"""Complete-only visual comparisons of two native trackers on one frozen cache.

The existing durable job owns the request and complete report. There is no partial
tracker checkpoint: a new launch always creates fresh native tracker instances.
"""

import json
from copy import deepcopy

from iris.store import DEFAULT_PROJECT_ID, _decode, new_id, now
from iris.temporal import _digest, _insert, _row, _text
from iris.temporal_detections import _coverage, _finite, _state
from iris.tracking_contracts import (
    make_profile,
    profile_hash,
    semantic_frame,
    validate_tracking_frame,
)
from iris.tracking_replay import REPORT_SCHEMA as REPLAY_SCHEMA
from iris.tracking_replay import TrackingReplayCancelled, replay_detection_cache

KIND = "tracking_compare"
REPORT_SCHEMA = "iris-tracking-comparison-v1"
MAX_COMPARISON_FRAMES = 500
# Keep a report below the workspace archive JSON column limit, including Unicode
# escaping performed by Store._encode, so every successful run stays portable.
MAX_REPORT_BYTES = 48 * 1024**2
LIMITATIONS = (
    "Visual comparison only: no temporal reference evaluation, quality score or ranking. "
    "Track IDs are local to each lane and sequence; equal numbers do not identify the same "
    "object across lanes. Each tracker replays the same saved detections once; repeatability "
    "is not checked. Memory counts available frame updates, not seconds. Missing source "
    "frames receive no synthetic updates. Lost-track predictions are not observations or "
    "human references. Cached detector timings are separate from current tracker timings."
)


def _profiles(cache, class_ids, gmc_method):
    known = {entry["id"] for entry in cache["config"]["detector"]["classes"]}
    if (
        not isinstance(class_ids, list)
        or not 1 <= len(class_ids) <= 100
        or any(type(value) is not int or value not in known for value in class_ids)
        or len(set(class_ids)) != len(class_ids)
    ):
        raise ValueError("Choose distinct native classes from this detector cache")
    if gmc_method not in ("none", "sparseOptFlow"):
        raise ValueError("BoT-SORT camera compensation must be none or sparseOptFlow")
    classes = sorted(class_ids)
    profiles = [
        make_profile("bytetrack", class_ids=classes),
        make_profile("botsort", class_ids=classes, gmc_method=gmc_method),
    ]
    if cache["config"]["detector"]["min_score"] > min(item["low_threshold"] for item in profiles):
        raise ValueError(
            "The saved detector score floor is above 0.1; "
            "explicitly calculate a lower-floor detector cache first"
        )
    return profiles


def _complete_inputs(conn, cache_id):
    cache, sequence, frames, attempts = _state(conn, cache_id)
    coverage = _coverage(cache, frames)
    if coverage["remaining_count"]:
        raise ValueError("Tracking comparison requires a complete detector cache")
    if len(frames) > MAX_COMPARISON_FRAMES:
        raise ValueError(
            f"The Studio comparator accepts at most {MAX_COMPARISON_FRAMES} available frames; "
            "prepare a shorter frozen sequence explicitly"
        )
    return cache, sequence, frames, attempts, coverage


def _request(job, cache, sequence, coverage):
    params = job["params"]
    if (
        job["kind"] != KIND
        or not isinstance(params, dict)
        or set(params)
        != {"name", "cache_id", "sequence_id", "cache_fingerprint", "result_sha256", "profiles"}
        or params["cache_id"] != cache["id"]
        or params["sequence_id"] != sequence["id"]
        or params["cache_fingerprint"] != cache["fingerprint"]
        or params["result_sha256"] != coverage["result_sha256"]
    ):
        raise ValueError("Tracking comparison no longer matches its frozen cache")
    if _text(params["name"], "Tracking comparison name") != params["name"]:
        raise ValueError("Tracking comparison name is not canonical")
    profiles = params["profiles"]
    if (
        not isinstance(profiles, list)
        or len(profiles) != 2
        or not all(isinstance(profile, dict) for profile in profiles)
        or profiles != _profiles(cache, profiles[0]["class_ids"], profiles[1]["gmc_method"])
    ):
        raise ValueError("Tracking comparison requires the two frozen native profiles")
    return params


def _validate_report(report, params, cache, sequence, frames, attempts):
    """Validate archived evidence against frozen cache rows without importing trackers."""
    if (
        not isinstance(report, dict)
        or set(report) != {"schema", "sequence", "cache_id", "lanes", "limitations"}
        or report["schema"] != REPORT_SCHEMA
        or report["sequence"] != sequence["manifest"]
        or report["cache_id"] != cache["id"]
        or report["limitations"] != LIMITATIONS
        or not isinstance(report["lanes"], list)
        or len(report["lanes"]) != 2
    ):
        raise ValueError("Tracking comparison report is incomplete or inconsistent")
    executions = {
        attempt["result"]["execution"]["signature_sha256"]: attempt["result"]["execution"][
            "signature"
        ]
        for attempt in attempts
        if attempt["result"]["execution"] is not None
    }
    payloads = [
        {
            "frame_id": frame["frame_id"],
            "payload_sha256": frame["payload_sha256"],
            "producer_job_id": frame["job_id"],
            "execution_signature_sha256": frame["payload"]["execution_signature_sha256"],
        }
        for frame in frames
    ]
    known = {entry["id"] for entry in cache["config"]["detector"]["classes"]}
    for lane, name, profile in zip(
        report["lanes"], ("ByteTrack", "BoT-SORT"), params["profiles"], strict=True
    ):
        if not isinstance(lane, dict) or set(lane) != {"name", "report"} or lane["name"] != name:
            raise ValueError("Tracking comparison lane identity is invalid")
        replay = lane["report"]
        if (
            not isinstance(replay, dict)
            or set(replay)
            != {
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
            or replay["schema"] != REPLAY_SCHEMA
            or replay["complete"] is not True
            or replay["sequence"] != sequence["manifest"]
            or replay["cache"]["id"] != cache["id"]
            or replay["cache"]["fingerprint"] != cache["fingerprint"]
            or replay["cache"]["result_sha256"] != params["result_sha256"]
            or replay["cache"]["config"] != cache["config"]
            or replay["profile"] != profile
            or replay["profile_sha256"] != profile_hash(profile)
            or replay["input_filter"]
            != {
                "min_score": cache["config"]["detector"]["min_score"],
                "class_ids": profile["class_ids"],
            }
            or replay["excluded_class_ids"] != sorted(known - set(profile["class_ids"]))
            or replay["source_images_required"] is not (profile["gmc_method"] != "none")
            or replay["source_payloads"] != payloads
            or replay["detector_executions"] != executions
            or not isinstance(replay["passes"], list)
            or len(replay["passes"]) != 1
        ):
            raise ValueError("Tracking lane is not a complete replay of the frozen inputs")
        replay_pass = replay["passes"][0]
        if (
            not isinstance(replay_pass, dict)
            or set(replay_pass)
            != {
                "pass_index",
                "semantic_sha256",
                "frames",
                "metadata",
                "runtime_sha256",
                "image_reads",
                "timing",
            }
            or type(replay_pass["pass_index"]) is not int
            or replay_pass["pass_index"] != 0
            or not isinstance(replay_pass["frames"], list)
            or len(replay_pass["frames"]) != len(frames)
            or not isinstance(replay_pass["metadata"], dict)
            or replay_pass["metadata"].get("algorithm") != profile["algorithm"]
            or _digest(replay_pass["metadata"]) != replay_pass["runtime_sha256"]
            or not isinstance(replay_pass["image_reads"], list)
            or len(replay_pass["image_reads"]) != len(frames)
            or not isinstance(replay_pass["timing"], dict)
            or set(replay_pass["timing"])
            != {
                "adapter_setup_ms",
                "replay_ms",
                "source_image_read_ms",
            }
            or not all(_finite(value) for value in replay_pass["timing"].values())
        ):
            raise ValueError("Tracking replay pass or runtime evidence is invalid")
        for read, source in zip(replay_pass["image_reads"], frames, strict=True):
            if (
                not isinstance(read, dict)
                or set(read) != {"frame_id", "image_read_ms"}
                or read["frame_id"] != source["frame_id"]
                or not _finite(read["image_read_ms"])
                or profile["gmc_method"] == "none"
                and read["image_read_ms"] != 0
            ):
                raise ValueError("Tracking source image timing evidence is invalid")
        if replay_pass["timing"]["source_image_read_ms"] != sum(
            read["image_read_ms"] for read in replay_pass["image_reads"]
        ):
            raise ValueError("Tracking source image timing total is inconsistent")
        last_observed = {}
        for position, (result, source) in enumerate(
            zip(replay_pass["frames"], frames, strict=True)
        ):
            filtered = {
                **source["payload"],
                "detections": [
                    row
                    for row in source["payload"]["detections"]
                    if row["label_id"] in profile["class_ids"]
                ],
            }
            validate_tracking_frame(result, filtered, profile)
            if result["sequence_id"] != sequence["id"] or result["update_index"] != position + 1:
                raise ValueError("Tracking lane changed the sequence or source frame order")
            for observation in result["observations"]:
                previous = last_observed.get(observation["track_id"])
                if previous is not None and any(
                    previous[key] != observation[key] for key in ("label_id", "label")
                ):
                    raise ValueError("A tracking identity changed native class within its lane")
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
                        "A predicted track does not refer to its last observed identity"
                    )
        digest = _digest(
            {
                "profile_sha256": replay["profile_sha256"],
                "cache_fingerprint": cache["fingerprint"],
                "sequence_sha256": cache["config"]["sequence_sha256"],
                "frames": [semantic_frame(result) for result in replay_pass["frames"]],
            }
        )
        repeatability = replay["repeatability"]
        if (
            replay_pass["semantic_sha256"] != digest
            or repeatability["status"] != "not_checked"
            or type(repeatability["passes"]) is not int
            or repeatability["passes"] != 1
            or repeatability["semantic_sha256"] != [digest]
        ):
            raise ValueError("Tracking replay semantic evidence or repeatability claim changed")
    if len(json.dumps(report, allow_nan=False).encode("utf-8")) > MAX_REPORT_BYTES:
        raise ValueError("Tracking report is too large; use a shorter frozen sequence")


def _checked_job(conn, job_id):
    job = _row(conn, "jobs", job_id)
    if job["kind"] != KIND:
        raise KeyError(job_id)
    cache, sequence, frames, attempts, coverage = _complete_inputs(conn, job["params"]["cache_id"])
    params = _request(job, cache, sequence, coverage)
    if job["status"] == "succeeded":
        if job["cancel_requested"]:
            raise ValueError("A cancelled tracking attempt cannot publish a report")
        _validate_report(job["result"], params, cache, sequence, frames, attempts)
    elif job["result"] is not None:
        raise ValueError("An unfinished tracking comparison cannot publish a report")
    return job, cache, sequence


def public_job(job):
    """Keep complete reports out of generic job polling and history responses."""
    if job["kind"] != KIND or job["result"] is None:
        return job
    report = job["result"]
    return {
        **job,
        "result": {
            "schema": REPORT_SCHEMA,
            "cache_id": job["params"]["cache_id"],
            "sequence_id": job["params"]["sequence_id"],
            "lane_count": len(report["lanes"]),
            "frame_count": len(report["sequence"]["frames"]),
        },
    }


def _detail(job, *, include_report):
    return {
        "id": job["id"],
        "name": job["params"]["name"],
        "sequence_id": job["params"]["sequence_id"],
        "cache_id": job["params"]["cache_id"],
        "job": public_job(job),
        "report": job["result"] if include_report and job["status"] == "succeeded" else None,
    }


def create_tracking_comparison(
    store, jobs, cache_id, *, name, class_ids, gmc_method, project_id=DEFAULT_PROJECT_ID
):
    from iris.tracking import tracking_status

    name = _text(name, "Tracking comparison name")
    with jobs.guard, store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        cache, sequence, _, _, coverage = _complete_inputs(conn, cache_id)
        if sequence["project_id"] != project_id:
            raise KeyError(cache_id)
        profiles = _profiles(cache, class_ids, gmc_method)
        status = tracking_status()
        if not status["available"]:
            raise ValueError(status["installation"])
        identifier = new_id()
        _insert(
            conn,
            "jobs",
            {
                "id": identifier,
                "kind": KIND,
                "status": "queued",
                "params": {
                    "name": name,
                    "cache_id": cache_id,
                    "sequence_id": sequence["id"],
                    "cache_fingerprint": cache["fingerprint"],
                    "result_sha256": coverage["result_sha256"],
                    "profiles": profiles,
                },
                "result": None,
                "created_at": now(),
                "message": "Waiting to replay ByteTrack and BoT-SORT from the saved detector cache",
            },
        )
    return get_tracking_comparison(store, identifier)


def get_tracking_comparison(store, job_id, *, include_report=True):
    with store.connect() as conn:
        conn.execute("BEGIN")
        job, _, _ = _checked_job(conn, job_id)
        return _detail(job, include_report=include_report)


def list_tracking_comparisons(store, cache_id):
    with store.connect() as conn:
        conn.execute("BEGIN")
        _row(conn, "temporal_detection_caches", cache_id)
        return [
            _detail(_decode(row), include_report=False)
            for row in conn.execute(
                "SELECT * FROM jobs WHERE kind=? AND json_extract(params,'$.cache_id')=? "
                "ORDER BY created_at,id",
                (KIND, cache_id),
            )
        ]


def run_tracking_comparison(store, job_id, progress, cancelled):
    with store.connect() as conn:
        conn.execute("BEGIN")
        job, _, sequence = _checked_job(conn, job_id)
    if job["status"] != "running" or cancelled():
        raise TrackingReplayCancelled("Tracking comparison stopped before replay")
    lanes = []
    for index, (name, profile) in enumerate(
        zip(("ByteTrack", "BoT-SORT"), job["params"]["profiles"], strict=True)
    ):
        progress(index / 2, f"Starting {name} from the same saved detector cache")
        report = replay_detection_cache(
            store,
            job["params"]["cache_id"],
            profile=deepcopy(profile),
            repeats=1,
            cancelled=cancelled,
            progress=lambda value, message, i=index, label=name: progress(
                (i + value) / 2, f"{label}: {message}"
            ),
        )
        lanes.append({"name": name, "report": report})
    if cancelled():
        raise TrackingReplayCancelled("Tracking comparison cancelled; no complete report published")
    result = {
        "schema": REPORT_SCHEMA,
        "sequence": sequence["manifest"],
        "cache_id": job["params"]["cache_id"],
        "lanes": lanes,
        "limitations": LIMITATIONS,
    }
    with store.connect() as conn:
        conn.execute("BEGIN")
        cache, sequence, frames, attempts, coverage = _complete_inputs(conn, result["cache_id"])
        params = _request(job, cache, sequence, coverage)
        _validate_report(result, params, cache, sequence, frames, attempts)
    if cancelled():
        raise TrackingReplayCancelled("Tracking comparison stopped before report publication")
    return result


def validate_tracking_comparison_records(connection):
    try:
        for row in connection.execute("SELECT id FROM jobs WHERE kind=?", (KIND,)):
            _checked_job(connection, row["id"])
    except (KeyError, TypeError, IndexError, OverflowError) as exc:
        raise ValueError("Tracking comparison evidence is invalid or missing") from exc
