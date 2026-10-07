"""Frozen detector caches with explicit, append-only continuation attempts.

Committed frames are authoritative, including empty results. Missing frames are
never negatives. Reading a cache needs neither its weights nor a model runtime.
"""

import math
from copy import deepcopy

from iris.projects import record_project
from iris.store import DEFAULT_PROJECT_ID, _decode, new_id, now
from iris.temporal import _digest, _insert, _row, _sequence_record, _text, _verify_sequence_media
from iris.temporal_detection_contracts import (
    cache_fingerprint,
    frame_validation_context,
    payload_hash,
    validate_cache_config,
    validate_frame_payload,
)
from iris.temporal_detector import (
    prepare_detector,
    saved_execution_signature,
    validate_execution_signature,
)
from iris.tiling import tile_boxes

KIND = "temporal_detect"
RESULT_SCHEMA = "iris-temporal-detection-attempt-v1"
TERMINAL = {"failed", "cancelled", "interrupted"}
LIMITATIONS = (
    "Saved results are detector outputs after native filtering, NMS and detection caps. "
    "A stricter score/class filter can reuse them; lowering the saved score floor or "
    "changing detector, preprocessing, NMS, tiling or caps requires another calculation. "
    "Stored timings describe the producing attempt, not cache-read performance."
)


class DetectionCacheConflict(RuntimeError):
    """The attempt or frozen evidence changed before publication or continuation."""


def _owned_sequence(conn, sequence_id, project_id):
    sequence = _sequence_record(conn, _row(conn, "temporal_sequences", sequence_id))
    if sequence["project_id"] != project_id:
        raise KeyError(sequence_id)
    return sequence


def _cache_record(conn, row):
    sequence = _sequence_record(conn, _row(conn, "temporal_sequences", row["sequence_id"]))
    config = validate_cache_config(row["config"], sequence["manifest"])
    if (
        config["sequence_id"] != row["sequence_id"]
        or cache_fingerprint(config) != row["fingerprint"]
        or _digest(row["config"]) != row["fingerprint"]
    ):
        raise ValueError("Detection cache identity or frozen configuration changed")
    _text(row["name"], "Detection cache name")
    initial = _row(conn, "jobs", row["job_id"])
    if (
        initial["kind"] != KIND
        or not isinstance(initial["params"], dict)
        or initial["params"].get("cache_id") != row["id"]
    ):
        raise ValueError("Detection cache initial job is missing or belongs to another cache")
    return {**row, "config": config}, sequence


def _finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value) and value >= 0
    except OverflowError:
        return False


def _execution(cache, value):
    if not isinstance(value, dict) or set(value) != {
        "signature",
        "signature_sha256",
        "metadata",
        "metadata_sha256",
        "load_ms",
        "warmup_ms",
        "warmup_frame_id",
    }:
        raise ValueError("Detection execution receipt has an unsupported format")
    checked = validate_execution_signature(cache["config"]["detector"], value["signature"])
    if (
        _digest(checked) != value["signature_sha256"]
        or _digest(value["signature"]) != value["signature_sha256"]
        or not isinstance(value["metadata"], dict)
        or _digest(value["metadata"]) != value["metadata_sha256"]
        or saved_execution_signature(cache["config"]["detector"], value["metadata"]) != checked
        or not _finite(value["load_ms"])
        or not _finite(value["warmup_ms"])
        or value["warmup_frame_id"] not in cache["config"]["frame_ids"]
    ):
        raise ValueError("Detection execution signature, timings or metadata changed")
    return value


def _initial_result(cache_id, inherited_count=0):
    return {
        "schema": RESULT_SCHEMA,
        "cache_id": cache_id,
        "worker_token": None,
        "execution": None,
        "inherited_count": inherited_count,
        "produced_count": 0,
        "completed_count": inherited_count,
        "cancelled": False,
    }


def _job_record(cache, job):
    params, result = job["params"], job["result"]
    if (
        job["kind"] != KIND
        or not isinstance(params, dict)
        or set(params) != {"cache_id", "cache_fingerprint", "recovery_of", "inherited_hashes"}
        or params["cache_id"] != cache["id"]
        or params["cache_fingerprint"] != cache["fingerprint"]
        or not isinstance(params["inherited_hashes"], list)
        or not isinstance(result, dict)
        or set(result)
        != {
            "schema",
            "cache_id",
            "worker_token",
            "execution",
            "inherited_count",
            "produced_count",
            "completed_count",
            "cancelled",
        }
        or result["schema"] != RESULT_SCHEMA
        or result["cache_id"] != cache["id"]
        or type(result["cancelled"]) is not bool
    ):
        raise ValueError("Detection attempt does not match its frozen cache")
    for key in ("inherited_count", "produced_count", "completed_count"):
        if type(result[key]) is not int or not 0 <= result[key] <= len(
            cache["config"]["frame_ids"]
        ):
            raise ValueError("Detection attempt counts are invalid")
    if (
        result["inherited_count"] != len(params["inherited_hashes"])
        or result["completed_count"] != result["inherited_count"] + result["produced_count"]
        or (
            result["worker_token"] is not None
            and (not isinstance(result["worker_token"], str) or len(result["worker_token"]) != 32)
        )
    ):
        raise ValueError("Detection attempt checkpoint is inconsistent")
    if result["execution"] is not None:
        if result["worker_token"] is None:
            raise ValueError("Detection execution has no worker claim")
        _execution(cache, result["execution"])
    if result["produced_count"] and result["execution"] is None:
        raise ValueError("Published detections have no execution receipt")
    if job["id"] == cache["job_id"] and (
        params["recovery_of"] is not None or params["inherited_hashes"]
    ):
        raise ValueError("The initial detection attempt cannot inherit other results")
    if job["id"] != cache["job_id"] and not isinstance(params["recovery_of"], str):
        raise ValueError("A detection continuation must identify its parent attempt")
    return job


def _state(conn, cache_id):
    cache, sequence = _cache_record(conn, _row(conn, "temporal_detection_caches", cache_id))
    jobs = {
        row["id"]: _job_record(cache, _decode(row))
        for row in conn.execute(
            "SELECT * FROM jobs WHERE kind=? AND json_extract(params,'$.cache_id')=? "
            "ORDER BY created_at,id",
            (KIND, cache_id),
        )
    }
    children = {}
    for job in jobs.values():
        parent_id = job["params"]["recovery_of"]
        if parent_id is not None:
            if (
                parent_id not in jobs
                or parent_id in children
                or jobs[parent_id]["status"] not in TERMINAL
            ):
                raise ValueError("Detection continuation history has a foreign or active parent")
            children[parent_id] = job["id"]
    chain, cursor = [], cache["job_id"]
    while cursor is not None:
        if cursor in chain or cursor not in jobs:
            raise ValueError("Detection continuation history is cyclic or incomplete")
        chain.append(cursor)
        cursor = children.get(cursor)
    if set(chain) != set(jobs):
        raise ValueError("Detection attempts do not form one continuation history")
    order = {frame_id: index for index, frame_id in enumerate(cache["config"]["frame_ids"])}
    entries = [
        _decode(row)
        for row in conn.execute(
            "SELECT * FROM temporal_detection_frames WHERE cache_id=?", (cache_id,)
        )
    ]
    if any(row["frame_id"] not in order for row in entries):
        raise ValueError("Detection cache contains a frame outside its frozen sequence")
    entries.sort(key=lambda row: order[row["frame_id"]])
    if [row["frame_id"] for row in entries] != cache["config"]["frame_ids"][: len(entries)]:
        raise ValueError("Committed detections must be a contiguous sequence prefix")
    execution_hashes = set()
    context = frame_validation_context(cache["config"], sequence["manifest"])
    for entry in entries:
        payload = validate_frame_payload(
            entry["payload"], cache["config"], sequence["manifest"], context=context
        )
        producer = jobs.get(entry["job_id"])
        if (
            entry["frame_id"] != payload["frame_id"]
            or payload_hash(payload) != entry["payload_sha256"]
            or _digest(entry["payload"]) != entry["payload_sha256"]
            or producer is None
            or producer["result"]["execution"] is None
            or payload["execution_signature_sha256"]
            != producer["result"]["execution"]["signature_sha256"]
        ):
            raise ValueError("Cached detections or their producer receipt changed")
        execution_hashes.add(payload["execution_signature_sha256"])
    if len(execution_hashes) > 1:
        raise ValueError("Detection cache mixes incompatible execution environments")
    hashes, prefix = [entry["payload_sha256"] for entry in entries], 0
    for job_id in chain:
        job = jobs[job_id]
        produced = [entry for entry in entries if entry["job_id"] == job_id]
        if (
            job["params"]["inherited_hashes"] != hashes[:prefix]
            or job["result"]["inherited_count"] != prefix
            or job["result"]["produced_count"] != len(produced)
            or produced != entries[prefix : prefix + len(produced)]
        ):
            raise ValueError("Detection attempt lost or rewrote its committed frame checkpoint")
        prefix += len(produced)
        if job["status"] == "succeeded" and prefix != len(order):
            raise ValueError("A successful detection attempt must cover the complete sequence")
    return cache, sequence, entries, [jobs[job_id] for job_id in chain]


def _coverage(cache, entries):
    completed, total = len(entries), len(cache["config"]["frame_ids"])
    return {
        "state": "complete" if completed == total else "partial" if completed else "empty",
        "total_count": total,
        "completed_count": completed,
        "remaining_count": total - completed,
        "result_sha256": _digest(
            {
                "fingerprint": cache["fingerprint"],
                "frames": [row["payload_sha256"] for row in entries],
            }
        )
        if completed == total
        else None,
    }


def _class_mapping(cache, sequence):
    contract = cache["config"]["detector"]["class_contract"]
    taxonomy = sequence["manifest"]["taxonomy"]
    if contract["taxonomy_id"] == "coco-2017-v1":
        return {
            str(item["coco_id"]): item["id"] for item in taxonomy["classes"] if "coco_id" in item
        }
    if contract["taxonomy"] != taxonomy:
        raise ValueError("Trained detector classes do not match the frozen sequence taxonomy")
    return {str(value): key for key, value in contract["output_class_mapping"].items()}


def get_detection_cache(store, cache_id):
    with store.connect() as conn:
        conn.execute("BEGIN")
        cache, sequence, entries, jobs = _state(conn, cache_id)
        mapping = _class_mapping(cache, sequence)
        return {
            **cache,
            "project_id": sequence["project_id"],
            "coverage": _coverage(cache, entries),
            "attempts": jobs,
            "latest_job_id": jobs[-1]["id"],
            "class_mapping": mapping,
            "unmapped_classes": [
                item["id"]
                for item in sequence["manifest"]["taxonomy"]["classes"]
                if item["id"] not in mapping.values()
            ],
            "limitations": LIMITATIONS,
        }


def list_detection_caches(store, sequence_id):
    if store.get("temporal_sequences", sequence_id) is None:
        raise KeyError(sequence_id)
    return [
        get_detection_cache(store, row["id"])
        for row in store.list("temporal_detection_caches", sequence_id=sequence_id)
    ]


def read_detection_cache(store, cache_id, *, min_score=None, class_ids=None):
    """Return a complete, derived view. Never recompute, mutate or renumber detections."""
    with store.connect() as conn:
        conn.execute("BEGIN")
        cache, sequence, entries, _ = _state(conn, cache_id)
        coverage = _coverage(cache, entries)
        if coverage["remaining_count"]:
            raise DetectionCacheConflict(
                "Detection cache is incomplete; continue the missing frames explicitly"
            )
        floor = cache["config"]["detector"]["min_score"]
        threshold = floor if min_score is None else min_score
        if not _finite(threshold) or not floor <= threshold <= 1:
            raise ValueError(
                "Requested score is below the saved cache floor; create a new lower-floor cache"
            )
        labels = {row["id"] for row in cache["config"]["detector"]["classes"]}
        if class_ids is not None and (
            not isinstance(class_ids, list)
            or not class_ids
            or len(class_ids) > len(labels)
            or any(type(value) is not int or value not in labels for value in class_ids)
            or len(set(class_ids)) != len(class_ids)
        ):
            raise ValueError(
                "Class filters must use distinct native labels from the frozen detector"
            )
        selected = labels if class_ids is None else set(class_ids)
        frames = [
            {
                **deepcopy(row["payload"]),
                "detections": [
                    deepcopy(detection)
                    for detection in row["payload"]["detections"]
                    if detection["score"] >= threshold and detection["label_id"] in selected
                ],
                "stored_payload_sha256": row["payload_sha256"],
                "producer_job_id": row["job_id"],
            }
            for row in entries
        ]
        return {
            "cache_id": cache_id,
            "cache_fingerprint": cache["fingerprint"],
            "result_sha256": coverage["result_sha256"],
            "filter": {"min_score": threshold, "class_ids": class_ids},
            "sequence": sequence["manifest"],
            "frames": frames,
            "limitations": LIMITATIONS,
        }


def _prepare(store, conn, sequence_id, project_id, settings, *, generation=None):
    sequence = _owned_sequence(conn, sequence_id, project_id)
    model = store.get("trained_models", settings["model_id"])
    if model is not None and record_project(store, "trained_models", model) != project_id:
        raise KeyError(settings["model_id"])
    detector = prepare_detector(store.root, **settings)
    config = validate_cache_config(
        {
            "schema": "iris-temporal-detection-cache-v1",
            "sequence_id": sequence_id,
            "sequence_sha256": sequence["manifest_sha256"],
            "detector": detector,
            "generation": generation,
            "frame_ids": [frame["frame_id"] for frame in sequence["manifest"]["frames"]],
        },
        sequence["manifest"],
    )
    _class_mapping({"config": config}, sequence)
    inference = detector["inference"]
    forward_passes = sum(
        len(tile_boxes(frame["width"], frame["height"], inference["tiling"]))
        if inference["mode"] == "tiled"
        else 1
        for frame in sequence["manifest"]["frames"]
    )
    return (
        sequence,
        config,
        {
            "frames": len(config["frame_ids"]),
            "forward_passes": forward_passes,
            "warmup_forward_passes_per_attempt": 1,
            "source_gap_frames": sum(
                gap["end_frame"] - gap["start_frame"] + 1 for gap in sequence["manifest"]["gaps"]
            ),
        },
    )


def preview_detection_cache(
    store,
    sequence_id,
    *,
    model_id,
    project_id=DEFAULT_PROJECT_ID,
    device="cpu",
    inference_mode="full",
    tile_size=640,
    overlap=0.2,
    min_score=0.001,
):
    settings = dict(
        model_id=model_id,
        device=device,
        inference_mode=inference_mode,
        tile_size=tile_size,
        overlap=overlap,
        min_score=min_score,
    )
    with store.connect() as conn:
        conn.execute("BEGIN")
        _, config, work = _prepare(store, conn, sequence_id, project_id, settings)
        fingerprint = cache_fingerprint(config)
        existing = conn.execute(
            "SELECT id FROM temporal_detection_caches WHERE fingerprint=?", (fingerprint,)
        ).fetchone()
        coverage = None
        # Empty caches still have useful, explicit zero coverage.
        if existing:
            state = _state(conn, existing["id"])
            coverage = _coverage(state[0], state[2])
        return {
            "config": config,
            "fingerprint": fingerprint,
            "work": work,
            "existing_cache_id": existing["id"] if existing else None,
            "coverage": coverage,
            "limitations": LIMITATIONS,
        }


def _queue(conn, cache_id, fingerprint, *, inherited_hashes=None, parent=None):
    identifier = new_id()
    inherited = inherited_hashes or []
    job = {
        "id": identifier,
        "kind": KIND,
        "status": "queued",
        "params": {
            "cache_id": cache_id,
            "cache_fingerprint": fingerprint,
            "recovery_of": parent,
            "inherited_hashes": inherited,
        },
        "result": _initial_result(cache_id, len(inherited)),
        "created_at": now(),
        "message": "Waiting to calculate the remaining temporal detections",
    }
    _insert(conn, "jobs", job)
    return identifier


def create_detection_cache(
    store,
    jobs,
    sequence_id,
    *,
    name,
    model_id,
    project_id=DEFAULT_PROJECT_ID,
    device="cpu",
    inference_mode="full",
    tile_size=640,
    overlap=0.2,
    min_score=0.001,
    force_new=False,
):
    name = _text(name, "Detection cache name")
    if type(force_new) is not bool:
        raise ValueError("force_new must be a boolean")
    settings = dict(
        model_id=model_id,
        device=device,
        inference_mode=inference_mode,
        tile_size=tile_size,
        overlap=overlap,
        min_score=min_score,
    )
    with jobs.guard, store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        sequence, config, _ = _prepare(
            store,
            conn,
            sequence_id,
            project_id,
            settings,
            generation=new_id() if force_new else None,
        )
        fingerprint = cache_fingerprint(config)
        existing = conn.execute(
            "SELECT id FROM temporal_detection_caches WHERE fingerprint=?", (fingerprint,)
        ).fetchone()
        if existing:
            _state(conn, existing["id"])
            cache_id, reused = existing["id"], True
        else:
            _verify_sequence_media(store, conn, sequence)
            cache_id, reused = new_id(), False
            job_id = _queue(conn, cache_id, fingerprint)
            _insert(
                conn,
                "temporal_detection_caches",
                {
                    "id": cache_id,
                    "sequence_id": sequence_id,
                    "name": name,
                    "config": config,
                    "fingerprint": fingerprint,
                    "job_id": job_id,
                    "created_at": now(),
                },
            )
            _state(conn, cache_id)
    return {**get_detection_cache(store, cache_id), "reused": reused}


def _recovery_preview(store, conn, job_id, project_id):
    job = _row(conn, "jobs", job_id)
    if job["kind"] != KIND:
        raise ValueError("Not a temporal detection attempt")
    raw = _row(conn, "temporal_detection_caches", job["params"].get("cache_id"))
    _owned_sequence(conn, raw["sequence_id"], project_id)
    cache, sequence, entries, attempts = _state(conn, raw["id"])
    coverage = _coverage(cache, entries)
    result = {
        "job_id": job_id,
        "cache_id": cache["id"],
        "available": False,
        "fingerprint": None,
        "successor_job_id": None,
        "reason": "",
        **{key: coverage[key] for key in ("total_count", "completed_count", "remaining_count")},
    }
    if attempts[-1]["id"] != job_id:
        result.update(
            successor_job_id=next(
                attempt["id"] for attempt in attempts if attempt["params"]["recovery_of"] == job_id
            ),
            reason="A continuation already exists; inspect its saved results",
        )
    elif job["status"] not in TERMINAL:
        result["reason"] = "Only failed, interrupted or cancelled attempts can be continued"
    elif not coverage["remaining_count"]:
        result["reason"] = "Every sequence frame is already saved; reuse the complete cache"
    else:
        try:
            verify_current_inputs(store, conn, cache, sequence)
        except (ValueError, RuntimeError, OSError) as exc:
            result["reason"] = str(exc)
        else:
            result.update(
                available=True,
                fingerprint=_digest(
                    {
                        "job": job,
                        "cache": cache["fingerprint"],
                        "frames": [row["payload_sha256"] for row in entries],
                    }
                ),
            )
    return result


def verify_current_inputs(store, conn, cache, sequence):
    detector = cache["config"]["detector"]
    inference = detector["inference"]
    tiling = inference.get("tiling", {})
    current = prepare_detector(
        store.root,
        detector["model_id"],
        device=detector["device"],
        inference_mode=inference["mode"],
        tile_size=tiling.get("tile_size", 640),
        overlap=tiling.get("overlap", 0.2),
        min_score=detector["min_score"],
    )
    if current != detector:
        raise DetectionCacheConflict(
            "Detector weights, runtime or preprocessing changed; create a fresh cache explicitly"
        )
    _verify_sequence_media(store, conn, sequence)


def preview_detection_recovery(store, job_id, project_id=DEFAULT_PROJECT_ID):
    with store.connect() as conn:
        conn.execute("BEGIN")
        return _recovery_preview(store, conn, job_id, project_id)


def recover_detection_cache(store, jobs, job_id, *, fingerprint, project_id=DEFAULT_PROJECT_ID):
    with jobs.guard, store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        preview = _recovery_preview(store, conn, job_id, project_id)
        if not preview["available"] or fingerprint != preview["fingerprint"]:
            raise DetectionCacheConflict(
                preview["reason"] or "Detection recovery preview changed; inspect it again"
            )
        cache, _, entries, _ = _state(conn, preview["cache_id"])
        identifier = _queue(
            conn,
            cache["id"],
            cache["fingerprint"],
            inherited_hashes=[row["payload_sha256"] for row in entries],
            parent=job_id,
        )
    return store.get("jobs", identifier)


def validate_detection_records(connection):
    """Check archived caches and receipts without loading weights or current runtimes."""
    try:
        for row in connection.execute("SELECT id FROM temporal_detection_caches"):
            _state(connection, row["id"])
        for row in connection.execute("SELECT * FROM jobs WHERE kind=?", (KIND,)):
            job = _decode(row)
            _row(connection, "temporal_detection_caches", job["params"]["cache_id"])
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError("Detection cache references are invalid or missing") from exc


def run_detection_cache(store, job_id, progress, cancelled, detector_factory=None):
    from iris.temporal_detection_worker import run

    return run(store, job_id, progress, cancelled, detector_factory=detector_factory)
