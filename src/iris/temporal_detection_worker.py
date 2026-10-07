"""Publish complete temporal frame outputs, with one claimed worker per attempt."""

import json
import time

from iris.inference import _load_verified_frame
from iris.media import _file_hash
from iris.prediction_taxonomy import validate_output_labels
from iris.store import _encode, new_id, now
from iris.temporal import _digest, _insert, _row
from iris.temporal_detection_contracts import (
    frame_validation_context,
    payload_hash,
    validate_frame_payload,
)
from iris.temporal_detections import (
    KIND,
    DetectionCacheConflict,
    _execution,
    _finite,
    _state,
    verify_current_inputs,
)
from iris.temporal_detector import detector_factory as default_factory
from iris.temporal_detector import verify_detector_metadata
from iris.tiling import TiledInferenceCancelled, tile_boxes, tiled_predict


def _active(job, token):
    return (
        job["status"] == "running"
        and not job["cancel_requested"]
        and job["result"]["worker_token"] == token
    )


def _write_result(conn, job_id, result):
    conn.execute(
        "UPDATE jobs SET result=? WHERE id=?", (_encode({"result": result})["result"], job_id)
    )


def _raw_prediction(prediction, frame, config):
    if not isinstance(prediction, dict) or not isinstance(prediction.get("detections"), list):
        raise ValueError("Detector must return a complete frame prediction")
    size = prediction.get("input_size")
    if (
        not isinstance(size, list)
        or any(type(value) is not int for value in size)
        or size != [frame["width"], frame["height"]]
    ):
        raise ValueError("Detector returned dimensions different from the frozen frame")
    cap = 100 if config["inference"]["mode"] == "full" else 300
    if len(prediction["detections"]) > cap:
        raise ValueError("Detector exceeded the frozen postprocessing detection cap")
    if any(not isinstance(detection, dict) for detection in prediction["detections"]):
        raise ValueError("Detector detections must be objects")
    validate_output_labels(prediction["detections"], config["class_contract"])
    for detection in prediction["detections"]:
        box, score = detection.get("box"), detection.get("score")
        if not isinstance(box, list) or len(box) != 4 or not all(_finite(value) for value in box):
            raise ValueError("Detector returned invalid box coordinates")
        if (
            not 0 <= box[0] < box[2] <= frame["width"]
            or not 0 <= box[1] < box[3] <= frame["height"]
        ):
            raise ValueError("Detector box does not fit the source frame")
        floor = config["native_filtering"]["score_threshold"]
        if (
            not _finite(score)
            or not floor <= score <= 1
            or score == floor
            and config["output_policy"]["native_score_comparison"] == "gt"
        ):
            raise ValueError("Detector score violates its frozen native threshold")
    timing = prediction.get("timing")
    if not isinstance(timing, dict) or any(
        not _finite(timing.get(field))
        for field in (
            "preprocess_ms",
            "inference_ms",
            "postprocess_ms",
            "total_ms",
        )
    ):
        raise ValueError("Detector did not return valid stage timings")


def _publish(store, cache, job_id, token, payload, position):
    """Output and attempt checkpoint commit together; no late/cancelled worker can append."""
    with store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        job = _row(conn, "jobs", job_id)
        if not _active(job, token):
            raise TiledInferenceCancelled()
        count = conn.execute(
            "SELECT count(*) FROM temporal_detection_frames WHERE cache_id=?", (cache["id"],)
        ).fetchone()[0]
        if count != position or job["result"]["completed_count"] != position:
            raise DetectionCacheConflict("Detection checkpoint changed before frame publication")
        _insert(
            conn,
            "temporal_detection_frames",
            {
                "id": new_id(),
                "cache_id": cache["id"],
                "frame_id": payload["frame_id"],
                "job_id": job_id,
                "payload": payload,
                "payload_sha256": payload_hash(payload),
                "created_at": now(),
            },
        )
        result = {
            **job["result"],
            "completed_count": position + 1,
            "produced_count": job["result"]["produced_count"] + 1,
        }
        _write_result(conn, job_id, result)
        conn.execute(
            "UPDATE jobs SET progress=? WHERE id=?",
            ((position + 1) / len(cache["config"]["frame_ids"]), job_id),
        )


def run(store, job_id, progress, cancelled, detector_factory=None):
    with store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        job = _row(conn, "jobs", job_id)
        if job["kind"] != KIND:
            raise ValueError("Not a temporal detection job")
        cache, sequence, entries, attempts = _state(conn, job["params"].get("cache_id"))
        if attempts[-1]["id"] != job_id or job["status"] != "running":
            raise DetectionCacheConflict("Only the latest running detection attempt can execute")
        if job["result"]["worker_token"] is not None:
            raise DetectionCacheConflict("This detection attempt was already claimed")
        if job["cancel_requested"]:
            return job["result"]
        token = new_id()
        _write_result(conn, job_id, {**job["result"], "worker_token": token})
    settings = cache["config"]["detector"]
    inference = settings["inference"]
    frames = sequence["manifest"]["frames"]
    context = frame_validation_context(cache["config"], sequence["manifest"])
    inherited = len(entries)

    def stopped():
        if cancelled():
            return True
        return not _active(store.get("jobs", job_id), token)

    detector = None
    try:
        if stopped():
            raise TiledInferenceCancelled()
        with store.connect() as conn:
            verify_current_inputs(store, conn, cache, sequence)
        progress(inherited / len(frames), "Loading the frozen temporal detector")
        started = time.perf_counter()
        detector = (detector_factory or default_factory)(store.root, settings)
        load_ms = (time.perf_counter() - started) * 1000
        metadata = json.loads(json.dumps(detector.metadata, allow_nan=False))
        signature = verify_detector_metadata(settings, metadata)
        signature_sha = _digest(signature)
        if entries and entries[0]["payload"]["execution_signature_sha256"] != signature_sha:
            raise DetectionCacheConflict(
                "Execution hardware or runtime changed; create a fresh cache explicitly"
            )
        if stopped():
            raise TiledInferenceCancelled()
        first = frames[inherited]
        frame = store.get("frames", first["frame_id"])
        if _file_hash(store.artifact_path(frame["path"])) != first["file_sha256"]:
            raise ValueError("Warmup source file changed")
        progress(inherited / len(frames), "Warming up the detector (separate from frame timings)")
        started = time.perf_counter()
        with _load_verified_frame(store, frame, first["sha256"]) as image:
            if inference["mode"] == "tiled":
                with image.crop(tuple(tile_boxes(*image.size, inference["tiling"])[0])) as crop:
                    detector.warmup(crop)
            else:
                detector.warmup(image)
        execution = {
            "signature": signature,
            "signature_sha256": signature_sha,
            "metadata": metadata,
            "metadata_sha256": _digest(metadata),
            "load_ms": load_ms,
            "warmup_ms": (time.perf_counter() - started) * 1000,
            "warmup_frame_id": first["frame_id"],
        }
        _execution(cache, execution)
        with store.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            current = _row(conn, "jobs", job_id)
            if not _active(current, token):
                raise TiledInferenceCancelled()
            _write_result(conn, job_id, {**current["result"], "execution": execution})
        for position in range(inherited, len(frames)):
            if stopped():
                raise TiledInferenceCancelled()
            frozen = frames[position]
            frame = store.get("frames", frozen["frame_id"])
            started = time.perf_counter()
            if _file_hash(store.artifact_path(frame["path"])) != frozen["file_sha256"]:
                raise ValueError("Source frame bytes changed after the cache was queued")
            with _load_verified_frame(store, frame, frozen["sha256"]) as image:
                decoded = time.perf_counter()
                prediction = (
                    tiled_predict(detector, image, inference["tiling"], cancelled=stopped)
                    if inference["mode"] == "tiled"
                    else detector.predict(image)
                )
            _raw_prediction(prediction, frozen, settings)
            filtered_at = time.perf_counter()
            detections = [
                {
                    "detection_index": index,
                    **{key: value[key] for key in ("label_id", "label", "score", "box")},
                }
                for index, value in enumerate(prediction["detections"])
                if value["score"] >= settings["min_score"]
            ]
            filter_ms = (time.perf_counter() - filtered_at) * 1000
            timing = prediction["timing"]
            payload = {
                "schema": "iris-temporal-detection-frame-v1",
                "cache_fingerprint": cache["fingerprint"],
                "frame_id": frozen["frame_id"],
                "frame_index": frozen["frame_index"],
                "timestamp_seconds": frozen["timestamp_seconds"],
                "frame_sha256": frozen["sha256"],
                "file_sha256": frozen["file_sha256"],
                "input_size": [frozen["width"], frozen["height"]],
                "detections": detections,
                "native_detection_count": len(prediction["detections"]),
                "execution_signature_sha256": signature_sha,
                "timing": {
                    **{
                        key: timing[key]
                        for key in ("preprocess_ms", "inference_ms", "postprocess_ms")
                    },
                    "decode_ms": (decoded - started) * 1000,
                    "crop_ms": timing.get("crop_ms", 0.0),
                    "merge_ms": timing.get("merge_ms", 0.0),
                    "filter_ms": filter_ms,
                    "total_ms": (time.perf_counter() - started) * 1000,
                },
                "work": {
                    "forward_passes": timing.get("forward_passes", 1),
                    "tile_count": timing.get("tile_count", 0),
                },
            }
            payload = validate_frame_payload(
                payload, cache["config"], sequence["manifest"], context=context
            )
            if stopped():
                raise TiledInferenceCancelled()
            _publish(store, cache, job_id, token, payload, position)
            progress(
                (position + 1) / len(frames),
                f"Saved {position + 1} / {len(frames)} temporal frame results",
            )
    except TiledInferenceCancelled:
        with store.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            current = _row(conn, "jobs", job_id)
            if current["status"] == "running" and current["result"]["worker_token"] == token:
                _write_result(conn, job_id, {**current["result"], "cancelled": True})
    finally:
        del detector
    return store.get("jobs", job_id)["result"]
