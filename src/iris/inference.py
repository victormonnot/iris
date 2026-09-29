"""Saved comparisons over a frozen frame selection; no automatic model downloads."""

import json
import math
import time
from collections.abc import Callable

from PIL import Image

from iris.media import _pixel_hash
from iris.models import TorchvisionDetector, catalog, get_spec
from iris.store import Store, new_id, now

MAX_COMPARISON_FRAMES = 100
PROTOCOL = {
    "version": "torchvision-forward-v1",
    "batch_size": 1,
    "warmup_frames": 1,
    "warmup_in_timings": False,
    "decode_ms": "Read local normalized PNG and verify its pixel hash",
    "preprocess_ms": "Convert RGB pixels to a tensor and transfer to the device",
    "inference_ms": (
        "Full model forward, including internal resize, normalization, NMS and rescaling"
    ),
    "postprocess_ms": "Transfer detections to CPU and convert them to JSON values",
    "total_ms": "Decode and detector call; excludes loading weights, warmup and database writes",
    "quality_metrics": "Unavailable: no human-validated reference labels in this increment",
}


def create_comparison(
    store: Store,
    jobs,
    session_id: str,
    *,
    name: str,
    frame_ids: list[str],
    model_ids: list[str],
    device: str = "cpu",
) -> dict:
    if not store.get("sessions", session_id):
        raise ValueError("Session not found")
    name = name.strip()
    if not name or len(name) > 160:
        raise ValueError("Comparison name must contain between 1 and 160 characters")
    if not 1 <= len(frame_ids) <= MAX_COMPARISON_FRAMES or len(set(frame_ids)) != len(frame_ids):
        raise ValueError("Choose between 1 and 100 distinct frames")
    if not 1 <= len(model_ids) <= 2 or len(set(model_ids)) != len(model_ids):
        raise ValueError("Choose one or two distinct models")
    if device not in {"cpu", "cuda"}:
        raise ValueError("Device must be cpu or cuda")
    frames = []
    for frame_id in frame_ids:
        frame = store.get("frames", frame_id)
        if frame is None or frame["session_id"] != session_id:
            raise ValueError("Every frame must belong to this flight session")
        frames.append(frame)
    for model_id in model_ids:
        get_spec(model_id)
    available = {model["id"]: model for model in catalog(store.root)}
    for model_id in model_ids:
        model = available.get(model_id)
        if model is None or model["status"] != "ready":
            reason = model.get("reason") if model else None
            raise RuntimeError(reason or f"Model {model_id} is not ready")
    comparison_id, job_id, created_at = new_id(), new_id(), now()
    config = {
        "device": device,
        "warmup": 1,
        "taxonomy": "coco-2017-v1",
        "class_mapping": {"person": 1, "car": 3},
        "frame_hashes": {frame["id"]: frame["sha256"] for frame in frames},
        "protocol": PROTOCOL,
    }
    # Publish the frozen input selection and its queue entry together. A worker
    # can never claim a job whose comparison has not been committed yet.
    with jobs.guard, store.connect() as conn:
        conn.execute(
            "INSERT INTO jobs (id,kind,status,params,message,created_at) VALUES (?,?,?,?,?,?)",
            (
                job_id,
                "infer",
                "queued",
                json.dumps({"comparison_id": comparison_id}),
                "Waiting to compare models on the selected frames",
                created_at,
            ),
        )
        conn.execute(
            "INSERT INTO comparisons "
            "(id,session_id,name,frame_ids,model_ids,config,job_id,created_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (
                comparison_id,
                session_id,
                name,
                json.dumps(frame_ids),
                json.dumps(model_ids),
                json.dumps(config),
                job_id,
                created_at,
            ),
        )
    return comparison_summary(store, store.get("comparisons", comparison_id))


def comparison_summary(store: Store, comparison: dict) -> dict:
    return {**comparison, "job": store.get("jobs", comparison["job_id"])}


def comparison_detail(store: Store, comparison_id: str) -> dict:
    comparison = store.get("comparisons", comparison_id)
    if comparison is None:
        raise KeyError(comparison_id)
    frames = []
    session = store.get("sessions", comparison["session_id"])
    for frame_id in comparison["frame_ids"]:
        frame = store.get("frames", frame_id)
        asset = store.get("assets", frame["asset_id"])
        frames.append(
            {
                **{key: value for key, value in frame.items() if key != "path"},
                "source_filename": asset["filename"],
                "session_name": session["name"],
                "scene_group": session["scene_group"],
            }
        )
    return {
        **comparison_summary(store, comparison),
        "frames": frames,
        "runs": store.list("runs", comparison_id=comparison_id),
        "predictions": store.list("predictions", comparison_id=comparison_id),
    }


def _load_verified_frame(store: Store, frame: dict, expected_hash: str) -> Image.Image:
    if frame["sha256"] != expected_hash:
        raise ValueError(f"Frame {frame['id']} changed since this comparison was created")
    with Image.open(store.artifact_path(frame["path"])) as source:
        image = source.convert("RGB")
        image.load()
    if image.size != (frame["width"], frame["height"]) or _pixel_hash(image) != expected_hash:
        image.close()
        raise ValueError(f"Frame {frame['id']} content no longer matches its recorded hash")
    return image


def _validate_prediction(prediction: dict, frame: dict):
    if prediction["input_size"] != [frame["width"], frame["height"]]:
        raise ValueError("Detector returned predictions for unexpected image dimensions")
    for detection in prediction["detections"]:
        box, score = detection["box"], detection["score"]
        if len(box) != 4 or not all(math.isfinite(value) for value in [*box, score]):
            raise ValueError("Detector returned nonfinite coordinates or confidence")
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= frame["width"] and 0 <= y1 < y2 <= frame["height"]):
            raise ValueError("Detector returned a box outside the original image")
        if not 0 <= score <= 1 or not isinstance(detection["label_id"], int):
            raise ValueError("Detector returned an invalid category or confidence")
    for key in ("preprocess_ms", "inference_ms", "postprocess_ms", "total_ms"):
        value = prediction["timing"][key]
        if not math.isfinite(value) or value < 0:
            raise ValueError("Detector returned an invalid timing measurement")


def run_comparison(
    store: Store,
    comparison_id: str,
    progress: Callable[[float, str], None],
    cancelled: Callable[[], bool],
    detector_factory=None,
) -> dict:
    comparison = store.get("comparisons", comparison_id)
    if comparison is None:
        raise ValueError("Comparison not found")
    if store.list("runs", comparison_id=comparison_id):
        raise ValueError("A comparison is immutable; create a new comparison to run it again")
    factory = detector_factory or TorchvisionDetector
    frame_ids, model_ids, config = (
        comparison["frame_ids"],
        comparison["model_ids"],
        comparison["config"],
    )
    result = {
        "comparison_id": comparison_id,
        "frames_total": len(frame_ids),
        "models_total": len(model_ids),
        "predictions_created": 0,
        "cancelled": False,
    }
    total = len(frame_ids) * len(model_ids)
    for model_id in model_ids:
        if cancelled():
            result["cancelled"] = True
            break
        progress(result["predictions_created"] / total, f"Loading {get_spec(model_id)['name']}")
        detector = factory(store.root, model_id, device=config["device"])
        try:
            run = store.insert(
                "runs",
                {
                    "id": new_id(),
                    "comparison_id": comparison_id,
                    "model_id": model_id,
                    "metadata": {**detector.metadata, "protocol": config["protocol"]},
                    "created_at": now(),
                },
            )
            first_frame = store.get("frames", frame_ids[0])
            with _load_verified_frame(
                store, first_frame, config["frame_hashes"][frame_ids[0]]
            ) as image:
                if cancelled():
                    result["cancelled"] = True
                    break
                progress(result["predictions_created"] / total, "Warming up model (not timed)")
                detector.warmup(image)
            for frame_id in frame_ids:
                if cancelled():
                    result["cancelled"] = True
                    break
                frame = store.get("frames", frame_id)
                started = time.perf_counter()
                with _load_verified_frame(store, frame, config["frame_hashes"][frame_id]) as image:
                    decoded = time.perf_counter()
                    prediction = detector.predict(image)
                elapsed_ms = (time.perf_counter() - started) * 1000
                _validate_prediction(prediction, frame)
                store.insert(
                    "predictions",
                    {
                        "id": new_id(),
                        "comparison_id": comparison_id,
                        "run_id": run["id"],
                        "frame_id": frame_id,
                        "model_id": model_id,
                        "detections": prediction["detections"],
                        "input_size": prediction["input_size"],
                        "timing": {
                            **prediction["timing"],
                            "decode_ms": (decoded - started) * 1000,
                            "total_ms": elapsed_ms,
                        },
                        "created_at": now(),
                    },
                )
                result["predictions_created"] += 1
                progress(
                    result["predictions_created"] / total,
                    f"Saved {result['predictions_created']} / {total} model/frame predictions",
                )
            if result["cancelled"]:
                break
        finally:
            del detector
    return result
