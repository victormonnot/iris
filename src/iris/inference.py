"""Saved comparisons over a frozen frame selection; no automatic model downloads."""

import json
import math
import time
from collections.abc import Callable

from PIL import Image

from iris.comparison_replay import comparison_replay
from iris.media import _pixel_hash
from iris.models import TorchvisionDetector, catalog, get_spec
from iris.prediction_taxonomy import output_contract, validate_output_labels
from iris.store import Store, new_id, now
from iris.tiling import (
    MAX_TILES_PER_FRAME,
    TiledInferenceCancelled,
    tile_boxes,
    tiled_predict,
    validate_tiling_config,
)

MAX_COMPARISON_FRAMES = 100
MAX_FORWARD_PASSES = 512
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
    "quality_metrics": "Not computed: this comparison reports predictions and timings only",
}

TILED_PROTOCOL = {
    **PROTOCOL,
    "version": "torchvision-tiled-v1",
    "warmup_frames": "One full image or first tile per run, matching the inference variant",
    "inference_ms": "Sum of model forward times across all tiles (or one full image)",
    "preprocess_ms": "Sum of detector preprocessing times across all passes",
    "postprocess_ms": "Sum of detector serialization times across all passes",
    "crop_ms": "Copy original pixels into each tile; no additional resize or padding",
    "merge_ms": "Map tile detections to original coordinates and apply class-aware NMS",
    "total_ms": (
        "Decode, verification, crops, all detector passes and merge; "
        "excludes weights, warmup, progress reporting and database writes"
    ),
    "merge": "Stable descending-score NMS per COCO class, IoU > 0.5; keep at most 300 boxes",
    "tile_layout": "Row-major, floor(size * (1 - overlap)) stride, final tile anchored to edge",
}


def comparison_lanes(comparison: dict) -> list[dict]:
    """Planned run identities; legacy comparisons always used full images."""
    model_ids = comparison["model_ids"]
    if (
        not isinstance(model_ids, list)
        or not 1 <= len(model_ids) <= 2
        or any(not isinstance(item, str) or not item for item in model_ids)
        or len(set(model_ids)) != len(model_ids)
    ):
        raise ValueError("Comparison has invalid model identities")
    config = comparison["config"]
    if not isinstance(config, dict) or not isinstance(config.get("inference", {}), dict):
        raise ValueError("Comparison has invalid inference settings")
    mode = config.get("inference", {}).get("mode", "full")
    if (
        not isinstance(mode, str)
        or mode not in {"full", "tiled", "paired"}
        or (mode == "paired" and len(model_ids) != 1)
    ):
        raise ValueError("Comparison has an invalid inference mode")
    expected = [
        {"model_id": model_id, "variant": variant}
        for model_id in model_ids
        for variant in (["full", "tiled"] if mode == "paired" else [mode])
    ]
    if "lanes" in config and config["lanes"] != expected:
        raise ValueError("Comparison run variants do not match its inference mode")
    return expected


def _work_plan(
    frames: list[dict],
    lanes: list[dict],
    inference: dict,
    *,
    max_forward_passes: int = MAX_FORWARD_PASSES,
) -> dict:
    tiled = any(lane["variant"] == "tiled" for lane in lanes)
    tiles = [
        {
            "frame_id": frame["id"],
            "input_size": [frame["width"], frame["height"]],
            "tile_count": len(tile_boxes(frame["width"], frame["height"], inference["tiling"]))
            if tiled
            else 0,
        }
        for frame in frames
    ]
    passes = sum(
        sum(item["tile_count"] for item in tiles) if lane["variant"] == "tiled" else len(frames)
        for lane in lanes
    )
    total = passes + len(lanes)
    if total > max_forward_passes:
        raise ValueError(
            f"This selection needs {total} detector passes including warmup; "
            f"the limit is {max_forward_passes}. Use fewer images or larger tiles."
        )
    return {
        "frames_total": len(frames),
        "forward_passes": passes,
        "warmup_passes": len(lanes),
        "total_forward_passes": total,
        "tiles": tiles,
        "limits": {
            "max_tiles_per_frame": MAX_TILES_PER_FRAME,
            "max_forward_passes": max_forward_passes,
        },
    }


def _prepare_comparison(
    store,
    session_id,
    *,
    name,
    frame_ids,
    model_ids,
    device="cpu",
    inference_mode="full",
    tile_size=640,
    overlap=0.2,
) -> tuple[list[dict], dict]:
    if not store.get("sessions", session_id):
        raise ValueError("Session not found")
    if not isinstance(name, str) or not 1 <= len(name.strip()) <= 160:
        raise ValueError("Comparison name must contain between 1 and 160 characters")
    if not 1 <= len(frame_ids) <= MAX_COMPARISON_FRAMES or len(set(frame_ids)) != len(frame_ids):
        raise ValueError("Choose between 1 and 100 distinct frames")
    if device not in {"cpu", "cuda"}:
        raise ValueError("Device must be cpu or cuda")
    tiling = validate_tiling_config(tile_size, overlap)
    inference = {"mode": inference_mode}
    if inference_mode != "full":
        inference.update(tiling=tiling, algorithm="iris-tiling-v1")
    lanes = comparison_lanes({"model_ids": model_ids, "config": {"inference": inference}})
    frames = []
    for frame_id in frame_ids:
        frame = store.get("frames", frame_id)
        if frame is None or frame["session_id"] != session_id:
            raise ValueError("Every frame must belong to this session")
        frames.append(frame)
    for model_id in model_ids:
        get_spec(model_id, store.root)
    return frames, {"lanes": lanes, "inference": inference, **_work_plan(frames, lanes, inference)}


def preview_comparison(store: Store, session_id: str, **settings) -> dict:
    """Count bounded detector work without loading weights, decoding images or writing jobs."""
    return _prepare_comparison(store, session_id, **settings)[1]


def create_comparison(
    store: Store,
    jobs,
    session_id: str,
    *,
    name: str,
    frame_ids: list[str],
    model_ids: list[str],
    device: str = "cpu",
    inference_mode: str = "full",
    tile_size: int = 640,
    overlap: float = 0.2,
) -> dict:
    frames, plan = _prepare_comparison(
        store,
        session_id,
        name=name,
        frame_ids=frame_ids,
        model_ids=model_ids,
        device=device,
        inference_mode=inference_mode,
        tile_size=tile_size,
        overlap=overlap,
    )
    name = name.strip()
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
        "model_hashes": {
            model_id: available[model_id]["weight_sha256"]
            for model_id in model_ids
            if available[model_id].get("weight_sha256")
        },
        "protocol": PROTOCOL if inference_mode == "full" else TILED_PROTOCOL,
        "inference": plan["inference"],
        "lanes": plan["lanes"],
        "work": {key: value for key, value in plan.items() if key not in {"inference", "lanes"}},
    }
    contracts = {model_id: output_contract(available[model_id]) for model_id in model_ids}
    if any(
        contract["taxonomy_id"] not in {"coco-2017-v1", "iris-objects-v1"}
        for contract in contracts.values()
    ):
        config.update(taxonomy="model-specific-v1", model_class_contracts=contracts)
        config.pop("class_mapping")
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
    runs = {
        (run["model_id"], run["variant"]): run
        for run in store.list("runs", comparison_id=comparison["id"])
    }
    return {
        **comparison,
        "job": store.get("jobs", comparison["job_id"]),
        "lanes": [
            {**lane, "run_id": runs.get((lane["model_id"], lane["variant"]), {}).get("id")}
            for lane in comparison_lanes(comparison)
        ],
    }


def comparison_detail(store: Store, comparison_id: str) -> dict:
    comparison = store.get("comparisons", comparison_id)
    if comparison is None:
        raise KeyError(comparison_id)
    frames, saved_frames, assets = [], [], {}
    session = store.get("sessions", comparison["session_id"])
    for frame_id in comparison["frame_ids"]:
        frame = store.get("frames", frame_id)
        asset = store.get("assets", frame["asset_id"])
        saved_frames.append(frame)
        assets[asset["id"]] = asset
        frames.append(
            {
                **{key: value for key, value in frame.items() if key != "path"},
                "source_filename": asset["filename"],
                "session_name": session["name"],
                "scene_group": session["scene_group"],
            }
        )
    summary = comparison_summary(store, comparison)
    predictions = store.list("predictions", comparison_id=comparison_id)
    return {
        **summary,
        "frames": frames,
        "runs": store.list("runs", comparison_id=comparison_id),
        "predictions": predictions,
        "replay": comparison_replay(store, saved_frames, assets, summary["lanes"], predictions),
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
    preannotation = "preannotation" in config
    if preannotation:
        from iris.preannotation import save_preannotation_prediction, validate_preannotation

        validate_preannotation(store, comparison)
    result = {
        "comparison_id": comparison_id,
        "frames_total": len(frame_ids),
        "models_total": len(model_ids),
        "predictions_created": 0,
        "cancelled": False,
    }
    if preannotation:
        result["preannotation"] = {
            "frames": [],
            "frames_ready": 0,
            "frames_issues": 0,
            "suggestions_created": 0,
        }
    lanes = comparison_lanes(comparison)
    inference = config.get("inference", {"mode": "full"})
    if inference["mode"] != "full":
        tiling = inference.get("tiling", {})
        if inference.get("algorithm") != "iris-tiling-v1" or tiling != validate_tiling_config(
            tiling.get("tile_size"), tiling.get("overlap")
        ):
            raise ValueError("Unsupported saved tiling protocol")
    frames = [store.get("frames", frame_id) for frame_id in frame_ids]
    if any(frame is None or frame["session_id"] != comparison["session_id"] for frame in frames):
        raise ValueError("The saved comparison contains missing or foreign frames")
    work = _work_plan(frames, lanes, inference)
    if "work" in config and work != config["work"]:
        raise ValueError("The saved inference plan no longer matches the selected frames")
    total = len(frame_ids) * len(lanes)
    contracts = config.get("model_class_contracts")
    if contracts is not None:
        if not isinstance(contracts, dict) or set(contracts) != set(model_ids):
            raise ValueError("Saved comparison class definitions do not match its models")
        for model_id in model_ids:
            if output_contract(get_spec(model_id, store.root)) != contracts[model_id]:
                raise ValueError("Checkpoint class definitions changed since comparison was queued")
    for model_id in model_ids:
        if cancelled():
            result["cancelled"] = True
            break
        progress(
            result["predictions_created"] / total,
            f"Loading {get_spec(model_id, store.root)['name']}",
        )
        detector = factory(store.root, model_id, device=config["device"])
        try:
            expected_hash = config.get("model_hashes", {}).get(model_id)
            if expected_hash and detector.metadata.get("weight_sha256") != expected_hash:
                raise ValueError("Checkpoint changed since this comparison was created")
            for lane in (lane for lane in lanes if lane["model_id"] == model_id):
                if cancelled():
                    raise TiledInferenceCancelled()
                variant = lane["variant"]
                tile_plan = (
                    {
                        frame["id"]: tile_boxes(
                            frame["width"], frame["height"], inference["tiling"]
                        )
                        for frame in frames
                    }
                    if variant == "tiled"
                    else {}
                )
                metadata = {
                    **detector.metadata,
                    "protocol": config["protocol"],
                    "inference": {"variant": variant},
                }
                if contracts is not None:
                    metadata["class_contract"] = contracts[model_id]
                if variant == "tiled":
                    metadata["inference"].update(
                        algorithm=inference["algorithm"],
                        **inference["tiling"],
                        tile_boxes=tile_plan,
                    )
                run = store.insert(
                    "runs",
                    {
                        "id": new_id(),
                        "comparison_id": comparison_id,
                        "model_id": model_id,
                        "variant": variant,
                        "metadata": metadata,
                        "created_at": now(),
                    },
                )
                with _load_verified_frame(
                    store, frames[0], config["frame_hashes"][frame_ids[0]]
                ) as image:
                    if cancelled():
                        raise TiledInferenceCancelled()
                    progress(
                        result["predictions_created"] / total,
                        f"Warming up {variant} run (not timed)",
                    )
                    if variant == "tiled":
                        with image.crop(tuple(tile_plan[frame_ids[0]][0])) as crop:
                            detector.warmup(crop)
                    else:
                        detector.warmup(image)
                for frame in frames:
                    if cancelled():
                        raise TiledInferenceCancelled()
                    progress_overhead = 0.0

                    def tile_progress(done, count, frame_id=frame["id"]):
                        nonlocal progress_overhead
                        reporting = time.perf_counter()
                        progress(
                            (result["predictions_created"] + done / count) / total,
                            f"Tiled frame {frame_ids.index(frame_id) + 1} / {len(frames)}: "
                            f"{done} / {count} tiles; merging before saving",
                        )
                        progress_overhead += time.perf_counter() - reporting

                    started = time.perf_counter()
                    with _load_verified_frame(
                        store, frame, config["frame_hashes"][frame["id"]]
                    ) as image:
                        decoded = time.perf_counter()
                        if variant == "tiled":
                            prediction = tiled_predict(
                                detector,
                                image,
                                inference["tiling"],
                                cancelled=cancelled,
                                progress=tile_progress,
                            )
                        else:
                            prediction = detector.predict(image)
                            if not preannotation or (
                                isinstance(prediction, dict)
                                and isinstance(prediction.get("timing"), dict)
                            ):
                                prediction["timing"].update(
                                    forward_passes=1,
                                    tile_count=0,
                                    crop_ms=0.0,
                                    merge_ms=0.0,
                                )
                    elapsed_ms = max(0, time.perf_counter() - started - progress_overhead) * 1000
                    if preannotation:
                        timing = (
                            prediction.get("timing", {}) if isinstance(prediction, dict) else {}
                        )
                        receipt = save_preannotation_prediction(
                            store,
                            comparison,
                            run,
                            frame,
                            prediction,
                            {
                                **(timing if isinstance(timing, dict) else {}),
                                "decode_ms": (decoded - started) * 1000,
                                "total_ms": elapsed_ms,
                            },
                            cancelled,
                        )
                        result["preannotation"]["frames"].append(
                            {
                                "frame_id": frame["id"],
                                **receipt,
                            }
                        )
                        result["preannotation"]["frames_ready"] += receipt["state"] in {
                            "pending_review",
                            "no_proposals",
                        }
                        result["preannotation"]["frames_issues"] += receipt["state"] in {
                            "conflict",
                            "invalid_output",
                        }
                        result["preannotation"]["suggestions_created"] += receipt["proposal_count"]
                    else:
                        _validate_prediction(prediction, frame)
                        if contracts is not None:
                            validate_output_labels(prediction["detections"], contracts[model_id])
                        if cancelled():
                            raise TiledInferenceCancelled()
                        store.insert(
                            "predictions",
                            {
                                "id": new_id(),
                                "comparison_id": comparison_id,
                                "run_id": run["id"],
                                "frame_id": frame["id"],
                                "model_id": model_id,
                                "detections": prediction["detections"],
                                "input_size": prediction["input_size"],
                                "metadata": prediction.get("metadata", {}),
                                "timing": {
                                    **prediction["timing"],
                                    "decode_ms": (decoded - started) * 1000,
                                    "total_ms": elapsed_ms,
                                },
                                "created_at": now(),
                            },
                        )
                    result["predictions_created"] += 1
                    if preannotation:
                        with store.connect() as conn:
                            conn.execute(
                                "UPDATE jobs SET result=? WHERE id=? "
                                "AND status IN ('queued','running')",
                                (json.dumps(result, allow_nan=False), comparison["job_id"]),
                            )
                    progress(
                        result["predictions_created"] / total,
                        f"Saved {result['predictions_created']} / {total} run/frame predictions",
                    )
                    if preannotation and cancelled():
                        raise TiledInferenceCancelled()
        except TiledInferenceCancelled:
            result["cancelled"] = True
            break
        finally:
            del detector
    return result
