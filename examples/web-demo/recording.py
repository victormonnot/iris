"""Closed public projection of two saved FPV evaluations, replayed without compute."""

import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path

from public_contract import FRAME_RECORD, TAXONOMY, project

MODEL_IDS = ("yolox_nano", "trained_78f290f12cdb4b1e8acd36206acebf83")
EVALUATION_IDS = ("239f13cb694141829860f2534dd6c7b2", "4530628bf91a4be49b4af47905625bab")
SOURCE_FRAMES = {
    1601: "5bffdf25c0b54935bea052e6c23e8ce0",
    1681: "54a87f36399c4d80bbdea2ed35abc169",
}
MODEL_HASHES = {
    MODEL_IDS[0]: "cd28f55fbbc1829f99d9ac9b38a16d259a22889739c8728ea877610201feff7b",
    MODEL_IDS[1]: "316b50c4ece77fc463c59342455c5e6af28005163502af6cbdd9fd5f33a62978",
}
PROTOCOL = {
    "version": "torchvision-forward-v1",
    "batch_size": 1,
    "warmup_frames": 1,
    "warmup_in_timings": False,
    "decode_ms": str,
    "preprocess_ms": str,
    "inference_ms": str,
    "postprocess_ms": str,
    "total_ms": str,
}
LANE = {"model_id": str, "variant": "full"}
FROZEN_TAXONOMY = {
    **TAXONOMY,
    "version": int,
    "parent_id": str,
    "created_at": str,
}
CLASS_CONTRACT = (
    {"taxonomy_id": "coco-2017-v1"},
    {
        "taxonomy_id": str,
        "taxonomy": FROZEN_TAXONOMY,
        "class_mapping": {"person": 1},
        "output_class_mapping": {"person": 1},
    },
)
CONFIG = {
    "device": "cuda:0",
    "warmup": 1,
    "taxonomy": "coco-2017-v1",
    "class_mapping": {"person": 1},
    "model_hashes": MODEL_HASHES,
    "protocol": PROTOCOL,
    "inference": {"mode": "full"},
    "lanes": [LANE],
    "model_class_contracts": {model: CLASS_CONTRACT for model in MODEL_IDS},
    "source_evaluation_ids": [str],
    "result_origin": "saved-evaluation-projection",
}
COMPARISON = {
    "id": "fpv-yolox-comparison",
    "session_id": str,
    "name": "YOLOX-Nano · original vs trained in Iris",
    "frame_ids": [str],
    "model_ids": [str],
    "config": CONFIG,
    "job_id": str,
    "created_at": str,
}
JOB = {
    "id": str,
    "kind": "infer",
    "status": "succeeded",
    "params": {"comparison_id": str},
    "result": {
        "comparison_id": str,
        "frames_total": 2,
        "models_total": 2,
        "predictions_created": 4,
        "cancelled": False,
    },
    "progress": 1.0,
    "message": "Saved evaluation results",
    "error": None,
    "created_at": str,
    "started_at": str,
    "finished_at": str,
    "cancel_requested": False,
}
RUN_METADATA = {
    "model_id": str,
    "architecture": "yolox_nano",
    "weights_name": str,
    "weight_sha256": str,
    "weight_source": (str, None),
    "license_url": str,
    "source_commit": str,
    "torch_version": str,
    "torchvision_version": str,
    "device": "cuda:0",
    "hardware": "NVIDIA GeForce RTX 4060",
    "threads": int,
    "interop_threads": int,
    "precision": "float32",
    "head_class_slots": int,
    "background_class": False,
    "internal_class_index_base": 0,
    "native_class_index_base": 1,
    "input_transform": {
        "color": "BGR",
        "tensor_range": [int],
        "exif_transpose": True,
        "fixed_size": [int],
        "dtype": "float32",
        "layout": "NCHW",
        "letterbox": {"alignment": "top_left", "value": 114, "interpolation": "opencv_linear"},
    },
    "native_filtering": {
        "score_threshold": float,
        "nms_iou_threshold": float,
        "max_detections_per_image": 100,
        "score": str,
        "nms": "class_aware",
    },
    "timing_protocol": PROTOCOL,
    "coordinates": str,
    "inference": {"variant": "full"},
    "source_evaluation_id": str,
    "class_contract": CLASS_CONTRACT,
}
RUN = {
    "id": str,
    "comparison_id": str,
    "model_id": str,
    "variant": "full",
    "metadata": RUN_METADATA,
    "created_at": str,
}
DETECTION_BASE = {
    "box": [float],
    "label_id": int,
    "label": str,
    "score": float,
    "native_label_id": int,
    "taxonomy_id": str,
}
DETECTION = ({**DETECTION_BASE, "ignored": True}, DETECTION_BASE)
TIMING = {
    "preprocess_ms": float,
    "inference_ms": float,
    "postprocess_ms": float,
    "total_ms": float,
    "decode_ms": float,
    "forward_passes": 1,
    "tile_count": 0,
    "crop_ms": 0.0,
    "merge_ms": 0.0,
}
PREDICTION = {
    "id": str,
    "comparison_id": str,
    "run_id": str,
    "frame_id": str,
    "model_id": str,
    "detections": [DETECTION],
    "timing": TIMING,
    "input_size": [int],
    "metadata": {"source_evaluation_id": str},
    "created_at": str,
}
MODEL = {
    "id": str,
    "name": str,
    "architecture": "yolox_nano",
    "weights_name": str,
    "weight_sha256": str,
    "license_url": str,
    "code_license_url": str,
    "classes": [{"id": int, "name": str}],
    "task": "object_detection",
    "inference": True,
    "training": True,
    "training_summary": str,
    "origin": ("official", "trained"),
    "parent_model_id": (str, None),
}
MODEL_IDENTITY = MODEL
SUMMARY = {**COMPARISON, "job": JOB, "lanes": [{**LANE, "run_id": str}]}
DETAIL = {
    **SUMMARY,
    "frames": [{**FRAME_RECORD, "source_filename": str, "session_name": str, "scene_group": str}],
    "runs": [RUN],
    "predictions": [PREDICTION],
    "replay": {
        "version": "iris-comparison-replay-v1",
        "mode": "saved-samples",
        "continuous_inference": False,
        "timestamps_approximate": True,
        "media_check": "path_and_size_only",
        "sources": [],
        "still_frame_ids": [str],
    },
}
FRAME_BINDING = {
    "frame_id": str,
    "asset_id": str,
    "coco_image_id": int,
    "source_filename": str,
    "source_sha256": str,
    "frame_pixel_sha256": str,
    "width": 640,
    "height": 480,
}
PROVENANCE = {
    "result_origin": "saved-evaluation-projection",
    "source_evaluations": [{"id": str, "model_id": str, "created_at": str}],
    "source_dataset_id": str,
    "source_dataset_manifest_sha256": str,
    "source_frame_ids": [str],
    "source_prediction_ids": [str],
    "source_training_id": str,
    "training_steps": 800,
    "new_inference": False,
    "network_used": False,
    "assembly": str,
    "model_filtering": str,
    "timing_scope": str,
    "limitations": str,
    "archive_sha256": str,
    "source_provenance_sha256": str,
}
RECORDING = {
    "format": "iris-demo-evaluation-recording-v1",
    "frames": [FRAME_BINDING],
    "comparison": COMPARISON,
    "job": JOB,
    "runs": [RUN],
    "predictions": [PREDICTION],
    "models": [MODEL],
    "provenance": PROVENANCE,
}
COMPARISON_SUMMARY = SUMMARY
COMPARISON_DETAIL = DETAIL


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(value):
    return bool(re.fullmatch(r"[0-9a-f]{64}", value))


def validate_recording(value):
    """Validate public fields, source identities and geometry before any database write."""
    from iris.models import COCO_CATEGORIES

    data = project(value, RECORDING)
    _require(data == value, "Recording contains fields outside the public contract")
    frames, comparison, job = data["frames"], data["comparison"], data["job"]
    _require(
        len(frames) == 2 and {f["coco_image_id"]: f["frame_id"] for f in frames} == SOURCE_FRAMES,
        "Recording must bind the two public FPV frames",
    )
    _require(
        comparison["frame_ids"] == list(SOURCE_FRAMES.values()),
        "Comparison frame selection differs",
    )
    _require(comparison["model_ids"] == list(MODEL_IDS), "Unexpected recorded detectors")
    _require(
        comparison["config"]["source_evaluation_ids"] == list(EVALUATION_IDS),
        "Unexpected evaluations",
    )
    _require(
        comparison["config"]["lanes"] == [{"model_id": m, "variant": "full"} for m in MODEL_IDS],
        "Invalid lanes",
    )
    _require(
        comparison["job_id"] == job["id"]
        and job["params"]["comparison_id"] == job["result"]["comparison_id"] == comparison["id"],
        "Saved-result wrapper does not match its comparison",
    )
    _require(
        len(data["models"]) == 2 and [m["id"] for m in data["models"]] == list(MODEL_IDS),
        "Invalid model identities",
    )
    models = {m["id"]: m for m in data["models"]}
    for mid, model in models.items():
        _require(model["weight_sha256"] == MODEL_HASHES[mid], "Model checkpoint hash differs")
    _require(
        models[MODEL_IDS[0]]["origin"] == "official"
        and models[MODEL_IDS[0]]["parent_model_id"] is None,
        "Invalid original lineage",
    )
    _require(
        models[MODEL_IDS[1]]["origin"] == "trained"
        and models[MODEL_IDS[1]]["parent_model_id"] == MODEL_IDS[0],
        "Invalid trained lineage",
    )
    _require(
        len(data["runs"]) == 2 and [r["model_id"] for r in data["runs"]] == list(MODEL_IDS),
        "Invalid runs",
    )
    runs = {r["id"]: r for r in data["runs"]}
    _require(len(runs) == 2, "Duplicate recorded run")
    for run, eid in zip(data["runs"], EVALUATION_IDS, strict=True):
        meta, model = run["metadata"], models[run["model_id"]]
        _require(
            run["comparison_id"] == comparison["id"] and meta["model_id"] == model["id"],
            "Invalid run links",
        )
        for key in ("architecture", "weights_name", "weight_sha256", "license_url"):
            _require(meta[key] == model[key], "Run differs from model identity")
        _require(meta["source_evaluation_id"] == eid, "Run source evaluation differs")
        _require(
            meta["class_contract"] == comparison["config"]["model_class_contracts"][model["id"]],
            "Recorded class definitions differ between the run and comparison",
        )
        _require(
            meta["timing_protocol"] == comparison["config"]["protocol"], "Timing protocols differ"
        )
        _require(meta["threads"] > 0 and meta["interop_threads"] > 0, "Invalid thread count")
    predictions = data["predictions"]
    _require(
        len(predictions) == 4 and len({r["id"] for r in predictions}) == 4,
        "Expected four saved predictions",
    )
    frame_ids = {f["frame_id"]: f for f in frames}
    pairs = set()
    for prediction in predictions:
        run, frame = runs.get(prediction["run_id"]), frame_ids.get(prediction["frame_id"])
        _require(
            run is not None
            and frame is not None
            and prediction["comparison_id"] == comparison["id"]
            and prediction["model_id"] == run["model_id"],
            "Prediction links do not match",
        )
        _require(
            prediction["metadata"]["source_evaluation_id"]
            == run["metadata"]["source_evaluation_id"],
            "Prediction source differs",
        )
        _require(
            prediction["input_size"] == [frame["width"], frame["height"]],
            "Prediction dimensions differ",
        )
        pairs.add((prediction["run_id"], prediction["frame_id"]))
        _require(len(prediction["detections"]) <= 100, "Prediction exceeds native detector limit")
        for detection in prediction["detections"]:
            box, label = detection["box"], detection["label_id"]
            _require(
                len(box) == 4
                and 0 <= box[0] < box[2] <= frame["width"]
                and 0 <= box[1] < box[3] <= frame["height"]
                and 0 <= detection["score"] <= 1,
                "Detection falls outside its image or score range",
            )
            _require(
                0 < label < len(COCO_CATEGORIES)
                and COCO_CATEGORIES[label] != "N/A"
                and detection["label"] == COCO_CATEGORIES[label],
                "Unexpected detection category",
            )
        _require(all(v >= 0 for v in prediction["timing"].values()), "Negative recorded timing")
    _require(len(pairs) == 4, "Duplicate or missing run/frame prediction")
    for frame in frames:
        _require(
            _digest(frame["source_sha256"]) and _digest(frame["frame_pixel_sha256"]),
            "Invalid frame digest",
        )
    provenance = data["provenance"]
    _require(
        [r["id"] for r in provenance["source_evaluations"]] == list(EVALUATION_IDS),
        "Invalid source evaluations",
    )
    _require(
        provenance["source_frame_ids"] == list(SOURCE_FRAMES.values()), "Invalid source frames"
    )
    _require(
        provenance["source_prediction_ids"] == [p["id"] for p in predictions],
        "Invalid source prediction identities",
    )
    for key in ("archive_sha256", "source_provenance_sha256", "source_dataset_manifest_sha256"):
        _require(_digest(provenance[key]), "Invalid provenance digest")
    return data


def load_recording(path=None):
    """Read the immutable evaluation projection; no Torch import or inference."""
    source = Path(path) if path is not None else Path(__file__).with_name("recording.json")
    return validate_recording(json.loads(source.read_text()))


def restore_recording(store, recording, session_id, frames_by_coco):
    """Replay verified records into four tables in a fresh store; no model is executed."""
    from PIL import Image

    from iris.media import _pixel_hash
    from iris.store import JSON_FIELDS

    data = deepcopy(validate_recording(recording))
    _require(store.get("sessions", session_id) is not None, "Target session does not exist")
    for table in ("jobs", "comparisons", "runs", "predictions"):
        _require(not store.list(table), "Recording restore requires a fresh example store")
    remap = {data["comparison"]["session_id"]: session_id}
    for binding in data["frames"]:
        frame = frames_by_coco.get(binding["coco_image_id"])
        _require(
            frame is not None and frame["session_id"] == session_id,
            "Recording frame is missing from the target session",
        )
        _require(
            [frame["width"], frame["height"], frame["sha256"]]
            == [binding["width"], binding["height"], binding["frame_pixel_sha256"]],
            "Recorded frame dimensions or pixel hash differ from the imported image",
        )
        stored_frame = store.get("frames", frame["id"])
        _require(
            stored_frame is not None
            and all(
                stored_frame[key] == frame[key]
                for key in ("session_id", "asset_id", "width", "height", "sha256")
            ),
            "Target frame does not match its stored record",
        )
        with Image.open(store.artifact_path(stored_frame["path"])) as source:
            _require(
                _pixel_hash(source.convert("RGB")) == binding["frame_pixel_sha256"],
                "Imported frame pixels differ from the recording",
            )
        asset = store.get("assets", frame["asset_id"])
        _require(
            asset is not None and asset["sha256"] == binding["source_sha256"],
            "Recorded source hash differs from the imported image",
        )
        with store.artifact_path(asset["path"]).open("rb") as source:
            _require(
                hashlib.file_digest(source, "sha256").hexdigest() == binding["source_sha256"],
                "Imported source bytes differ from the recording",
            )
        remap[binding["frame_id"]] = frame["id"]
        remap[binding["asset_id"]] = asset["id"]

    def mapped(value):
        if isinstance(value, dict):
            return {key: mapped(child) for key, child in value.items()}
        if isinstance(value, list):
            return [mapped(child) for child in value]
        return remap.get(value, value) if isinstance(value, str) else value

    comparison = mapped(data["comparison"])
    comparison["config"]["frame_hashes"] = {
        remap[frame["frame_id"]]: frame["frame_pixel_sha256"] for frame in data["frames"]
    }
    rows = [("jobs", mapped(data["job"])), ("comparisons", comparison)]
    rows += [("runs", mapped(row)) for row in data["runs"]]
    rows += [("predictions", mapped(row)) for row in data["predictions"]]
    # All input is validated before one transaction inserts only these known tables.
    with store.connect() as connection:
        for table, row in rows:
            values = [
                json.dumps(value, allow_nan=False)
                if key in JSON_FIELDS and value is not None
                else value
                for key, value in row.items()
            ]
            connection.execute(
                f"INSERT INTO {table} ({','.join(row)}) VALUES ({','.join('?' for _ in row)})",
                values,
            )
    return comparison["id"], remap
