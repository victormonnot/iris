"""Public, offline detector recording and strict replay into a fresh example store."""

import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path

from public_contract import FRAME_RECORD, project

MODEL_IDS = (
    "ssdlite320_mobilenet_v3_large",
    "fasterrcnn_mobilenet_v3_large_320_fpn",
)
MODEL_HASHES = {name: str for name in MODEL_IDS}
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
    "quality_metrics": str,
}
LANE = {"model_id": str, "variant": "full"}
CONFIG = {
    "device": "cpu",
    "warmup": 1,
    "taxonomy": "coco-2017-v1",
    "class_mapping": {"person": 1, "car": 3},
    "model_hashes": MODEL_HASHES,
    "protocol": PROTOCOL,
    "inference": {"mode": "full"},
    "lanes": [LANE],
    "work": {
        "frames_total": 2,
        "forward_passes": 4,
        "warmup_passes": 2,
        "total_forward_passes": 6,
        "tiles": [{"frame_id": str, "input_size": [int], "tile_count": 0}],
        "limits": {"max_tiles_per_frame": int, "max_forward_passes": int},
    },
}
COMPARISON = {
    "id": str,
    "session_id": str,
    "name": str,
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
    "progress": float,
    "message": "Comparison complete",
    "error": None,
    "created_at": str,
    "started_at": str,
    "finished_at": str,
    "cancel_requested": False,
}
FILTERING = {
    "score_threshold": float,
    "nms_iou_threshold": float,
    "max_detections_per_image": int,
    "ssdlite_topk_candidates_per_class": (int, None),
}
RUN_METADATA = {
    "model_id": str,
    "architecture": str,
    "weights_name": str,
    "weight_sha256": str,
    "weight_source": str,
    "license_url": str,
    "torch_version": str,
    "torchvision_version": str,
    "device": "cpu",
    "threads": int,
    "interop_threads": int,
    "precision": "float32",
    "head_class_slots": 91,
    "input_transform": {
        "color": "RGB",
        "tensor_range": [int],
        "exif_transpose": True,
        "image_mean": [float],
        "image_std": [float],
        "min_size": [int],
        "max_size": int,
        "fixed_size": ([int], None),
        "size_divisible": int,
    },
    "native_filtering": (
        {
            **FILTERING,
            "rpn": {
                "score_threshold": float,
                "nms_iou_threshold": float,
                "pre_nms_top_n": int,
                "post_nms_top_n": int,
            },
        },
        FILTERING,
    ),
    "timing_protocol": {
        "preprocess_ms": str,
        "inference_ms": str,
        "postprocess_ms": str,
        "total_ms": str,
        "synchronization": str,
        "warmup": str,
        "batch_size": 1,
    },
    "coordinates": str,
    "protocol": PROTOCOL,
    "inference": {"variant": "full"},
}
RUN = {
    "id": str,
    "comparison_id": str,
    "model_id": str,
    "variant": "full",
    "metadata": RUN_METADATA,
    "created_at": str,
}
DETECTION = {"box": [float], "label_id": int, "label": str, "score": float}
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
    "metadata": {},
    "created_at": str,
}
MODEL_IDENTITY = {
    "id": str,
    "name": str,
    "architecture": str,
    "weight_filename": str,
    "expected_hash_prefix": str,
    "download_bytes": int,
    "weights_name": str,
    "weight_url": str,
    "license_url": str,
    "code_license_url": str,
    "classes": [{"id": int, "name": str}],
    "task": "object_detection",
    "inference": True,
    "training": bool,
    "training_scopes": [
        {
            "id": str,
            "label": str,
            "description": str,
            "trainable_modules": [str],
        }
    ],
    "training_summary": str,
    "origin": "official",
    "weight_sha256": str,
}
# These are recorded identities. Runtime availability belongs to the executing adapter.
MODEL = MODEL_IDENTITY
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
    "width": int,
    "height": int,
}
PROVENANCE = {
    "started_at": str,
    "completed_at": str,
    "iris_version": str,
    "iris_git_commit": str,
    "runtime_versions": {"torch": str, "torchvision": str, "pillow": str, "numpy": str},
    "entrypoint": "iris.inference.run_comparison",
    "inference_source_sha256": {
        name: str
        for name in (
            "inference.py",
            "models.py",
            "media.py",
            "store.py",
            "coco_import.py",
        )
    },
    "archive_sha256": str,
    "source_provenance_sha256": str,
    "network_used": False,
    "human_annotation_reviews": 0,
    "model_filtering": str,
    "timing_scope": str,
}
RECORDING = {
    "format": "iris-demo-recording-v1",
    "frames": [FRAME_BINDING],
    "comparison": COMPARISON,
    "job": JOB,
    "runs": [RUN],
    "predictions": [PREDICTION],
    "models": [MODEL_IDENTITY],
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
    """Validate the closed public format and all cross-record identities before writing."""
    from iris.inference import PROTOCOL as actual_protocol
    from iris.models import COCO_CATEGORIES, TIMING_PROTOCOL, get_spec

    data = project(value, RECORDING)
    _require(data == value, "Recording contains fields outside the public contract")
    frames, comparison, job = data["frames"], data["comparison"], data["job"]
    _require(
        len(frames) == 2 and {frame["coco_image_id"] for frame in frames} == {6723, 32941},
        "Recording must bind the two public street-scenes images",
    )
    _require(len({frame["frame_id"] for frame in frames}) == 2, "Duplicate recorded frame")
    _require(
        comparison["frame_ids"] == [frame["frame_id"] for frame in frames],
        "Comparison frame selection differs from its bindings",
    )
    _require(comparison["model_ids"] == list(MODEL_IDS), "Unexpected recorded detectors")
    _require(
        comparison["config"]["lanes"]
        == [{"model_id": model, "variant": "full"} for model in MODEL_IDS],
        "Comparison lanes differ from its detectors",
    )
    _require(comparison["config"]["protocol"] == actual_protocol, "Unexpected timing protocol")
    _require(
        comparison["job_id"] == job["id"]
        and job["params"]["comparison_id"] == job["result"]["comparison_id"] == comparison["id"]
        and job["progress"] == 1,
        "Recorded comparison job does not match its completed result",
    )
    frame_ids = {frame["frame_id"]: frame for frame in frames}
    _require(
        comparison["config"]["work"]["tiles"]
        == [
            {
                "frame_id": frame["frame_id"],
                "input_size": [frame["width"], frame["height"]],
                "tile_count": 0,
            }
            for frame in frames
        ],
        "Recorded work plan does not match its images",
    )
    _require(
        len(data["models"]) == 2 and {model["id"] for model in data["models"]} == set(MODEL_IDS),
        "Recorded model identities are incomplete",
    )
    models = {model["id"]: model for model in data["models"]}
    for model_id, model in models.items():
        spec = get_spec(model_id)
        _require(
            all(model[key] == spec[key] for key in MODEL_IDENTITY if key != "weight_sha256"),
            "Recorded model identity differs from the official catalogue",
        )
        digest = model["weight_sha256"]
        _require(
            _digest(digest)
            and digest.startswith(spec["expected_hash_prefix"])
            and digest == comparison["config"]["model_hashes"][model_id],
            "Recorded model hash does not match its comparison",
        )
    _require(
        len(data["runs"]) == 2 and {run["model_id"] for run in data["runs"]} == set(MODEL_IDS),
        "Recording must contain one run per detector",
    )
    runs = {run["id"]: run for run in data["runs"]}
    _require(len(runs) == 2, "Duplicate recorded run")
    for run in runs.values():
        model = models[run["model_id"]]
        meta = run["metadata"]
        _require(
            run["comparison_id"] == comparison["id"] and meta["model_id"] == model["id"],
            "Run belongs to another comparison or detector",
        )
        for key in ("architecture", "weights_name", "weight_sha256", "license_url"):
            _require(meta[key] == model[key], "Run does not match its model identity")
        _require(meta["weight_source"] == model["weight_url"], "Run weight source differs")
        _require(
            meta["protocol"] == actual_protocol and meta["timing_protocol"] == TIMING_PROTOCOL,
            "Run timing protocol differs",
        )
        _require(meta["threads"] > 0 and meta["interop_threads"] > 0, "Invalid CPU thread count")
        for name in ("torch", "torchvision"):
            _require(
                meta[f"{name}_version"] == data["provenance"]["runtime_versions"][name],
                "Run runtime differs from its provenance",
            )
    predictions = data["predictions"]
    _require(
        len(predictions) == 4 and len({row["id"] for row in predictions}) == 4,
        "Recording must contain four distinct prediction results",
    )
    pairs = set()
    for prediction in predictions:
        run = runs.get(prediction["run_id"])
        frame = frame_ids.get(prediction["frame_id"])
        _require(
            run is not None
            and frame is not None
            and prediction["comparison_id"] == comparison["id"]
            and prediction["model_id"] == run["model_id"],
            "Prediction links do not match",
        )
        _require(
            prediction["input_size"] == [frame["width"], frame["height"]],
            "Prediction dimensions differ from its frame",
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
        _require(
            all(value >= 0 for value in prediction["timing"].values()), "Negative recorded timing"
        )
    _require(len(pairs) == 4, "Duplicate or missing run/frame prediction")
    for frame in frames:
        _require(
            frame["width"] > 0
            and frame["height"] > 0
            and _digest(frame["source_sha256"])
            and _digest(frame["frame_pixel_sha256"]),
            "Invalid recorded frame binding",
        )
    provenance = data["provenance"]
    for key in ("archive_sha256", "source_provenance_sha256"):
        _require(_digest(provenance[key]), "Invalid provenance digest")
    _require(
        all(_digest(value) for value in provenance["inference_source_sha256"].values()),
        "Invalid inference source digest",
    )
    return data


def load_recording(path=None):
    """Read the checked-in real capture without importing Torch or running inference."""
    source = Path(path) if path is not None else Path(__file__).with_name("recording.json")
    return validate_recording(json.loads(source.read_text()))


def make_recording(detail, frames, models, provenance):
    """Project a genuine native run; discard logs, host platform and readiness claims."""
    for frame in frames:
        _require(
            detail["config"]["frame_hashes"][frame["frame_id"]] == frame["frame_pixel_sha256"],
            "Captured comparison hash differs from its source frame",
        )
    return validate_recording(
        project(
            {
                "format": "iris-demo-recording-v1",
                "frames": frames,
                "comparison": detail,
                "job": detail["job"],
                "runs": detail["runs"],
                "predictions": detail["predictions"],
                "models": models,
                "provenance": {**provenance, "entrypoint": "iris.inference.run_comparison"},
            },
            RECORDING,
        )
    )


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
