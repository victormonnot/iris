"""Frozen held-out evaluations and explicit, auditable model reference decisions."""

import json
import math
import time
from collections.abc import Callable

from iris.datasets import MAX_FRAMES, load_manifest
from iris.inference import PROTOCOL as TIMING_PROTOCOL
from iris.inference import _validate_prediction
from iris.metrics import evaluate_predictions, get_protocol
from iris.models import TorchvisionDetector, catalog
from iris.store import Store, _decode, new_id, now
from iris.training import _read_training_image


class ReferenceConflict(RuntimeError):
    """Another reviewer changed the reference after the selection view was loaded."""


PRETRAINING_WARNING = (
    "Official COCO pretraining data cannot be checked against these images. Local training "
    "overlap checks do not prove independence from pretraining or related scenes."
)


def _heldout_frames(store: Store, dataset_id: str, split: str) -> tuple[dict, list[dict]]:
    dataset = store.get("dataset_versions", dataset_id)
    if dataset is None:
        raise ValueError("Dataset version not found")
    manifest = load_manifest(store, dataset_id)
    frames = [frame for frame in manifest["frames"] if frame["split"] == split]
    if not 1 <= len(frames) <= MAX_FRAMES:
        raise ValueError(f"Choose a dataset with 1 to {MAX_FRAMES} images in its {split} split")
    return dataset, frames


def _model_lineage(model: dict, available: dict, frames: list[dict]) -> list[dict]:
    """Require known ancestry and check both groups and pixels at every local ancestor."""
    lineage, visited = [], set()
    current = model
    while True:
        identifier = current["id"]
        if identifier in visited:
            raise ValueError("Checkpoint ancestry contains a cycle")
        visited.add(identifier)
        classes = {item["id"]: item["name"] for item in current.get("classes", [])}
        if classes.get(1) != "person" or classes.get(3) != "car":
            raise ValueError("Checkpoint taxonomy must expose COCO person=1 and car=3")
        origin = current.get("origin")
        if origin == "official":
            lineage.append({"model_id": identifier, "origin": origin})
            break
        if origin != "trained" or current.get("taxonomy_id") != "iris-objects-v1":
            raise ValueError("Checkpoint has unknown training provenance or taxonomy")
        provenance = current.get("provenance", {})
        groups = provenance.get("training_scene_groups")
        hashes = provenance.get("training_frame_hashes")
        if any(
            not isinstance(values, list)
            or not values
            or any(not isinstance(value, str) or not value for value in values)
            for values in (groups, hashes)
        ):
            raise ValueError("Checkpoint training provenance is incomplete")
        if any(frame["scene_group"] in groups or frame["sha256"] in hashes for frame in frames):
            raise ValueError(
                "Evaluation group or image was used to train this checkpoint or ancestor"
            )
        lineage.append(
            {
                "model_id": identifier,
                "origin": origin,
                "training_scene_groups": groups,
                "training_frame_hashes": hashes,
                "parent_model_id": current.get("parent_model_id"),
                "parent_weight_sha256": provenance.get("parent_weight_sha256"),
            }
        )
        parent_id = current.get("parent_model_id")
        if parent_id not in available:
            raise ValueError("Checkpoint has an unknown ancestor; held-out independence is unknown")
        current = available[parent_id]
        parent_hash = current.get("weight_sha256")
        recorded_hash = provenance.get("parent_weight_sha256")
        if not isinstance(recorded_hash, str) or len(recorded_hash) != 64:
            raise ValueError("Checkpoint ancestry has no verified parent digest")
        if parent_hash is not None and parent_hash != recorded_hash:
            raise ValueError("Checkpoint ancestry does not match its recorded parent digest")
    return lineage


def _ready_models(store: Store, model_ids: list[str], frames: list[dict]) -> tuple[dict, dict]:
    available = {model["id"]: model for model in catalog(store.root)}
    selected, lineages = {}, {}
    for identifier in model_ids:
        model = available.get(identifier)
        if model is None:
            raise ValueError(f"Unknown detector: {identifier}")
        if model.get("status") != "ready":
            raise RuntimeError(model.get("reason") or f"Model {identifier} is not ready")
        digest = model.get("weight_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("Checkpoint has no verified SHA-256 digest")
        lineages[identifier] = _model_lineage(model, available, frames)
        selected[identifier] = model
    return selected, lineages


def _complete_models(store: Store, evaluation: dict) -> list[dict]:
    job = store.get("jobs", evaluation["job_id"])
    rows = store.list("evaluation_models", evaluation_id=evaluation["id"])
    if (
        job is None
        or job["status"] != "succeeded"
        or len(rows) != len(evaluation["model_ids"])
        or {row["model_id"] for row in rows} != set(evaluation["model_ids"])
        or any(row["metrics"] is None for row in rows)
    ):
        raise ValueError("Evaluation must finish successfully for every model first")
    return rows


def _validation_audit(
    store: Store, validation_id: str | None, dataset_id: str, model_ids: list[str], config: dict
) -> None:
    previous = store.get("evaluations", validation_id) if validation_id else None
    if previous is None or previous["split"] != "val":
        raise ValueError("A test audit requires a completed validation evaluation")
    _complete_models(store, previous)
    if previous["dataset_id"] != dataset_id or previous["model_ids"] != model_ids:
        raise ValueError("A test audit must use the same dataset and models as validation")
    for key in (
        "dataset_manifest_sha256",
        "model_hashes",
        "confidence_threshold",
        "iou_threshold",
        "device",
        "protocol",
        "model_lineages",
    ):
        if previous["config"].get(key) != config.get(key):
            raise ValueError(
                "A test audit must retain validation checkpoints and evaluation settings"
            )


def create_evaluation(
    store: Store,
    jobs,
    *,
    name: str,
    dataset_id: str,
    split: str = "val",
    model_ids: list[str],
    confidence_threshold: float = 0.5,
    iou_threshold: float = 0.5,
    device: str = "cpu",
    validation_evaluation_id: str | None = None,
) -> dict:
    if not isinstance(name, str) or not 1 <= len(name.strip()) <= 160:
        raise ValueError("Evaluation name must contain between 1 and 160 characters")
    if split not in {"val", "test"}:
        raise ValueError("Evaluate a held-out val or test split")
    if (
        not isinstance(model_ids, list)
        or not 1 <= len(model_ids) <= 2
        or any(not isinstance(identifier, str) or not identifier for identifier in model_ids)
        or len(set(model_ids)) != len(model_ids)
    ):
        raise ValueError("Choose one or two distinct models")
    if device not in {"cpu", "cuda"}:
        raise ValueError("Device must be cpu or cuda")
    for threshold in (confidence_threshold, iou_threshold):
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold)
            or not 0 <= threshold <= 1
        ):
            raise ValueError("Confidence and IoU thresholds must be finite values between 0 and 1")
    if iou_threshold == 0:
        raise ValueError("IoU threshold must be greater than zero")
    if split == "val" and validation_evaluation_id is not None:
        raise ValueError("Validation evaluations cannot be linked as test audits")
    dataset, frames = _heldout_frames(store, dataset_id, split)
    selected, lineages = _ready_models(store, model_ids, frames)
    config = {
        "dataset_name": dataset["name"],
        "dataset_manifest_sha256": dataset["manifest_sha256"],
        "frame_ids": [frame["frame_id"] for frame in frames],
        "frame_hashes": {frame["frame_id"]: frame["sha256"] for frame in frames},
        "model_hashes": {
            identifier: selected[identifier]["weight_sha256"] for identifier in model_ids
        },
        "model_names": {identifier: selected[identifier]["name"] for identifier in model_ids},
        "model_lineages": lineages,
        "confidence_threshold": float(confidence_threshold),
        "iou_threshold": float(iou_threshold),
        "device": device,
        "warmup": 1,
        "taxonomy_id": "iris-objects-v1",
        "class_mapping": {"person": 1, "car": 3},
        "protocol": get_protocol(
            confidence_threshold=confidence_threshold, iou_threshold=iou_threshold
        ),
        "timing_protocol": {
            key: value for key, value in TIMING_PROTOCOL.items() if key != "quality_metrics"
        },
        "warnings": list(
            dict.fromkeys([*dataset["summary"].get("warnings", []), PRETRAINING_WARNING])
        ),
        "validation_evaluation_id": validation_evaluation_id,
    }
    if split == "test":
        _validation_audit(store, validation_evaluation_id, dataset_id, model_ids, config)
        config["warnings"].append(
            "This test audit retains validation settings. Repeated inspection of test results "
            "can bias later model choices; reference selection uses validation only."
        )
    identifier, job_id, created_at = new_id(), new_id(), now()
    with jobs.guard, store.connect() as connection:
        connection.execute(
            "INSERT INTO jobs (id,kind,status,params,message,created_at) VALUES (?,?,?,?,?,?)",
            (
                job_id,
                "evaluate",
                "queued",
                json.dumps({"evaluation_id": identifier}),
                "Waiting to evaluate frozen held-out images",
                created_at,
            ),
        )
        connection.execute(
            "INSERT INTO evaluations (id,name,dataset_id,split,model_ids,config,job_id,created_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (
                identifier,
                name.strip(),
                dataset_id,
                split,
                json.dumps(model_ids),
                json.dumps(config, allow_nan=False),
                job_id,
                created_at,
            ),
        )
    return evaluation_summary(store, store.get("evaluations", identifier))


def evaluation_summary(store: Store, row: dict) -> dict:
    return {**row, "job": store.get("jobs", row["job_id"])}


def evaluation_detail(store: Store, evaluation_id: str) -> dict:
    row = store.get("evaluations", evaluation_id)
    if row is None:
        raise KeyError(evaluation_id)
    dataset, frames = _heldout_frames(store, row["dataset_id"], row["split"])
    if dataset["manifest_sha256"] != row["config"]["dataset_manifest_sha256"]:
        raise ValueError("Dataset changed since evaluation was queued")
    return {
        **evaluation_summary(store, row),
        "frames": [
            {
                **{key: value for key, value in frame.items() if key != "image_path"},
                "id": frame["frame_id"],
                "image_url": f"/api/datasets/{row['dataset_id']}/frames/{frame['frame_id']}/image",
            }
            for frame in frames
        ],
        "models": store.list("evaluation_models", evaluation_id=evaluation_id),
        "predictions": store.list("evaluation_predictions", evaluation_id=evaluation_id),
    }


def _publish_metrics(
    store: Store, evaluation: dict, model_row: dict, metrics: dict, result: dict, *, final: bool
) -> bool:
    """Make a completed score visible together with terminal success on the final model."""
    with store.connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        job = connection.execute(
            "SELECT status,cancel_requested FROM jobs WHERE id=?", (evaluation["job_id"],)
        ).fetchone()
        if job["cancel_requested"] or job["status"] not in {"queued", "running"}:
            return False
        connection.execute(
            "UPDATE evaluation_models SET metrics=? WHERE id=?",
            (json.dumps(metrics, allow_nan=False), model_row["id"]),
        )
        if final:
            connection.execute(
                "UPDATE jobs SET status='succeeded',result=?,progress=1,finished_at=?,message=? "
                "WHERE id=?",
                (json.dumps(result), now(), "Held-out evaluation complete", evaluation["job_id"]),
            )
    return True


def run_evaluation(
    store: Store,
    evaluation_id: str,
    progress: Callable[[float, str], None],
    cancelled: Callable[[], bool],
    detector_factory=None,
) -> dict:
    row = store.get("evaluations", evaluation_id)
    if row is None:
        raise ValueError("Evaluation not found")
    job = store.get("jobs", row["job_id"])
    if store.list("evaluation_models", evaluation_id=evaluation_id) or job["status"] not in {
        "queued",
        "running",
    }:
        raise ValueError("An evaluation is immutable; create a new evaluation to retry")
    config, model_ids = row["config"], row["model_ids"]
    result = {
        "evaluation_id": evaluation_id,
        "frames_total": len(config["frame_ids"]),
        "models_total": len(model_ids),
        "models_completed": 0,
        "predictions_created": 0,
        "cancelled": False,
    }
    if cancelled():
        return {**result, "cancelled": True}
    dataset, frames = _heldout_frames(store, row["dataset_id"], row["split"])
    if (
        dataset["manifest_sha256"] != config["dataset_manifest_sha256"]
        or [frame["frame_id"] for frame in frames] != config["frame_ids"]
        or {frame["frame_id"]: frame["sha256"] for frame in frames} != config["frame_hashes"]
    ):
        raise ValueError("Dataset changed since evaluation was queued")
    selected, lineages = _ready_models(store, model_ids, frames)
    if {identifier: selected[identifier]["weight_sha256"] for identifier in model_ids} != config[
        "model_hashes"
    ] or lineages != config["model_lineages"]:
        raise ValueError("Checkpoint or training provenance changed since evaluation was queued")
    if (
        get_protocol(
            confidence_threshold=config["confidence_threshold"],
            iou_threshold=config["iou_threshold"],
        )
        != config["protocol"]
    ):
        raise ValueError("Evaluation protocol changed since this evaluation was queued")
    if row["split"] == "test":
        _validation_audit(
            store, config["validation_evaluation_id"], row["dataset_id"], model_ids, config
        )
    total = len(frames) * len(model_ids)
    for model_index, model_id in enumerate(model_ids):
        if cancelled():
            return {**result, "cancelled": True}
        progress(result["predictions_created"] / total, f"Loading {selected[model_id]['name']}")
        detector = (detector_factory or TorchvisionDetector)(
            store.root, model_id, device=config["device"]
        )
        try:
            if detector.metadata.get("weight_sha256") != config["model_hashes"][model_id]:
                raise ValueError("Checkpoint changed since evaluation was queued")
            model_row = store.insert(
                "evaluation_models",
                {
                    "id": new_id(),
                    "evaluation_id": evaluation_id,
                    "model_id": model_id,
                    "metadata": {
                        **detector.metadata,
                        "model_name": config["model_names"][model_id],
                        "protocol": config["protocol"],
                        "lineage": lineages[model_id],
                    },
                    "metrics": None,
                    "created_at": now(),
                },
            )
            with _read_training_image(store, frames[0]) as image:
                if cancelled():
                    return {**result, "cancelled": True}
                progress(result["predictions_created"] / total, "Warming up model (not timed)")
                detector.warmup(image)
            predictions = []
            for frame in frames:
                if cancelled():
                    return {**result, "cancelled": True}
                started = time.perf_counter()
                with _read_training_image(store, frame) as image:
                    decoded = time.perf_counter()
                    prediction = detector.predict(image)
                elapsed = (time.perf_counter() - started) * 1000
                _validate_prediction(prediction, frame)
                prediction_row = store.insert(
                    "evaluation_predictions",
                    {
                        "id": new_id(),
                        "evaluation_id": evaluation_id,
                        "evaluation_model_id": model_row["id"],
                        "model_id": model_id,
                        "frame_id": frame["frame_id"],
                        "detections": prediction["detections"],
                        "input_size": prediction["input_size"],
                        "timing": {
                            **prediction["timing"],
                            "decode_ms": (decoded - started) * 1000,
                            "total_ms": elapsed,
                        },
                        "created_at": now(),
                    },
                )
                predictions.append(prediction_row)
                result["predictions_created"] += 1
                progress(
                    result["predictions_created"] / total,
                    f"Saved {result['predictions_created']} / {total} held-out predictions",
                )
            if cancelled():
                return {**result, "cancelled": True}
            metrics = evaluate_predictions(
                frames,
                predictions,
                confidence_threshold=config["confidence_threshold"],
                iou_threshold=config["iou_threshold"],
            )
            if cancelled():
                return {**result, "cancelled": True}
            completed = {**result, "models_completed": result["models_completed"] + 1}
            if not _publish_metrics(
                store, row, model_row, metrics, completed, final=model_index == len(model_ids) - 1
            ):
                return {**result, "cancelled": True}
            result = completed
        finally:
            del detector
    return result


def reference_history(store: Store) -> dict:
    history = list(reversed(store.list("model_references")))
    return {"current": history[0] if history else None, "history": history}


def promote_reference(
    store: Store,
    *,
    evaluation_id: str,
    model_id: str,
    reviewer: str,
    notes: str,
    expected_previous_id: str | None = None,
) -> dict:
    if not isinstance(reviewer, str) or not 1 <= len(reviewer.strip()) <= 120:
        raise ValueError("Record a reviewer name of 1 to 120 characters")
    if not isinstance(notes, str) or not 1 <= len(notes.strip()) <= 2000:
        raise ValueError("Record a selection reason of 1 to 2000 characters")
    row = store.get("evaluations", evaluation_id)
    if row is None or row["split"] != "val":
        raise ValueError("Reference selection requires a validation evaluation, never a test audit")
    models = _complete_models(store, row)
    selected = next((model for model in models if model["model_id"] == model_id), None)
    if selected is None:
        raise ValueError("Choose a model from this completed evaluation")
    dataset, frames = _heldout_frames(store, row["dataset_id"], "val")
    available, lineages = _ready_models(store, [model_id], frames)
    if (
        dataset["manifest_sha256"] != row["config"]["dataset_manifest_sha256"]
        or available[model_id]["weight_sha256"] != row["config"]["model_hashes"][model_id]
        or lineages[model_id] != row["config"]["model_lineages"][model_id]
    ):
        raise ValueError("Dataset, checkpoint or training provenance changed since evaluation")
    metadata = {
        "model_name": row["config"]["model_names"][model_id],
        "weight_sha256": row["config"]["model_hashes"][model_id],
        "dataset_id": row["dataset_id"],
        "dataset_name": row["config"]["dataset_name"],
        "dataset_manifest_sha256": row["config"]["dataset_manifest_sha256"],
        "split": "val",
        "metrics": selected["metrics"],
        "protocol": row["config"]["protocol"],
        "confidence_threshold": row["config"]["confidence_threshold"],
        "iou_threshold": row["config"]["iou_threshold"],
        "warnings": row["config"]["warnings"],
        "previous_reference_id": expected_previous_id,
    }
    identifier = new_id()
    with store.connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        previous = connection.execute(
            "SELECT id FROM model_references ORDER BY created_at DESC,id DESC LIMIT 1"
        ).fetchone()
        if (previous["id"] if previous else None) != expected_previous_id:
            raise ReferenceConflict(
                "Reference changed since this view was loaded; refresh before selecting"
            )
        connection.execute(
            "INSERT INTO model_references "
            "(id,evaluation_id,model_id,reviewer,notes,metadata,created_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (
                identifier,
                evaluation_id,
                model_id,
                reviewer.strip(),
                notes.strip(),
                json.dumps(metadata, allow_nan=False),
                now(),
            ),
        )
        result = _decode(
            connection.execute(
                "SELECT * FROM model_references WHERE id=?", (identifier,)
            ).fetchone()
        )
    return result
