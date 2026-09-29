"""Bounded, offline CPU fine-tuning of a detector's prediction head.

Only frozen training images are opened. Validation and test examples remain
reserved for a separate quality evaluation; training loss is not an accuracy metric.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from PIL import Image

from iris.media import _pixel_hash
from iris.models import (
    IRIS_NATIVE_TO_COCO,
    TRAINING_ARCHITECTURE,
    TorchvisionDetector,
    catalog,
)
from iris.store import Store, new_id, now

MAX_STEPS = 200
CLASS_MAPPING = {"person": 1, "car": 2}


def _manifest(store: Store, dataset_id: str) -> dict:
    # Keep importing the optional dataset/training surfaces independent at startup.
    from iris.datasets import load_manifest

    manifest = load_manifest(store, dataset_id, verify_images=False)
    taxonomy = manifest.get("taxonomy_id", manifest.get("taxonomy"))
    if isinstance(taxonomy, dict):
        taxonomy = taxonomy.get("id")
    if taxonomy != "iris-objects-v1" or manifest.get("class_mapping") != CLASS_MAPPING:
        raise ValueError("Training requires the iris-objects-v1 person/car dataset mapping")
    return manifest


def _ready_parent(store: Store, parent_model_id: str) -> dict:
    parent = next((item for item in catalog(store.root) if item["id"] == parent_model_id), None)
    if parent is None or parent.get("architecture") != TRAINING_ARCHITECTURE:
        raise ValueError(
            "Choose the Faster R-CNN MobileNetV3-Large 320 FPN detector or its trained descendants"
        )
    if parent["status"] != "ready":
        raise RuntimeError(parent.get("reason") or "Parent checkpoint is unavailable")
    return parent


def _check_holdouts(manifest: dict, parent: dict) -> None:
    history = parent.get("provenance", {})
    previous_groups = set(history.get("training_scene_groups", []))
    previous_hashes = set(history.get("training_frame_hashes", []))
    heldout = [frame for frame in manifest["frames"] if frame["split"] != "train"]
    if any(
        frame["scene_group"] in previous_groups or frame["sha256"] in previous_hashes
        for frame in heldout
    ):
        raise ValueError(
            "A held-out group or image was already used to train this parent checkpoint"
        )


def create_training(
    store: Store,
    jobs,
    *,
    name: str,
    dataset_id: str,
    parent_model_id: str,
    steps: int = 20,
    learning_rate: float = 0.001,
    seed: int = 0,
) -> dict:
    name = name.strip()
    if not 1 <= len(name) <= 160:
        raise ValueError("Training name must contain between 1 and 160 characters")
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= MAX_STEPS:
        raise ValueError("Choose between 1 and 200 training steps")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2147483647:
        raise ValueError("Seed must be an integer between 0 and 2147483647")
    if (
        isinstance(learning_rate, bool)
        or not math.isfinite(learning_rate)
        or not 0 < learning_rate <= 0.1
    ):
        raise ValueError("Learning rate must be finite, greater than zero and at most 0.1")
    dataset = store.get("dataset_versions", dataset_id)
    if dataset is None:
        raise ValueError("Dataset version not found")
    manifest = _manifest(store, dataset_id)
    training_frames = [frame for frame in manifest["frames"] if frame["split"] == "train"]
    if not training_frames:
        raise ValueError("The dataset needs at least one training image")
    if not any(frame["boxes"] for frame in training_frames):
        raise ValueError("Training needs at least one positive person or car annotation")
    parent = _ready_parent(store, parent_model_id)
    _check_holdouts(manifest, parent)
    config = {
        "steps": steps,
        "learning_rate": float(learning_rate),
        "seed": seed,
        "device": "cpu",
        "scope": "prediction_head_only",
        "batch_size": 1,
        "optimizer": "SGD",
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "dataset_manifest_sha256": dataset["manifest_sha256"],
        "parent_weight_sha256": parent["weight_sha256"],
        "taxonomy_id": "iris-objects-v1",
        "class_mapping": CLASS_MAPPING,
        "quality_metrics": "Not computed; training loss does not measure detection quality",
    }
    training_id, job_id, created_at = new_id(), new_id(), now()
    with jobs.guard, store.connect() as connection:
        connection.execute(
            "INSERT INTO jobs (id,kind,status,params,message,created_at) VALUES (?,?,?,?,?,?)",
            (
                job_id,
                "train",
                "queued",
                json.dumps({"training_id": training_id}),
                "Waiting for local CPU fine-tuning",
                created_at,
            ),
        )
        connection.execute(
            "INSERT INTO training_runs "
            "(id,name,dataset_id,parent_model_id,config,job_id,created_at) VALUES (?,?,?,?,?,?,?)",
            (
                training_id,
                name,
                dataset_id,
                parent_model_id,
                json.dumps(config),
                job_id,
                created_at,
            ),
        )
    return training_detail(store, training_id)


def training_detail(store: Store, training_id: str) -> dict:
    training = store.get("training_runs", training_id)
    if training is None:
        raise KeyError(training_id)
    checkpoint = (
        store.get("trained_models", training["checkpoint_id"])
        if training["checkpoint_id"]
        else None
    )
    if checkpoint:
        checkpoint = {key: value for key, value in checkpoint.items() if key != "path"}
    return {**training, "job": store.get("jobs", training["job_id"]), "checkpoint": checkpoint}


def _read_training_image(store: Store, frame: dict) -> Image.Image:
    path = store.artifact_path(frame["image_path"])
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if frame.get("image_file_sha256") and digest != frame["image_file_sha256"]:
        raise ValueError("Frozen training image file no longer matches its manifest hash")
    with Image.open(path) as source:
        image = source.convert("RGB")
        image.load()
    if image.size != (frame["width"], frame["height"]) or _pixel_hash(image) != frame["sha256"]:
        image.close()
        raise ValueError("Frozen training image no longer matches its manifest hash")
    return image


class _HeadTrainer:
    def __init__(self, root: Path, parent_id: str, config: dict):
        import torch
        import torchvision

        self.torch = torch
        torch.manual_seed(config["seed"])
        torch.use_deterministic_algorithms(True)
        self.detector = TorchvisionDetector(root, parent_id, device="cpu")
        if self.detector.metadata["weight_sha256"] != config["parent_weight_sha256"]:
            raise ValueError("Parent checkpoint changed since training was queued")
        self.model = self.detector.model
        if self.detector.spec.get("origin") != "trained":
            previous = self.model.roi_heads.box_predictor
            predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
                previous.cls_score.in_features, 3
            )
            # Preserve the parent's learned background/person/car initialization.
            classes = [0, 1, 3]
            box_rows = [
                category * 4 + coordinate for category in classes for coordinate in range(4)
            ]
            with torch.no_grad():
                predictor.cls_score.weight.copy_(previous.cls_score.weight[classes])
                predictor.cls_score.bias.copy_(previous.cls_score.bias[classes])
                predictor.bbox_pred.weight.copy_(previous.bbox_pred.weight[box_rows])
                predictor.bbox_pred.bias.copy_(previous.bbox_pred.bias[box_rows])
            self.model.roi_heads.box_predictor = predictor
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.parameters = list(self.model.roi_heads.box_predictor.parameters())
        for parameter in self.parameters:
            parameter.requires_grad_(True)
        self.initial = [parameter.detach().clone() for parameter in self.parameters]
        self.model.train()
        self.model.backbone.eval()
        self.optimizer = torch.optim.SGD(
            self.parameters,
            lr=config["learning_rate"],
            momentum=config["momentum"],
            weight_decay=config["weight_decay"],
        )
        self.metadata = {
            **self.detector.metadata,
            "trainable_parameters": sum(parameter.numel() for parameter in self.parameters),
            "total_parameters": sum(parameter.numel() for parameter in self.model.parameters()),
            "trainable_modules": ["roi_heads.box_predictor"],
            "deterministic_algorithms": True,
            "head_initialization": (
                "Copied parent background/person/car classifier and box regression rows"
            ),
            "head_class_slots": 3,
            "native_to_coco": IRIS_NATIVE_TO_COCO,
            "validation_consumed": False,
            "test_consumed": False,
        }

    def step(self, image: Image.Image, boxes: list[dict]) -> dict:
        torch = self.torch
        tensor = self.detector.functional.pil_to_tensor(image).to(dtype=torch.float32) / 255
        coordinates = torch.tensor([box["box"] for box in boxes], dtype=torch.float32).reshape(
            -1, 4
        )
        labels = torch.tensor([CLASS_MAPPING[box["label"]] for box in boxes], dtype=torch.int64)
        self.optimizer.zero_grad(set_to_none=True)
        losses = self.model([tensor], [{"boxes": coordinates, "labels": labels}])
        loss = sum(losses.values())
        if not torch.isfinite(loss).item():
            raise ValueError("Nonfinite training loss; no checkpoint was published")
        loss.backward()
        if any(
            parameter.grad is None or not torch.isfinite(parameter.grad).all().item()
            for parameter in self.parameters
        ):
            raise ValueError("Missing or nonfinite prediction-head gradients")
        self.optimizer.step()
        if any(not torch.isfinite(parameter).all().item() for parameter in self.parameters):
            raise ValueError("Nonfinite prediction-head weights; no checkpoint was published")
        return {
            "loss": float(loss.detach()),
            "losses": {name: float(value.detach()) for name, value in losses.items()},
        }

    def write_checkpoint(self, path: Path) -> dict:
        changed = any(
            not self.torch.equal(before, parameter.detach())
            for before, parameter in zip(self.initial, self.parameters, strict=True)
        )
        if not changed:
            raise ValueError("No prediction-head weights changed; no checkpoint was published")
        self.torch.save(self.model.eval().state_dict(), path)
        return {"head_weights_changed": True}


def run_training(
    store: Store,
    training_id: str,
    progress: Callable[[float, str], None],
    cancelled: Callable[[], bool],
    trainer_factory=None,
) -> dict:
    training = store.get("training_runs", training_id)
    if training is None:
        raise ValueError("Training run not found")
    if training["history"] or training["checkpoint_id"]:
        raise ValueError("A training run is immutable; create a new run to retry")
    result = {
        "training_id": training_id,
        "steps_completed": 0,
        "checkpoint_id": None,
        "cancelled": False,
    }
    if cancelled():
        return {**result, "cancelled": True}
    config = training["config"]
    dataset = store.get("dataset_versions", training["dataset_id"])
    if dataset is None or dataset["manifest_sha256"] != config["dataset_manifest_sha256"]:
        raise ValueError("Dataset changed since training was queued")
    manifest = _manifest(store, training["dataset_id"])
    parent = _ready_parent(store, training["parent_model_id"])
    if parent["weight_sha256"] != config["parent_weight_sha256"]:
        raise ValueError("Parent checkpoint changed since training was queued")
    _check_holdouts(manifest, parent)
    frames = [frame for frame in manifest["frames"] if frame["split"] == "train"]
    progress(0, "Loading the local parent checkpoint; freezing backbone and proposal network")
    trainer = (trainer_factory or _HeadTrainer)(store.root, training["parent_model_id"], config)
    metadata = {
        **trainer.metadata,
        "dataset_id": dataset["id"],
        "dataset_manifest_sha256": dataset["manifest_sha256"],
        "parent_model_id": training["parent_model_id"],
        "parent_weight_sha256": config["parent_weight_sha256"],
        "training_scene_groups": sorted(
            set(parent.get("provenance", {}).get("training_scene_groups", []))
            | {frame["scene_group"] for frame in frames}
        ),
        "training_frame_hashes": sorted(
            set(parent.get("provenance", {}).get("training_frame_hashes", []))
            | {frame["sha256"] for frame in frames}
        ),
        "config": config,
        "quality_metrics": None,
    }
    store.update("training_runs", training_id, {"metadata": metadata})
    randomizer, order, history = random.Random(config["seed"]), [], []
    started = time.perf_counter()
    for step in range(1, config["steps"] + 1):
        if cancelled():
            return {**result, "cancelled": True}
        if not order:
            order = list(range(len(frames)))
            randomizer.shuffle(order)
        frame = frames[order.pop()]
        with _read_training_image(store, frame) as image:
            measurement = trainer.step(image, frame["boxes"])
        if not math.isfinite(measurement["loss"]):
            raise ValueError("Nonfinite training loss")
        history.append(
            {
                "step": step,
                "frame_id": frame["frame_id"],
                **measurement,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        store.update("training_runs", training_id, {"history": history})
        result["steps_completed"] = step
        progress(
            step / config["steps"],
            f"CPU head fine-tuning: step {step}/{config['steps']}, "
            f"training loss {measurement['loss']:.4f}",
        )
    if cancelled():
        return {**result, "cancelled": True}
    directory = store.artifact_path("models/trained")
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_id = "trained_" + new_id()
    destination = directory / (checkpoint_id + ".pth")
    with tempfile.NamedTemporaryFile(dir=directory, suffix=".part", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    published = False
    try:
        metadata.update(trainer.write_checkpoint(temporary_path))
        with temporary_path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        with temporary_path.open("rb") as source:
            os.fsync(source.fileno())
        if cancelled():
            return {**result, "cancelled": True}
        temporary_path.replace(destination)
        # Registration and completed-run linkage are one transaction. A cancelled
        # job cannot expose a completed model; unfinished files stay unregistered.
        with store.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            job = connection.execute(
                "SELECT status,cancel_requested FROM jobs WHERE id=?", (training["job_id"],)
            ).fetchone()
            if job["cancel_requested"] or job["status"] not in {"queued", "running"}:
                return {**result, "cancelled": True}
            connection.execute(
                "INSERT INTO trained_models "
                "(id,name,training_id,parent_model_id,architecture,path,weight_sha256,"
                "metadata,created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    checkpoint_id,
                    training["name"],
                    training_id,
                    training["parent_model_id"],
                    TRAINING_ARCHITECTURE,
                    str(destination.relative_to(store.root)),
                    digest,
                    json.dumps(metadata),
                    now(),
                ),
            )
            connection.execute(
                "UPDATE training_runs SET checkpoint_id=?,metadata=? WHERE id=?",
                (checkpoint_id, json.dumps(metadata), training_id),
            )
            result["checkpoint_id"] = checkpoint_id
            connection.execute(
                "UPDATE jobs SET status='succeeded',result=?,progress=1,finished_at=?,message=? "
                "WHERE id=?",
                (
                    json.dumps(result),
                    now(),
                    "CPU fine-tuning complete; checkpoint available for comparison",
                    training["job_id"],
                ),
            )
        published = True
        return result
    finally:
        temporary_path.unlink(missing_ok=True)
        if not published:
            destination.unlink(missing_ok=True)
