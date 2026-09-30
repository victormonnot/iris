"""Bounded, offline CPU fine-tuning with an explicit per-run training depth.

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
from copy import deepcopy
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
SCOPE_VERSION = 1
TRAINING_SCOPES = {
    "prediction_head_only": {
        "id": "prediction_head_only",
        "label": "Prediction head only",
        "description": "Adjust the person/car prediction head while keeping visual features fixed.",
        "trainable_modules": ["roi_heads.box_predictor"],
    },
    "partial_backbone": {
        "id": "partial_backbone",
        "label": "Last backbone stage and detector",
        "description": (
            "Adapt the final visual feature stage, feature pyramid, proposals and ROI heads."
        ),
        "trainable_modules": [
            "backbone.body.13",
            "backbone.body.14",
            "backbone.body.15",
            "backbone.body.16",
            "backbone.fpn",
            "rpn",
            "roi_heads",
        ],
    },
    "full_model": {
        "id": "full_model",
        "label": "All detector layers",
        "description": (
            "Adapt all visual features and detection layers, including early image features."
        ),
        "trainable_modules": ["backbone", "rpn", "roi_heads"],
    },
}


def training_scope(scope: str) -> dict:
    if not isinstance(scope, str) or scope not in TRAINING_SCOPES:
        raise ValueError("Choose prediction_head_only, partial_backbone or full_model")
    return deepcopy(TRAINING_SCOPES[scope])


def _scope_from_config(config: dict) -> dict:
    scope = training_scope(config.get("scope", "prediction_head_only"))
    if "scope_version" not in config:
        if scope["id"] != "prediction_head_only" or "trainable_modules" in config:
            raise ValueError("Unsupported legacy training scope contract; create a new run")
    elif (
        config.get("scope") != scope["id"]
        or type(config["scope_version"]) is not int
        or config["scope_version"] != SCOPE_VERSION
        or config.get("trainable_modules") != scope["trainable_modules"]
    ):
        raise ValueError("Training scope contract changed; create a new run")
    return scope


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


def _prepare_training(
    store: Store,
    *,
    name: str,
    dataset_id: str,
    parent_model_id: str,
    steps: int = 20,
    learning_rate: float = 0.001,
    seed: int = 0,
    scope: str = "prediction_head_only",
) -> tuple:
    selected_scope = training_scope(scope)
    if not isinstance(name, str):
        raise ValueError("Training name must contain between 1 and 160 characters")
    name = name.strip()
    if not 1 <= len(name) <= 160:
        raise ValueError("Training name must contain between 1 and 160 characters")
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= MAX_STEPS:
        raise ValueError("Choose between 1 and 200 training steps")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2147483647:
        raise ValueError("Seed must be an integer between 0 and 2147483647")
    if (
        type(learning_rate) not in (int, float)
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
        "scope": selected_scope["id"],
        "scope_version": SCOPE_VERSION,
        "trainable_modules": selected_scope["trainable_modules"],
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
    return name, config, dataset, parent, training_frames


def preview_training(
    store: Store,
    *,
    name: str,
    dataset_id: str,
    parent_model_id: str,
    steps: int = 20,
    learning_rate: float = 0.001,
    seed: int = 0,
    scope: str = "prediction_head_only",
) -> dict:
    """Validate a run and describe its bounded work without loading model parameters."""
    _, config, dataset, parent, frames = _prepare_training(
        store,
        name=name,
        dataset_id=dataset_id,
        parent_model_id=parent_model_id,
        steps=steps,
        learning_rate=learning_rate,
        seed=seed,
        scope=scope,
    )
    return {
        "config": config,
        "scope": training_scope(scope),
        "dataset": {
            "id": dataset["id"],
            "name": dataset["name"],
            "train_images": len(frames),
            "positive_train_images": sum(bool(frame["boxes"]) for frame in frames),
            "annotation_count": sum(len(frame["boxes"]) for frame in frames),
        },
        "workload": {
            "steps": steps,
            "batch_size": 1,
            "image_visits": steps,
            "unique_images_min": min(steps, len(frames)),
            "full_passes": steps // len(frames),
            "remainder_images": steps % len(frames),
            "device": "cpu",
        },
        "parent": {key: parent[key] for key in ("id", "name", "weight_sha256")},
        "notes": [
            "Only frozen training images are consumed; validation and test images stay reserved.",
            "Training loss does not measure detection quality; compare the resulting "
            "checkpoint on validation data.",
            "Frozen batch-normalization statistics remain unchanged for every training depth.",
            "Exact parameter counts are recorded after the worker loads the checkpoint.",
            *(
                [
                    "Deeper adaptation needs more CPU memory and computation; "
                    "use a small learning rate and a short first run."
                ]
                if scope != "prediction_head_only"
                else []
            ),
        ],
    }


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
    scope: str = "prediction_head_only",
) -> dict:
    name, config, _, _, _ = _prepare_training(
        store,
        name=name,
        dataset_id=dataset_id,
        parent_model_id=parent_model_id,
        steps=steps,
        learning_rate=learning_rate,
        seed=seed,
        scope=scope,
    )
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


def _matches_module(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(prefix + ".")


def _tensor_digest(tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


class _HeadTrainer:
    """Train the chosen scope; retain the original private class name for compatibility."""

    def __init__(self, root: Path, parent_id: str, config: dict):
        import torch
        import torchvision

        self.torch = torch
        self.scope = _scope_from_config(config)
        torch.manual_seed(config["seed"])
        torch.use_deterministic_algorithms(True)
        self.detector = TorchvisionDetector(root, parent_id, device="cpu")
        if self.detector.metadata["weight_sha256"] != config["parent_weight_sha256"]:
            raise ValueError("Parent checkpoint changed since training was queued")
        self.model = self.detector.model
        inference_proposal_threshold = self.model.rpn.score_thresh
        # The MobileNet320 inference preset drops RPN proposals below 0.05.
        # A negative image can then contain no sampled ROIs, making both ROI
        # losses undefined. Keep background proposals during optimization;
        # loading the saved weights through the inference adapter keeps its
        # original filtering settings, independent of this training-only change.
        self.model.rpn.score_thresh = 0.0
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
        modules = dict(self.model.named_modules())
        selected_modules = self.scope["trainable_modules"]
        if any(prefix not in modules for prefix in selected_modules):
            raise ValueError("Detector module layout does not match the selected training scope")
        if self.scope["id"] == "partial_backbone" and list(self.model.backbone.body._modules) != [
            str(index) for index in range(17)
        ]:
            raise ValueError("The partial scope requires the supported 17-block MobileNet backbone")
        if any(
            isinstance(module, torch.nn.modules.batchnorm._BatchNorm) for module in modules.values()
        ):
            raise ValueError("The training detector must retain frozen batch normalization")
        self.named_parameters = dict(self.model.named_parameters())
        self.selected_parameters = {}
        self.frozen_parameters = {}
        for name, parameter in self.named_parameters.items():
            selected = any(_matches_module(name, prefix) for prefix in selected_modules)
            parameter.requires_grad_(selected)
            if selected:
                self.selected_parameters[name] = parameter
            else:
                self.frozen_parameters[name] = parameter
        if any(
            not any(_matches_module(name, prefix) for name in self.selected_parameters)
            for prefix in selected_modules
        ) or (self.scope["id"] == "full_model" and self.frozen_parameters):
            raise ValueError("The selected scope contains missing or unsupported model parameters")
        self.parameters = list(self.selected_parameters.values())
        if any(not torch.isfinite(parameter).all().item() for parameter in self.parameters):
            raise ValueError("Parent checkpoint contains nonfinite trainable weights")
        self.initial = {
            name: parameter.detach().clone() for name, parameter in self.selected_parameters.items()
        }
        self.frozen_initial = {
            name: _tensor_digest(parameter) for name, parameter in self.frozen_parameters.items()
        }
        self.initial_buffers = {
            name: value.detach().clone() for name, value in self.model.named_buffers()
        }
        self.frozen_batchnorm_modules = [
            name
            for name, module in modules.items()
            if isinstance(module, torchvision.ops.misc.FrozenBatchNorm2d)
        ]
        self.gradient_modules = set()
        self.model.train()
        if self.scope["id"] == "prediction_head_only":
            self.model.backbone.eval()
        # FrozenBatchNorm forwards do not update statistics even in train mode;
        # keep the module mode explicit and verify all buffers before publication.
        for name in self.frozen_batchnorm_modules:
            modules[name].eval()
        self.optimizer = torch.optim.SGD(
            self.parameters,
            lr=config["learning_rate"],
            momentum=config["momentum"],
            weight_decay=config["weight_decay"],
        )
        self.metadata = {
            **self.detector.metadata,
            "trainable_parameters": sum(parameter.numel() for parameter in self.parameters),
            "frozen_parameters": sum(
                parameter.numel() for parameter in self.frozen_parameters.values()
            ),
            "total_parameters": sum(parameter.numel() for parameter in self.model.parameters()),
            "scope": self.scope["id"],
            "scope_version": config.get("scope_version", 0),
            "trainable_modules": selected_modules,
            "frozen_batchnorm_modules": self.frozen_batchnorm_modules,
            "training_proposal_filtering": {
                "rpn_score_threshold": self.model.rpn.score_thresh,
                "parent_inference_rpn_score_threshold": inference_proposal_threshold,
                "rpn_nms_iou_threshold": self.model.rpn.nms_thresh,
                "pre_nms_top_n": self.model.rpn.pre_nms_top_n(),
                "post_nms_top_n": self.model.rpn.post_nms_top_n(),
                "reason": "Retain background proposals for negative training images",
            },
            "deterministic_algorithms": True,
            "head_initialization": (
                "Preserved the trained parent's three-class prediction head"
                if self.detector.spec.get("origin") == "trained"
                else "Copied parent background/person/car classifier and box regression rows"
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
        for name, parameter in self.selected_parameters.items():
            if parameter.grad is None or not torch.isfinite(parameter.grad).all().item():
                raise ValueError(f"Missing or nonfinite trainable gradient: {name}")
            if torch.any(parameter.grad != 0).item():
                self.gradient_modules.update(
                    prefix
                    for prefix in self.scope["trainable_modules"]
                    if _matches_module(name, prefix)
                )
        if any(parameter.grad is not None for parameter in self.frozen_parameters.values()):
            raise ValueError("A frozen parameter unexpectedly received a gradient")
        self.optimizer.step()
        if any(not torch.isfinite(parameter).all().item() for parameter in self.parameters):
            raise ValueError("Nonfinite trainable weights; no checkpoint was published")
        return {
            "loss": float(loss.detach()),
            "losses": {name: float(value.detach()) for name, value in losses.items()},
        }

    def write_checkpoint(self, path: Path) -> dict:
        if any(not self.torch.isfinite(parameter).all().item() for parameter in self.parameters):
            raise ValueError("Nonfinite trainable weights; no checkpoint was published")
        changes = {
            name: not self.torch.equal(self.initial[name], parameter.detach())
            for name, parameter in self.selected_parameters.items()
        }
        if not any(changes.values()):
            raise ValueError("No trainable weights changed; no checkpoint was published")
        if any(
            _tensor_digest(parameter) != self.frozen_initial[name]
            for name, parameter in self.frozen_parameters.items()
        ):
            raise ValueError("Frozen model weights changed; no checkpoint was published")
        current_buffers = dict(self.model.named_buffers())
        if current_buffers.keys() != self.initial_buffers.keys() or any(
            not self.torch.equal(before, current_buffers[name])
            for name, before in self.initial_buffers.items()
        ):
            raise ValueError("Frozen model buffers changed; no checkpoint was published")
        module_changes = {
            prefix: any(
                changed and _matches_module(name, prefix) for name, changed in changes.items()
            )
            for prefix in self.scope["trainable_modules"]
        }
        self.torch.save(self.model.eval().state_dict(), path)
        return {
            "head_weights_changed": any(
                changed and _matches_module(name, "roi_heads.box_predictor")
                for name, changed in changes.items()
            ),
            "trainable_weights_changed": True,
            "changed_trainable_modules": [
                prefix for prefix, changed in module_changes.items() if changed
            ],
            "trainable_module_changes": module_changes,
            "modules_with_nonzero_gradients": sorted(self.gradient_modules),
            "frozen_parameters_unchanged": True,
            "frozen_batchnorm_buffers_unchanged": True,
            "model_buffers_unchanged": True,
        }


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
    selected_scope = _scope_from_config(config)
    dataset = store.get("dataset_versions", training["dataset_id"])
    if dataset is None or dataset["manifest_sha256"] != config["dataset_manifest_sha256"]:
        raise ValueError("Dataset changed since training was queued")
    manifest = _manifest(store, training["dataset_id"])
    parent = _ready_parent(store, training["parent_model_id"])
    if parent["weight_sha256"] != config["parent_weight_sha256"]:
        raise ValueError("Parent checkpoint changed since training was queued")
    _check_holdouts(manifest, parent)
    frames = [frame for frame in manifest["frames"] if frame["split"] == "train"]
    progress(0, f"Loading the local parent checkpoint; scope: {selected_scope['label']}")
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
        "scope": selected_scope["id"],
        "scope_version": config.get("scope_version", 0),
        "trainable_modules": selected_scope["trainable_modules"],
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
            f"CPU fine-tuning ({selected_scope['label']}): step {step}/{config['steps']}, "
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
