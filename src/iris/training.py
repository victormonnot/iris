"""Bounded, offline CPU/CUDA fine-tuning with an explicit per-run training depth.

Only frozen training images are opened. Validation and test examples remain
reserved for a separate quality evaluation; training loss is not an accuracy metric.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sys
import tempfile
import time
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path

from PIL import Image

from iris.media import _pixel_hash
from iris.model_taxonomy import class_contract, compatible_parent, dataset_contract
from iris.models import (
    IRIS_NATIVE_TO_COCO,
    TRAINING_ARCHITECTURE,
    TorchvisionDetector,
    catalog,
)
from iris.store import Store, new_id, now
from iris.training_architectures import (
    SCOPE_VERSION,
    SSDLITE,
    TRAINING_ARCHITECTURES,
    YOLOX,
    training_scope,
)
from iris.training_architectures import (
    TRAINING_SCOPES as TRAINING_SCOPES,  # Preserve the historical public scopes.
)
from iris.training_device import CUDA_PROTOCOL, device_label, normalize_device, resolve_device

MAX_STEPS = 10000
CLASS_MAPPING = {"person": 1, "car": 2}


def _scope_from_config(config: dict) -> dict:
    architecture = config.get("architecture", TRAINING_ARCHITECTURE)
    scope = training_scope(config.get("scope", "prediction_head_only"), architecture)
    if architecture == SSDLITE:
        from iris.ssdlite_training import POLICY

        if config.get("training_adapter") != POLICY or "scope_version" not in config:
            raise ValueError("SSDLite training adapter changed; create a new run")
    if architecture == YOLOX:
        from iris.yolox_spec import POLICY

        if config.get("training_adapter") != POLICY or "scope_version" not in config:
            raise ValueError("YOLOX training adapter changed; create a new run")
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

    return load_manifest(store, dataset_id, verify_images=False)


def _ready_parent(store: Store, parent_model_id: str) -> dict:
    parent = next((item for item in catalog(store.root) if item["id"] == parent_model_id), None)
    if parent is None or parent.get("architecture") not in TRAINING_ARCHITECTURES:
        raise ValueError(
            "Choose a supported Faster R-CNN, SSDLite or YOLOX detector, or its trained descendants"
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
    checkpoint_interval: int | None = None,
    device: str = "cpu",
) -> tuple:
    training_scope(scope)  # Reject an invalid scope before accessing a dataset.
    device = normalize_device(device)
    if not isinstance(name, str):
        raise ValueError("Training name must contain between 1 and 160 characters")
    name = name.strip()
    if not 1 <= len(name) <= 160:
        raise ValueError("Training name must contain between 1 and 160 characters")
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= MAX_STEPS:
        raise ValueError("Choose between 1 and 10,000 training steps")
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
    contract = dataset_contract(manifest)
    training_frames = [frame for frame in manifest["frames"] if frame["split"] == "train"]
    if not training_frames:
        raise ValueError("The dataset needs at least one training image")
    if not any(frame["boxes"] for frame in training_frames):
        raise ValueError("Training needs at least one positive annotation in the selected classes")
    parent = _ready_parent(store, parent_model_id)
    selected_scope = training_scope(scope, parent["architecture"])
    compatible_parent(parent, contract)
    _check_holdouts(manifest, parent)
    device, device_identity = resolve_device(device)
    config = {
        "steps": steps,
        "learning_rate": float(learning_rate),
        "seed": seed,
        "device": device,
        "scope": selected_scope["id"],
        "scope_version": SCOPE_VERSION,
        "trainable_modules": selected_scope["trainable_modules"],
        "batch_size": 1,
        "optimizer": "SGD",
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "dataset_manifest_sha256": dataset["manifest_sha256"],
        "parent_weight_sha256": parent["weight_sha256"],
        **contract,
        "quality_metrics": "Not computed; training loss does not measure detection quality",
    }
    if parent["architecture"] == SSDLITE:
        from iris.ssdlite_training import POLICY

        config.update(architecture=SSDLITE, training_adapter=deepcopy(POLICY))
    if parent["architecture"] == YOLOX:
        from iris.yolox_spec import POLICY

        config.update(architecture=YOLOX, training_adapter=deepcopy(POLICY))
    if device != "cpu":
        config.update(
            device_identity=device_identity,
            precision="float32",
            deterministic_algorithms=False,
        )
    if checkpoint_interval is not None or steps > 200 or device != "cpu":
        from iris.training_recovery import PROTOCOL, validate_config

        config.update(
            checkpoint_protocol=PROTOCOL if device == "cpu" else CUDA_PROTOCOL,
            checkpoint_interval=50 if checkpoint_interval is None else checkpoint_interval,
            history_interval=10,
        )
        validate_config(config)
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
    checkpoint_interval: int | None = None,
    device: str = "cpu",
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
        checkpoint_interval=checkpoint_interval,
        device=device,
    )
    from iris.training_recovery import digest, durable

    request_id = new_id()
    fingerprint = digest(
        {
            "name": name.strip(),
            "dataset_id": dataset_id,
            "parent_model_id": parent_model_id,
            "config": config,
            "request_id": request_id,
        }
    )
    return {
        "request_id": request_id,
        "fingerprint": fingerprint,
        "resume_supported": durable(config),
        "config": config,
        "scope": _scope_from_config(config),
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
            "device": config["device"],
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
                    "SSDLite uses a fixed 320 × 320 input. Small objects may lose detail; "
                    "compare validation quality before selecting a model.",
                    "On empty training images, the three hardest background anchors contribute "
                    "classification loss. Positive images retain the native SSD loss.",
                ]
                if parent["architecture"] == SSDLITE
                else []
            ),
            *(
                [
                    "YOLOX-Nano uses fixed 416 × 416 BGR letterboxing and native SimOTA loss. "
                    "This bounded fine-tuning uses no augmentation or EMA. Compare validation "
                    "quality before deployment; it does not reproduce the full upstream recipe.",
                    "YOLOX has no background class. Empty images contribute objectness loss; "
                    "box and class losses stay connected with zero positive targets.",
                    "Finite YOLOX gradients are clipped to a global L2 norm of 10 before SGD. "
                    "The unclipped norm and clipping flag are recorded for every step. "
                    "Nonfinite gradients still stop the attempt without publishing weights.",
                ]
                if parent["architecture"] == YOLOX
                else []
            ),
            *(
                [
                    f"Save optimizer and RNG state every {config['checkpoint_interval']} "
                    f"steps and at completion. "
                    "Keep the latest two states per attempt (up to 512 MiB each); source "
                    "states stay with resumed attempts.",
                    "After an interruption, explicitly preview a new attempt from the "
                    "latest durable state. "
                    "Recorded work after that state is recomputed. Float32, batch one; "
                    "continuation requires the original runtime and training device.",
                ]
                if durable(config)
                else [
                    "This short run has no optimizer recovery. Enable checkpointing for "
                    "durable state."
                ]
            ),
            *(
                [
                    "Deeper adaptation needs more memory and computation on the selected device; "
                    "use a small learning rate and a short first run."
                ]
                if scope != "prediction_head_only"
                else []
            ),
        ],
        "device_notes": [
            "Choose CPU or CUDA independently for comparison and evaluation. "
            "The current YOLOX ONNX export targets OpenCV CPU."
            if parent["architecture"] == YOLOX
            else "The training device does not restrict where the completed model can run. "
            "Choose CPU or CUDA independently for comparison, evaluation and export.",
            *(
                [
                    "CUDA training saves CPU and selected-GPU random state, but CUDA "
                    "operations may be nondeterministic. Exact repeatability is not promised.",
                    "GPU memory exhaustion stops this attempt and preserves published states. "
                    "There is no automatic CPU fallback or change of training settings.",
                ]
                if config["device"] != "cpu"
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
    checkpoint_interval: int | None = None,
    request_id: str | None = None,
    expected_fingerprint: str | None = None,
    device: str = "cpu",
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
        checkpoint_interval=checkpoint_interval,
        device=device,
    )
    from iris.training_recovery import _ID, digest, durable

    if durable(config):
        if (
            not isinstance(request_id, str)
            or not _ID.fullmatch(request_id)
            or expected_fingerprint
            != digest(
                {
                    "name": name,
                    "dataset_id": dataset_id,
                    "parent_model_id": parent_model_id,
                    "config": config,
                    "request_id": request_id,
                }
            )
        ):
            raise ValueError(
                "Training inputs changed or the preview is missing; preview this run again"
            )
        config["request_id"] = request_id
    training_id, job_id, created_at = new_id(), new_id(), now()
    with jobs.guard, store.connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        if durable(config):
            previous = connection.execute(
                "SELECT id,name,config,dataset_id,parent_model_id FROM training_runs "
                "WHERE json_extract(config,'$.request_id')=?",
                (request_id,),
            ).fetchone()
            if previous:
                if (
                    json.loads(previous["config"]) != config
                    or previous["name"] != name
                    or previous["dataset_id"] != dataset_id
                    or previous["parent_model_id"] != parent_model_id
                ):
                    raise ValueError("This preview was already used with different training inputs")
                return training_detail(store, previous["id"])
        connection.execute(
            "INSERT INTO jobs (id,kind,status,params,message,created_at) VALUES (?,?,?,?,?,?)",
            (
                job_id,
                "train",
                "queued",
                json.dumps({"training_id": training_id}),
                f"Waiting for local {device_label(config['device'])} fine-tuning",
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
    from iris.training_recovery import recovery_summary

    return {
        **training,
        "job": store.get("jobs", training["job_id"]),
        "checkpoint": checkpoint,
        "checkpoints": store.list("training_checkpoints", training_id=training_id),
        "recovery": recovery_summary(store, training),
    }


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
        self.contract = class_contract(config)
        self.class_mapping = self.contract["class_mapping"]
        self.device = torch.device(normalize_device(config.get("device", "cpu")))
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable; this GPU attempt cannot fall back to CPU.")
            torch.cuda.set_device(self.device)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
        torch.manual_seed(config["seed"])
        torch.use_deterministic_algorithms(self.device.type == "cpu")
        self.detector = TorchvisionDetector(root, parent_id, device=str(self.device))
        if self.detector.metadata["weight_sha256"] != config["parent_weight_sha256"]:
            raise ValueError("Parent checkpoint changed since training was queued")
        compatible_parent(self.detector.spec, self.contract)
        self.model = self.detector.model
        self.architecture = config.get("architecture", TRAINING_ARCHITECTURE)
        if self.detector.spec.get("architecture", TRAINING_ARCHITECTURE) != self.architecture:
            raise ValueError("Parent architecture differs from the frozen training configuration")
        adapter_metadata = {}
        if self.architecture == SSDLITE:
            from iris.ssdlite_training import configure_ssdlite_training, prepare_ssdlite_head

            adapter_metadata.update(
                prepare_ssdlite_head(
                    self.model,
                    torch,
                    self.contract,
                    trained=self.detector.spec.get("origin") == "trained",
                )
            )
            adapter_metadata.update(configure_ssdlite_training(self.model, torch))
        elif self.architecture == YOLOX:
            from iris.yolox_training import prepare_yolox_head

            adapter_metadata.update(
                prepare_yolox_head(
                    self.model,
                    torch,
                    self.contract,
                    trained=self.detector.spec.get("origin") == "trained",
                )
            )
        else:
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
                    previous.cls_score.in_features, len(self.class_mapping) + 1
                ).to(self.device)
                # Keep seeded initialization for unmapped classes. Only an explicit
                # native COCO mapping authorizes copying an official category's rows.
                source_rows = {
                    0: 0,
                    **{
                        self.class_mapping[item["id"]]: item["coco_id"]
                        for item in self.contract["taxonomy"]["classes"]
                        if item.get("coco_id") is not None
                    },
                }
                with torch.no_grad():
                    for target, source in source_rows.items():
                        predictor.cls_score.weight[target].copy_(previous.cls_score.weight[source])
                        predictor.cls_score.bias[target].copy_(previous.cls_score.bias[source])
                        predictor.bbox_pred.weight[target * 4 : (target + 1) * 4].copy_(
                            previous.bbox_pred.weight[source * 4 : (source + 1) * 4]
                        )
                        predictor.bbox_pred.bias[target * 4 : (target + 1) * 4].copy_(
                            previous.bbox_pred.bias[source * 4 : (source + 1) * 4]
                        )
                self.model.roi_heads.box_predictor = predictor
            if (
                self.model.roi_heads.box_predictor.cls_score.out_features
                != len(self.class_mapping) + 1
                or self.model.roi_heads.box_predictor.bbox_pred.out_features
                != (len(self.class_mapping) + 1) * 4
            ):
                raise ValueError("Parent prediction head does not match its frozen class mapping")
        modules = dict(self.model.named_modules())
        selected_modules = self.scope["trainable_modules"]
        if any(prefix not in modules for prefix in selected_modules):
            raise ValueError("Detector module layout does not match the selected training scope")
        if self.architecture == TRAINING_ARCHITECTURE:
            if self.scope["id"] == "partial_backbone" and list(
                self.model.backbone.body._modules
            ) != [str(index) for index in range(17)]:
                raise ValueError(
                    "The partial scope requires the supported 17-block MobileNet backbone"
                )
            if any(
                isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
                for module in modules.values()
            ):
                raise ValueError("The training detector must retain frozen batch normalization")
        elif self.architecture == SSDLITE and (
            list(self.model.backbone.features._modules) != ["0", "1"]
            or list(self.model.backbone.extra._modules) != [str(index) for index in range(4)]
        ):
            raise ValueError("Unsupported SSDLite feature layout")
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
            name: parameter.detach().cpu().clone()
            for name, parameter in self.selected_parameters.items()
        }
        self.frozen_initial = {
            name: _tensor_digest(parameter) for name, parameter in self.frozen_parameters.items()
        }
        self.initial_buffers = {
            name: value.detach().cpu().clone() for name, value in self.model.named_buffers()
        }
        self.frozen_batchnorm_modules = [
            name
            for name, module in modules.items()
            if isinstance(module, torchvision.ops.misc.FrozenBatchNorm2d)
            or (
                self.architecture in (SSDLITE, YOLOX)
                and isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
            )
        ]
        self.gradient_modules = set()
        self.model.train()
        if self.scope["id"] == "prediction_head_only":
            self.model.backbone.eval()
        # Preserve all running statistics, including ordinary SSDLite BatchNorm.
        # Eval mode retains affine gradients and supports batch-one 1x1 feature maps.
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
            "architecture": self.architecture,
            **self.contract,
            "trainable_parameters": sum(parameter.numel() for parameter in self.parameters),
            "frozen_parameters": sum(
                parameter.numel() for parameter in self.frozen_parameters.values()
            ),
            "total_parameters": sum(parameter.numel() for parameter in self.model.parameters()),
            "scope": self.scope["id"],
            "scope_version": config.get("scope_version", 0),
            "trainable_modules": selected_modules,
            "frozen_batchnorm_modules": self.frozen_batchnorm_modules,
            **(
                {
                    "training_proposal_filtering": {
                        "rpn_score_threshold": self.model.rpn.score_thresh,
                        "parent_inference_rpn_score_threshold": inference_proposal_threshold,
                        "rpn_nms_iou_threshold": self.model.rpn.nms_thresh,
                        "pre_nms_top_n": self.model.rpn.pre_nms_top_n(),
                        "post_nms_top_n": self.model.rpn.post_nms_top_n(),
                        "reason": "Retain background proposals for negative training images",
                    },
                }
                if self.architecture == TRAINING_ARCHITECTURE
                else adapter_metadata
            ),
            "deterministic_algorithms": self.device.type == "cpu",
            "training_device": str(self.device),
            "checkpoint_storage_device": "cpu",
            "inference_devices": ["cpu", "cuda"],
            "head_initialization": (
                "Preserved the trained parent's compatible prediction head"
                if self.detector.spec.get("origin") == "trained"
                else "Copied background and explicitly mapped COCO rows; seeded new class rows"
            ),
            "head_initialization_rows": {
                item["id"]: (
                    "parent"
                    if self.detector.spec.get("origin") == "trained"
                    else item.get("coco_id")
                )
                for item in self.contract["taxonomy"]["classes"]
            },
            "head_class_slots": len(self.class_mapping) + 1,
            "validation_consumed": False,
            "test_consumed": False,
        }
        if self.architecture == YOLOX:
            self.metadata.update(adapter_metadata)
            self.metadata.pop("native_to_coco", None)
        if self.contract["taxonomy_id"] == "iris-objects-v1":
            self.metadata["native_to_coco"] = IRIS_NATIVE_TO_COCO

    def resume_runtime(self):
        from iris.training_state import runtime_identity

        return runtime_identity(self)

    def write_resume_state(self, path, *, binding, sampler):
        from iris.training_state import write_state

        write_state(self, path, binding=binding, sampler=sampler)

    def load_resume_state(self, path, *, binding, sampler, expected_sha256):
        from iris.training_state import load_state

        load_state(self, path, binding=binding, sampler=sampler, expected_sha256=expected_sha256)

    def step(self, image: Image.Image, boxes: list[dict]) -> dict:
        try:
            return self._step(image, boxes)
        except self.torch.cuda.OutOfMemoryError as exc:
            raise RuntimeError(
                "GPU memory exhausted. Published recovery states remain available. "
                "Free GPU memory before continuing, or prepare a separate run with "
                "a smaller training scope. No CPU fallback was performed."
            ) from exc

    def _step(self, image: Image.Image, boxes: list[dict]) -> dict:
        torch = self.torch
        self.optimizer.zero_grad(set_to_none=True)
        if self.architecture == YOLOX:
            from iris.yolox_training import training_losses

            loss, losses = training_losses(
                self.model,
                image,
                boxes,
                self.class_mapping,
                torch,
                self.device,
            )
        else:
            tensor = (
                self.detector.functional.pil_to_tensor(image).to(
                    device=self.device, dtype=torch.float32
                )
                / 255
            )
            coordinates = torch.tensor(
                [box["box"] for box in boxes], dtype=torch.float32, device=self.device
            ).reshape(-1, 4)
            labels = torch.tensor(
                [self.class_mapping[box["label"]] for box in boxes],
                dtype=torch.int64,
                device=self.device,
            )
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
        gradient_measurement = {}
        if self.architecture == YOLOX:
            from iris.yolox_training import clip_gradients

            gradient_measurement = clip_gradients(self.parameters, torch)
        self.optimizer.step()
        if any(not torch.isfinite(parameter).all().item() for parameter in self.parameters):
            raise ValueError("Nonfinite trainable weights; no checkpoint was published")
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return {
            "loss": float(loss.detach()),
            "losses": {name: float(value.detach()) for name, value in losses.items()},
            **gradient_measurement,
        }

    def write_checkpoint(self, path: Path) -> dict:
        if any(not self.torch.isfinite(parameter).all().item() for parameter in self.parameters):
            raise ValueError("Nonfinite trainable weights; no checkpoint was published")
        changes = {
            name: not self.torch.equal(self.initial[name].cpu(), parameter.detach().cpu())
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
            not self.torch.equal(before.cpu(), current_buffers[name].detach().cpu())
            for name, before in self.initial_buffers.items()
        ):
            raise ValueError("Frozen model buffers changed; no checkpoint was published")
        module_changes = {
            prefix: any(
                changed and _matches_module(name, prefix) for name, changed in changes.items()
            )
            for prefix in self.scope["trainable_modules"]
        }
        self.torch.save(
            {name: value.detach().cpu() for name, value in self.model.eval().state_dict().items()},
            path,
        )
        return {
            "head_weights_changed": any(
                changed
                and _matches_module(
                    name,
                    "head" if self.architecture in (SSDLITE, YOLOX) else "roi_heads.box_predictor",
                )
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
    try:
        return _run_training(store, training_id, progress, cancelled, trainer_factory)
    except RuntimeError as exc:
        torch = sys.modules.get("torch")
        row = store.get("training_runs", training_id)
        if (
            torch is not None
            and isinstance(exc, torch.cuda.OutOfMemoryError)
            and row
            and row["config"].get("device", "cpu") != "cpu"
        ):
            raise RuntimeError(
                "GPU memory exhausted while loading, training or restoring state. "
                "Any published recovery states remain available. Free GPU memory before "
                "continuing, or prepare a separate run with a smaller training scope. "
                "No CPU fallback was performed."
            ) from exc
        raise


def _run_training(store, training_id, progress, cancelled, trainer_factory):
    training = store.get("training_runs", training_id)
    if training is None:
        raise ValueError("Training run not found")
    from iris import training_recovery as recovery

    resumable = recovery.durable(training["config"])
    if training["checkpoint_id"] or (
        training["history"] and not (resumable and training["config"].get("resume_from"))
    ):
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
    device = normalize_device(config.get("device", "cpu"))
    if device != "cpu":
        resolve_device(device, expected=config.get("device_identity"))
    if resumable:
        recovery.validate_config(config)
    selected_scope = _scope_from_config(config)
    dataset = store.get("dataset_versions", training["dataset_id"])
    if dataset is None or dataset["manifest_sha256"] != config["dataset_manifest_sha256"]:
        raise ValueError("Dataset changed since training was queued")
    manifest = _manifest(store, training["dataset_id"])
    contract = dataset_contract(manifest)
    if class_contract(config) != contract:
        raise ValueError("Dataset class definitions or training mappings changed since queueing")
    parent = _ready_parent(store, training["parent_model_id"])
    if parent["weight_sha256"] != config["parent_weight_sha256"]:
        raise ValueError("Parent checkpoint changed since training was queued")
    if parent["architecture"] != config.get("architecture", TRAINING_ARCHITECTURE):
        raise ValueError("Parent architecture differs from the frozen training configuration")
    compatible_parent(parent, contract)
    _check_holdouts(manifest, parent)
    frames = [frame for frame in manifest["frames"] if frame["split"] == "train"]
    progress(0, f"Loading the local parent checkpoint; scope: {selected_scope['label']}")
    token = recovery.claim_attempt(store, training) if resumable else None
    trainer = (trainer_factory or _HeadTrainer)(store.root, training["parent_model_id"], config)
    metadata = {
        **trainer.metadata,
        "architecture": parent["architecture"],
        **contract,
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
    if resumable:
        if not recovery.save_metadata(store, training, metadata, token):
            return {**result, "cancelled": True}
        history = list(training["history"])
        randomizer, order, _, sampler = recovery.sampling(config["seed"], frames, len(history))
        if config.get("resume_from"):
            checkpoint = store.get("training_checkpoints", config["resume_from"]["checkpoint_id"])
            with store.connect() as connection:
                recovery.validate_training_recoveries(connection, store.root)
            expected = recovery.binding(training, history, trainer.resume_runtime())
            if recovery.canonical(checkpoint["metadata"]) != recovery.canonical(
                {"binding": expected, "sampler": sampler}
            ):
                raise ValueError("Training state or runtime changed; cannot resume this attempt")
            trainer.load_resume_state(
                store.artifact_path(checkpoint["path"]),
                binding=expected,
                sampler=sampler,
                expected_sha256=checkpoint["state_sha256"],
            )
        result["steps_completed"] = len(history)
    else:
        store.update("training_runs", training_id, {"metadata": metadata})
        randomizer, order, history = random.Random(config["seed"]), [], []
    elapsed_before = history[-1]["elapsed_seconds"] if history else 0.0
    started = time.perf_counter()
    last_saved = len(history)

    def save_state():
        nonlocal last_saved
        if resumable and history and len(history) != last_saved:
            sampler = json.loads(
                recovery.canonical(
                    {"random_state": randomizer.getstate(), "remaining_order": order}
                )
            )
            saved = recovery.save_checkpoint(store, training, trainer, history, sampler, token)
            if saved:
                last_saved = len(history)
            return bool(saved)
        return True

    for step in range(len(history) + 1, config["steps"] + 1):
        if cancelled():
            save_state()
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
                "elapsed_seconds": elapsed_before + time.perf_counter() - started,
            }
        )
        if resumable:
            if step % config["checkpoint_interval"] == 0 or step == config["steps"]:
                if not save_state():
                    return {**result, "cancelled": True}
            elif step % config["history_interval"] == 0:
                if not recovery.save_history(store, training, history, token):
                    return {**result, "cancelled": True}
        else:
            store.update("training_runs", training_id, {"history": history})
        result["steps_completed"] = step
        progress(
            step / config["steps"],
            f"{device_label(device)} fine-tuning ({selected_scope['label']}): "
            f"step {step}/{config['steps']}, "
            f"training loss {measurement['loss']:.4f}",
        )
    if cancelled():
        save_state()
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
                "SELECT status,cancel_requested,params FROM jobs WHERE id=?",
                (training["job_id"],),
            ).fetchone()
            if (
                job["cancel_requested"]
                or job["status"] not in {"queued", "running"}
                or (resumable and json.loads(job["params"]).get("training_claim") != token)
            ):
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
                    parent["architecture"],
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
                    f"{device_label(device)} fine-tuning complete; "
                    "checkpoint available for comparison",
                    training["job_id"],
                ),
            )
        published = True
        return result
    finally:
        temporary_path.unlink(missing_ok=True)
        if not published:
            destination.unlink(missing_ok=True)
