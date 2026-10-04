"""Frozen, offline SAM 3 image proposals; the optional CUDA runtime is isolated.

Only pixels and explicitly chosen short class phrases reach the runtime. Human
reference boxes, corrections, notes and workspace databases never enter it.
"""

import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path

from PIL import Image

from iris.dataset_manifest import taxonomy_mappings
from iris.preannotation_contracts import OUTPUT_PROTOCOL, normalize_output

PROTOCOL = "iris-sam3-preannotation-v1"
RAW_PROTOCOL = "iris-sam3-native-boxes-v1"
MODEL = "sam3"
CODE_REVISION = "2345a4ad109ac29c569da749c91d84f10dc08c40"
HF_REVISION = "3c879f39826c281e95690f02c7821c4de09afae7"
CHECKPOINT_SHA256 = "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e"
CHECKPOINT_SIZE = 3_450_062_241
CHECKPOINT_PATH = "models/sam3/sam3.pt"
MAX_PROPOSALS = 100
MAX_NATIVE_BOXES = 300
MAX_IMAGE_PIXELS = 16_777_216
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
RUNTIME_REQUIRED = {
    "python_min": "3.12",
    "python_max_exclusive": "3.13",
    "platform": "linux",
    "torch": "2.10.0",
    "torchvision": "0.25.0",
    "numpy": "1.26.4",
    "cuda_min": "12.6",
    "bfloat16": True,
    "isolated": True,
}
_WEIGHT_CACHE = {}


class ProviderResponseError(RuntimeError):
    """Preserve returned native evidence when normalization or execution fails."""

    def __init__(self, message, *, raw_response=None, metadata=None):
        super().__init__(message)
        self.raw_response = raw_response
        self.metadata = deepcopy(metadata or {})


def _json(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (ValueError, TypeError, RecursionError) as exc:
        raise ValueError("SAM configuration and output must contain finite JSON values") from exc


def _number(value):
    try:
        return type(value) in {float, int} and math.isfinite(value)
    except OverflowError:
        return False


def _prompts(taxonomy, value):
    classes = taxonomy["classes"]
    if not 1 <= len(classes) <= 100:
        raise ValueError("SAM requires between 1 and 100 frozen classes")
    if value is None:
        value = [{"class_id": item["id"], "text": item["name"]} for item in classes]
    elif isinstance(value, dict):
        if set(value) != {item["id"] for item in classes}:
            raise ValueError("Supply exactly one SAM text phrase for every frozen class")
        value = [{"class_id": item["id"], "text": value[item["id"]]} for item in classes]
    if not isinstance(value, list) or len(value) != len(classes):
        raise ValueError("Supply exactly one SAM text phrase for every frozen class")
    result = []
    for item, label in zip(value, classes, strict=True):
        if (
            not isinstance(item, dict)
            or set(item) != {"class_id", "text"}
            or item["class_id"] != label["id"]
            or not isinstance(item["text"], str)
            or not 1 <= len(item["text"].strip()) <= 120
            or not item["text"].isprintable()
        ):
            raise ValueError(
                "SAM phrases must follow frozen class order and contain 1–120 printable characters"
            )
        result.append({"class_id": item["class_id"], "text": item["text"].strip()})
    return result


def freeze_config(taxonomy, *, model=MODEL, class_prompts=None, threshold=0.5, device="cuda"):
    """Prepare an immutable configuration without inspecting installations or weights."""
    taxonomy_mappings(taxonomy)
    if model != MODEL or device != "cuda":
        raise ValueError("This profile supports SAM 3 images on CUDA only, without CPU fallback")
    if not _number(threshold) or not 0 <= threshold <= 1:
        raise ValueError("SAM threshold must be a finite number between zero and one")
    return {
        "protocol": PROTOCOL,
        "provider": MODEL,
        "model": MODEL,
        "local_only": True,
        "external": False,
        "taxonomy": deepcopy(taxonomy),
        "taxonomy_id": taxonomy["id"],
        "prompts": _prompts(taxonomy, class_prompts),
        "settings": {"threshold": float(threshold), "device": "cuda", "precision": "bfloat16"},
        "code_revision": CODE_REVISION,
        "source": {
            "repository": "https://github.com/facebookresearch/sam3",
            "hf_repository": "facebook/sam3",
            "hf_revision": HF_REVISION,
            "checkpoint_file": "sam3.pt",
            "license": "SAM License",
        },
        "weights": {
            "path": CHECKPOINT_PATH,
            "sha256": CHECKPOINT_SHA256,
            "size_bytes": CHECKPOINT_SIZE,
        },
        "runtime_required": deepcopy(RUNTIME_REQUIRED),
        "preprocessing": {
            "input": "original_oriented_rgb_pixels",
            "resolution": 1008,
            "resize": "square_bilinear",
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
        },
        "postprocessing": {
            "source": "native_detector_boxes_not_mask_bounds",
            "native_coordinates": "xyxy_normalized",
            "output_coordinates": "xyxy_original_pixels",
            "score": "sigmoid_detection_logit_times_sigmoid_presence_logit",
            "threshold_comparison": ">",
            "clip_to_image": True,
            "cross_class_nms": False,
            "masks_retained": False,
            "partial_outputs": "error_without_proposals",
        },
        "limits": {
            "max_classes": 100,
            "max_prompt_characters": 120,
            "max_prompt_tokens": 30,
            "max_native_boxes_per_prompt": MAX_NATIVE_BOXES,
            "max_proposals": MAX_PROPOSALS,
            "max_image_pixels": MAX_IMAGE_PIXELS,
            "max_image_bytes": 16 * 1024 * 1024,
            "timeout_seconds": 300,
        },
    }


def validate_frozen_config(config):
    """Pure historical validation; this version's profile constants are immutable."""
    if not isinstance(config, dict) or not isinstance(config.get("settings"), dict):
        raise ValueError("A complete frozen SAM configuration is required")
    expected = freeze_config(
        config.get("taxonomy"),
        model=config.get("model"),
        class_prompts=config.get("prompts"),
        threshold=config["settings"].get("threshold"),
        device=config["settings"].get("device"),
    )
    if _json(config) != _json(expected):
        raise ValueError("The frozen SAM profile, prompts or model identity are inconsistent")
    return deepcopy(expected)


def _weights_status(root, *, force=False):
    result = {
        "available": False,
        "path": CHECKPOINT_PATH,
        "expected_sha256": CHECKPOINT_SHA256,
        "sha256": CHECKPOINT_SHA256,
        "size_bytes": CHECKPOINT_SIZE,
    }
    root = Path(root).resolve()
    original_path = root / CHECKPOINT_PATH
    path = original_path.resolve()
    if original_path.is_symlink() or not path.is_relative_to(root):
        return {
            **result,
            "status": "invalid_weights",
            "reason": "SAM weights must be a regular file inside the workspace",
        }
    try:
        stat = path.stat()
        if not path.is_file() or stat.st_size != CHECKPOINT_SIZE:
            return {
                **result,
                "status": "invalid_weights",
                "reason": "SAM checkpoint size differs from the official pinned checkpoint",
            }
        key = (
            str(path),
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )
        digest = None if force else _WEIGHT_CACHE.get(key)
        if digest is None:
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            after = path.stat()
            after_key = (
                str(path),
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            if after_key != key:
                raise ValueError("SAM checkpoint changed while its identity was checked")
            if len(_WEIGHT_CACHE) >= 16:
                _WEIGHT_CACHE.clear()
            _WEIGHT_CACHE[key] = digest
        if digest != CHECKPOINT_SHA256:
            return {
                **result,
                "status": "invalid_weights",
                "reason": "SAM checkpoint SHA-256 differs from the official pinned checkpoint",
            }
    except FileNotFoundError:
        return {
            **result,
            "status": "missing_weights",
            "reason": "Provision the licensed SAM checkpoint locally; IRIS never downloads it",
        }
    except (OSError, ValueError):
        return {
            **result,
            "status": "invalid_weights",
            "reason": "SAM checkpoint could not be read and verified consistently",
        }
    return {**result, "available": True, "status": "ready", "reason": None}


def provider_status(root, config=None, *, force=False):
    """Local readiness only; no model construction, download or network request."""
    from iris.sam_runtime import runtime_status

    status = {
        "model": MODEL,
        "model_id": MODEL,
        "name": "SAM 3 · local text prompts",
        "local_only": True,
        "external": False,
        "ready": False,
    }
    if config is not None:
        try:
            validate_frozen_config(config)
        except (ValueError, TypeError, KeyError) as exc:
            return {**status, "status": "invalid_config", "reason": str(exc)}
    runtime = runtime_status(config, force=force)
    weights = _weights_status(root, force=force)
    result = {
        **status,
        "runtime": runtime,
        "weights": weights,
        "devices": [{"id": "cuda", "available": runtime["ready"], "reason": runtime.get("reason")}],
    }
    if not weights["available"]:
        return {**result, "status": weights["status"], "reason": weights["reason"]}
    if not runtime["ready"]:
        return {**result, "status": runtime["status"], "reason": runtime["reason"]}
    return {
        **result,
        "ready": True,
        "status": "ready",
        "reason": "Pinned local runtime and checkpoint verified; inference has not been tested",
    }


def _normalize_response(raw, config, width, height):
    if type(width) is not int or type(height) is not int or min(width, height) <= 0:
        raise ValueError("SAM original image dimensions must be positive integers")
    if width * height > MAX_IMAGE_PIXELS:
        raise ValueError("SAM image exceeds the frozen pixel limit")
    if len(_json(raw)) > MAX_RESPONSE_BYTES:
        raise ValueError("SAM native output exceeds the 2 MiB evidence limit")
    coordinates = {
        "format": "xyxy",
        "space": "normalized",
        "image_size": [width, height],
        "to_original": {"scale": [width, height], "offset": [0, 0]},
    }
    if (
        not isinstance(raw, dict)
        or raw.get("protocol") != RAW_PROTOCOL
        or raw.get("complete") is not True
        or _json(raw.get("image")) != _json({"width": width, "height": height})
        or _json(raw.get("coordinates")) != _json(coordinates)
        or not isinstance(raw.get("prompts"), list)
        or len(raw["prompts"]) != len(config["prompts"])
        or not isinstance(raw.get("metadata", {}), dict)
    ):
        raise ValueError(
            "SAM output is incomplete or differs from its frozen image/prompt contract"
        )
    proposals, filtered, clipped_count = [], 0, 0
    for prompt_index, (row, prompt) in enumerate(
        zip(raw["prompts"], config["prompts"], strict=True)
    ):
        if (
            not isinstance(row, dict)
            or row.get("class_id") != prompt["class_id"]
            or row.get("text") != prompt["text"]
            or row.get("error") is not None
            or not all(
                isinstance(row.get(key), list) for key in ("boxes", "scores", "native_indices")
            )
        ):
            raise ValueError(
                "Every SAM phrase must have one complete matching output, including empty results"
            )
        boxes, scores, indices = row["boxes"], row["scores"], row["native_indices"]
        if not len(boxes) == len(scores) == len(indices) or len(boxes) > MAX_NATIVE_BOXES:
            raise ValueError(
                "SAM native boxes, scores and query indices must match within the bound"
            )
        seen = set()
        for box, score, index in zip(boxes, scores, indices, strict=True):
            if (
                type(index) is not int
                or not 0 <= index < MAX_NATIVE_BOXES
                or index in seen
                or not _number(score)
                or not 0 <= score <= 1
                or not isinstance(box, list)
                or len(box) != 4
                or not all(_number(value) for value in box)
                or box[0] >= box[2]
                or box[1] >= box[3]
            ):
                raise ValueError("SAM native boxes, scores and indices must be finite and valid")
            seen.add(index)
            if score <= config["settings"]["threshold"]:
                filtered += 1
                continue
            clipped = [min(1.0, max(0.0, value)) for value in box]
            if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
                raise ValueError("SAM box has no positive area inside the original image")
            changed = clipped != box
            clipped_count += changed
            proposals.append(
                {
                    "id": f"sam3-{prompt_index}-{index}",
                    "label": prompt["class_id"],
                    "box": [
                        value * (width if axis % 2 == 0 else height)
                        for axis, value in enumerate(clipped)
                    ],
                    "score": float(score),
                    "source": {
                        "provider": MODEL,
                        "prompt": prompt["text"],
                        "prompt_index": prompt_index,
                        "native_index": index,
                        "native_box": deepcopy(box),
                        "native_coordinates": "xyxy_normalized",
                        "clipped": changed,
                        "box_origin": "native_detector",
                    },
                }
            )
            if len(proposals) > MAX_PROPOSALS:
                raise ValueError(
                    "SAM produced more than 100 retained proposals; no outputs were truncated"
                )
    result = normalize_output(
        {
            "protocol": OUTPUT_PROTOCOL,
            "taxonomy_id": config["taxonomy_id"],
            "coordinates": {
                "format": "xyxy",
                "space": "original_pixels",
                "image_size": [width, height],
                "to_original": {"scale": [1, 1], "offset": [0, 0]},
            },
            "proposals": proposals,
        },
        config["taxonomy"],
        width=width,
        height=height,
    )
    result.update(filtered_count=filtered, clipped_count=clipped_count)
    if clipped_count:
        result["warnings"].append(
            f"{clipped_count} native SAM boxes were clipped to image bounds; "
            "original boxes remain in raw evidence."
        )
    result["warnings"].append(
        "Each class phrase is evaluated independently; "
        "overlapping proposals of different classes are retained."
    )
    return result


def normalize_response(raw, config, *, width, height):
    """Validate every native output before applying threshold and documented clipping."""
    checked = validate_frozen_config(config)
    try:
        return _normalize_response(raw, checked, width, height)
    except ValueError as exc:
        raise ProviderResponseError(str(exc), raw_response=raw) from exc


class Sam3Preannotator:
    """Load once through the isolated local runner, then process images without references."""

    def __init__(self, root, frozen_config, *, cancelled=lambda: False):
        from iris.sam_runtime import SamRuntime

        self.config = validate_frozen_config(frozen_config)
        self._runtime = None
        if cancelled():
            raise ProviderResponseError("SAM initialization was cancelled")
        status = provider_status(root, self.config, force=True)
        if not status["ready"]:
            raise ProviderResponseError(status["reason"], metadata={"readiness": status})
        try:
            self._runtime = SamRuntime(
                self.config, Path(root).resolve() / CHECKPOINT_PATH, cancelled=cancelled
            )
        except Exception as exc:
            raise ProviderResponseError(
                str(exc) or type(exc).__name__,
                raw_response=getattr(exc, "raw_response", None),
                metadata=getattr(exc, "metadata", {"readiness": status}),
            ) from exc
        self.metadata = deepcopy(self._runtime.metadata)
        if not isinstance(self.metadata.get("runtime_identity"), dict):
            self.close()
            raise ProviderResponseError("Loaded SAM runtime did not report its actual identity")
        self.metadata.setdefault("load_ms", self.metadata.get("model_load_ms"))

    def predict(self, image, *, cancelled=lambda: False):
        if self._runtime is None:
            raise ProviderResponseError("SAM runtime is closed")
        if not isinstance(image, Image.Image) or min(image.size) <= 0:
            raise ValueError("SAM requires one decoded original image")
        if image.width * image.height > MAX_IMAGE_PIXELS:
            raise ValueError("SAM image exceeds the frozen pixel limit")
        if cancelled():
            raise ProviderResponseError("SAM prediction was cancelled")
        try:
            return self._runtime.predict(
                image,
                class_prompts=deepcopy(self.config["prompts"]),
                threshold=self.config["settings"]["threshold"],
                cancelled=cancelled,
            )
        except Exception as exc:
            raise ProviderResponseError(
                str(exc) or type(exc).__name__,
                raw_response=getattr(exc, "raw_response", None),
                metadata=getattr(exc, "metadata", self.metadata),
            ) from exc

    def close(self):
        if self._runtime is not None:
            self._runtime.close()
            self._runtime = None
