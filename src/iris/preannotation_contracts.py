"""Frozen preannotation capabilities and validated boxes, independent of model runtimes.

Only the existing detector and candidate reviewers have executable capabilities.
The generic output contract permits future adapters to be tested without advertising
an unimplemented multimodal, segmentation or combined provider.
"""

from __future__ import annotations

import json
import math
from copy import deepcopy

from iris.dataset_manifest import taxonomy_mappings
from iris.model_taxonomy import class_contract
from iris.models import get_spec
from iris.prediction_taxonomy import COCO_TAXONOMY, output_contract, validate_output_labels
from iris.taxonomies import TAXONOMY

CAPABILITIES_PROTOCOL = "iris-preannotation-capabilities-v1"
CONTRACT_PROTOCOL = "iris-detector-preannotation-v1"
OUTPUT_PROTOCOL = "iris-box-proposals-v1"
MAX_OUTPUTS = 300
MAX_OUTPUT_BYTES = 1024 * 1024
SCORE_WARNING = "Provider scores are not calibrated or comparable across models or providers."
OMISSION_WARNING = (
    "Proposals may omit objects. No proposals do not establish that the image is negative; "
    "review the whole image before human validation."
)


def provider_capabilities(provider: str) -> dict:
    """Describe implemented operations only; this does not probe or load a provider."""
    if provider not in {"local_detector", "ollama", "alibaba"}:
        raise ValueError("Unknown implemented preannotation provider")
    detector = provider == "local_detector"
    return {
        "protocol": CAPABILITIES_PROTOCOL,
        "provider": provider,
        "operations": ["propose_boxes" if detector else "review_candidates"],
        "execution": "external" if provider == "alibaba" else "local",
        "creates_boxes": detector,
        "requires_candidates": not detector,
        "custom_classes": detector,
        "class_support": "checkpoint_or_explicit_mapping" if detector else "builtin_only",
        "taxonomy_ids": None if detector else [TAXONOMY["id"]],
        "max_candidates": None if detector else 8,
        "max_proposals": 100 if detector else 8,
        "scores_comparable": False,
        "automatic_validation": False,
        "creates_class_definitions": False,
        "omission_search": "detector_outputs" if detector else "scene_notes_only",
    }


def _detector_contract(model_id: str, source_contract: dict, taxonomy: dict) -> dict:
    taxonomy_mappings(taxonomy)
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("A detector contract requires its model identity")
    if not isinstance(source_contract, dict):
        raise ValueError("A detector contract requires frozen output class definitions")
    if source_contract.get("taxonomy_id") == COCO_TAXONOMY:
        if source_contract != {"taxonomy_id": COCO_TAXONOMY}:
            raise ValueError("Unexpected official detector class contract")
        mapping = {
            str(item["coco_id"]): item["id"] for item in taxonomy["classes"] if "coco_id" in item
        }
    else:
        checked = class_contract(source_contract)
        if checked != source_contract or checked["taxonomy"] != taxonomy:
            raise ValueError("The trained detector requires the exact same class definitions")
        mapping = {
            str(identifier): label for label, identifier in checked["output_class_mapping"].items()
        }
    supported = [item["id"] for item in taxonomy["classes"] if item["id"] in mapping.values()]
    unsupported = [item["id"] for item in taxonomy["classes"] if item["id"] not in supported]
    warnings = [SCORE_WARNING, OMISSION_WARNING]
    if unsupported:
        warnings.append(
            "This detector has no explicit class mapping for: " + ", ".join(unsupported) + "."
        )
    return {
        "protocol": CONTRACT_PROTOCOL,
        "provider": "local_detector",
        "operation": "propose_boxes",
        "model_id": model_id,
        "source_contract": deepcopy(source_contract),
        "taxonomy": deepcopy(taxonomy),
        "taxonomy_id": taxonomy["id"],
        "label_mapping": mapping,
        "supported_class_ids": supported,
        "unsupported_class_ids": unsupported,
        "coverage_complete": not unsupported,
        "coordinates": "xyxy_pixels_original_image",
        "score_semantics": "detector_score_not_calibrated",
        "warnings": warnings,
    }


def build_contract(store, model_id: str, taxonomy: dict) -> dict:
    """Freeze class coverage without importing ML libraries or looking at annotations."""
    return _detector_contract(model_id, output_contract(get_spec(model_id, store.root)), taxonomy)


def validate_contract(contract: dict) -> dict:
    """Validate a saved mapping without consulting today's project or model registry."""
    if not isinstance(contract, dict):
        raise ValueError("The preannotation class contract must be an object")
    expected = _detector_contract(
        contract.get("model_id"), contract.get("source_contract"), contract.get("taxonomy")
    )
    if contract != expected:
        raise ValueError("The frozen preannotation mapping or class coverage changed")
    if type(contract["coverage_complete"]) is not bool:
        raise ValueError("Frozen preannotation coverage must be explicit")
    return expected


def _number(value):
    if type(value) not in {int, float}:
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _size(width, height):
    if (
        type(width) is not int
        or type(height) is not int
        or min(width, height) <= 0
        or not _number(width)
        or not _number(height)
    ):
        raise ValueError("Original image dimensions must be positive integers")


def _coordinates(box, width, height):
    if (
        not isinstance(box, list)
        or len(box) != 4
        or not all(_number(value) for value in box)
        or not 0 <= box[0] < box[2] <= width
        or not 0 <= box[1] < box[3] <= height
    ):
        raise ValueError(
            "Proposed boxes require finite in-bounds xyxy coordinates of positive area"
        )
    return [float(value) for value in box]


def normalize_output(payload: dict, taxonomy: dict, *, width: int, height: int) -> dict:
    """Validate complete proposals and an explicit original/normalized coordinate transform.

    Unrecognized classes fail rather than inventing definitions or silently dropping
    boxes. Adapters must persist their raw output before calling this validator.
    Pixel inputs refer to the original oriented image; normalized inputs refer to
    that entire image. Crops, masks and iterative protocols require their own
    explicit adapter conversion before reaching this boundary.
    """
    taxonomy_mappings(taxonomy)
    _size(width, height)
    try:
        encoded = json.dumps(payload, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError("Provider output must be finite JSON") from exc
    if len(encoded) > MAX_OUTPUT_BYTES:
        raise ValueError("Provider output exceeds the 1 MiB contract limit")
    if (
        not isinstance(payload, dict)
        or set(payload) != {"protocol", "taxonomy_id", "coordinates", "proposals"}
        or payload["protocol"] != OUTPUT_PROTOCOL
        or payload["taxonomy_id"] != taxonomy["id"]
        or not isinstance(payload["proposals"], list)
        or len(payload["proposals"]) > MAX_OUTPUTS
    ):
        raise ValueError("Provider output must use the frozen bounded proposal protocol")
    coordinates = payload["coordinates"]
    if not isinstance(coordinates, dict) or coordinates.get("space") not in {
        "original_pixels",
        "normalized",
    }:
        raise ValueError("Declare original_pixels or normalized coordinates explicitly")
    normalized = coordinates["space"] == "normalized"
    scale = [width, height] if normalized else [1, 1]
    expected = {
        "format": "xyxy",
        "space": coordinates["space"],
        "image_size": [width, height],
        "to_original": {"scale": scale, "offset": [0, 0]},
    }
    if (
        coordinates != expected
        or any(type(value) is not int for value in coordinates.get("image_size", []))
        or not all(
            _number(value) for values in coordinates["to_original"].values() for value in values
        )
    ):
        raise ValueError("Coordinate transformation must explicitly match the original image")
    labels = {item["id"] for item in taxonomy["classes"]}
    proposals, identifiers = [], set()
    for index, item in enumerate(payload["proposals"]):
        if (
            not isinstance(item, dict)
            or not {"id", "label", "box"} <= item.keys()
            or item.keys() - {"id", "label", "box", "score", "uncertain", "reason", "source"}
            or not isinstance(item["id"], str)
            or not 1 <= len(item["id"]) <= 128
            or not item["id"].strip()
            or item["id"] in identifiers
        ):
            raise ValueError("Each proposal needs a unique source ID, class ID and box")
        if not isinstance(item["label"], str) or item["label"] not in labels:
            raise ValueError("A proposed class must belong to the frozen taxonomy")
        identifiers.add(item["id"])
        box = _coordinates(item["box"], 1 if normalized else width, 1 if normalized else height)
        original = [value * scale[index % 2] for index, value in enumerate(box)]
        score = item.get("score")
        if score is not None and (not _number(score) or not 0 <= score <= 1):
            raise ValueError("A provided score must be finite and between zero and one")
        uncertain = item.get("uncertain", False)
        reason = item.get("reason", "")
        source = item.get("source", {})
        if (
            type(uncertain) is not bool
            or not isinstance(reason, str)
            or len(reason) > 2000
            or not isinstance(source, dict)
        ):
            raise ValueError("Invalid proposal uncertainty, reason or source provenance")
        proposals.append(
            {
                "id": item["id"],
                "label": item["label"],
                "box": original,
                "score": float(score) if score is not None else None,
                "uncertain": uncertain,
                "reason": reason,
                "source": deepcopy(source),
                "geometry": {
                    "raw_box": deepcopy(item["box"]),
                    "coordinates": deepcopy(coordinates),
                    "source_index": index,
                },
            }
        )
    return {
        "protocol": OUTPUT_PROTOCOL,
        "taxonomy_id": taxonomy["id"],
        "proposals": proposals,
        "raw_output": deepcopy(payload),
        "warnings": [SCORE_WARNING, OMISSION_WARNING],
    }


def normalize_candidates(
    detections: list, contract: dict, threshold: float, *, width: int, height: int
) -> dict:
    """Adapt existing detector outputs, preserving every raw detection and source index."""
    contract = validate_contract(contract)
    _size(width, height)
    if not _number(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Preannotation threshold must be between zero and one")
    if (
        not isinstance(detections, list)
        or len(detections) > MAX_OUTPUTS
        or any(not isinstance(item, dict) for item in detections)
    ):
        raise ValueError("Detector outputs must be a bounded list of detection objects")
    try:
        if len(json.dumps(detections, allow_nan=False).encode("utf-8")) > MAX_OUTPUT_BYTES:
            raise ValueError("Detector raw output exceeds the 1 MiB contract limit")
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError("Detector raw output must be bounded finite JSON") from exc
    validate_output_labels(detections, contract["source_contract"])
    proposals, filtered, unmapped = [], 0, 0
    for index, detection in enumerate(detections):
        box = _coordinates(detection.get("box"), width, height)
        score = detection.get("score")
        if not _number(score) or not 0 <= score <= 1:
            raise ValueError("Detector output has invalid confidence")
        label = contract["label_mapping"].get(str(detection["label_id"]))
        if label is None:
            unmapped += 1
            continue
        if score < threshold:
            filtered += 1
            continue
        proposals.append(
            {
                "id": f"detection-{index}",
                "label": label,
                "box": box,
                "score": score,
                "source": {"detection_index": index, "raw_detection": deepcopy(detection)},
            }
        )
    result = normalize_output(
        {
            "protocol": OUTPUT_PROTOCOL,
            "taxonomy_id": contract["taxonomy_id"],
            "coordinates": {
                "format": "xyxy",
                "space": "original_pixels",
                "image_size": [width, height],
                "to_original": {"scale": [1, 1], "offset": [0, 0]},
            },
            "proposals": proposals,
        },
        contract["taxonomy"],
        width=width,
        height=height,
    )
    for proposal in result["proposals"]:
        detection = proposal["source"]["raw_detection"]
        proposal.update(
            source_index=proposal["source"]["detection_index"],
            original_label_id=detection["label_id"],
            original_label=detection["label"],
        )
    result.update(
        raw_output=deepcopy(detections),
        filtered_count=filtered,
        unmapped_count=unmapped,
        warnings=deepcopy(contract["warnings"]),
    )
    return result
