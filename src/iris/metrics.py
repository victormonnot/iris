"""Detection metrics for the reviewed IRIS taxonomy, using the official COCO evaluator.

AP consumes every saved native detection (before the UI confidence filter). The
confidence-specific counts are a separate, explicitly recorded operating point.
No crowd regions, ignored objects, masks or synthetic ground truth are inferred.
"""

from __future__ import annotations

import contextlib
import importlib.metadata
import io
import math

from iris.models import COCO_CATEGORIES

PROTOCOL_ID = "coco-bbox-iris-v1"
CLASS_IDS = {"person": 1, "car": 3}
MAX_DETECTIONS = 100


def _number(value, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{description} must be a finite number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{description} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{description} must be a finite number")
    return number


def _prediction_limit(value: int) -> int:
    if type(value) is not int or value not in (100, 300):
        raise ValueError("The saved prediction limit must be 100 or 300")
    return value


def get_protocol(
    confidence_threshold: float = 0.5,
    iou_threshold: float = 0.5,
    *,
    max_detections: int = MAX_DETECTIONS,
) -> dict:
    """Return the serializable metric definition, including the installed engine version."""
    confidence = _number(confidence_threshold, "Confidence threshold")
    iou = _number(iou_threshold, "IoU threshold")
    if not 0 <= confidence <= 1:
        raise ValueError("Confidence threshold must be between 0 and 1")
    if not 0 < iou <= 1:
        raise ValueError("IoU threshold must be greater than 0 and at most 1")
    limit = _prediction_limit(max_detections)
    protocol = {
        "id": PROTOCOL_ID,
        "engine": "pycocotools.COCOeval",
        "engine_version": importlib.metadata.version("pycocotools"),
        "numpy_version": importlib.metadata.version("numpy"),
        "task": "bbox",
        "taxonomy_id": "iris-objects-v1",
        "classes": dict(CLASS_IDS),
        "coordinate_format": "absolute pixel xyxy, continuous coordinates without +1",
        "ap_iou_thresholds": [round(0.5 + index * 0.05, 2) for index in range(10)],
        "ap_recall_thresholds": [index / 100 for index in range(101)],
        "area_range": "all (COCO default: 0 to 10000000000 square pixels)",
        "max_dets": [1, 10, MAX_DETECTIONS],
        "native_max_detections_per_image": MAX_DETECTIONS,
        "ap_confidence_filter": None,
        "confidence_threshold": confidence,
        "iou_threshold": iou,
        "operating_point": {
            "confidence_comparison": ">=",
            "iou_comparison": ">=",
            "matching": "greedy one-to-one by class, descending confidence",
            "score_ties": "original detection order within each image",
            "iou_ties": "last still-unmatched ground-truth box in manifest order",
            "aggregation": "micro across frames and the two project classes",
        },
        "ap_score_ties": "manifest frame order, then original detection order (stable COCO sort)",
        "empty_ground_truth_class_ap": "null; excluded from macro AP",
        "zero_denominator_precision_or_recall": None,
        "limitations": [
            "Only person and car are evaluated; other valid COCO classes are counted as ignored.",
            "All reviewed boxes are ordinary non-crowd objects; no ignored areas are supported.",
            "AP uses native detector outputs after its score floor, NMS and cap of 100 "
            "detections per image; "
            "it cannot recover discarded detections.",
            "These are full-image metrics; no object-size breakdown or uncertainty estimate.",
        ],
    }
    if limit == 300:
        # Keep the historical definition byte-for-byte equivalent for old runs.
        # The accepted pipeline output grows; the COCO AP maxDets stays 100.
        protocol.pop("native_max_detections_per_image")
        protocol.update(
            id="coco-bbox-iris-v2",
            max_saved_detections_per_image=limit,
            native_max_detections_per_call=MAX_DETECTIONS,
            limitations=[
                *protocol["limitations"][:2],
                "Each run scores its final original-image predictions, after detector filtering "
                "and any recorded tile merging; discarded predictions cannot be recovered.",
                "COCO AP keeps maxDets=100. Operating-point counts use every saved prediction "
                "above threshold, including merged outputs beyond that AP limit.",
                "Metrics have no object-size breakdown or uncertainty estimate.",
            ],
        )
    return protocol


def _box(value, width: int, height: int) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("A box must contain four xyxy coordinates")
    x1, y1, x2, y2 = [_number(item, "Box coordinate") for item in value]
    if not 0 <= x1 < x2 <= width or not 0 <= y1 < y2 <= height:
        raise ValueError("A box must have positive area and lie within its image")
    return [x1, y1, x2, y2]


def _validate(
    frames: list[dict],
    predictions: list[dict],
    *,
    max_detections: int = MAX_DETECTIONS,
) -> tuple[list[dict], int]:
    limit = _prediction_limit(max_detections)
    if not isinstance(frames, list) or not frames:
        raise ValueError("Evaluation requires at least one reviewed frame")
    normalized, seen = [], set()
    for frame in frames:
        if not isinstance(frame, dict):
            raise ValueError("Each evaluation frame must be an object")
        identifier = frame.get("frame_id")
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError("Evaluation frame IDs must be nonempty and distinct")
        seen.add(identifier)
        width, height = frame.get("width"), frame.get("height")
        if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
            raise ValueError("Evaluation image dimensions must be positive integers")
        if not isinstance(frame.get("boxes"), list):
            raise ValueError("Every evaluation frame must have a reviewed boxes list")
        boxes = []
        for item in frame["boxes"]:
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("label"), str)
                or item["label"] not in CLASS_IDS
            ):
                raise ValueError("Ground truth must use the person/car project taxonomy")
            if item.get("iscrowd", 0) != 0 or item.get("ignore", 0) != 0:
                raise ValueError("Crowd and ignored ground-truth boxes are unsupported")
            boxes.append({"label": item["label"], "box": _box(item.get("box"), width, height)})
        normalized.append(
            {"frame_id": identifier, "width": width, "height": height, "boxes": boxes}
        )
    if not isinstance(predictions, list):
        raise ValueError("Predictions must contain exactly one row for every evaluation frame")
    rows = {}
    for row in predictions:
        if not isinstance(row, dict):
            raise ValueError("Each prediction row must be an object")
        identifier = row.get("frame_id")
        if not isinstance(identifier, str) or identifier not in seen or identifier in rows:
            raise ValueError("Prediction frame IDs must match the evaluation frames exactly once")
        rows[identifier] = row
    if set(rows) != seen:
        raise ValueError("Predictions are missing evaluation frames, including explicit empty rows")
    ignored = 0
    for frame in normalized:
        raw = rows[frame["frame_id"]].get("detections")
        if not isinstance(raw, list) or len(raw) > limit:
            raise ValueError(f"Every frame must contain a list of at most {limit} saved detections")
        frame["detections"] = []
        for index, item in enumerate(raw):
            if not isinstance(item, dict):
                raise ValueError("Every detection must be an object")
            category = item.get("label_id")
            if (
                type(category) is not int
                or not 0 < category < len(COCO_CATEGORIES)
                or COCO_CATEGORIES[category] == "N/A"
                or item.get("label") != COCO_CATEGORIES[category]
            ):
                raise ValueError("Detection class ID/name must use the canonical COCO taxonomy")
            box = _box(item.get("box"), frame["width"], frame["height"])
            score = _number(item.get("score"), "Detection score")
            if not 0 <= score <= 1:
                raise ValueError("Detection scores must be between 0 and 1")
            if category not in CLASS_IDS.values():
                ignored += 1
                continue
            frame["detections"].append(
                {
                    "index": index,
                    "label_id": category,
                    "label": item["label"],
                    "box": box,
                    "score": score,
                }
            )
    return normalized, ignored


def _xywh(box: list[float]) -> list[float]:
    return [box[0], box[1], box[2] - box[0], box[3] - box[1]]


def _average_precision(frames: list[dict]) -> dict:
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    images, ground_truth, detections = [], [], []
    categories = [{"id": identifier, "name": label} for label, identifier in CLASS_IDS.items()]
    for image_id, frame in enumerate(frames, 1):
        images.append({"id": image_id, "width": frame["width"], "height": frame["height"]})
        for target, source in ((ground_truth, frame["boxes"]), (detections, frame["detections"])):
            for item in source:
                box = _xywh(item["box"])
                annotation = {
                    "id": len(target) + 1,
                    "image_id": image_id,
                    "category_id": CLASS_IDS[item["label"]],
                    "bbox": box,
                    "area": box[2] * box[3],
                    "iscrowd": 0,
                }
                if "score" in item:
                    annotation["score"] = item["score"]
                target.append(annotation)
    # Building the result COCO directly also handles an entirely empty result set;
    # COCO.loadRes assumes at least one result in several pycocotools releases.
    with contextlib.redirect_stdout(io.StringIO()):
        reference, predicted = COCO(), COCO()
        for obj, annotations in ((reference, ground_truth), (predicted, detections)):
            obj.dataset = {"images": images, "categories": categories, "annotations": annotations}
            obj.createIndex()
        evaluator = COCOeval(reference, predicted, "bbox")
        evaluator.params.imgIds = [item["id"] for item in images]
        evaluator.params.catIds = list(CLASS_IDS.values())
        evaluator.params.iouThrs = np.linspace(0.5, 0.95, 10)
        evaluator.params.recThrs = np.linspace(0.0, 1.0, 101)
        evaluator.params.areaRng = [[0, 1e10]]
        evaluator.params.areaRngLbl = ["all"]
        evaluator.params.maxDets = [1, 10, MAX_DETECTIONS]
        evaluator.evaluate()
        evaluator.accumulate()
    precision = evaluator.eval["precision"][:, :, :, 0, 2]

    def mean(values):
        valid = values[values >= 0]
        return float(valid.mean()) if valid.size else None

    result = {
        "map": mean(precision),
        "map50": mean(precision[0]),
        "map75": mean(precision[5]),
        "per_class": {},
    }
    for index, label in enumerate(CLASS_IDS):
        values = precision[:, :, index]
        result["per_class"][label] = {
            "ap": mean(values),
            "ap50": mean(values[0]),
            "ap75": mean(values[5]),
        }
    return result


def _iou(first: list[float], second: list[float]) -> float:
    intersection = max(0.0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0.0, min(first[3], second[3]) - max(first[1], second[1])
    )
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / (first_area + second_area - intersection)


def _rates(tp: int, fp: int, fn: int) -> dict:
    return {
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def evaluate_predictions(
    frames: list[dict],
    predictions: list[dict],
    *,
    confidence_threshold: float = 0.5,
    iou_threshold: float = 0.5,
    max_detections: int = MAX_DETECTIONS,
) -> dict:
    """Score complete saved predictions against a frozen, reviewed frame list.

    Frame and detection order are preserved, making equal-score results repeatable
    and error-example indices traceable to the unchanged source artifacts.
    """
    protocol = get_protocol(confidence_threshold, iou_threshold, max_detections=max_detections)
    normalized, ignored = _validate(frames, predictions, max_detections=max_detections)
    confidence, threshold = protocol["confidence_threshold"], protocol["iou_threshold"]
    ap = _average_precision(normalized)
    totals = {label: {"tp": 0, "fp": 0, "fn": 0, "support": 0} for label in CLASS_IDS}
    details = []
    for frame in normalized:
        matched, matches, false_positives = set(), [], []
        for item in frame["boxes"]:
            totals[item["label"]]["support"] += 1
        ranked = sorted(frame["detections"], key=lambda item: -item["score"])
        for detection in ranked:
            if detection["score"] < confidence:
                continue
            best, best_iou = None, threshold
            for index, target in enumerate(frame["boxes"]):
                if index in matched or target["label"] != detection["label"]:
                    continue
                overlap = _iou(detection["box"], target["box"])
                if overlap >= best_iou:
                    best, best_iou = index, overlap
            if best is None:
                false_positives.append(detection["index"])
                totals[detection["label"]]["fp"] += 1
            else:
                matched.add(best)
                totals[detection["label"]]["tp"] += 1
                matches.append(
                    {
                        "label": detection["label"],
                        "ground_truth_index": best,
                        "detection_index": detection["index"],
                        "iou": best_iou,
                    }
                )
        false_negatives = [index for index in range(len(frame["boxes"])) if index not in matched]
        for index in false_negatives:
            totals[frame["boxes"][index]["label"]]["fn"] += 1
        details.append(
            {
                "frame_id": frame["frame_id"],
                "tp": len(matches),
                "fp": len(false_positives),
                "fn": len(false_negatives),
                "matches": matches,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
            }
        )
    per_class = [
        {
            "label": label,
            **ap["per_class"][label],
            **_rates(counts["tp"], counts["fp"], counts["fn"]),
            "support": counts["support"],
            "predictions": counts["tp"] + counts["fp"],
        }
        for label, counts in totals.items()
    ]
    tp, fp, fn = (sum(counts[key] for counts in totals.values()) for key in ("tp", "fp", "fn"))
    evaluated_classes = [label for label, counts in totals.items() if counts["support"]]
    warnings = []
    if ignored:
        warnings.append(f"Ignored {ignored} detections from valid COCO classes outside person/car.")
    if not evaluated_classes:
        warnings.append("No ground-truth objects: AP and recall are undefined, not perfect scores.")
    elif len(evaluated_classes) < len(CLASS_IDS):
        warnings.append("Classes without ground-truth objects are excluded from macro AP.")
    return {
        "protocol": protocol,
        "summary": {
            **{key: ap[key] for key in ("map", "map50", "map75")},
            **_rates(tp, fp, fn),
            "ground_truth_count": tp + fn,
            "prediction_count": tp + fp,
            "frame_count": len(normalized),
            "evaluated_classes": evaluated_classes,
            "ignored_prediction_count": ignored,
            "native_prediction_count": sum(len(row["detections"]) for row in predictions),
            "project_prediction_count_before_threshold": sum(
                len(frame["detections"]) for frame in normalized
            ),
        },
        "per_class": per_class,
        "frames": details,
        "warnings": warnings,
    }
