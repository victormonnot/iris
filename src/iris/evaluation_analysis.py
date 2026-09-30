"""Inspect saved operating-point errors without rerunning metrics or detectors."""

import math
import re

from iris.evaluation import (
    _timing_protocol,
    evaluation_detail,
    evaluation_inference,
    evaluation_lanes,
)
from iris.metrics import CLASS_IDS, _iou, _number, _validate, get_protocol
from iris.store import Store
from iris.tiling import _validate_tile_prediction, tile_boxes

PROTOCOL = "iris-error-analysis-v1"
RUN_PROTOCOL = "iris-error-analysis-v2"
FILTERS = ("all", "person", "car")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
WARNINGS = [
    "Counts use the saved confidence and IoU operating point, not COCO AP matching.",
    "New misses and recovered objects compare the same frozen ground-truth indices; "
    "they can both occur on one image even when the total miss count is unchanged.",
    "False-positive delta compares counts, not identities of false-positive objects.",
    "This analysis reads saved results and recorded hashes; it does not recheck image pixels "
    "or checkpoint files and does not establish generalization beyond this dataset.",
]


def _require(condition: bool, description: str):
    if not condition:
        raise ValueError(description)


def _count(value, description: str) -> int:
    _require(type(value) is int and value >= 0, f"Saved {description} must be a nonnegative count")
    return value


def _indices(values, limit: int, description: str) -> set[int]:
    _require(isinstance(values, list), f"Saved {description} must be an index list")
    _require(
        all(type(index) is int and 0 <= index < limit for index in values),
        f"Saved {description} contains an invalid source index",
    )
    _require(len(set(values)) == len(values), f"Saved {description} repeats a source index")
    return set(values)


def _recorded_protocol(config: dict) -> dict:
    confidence = _number(config["confidence_threshold"], "Saved confidence threshold")
    iou = _number(config["iou_threshold"], "Saved IoU threshold")
    max_detections = 300 if config.get("inference", {}).get("mode", "full") != "full" else 100
    supported = (
        get_protocol(confidence, iou, max_detections=max_detections)
        if max_detections != 100
        else get_protocol(confidence, iou)
    )
    recorded = config["protocol"]
    _require(isinstance(recorded, dict), "Saved evaluation protocol is missing")
    # Recorded outputs remain readable after dependency upgrades: no metric engine runs here.
    versions = {"engine_version", "numpy_version"}
    _require(
        {key: value for key, value in recorded.items() if key not in versions}
        == {key: value for key, value in supported.items() if key not in versions}
        and all(isinstance(recorded.get(key), str) and recorded[key] for key in versions),
        "Saved evaluation uses an unsupported or inconsistent matching protocol",
    )
    return recorded


def _validate_identity(detail: dict, evaluation_id: str) -> tuple[list[str], dict]:
    _require(detail["id"] == evaluation_id, "Saved evaluation identity is inconsistent")
    job = detail["job"]
    _require(
        isinstance(job, dict)
        and job["id"] == detail["job_id"]
        and job["kind"] == "evaluate"
        and job["params"].get("evaluation_id") == evaluation_id
        and job["status"] == "succeeded",
        "Evaluation must finish successfully before analyzing errors",
    )
    model_ids, config, frames = detail["model_ids"], detail["config"], detail["frames"]
    _require(
        isinstance(model_ids, list)
        and 1 <= len(model_ids) <= 2
        and all(isinstance(identifier, str) and identifier for identifier in model_ids)
        and len(set(model_ids)) == len(model_ids),
        "Saved evaluation requires one or two distinct ordered models",
    )
    _require(
        detail["split"] in {"val", "test"}
        and config["taxonomy_id"] == "iris-objects-v1"
        and config["class_mapping"] == CLASS_IDS,
        "Saved evaluation split or taxonomy is incompatible",
    )
    protocol = _recorded_protocol(config)
    _require(
        config["frame_ids"] == [frame["frame_id"] for frame in frames]
        and config["frame_hashes"] == {frame["frame_id"]: frame["sha256"] for frame in frames},
        "Saved evaluation frames or recorded hashes differ from the frozen split",
    )
    _require(
        isinstance(config["dataset_manifest_sha256"], str)
        and _SHA256.fullmatch(config["dataset_manifest_sha256"]) is not None
        and set(config["model_hashes"]) == set(model_ids)
        and set(config["model_names"]) == set(model_ids)
        and set(config["model_lineages"]) == set(model_ids),
        "Saved evaluation model or manifest identity is incomplete",
    )
    for identifier in model_ids:
        _require(
            isinstance(config["model_hashes"][identifier], str)
            and _SHA256.fullmatch(config["model_hashes"][identifier]) is not None
            and isinstance(config["model_names"][identifier], str)
            and bool(config["model_names"][identifier].strip()),
            "Saved model names or checkpoint hashes are invalid",
        )
    for frame in frames:
        annotation = frame["annotation"]
        _require(
            frame["split"] == detail["split"]
            and isinstance(frame["sha256"], str)
            and _SHA256.fullmatch(frame["sha256"]) is not None
            and annotation["status"] == "validated"
            and annotation["taxonomy_id"] == "iris-objects-v1"
            and annotation["frame_id"] == frame["frame_id"]
            and annotation["frame_sha256"] == frame["sha256"]
            and annotation["boxes"] == frame["boxes"]
            and annotation["id"] == frame["annotation_revision_id"]
            and type(annotation["revision"]) is int
            and annotation["revision"] == frame["revision"]
            and annotation["revision"] > 0
            and isinstance(annotation["reviewer"], str)
            and bool(annotation["reviewer"].strip()),
            "Frozen ground truth does not match its validated annotation revision",
        )
        _require(
            all(
                isinstance(value, str) and value
                for value in (
                    frame["source"]["filename"],
                    frame["scene_group"],
                    frame["session_id"],
                )
            ),
            "Frozen frame source identity is incomplete",
        )
    return model_ids, protocol


def _ordered_runs(detail: dict, model_ids: list[str]) -> tuple[list[dict], bool]:
    """Keep checkpoint identities distinct from each saved inference execution."""
    models, config = detail["models"], detail["config"]
    versioned = "inference" in config
    _require(isinstance(models, list), "Saved evaluation model runs must be a list")
    if not versioned:
        expected = [(identifier, "full") for identifier in model_ids]
    else:
        mode = evaluation_inference(detail)["mode"]
        expected = [(lane["model_id"], lane["variant"]) for lane in evaluation_lanes(detail)]
        _require(
            config.get("lanes")
            == [{"model_id": identifier, "variant": variant} for identifier, variant in expected],
            "Saved evaluation run order differs from its inference plan",
        )
        _require(
            config.get("timing_protocol")
            == _timing_protocol("full" if mode == "full" else "tiled"),
            "Saved evaluation timing protocol differs from its inference plan",
        )
    _require(
        len(models) == len(expected)
        and {(model["model_id"], model.get("variant", "full")) for model in models} == set(expected)
        and all(isinstance(model["id"], str) and model["id"] for model in models)
        and len({model["id"] for model in models}) == len(models),
        "Saved evaluation requires complete predictions and metrics for every model run",
    )
    by_identity = {(model["model_id"], model.get("variant", "full")): model for model in models}
    ordered = [by_identity[identity] for identity in expected]
    if versioned:
        _require(
            detail.get("lanes")
            == [
                {
                    "model_id": model["model_id"],
                    "variant": model["variant"],
                    "evaluation_model_id": model["id"],
                }
                for model in ordered
            ],
            "Saved evaluation lane identities are inconsistent",
        )
    by_id = {model["id"]: model for model in models}
    _require(
        all(
            prediction["evaluation_model_id"] in by_id
            and prediction["model_id"] == by_id[prediction["evaluation_model_id"]]["model_id"]
            for prediction in detail["predictions"]
        ),
        "Saved prediction references an unknown or inconsistent model run",
    )
    return ordered, versioned


def _run_inference(detail: dict, model: dict) -> dict:
    variant = model["variant"]
    inference = {"variant": variant}
    if variant == "tiled":
        config = detail["config"]["inference"]
        inference.update(
            algorithm=config["algorithm"],
            **config["tiling"],
            tile_boxes={
                frame["frame_id"]: tile_boxes(frame["width"], frame["height"], config["tiling"])
                for frame in detail["frames"]
            },
        )
    _require(
        model["metadata"].get("inference") == inference
        and model["metadata"].get("timing_protocol") == _timing_protocol(variant),
        "Saved model run differs from its recorded inference plan",
    )
    return inference


def _prediction_inference(prediction: dict, inference: dict):
    metadata = prediction["metadata"]
    _require(isinstance(metadata, dict), "Saved prediction metadata must be an object")
    if inference["variant"] == "full":
        _require("tiles" not in metadata, "Saved full-image prediction contains tiled results")
        return
    boxes, tiles = inference["tile_boxes"][prediction["frame_id"]], metadata.get("tiles")
    _require(
        isinstance(tiles, list) and len(tiles) == len(boxes),
        "Saved tiled prediction is missing its complete crop results",
    )
    sources = []
    for index, (box, tile) in enumerate(zip(boxes, tiles, strict=True)):
        _require(
            type(tile["tile_index"]) is int and tile["tile_index"] == index and tile["box"] == box,
            "Saved prediction crops differ from the recorded tile plan",
        )
        _validate_tile_prediction(tile, box[2] - box[0], box[3] - box[1])
        sources.append(
            [
                {
                    **detection,
                    "box": [
                        detection["box"][0] + box[0],
                        detection["box"][1] + box[1],
                        detection["box"][2] + box[0],
                        detection["box"][3] + box[1],
                    ],
                    "tile_index": index,
                }
                for detection in tile["detections"]
            ]
        )
    for detection in prediction["detections"]:
        index = detection.get("tile_index")
        _require(
            type(index) is int and 0 <= index < len(sources) and detection in sources[index],
            "Saved merged detection has no matching original-pixel crop result",
        )


def _model_results(detail: dict, model: dict, protocol: dict, *, versioned: bool) -> dict:
    identifier, config = model["model_id"], detail["config"]
    metadata, metrics = model["metadata"], model["metrics"]
    device = metadata.get("device")
    compatible_device = (config["device"] == "cpu" and device == "cpu") or (
        config["device"] == "cuda"
        and isinstance(device, str)
        and (device == "cuda" or re.fullmatch(r"cuda:[0-9]+", device) is not None)
    )
    _require(
        model["evaluation_id"] == detail["id"]
        and metadata["model_id"] == identifier
        and metadata["weight_sha256"] == config["model_hashes"][identifier]
        and metadata["model_name"] == config["model_names"][identifier]
        and metadata["protocol"] == protocol
        and metadata["lineage"] == config["model_lineages"][identifier]
        and compatible_device
        and isinstance(metrics, dict)
        and metrics["protocol"] == protocol,
        "Saved model metrics or checkpoint identity is incomplete or inconsistent",
    )
    inference = _run_inference(detail, model) if versioned else None
    predictions = [
        row for row in detail["predictions"] if row["evaluation_model_id"] == model["id"]
    ]
    for prediction in predictions:
        _require(
            prediction["evaluation_id"] == detail["id"] and prediction["model_id"] == identifier,
            "Saved prediction has an inconsistent evaluation or model identity",
        )
        if inference is not None:
            _prediction_inference(prediction, inference)
    if inference is not None and inference["variant"] == "full":
        _require(
            all(len(prediction["detections"]) <= 100 for prediction in predictions),
            "Saved full-image prediction exceeds the native detector output limit",
        )
    normalized, ignored = _validate(
        detail["frames"],
        predictions,
        max_detections=protocol.get("max_saved_detections_per_image", 100),
    )
    by_prediction = {row["frame_id"]: row for row in predictions}
    rows = metrics["frames"]
    _require(
        isinstance(rows, list) and [row["frame_id"] for row in rows] == config["frame_ids"],
        "Saved error rows must match every frozen frame in source order",
    )
    results = {}
    for frame, row in zip(normalized, rows, strict=True):
        _require(
            by_prediction[frame["frame_id"]]["input_size"] == [frame["width"], frame["height"]],
            "Saved prediction dimensions differ from the frozen image",
        )
        results[frame["frame_id"]] = _frame_results(frame, row, protocol)
    _validate_totals(metrics, results, normalized, predictions, ignored)
    return results


def _frame_results(frame: dict, row: dict, protocol: dict) -> dict:
    truth = frame["boxes"]
    retained = {
        item["index"]: item
        for item in frame["detections"]
        if item["score"] >= protocol["confidence_threshold"]
    }
    # Source indices retain gaps left by non-project classes and low-confidence detections.
    matches = row["matches"]
    _require(isinstance(matches, list), "Saved matches must be a list")
    false_negatives = _indices(row["false_negatives"], len(truth), "false negatives")
    false_positives = row["false_positives"]
    _require(
        isinstance(false_positives, list)
        and all(type(index) is int and index in retained for index in false_positives)
        and len(set(false_positives)) == len(false_positives),
        "Saved false positives must identify distinct retained source detections",
    )
    matched_truth, matched_detections = set(), set()
    counts = {label: {"tp": 0, "fp": 0, "fn": 0} for label in CLASS_IDS}
    for match in matches:
        target, detection = match["ground_truth_index"], match["detection_index"]
        _require(
            type(target) is int
            and 0 <= target < len(truth)
            and type(detection) is int
            and detection in retained
            and target not in matched_truth
            and detection not in matched_detections,
            "Saved match contains an invalid or reused source index",
        )
        label = truth[target]["label"]
        overlap = _number(match["iou"], "Saved match IoU")
        _require(
            match["label"] == label == retained[detection]["label"]
            and protocol["iou_threshold"] <= overlap <= 1
            and math.isclose(
                overlap, _iou(truth[target]["box"], retained[detection]["box"]), abs_tol=1e-12
            ),
            "Saved match class or IoU disagrees with its referenced source boxes",
        )
        matched_truth.add(target)
        matched_detections.add(detection)
        counts[label]["tp"] += 1
    _require(
        not matched_truth & false_negatives
        and matched_truth | false_negatives == set(range(len(truth)))
        and not matched_detections & set(false_positives)
        and matched_detections | set(false_positives) == set(retained),
        "Saved matches and errors do not partition frozen objects and retained detections",
    )
    for index in false_negatives:
        counts[truth[index]["label"]]["fn"] += 1
    for index in false_positives:
        counts[retained[index]["label"]]["fp"] += 1
    counts["all"] = {
        key: sum(counts[label][key] for label in CLASS_IDS) for key in ("tp", "fp", "fn")
    }
    _require(
        all(_count(row[key], key) == counts["all"][key] for key in ("tp", "fp", "fn")),
        "Saved frame error counts disagree with their source indices",
    )
    return {"counts": counts, "matched": matched_truth, "missed": false_negatives}


def _validate_totals(metrics, results, frames, predictions, ignored):
    totals = {
        label: {
            key: sum(result["counts"][label][key] for result in results.values())
            for key in ("tp", "fp", "fn")
        }
        for label in FILTERS
    }
    summary = metrics["summary"]
    expected = {
        **totals["all"],
        "ground_truth_count": totals["all"]["tp"] + totals["all"]["fn"],
        "prediction_count": totals["all"]["tp"] + totals["all"]["fp"],
        "frame_count": len(frames),
        "ignored_prediction_count": ignored,
        "native_prediction_count": sum(len(row["detections"]) for row in predictions),
        "project_prediction_count_before_threshold": sum(
            len(frame["detections"]) for frame in frames
        ),
    }
    _require(
        all(_count(summary[key], key) == value for key, value in expected.items()),
        "Saved summary counts disagree with the recorded frame errors",
    )
    classes = metrics["per_class"]
    _require(
        isinstance(classes, list)
        and len(classes) == len(CLASS_IDS)
        and {row["label"] for row in classes} == set(CLASS_IDS),
        "Saved per-class metrics are incomplete",
    )
    for row in classes:
        expected = totals[row["label"]]
        expected = {
            **expected,
            "support": expected["tp"] + expected["fn"],
            "predictions": expected["tp"] + expected["fp"],
        }
        _require(
            all(_count(row[key], key) == value for key, value in expected.items()),
            "Saved class counts disagree with the recorded frame errors",
        )


def _analyze(detail: dict, evaluation_id: str) -> dict:
    model_ids, protocol = _validate_identity(detail, evaluation_id)
    models, versioned = _ordered_runs(detail, model_ids)
    collection = "runs" if versioned else "models"
    identifiers = [model["id"] if versioned else model["model_id"] for model in models]
    results = {
        identifier: _model_results(detail, model, protocol, versioned=versioned)
        for identifier, model in zip(identifiers, models, strict=True)
    }
    identity = "run_id" if versioned else "model_id"
    comparison = (
        {f"baseline_{identity}": identifiers[0], f"candidate_{identity}": identifiers[1]}
        if len(identifiers) == 2
        else None
    )
    frames = []
    for position, frame in enumerate(detail["frames"], 1):
        identifier, truth = frame["frame_id"], frame["boxes"]
        counts = {}
        for label in FILTERS:
            indices = {
                index for index, box in enumerate(truth) if label == "all" or box["label"] == label
            }
            changes = None
            if comparison:
                baseline, candidate = (results[run_id][identifier] for run_id in identifiers)
                new = sorted(baseline["matched"] & candidate["missed"] & indices)
                recovered = sorted(baseline["missed"] & candidate["matched"] & indices)
                changes = {
                    "new_misses": len(new),
                    "recovered": len(recovered),
                    "fp_delta": candidate["counts"][label]["fp"] - baseline["counts"][label]["fp"],
                    "new_miss_indices": new,
                    "recovered_indices": recovered,
                }
            counts[label] = {
                "ground_truth_count": len(indices),
                collection: {
                    run_id: results[run_id][identifier]["counts"][label] for run_id in identifiers
                },
                "changes": changes,
            }
        frames.append(
            {
                "frame_id": identifier,
                "source_filename": frame["source"]["filename"],
                "scene_group": frame["scene_group"],
                "session_id": frame["session_id"],
                "width": frame["width"],
                "height": frame["height"],
                "position": position,
                "counts": counts,
            }
        )
    summary = {}
    for label in FILTERS:
        stats = [frame["counts"][label] for frame in frames]
        summary[label] = {
            "frame_count": len(frames),
            "ground_truth_count": sum(item["ground_truth_count"] for item in stats),
            collection: {
                run_id: {
                    **{
                        key: sum(item[collection][run_id][key] for item in stats)
                        for key in ("tp", "fp", "fn")
                    },
                    "error_frames": sum(
                        bool(item[collection][run_id]["fp"] or item[collection][run_id]["fn"])
                        for item in stats
                    ),
                }
                for run_id in identifiers
            },
            "changes": {
                key: sum(item["changes"][key] for item in stats)
                for key in ("new_misses", "recovered", "fp_delta")
            }
            if comparison
            else None,
        }
    warnings = [*WARNINGS, *detail["config"]["warnings"]]
    for model in models:
        warnings.extend(model["metrics"]["warnings"])
    _require(all(isinstance(warning, str) for warning in warnings), "Saved warnings are invalid")
    return {
        "evaluation_id": evaluation_id,
        "dataset_id": detail["dataset_id"],
        "split": detail["split"],
        "protocol": RUN_PROTOCOL if versioned else PROTOCOL,
        "confidence_threshold": protocol["confidence_threshold"],
        "iou_threshold": protocol["iou_threshold"],
        collection: [
            {
                "id": model["id"],
                "model_id": model["model_id"],
                "variant": model["variant"],
                "name": detail["config"]["model_names"][model["model_id"]]
                + (" · Full image" if model["variant"] == "full" else " · Tiled"),
            }
            for model in models
        ]
        if versioned
        else [
            {"id": identifier, "name": detail["config"]["model_names"][identifier]}
            for identifier in model_ids
        ],
        "comparison": comparison,
        "warnings": list(dict.fromkeys(warnings)),
        "summary": summary,
        "frames": frames,
    }


def analyze_evaluation(store: Store, evaluation_id: str) -> dict:
    """Read persisted operating-point matches and expose frame/class error changes."""
    try:
        detail = evaluation_detail(store, evaluation_id)
    except KeyError as exc:
        if exc.args == (evaluation_id,):
            raise
        raise ValueError("Saved evaluation is missing required data") from exc
    except (TypeError, IndexError, AttributeError) as exc:
        raise ValueError("Saved evaluation has incompatible data") from exc
    try:
        return _analyze(detail, evaluation_id)
    except (
        KeyError,
        TypeError,
        IndexError,
        AttributeError,
        ZeroDivisionError,
        OverflowError,
    ) as exc:
        raise ValueError("Saved evaluation is incomplete or contains inconsistent data") from exc
