"""Immutable experiment evidence with separately editable human interpretation."""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
import shutil
from collections import Counter
from copy import deepcopy
from pathlib import PurePosixPath

from PIL import Image

from iris.datasets import load_manifest
from iris.evaluation import evaluation_detail
from iris.evaluation_analysis import _analyze
from iris.store import Store, new_id, now
from iris.training import _read_training_image

MAX_EXAMPLES = 6
MAX_IMAGE_BYTES = 2 * 1024 * 1024
SNAPSHOT_VERSION = 1
_SAFE_ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
_CONFIG_FIELDS = (
    "confidence_threshold",
    "iou_threshold",
    "taxonomy_id",
    "device",
    "validation_evaluation_id",
    "dataset_manifest_sha256",
    "dataset_name",
    "warnings",
)
_TRAINING_CONFIG_FIELDS = (
    "steps",
    "learning_rate",
    "seed",
    "device",
    "scope",
    "scope_version",
    "trainable_modules",
    "batch_size",
    "optimizer",
    "momentum",
    "weight_decay",
    "dataset_manifest_sha256",
    "parent_weight_sha256",
    "taxonomy_id",
)
_TRAINING_METADATA_FIELDS = (
    "scope",
    "scope_version",
    "trainable_modules",
    "trainable_parameters",
    "frozen_parameters",
    "total_parameters",
    "changed_trainable_modules",
    "head_weights_changed",
    "frozen_parameters_unchanged",
    "frozen_batchnorm_buffers_unchanged",
    "model_buffers_unchanged",
    "head_initialization",
    "head_class_slots",
    "validation_consumed",
    "test_consumed",
)
_SUMMARY_FIELDS = (
    "map",
    "map50",
    "map75",
    "precision",
    "recall",
    "f1",
    "tp",
    "fp",
    "fn",
    "ground_truth_count",
    "prediction_count",
    "frame_count",
    "evaluated_classes",
    "ignored_prediction_count",
    "native_prediction_count",
    "project_prediction_count_before_threshold",
)
_CLASS_FIELDS = (
    "label",
    "ap",
    "ap50",
    "ap75",
    "precision",
    "recall",
    "f1",
    "tp",
    "fp",
    "fn",
    "support",
    "predictions",
)


class ExperimentConflict(RuntimeError):
    """Another edit changed the report revision before this update."""


def _pick(value: dict, keys) -> dict:
    if not isinstance(value, dict):
        raise ValueError("Saved experiment source contains an invalid object")
    result = {key: deepcopy(value[key]) for key in keys if key in value}

    def leaf(item):
        return (
            isinstance(item, (str, int, float, bool))
            or item is None
            or (isinstance(item, list) and all(leaf(child) for child in item))
        )

    if not all(leaf(item) for item in result.values()):
        raise ValueError("Saved experiment source contains an unexpected nested object")
    return result


def _canonical(value) -> bytes:
    try:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ValueError("Experiment evidence must contain finite JSON values") from exc


def _digest(value) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _text(value, field, maximum, *, required=False):
    if not isinstance(value, str) or len(value) > maximum or (required and not value.strip()):
        raise ValueError(
            f"{field} must contain {'1–' if required else 'at most '}{maximum} characters"
        )
    return value.strip()


def _editorial(title, objective, conclusion):
    return {
        "title": _text(title, "Title", 160, required=True),
        "objective": _text(objective, "Objective", 4000),
        "conclusion": _text(conclusion, "Conclusion", 4000),
    }


def _inference(config):
    value = config.get("inference", {"mode": "full"})
    result = _pick(value, ("mode", "algorithm"))
    if "tiling" in value:
        result["tiling"] = _pick(
            value["tiling"], ("tile_size", "overlap", "merge_iou", "max_detections")
        )
    return result


def _protocol(protocol):
    # The saved matching protocol has already passed _analyze's exact schema
    # check, including its recorded engine versions; it cannot carry extra keys.
    return deepcopy(protocol)


def _training_summary(store, model_id, expected_sha256=None):
    checkpoint = store.get("trained_models", model_id)
    if checkpoint is None:
        return None
    if expected_sha256 is not None and checkpoint["weight_sha256"] != expected_sha256:
        raise ValueError("Training lineage does not match its recorded checkpoint hash")
    run = store.get("training_runs", checkpoint["training_id"])
    if (
        run is None
        or run["checkpoint_id"] != model_id
        or run["parent_model_id"] != checkpoint["parent_model_id"]
    ):
        return None
    dataset = store.get("dataset_versions", run["dataset_id"])
    config = run["config"]
    if dataset is None or dataset["manifest_sha256"] != config.get("dataset_manifest_sha256"):
        return None
    history = run["history"]
    return {
        "id": run["id"],
        "name": run["name"],
        "dataset_id": run["dataset_id"],
        "dataset_name": dataset["name"],
        "dataset_manifest_sha256": dataset["manifest_sha256"],
        "parent_model_id": run["parent_model_id"],
        "parent_weight_sha256": config.get("parent_weight_sha256"),
        "created_at": run["created_at"],
        "config": _pick(config, _TRAINING_CONFIG_FIELDS),
        "metadata": _pick(run["metadata"], _TRAINING_METADATA_FIELDS),
        "history_summary": {
            "steps_completed": len(history),
            "first_loss": history[0].get("loss") if history else None,
            "last_loss": history[-1].get("loss") if history else None,
        },
    }


def _lineage(store, recorded, expected_sha256):
    if not isinstance(recorded, list) or len(recorded) > 64:
        raise ValueError("Saved model lineage is invalid")
    result = []
    for item in recorded:
        safe = _pick(
            item,
            (
                "model_id",
                "origin",
                "parent_model_id",
                "parent_weight_sha256",
                "training_scene_groups",
            ),
        )
        if "training_frame_hashes" in item:
            safe["training_frame_count"] = len(item["training_frame_hashes"])
        if item.get("origin") == "trained" and isinstance(item.get("model_id"), str):
            safe["training"] = _training_summary(store, item["model_id"], expected_sha256)
        result.append(safe)
        expected_sha256 = item.get("parent_weight_sha256")
    return result


def _runtime(metadata):
    result = _pick(
        metadata,
        ("device", "hardware", "precision", "torch_version", "torchvision_version", "threads"),
    )
    if "input_transform" in metadata:
        result["input_transform"] = _pick(
            metadata["input_transform"],
            (
                "color",
                "tensor_range",
                "exif_transpose",
                "image_mean",
                "image_std",
                "min_size",
                "max_size",
                "fixed_size",
                "size_divisible",
            ),
        )
    if "timing_protocol" in metadata:
        result["timing_protocol"] = _pick(
            metadata["timing_protocol"],
            (
                "id",
                "version",
                "preprocess_ms",
                "inference_ms",
                "postprocess_ms",
                "total_ms",
                "synchronization",
                "warmup",
                "warmup_frames",
                "warmup_in_timings",
                "decode_ms",
                "batch_size",
                "crop_ms",
                "merge_ms",
                "merge",
                "tile_layout",
                "device",
                "warmup_iterations",
                "scope",
                "synchronize",
                "description",
                "includes",
                "excludes",
            ),
        )
    return result


def _timing(predictions):
    values = []
    for row in predictions:
        value = row["timing"].get("total_ms")
        if value is None:
            continue
        if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
            raise ValueError("Saved prediction timing must be finite and nonnegative")
        values.append(float(value))
    complete = bool(values) and len(values) == len(predictions)
    return {
        "frame_count": len(predictions),
        "measured_frame_count": len(values),
        "mean_total_ms": sum(values) / len(values) if complete else None,
        "min_total_ms": min(values) if complete else None,
        "max_total_ms": max(values) if complete else None,
    }


def _metrics(metrics):
    result = {
        "protocol": _protocol(metrics["protocol"]),
        "summary": _pick(metrics["summary"], _SUMMARY_FIELDS),
        "per_class": [_pick(row, _CLASS_FIELDS) for row in metrics["per_class"]],
        "warnings": deepcopy(metrics["warnings"]),
    }
    for row in [result["summary"], *result["per_class"]]:
        for key in ("map", "map50", "map75", "ap", "ap50", "ap75", "precision", "recall", "f1"):
            if key in row and row[key] is not None:
                value = row[key]
                if (
                    type(value) not in (int, float)
                    or not math.isfinite(value)
                    or not 0 <= value <= 1
                ):
                    raise ValueError(
                        "Saved metric rates must be finite values between zero and one"
                    )
    return result


def _counts(counts, identifiers, *, summary=False):
    result = {}
    for label in ("all", "person", "car"):
        item = counts[label]
        source = item.get("runs", item.get("models"))
        result[label] = {
            **_pick(
                item, ("ground_truth_count", "frame_count") if summary else ("ground_truth_count",)
            ),
            "runs": {
                run_id: _pick(source[old_id], ("tp", "fp", "fn", "error_frames"))
                for old_id, run_id in identifiers.items()
            },
            "changes": _pick(
                item["changes"],
                ("new_misses", "recovered", "fp_delta", "new_miss_indices", "recovered_indices"),
            )
            if item["changes"] is not None
            else None,
        }
    return result


def _references(store, evaluation_id, captured_at):
    rows = store.list("model_references")
    return {
        "historical": True,
        "captured_at": captured_at,
        "current_reference_id": rows[-1]["id"] if rows else None,
        "decisions": [
            {
                **_pick(
                    row, ("id", "evaluation_id", "model_id", "reviewer", "notes", "created_at")
                ),
                **_pick(row["metadata"], ("variant", "evaluation_model_id", "model_name")),
            }
            for row in rows
            if row["evaluation_id"] == evaluation_id
        ],
    }


def _prepare(store, evaluation_id, captured_at):
    try:
        detail = evaluation_detail(store, evaluation_id)
        analysis = _analyze(detail, evaluation_id)
        models = {row["id"]: row for row in detail["models"]}
        versioned = "runs" in analysis
        identifiers = {
            lane["evaluation_model_id"] if versioned else lane["model_id"]: lane[
                "evaluation_model_id"
            ]
            for lane in detail["lanes"]
        }
        lanes = []
        for lane in detail["lanes"]:
            row = models[lane["evaluation_model_id"]]
            model_id = row["model_id"]
            history = detail["config"]["model_lineages"][model_id]
            checkpoint = store.get("trained_models", model_id)
            if (
                checkpoint is not None
                and checkpoint["weight_sha256"] != detail["config"]["model_hashes"][model_id]
            ):
                raise ValueError("Training provenance does not match the evaluated checkpoint hash")
            trained = _training_summary(store, model_id)
            status = (
                "recorded"
                if trained
                else "pretrained"
                if history and history[0].get("origin") == "official"
                else "unavailable"
            )
            lanes.append(
                {
                    "id": row["id"],
                    "model_id": model_id,
                    "variant": lane["variant"],
                    "name": detail["config"]["model_names"][model_id]
                    + (" · Tiled" if lane["variant"] == "tiled" else " · Full image"),
                    "weight_sha256": detail["config"]["model_hashes"][model_id],
                    "metrics": _metrics(row["metrics"]),
                    "timing": _timing(
                        [
                            prediction
                            for prediction in detail["predictions"]
                            if prediction["evaluation_model_id"] == row["id"]
                        ]
                    ),
                    "runtime": _runtime(row["metadata"]),
                    "training": trained,
                    "training_status": status,
                    "lineage": _lineage(store, history, detail["config"]["model_hashes"][model_id]),
                }
            )
        dataset = store.get("dataset_versions", detail["dataset_id"])
        summary = _pick(
            dataset["summary"],
            (
                "frame_count",
                "box_count",
                "negative_count",
                "scene_groups",
                "near_duplicate_cross_split_pairs",
                "warnings",
            ),
        )
        for key in ("split_counts", "class_counts", "split_class_counts"):
            if key in dataset["summary"]:
                keys = ("train", "val", "test") if key != "class_counts" else ("person", "car")
                if key == "split_class_counts":
                    summary[key] = {
                        split: _pick(dataset["summary"][key][split], ("person", "car"))
                        for split in keys
                        if split in dataset["summary"][key]
                    }
                else:
                    summary[key] = _pick(dataset["summary"][key], keys)
        config = _pick(detail["config"], _CONFIG_FIELDS)
        config.update(
            inference=_inference(detail["config"]), protocol=_protocol(detail["config"]["protocol"])
        )
        normalized_frames = {
            frame["frame_id"]: _counts(frame["counts"], identifiers) for frame in analysis["frames"]
        }
        sources = {}
        for frame in detail["frames"]:
            attribution = _attribution(frame["source"])
            if attribution:
                sources[_digest(attribution)] = attribution
        snapshot = {
            "version": SNAPSHOT_VERSION,
            "captured_at": captured_at,
            "evaluation": {
                **_pick(detail, ("id", "name", "dataset_id", "split", "created_at")),
                "config": config,
            },
            "dataset": {
                "id": dataset["id"],
                "name": dataset["name"],
                "manifest_sha256": dataset["manifest_sha256"],
                "summary": summary,
                "sources": list(sources.values()),
                "source_groups": [
                    {"scene_group": group, "frame_count": count}
                    for group, count in sorted(
                        Counter(frame["scene_group"] for frame in detail["frames"]).items()
                    )
                ],
            },
            "lanes": lanes,
            "error_analysis": {
                **_pick(
                    analysis, ("protocol", "confidence_threshold", "iou_threshold", "warnings")
                ),
                "comparison": {
                    "baseline_run_id": lanes[0]["id"],
                    "candidate_run_id": lanes[1]["id"],
                }
                if len(lanes) == 2
                else None,
                "summary": _counts(analysis["summary"], identifiers, summary=True),
            },
            "reference_decisions": _references(store, evaluation_id, captured_at),
            "examples": [],
        }
        available = [
            {
                "frame_id": frame["frame_id"],
                "source_filename": _filename(frame["source"]["filename"]),
                "scene_group": frame["scene_group"],
                "width": frame["width"],
                "height": frame["height"],
                "timestamp_seconds": frame["source"].get("timestamp_seconds"),
                "image_url": frame["image_url"],
                "counts": normalized_frames[frame["frame_id"]],
            }
            for frame in detail["frames"]
        ]
        _canonical(snapshot)
        _canonical(available)
        return snapshot, available, detail
    except KeyError as exc:
        if exc.args == (evaluation_id,):
            raise
        raise ValueError("Saved evaluation is missing required experiment evidence") from exc
    except (TypeError, IndexError, AttributeError, OverflowError, ZeroDivisionError) as exc:
        raise ValueError("Saved evaluation contains inconsistent experiment evidence") from exc


def _filename(value):
    return PurePosixPath(value.replace("\\", "/")).name


def _attribution(source):
    provenance = source.get("metadata", {}).get("dataset_import")
    if provenance is None:
        return None
    return _pick(provenance, ("source_url", "license_name", "attribution"))


def preview_experiment(store: Store, evaluation_id: str) -> dict:
    snapshot, examples, _ = _prepare(store, evaluation_id, now())
    return {"snapshot": snapshot, "available_examples": examples}


def _example(detail, frame, counts):
    result = {
        "frame_id": frame["frame_id"],
        "width": frame["width"],
        "height": frame["height"],
        "scene_group": frame["scene_group"],
        "source": {
            **_pick(frame["source"], ("timestamp_seconds", "frame_index")),
            "filename": _filename(frame["source"]["filename"]),
            "attribution": _attribution(frame["source"]),
        },
        "ground_truth": [_pick(box, ("id", "label", "box")) for box in frame["boxes"]],
        "counts": counts,
        "lanes": [],
    }
    for lane in detail["lanes"]:
        run_id = lane["evaluation_model_id"]
        model = next(row for row in detail["models"] if row["id"] == run_id)
        prediction = next(
            row
            for row in detail["predictions"]
            if row["evaluation_model_id"] == run_id and row["frame_id"] == frame["frame_id"]
        )
        errors = next(
            row for row in model["metrics"]["frames"] if row["frame_id"] == frame["frame_id"]
        )
        saved_errors = _pick(
            errors, ("frame_id", "tp", "fp", "fn", "false_positives", "false_negatives")
        )
        saved_errors["matches"] = [
            _pick(item, ("label", "ground_truth_index", "detection_index", "iou"))
            for item in errors["matches"]
        ]
        result["lanes"].append(
            {
                "run_id": run_id,
                "model_id": lane["model_id"],
                "variant": lane["variant"],
                "detections": [
                    _pick(
                        item, ("label", "label_id", "native_label_id", "box", "score", "tile_index")
                    )
                    for item in prediction["detections"]
                ],
                "errors": saved_errors,
            }
        )
    return result


def create_experiment(
    store: Store,
    *,
    evaluation_id: str,
    title: str,
    objective="",
    conclusion="",
    example_frame_ids=None,
) -> dict:
    editorial = _editorial(title, objective, conclusion)
    frame_ids = [] if example_frame_ids is None else example_frame_ids
    if (
        not isinstance(frame_ids, list)
        or len(frame_ids) > MAX_EXAMPLES
        or any(not isinstance(value, str) or not _SAFE_ID.fullmatch(value) for value in frame_ids)
        or len(set(frame_ids)) != len(frame_ids)
    ):
        raise ValueError("Choose at most six distinct evaluated example frames")
    captured = now()
    snapshot, available, detail = _prepare(store, evaluation_id, captured)
    source_hash = _digest({"snapshot": snapshot, "evidence": detail})
    candidates = {frame["frame_id"]: frame for frame in available}
    if not set(frame_ids) <= candidates.keys():
        raise ValueError("Examples must belong to this evaluated dataset split")
    manifest = load_manifest(store, detail["dataset_id"])
    frozen = {frame["frame_id"]: frame for frame in manifest["frames"]}
    frames = {frame["frame_id"]: frame for frame in detail["frames"]}
    report_id = new_id()
    directory = store.artifact_path(f"reports/{report_id}")
    images = []
    published = False
    try:
        if frame_ids:
            directory.mkdir(parents=True)
        for frame_id in frame_ids:
            with _read_training_image(store, frozen[frame_id]) as image:
                image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                image.info.clear()
                output = io.BytesIO()
                image.save(output, format="JPEG", quality=88)
                content = output.getvalue()
                if len(content) > MAX_IMAGE_BYTES:
                    raise ValueError("Experiment image exceeds the 2 MiB limit")
                path = store.artifact_path(f"reports/{report_id}/{frame_id}.jpg")
                path.write_bytes(content)
                item = {
                    "frame_id": frame_id,
                    "width": image.width,
                    "height": image.height,
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "size_bytes": len(content),
                    "path": path.relative_to(store.root).as_posix(),
                }
                images.append(item)
                example = _example(detail, frames[frame_id], candidates[frame_id]["counts"])
                example.update(
                    image_sha256=item["sha256"], image_width=image.width, image_height=image.height
                )
                snapshot["examples"].append(example)
        snapshot_hash = _digest(snapshot)
        with store.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            current, _, current_detail = _prepare(store, evaluation_id, captured)
            if _digest({"snapshot": current, "evidence": current_detail}) != source_hash:
                raise ValueError(
                    "Evaluation evidence changed while preparing the report; try again"
                )
            conn.execute(
                "INSERT INTO experiment_reports "
                "(id,evaluation_id,title,objective,conclusion,revision,snapshot,"
                "snapshot_sha256,images,created_at,updated_at) VALUES (?,?,?,?,?,1,?,?,?,?,?)",
                (
                    report_id,
                    evaluation_id,
                    editorial["title"],
                    editorial["objective"],
                    editorial["conclusion"],
                    _canonical(snapshot).decode(),
                    snapshot_hash,
                    _canonical(images).decode(),
                    captured,
                    captured,
                ),
            )
        published = True
        return experiment_detail(store, report_id)
    except BaseException as exc:
        if not published:
            shutil.rmtree(directory, ignore_errors=True)
        if isinstance(exc, OSError):
            raise ValueError("Experiment example image is missing or unreadable") from exc
        raise


def _verified_record(store, report_id):
    record = store.get("experiment_reports", report_id)
    if record is None:
        raise KeyError(report_id)
    snapshot = record["snapshot"]
    if (
        not isinstance(snapshot, dict)
        or type(snapshot.get("version")) is not int
        or snapshot.get("version") != SNAPSHOT_VERSION
        or _digest(snapshot) != record["snapshot_sha256"]
    ):
        raise ValueError("Experiment evidence no longer matches its saved hash")
    if type(record["revision"]) is not int or record["revision"] < 1:
        raise ValueError("Experiment revision must be a positive integer")
    if snapshot.get("evaluation", {}).get("id") != record["evaluation_id"]:
        raise ValueError("Experiment evaluation identity is inconsistent")
    examples, images = snapshot.get("examples"), record["images"]
    if (
        not isinstance(examples, list)
        or not isinstance(images, list)
        or len(examples) > MAX_EXAMPLES
        or len(examples) != len(images)
    ):
        raise ValueError("Experiment example images are inconsistent")
    identifiers = set()
    for example, image in zip(examples, images, strict=True):
        if not isinstance(example, dict) or not isinstance(image, dict):
            raise ValueError("Experiment image metadata must contain objects")
        frame_id = example.get("frame_id")
        if (
            not isinstance(frame_id, str)
            or not _SAFE_ID.fullmatch(frame_id)
            or frame_id in identifiers
            or image.get("frame_id") != frame_id
            or image.get("path") != f"reports/{report_id}/{frame_id}.jpg"
            or image.get("sha256") != example.get("image_sha256")
            or image.get("width") != example.get("image_width")
            or image.get("height") != example.get("image_height")
            or type(image.get("size_bytes")) is not int
            or not 0 < image["size_bytes"] <= MAX_IMAGE_BYTES
        ):
            raise ValueError("Experiment image identity is inconsistent")
        identifiers.add(frame_id)
    _editorial(record["title"], record["objective"], record["conclusion"])
    return record


def experiment_detail(store: Store, report_id: str) -> dict:
    record = _verified_record(store, report_id)
    return {
        **record,
        "images": [
            {
                **{key: value for key, value in image.items() if key != "path"},
                "url": f"/api/experiments/{report_id}/images/{image['frame_id']}",
            }
            for image in record["images"]
        ],
    }


def list_experiments(store: Store) -> list[dict]:
    result = []
    for row in reversed(store.list("experiment_reports")):
        record = _verified_record(store, row["id"])
        snapshot = record["snapshot"]
        result.append(
            {
                **_pick(
                    record,
                    (
                        "id",
                        "evaluation_id",
                        "title",
                        "objective",
                        "conclusion",
                        "revision",
                        "snapshot_sha256",
                        "created_at",
                        "updated_at",
                    ),
                ),
                "evaluation": _pick(snapshot["evaluation"], ("id", "name", "split")),
                "dataset": _pick(snapshot["dataset"], ("id", "name")),
                "lanes": [
                    _pick(lane, ("id", "model_id", "variant", "name")) for lane in snapshot["lanes"]
                ],
                "example_count": len(snapshot["examples"]),
            }
        )
    return result


def update_experiment(
    store: Store,
    report_id: str,
    *,
    expected_revision: int,
    title: str,
    objective: str,
    conclusion: str,
) -> dict:
    fields = _editorial(title, objective, conclusion)
    if type(expected_revision) is not int or expected_revision < 1:
        raise ValueError("expected_revision must be a positive integer")
    with store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        _verified_record(store, report_id)
        changed = conn.execute(
            "UPDATE experiment_reports SET title=?,objective=?,conclusion=?,"
            "revision=revision+1,updated_at=? WHERE id=? AND revision=?",
            (
                fields["title"],
                fields["objective"],
                fields["conclusion"],
                now(),
                report_id,
                expected_revision,
            ),
        ).rowcount
        if changed != 1:
            raise ExperimentConflict(
                "This experiment changed in another view; reload before saving"
            )
    return experiment_detail(store, report_id)


def read_experiment_image(store: Store, report_id: str, frame_id: str) -> bytes:
    record = _verified_record(store, report_id)
    item = next((image for image in record["images"] if image["frame_id"] == frame_id), None)
    if item is None:
        raise KeyError(frame_id)
    path = store.artifact_path(item["path"])
    try:
        if path.stat().st_size != item["size_bytes"]:
            raise ValueError("Experiment image no longer matches its saved size")
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != item["sha256"]:
            raise ValueError("Experiment image no longer matches its saved hash")
        with Image.open(io.BytesIO(content)) as image:
            if (
                image.format != "JPEG"
                or image.mode != "RGB"
                or image.size != (item["width"], item["height"])
                or not 0 < max(image.size) <= 1024
                or image.getexif()
                or image.info.get("icc_profile")
                or image.info.get("comment")
            ):
                raise ValueError("Experiment image encoding is invalid")
            image.verify()
        return content
    except OSError as exc:
        raise ValueError("Experiment image is missing or unreadable") from exc
