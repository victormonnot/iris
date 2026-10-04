"""Read-only human review progress and disagreement between saved detector outputs."""

from iris.annotations import COCO_MAPPING, TAXONOMY, _coordinates, _finite_number, _latest
from iris.datasets import _reservations
from iris.inference import comparison_lanes
from iris.models import COCO_CATEGORIES, get_spec
from iris.prediction_taxonomy import annotation_mapping, validate_output_labels
from iris.review_hints import review_hints
from iris.store import Store, _decode
from iris.taxonomies import _get

PROTOCOL = "iris-review-disagreement-v2"
MATCHING = (
    "Maximum-cardinality same-class matching at the chosen IoU; deterministic augmenting "
    "paths visit original left indices, then neighbors by descending IoU and right index. "
    "This does not maximize summed IoU. Remaining different-class overlaps are informative "
    "conflicts and remain unmatched."
)
WARNINGS = [
    "Disagreement is a review ordering aid, not an accuracy or uncertainty estimate. "
    "Confidence scores are not calibrated across models.",
    "Agreement and no detections can both hide missed objects. Review the whole image; "
    "neither result validates an annotation.",
    "Only saved predictions mapped to this image's class definitions above the chosen "
    "confidence threshold are compared. Both sources must cover every target class. "
    "Each detector's native filtering and detection limits already apply.",
    "Keep reserved train, validation and test splits. Prioritizing a review never changes "
    "selection, annotations or split assignments.",
]
_COCO_IDS = {index for index, label in enumerate(COCO_CATEGORIES) if index and label != "N/A"}


def _thresholds(confidence_threshold, iou_threshold):
    if not _finite_number(confidence_threshold) or not 0 <= confidence_threshold <= 1:
        raise ValueError("Confidence threshold must be between 0 and 1")
    if not _finite_number(iou_threshold) or not 0 < iou_threshold <= 1:
        raise ValueError("IoU threshold must be greater than 0 and at most 1")


def _filtered(detections, width, height, threshold, mapping=None, limit=100):
    if not isinstance(detections, list) or len(detections) > limit:
        raise ValueError(f"Saved detections must be a list of at most {limit} native outputs")
    result = []
    for index, detection in enumerate(detections):
        if not isinstance(detection, dict):
            raise ValueError("Each saved detection must be an object")
        category, score = detection.get("label_id"), detection.get("score")
        if type(category) is not int or (mapping is None and category not in _COCO_IDS):
            raise ValueError("Saved detections must use canonical COCO category IDs")
        if not _finite_number(score) or not 0 <= score <= 1:
            raise ValueError("Saved detection confidence must be between 0 and 1")
        box = _coordinates(detection.get("box"), {"width": width, "height": height})
        area = (box[2] - box[0]) * (box[3] - box[1])
        if not _finite_number(area) or area <= 0:
            raise ValueError("Saved detection box area must be finite and positive")
        resolved = COCO_MAPPING if mapping is None else mapping
        if category in resolved and score >= threshold:
            result.append({"index": index, "category": resolved[category], "box": box})
    return result


def _iou(left, right):
    intersection = max(0.0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0.0, min(left[3], right[3]) - max(left[1], right[1])
    )
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    scale = max(left_area, right_area)
    normalized_intersection = intersection / scale
    return normalized_intersection / (
        left_area / scale + right_area / scale - normalized_intersection
    )


def _match(left, right, threshold, *, same_class):
    neighbors = {}
    overlaps = {}
    for left_box in left:
        candidates = []
        for right_box in right:
            if (left_box["category"] == right_box["category"]) != same_class:
                continue
            overlap = _iou(left_box["box"], right_box["box"])
            if overlap >= threshold:
                candidates.append((-overlap, right_box["index"]))
                overlaps[left_box["index"], right_box["index"]] = overlap
        neighbors[left_box["index"]] = [index for _, index in sorted(candidates)]
    assigned_right = {}

    def augment(left_index, visited):
        for right_index in neighbors[left_index]:
            if right_index in visited:
                continue
            visited.add(right_index)
            previous = assigned_right.get(right_index)
            if previous is None or augment(previous, visited):
                assigned_right[right_index] = left_index
                return True
        return False

    for left_index in sorted(neighbors):
        augment(left_index, set())
    pairs = sorted((left_index, right_index) for right_index, left_index in assigned_right.items())
    matches = [
        {
            "left_index": left_index,
            "right_index": right_index,
            "iou": overlaps[left_index, right_index],
        }
        for left_index, right_index in pairs
    ]
    return matches, set(assigned_right.values()), set(assigned_right)


def assess_disagreement(
    left_detections,
    right_detections,
    width,
    height,
    confidence_threshold=0.5,
    iou_threshold=0.5,
    *,
    mappings=None,
    limits=(100, 100),
) -> dict:
    """Match canonical person/car outputs without using any reference annotations.

    Maximum-cardinality pairs keep the unmatched ratio invariant to model order.
    Conflicting classes remain unmatched even when their geometry overlaps.
    """
    _thresholds(confidence_threshold, iou_threshold)
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive integers")
    mappings = mappings or (None, None)
    left = _filtered(left_detections, width, height, confidence_threshold, mappings[0], limits[0])
    right = _filtered(right_detections, width, height, confidence_threshold, mappings[1], limits[1])
    matches, matched_left, matched_right = _match(left, right, iou_threshold, same_class=True)
    unmatched_left = [item for item in left if item["index"] not in matched_left]
    unmatched_right = [item for item in right if item["index"] not in matched_right]
    conflicts, _, _ = _match(unmatched_left, unmatched_right, iou_threshold, same_class=False)
    total = len(left) + len(right)
    unmatched = len(unmatched_left) + len(unmatched_right)
    if not total:
        status = "no_detections"
        reason = "Neither model has a saved target-class detection above this threshold."
    elif unmatched:
        status = "disagreement"
        reason = f"{unmatched} of {total} detections have no same-class match at this IoU."
    else:
        status = "agreement"
        reason = "All retained detections have a same-class match at this IoU."
    return {
        "status": status,
        "reason": reason,
        "disagreement": unmatched / total if total else None,
        "counts": [len(left), len(right)],
        "matched_count": len(matches),
        "unmatched_counts": [len(unmatched_left), len(unmatched_right)],
        "class_conflicts": len(conflicts),
        "prediction_ids": [],
        "matches": matches,
        "unmatched_indices": [
            [item["index"] for item in unmatched_left],
            [item["index"] for item in unmatched_right],
        ],
        "class_conflict_pairs": conflicts,
    }


def _unavailable(reason):
    return {
        "status": "unavailable",
        "reason": reason,
        "disagreement": None,
        "counts": None,
        "matched_count": None,
        "unmatched_counts": None,
        "class_conflicts": None,
        "prediction_ids": [],
        "matches": [],
        "unmatched_indices": [[], []],
        "class_conflict_pairs": [],
    }


def _comparison(conn, session_id, comparison_id):
    comparison = _decode(
        conn.execute("SELECT * FROM comparisons WHERE id=?", (comparison_id,)).fetchone()
    )
    if comparison is None:
        raise KeyError(comparison_id)
    if comparison["session_id"] != session_id:
        raise ValueError("Choose a comparison from this session")
    job = conn.execute("SELECT status FROM jobs WHERE id=?", (comparison["job_id"],)).fetchone()
    if job is None or job["status"] != "succeeded":
        raise ValueError("Choose a completed comparison")
    try:
        lanes = comparison_lanes(comparison)
    except (ValueError, TypeError, KeyError) as exc:
        raise ValueError("Choose a comparison of two distinct model/inference runs") from exc
    if len(lanes) != 2:
        raise ValueError("Choose a comparison of two distinct model/inference runs")
    if not isinstance(comparison["config"], dict) or (
        comparison["config"].get("taxonomy") not in {"coco-2017-v1", "model-specific-v1"}
    ):
        raise ValueError("The comparison must use COCO or frozen model-specific class definitions")
    if not isinstance(comparison["frame_ids"], list) or not all(
        isinstance(frame_id, str) for frame_id in comparison["frame_ids"]
    ):
        raise ValueError("The comparison has an invalid frame selection")
    runs = {}
    for row in conn.execute("SELECT * FROM runs WHERE comparison_id=?", (comparison_id,)):
        run = _decode(row)
        runs.setdefault((run["model_id"], run.get("variant", "full")), []).append(run)
    names = []
    for lane in lanes:
        model_id = lane["model_id"]
        trained = conn.execute("SELECT name FROM trained_models WHERE id=?", (model_id,)).fetchone()
        if trained:
            name = trained["name"]
        else:
            try:
                name = get_spec(model_id)["name"]
            except ValueError:
                name = model_id
        if lane["variant"] == "tiled" or len(comparison["model_ids"]) == 1:
            name += " · " + ("Tiled" if lane["variant"] == "tiled" else "Full image")
        names.append({"id": model_id, "name": name})
    summary = {
        **{key: comparison[key] for key in ("id", "name", "model_ids", "created_at")},
        "models": names,
        "lanes": [
            {
                **lane,
                "run_id": matches[0]["id"] if len(matches) == 1 else None,
            }
            for lane in lanes
            for matches in [runs.get((lane["model_id"], lane["variant"]), [])]
        ],
    }
    predictions = {}
    for row in conn.execute(
        "SELECT * FROM predictions WHERE comparison_id=? ORDER BY created_at,id", (comparison_id,)
    ):
        prediction = _decode(row)
        predictions.setdefault((prediction["frame_id"], prediction["run_id"]), []).append(
            prediction
        )
    return comparison, summary, runs, predictions


def _frame_signal(frame, comparison, runs, predictions, confidence, iou, taxonomy):
    if frame["id"] not in comparison["frame_ids"]:
        return _unavailable("This selected image was not included in the saved comparison.")
    hashes = comparison["config"].get("frame_hashes")
    if not isinstance(hashes, dict) or hashes.get(frame["id"]) != frame["sha256"]:
        return _unavailable("The image hash does not match the saved comparison input.")
    expected_runs = {
        run["id"]
        for lane in comparison_lanes(comparison)
        for run in runs.get((lane["model_id"], lane["variant"]), [])
    }
    if any(
        saved_frame_id == frame["id"] and run_id not in expected_runs
        for saved_frame_id, run_id in predictions
    ):
        return _unavailable("The saved prediction does not match its model run.")
    pair, mappings, limits = [], [], []
    for lane in comparison_lanes(comparison):
        model_id, variant = lane["model_id"], lane["variant"]
        matches = runs.get((model_id, variant), [])
        run = matches[0] if len(matches) == 1 else None
        records = predictions.get((frame["id"], run["id"]), []) if run else []
        if run is None or len(records) != 1:
            return _unavailable("A saved model run or prediction is missing or ambiguous.")
        prediction = records[0]
        if prediction["run_id"] != run["id"] or prediction["model_id"] != run["model_id"]:
            return _unavailable("The saved prediction does not match its model run.")
        if prediction["input_size"] != [frame["width"], frame["height"]]:
            return _unavailable("The saved prediction dimensions do not match the image.")
        metadata = run["metadata"]
        if not isinstance(metadata, dict) or metadata.get("model_id", model_id) != model_id:
            return _unavailable("The saved run metadata does not match its model.")
        inference = metadata.get("inference", {})
        if not isinstance(inference, dict) or inference.get("variant", variant) != variant:
            return _unavailable("The saved run metadata does not match its inference mode.")
        try:
            mapping, _ = annotation_mapping(comparison, run, taxonomy)
            contracts = comparison["config"].get("model_class_contracts")
            if contracts is not None:
                validate_output_labels(prediction["detections"], contracts[model_id])
            else:
                source_ids = _COCO_IDS
                if any(
                    key in metadata
                    for key in ("training_id", "taxonomy_id", "taxonomy", "output_class_mapping")
                ):
                    from iris.model_taxonomy import class_contract

                    legacy = class_contract(metadata)
                    if legacy["taxonomy_id"] != TAXONOMY["id"]:
                        raise ValueError("Custom trained outputs require a frozen model contract")
                    source_ids = set(legacy["output_class_mapping"].values())
                    mapping = {key: value for key, value in mapping.items() if key in source_ids}
                if any(
                    type(item.get("label_id")) is not int or item["label_id"] not in source_ids
                    for item in prediction["detections"]
                ):
                    raise ValueError(
                        "Saved detection IDs do not match their source class namespace"
                    )
            receipt = prediction.get("metadata", {}).get("preannotation")
            if ("preannotation" in comparison["config"] or receipt is not None) and (
                not isinstance(receipt, dict)
                or receipt.get("state") not in {"pending_review", "no_proposals"}
            ):
                raise ValueError("Saved preannotation output has no valid publication receipt")
            if set(mapping.values()) != {item["id"] for item in taxonomy["classes"]}:
                return _unavailable(
                    "Both detectors must cover every saved class. Some custom class definitions "
                    "have no compatible output; inspect the image manually."
                )
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            return _unavailable(f"Saved class definitions cannot be compared: {exc}")
        mappings.append(mapping)
        limits.append(300 if variant == "tiled" else 100)
        pair.append(prediction)
    try:
        signal = assess_disagreement(
            pair[0]["detections"],
            pair[1]["detections"],
            frame["width"],
            frame["height"],
            confidence,
            iou,
            mappings=mappings,
            limits=limits,
        )
    except ValueError as exc:
        return _unavailable(f"Saved detections cannot be compared: {exc}")
    signal["prediction_ids"] = [prediction["id"] for prediction in pair]
    return signal


def review_queue(
    store: Store,
    session_id: str,
    *,
    comparison_id: str | None = None,
    confidence_threshold: float = 0.5,
    iou_threshold: float = 0.5,
) -> dict:
    """Read all selected frames in source order from one SQLite snapshot.

    Saved detector geometry supplies an optional review signal. Annotation state
    supplies progress only; labels are never used to calculate disagreement.
    """
    _thresholds(confidence_threshold, iou_threshold)
    counts = dict.fromkeys(
        ("total", "needs_review", "unannotated", "draft", "pending", "validated"), 0
    )
    frames, summary = [], None
    with store.connect() as conn:
        conn.execute("BEGIN")
        session = conn.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        if session is None:
            raise KeyError(session_id)
        project_taxonomy_id = conn.execute(
            "SELECT taxonomy_id FROM projects WHERE id=?", (session["project_id"],)
        ).fetchone()["taxonomy_id"]
        saved = _comparison(conn, session_id, comparison_id) if comparison_id is not None else None
        if saved:
            comparison, summary, runs, predictions = saved
        reserved_groups, reserved_pixels = _reservations(store, conn, session["project_id"])
        for row in conn.execute(
            "SELECT f.*, a.filename AS source_filename FROM frames f "
            "JOIN assets a ON a.id=f.asset_id WHERE f.session_id=? AND f.selected=1 "
            "ORDER BY f.created_at,f.id",
            (session_id,),
        ):
            frame = _decode(row)
            latest = _latest(conn, frame["id"])
            taxonomy_id = latest["taxonomy_id"] if latest else frame["taxonomy_id"]
            decisions = latest["decisions"] if latest else {}
            suggestion_records = [
                _decode(row)
                for row in conn.execute(
                    "SELECT * FROM annotation_suggestions WHERE frame_id=?", (frame["id"],)
                )
            ]
            suggestions = {item["id"] for item in suggestion_records}
            pending = len(suggestions - decisions.keys())
            annotation_status = latest["status"] if latest else "unannotated"
            status = "pending" if pending else annotation_status
            counts[status] += 1
            counts["total"] += 1
            counts["needs_review"] += status != "validated"
            group_split = reserved_groups.get(session["scene_group"])
            pixel_split = reserved_pixels.get(frame["sha256"])
            if group_split and pixel_split and group_split != pixel_split:
                raise ValueError("The image and scene group have conflicting reserved splits")
            taxonomy = _get(conn, taxonomy_id, session["project_id"])
            signal = (
                _frame_signal(
                    frame,
                    comparison,
                    runs,
                    predictions,
                    confidence_threshold,
                    iou_threshold,
                    taxonomy,
                )
                if saved
                else _unavailable("Choose a completed two-run comparison for a review signal.")
            )
            frames.append(
                {
                    **{
                        key: frame[key]
                        for key in (
                            "id",
                            "session_id",
                            "asset_id",
                            "source_filename",
                            "frame_index",
                            "timestamp_seconds",
                            "width",
                            "height",
                            "sha256",
                        )
                    },
                    "review_status": status,
                    "taxonomy_id": taxonomy_id,
                    "taxonomy_outdated": taxonomy_id != project_taxonomy_id,
                    "annotation_status": annotation_status,
                    "revision": latest["revision"] if latest else 0,
                    "pending_count": pending,
                    "box_count": len(latest["boxes"]) if latest else 0,
                    "reviewer": latest["reviewer"] if latest else "",
                    "reserved_split": group_split or pixel_split,
                    "signal": signal,
                    "hints": review_hints(
                        conn, frame, taxonomy, suggestion_records, decisions, signal
                    ),
                }
            )
    return {
        "session_id": session_id,
        "taxonomy_id": project_taxonomy_id,
        "config": {
            "comparison_id": comparison_id,
            "confidence_threshold": confidence_threshold,
            "iou_threshold": iou_threshold,
            "protocol": PROTOCOL,
            "matching": MATCHING,
        },
        "comparison": summary,
        "counts": counts,
        "warnings": list(WARNINGS),
        "frames": frames,
    }
