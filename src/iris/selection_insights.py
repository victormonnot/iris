"""Bounded, read-only gallery hints from saved reviews, pixels hashes and predictions."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import PurePosixPath

from iris.annotations import MAX_BOXES, _coordinates, _finite_number, _latest
from iris.dataset_manifest import taxonomy_mappings
from iris.inference import comparison_lanes
from iris.model_taxonomy import class_contract
from iris.prediction_taxonomy import COCO_TAXONOMY, annotation_mapping, validate_output_labels
from iris.store import DEFAULT_PROJECT_ID, Store, _decode
from iris.taxonomies import TAXONOMY, _get

PROTOCOL = "iris-selection-insights-v1"
MAX_FRAMES = 2000
MAX_SIMILARITY_FRAMES = 1000
MAX_RELATED_IDS = 20
MAX_PREDICTION_SOURCES = 20
MAX_JSON_BYTES = 2 * 1024 * 1024
SIMILARITY_DISTANCE = 6
LOW_SCORE_MIN = 0.1
LOW_SCORE_MAX = 0.5
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_DHASH = re.compile(r"[0-9a-f]{16}\Z")
WARNINGS = [
    "These are selection hints from saved data, not model accuracy or uncertainty estimates. "
    "Low scores and no target predictions never prove an error or an object's absence.",
    "Positive and negative labels require a human-validated annotation with no pending "
    "suggestions. Unreviewed empty images are not negatives.",
    "Similar images use a 64-bit dHash distance of at most 6. This heuristic can group "
    "different scenes and miss similar ones; inspect the linked images before selecting.",
    "No images, selections, annotations or split assignments are changed. Keep related "
    "scenes in the same dataset split.",
]


def _json(raw):
    if not isinstance(raw, str) or len(raw) > MAX_JSON_BYTES:
        raise ValueError("Saved prediction metadata exceeds the selection-insight limit")
    return json.loads(raw)


def _annotation(conn, frame, taxonomy, latest):
    decisions = latest["decisions"] if latest else {}
    if not isinstance(decisions, dict):
        decisions = {}
    reviewed = {
        key: state
        for key, state in decisions.items()
        if isinstance(key, str)
        and isinstance(state, str)
        and state in {"accepted", "corrected", "rejected"}
    }
    pending = conn.execute(
        "SELECT COUNT(*) FROM annotation_suggestions s WHERE s.frame_id=? "
        "AND NOT EXISTS (SELECT 1 FROM json_each(?) d WHERE d.key=s.id)",
        (frame["id"], json.dumps(reviewed)),
    ).fetchone()[0]
    counts = {item["id"]: 0 for item in taxonomy["classes"]}
    valid, error = True, None
    boxes = latest["boxes"] if latest else []
    try:
        if (
            type(frame["width"]) is not int
            or type(frame["height"]) is not int
            or min(frame["width"], frame["height"]) <= 0
            or not isinstance(frame["sha256"], str)
            or not _HASH.fullmatch(frame["sha256"])
            or latest
            and not isinstance(latest["decisions"], dict)
        ):
            raise ValueError("Saved annotation has an invalid image identity or review decisions")
        if not isinstance(boxes, list) or len(boxes) > MAX_BOXES:
            raise ValueError("Saved annotation has an invalid boxes list")
        for box in boxes:
            if not isinstance(box, dict) or not isinstance(box.get("label"), str):
                raise ValueError("Saved annotation contains an invalid label")
            if box["label"] not in counts:
                raise ValueError("Saved annotation label differs from its class version")
            _coordinates(box.get("box"), frame)
            counts[box["label"]] += 1
        if latest and latest["frame_sha256"] != frame["sha256"]:
            raise ValueError("Saved annotation refers to different image pixels")
        if latest and latest["status"] == "validated" and not latest["reviewer"].strip():
            raise ValueError("Saved annotation has no recorded human reviewer")
    except (ValueError, TypeError, KeyError, AttributeError):
        valid, error = False, "Saved annotation is inconsistent; open it for review."
        counts = {}
    annotation_status = latest["status"] if latest else "unannotated"
    status = "pending_suggestions" if pending else annotation_status if valid else "draft"
    validated = valid and status == "validated"
    return latest, {
        "annotation_status": annotation_status,
        "review_status": status,
        "revision": latest["revision"] if latest else 0,
        "pending_count": pending,
        "box_count": len(boxes) if valid else None,
        "class_counts": counts,
        "negative": not boxes if validated else None,
        "positive": bool(boxes) if validated else None,
        "annotation_valid": valid,
        "annotation_error": error,
    }


def _unavailable(reason, *, scanned=0, truncated=False):
    return {
        "low_confidence_count": None,
        "no_target_predictions": None,
        "prediction_source_id": None,
        "prediction_signal_status": "unavailable",
        "prediction_signal_reason": reason,
        "prediction_target_class_ids": [],
        "prediction_mapping_complete": None,
        "prediction_sources_inspected": scanned,
        "prediction_sources_truncated": truncated,
    }


def _source_signal(raw, frame, taxonomy):
    comparison = {
        "id": raw["comparison_id"],
        "config": _json(raw["comparison_config"]),
        "frame_ids": _json(raw["comparison_frames"]),
        "model_ids": _json(raw["comparison_models"]),
    }
    run = {
        "id": raw["run_id"],
        "model_id": raw["run_model_id"],
        "variant": raw["variant"],
        "metadata": _json(raw["run_metadata"]),
    }
    config = comparison["config"]
    input_size = _json(raw["input_size"])
    if (
        not isinstance(run["metadata"], dict)
        or run["metadata"].get("model_id", run["model_id"]) != run["model_id"]
        or not isinstance(input_size, list)
        or any(type(value) is not int for value in input_size)
        or raw["run_comparison_id"] != raw["comparison_id"]
        or raw["model_id"] != run["model_id"]
        or not isinstance(comparison["frame_ids"], list)
        or frame["id"] not in comparison["frame_ids"]
        or config.get("frame_hashes", {}).get(frame["id"]) != frame["sha256"]
        or {"model_id": run["model_id"], "variant": run["variant"]}
        not in comparison_lanes(comparison)
        or input_size != [frame["width"], frame["height"]]
    ):
        raise ValueError("Saved prediction source does not match this image and model run")
    mapping, source_taxonomy = annotation_mapping(comparison, run, taxonomy)
    contract = (
        {"taxonomy_id": COCO_TAXONOMY}
        if source_taxonomy == COCO_TAXONOMY
        else config["model_class_contracts"][run["model_id"]]
    )
    if config.get("model_class_contracts") is None and any(
        key in run["metadata"]
        for key in (
            "taxonomy_id",
            "taxonomy",
            "output_class_mapping",
            "native_to_coco",
            "training_id",
        )
    ):
        # Legacy person/car fine-tunes shared the COCO comparison namespace, but
        # their two-class head could never cover every later COCO-mapped class.
        contract = class_contract(run["metadata"])
        if contract["taxonomy_id"] != TAXONOMY["id"]:
            raise ValueError("Saved trained class definitions require a recorded model contract")
        native = run["metadata"].get("native_to_coco")
        if native is not None and (
            not isinstance(native, dict)
            or {str(key): value for key, value in native.items()} != {"1": 1, "2": 3}
            or any(type(value) is not int for value in native.values())
        ):
            raise ValueError("Saved trained output mapping is inconsistent")
        outputs = set(contract["output_class_mapping"].values())
        mapping = {category: label for category, label in mapping.items() if category in outputs}
    if not mapping:
        return None
    detections = _json(raw["detections"])
    limit = 300 if run["variant"] == "tiled" else 100
    if (
        not isinstance(detections, list)
        or len(detections) > limit
        or any(not isinstance(item, dict) for item in detections)
    ):
        raise ValueError("Saved prediction has an invalid detection list")
    validate_output_labels(detections, contract)
    targets = []
    for detection in detections:
        score = detection.get("score")
        if not _finite_number(score) or not 0 <= score <= 1:
            raise ValueError("Saved prediction has an invalid confidence")
        _coordinates(detection.get("box"), frame)
        if detection["label_id"] in mapping:
            targets.append(detection)
    classes = [item["id"] for item in taxonomy["classes"] if item["id"] in mapping.values()]
    complete = len(classes) == len(taxonomy["classes"])
    return {
        "prediction_source_id": raw["id"],
        "prediction_model_id": raw["model_id"],
        "prediction_run_id": raw["run_id"],
        "prediction_comparison_id": raw["comparison_id"],
        "prediction_created_at": raw["created_at"],
        "prediction_variant": raw["variant"],
        "prediction_signal_status": "available",
        "prediction_signal_reason": None
        if complete
        else "Only explicitly mapped classes are covered by this saved source.",
        "prediction_target_class_ids": classes,
        "prediction_mapping_complete": complete,
        "low_confidence_count": sum(
            LOW_SCORE_MIN <= item["score"] < LOW_SCORE_MAX for item in targets
        ),
        "no_target_predictions": not targets if complete else None,
    }


def _candidate_ids(conn, session_id):
    # Rank only small identity columns once for this session. Fetch large detector
    # payloads in per-frame batches; there is no per-frame scan of all predictions.
    rows = conn.execute(
        "WITH ranked AS (SELECT p.id,p.frame_id,ROW_NUMBER() OVER ("
        "PARTITION BY p.frame_id ORDER BY p.created_at DESC,p.id DESC) AS position "
        "FROM comparisons c JOIN jobs j ON j.id=c.job_id "
        "JOIN predictions p ON p.comparison_id=c.id "
        "WHERE c.session_id=? AND j.status='succeeded' AND p.frame_id IN ("
        "SELECT id FROM frames WHERE session_id=? ORDER BY created_at,id LIMIT ?)) "
        "SELECT id,frame_id FROM ranked WHERE position<=? ORDER BY frame_id,position",
        (session_id, session_id, MAX_FRAMES, MAX_PREDICTION_SOURCES + 1),
    )
    candidates = defaultdict(list)
    for row in rows:
        candidates[row["frame_id"]].append(row["id"])
    return candidates


def _prediction_signal(conn, frame, taxonomy, candidate_ids):
    if not candidate_ids:
        return _unavailable("No completed saved prediction is available for this image.")
    placeholders = ",".join("?" for _ in candidate_ids)
    candidates = conn.execute(
        "SELECT p.id,p.comparison_id,p.run_id,p.model_id,p.detections,p.input_size,p.created_at, "
        "c.config AS comparison_config,c.frame_ids AS comparison_frames, "
        "c.model_ids AS comparison_models,r.model_id AS run_model_id, "
        "r.comparison_id AS run_comparison_id,r.metadata AS run_metadata,r.variant "
        "FROM predictions p JOIN comparisons c ON c.id=p.comparison_id "
        "JOIN runs r ON r.id=p.run_id JOIN jobs j ON j.id=c.job_id "
        f"WHERE p.id IN ({placeholders}) AND c.session_id=? AND j.status='succeeded' "
        "ORDER BY p.created_at DESC,p.id DESC",
        (*candidate_ids, frame["session_id"]),
    ).fetchall()
    truncated = len(candidates) > MAX_PREDICTION_SOURCES
    for scanned, raw in enumerate(candidates[:MAX_PREDICTION_SOURCES], 1):
        try:
            signal = _source_signal(raw, frame, taxonomy)
        except (ValueError, KeyError, TypeError, AttributeError, IndexError, OverflowError):
            continue
        if signal is not None:
            return {
                **signal,
                "prediction_sources_inspected": scanned,
                "prediction_sources_truncated": truncated,
            }
    return _unavailable(
        "No compatible, valid completed prediction was found in the inspected saved sources."
        if candidates
        else "No completed saved prediction is available for this image.",
        scanned=min(len(candidates), MAX_PREDICTION_SOURCES),
        truncated=truncated,
    )


def _related(frames, output):
    by_hash = defaultdict(list)
    by_asset = defaultdict(list)
    for frame in frames:
        if isinstance(frame["sha256"], str) and _HASH.fullmatch(frame["sha256"]):
            by_hash[frame["sha256"]].append(frame["id"])
        by_asset[frame["asset_id"]].append(frame)
    for group in by_hash.values():
        for identifier in group:
            others = [value for value in group if value != identifier]
            output[identifier]["exact_duplicate_ids"] = others[:MAX_RELATED_IDS]
            output[identifier]["exact_duplicate_count"] = len(others)
    for group in by_asset.values():
        ordered = sorted(
            group,
            key=lambda frame: (
                frame["timestamp_seconds"]
                if _finite_number(frame["timestamp_seconds"])
                else float("inf"),
                frame["frame_index"] if type(frame["frame_index"]) is int else float("inf"),
                frame["created_at"],
                frame["id"],
            ),
        )
        for index, frame in enumerate(ordered):
            output[frame["id"]]["neighbor_frame_ids"] = [
                other["id"]
                for other in ordered[max(0, index - 2) : index + 3]
                if other["id"] != frame["id"]
            ]
    inspected = frames[:MAX_SIMILARITY_FRAMES]
    valid = []
    for frame in inspected:
        digest = frame["perceptual_hash"]
        if isinstance(digest, str) and _DHASH.fullmatch(digest):
            output[frame["id"]]["similarity_inspected"] = True
            valid.append((frame, int(digest, 16)))
    pairs = 0
    for index, (left, digest) in enumerate(valid):
        for right, other in valid[index + 1 :]:
            pairs += 1
            if (
                left["sha256"] == right["sha256"]
                or (digest ^ other).bit_count() > SIMILARITY_DISTANCE
            ):
                continue
            for source, target in ((left, right), (right, left)):
                row = output[source["id"]]
                row["similar_frame_count"] += 1
                if len(row["similar_frame_ids"]) < MAX_RELATED_IDS:
                    row["similar_frame_ids"].append(target["id"])
    return len(valid), pairs


def selection_insights(store: Store, session_id: str, project_id: str = DEFAULT_PROJECT_ID) -> dict:
    """Inspect one session snapshot; never infer negatives or mutate selected media."""
    with store.connect() as conn:
        conn.execute("BEGIN")
        session = conn.execute(
            "SELECT s.*,p.taxonomy_id AS current_taxonomy_id FROM sessions s "
            "JOIN projects p ON p.id=s.project_id WHERE s.id=? AND s.project_id=?",
            (session_id, project_id),
        ).fetchone()
        if session is None:
            raise KeyError(session_id)
        total = conn.execute(
            "SELECT COUNT(*) FROM frames WHERE session_id=?", (session_id,)
        ).fetchone()[0]
        frames = [
            _decode(row)
            for row in conn.execute(
                "SELECT f.*,a.filename AS source_filename,a.kind AS media_kind FROM frames f "
                "JOIN assets a ON a.id=f.asset_id AND a.session_id=f.session_id "
                "WHERE f.session_id=? ORDER BY f.created_at,f.id LIMIT ?",
                (session_id, MAX_FRAMES),
            )
        ]
        output, taxonomies = {}, {}
        candidates = _candidate_ids(conn, session_id)
        for frame in frames:
            latest = _latest(conn, frame["id"])
            taxonomy_id = latest["taxonomy_id"] if latest else frame["taxonomy_id"]
            if taxonomy_id not in taxonomies:
                taxonomies[taxonomy_id] = _get(conn, taxonomy_id, project_id)
                taxonomy_mappings(taxonomies[taxonomy_id])
            taxonomy = taxonomies[taxonomy_id]
            _, annotation = _annotation(conn, frame, taxonomy, latest)
            output[frame["id"]] = {
                "frame_id": frame["id"],
                **annotation,
                "taxonomy_id": taxonomy_id,
                "taxonomy_outdated": taxonomy_id != session["current_taxonomy_id"],
                "source": {
                    "asset_id": frame["asset_id"],
                    "filename": PurePosixPath(frame["source_filename"].replace("\\", "/")).name,
                    "media_kind": frame["media_kind"],
                    "frame_index": frame["frame_index"],
                    "timestamp_seconds": frame["timestamp_seconds"],
                },
                "exact_duplicate_ids": [],
                "exact_duplicate_count": 0,
                "similar_frame_ids": [],
                "similar_frame_count": 0,
                "similarity_inspected": False,
                "neighbor_frame_ids": [],
                **_prediction_signal(conn, frame, taxonomy, candidates[frame["id"]]),
            }
        inspected, pairs = _related(frames, output)
    rows = list(output.values())
    counts = Counter(row["review_status"] for row in rows)
    summary = {
        "total_frames": total,
        "returned_frames": len(rows),
        "review_counts": {
            status: counts[status]
            for status in ("unannotated", "draft", "validated", "pending_suggestions")
        },
        "positive_frames": sum(row["positive"] is True for row in rows),
        "negative_frames": sum(row["negative"] is True for row in rows),
        "signal_available_frames": sum(
            row["prediction_signal_status"] == "available" for row in rows
        ),
        "low_confidence_frames": sum(bool(row["low_confidence_count"]) for row in rows),
        "no_target_prediction_frames": sum(row["no_target_predictions"] is True for row in rows),
        "exact_duplicate_frames": sum(bool(row["exact_duplicate_count"]) for row in rows),
        "similar_frames": sum(bool(row["similar_frame_count"]) for row in rows),
    }
    limits = {
        "max_frames": MAX_FRAMES,
        "frames_truncated": total > len(rows),
        "max_similarity_frames": MAX_SIMILARITY_FRAMES,
        "similarity_frames_inspected": inspected,
        "similarity_truncated": len(frames) > MAX_SIMILARITY_FRAMES,
        "similarity_pairs_checked": pairs,
        "dhash_distance": SIMILARITY_DISTANCE,
        "max_related_ids_per_frame": MAX_RELATED_IDS,
        "max_prediction_sources_per_frame": MAX_PREDICTION_SOURCES,
        "low_confidence_min_inclusive": LOW_SCORE_MIN,
        "low_confidence_max_exclusive": LOW_SCORE_MAX,
    }
    warnings = list(WARNINGS)
    if limits["frames_truncated"] or limits["similarity_truncated"]:
        warnings.append(
            "The session exceeds inspection limits; hints cover the earliest returned images only."
        )
    if any(row["prediction_sources_truncated"] for row in rows):
        warnings.append(
            "Some frames have more saved sources than inspected; "
            "a compatible older source may be omitted."
        )
    return {
        "protocol": PROTOCOL,
        "session_id": session_id,
        "project_id": project_id,
        "frames": rows,
        "summary": summary,
        "limits": limits,
        "warnings": warnings,
    }
