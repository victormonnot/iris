"""Human annotation revisions and reviewable, traceable automatic suggestions."""

import hashlib
import json
import math
import sqlite3

from iris.inference import _load_verified_frame
from iris.store import Store, _decode, new_id, now
from iris.taxonomies import TAXONOMY, get_taxonomy

CLASS_IDS = {item["id"] for item in TAXONOMY["classes"]}
COCO_MAPPING = {item["coco_id"]: item["id"] for item in TAXONOMY["classes"]}
MAX_BOXES = 500
MAX_DETECTOR_SUGGESTIONS = 100


class AnnotationConflict(RuntimeError):
    """The caller edited an annotation that has since acquired a new revision."""


def _frame(store: Store, frame_id: str) -> dict:
    frame = store.get("frames", frame_id)
    if frame is None:
        raise KeyError(frame_id)
    return frame


def _latest(conn: sqlite3.Connection, frame_id: str) -> dict | None:
    return _decode(
        conn.execute(
            "SELECT * FROM annotation_revisions WHERE frame_id=? ORDER BY revision DESC LIMIT 1",
            (frame_id,),
        ).fetchone()
    )


def _check_revision(latest: dict | None, expected_revision: int):
    if type(expected_revision) is not int or expected_revision < 0:
        raise ValueError("Expected revision must be a nonnegative integer")
    actual = latest["revision"] if latest else 0
    if actual != expected_revision:
        raise AnnotationConflict(
            f"Annotation changed: expected revision {expected_revision}, found {actual}. "
            "Reload the image before saving."
        )


def _taxonomy_id(frame: dict, latest: dict | None) -> str:
    return latest["taxonomy_id"] if latest else frame["taxonomy_id"]


def _check_taxonomy(actual: str, expected: str | None):
    if expected is not None and expected != actual:
        raise AnnotationConflict(
            "The annotation class version changed. Reload the image before saving."
        )


def _coco_mapping(taxonomy: dict) -> dict[int, str]:
    return {
        item["coco_id"]: item["id"]
        for item in taxonomy["classes"]
        if type(item.get("coco_id")) is int
    }


def require_revision(store: Store, frame_id: str, expected_revision: int) -> dict:
    """Check an editor snapshot; mutating callers must also check within their transaction."""
    annotation = get_annotation(store, frame_id)
    _check_revision(annotation, expected_revision)
    return annotation


def get_annotation(store: Store, frame_id: str) -> dict:
    frame = _frame(store, frame_id)
    asset = store.get("assets", frame["asset_id"])
    session = store.get("sessions", frame["session_id"])
    # Read the revisions and suggestions in one SQLite snapshot, including when a
    # background assistance job is publishing new suggestions.
    with store.connect() as conn:
        conn.execute("BEGIN")
        project = conn.execute(
            "SELECT * FROM projects WHERE id=?", (session["project_id"],)
        ).fetchone()
        revisions = [
            _decode(row)
            for row in conn.execute(
                "SELECT * FROM annotation_revisions WHERE frame_id=? ORDER BY revision DESC",
                (frame_id,),
            )
        ]
        suggestions = [
            _decode(row)
            for row in conn.execute(
                "SELECT * FROM annotation_suggestions WHERE frame_id=? ORDER BY created_at,id",
                (frame_id,),
            )
        ]
    latest = revisions[0] if revisions else None
    taxonomy = get_taxonomy(store, _taxonomy_id(frame, latest), session["project_id"])
    current_taxonomy = get_taxonomy(store, project["taxonomy_id"], session["project_id"])
    coco_mapping = _coco_mapping(taxonomy)
    decisions = latest["decisions"] if latest else {}
    prediction_sources = []
    for prediction in store.list("predictions", frame_id=frame_id):
        comparison = store.get("comparisons", prediction["comparison_id"])
        run = store.get("runs", prediction["run_id"])
        if comparison["config"].get("taxonomy") != "coco-2017-v1":
            continue
        prediction_sources.append(
            {
                "id": prediction["id"],
                "model_id": prediction["model_id"],
                "variant": run.get("variant", "full"),
                "run_id": prediction["run_id"],
                "comparison_id": prediction["comparison_id"],
                "comparison_name": comparison["name"],
                "detection_count": sum(
                    type(detection.get("label_id")) is int and detection["label_id"] in coco_mapping
                    for detection in prediction["detections"]
                ),
                "created_at": prediction["created_at"],
            }
        )
    return {
        "frame": {
            **{key: value for key, value in frame.items() if key != "path"},
            "source_filename": asset["filename"],
            "session_name": session["name"],
            "scene_group": session["scene_group"],
        },
        "taxonomy": taxonomy,
        "taxonomy_id": taxonomy["id"],
        "current_taxonomy": current_taxonomy,
        "taxonomy_outdated": taxonomy["id"] != current_taxonomy["id"],
        "revision": latest["revision"] if latest else 0,
        "status": latest["status"] if latest else "unannotated",
        "boxes": latest["boxes"] if latest else [],
        "decisions": decisions,
        "reviewer": latest["reviewer"] if latest else "",
        "notes": latest["notes"] if latest else "",
        "frame_sha256": latest["frame_sha256"] if latest else frame["sha256"],
        "history": [
            {
                "id": revision["id"],
                "revision": revision["revision"],
                "status": revision["status"],
                "taxonomy_id": revision["taxonomy_id"],
                "taxonomy": get_taxonomy(store, revision["taxonomy_id"], session["project_id"]),
                "box_count": len(revision["boxes"]),
                "reviewer": revision["reviewer"],
                "notes": revision["notes"],
                "created_at": revision["created_at"],
            }
            for revision in revisions
        ],
        "suggestions": [
            {
                **suggestion,
                "state": decisions.get(suggestion["id"], "pending"),
                "taxonomy_outdated": suggestion["metadata"].get("target_taxonomy", TAXONOMY["id"])
                != taxonomy["id"],
            }
            for suggestion in suggestions
        ],
        "prediction_sources": prediction_sources,
    }


def _finite_number(value) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _coordinates(box, frame: dict) -> list[float]:
    if (
        not isinstance(box, (list, tuple))
        or len(box) != 4
        or not all(_finite_number(value) for value in box)
    ):
        raise ValueError("Each box must contain four finite numeric xyxy pixel coordinates")
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= frame["width"] and 0 <= y1 < y2 <= frame["height"]):
        raise ValueError("Boxes must have positive area and stay inside the original image")
    return [float(value) for value in box]


def _validate_boxes(
    boxes: list[dict], frame: dict, suggestions: dict, taxonomy: dict | None = None
) -> list[dict]:
    taxonomy = TAXONOMY if taxonomy is None else taxonomy
    class_ids = {item["id"] for item in taxonomy["classes"]}
    if not isinstance(boxes, list) or len(boxes) > MAX_BOXES:
        raise ValueError(f"Annotations support at most {MAX_BOXES} boxes per image")
    result, identifiers, referenced = [], set(), set()
    for box in boxes:
        if not isinstance(box, dict):
            raise ValueError("Each annotation box must be an object")
        identifier, label = box.get("id"), box.get("label")
        if (
            not isinstance(identifier, str)
            or not identifier.strip()
            or len(identifier) > 128
            or identifier in identifiers
        ):
            raise ValueError("Each annotation box requires a distinct nonempty ID (128 max)")
        identifiers.add(identifier)
        if not isinstance(label, str) or label not in class_ids:
            raise ValueError("Annotation label must belong to this image's class version")
        coordinates = _coordinates(box.get("box"), frame)
        suggestion_id = box.get("suggestion_id")
        if suggestion_id is not None and (
            not isinstance(suggestion_id, str) or suggestion_id not in suggestions
        ):
            raise ValueError("A referenced suggestion must belong to this image")
        if suggestion_id in referenced:
            raise ValueError("A suggestion cannot be used by multiple annotation boxes")
        if suggestion_id is None:
            review_state, source = "manual", {"kind": "manual"}
        else:
            referenced.add(suggestion_id)
            suggestion = suggestions[suggestion_id]
            target = suggestion["metadata"].get("target_box_id")
            if suggestion["kind"] == "multimodal" and target and target != identifier:
                raise ValueError("A saved-box review must be applied to its original box ID")
            original = _coordinates(suggestion["box"], frame)
            review_state = (
                "accepted"
                if label == suggestion["label"] and coordinates == original
                else "corrected"
            )
            source = {
                "kind": suggestion["kind"],
                "suggestion_id": suggestion_id,
                "metadata": suggestion["metadata"],
            }
        result.append(
            {
                "id": identifier,
                "label": label,
                "box": coordinates,
                "suggestion_id": suggestion_id,
                "review_state": review_state,
                "source": source,
            }
        )
    return result


def _review_replaces_box(conn, suggestion: dict, old: dict, latest: dict) -> bool:
    """A model's review can replace its unchanged target without losing the earlier origin."""
    metadata = suggestion["metadata"]
    source, base = metadata.get("source", {}), metadata.get("base_revision")
    if (
        suggestion["kind"] != "multimodal"
        or metadata.get("target_box_id") != old["id"]
        or not isinstance(source, dict)
        or source.get("kind") != "annotation"
        or source.get("target_box_id") != old["id"]
        or type(base) is not int
        or source.get("revision") != base
        or base > latest["revision"]
        or metadata.get("frame_sha256") != latest["frame_sha256"]
    ):
        return False
    original = _decode(
        conn.execute(
            "SELECT * FROM annotation_revisions WHERE frame_id=? AND revision=?",
            (latest["frame_id"], base),
        ).fetchone()
    )
    if original is None or original["frame_sha256"] != latest["frame_sha256"]:
        return False
    target = next((box for box in original["boxes"] if box["id"] == old["id"]), None)
    return target is not None and all(
        target[key] == old[key] for key in ("label", "box", "suggestion_id")
    )


def _validate_decisions(conn, decisions, boxes: list[dict], suggestions: dict, latest: dict | None):
    if not isinstance(decisions, dict) or any(
        not isinstance(key, str)
        or key not in suggestions
        or not isinstance(value, str)
        or value not in {"accepted", "corrected", "rejected"}
        for key, value in decisions.items()
    ):
        raise ValueError("Review decisions must reference this image's suggestions")
    if latest and not latest["decisions"].keys() <= decisions.keys():
        raise ValueError("Keep previous suggestion decisions in the new annotation snapshot")
    references = {box["suggestion_id"]: box for box in boxes if box["suggestion_id"]}
    for suggestion_id, box in references.items():
        if decisions.get(suggestion_id) != box["review_state"]:
            raise ValueError("Each suggested box requires its matching accepted/corrected decision")
    for suggestion_id, decision in decisions.items():
        if decision == "rejected" and suggestion_id in references:
            raise ValueError("A rejected suggestion cannot remain in the annotation boxes")
        if decision in {"accepted", "corrected"} and suggestion_id not in references:
            raise ValueError("An accepted or corrected suggestion must have exactly one box")
    if latest:
        # An explicit multimodal review may replace an unchanged target. Every
        # earlier origin remains in the immutable history and review decisions.
        previous = {box["id"]: box for box in latest["boxes"]}
        for box in boxes:
            old = previous.get(box["id"])
            if old and old["suggestion_id"] != box["suggestion_id"]:
                replacement = suggestions.get(box["suggestion_id"])
                if replacement is None or not _review_replaces_box(conn, replacement, old, latest):
                    raise ValueError(
                        "An existing box can only change origin through a review of its "
                        "unchanged saved annotation"
                    )


def save_annotation(
    store: Store,
    frame_id: str,
    *,
    expected_revision: int,
    boxes: list[dict],
    decisions: dict,
    status: str = "draft",
    reviewer: str = "",
    notes: str = "",
    taxonomy_id: str | None = None,
) -> dict:
    frame = _frame(store, frame_id)
    if not isinstance(status, str) or status not in {"draft", "validated"}:
        raise ValueError("Annotation status must be draft or validated")
    if not isinstance(reviewer, str) or len(reviewer.strip()) > 120:
        raise ValueError("Reviewer must contain at most 120 characters")
    if not isinstance(notes, str) or len(notes) > 4000:
        raise ValueError("Annotation notes must contain at most 4000 characters")
    reviewer = reviewer.strip()
    if status == "validated" and not reviewer:
        raise ValueError("Human validation requires a reviewer name")
    with store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        latest = _latest(conn, frame_id)
        _check_revision(latest, expected_revision)
        resolved_id = _taxonomy_id(frame, latest)
        _check_taxonomy(resolved_id, taxonomy_id)
        session = conn.execute(
            "SELECT project_id FROM sessions WHERE id=?", (frame["session_id"],)
        ).fetchone()
        taxonomy = get_taxonomy(store, resolved_id, session["project_id"])
        expected_hash = latest["frame_sha256"] if latest else frame["sha256"]
        with _load_verified_frame(store, frame, expected_hash):
            pass
        suggestions = {
            row["id"]: _decode(row)
            for row in conn.execute(
                "SELECT * FROM annotation_suggestions WHERE frame_id=?", (frame_id,)
            )
        }
        normalized = _validate_boxes(boxes, frame, suggestions, taxonomy)
        _validate_decisions(conn, decisions, normalized, suggestions, latest)
        if status == "validated" and suggestions.keys() - decisions.keys():
            raise ValueError("Resolve every pending suggestion before validating this image")
        conn.execute(
            "INSERT INTO annotation_revisions "
            "(id,frame_id,revision,status,taxonomy_id,frame_sha256,boxes,decisions,reviewer,notes,"
            "created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                new_id(),
                frame_id,
                expected_revision + 1,
                status,
                resolved_id,
                expected_hash,
                json.dumps(normalized, allow_nan=False),
                json.dumps(decisions, allow_nan=False),
                reviewer,
                notes,
                now(),
            ),
        )
    return get_annotation(store, frame_id)


def adopt_taxonomy(
    store: Store,
    frame_id: str,
    *,
    expected_revision: int,
    expected_taxonomy_id: str,
    target_taxonomy_id: str,
) -> dict:
    """Explicitly move this image to the project's current classes as a new draft."""
    frame = _frame(store, frame_id)
    with store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        latest = _latest(conn, frame_id)
        _check_revision(latest, expected_revision)
        resolved_id = _taxonomy_id(frame, latest)
        if not isinstance(expected_taxonomy_id, str) or not expected_taxonomy_id:
            raise ValueError("Provide the image's current class version")
        _check_taxonomy(resolved_id, expected_taxonomy_id)
        project = conn.execute(
            "SELECT p.* FROM projects p JOIN sessions s ON s.project_id=p.id WHERE s.id=?",
            (frame["session_id"],),
        ).fetchone()
        if target_taxonomy_id != project["taxonomy_id"]:
            raise AnnotationConflict(
                "The project's current class version changed. Reload before adopting it."
            )
        if target_taxonomy_id == resolved_id:
            raise ValueError("This image already uses the project's current class version")
        taxonomy = get_taxonomy(store, target_taxonomy_id, project["id"])
        class_ids = {item["id"] for item in taxonomy["classes"]}
        boxes = latest["boxes"] if latest else []
        if any(box["label"] not in class_ids for box in boxes):
            raise ValueError(
                "Some saved boxes use classes absent from the current project version. "
                "Keep the image's existing version or explicitly remove those boxes and save "
                "before adopting the new classes."
            )
        expected_hash = latest["frame_sha256"] if latest else frame["sha256"]
        with _load_verified_frame(store, frame, expected_hash):
            pass
        for box in boxes:
            _coordinates(box["box"], frame)
        conn.execute(
            "INSERT INTO annotation_revisions "
            "(id,frame_id,revision,status,taxonomy_id,frame_sha256,boxes,decisions,reviewer,notes,"
            "created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                new_id(),
                frame_id,
                expected_revision + 1,
                "draft",
                target_taxonomy_id,
                expected_hash,
                json.dumps(boxes, allow_nan=False),
                json.dumps(latest["decisions"] if latest else {}, allow_nan=False),
                "",
                latest["notes"] if latest else "",
                now(),
            ),
        )
    return get_annotation(store, frame_id)


def add_detector_suggestions(
    store: Store,
    frame_id: str,
    *,
    prediction_id: str,
    threshold: float = 0.5,
    expected_revision: int,
) -> dict:
    frame = _frame(store, frame_id)
    if not _finite_number(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Detector suggestion threshold must be between 0 and 1")
    prediction = store.get("predictions", prediction_id)
    if prediction is None or prediction["frame_id"] != frame_id:
        raise ValueError("The saved prediction must belong to this image")
    comparison = store.get("comparisons", prediction["comparison_id"])
    run = store.get("runs", prediction["run_id"])
    config = comparison["config"]
    if (
        config.get("taxonomy") != "coco-2017-v1"
        or run["comparison_id"] != comparison["id"]
        or run["model_id"] != prediction["model_id"]
        or prediction["input_size"] != [frame["width"], frame["height"]]
        or frame_id not in comparison["frame_ids"]
        or not config.get("frame_hashes", {}).get(frame_id)
    ):
        raise ValueError(
            "The prediction's taxonomy, image dimensions or provenance is incompatible"
        )
    annotation = require_revision(store, frame_id, expected_revision)
    taxonomy = annotation["taxonomy"]
    coco_mapping = _coco_mapping(taxonomy)
    proposals = []
    for index, detection in enumerate(prediction["detections"]):
        label_id, score = detection.get("label_id"), detection.get("score")
        if type(label_id) is not int or label_id not in coco_mapping:
            continue
        if not _finite_number(score) or not 0 <= score <= 1:
            raise ValueError("The saved prediction contains an invalid confidence")
        if score < threshold:
            continue
        coordinates = _coordinates(detection.get("box"), frame)
        if len(proposals) == MAX_DETECTOR_SUGGESTIONS:
            raise ValueError("More than 100 detector proposals; raise the confidence threshold")
        identity = f"detector:{prediction_id}:{index}"
        if taxonomy["id"] != TAXONOMY["id"]:
            identity += f":{taxonomy['id']}"
        proposals.append(
            {
                "id": hashlib.sha256(identity.encode()).hexdigest(),
                "label": coco_mapping[label_id],
                "box": coordinates,
                "metadata": {
                    "prediction_id": prediction_id,
                    "detection_index": index,
                    "comparison_id": comparison["id"],
                    "run_id": run["id"],
                    "model_id": prediction["model_id"],
                    "model_metadata": run["metadata"],
                    "score": float(score),
                    "threshold": float(threshold),
                    "original_label_id": label_id,
                    "original_label": detection.get("label"),
                    "source_taxonomy": "coco-2017-v1",
                    "target_taxonomy": taxonomy["id"],
                    "frame_sha256": config["frame_hashes"][frame_id],
                },
            }
        )
    with store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        latest = _latest(conn, frame_id)
        _check_revision(latest, expected_revision)
        _check_taxonomy(_taxonomy_id(frame, latest), taxonomy["id"])
        with _load_verified_frame(store, frame, config["frame_hashes"][frame_id]):
            pass
        if latest and latest["frame_sha256"] != frame["sha256"]:
            raise ValueError("The image no longer matches its annotation history")
        created_at = now()
        for proposal in proposals:
            conn.execute(
                "INSERT OR IGNORE INTO annotation_suggestions "
                "(id,frame_id,job_id,kind,label,box,metadata,created_at) VALUES (?,?,?,?,?,?,?,?)",
                (
                    proposal["id"],
                    frame_id,
                    comparison["job_id"],
                    "detector",
                    proposal["label"],
                    json.dumps(proposal["box"], allow_nan=False),
                    json.dumps(proposal["metadata"], allow_nan=False),
                    created_at,
                ),
            )
    return get_annotation(store, frame_id)
