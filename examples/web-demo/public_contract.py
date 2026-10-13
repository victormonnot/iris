"""Explicit public projections of the FPV API seed.

These schemas describe imported, unreviewed images, optionally accompanied by
recorded predictions. They are deliberately narrower than the general Iris API.
They validate structure and seed state; the fixture builder must separately
verify source hashes, identities, geometry, and the contents of public strings.
"""

import math


def project(value, schema):
    """Copy only declared fields, rejecting missing fields and unexpected shapes.

    Nonempty dictionaries require every declared key and discard other keys.
    A one-element schema list projects every list item; empty lists/dictionaries
    require an actually empty container. Tuples are explicit schema alternatives.
    Scalar type schemas are exact, except float also accepts finite integers;
    scalar literals (including None) require that value and its exact type.
    Invalid input raises ValueError with its JSON location, without its contents.
    """
    return _project(value, schema, "$")


def _project(value, schema, location):
    if isinstance(schema, dict):
        if type(value) is not dict:
            raise ValueError(f"{location}: expected an object")
        if not schema and value:
            raise ValueError(f"{location}: expected an empty object")
        result = {}
        for key, child in schema.items():
            if key not in value:
                raise ValueError(f"{location}.{key}: required field is missing")
            result[key] = _project(value[key], child, f"{location}.{key}")
        return result
    if isinstance(schema, list):
        if len(schema) > 1:
            raise TypeError("A list schema must be empty or contain one item schema")
        if type(value) is not list:
            raise ValueError(f"{location}: expected a list")
        if not schema:
            if value:
                raise ValueError(f"{location}: expected an empty list")
            return []
        return [
            _project(item, schema[0], f"{location}[{index}]") for index, item in enumerate(value)
        ]
    if isinstance(schema, tuple):
        if not schema:
            raise TypeError("A schema union must contain an alternative")
        for alternative in schema:
            try:
                return _project(value, alternative, location)
            except ValueError:
                pass
        raise ValueError(f"{location}: value does not match an allowed shape")
    if schema is float:
        if type(value) not in (int, float):
            raise ValueError(f"{location}: expected a finite number")
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError(f"{location}: expected a finite number")
        return value
    if schema in (str, int, bool, type(None)):
        if type(value) is not schema:
            raise ValueError(f"{location}: expected {schema.__name__}")
        return value
    if schema is None or type(schema) in (str, int, float, bool):
        if type(schema) is float and not math.isfinite(schema):
            raise TypeError("A scalar schema literal must be finite")
        if type(value) is not type(schema) or value != schema:
            raise ValueError(f"{location}: unexpected scalar value")
        return value
    raise TypeError("Unsupported public projection schema")


PROJECT = {
    "id": str,
    "name": str,
    "description": str,
    "taxonomy_id": str,
    "created_at": str,
}

CLASS = {"id": str, "name": str, "definition": str, "coco_id": (int, None)}
TAXONOMY = {
    "id": str,
    "box_format": str,
    "classes": [CLASS],
    "review_guidance": str,
}
VERSIONED_TAXONOMY = {
    **TAXONOMY,
    "version": int,
    "parent_id": (str, None),
    "created_at": str,
}
TAXONOMIES = {
    "current_taxonomy_id": str,
    "versions": [VERSIONED_TAXONOMY],
}

SESSION = {
    "project_id": str,
    "id": str,
    "name": str,
    "scene_group": str,
    "created_at": str,
}

COCO_CATEGORY = {"id": int, "name": str, "supercategory": str}
COCO_ANNOTATION = {
    "id": int,
    "image_id": int,
    "category_id": int,
    "bbox": [float],
    "area": float,
    "iscrowd": int,
    "segmentation": [[float]],
}
COCO_IMAGE = {
    "id": int,
    "license": int,
    "file_name": str,
    "coco_url": str,
    "flickr_url": str,
    "date_captured": str,
    "width": int,
    "height": int,
}
IMPORT_PROVENANCE = {
    "id": str,
    "taxonomy": VERSIONED_TAXONOMY,
    "archive_sha256": str,
    "archive_filename": str,
    "json_filename": str,
    "source_url": str,
    "license_name": str,
    "attribution": str,
    "source_split": "val",
    "category_mapping": {"1": "person"},
    "coco_image": COCO_IMAGE,
    "original_annotations": [COCO_ANNOTATION],
    "original_categories": [COCO_CATEGORY],
}

ASSET = {
    "id": str,
    "session_id": str,
    "filename": str,
    "kind": "image",
    "sha256": str,
    "size_bytes": int,
    "metadata": {
        "format": str,
        "width": int,
        "height": int,
        "dataset_import": IMPORT_PROVENANCE,
    },
    "created_at": str,
}
FRAME_RECORD = {
    "taxonomy_id": str,
    "id": str,
    "session_id": str,
    "asset_id": str,
    "frame_index": None,
    "timestamp_seconds": None,
    "width": int,
    "height": int,
    "sha256": str,
    "perceptual_hash": str,
    "selected": bool,
    "extraction": {"method": "coco_import", "dataset_import": IMPORT_PROVENANCE},
    "created_at": str,
}
FRAME = {**FRAME_RECORD, "duplicate_count": int}

PREDICTION_SOURCE = {
    "id": str,
    "model_id": str,
    "variant": str,
    "run_id": str,
    "comparison_id": str,
    "comparison_name": str,
    "detection_count": int,
    "created_at": str,
}
SUGGESTION = {
    "id": str,
    "frame_id": str,
    "job_id": None,
    "kind": "imported",
    "label": str,
    "box": [float],
    "metadata": {
        "import_id": str,
        "archive_sha256": str,
        "source_url": str,
        "license_name": str,
        "attribution": str,
        "source_split": "val",
        "source_category": COCO_CATEGORY,
        "source_annotation": COCO_ANNOTATION,
        "source_image_id": int,
        "mapping_target": str,
        "target_taxonomy": str,
    },
    "created_at": str,
    "state": "pending",
    "taxonomy_outdated": False,
}
ANNOTATION = {
    "frame": {
        **FRAME_RECORD,
        "source_filename": str,
        "session_name": str,
        "scene_group": str,
    },
    "taxonomy": VERSIONED_TAXONOMY,
    "taxonomy_id": str,
    "current_taxonomy": VERSIONED_TAXONOMY,
    "taxonomy_outdated": False,
    "revision": 0,
    "status": "unannotated",
    "boxes": [],
    "decisions": {},
    "reviewer": "",
    "notes": "",
    "frame_sha256": str,
    "history": [],
    "suggestions": [SUGGESTION],
    "prediction_sources": [PREDICTION_SOURCE],
}

INSIGHT_FRAME = {
    "frame_id": str,
    "annotation_status": "unannotated",
    "review_status": "pending_suggestions",
    "revision": 0,
    "pending_count": int,
    "box_count": 0,
    "class_counts": {"person": 0},
    "negative": None,
    "positive": None,
    "annotation_valid": True,
    "annotation_error": None,
    "taxonomy_id": str,
    "taxonomy_outdated": False,
    "source": {
        "asset_id": str,
        "filename": str,
        "media_kind": "image",
        "frame_index": None,
        "timestamp_seconds": None,
    },
    "exact_duplicate_ids": [str],
    "exact_duplicate_count": int,
    "similar_frame_ids": [str],
    "similar_frame_count": int,
    "similarity_inspected": bool,
    "neighbor_frame_ids": [str],
    "prediction_sources_inspected": int,
    "prediction_sources_truncated": bool,
}
INSIGHT_WITH_PREDICTION = {
    **INSIGHT_FRAME,
    "prediction_source_id": str,
    "prediction_model_id": str,
    "prediction_run_id": str,
    "prediction_comparison_id": str,
    "prediction_created_at": str,
    "prediction_variant": str,
    "prediction_signal_status": "available",
    "prediction_signal_reason": None,
    "prediction_target_class_ids": [str],
    "prediction_mapping_complete": bool,
    "low_confidence_count": int,
    "no_target_predictions": bool,
}
INSIGHT_WITHOUT_PREDICTION = {
    **INSIGHT_FRAME,
    "prediction_source_id": None,
    "prediction_signal_status": "unavailable",
    "prediction_signal_reason": str,
    "prediction_target_class_ids": [],
    "prediction_mapping_complete": None,
    "low_confidence_count": None,
    "no_target_predictions": None,
}
INSIGHTS = {
    "protocol": str,
    "session_id": str,
    "project_id": str,
    "frames": [(INSIGHT_WITH_PREDICTION, INSIGHT_WITHOUT_PREDICTION)],
    "summary": {
        "total_frames": int,
        "returned_frames": int,
        "review_counts": {
            "unannotated": int,
            "draft": 0,
            "validated": 0,
            "pending_suggestions": int,
        },
        "positive_frames": 0,
        "negative_frames": 0,
        "signal_available_frames": int,
        "low_confidence_frames": int,
        "no_target_prediction_frames": int,
        "exact_duplicate_frames": int,
        "similar_frames": int,
    },
    "limits": {
        "max_frames": int,
        "frames_truncated": bool,
        "max_similarity_frames": int,
        "similarity_frames_inspected": int,
        "similarity_truncated": bool,
        "similarity_pairs_checked": int,
        "dhash_distance": int,
        "max_related_ids_per_frame": int,
        "max_prediction_sources_per_frame": int,
        "low_confidence_min_inclusive": float,
        "low_confidence_max_exclusive": float,
    },
    "warnings": [str],
}

MATCH = {"left_index": int, "right_index": int, "iou": float}
REVIEW_SIGNAL = {
    "status": str,
    "reason": str,
    "disagreement": (float, None),
    "counts": ([int], None),
    "matched_count": (int, None),
    "unmatched_counts": ([int], None),
    "class_conflicts": (int, None),
    "prediction_ids": [str],
    "matches": [MATCH],
    "unmatched_indices": [[int]],
    "class_conflict_pairs": [MATCH],
}
REVIEW_HINTS = {
    "protocol": str,
    "low_confidence_count": int,
    "uncertain_count": int,
    "possible_omission": bool,
    "reasons": [str],
    "unreadable_proposal_count": int,
    "prediction_id": (str, None),
    "prediction_status": str,
    "prediction_reason": (str, None),
    "sources_inspected": int,
    "sources_truncated": bool,
    "score_note": str,
}
REVIEW_COMPARISON = {
    "id": str,
    "name": str,
    "model_ids": [str],
    "created_at": str,
    "models": [{"id": str, "name": str}],
    "lanes": [{"model_id": str, "variant": str, "run_id": str}],
}
REVIEW_FRAME = {
    "id": str,
    "session_id": str,
    "asset_id": str,
    "source_filename": str,
    "frame_index": None,
    "timestamp_seconds": None,
    "width": int,
    "height": int,
    "sha256": str,
    "review_status": "pending",
    "taxonomy_id": str,
    "taxonomy_outdated": False,
    "annotation_status": "unannotated",
    "revision": 0,
    "pending_count": int,
    "box_count": 0,
    "reviewer": "",
    "reserved_split": "val",
    "signal": REVIEW_SIGNAL,
    "hints": REVIEW_HINTS,
}
REVIEW_QUEUE = {
    "session_id": str,
    "taxonomy_id": str,
    "config": {
        "comparison_id": (str, None),
        "confidence_threshold": float,
        "iou_threshold": float,
        "protocol": str,
        "matching": str,
    },
    "comparison": (REVIEW_COMPARISON, None),
    "counts": {
        "total": int,
        "needs_review": int,
        "unannotated": int,
        "draft": 0,
        "pending": int,
        "validated": 0,
    },
    "warnings": [str],
    "frames": [REVIEW_FRAME],
}
