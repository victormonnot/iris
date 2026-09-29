"""Explicit, bounded local multimodal review, separate from human annotations."""

import json
import math
from collections.abc import Callable

from iris.annotations import TAXONOMY, AnnotationConflict, _coordinates, require_revision
from iris.assistance_provider import OllamaReviewer, provider_status
from iris.inference import _load_verified_frame
from iris.store import Store, new_id, now

MAX_CANDIDATES = 8


def _candidates(
    store: Store, frame: dict, annotation: dict, prediction_id, threshold
) -> list[dict]:
    if prediction_id is None:
        return [
            {
                "id": f"annotation-{index}",
                "label": box["label"],
                "box": box["box"],
                "source": {
                    "kind": "annotation",
                    "revision": annotation["revision"],
                    "target_box_id": box["id"],
                },
            }
            for index, box in enumerate(annotation["boxes"])
        ]
    prediction = store.get("predictions", prediction_id)
    if prediction is None or prediction["frame_id"] != frame["id"]:
        raise ValueError("The saved prediction must belong to this frame")
    comparison = store.get("comparisons", prediction["comparison_id"])
    if comparison["config"]["frame_hashes"][frame["id"]] != frame["sha256"]:
        raise ValueError("The prediction was produced from a different frame revision")
    mapping = {item["coco_id"]: item["id"] for item in TAXONOMY["classes"]}
    run = store.get("runs", prediction["run_id"])
    if (
        comparison["config"].get("taxonomy") != "coco-2017-v1"
        or prediction["input_size"] != [frame["width"], frame["height"]]
        or frame["id"] not in comparison["frame_ids"]
        or run["comparison_id"] != comparison["id"]
        or run["model_id"] != prediction["model_id"]
    ):
        raise ValueError(
            "The prediction's taxonomy, image dimensions or provenance is incompatible"
        )
    for detection in prediction["detections"]:
        if type(detection.get("label_id")) is not int:
            raise ValueError("The prediction contains an invalid category")
        score = detection.get("score")
        if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("The prediction contains an invalid confidence")
        _coordinates(detection.get("box"), frame)
    return [
        {
            "id": f"{prediction_id}-{index}",
            "label": mapping[detection["label_id"]],
            "box": detection["box"],
            "source": {
                "kind": "prediction",
                "prediction_id": prediction_id,
                "detection_index": index,
                "score": detection["score"],
                "model_id": prediction["model_id"],
                "run_metadata": run["metadata"],
            },
        }
        for index, detection in enumerate(prediction["detections"])
        if detection["label_id"] in mapping and detection["score"] >= threshold
    ]


def request_assistance(
    store: Store,
    jobs,
    frame_id: str,
    *,
    expected_revision: int,
    prediction_id: str | None = None,
    threshold: float = 0.5,
    instructions: str = "",
) -> dict:
    annotation = require_revision(store, frame_id, expected_revision)
    frame = store.get("frames", frame_id)
    if (
        type(threshold) not in (int, float)
        or not math.isfinite(threshold)
        or not 0 <= threshold <= 1
    ):
        raise ValueError("Confidence threshold must be between 0 and 1")
    if not isinstance(instructions, str) or len(instructions) > 2000:
        raise ValueError("Additional instructions are limited to 2000 characters")
    candidates = _candidates(store, frame, annotation, prediction_id, threshold)
    if not 1 <= len(candidates) <= MAX_CANDIDATES:
        raise ValueError(
            "Choose a source containing 1–8 person/car boxes; adjust the threshold if needed"
        )
    with _load_verified_frame(store, frame, frame["sha256"]):
        pass
    status = provider_status()
    if status["status"] != "ready":
        raise RuntimeError(status.get("reason") or "The local annotation provider is unavailable")
    config = {
        "provider": {"endpoint": status["endpoint"], "model": status["model"]},
        "model_digest": status.get("model_digest"),
        "frame_sha256": frame["sha256"],
        "base_revision": expected_revision,
        "taxonomy_id": TAXONOMY["id"],
        "instructions": instructions.strip(),
        "prediction_id": prediction_id,
        "threshold": threshold,
        "max_candidates": MAX_CANDIDATES,
    }
    job_id, record_id, created_at = new_id(), new_id(), now()
    with jobs.guard, store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        latest = conn.execute(
            "SELECT COALESCE(MAX(revision),0) FROM annotation_revisions WHERE frame_id=?",
            (frame_id,),
        ).fetchone()[0]
        if latest != expected_revision:
            raise AnnotationConflict(
                "Annotations changed while preparing assistance; reload the frame"
            )
        active = conn.execute(
            "SELECT 1 FROM assistance_records a JOIN jobs j ON j.id=a.job_id "
            "WHERE a.frame_id=? AND j.status IN ('queued','running')",
            (frame_id,),
        ).fetchone()
        if active:
            raise RuntimeError("An assistance request is already queued or running for this frame")
        conn.execute(
            "INSERT INTO jobs (id,kind,status,params,message,created_at) VALUES (?,?,?,?,?,?)",
            (
                job_id,
                "assist",
                "queued",
                json.dumps({"frame_id": frame_id, "assistance_id": record_id}),
                "Waiting for explicit local multimodal review",
                created_at,
            ),
        )
        conn.execute(
            "INSERT INTO assistance_records (id,frame_id,job_id,config,candidates,created_at) "
            "VALUES (?,?,?,?,?,?)",
            (record_id, frame_id, job_id, json.dumps(config), json.dumps(candidates), created_at),
        )
    return store.get("jobs", job_id)


def run_assistance(
    store: Store,
    record_id: str,
    progress: Callable[[float, str], None],
    cancelled: Callable[[], bool],
    reviewer_factory=None,
) -> dict:
    record = store.get("assistance_records", record_id)
    if record is None:
        raise ValueError("Assistance request not found")
    result = {
        "assistance_id": record_id,
        "frame_id": record["frame_id"],
        "suggestions_created": 0,
        "cancelled": False,
    }
    if cancelled():
        result["cancelled"] = True
        return result
    if record["raw_response"] is not None:
        raise ValueError("An assistance request is immutable; create a new request to retry")
    frame = store.get("frames", record["frame_id"])
    config = record["config"]
    try:
        factory = reviewer_factory or OllamaReviewer
        reviewer = factory(config=config["provider"])
        if (
            config.get("model_digest")
            and reviewer.metadata.get("model_digest") != config["model_digest"]
        ):
            raise ValueError("The local multimodal model changed after this job was queued")
        store.update("assistance_records", record_id, {"metadata": reviewer.metadata})
        with _load_verified_frame(store, frame, config["frame_sha256"]) as image:
            if cancelled():
                result["cancelled"] = True
                return result
            progress(0.1, f"Reviewing {len(record['candidates'])} candidates with local Ollama")
            reviewed = reviewer.review(
                image,
                [
                    {key: candidate[key] for key in ("id", "label", "box")}
                    for candidate in record["candidates"]
                ],
                instructions=config["instructions"],
            )
        store.update(
            "assistance_records",
            record_id,
            {
                "prompt": reviewed["prompt"],
                "metadata": reviewed["metadata"],
                "raw_response": reviewed["raw_response"],
            },
        )
        if cancelled():
            result["cancelled"] = True
            return result
        # Verify the source again before publishing suggestions; human revisions
        # may have advanced meanwhile, but no revision is ever changed by this job.
        with _load_verified_frame(store, frame, config["frame_sha256"]):
            pass
        candidates = {candidate["id"]: candidate for candidate in record["candidates"]}
        reviews = reviewed["reviews"]
        if len(reviews) != len(candidates) or {r["candidate_id"] for r in reviews} != set(
            candidates
        ):
            raise ValueError("The provider must review every candidate exactly once")
        if cancelled():
            result["cancelled"] = True
            return result
        with store.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            current = conn.execute(
                "SELECT status,cancel_requested FROM jobs WHERE id=?", (record["job_id"],)
            ).fetchone()
            if current["cancel_requested"] or current["status"] not in {"queued", "running"}:
                result["cancelled"] = True
                return result
            for review in reviews:
                candidate = candidates[review["candidate_id"]]
                label = review["label"]
                if label not in {"person", "car", "none", "uncertain"}:
                    raise ValueError("The provider returned an unsupported category")
                recommendation = (
                    "reject"
                    if label == "none"
                    else (
                        "uncertain"
                        if label == "uncertain"
                        else ("keep" if label == candidate["label"] else "change")
                    )
                )
                metadata = {
                    "assistance_id": record_id,
                    "provider": reviewed["metadata"],
                    "candidate_id": candidate["id"],
                    "source": candidate["source"],
                    "base_revision": config["base_revision"],
                    "frame_sha256": config["frame_sha256"],
                    "recommendation": recommendation,
                    "reason": review["reason"],
                    "scene_notes": reviewed["scene_notes"],
                }
                if "target_box_id" in candidate["source"]:
                    metadata["target_box_id"] = candidate["source"]["target_box_id"]
                conn.execute(
                    "INSERT INTO annotation_suggestions "
                    "(id,frame_id,job_id,kind,label,box,metadata,created_at) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (
                        new_id(),
                        frame["id"],
                        record["job_id"],
                        "multimodal",
                        label if label in {"person", "car"} else candidate["label"],
                        json.dumps(candidate["box"]),
                        json.dumps(metadata),
                        now(),
                    ),
                )
        result["suggestions_created"] = len(reviews)
        progress(1, "Multimodal proposals saved; human review is required")
        return result
    except Exception as exc:
        update = {"error": str(exc)}
        for field in ("raw_response", "metadata", "prompt"):
            value = getattr(exc, field, None)
            if value is not None:
                update[field] = value
        store.update("assistance_records", record_id, update)
        raise
