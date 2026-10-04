"""Explicit local detector proposals over frozen image and annotation inputs."""

import hashlib
import json
import math
from copy import deepcopy

from iris import inference
from iris.annotations import MAX_DETECTOR_SUGGESTIONS, _latest, _taxonomy_id
from iris.prediction_taxonomy import output_contract
from iris.projects import record_project
from iris.store import Store, _decode, new_id, now
from iris.taxonomies import get_taxonomy

PROTOCOL = "iris-local-preannotation-v1"
MAX_FRAMES = 25
FRAME_FIELDS = ("id", "session_id", "asset_id", "sha256", "path", "width", "height", "taxonomy_id")
ACTIVE = {"queued", "running"}


class PreannotationConflict(RuntimeError):
    """The confirmed preview no longer represents the queued inputs."""


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _history(conn, session_id):
    count = conn.execute(
        "SELECT COUNT(*) FROM comparisons WHERE session_id=? "
        "AND json_extract(config,'$.preannotation.protocol')=?",
        (session_id, PROTOCOL),
    ).fetchone()[0]
    latest = conn.execute(
        "SELECT id FROM comparisons WHERE session_id=? "
        "AND json_extract(config,'$.preannotation.protocol')=? "
        "ORDER BY created_at DESC,id DESC LIMIT 1",
        (session_id, PROTOCOL),
    ).fetchone()
    return {"count": count, "latest_id": latest["id"] if latest else None}


def _receipt(conn, session_id, fingerprint):
    return _decode(
        conn.execute(
            "SELECT * FROM comparisons WHERE session_id=? "
            "AND json_extract(config,'$.preannotation.protocol')=? "
            "AND json_extract(config,'$.preannotation.fingerprint')=? "
            "ORDER BY created_at,id LIMIT 1",
            (session_id, PROTOCOL, fingerprint),
        ).fetchone()
    )


def _snapshot(conn, frame_id):
    frame = _decode(conn.execute("SELECT * FROM frames WHERE id=?", (frame_id,)).fetchone())
    if frame is None:
        raise ValueError("A requested image no longer exists")
    revision = _latest(conn, frame_id)
    return (
        frame,
        revision,
        {
            "frame": {key: frame[key] for key in FRAME_FIELDS},
            "revision_id": revision["id"] if revision else None,
            "revision": revision["revision"] if revision else 0,
            "taxonomy_id": _taxonomy_id(frame, revision),
            "annotation_frame_sha256": revision["frame_sha256"] if revision else frame["sha256"],
        },
    )


def _prepare(
    store,
    session_id,
    *,
    frame_ids,
    model_id,
    threshold=0.5,
    device="cpu",
    inference_mode="full",
    tile_size=640,
    overlap=0.2,
):
    from iris.preannotation_contracts import build_contract

    if (
        not isinstance(frame_ids, list)
        or not 1 <= len(frame_ids) <= MAX_FRAMES
        or any(not isinstance(identifier, str) or not identifier for identifier in frame_ids)
        or len(set(frame_ids)) != len(frame_ids)
    ):
        raise ValueError("Choose between 1 and 25 distinct images")
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("Choose one installed local detector")
    if (
        type(threshold) not in (int, float)
        or not math.isfinite(threshold)
        or not 0 <= threshold <= 1
    ):
        raise ValueError("Detector confidence threshold must be between 0 and 1")
    if inference_mode not in {"full", "tiled"}:
        raise ValueError("Preannotation supports one full or tiled detector run")
    session = store.get("sessions", session_id)
    if session is None:
        raise KeyError(session_id)
    trained = store.get("trained_models", model_id)
    if trained and record_project(store, "trained_models", trained) != session["project_id"]:
        raise ValueError("The checkpoint must belong to this image's project")
    available = {row["id"]: row for row in inference.catalog(store.root)}
    model = available.get(model_id)
    if model is None or model["status"] != "ready":
        raise ValueError((model or {}).get("reason") or "This local detector is not ready")
    weight_hash = model.get("weight_sha256")
    if not isinstance(weight_hash, str) or len(weight_hash) != 64:
        raise ValueError("The local detector has no verified checkpoint identity")
    settings = {
        "frame_ids": frame_ids,
        "model_id": model_id,
        "threshold": float(threshold),
        "device": device,
        "inference_mode": inference_mode,
        "tile_size": tile_size,
        "overlap": overlap,
    }
    _, plan = inference._prepare_comparison(
        store,
        session_id,
        name="Local preannotation",
        frame_ids=frame_ids,
        model_ids=[model_id],
        device=device,
        inference_mode=inference_mode,
        tile_size=tile_size,
        overlap=overlap,
    )
    prepared, rows, warnings = [], [], []
    with store.connect() as conn:
        conn.execute("BEGIN")
        generation = _history(conn, session_id)
        active = [
            _decode(row)
            for row in conn.execute(
                "SELECT c.* FROM comparisons c JOIN jobs j ON j.id=c.job_id "
                "WHERE c.session_id=? AND json_extract(c.config,'$.preannotation.protocol')=? "
                "AND j.status IN ('queued','running')",
                (session_id, PROTOCOL),
            )
        ]
        for frame_id in frame_ids:
            frame, revision, snapshot = _snapshot(conn, frame_id)
            if frame["session_id"] != session_id:
                raise ValueError("Every image must belong to this session")
            taxonomy = get_taxonomy(store, snapshot["taxonomy_id"], session["project_id"])
            row = {
                "frame_id": frame_id,
                "eligible": False,
                "reason": None,
                "base_revision": snapshot["revision"],
                "taxonomy_id": taxonomy["id"],
                "source_filename": store.get("assets", frame["asset_id"])["filename"],
                "coverage": None,
            }
            contract = None
            try:
                contract = build_contract(store, model_id, taxonomy)
                row["coverage"] = {
                    key: contract[key]
                    for key in (
                        "supported_class_ids",
                        "unsupported_class_ids",
                        "coverage_complete",
                    )
                }
                if not contract["supported_class_ids"]:
                    raise ValueError("This detector covers none of this image's class definitions")
                if snapshot["annotation_frame_sha256"] != frame["sha256"]:
                    raise ValueError("The saved annotations refer to different image pixels")
                if any(frame_id in other["frame_ids"] for other in active):
                    raise ValueError(
                        "A preannotation request is already queued or running for this image"
                    )
                with inference._load_verified_frame(store, frame, frame["sha256"]):
                    pass
                row["eligible"] = True
                if not contract["coverage_complete"]:
                    warnings.append(
                        f"{row['source_filename']}: detector does not cover "
                        + ", ".join(contract["unsupported_class_ids"])
                    )
            except (ValueError, OSError) as exc:
                row["reason"] = str(exc)
            rows.append(row)
            prepared.append({"snapshot": snapshot, "contract": contract, "row": row})
    eligible_ids = {row["frame_id"] for row in rows if row["eligible"]}
    eligible_frames = [
        store.get("frames", frame_id) for frame_id in frame_ids if frame_id in eligible_ids
    ]
    work = inference._work_plan(eligible_frames, plan["lanes"], plan["inference"])
    if not eligible_frames:
        work.update(warmup_passes=0, total_forward_passes=0)
    source = {
        "model_id": model_id,
        "name": model["name"],
        "provider": "local_detector",
        "operation": "propose_boxes",
        "weight_sha256": weight_hash,
        "source_contract": output_contract(model),
    }
    preview = {
        "session_id": session_id,
        "frame_ids": frame_ids,
        "model_id": model_id,
        "threshold": float(threshold),
        "config": settings,
        "source": source,
        "work": work,
        "eligible_count": len(eligible_ids),
        "excluded_count": len(frame_ids) - len(eligible_ids),
        "frames": rows,
        "warnings": [
            *warnings,
            "Proposals require human review. An empty result does not establish absence.",
        ],
        "fingerprint": _digest(
            {
                "session_id": session_id,
                "project_id": session["project_id"],
                "settings": settings,
                "source": source,
                "prepared": prepared,
                "history": generation,
            }
        ),
    }
    return preview, prepared, plan


def preview_preannotation(store: Store, session_id: str, **settings) -> dict:
    """Inspect inputs and bounded work without loading a detector or creating records."""
    return _prepare(store, session_id, **settings)[0]


def create_preannotation(
    store: Store,
    jobs,
    session_id: str,
    *,
    name: str,
    expected_fingerprint: str,
    **settings,
) -> dict:
    if not isinstance(name, str) or not 1 <= len(name.strip()) <= 160:
        raise ValueError("Preannotation name must contain between 1 and 160 characters")
    if not isinstance(expected_fingerprint, str) or len(expected_fingerprint) != 64:
        raise ValueError("Preview the detector request before confirming it")
    with jobs.guard, store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        existing = _receipt(conn, session_id, expected_fingerprint)
        if existing:
            identifier = existing["id"]
        else:
            preview, prepared, plan = _prepare(store, session_id, **settings)
            if preview["fingerprint"] != expected_fingerprint:
                raise PreannotationConflict("The preannotation inputs changed; preview them again")
            eligible = [item for item in prepared if item["row"]["eligible"]]
            if not eligible:
                raise ValueError("No images are eligible for this detector")
            identifier, job_id, created = new_id(), new_id(), now()
            model_id = preview["model_id"]
            frozen = {
                "protocol": PROTOCOL,
                "fingerprint": expected_fingerprint,
                "requested_frame_ids": preview["frame_ids"],
                "threshold": preview["threshold"],
                "source": preview["source"],
                "settings": preview["config"],
                "frames": eligible,
                "excluded": [row for row in preview["frames"] if not row["eligible"]],
                "warnings": preview["warnings"],
            }
            config = {
                "device": preview["config"]["device"],
                "warmup": 1,
                "taxonomy": "model-specific-v1",
                "model_class_contracts": {model_id: preview["source"]["source_contract"]},
                "frame_hashes": {
                    item["snapshot"]["frame"]["id"]: item["snapshot"]["frame"]["sha256"]
                    for item in eligible
                },
                "model_hashes": {model_id: preview["source"]["weight_sha256"]},
                "protocol": inference.PROTOCOL
                if plan["inference"]["mode"] == "full"
                else inference.TILED_PROTOCOL,
                "inference": plan["inference"],
                "lanes": plan["lanes"],
                "work": preview["work"],
                "preannotation": frozen,
            }
            frame_ids = [item["snapshot"]["frame"]["id"] for item in eligible]
            conn.execute(
                "INSERT INTO jobs (id,kind,status,params,message,created_at) VALUES (?,?,?,?,?,?)",
                (
                    job_id,
                    "infer",
                    "queued",
                    json.dumps({"comparison_id": identifier}),
                    "Waiting for local detector proposals",
                    created,
                ),
            )
            conn.execute(
                "INSERT INTO comparisons "
                "(id,session_id,name,frame_ids,model_ids,config,job_id,created_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (
                    identifier,
                    session_id,
                    name.strip(),
                    json.dumps(frame_ids),
                    json.dumps([model_id]),
                    json.dumps(config, allow_nan=False),
                    job_id,
                    created,
                ),
            )
    return preannotation_detail(store, identifier)


def validate_preannotation(store, comparison):
    """Recheck the frozen source before model loading; live annotation edits stay conflicts."""
    from iris.preannotation_contracts import build_contract, validate_contract

    config = comparison["config"]
    frozen = config.get("preannotation")
    if not isinstance(frozen, dict) or frozen.get("protocol") != PROTOCOL:
        raise ValueError("Unsupported saved preannotation protocol")
    job = store.get("jobs", comparison["job_id"])
    if (
        job is None
        or job["kind"] != "infer"
        or job["params"].get("comparison_id") != comparison["id"]
    ):
        raise ValueError("The preannotation job provenance is inconsistent")
    if job["status"] not in ACTIVE:
        raise ValueError("A terminal preannotation is immutable; prepare a new explicit request")
    if len(comparison["model_ids"]) != 1 or config["inference"]["mode"] not in {"full", "tiled"}:
        raise ValueError("Saved preannotation must use one full or tiled detector")
    model_id = comparison["model_ids"][0]
    if (
        frozen["source"]["model_id"] != model_id
        or frozen["source"]["weight_sha256"] != config["model_hashes"].get(model_id)
        or frozen["source"]["source_contract"] != config["model_class_contracts"].get(model_id)
        or [item["snapshot"]["frame"]["id"] for item in frozen["frames"]] != comparison["frame_ids"]
    ):
        raise ValueError("Saved preannotation provenance is inconsistent")
    for item in frozen["frames"]:
        checked = validate_contract(item["contract"])
        if build_contract(store, model_id, checked["taxonomy"]) != item["contract"]:
            raise ValueError("The detector class contract changed since preannotation was queued")
        snapshot = item["snapshot"]
        if (
            checked["source_contract"] != frozen["source"]["source_contract"]
            or checked["taxonomy_id"] != snapshot["taxonomy_id"]
            or config["frame_hashes"].get(snapshot["frame"]["id"]) != snapshot["frame"]["sha256"]
        ):
            raise ValueError("Saved preannotation classes or image identity are inconsistent")


def _publication_receipt(store, prediction_id, receipt):
    with store.connect() as conn:
        current = _decode(
            conn.execute("SELECT * FROM predictions WHERE id=?", (prediction_id,)).fetchone()
        )
        conn.execute(
            "UPDATE predictions SET metadata=? WHERE id=?",
            (
                json.dumps({**current["metadata"], "preannotation": receipt}, allow_nan=False),
                prediction_id,
            ),
        )


def save_preannotation_prediction(store, comparison, run, frame, output, timing, cancelled):
    """Persist detector evidence first, then atomically publish guarded pending proposals."""
    from iris.preannotation_contracts import normalize_candidates

    frozen = comparison["config"]["preannotation"]
    item = next(item for item in frozen["frames"] if item["snapshot"]["frame"]["id"] == frame["id"])
    snapshot = item["snapshot"]
    receipt = {
        "state": "raw_saved",
        "reason": "Detector output saved; proposals not yet published",
        "proposal_count": 0,
        "filtered_count": 0,
        "unmapped_count": 0,
        "base_revision": snapshot["revision"],
        "taxonomy_id": snapshot["taxonomy_id"],
    }
    metadata = deepcopy(output.get("metadata", {})) if isinstance(output, dict) else {}
    metadata = metadata if isinstance(metadata, dict) else {}
    invalid = None
    try:
        serialized = json.dumps(output, allow_nan=False)
        if len(serialized.encode()) > 2 * 1024 * 1024:
            raise ValueError("Detector output exceeds the 2 MiB evidence limit")
        metadata["raw_output"] = deepcopy(output)
        inference._validate_prediction(output, frame)
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        invalid = str(exc) or "Detector returned invalid output"
        if "raw_output" not in metadata:
            metadata = {"raw_output_summary": repr(output)[:4000], "raw_output_not_json": True}
    prediction = store.insert(
        "predictions",
        {
            "id": new_id(),
            "comparison_id": comparison["id"],
            "run_id": run["id"],
            "frame_id": frame["id"],
            "model_id": run["model_id"],
            "detections": [] if invalid else output["detections"],
            "input_size": [frame["width"], frame["height"]],
            "metadata": {**metadata, "preannotation": receipt},
            "timing": timing
            if not invalid
            else {
                key: 0.0
                for key in (
                    "decode_ms",
                    "preprocess_ms",
                    "inference_ms",
                    "postprocess_ms",
                    "total_ms",
                )
            },
            "created_at": now(),
        },
    )
    try:
        if invalid:
            raise ValueError(invalid)
        normalized = normalize_candidates(
            output["detections"],
            item["contract"],
            frozen["threshold"],
            width=frame["width"],
            height=frame["height"],
        )
        proposals = normalized["proposals"]
        receipt.update(
            filtered_count=normalized["filtered_count"],
            unmapped_count=normalized["unmapped_count"],
            warnings=normalized["warnings"],
        )
        if len(proposals) > MAX_DETECTOR_SUGGESTIONS:
            raise ValueError(
                "More than 100 detector proposals; create a new request with a higher threshold"
            )
        if cancelled():
            receipt.update(state="cancelled", reason="Request stopped after saving detector output")
        else:
            with store.connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                current_job = conn.execute(
                    "SELECT kind,params,status,cancel_requested FROM jobs WHERE id=?",
                    (comparison["job_id"],),
                ).fetchone()
                if (
                    current_job is None
                    or current_job["kind"] != "infer"
                    or json.loads(current_job["params"]).get("comparison_id") != comparison["id"]
                ):
                    raise PreannotationConflict("The preannotation job provenance changed")
                if current_job["status"] not in ACTIVE or current_job["cancel_requested"]:
                    receipt.update(
                        state="cancelled", reason="Request stopped after saving detector output"
                    )
                else:
                    current_frame, _, current = _snapshot(conn, frame["id"])
                    if current != snapshot:
                        raise PreannotationConflict(
                            "Image pixels, classes or human annotation revision changed; "
                            "review the saved output explicitly"
                        )
                    try:
                        with inference._load_verified_frame(
                            store, current_frame, snapshot["frame"]["sha256"]
                        ):
                            pass
                    except (ValueError, OSError) as exc:
                        raise PreannotationConflict(str(exc)) from exc
                    for proposal in proposals:
                        identity = f"detector:{prediction['id']}:{proposal['source_index']}"
                        if snapshot["taxonomy_id"] != "iris-objects-v1":
                            identity += f":{snapshot['taxonomy_id']}"
                        proposal_metadata = {
                            "prediction_id": prediction["id"],
                            "detection_index": proposal["source_index"],
                            "comparison_id": comparison["id"],
                            "run_id": run["id"],
                            "model_id": run["model_id"],
                            "model_metadata": run["metadata"],
                            "score": proposal["score"],
                            "threshold": frozen["threshold"],
                            "original_label_id": proposal["original_label_id"],
                            "original_label": proposal["original_label"],
                            "source_taxonomy": item["contract"]["source_contract"]["taxonomy_id"],
                            "target_taxonomy": snapshot["taxonomy_id"],
                            "frame_sha256": snapshot["frame"]["sha256"],
                            "preannotation_id": comparison["id"],
                            "base_revision": snapshot["revision"],
                            "base_revision_id": snapshot["revision_id"],
                            "contract": item["contract"],
                            "geometry": proposal["geometry"],
                            "source": proposal["source"],
                        }
                        conn.execute(
                            "INSERT INTO annotation_suggestions "
                            "(id,frame_id,job_id,kind,label,box,metadata,created_at) "
                            "VALUES (?,?,?,?,?,?,?,?)",
                            (
                                hashlib.sha256(identity.encode()).hexdigest(),
                                frame["id"],
                                comparison["job_id"],
                                "detector",
                                proposal["label"],
                                json.dumps(proposal["box"]),
                                json.dumps(proposal_metadata, allow_nan=False),
                                now(),
                            ),
                        )
                    receipt.update(
                        state="pending_review" if proposals else "no_proposals",
                        reason=None,
                        proposal_count=len(proposals),
                    )
                conn.execute(
                    "UPDATE predictions SET metadata=? WHERE id=?",
                    (
                        json.dumps(
                            {**prediction["metadata"], "preannotation": receipt}, allow_nan=False
                        ),
                        prediction["id"],
                    ),
                )
                return receipt
    except PreannotationConflict as exc:
        receipt.update(state="conflict", reason=str(exc))
    except OSError as exc:
        receipt.update(state="conflict", reason=str(exc))
    except (ValueError, TypeError, KeyError) as exc:
        receipt.update(state="invalid_output", reason=str(exc))
    _publication_receipt(store, prediction["id"], receipt)
    return receipt


def preannotation_detail(store: Store, comparison_id: str) -> dict:
    comparison = store.get("comparisons", comparison_id)
    if (
        comparison is None
        or comparison["config"].get("preannotation", {}).get("protocol") != PROTOCOL
    ):
        raise KeyError(comparison_id)
    frozen = comparison["config"]["preannotation"]
    job = store.get("jobs", comparison["job_id"])
    predictions = store.list("predictions", comparison_id=comparison_id)
    by_frame = {prediction["frame_id"]: prediction for prediction in predictions}
    frames = []
    for item in frozen["frames"]:
        frame_id = item["snapshot"]["frame"]["id"]
        prediction = by_frame.get(frame_id)
        receipt = prediction["metadata"].get("preannotation") if prediction else None
        state = job["status"] if job["status"] != "succeeded" else "failed"
        frames.append(
            {
                **item["row"],
                "state": state,
                "reason": job["error"],
                "prediction_id": prediction["id"] if prediction else None,
                "proposal_count": 0,
                "filtered_count": 0,
                "unmapped_count": 0,
                **(receipt or {}),
            }
        )
    states = (
        "queued",
        "running",
        "pending_review",
        "no_proposals",
        "conflict",
        "invalid_output",
        "raw_saved",
        "cancelled",
        "failed",
        "interrupted",
    )
    return {
        **comparison,
        "job": job,
        "frames": frames,
        "predictions": predictions,
        "runs": store.list("runs", comparison_id=comparison_id),
        "counts": {
            "total": len(frames),
            "predictions": len(predictions),
            "proposals": sum(frame["proposal_count"] for frame in frames),
            **{state: sum(frame["state"] == state for frame in frames) for state in states},
        },
    }


def list_preannotations(store: Store, session_id: str) -> list[dict]:
    if store.get("sessions", session_id) is None:
        raise KeyError(session_id)
    rows = []
    for comparison in store.list("comparisons", session_id=session_id):
        if comparison["config"].get("preannotation", {}).get("protocol") == PROTOCOL:
            detail = preannotation_detail(store, comparison["id"])
            rows.append(
                {key: value for key, value in detail.items() if key not in {"predictions", "runs"}}
            )
    return rows
