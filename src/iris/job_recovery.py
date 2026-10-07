"""Explicit continuation of frozen local extraction plans; never resubmit model/provider work."""

import hashlib
import json
from copy import deepcopy

from iris.projects import record_project
from iris.store import DEFAULT_PROJECT_ID, Store, new_id, now
from iris.taxonomies import get_taxonomy

PROTOCOL = "iris-extraction-checkpoint-v1"
TERMINAL = {"failed", "cancelled", "interrupted"}
_STATES = {"created", "skipped_existing", "skipped_exact", "skipped_similar"}
_IDENTITY = (
    "id",
    "asset_id",
    "frame_index",
    "sha256",
    "perceptual_hash",
    "width",
    "height",
    "path",
    "taxonomy_id",
)


class RecoveryConflict(RuntimeError):
    """The preview's durable inputs changed, or another continuation already exists."""


def _hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _inventory(frames):
    return _hash(
        [
            {key: frame[key] for key in _IDENTITY}
            for frame in sorted(frames, key=lambda row: row["id"])
        ]
    )


def _planned(store, asset, config):
    if config.get("sampling_mode") == "passages":
        from iris.video_reviews import validate_passage_extraction

        plan = validate_passage_extraction(store, config.get("passages_plan"))
        if plan["asset_id"] != asset["id"] or plan["source_sha256"] != asset["sha256"]:
            raise ValueError("The passage extraction plan no longer matches its video")
        return plan
    from iris.media import preview_extraction

    return preview_extraction(store, asset["id"], config)


def prepare_extraction_contract(
    store: Store, operation_id: str, asset_id: str, config: dict
) -> dict:
    """Freeze position/taxonomy/dedup inputs at submission, without decoding the video."""
    asset = store.get("assets", asset_id)
    if asset is None or asset["kind"] != "video":
        raise ValueError("Extraction requires an imported video")
    session = store.get("sessions", asset["session_id"])
    project = store.get("projects", session["project_id"])
    contract = {
        "protocol": PROTOCOL,
        "operation_id": operation_id,
        "plan": _planned(store, asset, config),
        "source_sha256": asset["sha256"],
        "taxonomy_id": project["taxonomy_id"],
        "project_id": session["project_id"],
        "baseline_fingerprint": _inventory(store.list("frames", asset_id=asset_id)),
        "config_fingerprint": _hash(config),
    }
    contract["plan_fingerprint"] = _hash(contract)
    return contract


def _result(asset_id, contract, units, *, cancelled=False, inherited=0):
    return {
        "asset_id": asset_id,
        "sampled": len(units),
        "completed_count": len(units),
        "remaining_count": contract["plan"]["planned_count"] - len(units),
        **{state: sum(unit["state"] == state for unit in units) for state in _STATES},
        "frame_ids": [unit["frame_id"] for unit in units if unit["state"] == "created"],
        "cancelled": cancelled,
        "timestamp_basis": "frame_index / nominal_fps",
        "plan": contract["plan"],
        "inherited_completed_count": inherited,
        "checkpoint": {
            "protocol": PROTOCOL,
            "plan_fingerprint": contract["plan_fingerprint"],
            "units": deepcopy(units),
        },
    }


def _saved_pixels(store, frame):
    from PIL import Image

    from iris.media import _pixel_hash

    with Image.open(store.artifact_path(frame["path"])) as original:
        with original.convert("RGB") as image:
            if (
                image.size != (frame["width"], frame["height"])
                or _pixel_hash(image) != frame["sha256"]
            ):
                raise ValueError("A saved extraction image changed; continuation is unavailable")


def _state(store, job, *, verify_source=True):
    """Validate saved units and recognize a frame committed just before a checkpoint crash."""
    from iris.media import _verified_video_source

    params = job["params"]
    contract = params.get("extraction_contract")
    if not isinstance(contract, dict) or contract.get("protocol") != PROTOCOL:
        raise ValueError(
            "This historical extraction has no durable frozen plan; "
            "prepare a new extraction explicitly"
        )
    if _hash(
        {key: value for key, value in contract.items() if key != "plan_fingerprint"}
    ) != contract.get("plan_fingerprint"):
        raise ValueError("The frozen extraction contract changed")
    if _hash(params["config"]) != contract["config_fingerprint"]:
        raise ValueError("The frozen extraction settings changed")
    asset = store.get("assets", params["asset_id"])
    if asset is None or asset["kind"] != "video" or asset["sha256"] != contract["source_sha256"]:
        raise ValueError("The original video identity changed")
    session = store.get("sessions", asset["session_id"])
    if session["project_id"] != contract["project_id"]:
        raise ValueError("The original video moved to another project")
    get_taxonomy(store, contract["taxonomy_id"], contract["project_id"])
    if _planned(store, asset, params["config"]) != contract["plan"]:
        raise ValueError("The original video sampling plan changed")
    root = store.get("jobs", contract["operation_id"])
    if root is None or root["params"].get("extraction_contract") != contract:
        raise ValueError("The original extraction plan is unavailable or inconsistent")
    if verify_source:
        _verified_video_source(store, asset)
    positions = contract["plan"]["positions"]
    checkpoint = (job.get("result") or {}).get("checkpoint")
    units = []
    if checkpoint is not None:
        if (
            not isinstance(checkpoint, dict)
            or checkpoint.get("protocol") != PROTOCOL
            or checkpoint.get("plan_fingerprint") != contract["plan_fingerprint"]
            or not isinstance(checkpoint.get("units"), list)
            or len(checkpoint["units"]) > len(positions)
        ):
            raise ValueError("The saved extraction checkpoint is inconsistent")
        units = deepcopy(checkpoint["units"])
    frames = store.list("frames", asset_id=asset["id"])
    own = [
        frame
        for frame in frames
        if frame["extraction"].get("operation_id") == contract["operation_id"]
    ]
    baseline = [
        frame
        for frame in frames
        if frame["extraction"].get("operation_id") != contract["operation_id"]
    ]
    if _inventory(baseline) != contract["baseline_fingerprint"]:
        raise ValueError("The video's existing-frame inventory changed; prepare a new extraction")
    by_id = {frame["id"]: frame for frame in frames}
    own_ids = {frame["id"] for frame in own}
    recorded = set()
    for index, unit in enumerate(units):
        if (
            not isinstance(unit, dict)
            or unit.get("position") != index
            or type(unit.get("position")) is not int
            or unit.get("frame_index") != positions[index]["frame_index"]
            or unit.get("state") not in _STATES
            or unit.get("frame_id") not in by_id
        ):
            raise ValueError("The saved extraction position ledger is inconsistent")
        frame = by_id[unit["frame_id"]]
        if unit["state"] == "created":
            if (
                frame["id"] not in own_ids
                or frame["id"] in recorded
                or frame["frame_index"] != unit["frame_index"]
                or frame["extraction"].get("unit_index") != index
            ):
                raise ValueError("A saved extracted position no longer matches its image")
            if unit.get("sha256") != frame["sha256"]:
                raise ValueError("A saved extracted image no longer matches its checkpoint hash")
            recorded.add(frame["id"])
        elif unit["state"] == "skipped_existing" and frame["frame_index"] != unit["frame_index"]:
            raise ValueError("A skipped position no longer matches its existing image")
        elif unit["state"] == "skipped_exact" and frame["sha256"] != unit.get("sha256"):
            raise ValueError("A skipped exact duplicate no longer matches its image")
        elif unit["state"] == "skipped_similar":
            distance = (
                int(frame["perceptual_hash"], 16) ^ int(unit["perceptual_hash"], 16)
            ).bit_count()
            if (
                params["config"].get("dedup_hamming") is None
                or distance > params["config"]["dedup_hamming"]
            ):
                raise ValueError("A skipped similar image no longer matches the frozen filter")
    for frame in frames:
        _saved_pixels(store, frame)
    for frame in own:
        provenance = frame["extraction"]
        if (
            provenance.get("plan_fingerprint") != contract["plan_fingerprint"]
            or frame["taxonomy_id"] != contract["taxonomy_id"]
        ):
            raise ValueError("Saved extraction provenance differs from its frozen plan")
    unrecorded = [frame for frame in own if frame["id"] not in recorded]
    if unrecorded:
        if (
            len(unrecorded) != 1
            or len(units) >= len(positions)
            or unrecorded[0]["frame_index"] != positions[len(units)]["frame_index"]
            or unrecorded[0]["extraction"].get("unit_index") != len(units)
        ):
            raise ValueError(
                "Uncheckpointed extraction images do not match the next planned position"
            )
        units.append(
            {
                "position": len(units),
                "frame_index": unrecorded[0]["frame_index"],
                "state": "created",
                "frame_id": unrecorded[0]["id"],
                "sha256": unrecorded[0]["sha256"],
            }
        )
    return asset, contract, units, frames


def _owned_job(store, job_id, project_id):
    job = store.get("jobs", job_id)
    if job is None or record_project(store, "jobs", job) != project_id:
        raise KeyError(job_id)
    return job


def preview_job_recovery(store: Store, job_id: str, project_id: str = DEFAULT_PROJECT_ID) -> dict:
    job = _owned_job(store, job_id, project_id)
    if job["kind"] == "temporal_detect":
        from iris.temporal_detections import preview_detection_recovery

        return {
            **preview_detection_recovery(store, job_id, project_id=project_id),
            "source_job_id": job_id,
            "mode": "continue_temporal_detection",
        }
    result = {
        "source_job_id": job_id,
        "available": False,
        "mode": "continue_extraction",
        "reason": None,
        "completed_count": 0,
        "remaining_count": 0,
        "total_count": 0,
        "fingerprint": None,
        "successor_job_id": None,
    }
    if job["kind"] != "extract":
        result["reason"] = (
            "This job cannot continue in place. Preserve its partial results and prepare "
            "a new explicit launch; external requests are never resent automatically."
        )
        return result
    if job["status"] not in TERMINAL:
        result["reason"] = "Only an interrupted, cancelled or failed extraction can be continued"
        return result
    with store.connect() as conn:
        successor = conn.execute(
            "SELECT id FROM jobs WHERE json_extract(params,'$.recovery_of')=? "
            "ORDER BY created_at,id LIMIT 1",
            (job_id,),
        ).fetchone()
        active = conn.execute(
            "SELECT id FROM jobs WHERE kind='extract' AND status IN ('queued','running') "
            "AND json_extract(params,'$.asset_id')=? LIMIT 1",
            (job["params"].get("asset_id"),),
        ).fetchone()
    if successor:
        result.update(
            successor_job_id=successor["id"],
            reason="A continuation already exists; inspect that attempt",
        )
        return result
    if active:
        result["reason"] = "Another extraction is queued or running for this video"
        return result
    try:
        _, contract, units, frames = _state(store, job)
    except (
        ValueError,
        OSError,
        KeyError,
        TypeError,
        AttributeError,
        OverflowError,
        IndexError,
    ) as exc:
        result["reason"] = str(exc) or "Saved extraction inputs are unavailable"
        return result
    result.update(
        completed_count=len(units),
        remaining_count=contract["plan"]["planned_count"] - len(units),
        total_count=contract["plan"]["planned_count"],
        taxonomy_id=contract["taxonomy_id"],
        source_sha256=contract["source_sha256"],
    )
    if not result["remaining_count"]:
        result["reason"] = (
            "Every planned position is already recorded; no remaining work needs continuation"
        )
        return result
    result.update(
        available=True,
        fingerprint=_hash(
            {
                "job_id": job_id,
                "status": job["status"],
                "contract": contract,
                "units": units,
                "inventory": _inventory(frames),
            }
        ),
    )
    return result


def recover_job(
    store: Store, jobs, job_id: str, *, fingerprint: str, project_id: str = DEFAULT_PROJECT_ID
) -> dict:
    candidate = _owned_job(store, job_id, project_id)
    if candidate["kind"] == "temporal_detect":
        from iris.temporal_detections import recover_detection_cache

        return recover_detection_cache(
            store, jobs, job_id, fingerprint=fingerprint, project_id=project_id
        )
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ValueError("Provide the current extraction recovery fingerprint")
    with jobs.guard, store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        preview = preview_job_recovery(store, job_id, project_id)
        if not preview["available"] or preview["fingerprint"] != fingerprint:
            raise RecoveryConflict(
                preview["reason"] or "Extraction inputs changed since the recovery preview"
            )
        parent = _owned_job(store, job_id, project_id)
        _, contract, units, _ = _state(store, parent)
        identifier = new_id()
        params = deepcopy(parent["params"])
        params.update(recovery_of=job_id, inherited_completed_count=len(units))
        result = _result(params["asset_id"], contract, units, inherited=len(units))
        conn.execute(
            "INSERT INTO jobs (id,kind,status,params,result,progress,message,created_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (
                identifier,
                "extract",
                "queued",
                json.dumps(params),
                json.dumps(result),
                len(units) / contract["plan"]["planned_count"],
                "Waiting to continue the remaining frozen extraction positions",
                now(),
            ),
        )
    return store.get("jobs", identifier)


def _persist_checkpoint(store, job_id, result):
    with store.connect() as conn:
        conn.execute(
            "UPDATE jobs SET result=?,progress=? WHERE id=? AND status IN ('queued','running')",
            (
                json.dumps(result),
                result["completed_count"] / result["plan"]["planned_count"],
                job_id,
            ),
        )


def run_durable_extraction(store: Store, job_id: str, progress, cancelled) -> dict:
    import cv2
    from PIL import Image

    from iris.media import _frame_record, _perceptual_hash, _pixel_hash, _video_position_error

    job = store.get("jobs", job_id)
    if job is None or job["kind"] != "extract":
        raise ValueError("Extraction job not found")
    if job["status"] not in {"queued", "running"}:
        raise ValueError("A terminal extraction is immutable; create an explicit continuation")
    if cancelled():
        return {**(job["result"] or {}), "cancelled": True}
    asset, contract, units, frames = _state(store, job)
    inherited = job["params"].get("inherited_completed_count", 0)
    result = _result(asset["id"], contract, units, inherited=inherited)
    _persist_checkpoint(store, job_id, result)
    existing = {frame["frame_index"]: frame for frame in frames}
    exact = {frame["sha256"]: frame for frame in frames}
    config = job["params"]["config"]
    provenance = dict(config)
    plan = contract["plan"]
    if plan["sampling_mode"] == "passages":
        provenance.update(
            sampling_algorithm=plan["algorithm"],
            video_review_id=plan["video_review_id"],
            passage_ids=plan["passage_ids"],
        )
    elif plan["sampling_mode"] == "uniform":
        from iris.media import ALGORITHM

        provenance["sampling_algorithm"] = ALGORITHM
        provenance["sampling_plan"] = {
            key: plan[key]
            for key in (
                "planned_count",
                "start_seconds",
                "end_seconds",
                "first_timestamp_seconds",
                "last_timestamp_seconds",
            )
        }
    dedup = config.get("dedup_hamming")
    capture = cv2.VideoCapture(str(store.artifact_path(asset["path"])))
    try:
        if not capture.isOpened():
            raise ValueError("The original video is missing or unreadable")
        for index in range(len(units), contract["plan"]["planned_count"]):
            if cancelled():
                break
            target = contract["plan"]["positions"][index]["frame_index"]
            unit = {"position": index, "frame_index": target}
            if target in existing:
                unit.update(state="skipped_existing", frame_id=existing[target]["id"])
            else:
                if not capture.set(cv2.CAP_PROP_POS_FRAMES, target):
                    raise _video_position_error("seek to", target)
                ok, pixels = capture.read()
                if not ok or pixels is None:
                    raise _video_position_error("decode", target)
                if cancelled():
                    break
                with Image.fromarray(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)) as image:
                    digest, phash = _pixel_hash(image), _perceptual_hash(image)
                    similar = next(
                        (
                            frame
                            for frame in frames
                            if dedup is not None
                            and frame["perceptual_hash"]
                            and (int(frame["perceptual_hash"], 16) ^ int(phash, 16)).bit_count()
                            <= dedup
                        ),
                        None,
                    )
                    if digest in exact:
                        unit.update(
                            state="skipped_exact", frame_id=exact[digest]["id"], sha256=digest
                        )
                    elif similar:
                        unit.update(
                            state="skipped_similar", frame_id=similar["id"], perceptual_hash=phash
                        )
                    else:
                        if cancelled():
                            break
                        frame = _frame_record(
                            store,
                            asset,
                            image,
                            frame_index=target,
                            timestamp_seconds=target / contract["plan"]["fps"],
                            taxonomy_id=contract["taxonomy_id"],
                            extraction={
                                **provenance,
                                "job_id": job_id,
                                "operation_id": contract["operation_id"],
                                "plan_fingerprint": contract["plan_fingerprint"],
                                "unit_index": index,
                            },
                        )
                        frames.append(frame)
                        existing[target], exact[digest] = frame, frame
                        unit.update(state="created", frame_id=frame["id"], sha256=frame["sha256"])
            units.append(unit)
            result = _result(asset["id"], contract, units, inherited=inherited)
            _persist_checkpoint(store, job_id, result)
            progress(
                len(units) / contract["plan"]["planned_count"],
                f"{len(units)} / {contract['plan']['planned_count']} positions recorded; "
                f"{result['created']} images preserved",
            )
        return {**result, "cancelled": cancelled()}
    finally:
        capture.release()
