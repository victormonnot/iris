"""Saved selected-object scenarios on immutable first-pass tracking observations.

Only deterministic JSON evidence is replayed. This service never executes a
tracker or detector and never modifies source observations or human references.
"""

import json
import re
import time
from copy import deepcopy

from iris.store import DEFAULT_PROJECT_ID, _decode, new_id, now
from iris.temporal import _digest, _insert, _reference_record, _row
from iris.tracking_comparisons import _checked_job
from iris.tracking_selection_contracts import canonicalize_request, selection_status
from iris.tracking_selection_engine import run_selection, validate_report
from iris.tracking_studies import _checked_study
from iris.tracking_studies import public_job as study_public_job

KIND = "tracking_selection"
MAX_FRAMES = 500
MAX_REPORT_BYTES = 48 * 1024**2


def _descriptor(value):
    if not isinstance(value, dict) or set(value) != {
        "kind",
        "job_id",
        "sequence_id",
        "profile_sha256",
    }:
        raise ValueError("Select an exact saved comparison lane or study profile")
    if value["kind"] not in ("comparison", "study"):
        raise ValueError("Selected-object sources must be a comparison or study")
    for key in ("job_id", "sequence_id"):
        text = value[key]
        if not isinstance(text, str) or not 1 <= len(text) <= 128 or text.strip() != text:
            raise ValueError("Selected-object source IDs must be bounded strings")
        try:
            text.encode("utf-8")
        except UnicodeError as exc:
            raise ValueError("Source IDs must be valid UTF-8") from exc
    if not isinstance(value["profile_sha256"], str) or not re.fullmatch(
        "[0-9a-f]{64}", value["profile_sha256"]
    ):
        raise ValueError("Choose a canonical tracker profile SHA-256")
    return deepcopy(value)


def _source(conn, descriptor, *, store=None, project_id=None):
    descriptor = _descriptor(descriptor)
    job = _row(conn, "jobs", descriptor["job_id"])
    expected_kind = "tracking_compare" if descriptor["kind"] == "comparison" else "tracking_study"
    if job["kind"] != expected_kind:
        raise KeyError(job["id"])
    if job["status"] != "succeeded":
        raise ValueError("Selected-object replay needs a complete saved source")
    inherited_dataset = None
    if descriptor["kind"] == "comparison":
        job, _, sequence = _checked_job(conn, job["id"])
        replays = [lane["report"] for lane in job["result"]["lanes"]]
    else:
        job, bundle = _checked_study(conn, job["id"], store=store, project_id=project_id)
        sources = [
            source
            for source in bundle["sources"]
            if source["sequence"]["id"] == descriptor["sequence_id"]
        ]
        if len(sources) != 1:
            raise ValueError("Choose an evaluated source from this exact saved study")
        sequence = sources[0]["sequence"]
        run = next(row for row in job["result"]["runs"] if row["sequence_id"] == sequence["id"])
        replays = run["replays"]
        inherited_dataset = {
            "dataset_id": bundle["dataset"]["id"],
            "manifest_sha256": bundle["dataset"]["manifest_sha256"],
            "split": run["split"],
        }
    if project_id is not None and sequence["project_id"] != project_id:
        raise KeyError(job["id"])
    if sequence["id"] != descriptor["sequence_id"]:
        raise ValueError("Choose a lane from the exact saved sequence")
    selected = [
        replay for replay in replays if replay["profile_sha256"] == descriptor["profile_sha256"]
    ]
    if len(selected) != 1:
        raise ValueError("Choose a profile present in this exact saved source")
    replay = selected[0]
    if not 1 <= len(replay["passes"][0]["frames"]) <= MAX_FRAMES:
        raise ValueError("Selected-object replay requires 1–500 available frames")
    binding = {
        "source": descriptor,
        "source_report_sha256": _digest(job["result"]),
        "sequence_sha256": sequence["manifest_sha256"],
        "cache_fingerprint": replay["cache"]["fingerprint"],
        "result_sha256": replay["cache"]["result_sha256"],
        "replay_sha256": _digest(replay),
        "first_pass_semantic_sha256": replay["passes"][0]["semantic_sha256"],
        "inherited_dataset": inherited_dataset,
        "reference_id": None,
        "reference_sha256": None,
        "reference_revision": None,
    }
    return {"source": descriptor, "sequence": sequence, "replay": replay, "source_binding": binding}


def _references(conn, sequence):
    result = []
    for raw in conn.execute(
        "SELECT * FROM temporal_references WHERE sequence_id=? ORDER BY revision,id",
        (sequence["id"],),
    ):
        reference = _reference_record(conn, _decode(raw), sequence)
        result.append({key: reference[key] for key in ("id", "revision", "summary", "created_at")})
    return result


def _source_context(conn, source):
    sequence = source["sequence"]
    memberships = []
    for raw in conn.execute(
        "SELECT * FROM temporal_datasets WHERE project_id=? ORDER BY created_at,id",
        (sequence["project_id"],),
    ):
        row = _decode(raw)
        for entry in row["manifest"]["entries"]:
            if entry["sequence_id"] == sequence["id"]:
                memberships.append(
                    {"dataset_id": row["id"], "name": row["name"], "split": entry["split"]}
                )
    return {
        "current_dataset_memberships": memberships,
        "independence": "No independent-test qualification is established by this scenario.",
        "repeatability": source["replay"]["repeatability"],
        "scope": "First saved replay pass only; no detector or tracker execution.",
    }


def status():
    return {**selection_status(), "max_report_bytes": MAX_REPORT_BYTES}


def source_detail(store, descriptor, *, project_id=DEFAULT_PROJECT_ID):
    with store.connect() as conn:
        conn.execute("BEGIN")
        source = _source(conn, descriptor, store=store, project_id=project_id)
        return {
            **source,
            "references": _references(conn, source["sequence"]),
            "context": _source_context(conn, source),
        }


def catalogue(store, *, project_id=DEFAULT_PROJECT_ID):
    with store.connect() as conn:
        conn.execute("BEGIN")
        result = []
        for raw in conn.execute(
            "SELECT * FROM jobs WHERE kind IN ('tracking_compare','tracking_study') "
            "AND status='succeeded' ORDER BY created_at,id"
        ):
            job = _decode(raw)
            if job["kind"] == "tracking_compare":
                sequence_id = job["params"]["sequence_id"]
                sequence = _row(conn, "temporal_sequences", sequence_id)
                if sequence["project_id"] != project_id:
                    continue
                job, _, sequence = _checked_job(conn, job["id"])
                items = [
                    (sequence, lane["report"], "comparison", lane["report"]["profile"]["algorithm"])
                    for lane in job["result"]["lanes"]
                ]
            else:
                dataset = _row(conn, "temporal_datasets", job["params"]["dataset_id"])
                if dataset["project_id"] != project_id:
                    continue
                job, bundle = _checked_study(conn, job["id"], store=store, project_id=project_id)
                sequences = {
                    source["sequence"]["id"]: source["sequence"] for source in bundle["sources"]
                }
                names = [bundle["request"]["baseline"], *bundle["request"]["candidates"]]
                items = [
                    (sequences[run["sequence_id"]], replay, "study", named["name"])
                    for run in job["result"]["runs"]
                    for replay, named in zip(run["replays"], names, strict=True)
                ]
            for sequence, replay, kind, profile_name in items:
                frame_count = len(replay["passes"][0]["frames"])
                if not 1 <= frame_count <= MAX_FRAMES:
                    continue
                descriptor = {
                    "kind": kind,
                    "job_id": job["id"],
                    "sequence_id": sequence["id"],
                    "profile_sha256": replay["profile_sha256"],
                }
                result.append(
                    {
                        "source": descriptor,
                        "name": f"{job['params']['name']} · {profile_name}",
                        "sequence_name": sequence["name"],
                        "profile": replay["profile"],
                        "frame_count": frame_count,
                        "references": _references(conn, sequence),
                        "taxonomy": sequence["manifest"]["taxonomy"],
                    }
                )
        return {"sources": result}


def _resolve(conn, payload, *, store=None, project_id=None):
    request = canonicalize_request(payload)
    source = _source(conn, request["source"], store=store, project_id=project_id)
    reference = None
    if request["evaluation"] is not None:
        row = _row(conn, "temporal_references", request["evaluation"]["reference_id"])
        if row["sequence_id"] != source["sequence"]["id"]:
            raise ValueError("Choose a reference revision from this exact source sequence")
        reference = _reference_record(conn, row, source["sequence"])
        source["source_binding"].update(
            {
                "reference_id": reference["id"],
                "reference_sha256": reference["payload_sha256"],
                "reference_revision": reference["revision"],
            }
        )
    return {
        "request": request,
        "sequence": source["sequence"],
        "replay": source["replay"],
        "reference": reference,
        "source_binding": source["source_binding"],
        "fingerprint": _digest({"request": request, "source_binding": source["source_binding"]}),
    }


def _bounded(report):
    if len(json.dumps(report, allow_nan=False).encode("utf-8")) > MAX_REPORT_BYTES:
        raise ValueError("Selected-object report exceeds the supported JSON size limit")


def _compute(bundle, checkpoint=None):
    report = run_selection(bundle, checkpoint=checkpoint)
    _bounded(report)
    return report


def preview_selection(store, payload, *, project_id=DEFAULT_PROJECT_ID):
    with store.connect() as conn:
        conn.execute("BEGIN")
        bundle = _resolve(conn, payload, store=store, project_id=project_id)
        # Pure replay also checks anchor validity and optional reference correspondence
        # before a durable attempt is admitted. It never executes native ML code.
        _compute(bundle)
        return {
            "request": bundle["request"],
            "fingerprint": bundle["fingerprint"],
            "source_binding": bundle["source_binding"],
            "frame_count": len(bundle["replay"]["passes"][0]["frames"]),
            "work": {
                "policies": 2,
                "frames": len(bundle["replay"]["passes"][0]["frames"]),
                "max_seconds": bundle["request"]["max_seconds"],
                "detector_runs": 0,
                "tracker_runs": 0,
            },
            "context": _source_context(conn, bundle),
            "reference_summary": bundle["reference"]["summary"] if bundle["reference"] else None,
        }


def _checked_selection(conn, job_id, *, store=None, project_id=None):
    job = _row(conn, "jobs", job_id)
    if job["kind"] != KIND:
        raise KeyError(job_id)
    params = job["params"]
    if not isinstance(params, dict) or set(params) != {
        "name",
        "sequence_id",
        "source_job_id",
        "request",
        "fingerprint",
    }:
        raise ValueError("Selected-object scenario request is invalid")
    bundle = _resolve(conn, params["request"], store=store, project_id=project_id)
    if (
        params["name"] != bundle["request"]["name"]
        or params["sequence_id"] != bundle["sequence"]["id"]
        or params["source_job_id"] != bundle["request"]["source"]["job_id"]
        or _digest(params["request"]) != _digest(bundle["request"])
        or params["fingerprint"] != bundle["fingerprint"]
    ):
        raise ValueError("Selected-object scenario changed its frozen source or request")
    if job["status"] == "succeeded":
        if job["cancel_requested"]:
            raise ValueError("A cancelled selected-object scenario cannot publish a report")
        _bounded(job["result"])
        validate_report(bundle, job["result"])
    elif job["result"] is not None:
        raise ValueError("An unfinished selected-object scenario cannot publish a report")
    return job, bundle


def public_job(job):
    job = study_public_job(job)
    if job["kind"] != KIND or job["result"] is None:
        return job
    report = job["result"]
    return {**job, "result": {"schema": report["schema"], "complete": True}}


def _public(job, *, include_report):
    return {
        "id": job["id"],
        "name": job["params"]["name"],
        "sequence_id": job["params"]["sequence_id"],
        "source_job_id": job["params"]["source_job_id"],
        "job": public_job(job),
        "report": job["result"] if include_report and job["status"] == "succeeded" else None,
    }


def create_selection(store, jobs, payload, *, expected_fingerprint, project_id=DEFAULT_PROJECT_ID):
    with jobs.guard, store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        bundle = _resolve(conn, payload, store=store, project_id=project_id)
        if expected_fingerprint != bundle["fingerprint"]:
            raise ValueError(
                "The selection preview changed; inspect a fresh preview before starting"
            )
        _compute(bundle)
        identifier = new_id()
        _insert(
            conn,
            "jobs",
            {
                "id": identifier,
                "kind": KIND,
                "status": "queued",
                "params": {
                    "name": bundle["request"]["name"],
                    "sequence_id": bundle["sequence"]["id"],
                    "source_job_id": bundle["request"]["source"]["job_id"],
                    "request": bundle["request"],
                    "fingerprint": bundle["fingerprint"],
                },
                "result": None,
                "created_at": now(),
                "message": "Waiting for selected-object replay of frozen observations",
            },
        )
    return get_selection(store, identifier, project_id=project_id)


def get_selection(store, job_id, *, project_id=DEFAULT_PROJECT_ID):
    with store.connect() as conn:
        conn.execute("BEGIN")
        job, _ = _checked_selection(conn, job_id, store=store, project_id=project_id)
        return _public(job, include_report=True)


def list_selections(store, *, project_id=DEFAULT_PROJECT_ID):
    with store.connect() as conn:
        conn.execute("BEGIN")
        result = []
        for row in conn.execute(
            "SELECT j.id FROM jobs j JOIN temporal_sequences s "
            "ON s.id=json_extract(j.params,'$.sequence_id') "
            "WHERE j.kind=? AND s.project_id=? ORDER BY j.created_at,j.id",
            (KIND, project_id),
        ):
            job, _ = _checked_selection(conn, row["id"], store=store, project_id=project_id)
            result.append(_public(job, include_report=False))
        return result


def run_tracking_selection(store, job_id, progress, cancelled):
    started = time.monotonic()
    with store.connect() as conn:
        conn.execute("BEGIN")
        job, bundle = _checked_selection(conn, job_id, store=store)

    def checkpoint():
        if cancelled():
            raise RuntimeError("Selected-object scenario cancelled before publication")
        if time.monotonic() - started >= bundle["request"]["max_seconds"]:
            raise ValueError("Selected-object scenario exceeded its wall-time budget")

    if job["status"] != "running":
        raise RuntimeError("Selected-object scenario is no longer running")
    checkpoint()
    progress(0.1, "Replaying the two fixed selection policies on saved observations")
    report = _compute(bundle, checkpoint=checkpoint)
    checkpoint()
    with store.connect() as conn:
        conn.execute("BEGIN")
        current, frozen = _checked_selection(conn, job_id, store=store)
        if current["params"] != job["params"]:
            raise ValueError("Selected-object scenario inputs changed during replay")
        report = validate_report(frozen, report, checkpoint=checkpoint)
    checkpoint()
    progress(1.0, "Selected-object scenario complete; no tracker or application policy changed")
    checkpoint()
    return report


def validate_tracking_selection_records(connection):
    try:
        for row in connection.execute("SELECT id FROM jobs WHERE kind=?", (KIND,)):
            _checked_selection(connection, row["id"])
    except (KeyError, TypeError, IndexError, OverflowError, RecursionError, StopIteration) as exc:
        raise ValueError("Selected-object evidence is invalid or missing") from exc
