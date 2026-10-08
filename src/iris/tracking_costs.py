"""Durable, complete-only measurements of a fresh local tracking pipeline."""

import json
import os
import tempfile
from pathlib import Path

from iris.store import DEFAULT_PROJECT_ID, new_id, now
from iris.temporal import _digest, _insert, _row, _text
from iris.tracking_comparisons import _checked_job, _detail
from iris.tracking_comparisons import public_job as comparison_public_job
from iris.tracking_cost_contracts import (
    MAX_REPORT_BYTES,
    cost_status,
    validate_cost_report,
    validate_cost_request,
)

KIND = "tracking_cost"
ORIGINS = {"local_worker", "imported_declaration"}


def _comparison(conn, comparison_id, *, project_id=None):
    job, _, sequence = _checked_job(conn, comparison_id)
    if project_id is not None and sequence["project_id"] != project_id:
        raise KeyError(comparison_id)
    if job["status"] != "succeeded":
        raise ValueError("Tracking cost requires a complete saved comparison")
    return _detail(job, include_report=True), sequence


def _bounded(report):
    try:
        size = len(json.dumps(report, allow_nan=False).encode("utf-8"))
    except (RecursionError, TypeError) as exc:
        raise ValueError("Tracking cost report must contain bounded JSON data") from exc
    if size > MAX_REPORT_BYTES:
        raise ValueError("Tracking cost report exceeds the supported JSON size limit")


def _report(report, comparison, config):
    _bounded(report)
    canonical = validate_cost_report(report, comparison)
    if _digest(canonical["request"]) != _digest(config):
        raise ValueError("Tracking cost report changed its frozen measurement request")
    return canonical


def _checked_cost(conn, job_id):
    job = _row(conn, "jobs", job_id)
    if job["kind"] != KIND:
        raise KeyError(job_id)
    params = job["params"]
    if not isinstance(params, dict) or set(params) != {
        "name",
        "comparison_id",
        "sequence_id",
        "comparison_sha256",
        "config",
        "origin",
    }:
        raise ValueError("Tracking cost request is invalid")
    comparison, sequence = _comparison(conn, params["comparison_id"])
    config = validate_cost_request(params["config"])
    if (
        params["name"] != _text(params["name"], "Tracking cost name")
        or params["origin"] not in ORIGINS
        or params["sequence_id"] != sequence["id"]
        or params["comparison_sha256"] != _digest(comparison["report"])
        or _digest(params["config"]) != _digest(config)
    ):
        raise ValueError("Tracking cost request no longer matches its frozen comparison")
    if params["origin"] == "imported_declaration" and job["status"] != "succeeded":
        raise ValueError("An imported declaration must be a complete, terminal report")
    if job["status"] == "succeeded":
        if job["cancel_requested"]:
            raise ValueError("A cancelled tracking cost attempt cannot publish a report")
        _report(job["result"], comparison, config)
    elif job["result"] is not None:
        raise ValueError("An unfinished tracking cost attempt cannot publish a report")
    return job, comparison, sequence


def public_job(job):
    job = comparison_public_job(job)
    if job["kind"] != KIND or job["result"] is None:
        return job
    report = job["result"]
    return {
        **job,
        "result": {
            "schema": report["schema"],
            "complete": True,
            "request": report["request"],
            "summary": report["summary"],
        },
    }


def _public(job, *, include_report):
    params = job["params"]
    return {
        "id": job["id"],
        "name": params["name"],
        "comparison_id": params["comparison_id"],
        "sequence_id": params["sequence_id"],
        "job": public_job(job),
        "report": job["result"] if include_report and job["status"] == "succeeded" else None,
        "origin": params["origin"],
    }


def status():
    return {**cost_status(), "max_report_bytes": MAX_REPORT_BYTES}


def create_cost_run(
    store,
    jobs,
    comparison_id,
    *,
    name,
    lane_index=0,
    device="cpu",
    repeats=1,
    policy="offline_all",
    cadence_fps=None,
    project_id=DEFAULT_PROJECT_ID,
):
    from iris.tracking import tracking_status

    name = _text(name, "Tracking cost name")
    config = validate_cost_request(
        {
            "lane_index": lane_index,
            "device": device,
            "repeats": repeats,
            "policy": policy,
            "cadence_fps": cadence_fps,
        }
    )
    with jobs.guard, store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        comparison, sequence = _comparison(conn, comparison_id, project_id=project_id)
        readiness = tracking_status()
        if not readiness["available"]:
            raise ValueError(readiness["installation"])
        identifier = new_id()
        _insert(
            conn,
            "jobs",
            {
                "id": identifier,
                "kind": KIND,
                "status": "queued",
                "params": {
                    "name": name,
                    "comparison_id": comparison_id,
                    "sequence_id": sequence["id"],
                    "comparison_sha256": _digest(comparison["report"]),
                    "config": config,
                    "origin": "local_worker",
                },
                "result": None,
                "created_at": now(),
                "message": "Waiting for a fresh detector and tracker measurement",
            },
        )
    return get_cost_run(store, identifier, project_id=project_id)


def import_cost_report(store, comparison_id, report, *, project_id=DEFAULT_PROJECT_ID):
    _bounded(report)
    with store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        comparison, sequence = _comparison(conn, comparison_id, project_id=project_id)
        canonical = validate_cost_report(report, comparison)
        config = validate_cost_request(canonical["request"])
        identifier, timestamp = new_id(), now()
        lane_name = comparison["report"]["lanes"][config["lane_index"]]["name"]
        _insert(
            conn,
            "jobs",
            {
                "id": identifier,
                "kind": KIND,
                "status": "succeeded",
                "params": {
                    "name": f"Imported {lane_name} {config['device']} declaration",
                    "comparison_id": comparison_id,
                    "sequence_id": sequence["id"],
                    "comparison_sha256": _digest(comparison["report"]),
                    "config": config,
                    "origin": "imported_declaration",
                },
                "result": canonical,
                "created_at": timestamp,
                "started_at": timestamp,
                "finished_at": timestamp,
                "progress": 1,
                "message": "Imported measurement declaration; execution is not authenticated",
            },
        )
    return get_cost_run(store, identifier, project_id=project_id)


def get_cost_run(store, job_id, *, project_id=DEFAULT_PROJECT_ID):
    with store.connect() as conn:
        conn.execute("BEGIN")
        job, _, sequence = _checked_cost(conn, job_id)
        if sequence["project_id"] != project_id:
            raise KeyError(job_id)
        return _public(job, include_report=True)


def list_cost_runs(store, comparison_id, *, project_id=DEFAULT_PROJECT_ID):
    with store.connect() as conn:
        conn.execute("BEGIN")
        _comparison(conn, comparison_id, project_id=project_id)
        return [
            _public(_checked_cost(conn, row["id"])[0], include_report=False)
            for row in conn.execute(
                "SELECT * FROM jobs WHERE kind=? AND json_extract(params,'$.comparison_id')=? "
                "ORDER BY created_at,id",
                (KIND, comparison_id),
            )
        ]


def run_cost_measurement(store, job_id, progress, cancelled):
    from iris.tracking_cost_runtime import TrackingCostCancelled, measure_tracking_cost

    with store.connect() as conn:
        conn.execute("BEGIN")
        job, comparison, _ = _checked_cost(conn, job_id)
    if job["status"] != "running" or job["params"]["origin"] != "local_worker" or cancelled():
        raise TrackingCostCancelled("Tracking cost measurement stopped before execution")
    report = measure_tracking_cost(
        store, comparison, **job["params"]["config"], progress=progress, cancelled=cancelled
    )
    if cancelled():
        raise TrackingCostCancelled("Tracking cost measurement stopped before publication")
    with store.connect() as conn:
        conn.execute("BEGIN")
        current, comparison, _ = _checked_cost(conn, job_id)
        if current["params"] != job["params"]:
            raise ValueError("Tracking cost inputs changed during measurement")
        report = _report(report, comparison, job["params"]["config"])
    if cancelled():
        raise TrackingCostCancelled("Tracking cost measurement stopped before publication")
    return report


def measure_to_file(store, comparison_id, destination, **settings):
    """Read existing evidence and atomically publish JSON, never overwrite a path."""
    from iris.tracking_cost_runtime import TrackingCostCancelled, measure_tracking_cost

    destination = Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Tracking cost report already exists: {destination}")
    if not destination.parent.is_dir():
        raise ValueError("The report destination directory must already exist")
    with store.connect() as conn:
        conn.execute("BEGIN")
        comparison, _ = _comparison(conn, comparison_id)
    config = validate_cost_request(
        {
            key: settings.get(key, default)
            for key, default in {
                "lane_index": 0,
                "device": "cpu",
                "repeats": 1,
                "policy": "offline_all",
                "cadence_fps": None,
            }.items()
        }
    )
    report = measure_tracking_cost(store, comparison, **settings)
    report = _report(report, comparison, config)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination.parent, prefix=".iris-cost-", delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if settings.get("cancelled") is not None and settings["cancelled"]():
            raise TrackingCostCancelled("Tracking cost measurement stopped before publication")
        os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return report


def validate_tracking_cost_records(connection):
    try:
        for row in connection.execute("SELECT id FROM jobs WHERE kind=?", (KIND,)):
            _checked_cost(connection, row["id"])
    except (KeyError, TypeError, IndexError, OverflowError) as exc:
        raise ValueError("Tracking cost evidence is invalid or missing") from exc
