"""Bounded profile studies on frozen development evidence; no automatic application."""

import json
import time

from iris.store import DEFAULT_PROJECT_ID, _decode, new_id, now
from iris.temporal import (
    _dataset_record,
    _digest,
    _insert,
    _reference_record,
    _reserve_sequence,
    _row,
    _sequence_record,
    temporal_reservations,
)
from iris.tracking_comparisons import _checked_job, _complete_inputs, _detail
from iris.tracking_contracts import profile_hash
from iris.tracking_cost_contracts import detector_recipe
from iris.tracking_costs import public_job as cost_public_job
from iris.tracking_metrics import evaluate_quality
from iris.tracking_study_contracts import (
    budget_for,
    canonicalize_request,
    study_status,
    validate_report,
)
from iris.tracking_study_contracts import (
    suggestions as suggestions,
)

KIND = "tracking_study"
MAX_REPORT_BYTES = 48 * 1024**2


def _resolve(conn, payload, *, store=None, project_id=None):
    request = canonicalize_request(payload)
    dataset = _dataset_record(conn, _row(conn, "temporal_datasets", request["dataset_id"]))
    if project_id is not None and dataset["project_id"] != project_id:
        raise KeyError(dataset["id"])
    entries = dataset["manifest"]["entries"]
    active = [entry for entry in entries if entry["split"] != "test"]
    if not 1 <= len(active) <= 4:
        raise ValueError("A study requires 1–4 development or validation sequences")
    selected = {source["sequence_id"]: source["comparison_id"] for source in request["sources"]}
    if set(selected) != {entry["sequence_id"] for entry in active}:
        raise ValueError("Select every non-test dataset sequence exactly once; test is reserved")
    if store is not None:
        from iris.datasets import _reservation_state

        reservations = _reservation_state(store, conn, dataset["project_id"])
    else:
        # Archive validation also checks image/import manifest reservations using its
        # verified archive inventory, after this database-only evidence validation.
        reservations = temporal_reservations(conn, dataset["project_id"])
    for entry in entries:
        sequence = _row(conn, "temporal_sequences", entry["sequence_id"])
        _reserve_sequence(sequence["manifest"], entry["split"], *reservations, in_project=True)
    profiles = [request["baseline"]["profile"], *[row["profile"] for row in request["candidates"]]]
    baseline_hash = profile_hash(profiles[0])
    recipe, sources, pins = None, [], []
    for entry in active:
        if entry["reference_id"] is None:
            raise ValueError("Every development sequence needs a frozen reference revision")
        job, cache, sequence = _checked_job(conn, selected[entry["sequence_id"]])
        if sequence["project_id"] != dataset["project_id"]:
            raise KeyError(job["id"])
        if sequence["id"] != entry["sequence_id"] or job["status"] != "succeeded":
            raise ValueError("Choose a completed comparison from each exact dataset sequence")
        cache, sequence, frames, attempts, coverage = _complete_inputs(conn, cache["id"])
        comparison = _detail(job, include_report=True)
        if baseline_hash not in {profile_hash(profile) for profile in job["params"]["profiles"]}:
            raise ValueError("The baseline must match a saved comparison lane on every source")
        current_recipe = detector_recipe(cache["config"]["detector"])
        if recipe is not None and current_recipe != recipe:
            raise ValueError("Study sources must share the same native detector recipe")
        recipe = current_recipe
        known = {item["id"] for item in recipe["classes"]}
        for profile in profiles:
            if not set(profile["class_ids"]) <= known:
                raise ValueError("Study profiles contain classes absent from the detector cache")
            if recipe["min_score"] > profile["low_threshold"]:
                raise ValueError("The detector cache score floor cannot support this profile")
        reference = _reference_record(
            conn, _row(conn, "temporal_references", entry["reference_id"]), sequence
        )
        # Check the explicit class correspondence and reference eligibility before
        # admitting expensive work. T6 preserves unavailable metrics as unavailable.
        evaluate_quality(
            comparison,
            reference,
            class_mapping=request["class_mapping"],
            iou_threshold=request["iou_threshold"],
        )
        sources.append(
            {
                "entry": entry,
                "sequence": sequence,
                "cache": cache,
                "comparison": comparison,
                "reference": reference,
                "frames": frames,
                "attempts": attempts,
            }
        )
        pins.append(
            {
                "sequence_id": sequence["id"],
                "comparison_id": job["id"],
                "comparison_sha256": _digest(comparison["report"]),
                "cache_fingerprint": cache["fingerprint"],
                "result_sha256": coverage["result_sha256"],
                "reference_sha256": reference["payload_sha256"],
            }
        )
    budget = budget_for(request, sources)
    if budget["required_updates"] > request["max_updates"]:
        raise ValueError("The requested study exceeds its explicit update budget")
    return {
        "request": request,
        "dataset": dataset,
        "sources": sources,
        "fingerprint": _digest(
            {"request": request, "dataset_sha256": dataset["manifest_sha256"], "sources": pins}
        ),
        "budget": budget,
    }


def _preview(bundle):
    entries = bundle["dataset"]["manifest"]["entries"]
    splits = {
        split: sum(row["split"] == split for row in entries) for split in ("train", "val", "test")
    }
    return {
        "request": bundle["request"],
        "fingerprint": bundle["fingerprint"],
        "budget": bundle["budget"],
        "coverage": {
            "development_only": splits["val"] == 0,
            "active_sequence_count": len(bundle["sources"]),
            "reserved_test_count": splits["test"],
            "splits": splits,
            "sources": [
                {
                    "sequence_id": source["sequence"]["id"],
                    "name": source["sequence"]["name"],
                    "split": source["entry"]["split"],
                    "reference_id": source["reference"]["id"],
                    "reference_summary": source["reference"]["summary"],
                }
                for source in bundle["sources"]
            ],
        },
    }


def status():
    from iris.tracking import tracking_status

    return {**study_status(), "runtime": tracking_status(), "max_report_bytes": MAX_REPORT_BYTES}


def catalogue(store, *, project_id=DEFAULT_PROJECT_ID):
    with store.connect() as conn:
        conn.execute("BEGIN")
        datasets = []
        for raw in conn.execute(
            "SELECT * FROM temporal_datasets WHERE project_id=? ORDER BY created_at,id",
            (project_id,),
        ):
            dataset = _dataset_record(conn, _decode(raw))
            datasets.append(
                {
                    "id": dataset["id"],
                    "name": dataset["name"],
                    "manifest_sha256": dataset["manifest_sha256"],
                    "entries": dataset["manifest"]["entries"],
                }
            )
        sequences = []
        for raw in conn.execute(
            "SELECT * FROM temporal_sequences WHERE project_id=? ORDER BY created_at,id",
            (project_id,),
        ):
            sequence = _sequence_record(conn, _decode(raw))
            references, comparisons = [], []
            for refraw in conn.execute(
                "SELECT * FROM temporal_references WHERE sequence_id=? ORDER BY revision,id",
                (sequence["id"],),
            ):
                reference = _reference_record(conn, _decode(refraw), sequence)
                references.append(
                    {key: reference[key] for key in ("id", "revision", "summary", "created_at")}
                )
            for jobraw in conn.execute(
                "SELECT id FROM jobs WHERE kind='tracking_compare' AND status='succeeded' "
                "AND json_extract(params,'$.sequence_id')=? ORDER BY created_at,id",
                (sequence["id"],),
            ):
                job, cache, _ = _checked_job(conn, jobraw["id"])
                detector = cache["config"]["detector"]
                comparisons.append(
                    {
                        "id": job["id"],
                        "name": job["params"]["name"],
                        "cache_id": cache["id"],
                        "profiles": job["params"]["profiles"],
                        "classes": detector["classes"],
                        "detector": {key: detector[key] for key in ("classes", "class_contract")},
                    }
                )
            sequences.append(
                {
                    "id": sequence["id"],
                    "name": sequence["name"],
                    "frame_count": len(sequence["manifest"]["frames"]),
                    "taxonomy": sequence["manifest"]["taxonomy"],
                    "references": references,
                    "comparisons": comparisons,
                }
            )
        return {"datasets": datasets, "sequences": sequences}


def preview_study(store, payload, *, project_id=DEFAULT_PROJECT_ID):
    with store.connect() as conn:
        conn.execute("BEGIN")
        return _preview(_resolve(conn, payload, store=store, project_id=project_id))


def _bounded(report):
    if len(json.dumps(report, allow_nan=False).encode("utf-8")) > MAX_REPORT_BYTES:
        raise ValueError("Tracking study report exceeds the supported JSON size limit")


def _checked_study(conn, job_id, *, store=None, project_id=None):
    job = _row(conn, "jobs", job_id)
    if job["kind"] != KIND:
        raise KeyError(job_id)
    params = job["params"]
    if not isinstance(params, dict) or set(params) != {
        "name",
        "dataset_id",
        "request",
        "fingerprint",
    }:
        raise ValueError("Tracking study request is invalid")
    bundle = _resolve(conn, params["request"], store=store, project_id=project_id)
    if (
        params["name"] != bundle["request"]["name"]
        or params["dataset_id"] != bundle["dataset"]["id"]
        or params["request"] != bundle["request"]
        or params["fingerprint"] != bundle["fingerprint"]
    ):
        raise ValueError("Tracking study changed its frozen sources or request")
    if job["status"] == "succeeded":
        if job["cancel_requested"]:
            raise ValueError("A cancelled tracking study cannot publish a report")
        _bounded(job["result"])
        validate_report(bundle, job["result"])
    elif job["result"] is not None:
        raise ValueError("An unfinished tracking study cannot publish a report")
    return job, bundle


def public_job(job):
    job = cost_public_job(job)
    if job["kind"] != KIND or job["result"] is None:
        return job
    report = job["result"]
    return {
        **job,
        "result": {"schema": report["schema"], "complete": True, "summary": report["summary"]},
    }


def _public(job, *, include_report):
    return {
        "id": job["id"],
        "name": job["params"]["name"],
        "dataset_id": job["params"]["dataset_id"],
        "job": public_job(job),
        "report": job["result"] if include_report and job["status"] == "succeeded" else None,
    }


def create_study(store, jobs, payload, *, expected_fingerprint, project_id=DEFAULT_PROJECT_ID):
    from iris.tracking import tracking_status

    with jobs.guard, store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        bundle = _resolve(conn, payload, store=store, project_id=project_id)
        if expected_fingerprint != bundle["fingerprint"]:
            raise ValueError("The study preview changed; inspect a fresh preview before starting")
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
                    "name": bundle["request"]["name"],
                    "dataset_id": bundle["dataset"]["id"],
                    "request": bundle["request"],
                    "fingerprint": bundle["fingerprint"],
                },
                "result": None,
                "created_at": now(),
                "message": "Waiting for bounded profile replays; test sequences remain reserved",
            },
        )
    return get_study(store, identifier, project_id=project_id)


def get_study(store, job_id, *, project_id=DEFAULT_PROJECT_ID):
    with store.connect() as conn:
        conn.execute("BEGIN")
        job, _ = _checked_study(conn, job_id, store=store, project_id=project_id)
        return _public(job, include_report=True)


def list_studies(store, *, project_id=DEFAULT_PROJECT_ID):
    with store.connect() as conn:
        conn.execute("BEGIN")
        result = []
        for row in conn.execute(
            "SELECT j.id FROM jobs j JOIN temporal_datasets d "
            "ON d.id=json_extract(j.params,'$.dataset_id') "
            "WHERE j.kind=? AND d.project_id=? ORDER BY j.created_at,j.id",
            (KIND, project_id),
        ):
            job, _ = _checked_study(conn, row["id"], store=store, project_id=project_id)
            result.append(_public(job, include_report=False))
        return result


def run_tracking_study(store, job_id, progress, cancelled):
    from iris.tracking_study_runtime import (
        TrackingStudyBudgetExceeded,
        TrackingStudyCancelled,
        run_study,
    )

    started = time.monotonic()
    with store.connect() as conn:
        conn.execute("BEGIN")
        job, bundle = _checked_study(conn, job_id, store=store)

    def stopped():
        if cancelled():
            return True
        if time.monotonic() - started >= bundle["request"]["max_seconds"]:
            raise TrackingStudyBudgetExceeded("Tracking study exceeded its wall-time budget")
        return False

    def checkpoint():
        if stopped():
            raise TrackingStudyCancelled("Tracking study stopped before publication")

    if job["status"] != "running" or stopped():
        raise TrackingStudyCancelled("Tracking study stopped before execution")
    report = run_study(store, bundle, progress=progress, cancelled=stopped)
    if stopped():
        raise TrackingStudyCancelled("Tracking study stopped before publication")
    _bounded(report)
    with store.connect() as conn:
        conn.execute("BEGIN")
        current, frozen = _checked_study(conn, job_id, store=store)
        if current["params"] != job["params"]:
            raise ValueError("Tracking study inputs changed during replay")
        report = validate_report(frozen, report, checkpoint=checkpoint)
    if stopped():
        raise TrackingStudyCancelled("Tracking study stopped before publication")
    return report


def validate_tracking_study_records(connection):
    try:
        for row in connection.execute("SELECT id FROM jobs WHERE kind=?", (KIND,)):
            _checked_study(connection, row["id"])
    except (KeyError, TypeError, IndexError, OverflowError, RecursionError) as exc:
        raise ValueError("Tracking study evidence is invalid or missing") from exc
