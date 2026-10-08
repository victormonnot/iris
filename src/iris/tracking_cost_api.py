"""Explicit local pipeline measurements and unverified target-machine declarations."""

import json
from typing import Any, Literal

from fastapi import HTTPException, Request, Response
from pydantic import Field, ValidationError

from iris import tracking_costs
from iris.temporal_api import RecordId, StrictInput


class CostCreate(StrictInput):
    name: str = Field(min_length=1, max_length=160)
    lane_index: int = Field(ge=0, le=1)
    device: Literal["cpu", "cuda"]
    repeats: int = Field(ge=1, le=5)
    policy: Literal["offline_all", "simulated_latest"]
    cadence_fps: float | None = Field(default=None, ge=0.1, le=240)


class CostImport(StrictInput):
    report: dict[str, Any] = Field(max_length=64)


async def _payload(request, model, limit):
    raw = bytearray()
    async for chunk in request.stream():
        if len(raw) + len(chunk) > limit:
            raise HTTPException(409, "Tracking measurement JSON exceeds its size limit")
        raw.extend(chunk)
    depth, quoted, escaped = 0, False, False
    for character in raw:
        if quoted:
            if escaped:
                escaped = False
            elif character == 92:
                escaped = True
            elif character == 34:
                quoted = False
        elif character == 34:
            quoted = True
        elif character in (91, 123):
            depth += 1
            if depth > 32:
                raise HTTPException(409, "Tracking measurement JSON is nested too deeply")
        elif character in (93, 125):
            depth -= 1

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON keys are not supported")
            result[key] = value
        return result

    def nonfinite(_value):
        raise ValueError("Non-finite JSON numbers are not supported")

    try:
        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite)
        return model.model_validate(value)
    except ValidationError as exc:
        raise HTTPException(
            422,
            [{key: error[key] for key in ("type", "loc", "msg")} for error in exc.errors()],
        ) from exc
    except (ValueError, RecursionError) as exc:
        raise HTTPException(409, "Expected bounded finite tracking measurement JSON") from exc


def install_tracking_cost_routes(app, store, jobs, require, active_project):
    def action(function):
        try:
            return function()
        except KeyError as exc:
            raise HTTPException(404, "Tracking source or measurement not found") from exc
        except (ValueError, RuntimeError, OSError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/temporal/tracking-cost-status")
    def status():
        return tracking_costs.status()

    @app.get("/api/temporal/tracking-comparisons/{comparison_id}/cost-runs")
    def listing(comparison_id: RecordId):
        require("jobs", comparison_id)
        return action(
            lambda: tracking_costs.list_cost_runs(
                store, comparison_id, project_id=active_project.get()
            )
        )

    @app.post("/api/temporal/tracking-comparisons/{comparison_id}/cost-runs", status_code=201)
    async def create(comparison_id: RecordId, request: Request):
        require("jobs", comparison_id)
        payload = await _payload(request, CostCreate, 16384)
        return action(
            lambda: tracking_costs.create_cost_run(
                store, jobs, comparison_id, project_id=active_project.get(), **payload.model_dump()
            )
        )

    @app.post(
        "/api/temporal/tracking-comparisons/{comparison_id}/cost-runs/import", status_code=201
    )
    async def import_report(comparison_id: RecordId, request: Request):
        require("jobs", comparison_id)
        payload = await _payload(request, CostImport, tracking_costs.MAX_REPORT_BYTES)
        return action(
            lambda: tracking_costs.import_cost_report(
                store, comparison_id, payload.report, project_id=active_project.get()
            )
        )

    @app.get("/api/temporal/tracking-cost-runs/{job_id}")
    def detail(job_id: RecordId):
        require("jobs", job_id)
        return action(
            lambda: tracking_costs.get_cost_run(store, job_id, project_id=active_project.get())
        )

    @app.get("/api/temporal/tracking-cost-runs/{job_id}/report")
    def download(job_id: RecordId):
        require("jobs", job_id)
        record = action(
            lambda: tracking_costs.get_cost_run(store, job_id, project_id=active_project.get())
        )
        if record["report"] is None:
            raise HTTPException(409, "This attempt has no complete measurement report")
        return Response(
            json.dumps(record["report"], ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            media_type="application/json",
            headers={
                "Content-Disposition": f'attachment; filename="iris-tracking-cost-{job_id}.json"'
            },
        )
