"""Explicit, project-scoped selected-object scenarios from frozen JSON evidence."""

import json
from typing import Literal

from fastapi import HTTPException, Request, Response
from pydantic import Field

from iris import tracking_selections
from iris.temporal_api import Identifier, RecordId, StrictInput
from iris.tracking_cost_api import _payload


class SelectionSource(StrictInput):
    kind: Literal["comparison", "study"]
    job_id: Identifier
    sequence_id: Identifier
    profile_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class SelectionAnchor(StrictInput):
    frame_id: Identifier
    detection_index: int = Field(ge=0, le=2**31 - 1)


class SelectionPolicy(StrictInput):
    policy_schema: Literal["iris-selection-policy-v1"] = Field(alias="schema")
    min_score: float
    min_iou: float
    max_center_distance: float
    max_area_ratio: float
    max_lost_seconds: float | None
    max_lost_updates: int
    recovery_confirmation_updates: int


class SelectionEvaluation(StrictInput):
    reference_id: Identifier
    identity_id: Identifier
    class_mapping: dict[str, str | None] = Field(min_length=1, max_length=100)
    iou_threshold: float


class SelectionRequest(StrictInput):
    name: str = Field(min_length=1, max_length=160)
    source: SelectionSource
    selection: SelectionAnchor
    release_frame_id: Identifier | None
    policy: SelectionPolicy
    evaluation: SelectionEvaluation | None
    max_seconds: float = Field(ge=1, le=120)


class SelectionCreate(SelectionRequest):
    expected_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


def install_tracking_selection_routes(app, store, jobs, require, active_project):
    def action(function):
        try:
            return function()
        except KeyError as exc:
            raise HTTPException(404, "Selected-object scenario or frozen source not found") from exc
        except (ValueError, RuntimeError, OSError) as exc:
            raise HTTPException(409, str(exc)) from exc

    def check_source(source):
        require("jobs", source.job_id)
        require("temporal_sequences", source.sequence_id)

    def check_request(payload):
        check_source(payload.source)
        if payload.evaluation is not None:
            require("temporal_references", payload.evaluation.reference_id)

    @app.get("/api/temporal/tracking-selection-status")
    def status():
        return action(tracking_selections.status)

    @app.get("/api/temporal/tracking-selection-sources")
    def sources():
        return action(lambda: tracking_selections.catalogue(store, project_id=active_project.get()))

    @app.post("/api/temporal/tracking-selection-source")
    async def source(request: Request):
        payload = await _payload(request, SelectionSource, 16384)
        check_source(payload)
        return action(
            lambda: tracking_selections.source_detail(
                store, payload.model_dump(by_alias=True), project_id=active_project.get()
            )
        )

    @app.post("/api/temporal/tracking-selections/preview")
    async def preview(request: Request):
        payload = await _payload(request, SelectionRequest, 65536)
        check_request(payload)
        return action(
            lambda: tracking_selections.preview_selection(
                store, payload.model_dump(by_alias=True), project_id=active_project.get()
            )
        )

    @app.post("/api/temporal/tracking-selections", status_code=201)
    async def create(request: Request):
        payload = await _payload(request, SelectionCreate, 65536)
        check_request(payload)
        values = payload.model_dump(by_alias=True)
        expected = values.pop("expected_fingerprint")
        return action(
            lambda: tracking_selections.create_selection(
                store, jobs, values, expected_fingerprint=expected, project_id=active_project.get()
            )
        )

    @app.get("/api/temporal/tracking-selections")
    def listing():
        return action(
            lambda: tracking_selections.list_selections(store, project_id=active_project.get())
        )

    @app.get("/api/temporal/tracking-selections/{job_id}")
    def detail(job_id: RecordId):
        require("jobs", job_id)
        return action(
            lambda: tracking_selections.get_selection(
                store, job_id, project_id=active_project.get()
            )
        )

    @app.get("/api/temporal/tracking-selections/{job_id}/report")
    def download(job_id: RecordId):
        require("jobs", job_id)
        record = action(
            lambda: tracking_selections.get_selection(
                store, job_id, project_id=active_project.get()
            )
        )
        if record["report"] is None:
            raise HTTPException(409, "This attempt has no complete selected-object report")
        return Response(
            json.dumps(record["report"], ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            media_type="application/json",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="iris-tracking-selection-{job_id}.json"'
                )
            },
        )
