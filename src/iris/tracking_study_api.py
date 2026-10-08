"""Explicit preview and complete-only tracking profile studies."""

import json
from typing import Any

from fastapi import HTTPException, Request, Response
from pydantic import Field

from iris import tracking_studies
from iris.temporal_api import Identifier, RecordId, StrictInput
from iris.tracking_cost_api import _payload


class StudySource(StrictInput):
    sequence_id: Identifier
    comparison_id: Identifier


class StudyProfile(StrictInput):
    name: str = Field(min_length=1, max_length=80)
    profile: dict[str, Any] = Field(min_length=1, max_length=32)


class StudyRequest(StrictInput):
    name: str = Field(min_length=1, max_length=160)
    dataset_id: Identifier
    sources: list[StudySource] = Field(min_length=1, max_length=4)
    baseline: StudyProfile
    candidates: list[StudyProfile] = Field(min_length=1, max_length=7)
    class_mapping: dict[str, str | None] = Field(min_length=1, max_length=100)
    iou_threshold: float = Field(default=0.5, ge=1e-12, le=1)
    repeats: int = Field(default=2, ge=1, le=3)
    max_updates: int = Field(default=5000, ge=1, le=20000)
    max_seconds: float = Field(default=120, ge=1, le=600)


class StudyCreate(StudyRequest):
    expected_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


class SuggestionsRequest(StrictInput):
    baseline_profile: dict[str, Any] = Field(min_length=1, max_length=32)


def install_tracking_study_routes(app, store, jobs, require, active_project):
    def action(function):
        try:
            return function()
        except KeyError as exc:
            raise HTTPException(404, "Tracking study or frozen source not found") from exc
        except (ValueError, RuntimeError, OSError) as exc:
            raise HTTPException(409, str(exc)) from exc

    def check_sources(payload):
        require("temporal_datasets", payload.dataset_id)
        for source in payload.sources:
            require("temporal_sequences", source.sequence_id)
            require("jobs", source.comparison_id)

    @app.get("/api/temporal/tracking-study-status")
    def status():
        return action(tracking_studies.status)

    @app.get("/api/temporal/tracking-study-sources")
    def sources():
        return action(lambda: tracking_studies.catalogue(store, project_id=active_project.get()))

    @app.post("/api/temporal/tracking-studies/suggestions")
    async def suggestions(request: Request):
        payload = await _payload(request, SuggestionsRequest, 16384)
        return action(lambda: tracking_studies.suggestions(payload.baseline_profile))

    @app.post("/api/temporal/tracking-studies/preview")
    async def preview(request: Request):
        payload = await _payload(request, StudyRequest, 131072)
        check_sources(payload)
        return action(
            lambda: tracking_studies.preview_study(
                store, payload.model_dump(), project_id=active_project.get()
            )
        )

    @app.post("/api/temporal/tracking-studies", status_code=201)
    async def create(request: Request):
        payload = await _payload(request, StudyCreate, 131072)
        check_sources(payload)
        values = payload.model_dump()
        expected = values.pop("expected_fingerprint")
        return action(
            lambda: tracking_studies.create_study(
                store, jobs, values, expected_fingerprint=expected, project_id=active_project.get()
            )
        )

    @app.get("/api/temporal/tracking-studies")
    def listing():
        return action(lambda: tracking_studies.list_studies(store, project_id=active_project.get()))

    @app.get("/api/temporal/tracking-studies/{job_id}")
    def detail(job_id: RecordId):
        require("jobs", job_id)
        return action(
            lambda: tracking_studies.get_study(store, job_id, project_id=active_project.get())
        )

    @app.get("/api/temporal/tracking-studies/{job_id}/report")
    def download(job_id: RecordId):
        require("jobs", job_id)
        record = action(
            lambda: tracking_studies.get_study(store, job_id, project_id=active_project.get())
        )
        if record["report"] is None:
            raise HTTPException(409, "This attempt has no complete tracking study report")
        return Response(
            json.dumps(record["report"], ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            media_type="application/json",
            headers={
                "Content-Disposition": f'attachment; filename="iris-tracking-study-{job_id}.json"'
            },
        )
