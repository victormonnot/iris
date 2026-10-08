"""Project-scoped, explicit packaging of frozen experimental pipeline contracts."""

import json
from typing import Literal

from fastapi import HTTPException, Request, Response
from fastapi.responses import FileResponse
from pydantic import Field

from iris import pipeline_bundles
from iris.temporal_api import Identifier, RecordId, StrictInput
from iris.tracking_cost_api import _payload
from iris.tracking_selection_api import SelectionSource


class BundleRequest(StrictInput):
    name: str = Field(min_length=1, max_length=160)
    source: SelectionSource
    selection_id: Identifier | None
    target_device: Literal["cpu", "cuda"]


class BundleCreate(BundleRequest):
    expected_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


def install_pipeline_bundle_routes(app, store, jobs, require, active_project):
    def action(function):
        try:
            return function()
        except KeyError as exc:
            raise HTTPException(404, "Pipeline package or frozen source not found") from exc
        except (ValueError, RuntimeError, OSError) as exc:
            raise HTTPException(409, str(exc)) from exc

    def check(payload):
        require("jobs", payload.source.job_id)
        require("temporal_sequences", payload.source.sequence_id)
        if payload.selection_id:
            require("jobs", payload.selection_id)

    def record(job_id):
        require("jobs", job_id)
        result = action(
            lambda: pipeline_bundles.get_bundle(store, job_id, project_id=active_project.get())
        )
        if result["bundle"] is None:
            raise HTTPException(409, "This attempt has no complete pipeline package")
        return result

    @app.get("/api/temporal/pipeline-bundle-status")
    def status():
        return pipeline_bundles.status()

    @app.get("/api/temporal/pipeline-bundle-sources")
    def sources():
        return action(lambda: pipeline_bundles.catalogue(store, project_id=active_project.get()))

    @app.post("/api/temporal/pipeline-bundles/preview")
    async def preview(request: Request):
        payload = await _payload(request, BundleRequest, 16384)
        check(payload)
        return action(
            lambda: pipeline_bundles.preview_bundle(
                store, payload.model_dump(), project_id=active_project.get()
            )
        )

    @app.post("/api/temporal/pipeline-bundles", status_code=201)
    async def create(request: Request):
        payload = await _payload(request, BundleCreate, 16384)
        check(payload)
        values = payload.model_dump()
        fingerprint = values.pop("expected_fingerprint")
        return action(
            lambda: pipeline_bundles.create_bundle(
                store,
                jobs,
                values,
                expected_fingerprint=fingerprint,
                project_id=active_project.get(),
            )
        )

    @app.get("/api/temporal/pipeline-bundles")
    def listing():
        return action(lambda: pipeline_bundles.list_bundles(store, project_id=active_project.get()))

    @app.get("/api/temporal/pipeline-bundles/{job_id}")
    def detail(job_id: RecordId):
        require("jobs", job_id)
        return action(
            lambda: pipeline_bundles.get_bundle(store, job_id, project_id=active_project.get())
        )

    @app.get("/api/temporal/pipeline-bundles/{job_id}/download")
    def download(job_id: RecordId):
        result = record(job_id)["bundle"]
        return FileResponse(
            store.artifact_path(result["path"]),
            media_type="application/zip",
            filename=f"iris-pipeline-{job_id}.zip",
        )

    @app.get("/api/temporal/pipeline-bundles/{job_id}/manifest")
    def manifest(job_id: RecordId):
        result = record(job_id)["bundle"]
        return Response(
            json.dumps(result["manifest"], indent=2, ensure_ascii=False) + "\n",
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="iris-pipeline-{job_id}.json"'},
        )
