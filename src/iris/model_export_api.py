"""Project-scoped native model export and bounded measurement import endpoints."""

from fastapi import HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from iris import model_exports as exports


class ExportPreview(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    trained_model_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,128}$")
    evaluation_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,128}$")
    frame_ids: list[str] = Field(min_length=1, max_length=8)
    name: str = Field(min_length=1, max_length=160)


class ExportCreate(ExportPreview):
    request_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,128}$")
    expected_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


def install_model_export_routes(app, store, require, active_project):
    def action(function):
        try:
            return function()
        except KeyError as exc:
            raise HTTPException(404, "Export source or record not found") from exc
        except (ValueError, RuntimeError, OSError, TypeError) as exc:
            raise HTTPException(409, str(exc)) from exc

    def sources(payload):
        require("trained_models", payload.trained_model_id)
        require("evaluations", payload.evaluation_id)

    @app.get("/api/model-exports/candidates")
    def candidates():
        return action(lambda: exports.candidates(store, active_project.get()))

    @app.get("/api/model-exports")
    def listing():
        return exports.list_exports(store, active_project.get())

    @app.post("/api/model-exports/preview")
    def preview(payload: ExportPreview):
        sources(payload)
        return action(lambda: exports.preview_export(store, **payload.model_dump()))

    @app.post("/api/model-exports", status_code=202)
    def create(payload: ExportCreate):
        sources(payload)
        return action(lambda: exports.create_export(store, **payload.model_dump()))

    @app.get("/api/model-exports/{export_id}")
    def detail(export_id: str):
        require("model_exports", export_id)
        return action(lambda: exports.export_detail(store, export_id))

    @app.get("/api/model-exports/{export_id}/download")
    def download(export_id: str):
        require("model_exports", export_id)
        return FileResponse(
            action(lambda: exports.download_path(store, export_id)),
            media_type="application/zip",
            filename=f"iris-model-{export_id}.zip",
            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
        )

    async def measurement(request):
        if request.headers.get("content-type", "").split(";")[0] != "application/json":
            raise HTTPException(415, "Upload the measurement as application/json")
        length = request.headers.get("content-length")
        if length and (not length.isdecimal() or int(length) > exports.MAX_MEASUREMENT_BYTES):
            raise HTTPException(413, "Measurement JSON exceeds the 8 MiB limit")
        raw = bytearray()
        async for chunk in request.stream():
            if len(raw) + len(chunk) > exports.MAX_MEASUREMENT_BYTES:
                raise HTTPException(413, "Measurement JSON exceeds the 8 MiB limit")
            raw.extend(chunk)
        return action(lambda: exports._json(bytes(raw)))

    @app.post("/api/model-exports/{export_id}/measurements/preview")
    async def preview_measurement(export_id: str, request: Request):
        require("model_exports", export_id)
        payload = await measurement(request)
        return action(lambda: exports.preview_measurement(store, export_id, payload))

    @app.post("/api/model-exports/{export_id}/measurements", status_code=201)
    async def save_measurement(
        export_id: str,
        request: Request,
        expected_fingerprint: str = Query(pattern=r"^[0-9a-f]{64}$"),
    ):
        require("model_exports", export_id)
        payload = await measurement(request)
        return action(
            lambda: exports.save_measurement(store, export_id, payload, expected_fingerprint)
        )

    @app.get("/api/model-export-measurements/{measurement_id}")
    def get_measurement(measurement_id: str):
        return require("model_export_measurements", measurement_id)
