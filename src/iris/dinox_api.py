"""Project-scoped hosted proposal batches and write-only local credentials."""

import json

from fastapi import HTTPException, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from iris import dinox_batches as batches
from iris import dinox_provider as provider


class Preview(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    frame_ids: list[str] = Field(min_length=1, max_length=25)
    threshold: float = Field(default=0.25, ge=0, le=1)
    class_prompts: dict[str, str] | None = None


class Create(Preview):
    name: str = Field(min_length=1, max_length=160)
    expected_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    approve_external: bool = False
    max_cost_cny: float = Field(default=0, ge=0, le=1000)


def install_dinox_routes(app, store, jobs, require):
    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, error: RequestValidationError):
        if "/dinox" not in request.url.path:
            return await request_validation_exception_handler(request, error)
        return JSONResponse(
            {
                "detail": [
                    {key: row[key] for key in ("type", "loc", "msg")} for row in error.errors()
                ]
            },
            status_code=422,
        )

    def action(function):
        try:
            return function()
        except KeyError as exc:
            raise HTTPException(404, "DINO-X batch or source image not found") from exc
        except (ValueError, RuntimeError, OSError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/dinox/provider")
    def status():
        return JSONResponse(provider.provider_status(), headers={"Cache-Control": "no-store"})

    @app.put("/api/dinox/key")
    async def save_key(request: Request):
        # No validation exception may echo the submitted secret into a response.
        try:
            raw = bytearray()
            async for chunk in request.stream():
                if len(raw) + len(chunk) > 2048:
                    raise ValueError("Oversized credential")
                raw.extend(chunk)
            payload = json.loads(raw)
            if not isinstance(payload, dict) or set(payload) != {"key"}:
                raise ValueError("Invalid credential object")
            provider.update_key(payload["key"])
        except (ValueError, RuntimeError, OSError):
            raise HTTPException(422, "Unable to save this DINO-X key securely") from None
        return JSONResponse(provider.provider_status(), headers={"Cache-Control": "no-store"})

    @app.delete("/api/dinox/key")
    def remove_key():
        action(provider.clear_key)
        return JSONResponse(provider.provider_status(), headers={"Cache-Control": "no-store"})

    @app.post("/api/sessions/{session_id}/dinox-batches/preview")
    def preview(session_id: str, payload: Preview):
        require("sessions", session_id)
        return action(lambda: batches.preview_batch(store, session_id, **payload.model_dump()))

    @app.post("/api/sessions/{session_id}/dinox-batches", status_code=202)
    def create(session_id: str, payload: Create):
        require("sessions", session_id)
        return action(lambda: batches.create_batch(store, jobs, session_id, **payload.model_dump()))

    @app.get("/api/sessions/{session_id}/dinox-batches")
    def history(session_id: str):
        require("sessions", session_id)
        return action(lambda: batches.list_batches(store, session_id))

    @app.get("/api/dinox-batches/{identifier}")
    def detail(identifier: str):
        require("dinox_batches", identifier)
        return action(lambda: batches.batch_detail(store, identifier))
