"""Project-scoped APIs for durable local detector caches and read-only filtering."""

from typing import Annotated, Literal

from fastapi import HTTPException, Response
from pydantic import Field

from iris import temporal_detections
from iris.temporal_api import Identifier, RecordId, StrictInput


class DetectionSettings(StrictInput):
    model_id: Identifier
    device: Literal["cpu", "cuda"] = "cpu"
    inference_mode: Literal["full", "tiled"] = "full"
    tile_size: int = Field(default=640, ge=128, le=2048)
    overlap: float = Field(default=0.2, ge=0, le=0.5)
    min_score: float = Field(default=0.001, ge=0.001, le=1)


class DetectionCacheCreate(DetectionSettings):
    name: str = Field(min_length=1, max_length=160)
    force_new: bool = False


class DetectionCacheRead(StrictInput):
    min_score: float | None = Field(default=None, ge=0.001, le=1)
    class_ids: list[Annotated[int, Field(ge=1, le=2**31 - 1)]] | None = Field(
        default=None, min_length=1, max_length=100
    )


def install_temporal_detection_routes(app, store, jobs, require, active_project):
    def action(function):
        try:
            return function()
        except KeyError as exc:
            raise HTTPException(404, "Temporal source, detector or cache not found") from exc
        except OSError as exc:
            raise HTTPException(
                409, "Local detector weights or source media are unreadable"
            ) from exc
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(409, str(exc)) from exc

    def require_model(model_id):
        if store.get("trained_models", model_id) is not None:
            require("trained_models", model_id)

    @app.get("/api/temporal/sequences/{sequence_id}/detection-caches")
    def caches(sequence_id: RecordId):
        require("temporal_sequences", sequence_id)
        return action(lambda: temporal_detections.list_detection_caches(store, sequence_id))

    @app.post("/api/temporal/sequences/{sequence_id}/detection-caches/preview")
    def preview(sequence_id: RecordId, payload: DetectionSettings):
        require("temporal_sequences", sequence_id)
        require_model(payload.model_id)
        return action(
            lambda: temporal_detections.preview_detection_cache(
                store, sequence_id, project_id=active_project.get(), **payload.model_dump()
            )
        )

    @app.post("/api/temporal/sequences/{sequence_id}/detection-caches", status_code=201)
    def create(sequence_id: RecordId, payload: DetectionCacheCreate, response: Response):
        require("temporal_sequences", sequence_id)
        require_model(payload.model_id)
        cache = action(
            lambda: temporal_detections.create_detection_cache(
                store, jobs, sequence_id, project_id=active_project.get(), **payload.model_dump()
            )
        )
        if cache["reused"]:
            response.status_code = 200
        return cache

    @app.get("/api/temporal/detection-caches/{cache_id}")
    def detail(cache_id: RecordId):
        require("temporal_detection_caches", cache_id)
        return action(lambda: temporal_detections.get_detection_cache(store, cache_id))

    @app.post("/api/temporal/detection-caches/{cache_id}/read")
    def filtered(cache_id: RecordId, payload: DetectionCacheRead):
        require("temporal_detection_caches", cache_id)
        return action(
            lambda: temporal_detections.read_detection_cache(
                store, cache_id, **payload.model_dump()
            )
        )

    @app.get("/api/temporal/detection-caches/{cache_id}/frames")
    def frames(cache_id: RecordId):
        require("temporal_detection_caches", cache_id)
        return action(lambda: temporal_detections.read_detection_cache(store, cache_id))
