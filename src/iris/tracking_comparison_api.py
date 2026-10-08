"""Project-scoped visual tracking jobs and verified frozen source pixels."""

import hashlib
from io import BytesIO
from typing import Annotated, Literal

from fastapi import HTTPException, Response
from PIL import Image
from pydantic import Field

from iris import tracking_comparisons
from iris.media import _pixel_hash
from iris.temporal import _row, _sequence_record
from iris.temporal_api import RecordId, StrictInput


class TrackingComparisonCreate(StrictInput):
    name: str = Field(min_length=1, max_length=160)
    class_ids: list[Annotated[int, Field(ge=1, le=2**31 - 1)]] = Field(min_length=1, max_length=100)
    gmc_method: Literal["none", "sparseOptFlow"]


def verified_frame_bytes(store, sequence_id, frame_id):
    # Verify one immutable byte snapshot and return those exact bytes. A FileResponse
    # would reopen the path after validation, allowing mismatched replacement pixels.
    with store.connect() as conn:
        conn.execute("BEGIN")
        sequence = _sequence_record(conn, _row(conn, "temporal_sequences", sequence_id))
        frozen = next(
            (frame for frame in sequence["manifest"]["frames"] if frame["frame_id"] == frame_id),
            None,
        )
        if frozen is None:
            raise KeyError(frame_id)
        frame = _row(conn, "frames", frame_id)
        content = store.artifact_path(frame["path"]).read_bytes()
    if hashlib.sha256(content).hexdigest() != frozen["file_sha256"]:
        raise ValueError("Source frame bytes no longer match the frozen temporal sequence")
    with Image.open(BytesIO(content)) as source:
        if source.format != "PNG":
            raise ValueError("Frozen temporal source frame must be a PNG")
        with source.convert("RGB") as image:
            image.load()
            if image.size != (frozen["width"], frozen["height"]) or (
                _pixel_hash(image) != frozen["sha256"]
            ):
                raise ValueError("Source pixels no longer match the frozen temporal sequence")
    return content


def install_tracking_comparison_routes(app, store, jobs, require, active_project):
    def action(function):
        try:
            return function()
        except KeyError as exc:
            raise HTTPException(404, "Temporal source, cache or comparison not found") from exc
        except OSError as exc:
            raise HTTPException(
                409, "Frozen temporal source image is missing or unreadable"
            ) from exc
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/temporal/tracking-status")
    def status():
        from iris.tracking import tracking_status

        return {**tracking_status(), "max_frames": tracking_comparisons.MAX_COMPARISON_FRAMES}

    @app.get("/api/temporal/detection-caches/{cache_id}/tracking-comparisons")
    def comparisons(cache_id: RecordId):
        require("temporal_detection_caches", cache_id)
        return action(lambda: tracking_comparisons.list_tracking_comparisons(store, cache_id))

    @app.post("/api/temporal/detection-caches/{cache_id}/tracking-comparisons", status_code=201)
    def create(cache_id: RecordId, payload: TrackingComparisonCreate):
        require("temporal_detection_caches", cache_id)
        return action(
            lambda: tracking_comparisons.create_tracking_comparison(
                store, jobs, cache_id, project_id=active_project.get(), **payload.model_dump()
            )
        )

    @app.get("/api/temporal/tracking-comparisons/{job_id}")
    def detail(job_id: RecordId):
        require("jobs", job_id)
        return action(lambda: tracking_comparisons.get_tracking_comparison(store, job_id))

    @app.get("/api/temporal/sequences/{sequence_id}/frames/{frame_id}/image")
    def frame_image(sequence_id: RecordId, frame_id: RecordId):
        require("temporal_sequences", sequence_id)
        require("frames", frame_id)
        return action(
            lambda: Response(
                verified_frame_bytes(store, sequence_id, frame_id),
                media_type="image/png",
                headers={"Cache-Control": "no-store"},
            )
        )
