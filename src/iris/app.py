"""HTTP interface for a single-user, loopback-only workspace."""

import os
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.responses import JSONResponse

from iris import __version__
from iris.annotations import (
    AnnotationConflict,
    add_detector_suggestions,
    get_annotation,
    save_annotation,
)
from iris.assistance import request_assistance
from iris.assistance_provider import provider_status
from iris.inference import comparison_detail, comparison_summary, create_comparison
from iris.jobs import JobManager
from iris.media import import_asset
from iris.models import catalog
from iris.store import Store, new_id, now

STATIC = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 2 * 1024**3


class SessionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=160)
    scene_group: str = Field(min_length=1, max_length=160)

    @field_validator("name", "scene_group")
    @classmethod
    def strip_nonempty(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("This field cannot be blank")
        return value


class SelectionInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    selected: bool


class ExtractionInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    interval_seconds: float = Field(default=2, ge=0.05, le=86400)
    start_seconds: float = Field(default=0, ge=0)
    end_seconds: float | None = Field(default=None, gt=0)
    max_frames: int = Field(default=100, ge=1, le=500)
    dedup_hamming: int | None = Field(default=None, ge=0, le=16)

    @model_validator(mode="after")
    def check_range(self):
        if self.end_seconds is not None and self.end_seconds <= self.start_seconds:
            raise ValueError("End time must be greater than start time")
        return self


class ComparisonInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    name: str = Field(min_length=1, max_length=160)
    frame_ids: list[str] = Field(min_length=1, max_length=100)
    model_ids: list[str] = Field(min_length=1, max_length=2)
    device: Literal["cpu", "cuda"] = "cpu"


class AnnotationInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    expected_revision: int = Field(ge=0)
    boxes: list[dict] = Field(max_length=500)
    decisions: dict[str, Literal["accepted", "corrected", "rejected"]]
    status: Literal["draft", "validated"] = "draft"
    reviewer: str = Field(default="", max_length=120)
    notes: str = Field(default="", max_length=4000)


class SuggestionsInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    expected_revision: int = Field(ge=0)
    prediction_id: str
    threshold: float = Field(default=0.5, ge=0, le=1)


class AssistanceInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    expected_revision: int = Field(ge=0)
    prediction_id: str | None = None
    threshold: float = Field(default=0.5, ge=0, le=1)
    instructions: str = Field(default="", max_length=2000)


def public(record: dict) -> dict:
    return {key: value for key, value in record.items() if key != "path"}


def create_app(data_dir: Path | None = None, *, run_jobs: bool = True) -> FastAPI:
    store = Store(data_dir or Path(os.getenv("IRIS_DATA_DIR", ".iris")))
    jobs = JobManager(store)
    import_lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app):
        if run_jobs:
            jobs.start()
        try:
            yield
        finally:
            if run_jobs:
                jobs.close()

    app = FastAPI(
        title="IRIS", version=__version__, lifespan=lifespan, docs_url=None, redoc_url=None
    )
    app.state.store = store
    app.state.jobs = jobs
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]"])

    @app.middleware("http")
    async def local_requests(request: Request, call_next):
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            origin = request.headers.get("origin")
            if origin:
                try:
                    parsed = urlsplit(origin)
                except ValueError:
                    return JSONResponse({"detail": "Invalid request origin"}, 403)
                if parsed.scheme != "http" or parsed.netloc != request.headers.get("host"):
                    return JSONResponse(
                        {"detail": "Only same-origin local requests are allowed"}, 403
                    )
            if request.headers.get("sec-fetch-site") == "cross-site":
                return JSONResponse({"detail": "Cross-site requests are not allowed"}, 403)
        length = request.headers.get("content-length")
        if length and (not length.isdecimal() or int(length) > MAX_UPLOAD_BYTES + 1024**2):
            return JSONResponse({"detail": "Upload exceeds the 2 GiB limit"}, 413)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' blob:; media-src 'self' blob:; "
            "style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; "
            "frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
        )
        return response

    def require(table: str, record_id: str):
        record = store.get(table, record_id)
        if record is None:
            raise HTTPException(404, "Record not found")
        return record

    @app.get("/api/system")
    def system():
        return {
            "version": __version__,
            "data_dir": str(store.root),
            "capabilities": {
                "media_import": True,
                "frame_extraction": True,
                "frame_selection": True,
                "inference": True,
                "annotation": True,
                "assisted_annotation": True,
                "training": False,
            },
        }

    @app.get("/api/sessions")
    def sessions():
        return store.list("sessions")

    @app.get("/api/models")
    def models():
        return catalog(store.root)

    @app.get("/api/sessions/{session_id}/comparisons")
    def comparisons(session_id: str):
        require("sessions", session_id)
        return [
            comparison_summary(store, row)
            for row in store.list("comparisons", session_id=session_id)
        ]

    @app.post("/api/sessions/{session_id}/comparisons", status_code=202)
    def compare(session_id: str, payload: ComparisonInput):
        require("sessions", session_id)
        try:
            return create_comparison(store, jobs, session_id, **payload.model_dump())
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/comparisons/{comparison_id}")
    def comparison(comparison_id: str):
        require("comparisons", comparison_id)
        return comparison_detail(store, comparison_id)

    @app.post("/api/sessions", status_code=201)
    def create_session(payload: SessionInput):
        return store.insert(
            "sessions", {"id": new_id(), **payload.model_dump(), "created_at": now()}
        )

    @app.get("/api/sessions/{session_id}")
    def session(session_id: str):
        return require("sessions", session_id)

    @app.get("/api/sessions/{session_id}/assets")
    def assets(session_id: str):
        require("sessions", session_id)
        return [public(asset) for asset in store.list("assets", session_id=session_id)]

    @app.post("/api/sessions/{session_id}/assets", status_code=201)
    def upload(session_id: str, file: UploadFile):
        require("sessions", session_id)
        uploads = store.root / "uploads"
        uploads.mkdir(exist_ok=True)
        staged = None
        try:
            with tempfile.NamedTemporaryFile(dir=uploads, delete=False) as target:
                staged = Path(target.name)
                size = 0
                while chunk := file.file.read(1024**2):
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        raise HTTPException(413, "Upload exceeds the 2 GiB limit")
                    target.write(chunk)
            with import_lock:
                return public(import_asset(store, session_id, staged, file.filename or "upload"))
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        finally:
            file.file.close()
            if staged:
                staged.unlink(missing_ok=True)

    @app.get("/api/sessions/{session_id}/frames")
    def frames(session_id: str):
        require("sessions", session_id)
        with store.connect() as conn:
            counts = dict(conn.execute("SELECT sha256, COUNT(*) - 1 FROM frames GROUP BY sha256"))
        return [
            {**public(frame), "duplicate_count": counts[frame["sha256"]]}
            for frame in store.list("frames", session_id=session_id)
        ]

    @app.patch("/api/frames/{frame_id}")
    def select_frame(frame_id: str, payload: SelectionInput):
        require("frames", frame_id)
        return public(store.update("frames", frame_id, payload.model_dump()))

    @app.get("/api/frames/{frame_id}/image")
    def frame_image(frame_id: str):
        record = require("frames", frame_id)
        return FileResponse(store.artifact_path(record["path"]), media_type="image/png")

    @app.get("/api/annotation-provider")
    def annotation_provider():
        return provider_status()

    @app.get("/api/frames/{frame_id}/annotation")
    def annotation(frame_id: str):
        require("frames", frame_id)
        return get_annotation(store, frame_id)

    @app.get("/api/frames/{frame_id}/annotation/revisions/{revision}")
    def annotation_revision(frame_id: str, revision: int):
        require("frames", frame_id)
        revisions = store.list("annotation_revisions", frame_id=frame_id, revision=revision)
        if not revisions:
            raise HTTPException(404, "Annotation revision not found")
        return revisions[0]

    def annotation_action(function, frame_id, payload):
        require("frames", frame_id)
        try:
            return function(**payload.model_dump())
        except AnnotationConflict as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        except OSError as exc:
            raise HTTPException(409, "The source image is missing or unreadable") from exc

    @app.put("/api/frames/{frame_id}/annotation")
    def write_annotation(frame_id: str, payload: AnnotationInput):
        return annotation_action(
            lambda **fields: save_annotation(store, frame_id, **fields), frame_id, payload
        )

    @app.post("/api/frames/{frame_id}/suggestions")
    def detector_suggestions(frame_id: str, payload: SuggestionsInput):
        return annotation_action(
            lambda **fields: add_detector_suggestions(store, frame_id, **fields), frame_id, payload
        )

    @app.post("/api/frames/{frame_id}/assist", status_code=202)
    def assist(frame_id: str, payload: AssistanceInput):
        return annotation_action(
            lambda **fields: request_assistance(store, jobs, frame_id, **fields), frame_id, payload
        )

    @app.get("/api/frames/{frame_id}/assistance")
    def assistance_history(frame_id: str):
        require("frames", frame_id)
        return [
            {**record, "job": store.get("jobs", record["job_id"])}
            for record in store.list("assistance_records", frame_id=frame_id)
        ]

    @app.get("/api/assets/{asset_id}/media")
    def asset_media(asset_id: str):
        record = require("assets", asset_id)
        image_types = {
            "JPEG": "image/jpeg",
            "PNG": "image/png",
            "WEBP": "image/webp",
            "BMP": "image/bmp",
            "TIFF": "image/tiff",
        }
        media_type = image_types.get(record["metadata"].get("format"), "application/octet-stream")
        if record["kind"] == "video":
            media_type = record["metadata"].get("media_type", "application/octet-stream")
        return FileResponse(
            store.artifact_path(record["path"]),
            filename=record["filename"],
            media_type=media_type,
            content_disposition_type="inline",
        )

    @app.post("/api/assets/{asset_id}/extract", status_code=202)
    def extract(asset_id: str, payload: ExtractionInput):
        record = require("assets", asset_id)
        if record["kind"] != "video":
            raise HTTPException(422, "Only videos need frame extraction")
        duration = record["metadata"].get("duration_seconds")
        if duration and payload.start_seconds >= duration:
            raise HTTPException(422, "Start time is beyond the end of this video")
        try:
            return jobs.submit(asset_id, payload.model_dump())
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/jobs")
    def list_jobs():
        return store.list("jobs")

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel(job_id: str):
        require("jobs", job_id)
        return jobs.cancel(job_id)

    @app.get("/api/jobs/{job_id}/log")
    def job_log(job_id: str):
        require("jobs", job_id)
        log = store.root / "logs" / f"{job_id}.log"
        if not log.exists():
            raise HTTPException(404, "No worker log is available for this job")
        return FileResponse(log, media_type="text/plain")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")

    return app
