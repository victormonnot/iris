"""HTTP interface for a single-user, loopback-only workspace."""

import os
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Query, Request, UploadFile
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
from iris.assistance_catalog import catalog as annotation_catalog
from iris.assistance_previews import preview_assistance, read_images
from iris.assistance_provider import provider_status
from iris.coco_import import commit_import, import_detail, preview_image_path, preview_import
from iris.datasets import create_dataset, dataset_candidates, dataset_detail, load_manifest
from iris.evaluation import (
    create_evaluation,
    evaluation_detail,
    evaluation_summary,
    promote_reference,
    reference_history,
)
from iris.inference import (
    _load_verified_frame,
    comparison_detail,
    comparison_summary,
    create_comparison,
)
from iris.jobs import JobManager
from iris.media import import_asset
from iris.models import catalog
from iris.review_queue import review_queue
from iris.store import Store, new_id, now
from iris.training import create_training, training_detail

STATIC = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 2 * 1024**3
MAX_DATASET_UPLOAD_BYTES = 64 * 1024**2


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


class DatasetInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    name: str = Field(min_length=1, max_length=160)
    frame_ids: list[str] = Field(min_length=2, max_length=1000)
    splits: dict[str, Literal["train", "val", "test"]]
    parent_id: str | None = None


class DatasetImportInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    name: str = Field(min_length=1, max_length=160)
    scene_group: str = Field(min_length=1, max_length=160)
    source_url: str = Field(min_length=1, max_length=2000)
    license_name: str = Field(min_length=1, max_length=500)
    attribution: str = Field(min_length=1, max_length=2000)
    source_split: Literal["train", "val", "test"] | None = None
    category_mapping: dict[str, Literal["person", "car", "exclude"]]


class TrainingInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    name: str = Field(min_length=1, max_length=160)
    dataset_id: str
    parent_model_id: str
    steps: int = Field(default=20, ge=1, le=200)
    learning_rate: float = Field(default=0.001, gt=0, le=0.1)
    seed: int = Field(default=0, ge=0, le=2147483647)


class EvaluationInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    name: str = Field(min_length=1, max_length=160)
    dataset_id: str
    split: Literal["val", "test"] = "val"
    model_ids: list[str] = Field(min_length=1, max_length=2)
    confidence_threshold: float = Field(default=0.5, ge=0, le=1)
    iou_threshold: float = Field(default=0.5, gt=0, le=1)
    device: Literal["cpu", "cuda"] = "cpu"
    validation_evaluation_id: str | None = None


class ReferenceInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    evaluation_id: str
    model_id: str
    reviewer: str = Field(min_length=1, max_length=120)
    notes: str = Field(min_length=1, max_length=2000)
    expected_previous_id: str | None


class SuggestionsInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    expected_revision: int = Field(ge=0)
    prediction_id: str
    threshold: float = Field(default=0.5, ge=0, le=1)


class AssistancePreviewInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    expected_revision: int = Field(ge=0)
    prediction_id: str | None = None
    threshold: float = Field(default=0.5, ge=0, le=1)
    instructions: str = Field(default="", max_length=2000)
    provider: Literal["ollama", "alibaba"] = "ollama"
    model: str | None = Field(default=None, max_length=160)


class AssistanceInput(AssistancePreviewInput):
    preview_id: str | None = Field(default=None, max_length=64)
    allow_external: bool = False
    max_cost_usd: float | None = Field(default=None, ge=0, le=100)


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
        dataset_upload = request.method == "POST" and request.url.path == "/api/dataset-imports"
        limit = MAX_DATASET_UPLOAD_BYTES if dataset_upload else MAX_UPLOAD_BYTES
        if length and (not length.isdecimal() or int(length) > limit + 1024**2):
            description = (
                "Dataset ZIP exceeds the 64 MiB limit"
                if dataset_upload
                else ("Upload exceeds the 2 GiB limit")
            )
            return JSONResponse({"detail": description}, 413)
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
                "dataset_versions": True,
                "coco_import": True,
                "training": True,
                "evaluation": True,
                "review_queue": True,
            },
        }

    @app.get("/api/sessions")
    def sessions():
        return store.list("sessions")

    @app.get("/api/models")
    def models():
        return catalog(store.root)

    @app.get("/api/dataset-candidates")
    def candidates():
        try:
            return dataset_candidates(store)
        except (OSError, ValueError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/dataset-imports", status_code=201)
    def preview_coco_import(file: UploadFile):
        uploads = store.root / "uploads"
        uploads.mkdir(exist_ok=True)
        staged = None
        try:
            with tempfile.NamedTemporaryFile(dir=uploads, delete=False) as target:
                staged = Path(target.name)
                size = 0
                while chunk := file.file.read(1024**2):
                    size += len(chunk)
                    if size > MAX_DATASET_UPLOAD_BYTES:
                        raise HTTPException(413, "Dataset ZIP exceeds the 64 MiB limit")
                    target.write(chunk)
            with import_lock:
                return preview_import(store, staged, file.filename or "dataset.zip")
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except (OSError, RuntimeError) as exc:
            raise HTTPException(409, str(exc)) from exc
        finally:
            file.file.close()
            if staged:
                staged.unlink(missing_ok=True)

    @app.get("/api/dataset-imports")
    def dataset_imports():
        return [
            {
                key: value
                for key, value in import_detail(store, row["id"]).items()
                if key != "images"
            }
            for row in store.list("dataset_imports")
        ]

    @app.get("/api/dataset-imports/{import_id}")
    def dataset_import(import_id: str):
        require("dataset_imports", import_id)
        return import_detail(store, import_id)

    @app.get("/api/dataset-imports/{import_id}/images/{image_id}")
    def dataset_import_image(import_id: str, image_id: str):
        require("dataset_imports", import_id)
        try:
            return FileResponse(
                preview_image_path(store, import_id, image_id), media_type="image/png"
            )
        except KeyError as exc:
            raise HTTPException(404, "Image is not part of this import") from exc
        except (ValueError, OSError, RuntimeError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/dataset-imports/{import_id}/commit", status_code=201)
    def confirm_dataset_import(import_id: str, payload: DatasetImportInput):
        require("dataset_imports", import_id)
        try:
            with import_lock:
                return commit_import(store, import_id, **payload.model_dump())
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except (OSError, RuntimeError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/datasets")
    def datasets():
        return [public(row) for row in store.list("dataset_versions")]

    @app.post("/api/datasets", status_code=201)
    def freeze_dataset(payload: DatasetInput):
        try:
            return public(create_dataset(store, **payload.model_dump()))
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except (OSError, RuntimeError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/datasets/{dataset_id}")
    def dataset(dataset_id: str):
        require("dataset_versions", dataset_id)
        try:
            return public(dataset_detail(store, dataset_id))
        except (OSError, ValueError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/datasets/{dataset_id}/manifest")
    def dataset_manifest(dataset_id: str):
        row = require("dataset_versions", dataset_id)
        try:
            load_manifest(store, dataset_id)
            return FileResponse(
                store.artifact_path(row["path"]),
                media_type="application/json",
                filename=f"iris-dataset-{dataset_id}.json",
            )
        except (OSError, ValueError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/datasets/{dataset_id}/frames/{frame_id}/image")
    def dataset_frame_image(dataset_id: str, frame_id: str):
        require("dataset_versions", dataset_id)
        try:
            manifest = load_manifest(store, dataset_id)
            frame = next((row for row in manifest["frames"] if row["frame_id"] == frame_id), None)
            if frame is None:
                raise HTTPException(404, "Frame is not part of this dataset version")
            with _load_verified_frame(
                store,
                {**frame, "id": frame_id, "path": frame["image_path"]},
                frame["sha256"],
            ):
                pass
            return FileResponse(store.artifact_path(frame["image_path"]), media_type="image/png")
        except (OSError, ValueError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/trainings")
    def trainings():
        return [training_detail(store, row["id"]) for row in store.list("training_runs")]

    @app.post("/api/trainings", status_code=202)
    def train(payload: TrainingInput):
        require("dataset_versions", payload.dataset_id)
        try:
            return create_training(store, jobs, **payload.model_dump())
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except (OSError, RuntimeError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/trainings/{training_id}")
    def training(training_id: str):
        require("training_runs", training_id)
        return training_detail(store, training_id)

    @app.get("/api/evaluations")
    def evaluations():
        return [evaluation_summary(store, row) for row in store.list("evaluations")]

    @app.post("/api/evaluations", status_code=202)
    def evaluate(payload: EvaluationInput):
        require("dataset_versions", payload.dataset_id)
        try:
            return create_evaluation(store, jobs, **payload.model_dump())
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except (OSError, RuntimeError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/evaluations/{evaluation_id}")
    def evaluation(evaluation_id: str):
        require("evaluations", evaluation_id)
        try:
            return evaluation_detail(store, evaluation_id)
        except (OSError, ValueError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/model-references")
    def references():
        return reference_history(store)

    @app.post("/api/model-references", status_code=201)
    def select_reference(payload: ReferenceInput):
        require("evaluations", payload.evaluation_id)
        try:
            return promote_reference(store, **payload.model_dump())
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except (OSError, RuntimeError) as exc:
            raise HTTPException(409, str(exc)) from exc

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

    @app.get("/api/sessions/{session_id}/review-queue")
    def session_review_queue(
        session_id: str,
        comparison_id: str | None = None,
        confidence_threshold: float = Query(default=0.5, ge=0, le=1),
        iou_threshold: float = Query(default=0.5, gt=0, le=1),
    ):
        require("sessions", session_id)
        try:
            return review_queue(
                store,
                session_id,
                comparison_id=comparison_id,
                confidence_threshold=confidence_threshold,
                iou_threshold=iou_threshold,
            )
        except KeyError as exc:
            raise HTTPException(404, "Comparison not found") from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except (OSError, RuntimeError) as exc:
            raise HTTPException(409, str(exc)) from exc

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

    @app.get("/api/annotation-providers")
    def annotation_providers():
        return annotation_catalog()

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

    @app.post("/api/frames/{frame_id}/assist/preview", status_code=201)
    def assist_preview(frame_id: str, payload: AssistancePreviewInput):
        return annotation_action(
            lambda **fields: preview_assistance(store, frame_id, **fields), frame_id, payload
        )

    @app.get("/api/assist-previews/{preview_id}/images/{index}")
    def preview_image(preview_id: str, index: int):
        from fastapi.responses import Response

        preview = require("assistance_previews", preview_id)
        if not 0 <= index < len(preview["images"]):
            raise HTTPException(404, "Preview image not found")
        try:
            content = read_images(store, preview)[index]
        except (OSError, ValueError) as exc:
            raise HTTPException(409, "Preview image changed or is unreadable") from exc
        return Response(content, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

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
