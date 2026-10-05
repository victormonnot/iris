"""HTTP interface for a single-user, loopback-only workspace."""

import os
import shutil
import tempfile
import threading
from contextlib import asynccontextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.responses import JSONResponse

from iris import __version__
from iris.annotations import (
    AnnotationConflict,
    add_detector_suggestions,
    adopt_taxonomy,
    get_annotation,
    save_annotation,
)
from iris.assistance import request_assistance
from iris.assistance_batches import (
    batch_detail,
    cancel_batch,
    create_batch,
    list_batches,
    preview_batch,
)
from iris.assistance_catalog import catalog as annotation_catalog
from iris.assistance_previews import preview_assistance, read_images
from iris.assistance_provider import provider_status
from iris.batch_recovery import preview_retry_batch, retry_batch
from iris.coco_import import commit_import, import_detail, preview_image_path, preview_import
from iris.comparison_replay import VIDEO_TYPES, local_media_file
from iris.dataset_export import ExportLimitError, build_coco_export
from iris.dataset_planning import preview_dataset_plan
from iris.datasets import (
    create_dataset,
    dataset_brief,
    dataset_candidates,
    dataset_detail,
    load_manifest,
)
from iris.evaluation import (
    create_evaluation,
    evaluation_detail,
    evaluation_summary,
    preview_evaluation,
    promote_reference,
    reference_history,
)
from iris.evaluation_analysis import analyze_evaluation
from iris.experiment_export import ExperimentExportLimitError, render_experiment_html
from iris.experiments import (
    ExperimentConflict,
    create_experiment,
    experiment_detail,
    list_experiments,
    preview_experiment,
    read_experiment_image,
    update_experiment,
)
from iris.inference import (
    _load_verified_frame,
    comparison_detail,
    comparison_summary,
    create_comparison,
    preview_comparison,
)
from iris.job_activity import job_detail
from iris.jobs import JobManager
from iris.media import import_asset, preview_extraction
from iris.models import catalog
from iris.preannotation import (
    create_preannotation,
    list_preannotations,
    preannotation_detail,
    preview_preannotation,
)
from iris.preannotation_contracts import provider_capabilities
from iris.projects import create_project, project_records, record_project
from iris.review_queue import review_queue
from iris.selection import SelectionConflict, set_selection
from iris.selection_insights import selection_insights
from iris.store import DEFAULT_PROJECT_ID, Store, new_id, now
from iris.taxonomies import TaxonomyConflict, get_taxonomy, list_taxonomies, publish_taxonomy
from iris.training import create_training, preview_training, training_detail
from iris.video_reviews import (
    get_review,
    list_reviews,
    prepare_review,
    preview_passage_extraction,
    queue_review,
    read_review_images,
)
from iris.workspace_archive import MAX_ARCHIVE_BYTES, ArchiveError, ArchiveLimitError
from iris.workspace_operations import WorkspaceBusy, WorkspaceOperations

STATIC = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 2 * 1024**3
MAX_DATASET_UPLOAD_BYTES = 64 * 1024**2


class WorkspaceRestoreInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    inspection_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    folder_name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$")


class ProjectInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    name: str = Field(min_length=1, max_length=160)
    description: str = Field(default="", max_length=2000)


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


class TaxonomyInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    expected_taxonomy_id: str = Field(min_length=1, max_length=128)
    classes: list[dict] = Field(min_length=1, max_length=100)


class AnnotationTaxonomyInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    expected_revision: int = Field(ge=0)
    expected_taxonomy_id: str = Field(min_length=1, max_length=128)
    target_taxonomy_id: str = Field(min_length=1, max_length=128)


class SelectionInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    selected: bool


class BatchSelectionInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    frame_ids: list[str] = Field(min_length=1, max_length=1000)
    selected: bool
    expected_selection: dict[str, bool] = Field(min_length=1, max_length=1000)


class ExtractionInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    sampling_mode: Literal["uniform", "interval"] = "interval"
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
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    name: str = Field(min_length=1, max_length=160)
    frame_ids: list[str] = Field(min_length=1, max_length=100)
    model_ids: list[str] = Field(min_length=1, max_length=2)
    device: Literal["cpu", "cuda"] = "cpu"
    inference_mode: Literal["full", "tiled", "paired"] = "full"
    tile_size: int = Field(default=640, ge=128, le=2048)
    overlap: float = Field(default=0.2, ge=0, le=0.5)


class AnnotationInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    expected_revision: int = Field(ge=0)
    taxonomy_id: str | None = Field(default=None, min_length=1, max_length=128)
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
    taxonomy_id: str | None = Field(default=None, min_length=1, max_length=128)
    expected_revisions: dict[str, str] | None = Field(default=None, max_length=1000)


class DatasetPlanInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    taxonomy_id: str | None = Field(default=None, min_length=1, max_length=128)
    ratios: dict[str, float] = Field(
        default_factory=lambda: {"train": 0.8, "val": 0.2, "test": 0.0},
        min_length=3,
        max_length=3,
    )
    seed: int = Field(default=0, ge=0, le=2147483647)


class DatasetImportInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    name: str = Field(min_length=1, max_length=160)
    scene_group: str = Field(min_length=1, max_length=160)
    source_url: str = Field(min_length=1, max_length=2000)
    license_name: str = Field(min_length=1, max_length=500)
    attribution: str = Field(min_length=1, max_length=2000)
    source_split: Literal["train", "val", "test"] | None = None
    category_mapping: dict[str, str]


class TrainingInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    name: str = Field(min_length=1, max_length=160)
    dataset_id: str
    parent_model_id: str
    scope: Literal["prediction_head_only", "partial_backbone", "full_model"] = (
        "prediction_head_only"
    )
    steps: int = Field(default=20, ge=1, le=10000)
    checkpoint_interval: int | None = Field(default=None, ge=1, le=1000)
    learning_rate: float = Field(default=0.001, gt=0, le=0.1)
    seed: int = Field(default=0, ge=0, le=2147483647)
    device: str = Field(default="cpu", pattern=r"^(?:cpu|cuda(?::(?:0|[1-9][0-9]{0,2}))?)$")


class TrainingCreateInput(TrainingInput):
    request_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,128}$")
    expected_fingerprint: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class TrainingResumeInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    expected_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


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
    inference_mode: Literal["full", "tiled", "paired"] = "full"
    tile_size: int = Field(default=640, ge=128, le=2048)
    overlap: float = Field(default=0.2, ge=0, le=0.5)


class ReferenceInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    evaluation_id: str
    model_id: str
    variant: Literal["full", "tiled"] | None = None
    reviewer: str = Field(min_length=1, max_length=120)
    notes: str = Field(min_length=1, max_length=2000)
    expected_previous_id: str | None


class ExperimentInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    evaluation_id: str = Field(min_length=1)
    title: str = Field(min_length=1, max_length=160)
    objective: str = Field(default="", max_length=4000)
    conclusion: str = Field(default="", max_length=4000)
    example_frame_ids: list[str] = Field(default_factory=list, max_length=6)


class ExperimentUpdateInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    expected_revision: int = Field(ge=1)
    title: str = Field(min_length=1, max_length=160)
    objective: str = Field(default="", max_length=4000)
    conclusion: str = Field(default="", max_length=4000)


class SuggestionsInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    expected_revision: int = Field(ge=0)
    prediction_id: str
    threshold: float = Field(default=0.5, ge=0, le=1)


class PreannotationPreviewInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    frame_ids: list[str] = Field(min_length=1, max_length=25)
    model_id: str = Field(min_length=1, max_length=160)
    threshold: float = Field(default=0.5, ge=0, le=1)
    inference_mode: Literal["full", "tiled"] = "full"
    tile_size: int = Field(default=640, ge=1)
    overlap: float = Field(default=0.2, ge=0, lt=1)
    device: Literal["cpu", "cuda"] = "cpu"


class PreannotationInput(PreannotationPreviewInput):
    name: str = Field(min_length=1, max_length=160)
    expected_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")


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


class VideoReviewInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    provider: Literal["ollama", "alibaba"] = "ollama"
    model: str | None = Field(default=None, max_length=160)
    start_seconds: float = Field(default=0, ge=0)
    end_seconds: float | None = Field(default=None, gt=0)
    sample_count: int = Field(default=8, ge=2, le=12)
    instructions: str = Field(default="", max_length=2000)

    @model_validator(mode="after")
    def check_range(self):
        if self.end_seconds is not None and self.end_seconds <= self.start_seconds:
            raise ValueError("End time must be greater than start time")
        return self


class VideoReviewRunInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    allow_external: bool = False
    max_cost_usd: float | None = Field(default=None, ge=0, le=100)


class PassageExtractionInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    passage_ids: list[str] = Field(min_length=1, max_length=6)
    frames_per_passage: int = Field(default=8, ge=1, le=50)
    context_seconds: float = Field(default=2, ge=0, le=30)
    coverage_frames: int = Field(default=8, ge=0, le=32)


class AssistanceBatchPreviewInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    frame_ids: list[str] = Field(min_length=1, max_length=25)
    source: Literal["annotations", "comparison"] = "annotations"
    comparison_id: str | None = None
    detector_model_id: str | None = None
    detector_variant: Literal["full", "tiled"] | None = None
    model: str = Field(min_length=1, max_length=160)
    threshold: float = Field(default=0.5, ge=0, le=1)
    instructions: str = Field(default="", max_length=2000)


class AssistanceBatchInput(AssistanceBatchPreviewInput):
    name: str = Field(min_length=1, max_length=160)
    expected_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")

    @field_validator("name")
    @classmethod
    def strip_name(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("Batch name cannot be blank")
        return value


class BatchRetryInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    name: str = Field(min_length=1, max_length=160)
    expected_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")

    @field_validator("name")
    @classmethod
    def strip_name(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("Batch name cannot be blank")
        return value


class RecoveryInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")


def public(record: dict) -> dict:
    return {key: value for key, value in record.items() if key != "path"}


class _DatasetExportResponse(FileResponse):
    """Release temporary disk space and the export slot even if the client disconnects."""

    def __init__(self, path: Path, dataset_id: str, export_lock):
        super().__init__(
            path,
            media_type="application/zip",
            filename=f"iris-dataset-{dataset_id}-coco.zip",
            headers={"Cache-Control": "no-store"},
        )
        self.export_lock = export_lock

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            try:
                Path(self.path).unlink(missing_ok=True)
            finally:
                self.export_lock.release()


class _WorkspaceArchiveResponse(FileResponse):
    def __init__(self, manager, identifier, path, filename):
        super().__init__(
            path,
            media_type="application/zip",
            filename=filename,
            headers={"Cache-Control": "no-store"},
        )
        self.manager = manager
        self.identifier = identifier

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            self.manager.release_download(self.identifier)


def create_app(data_dir: Path | None = None, *, run_jobs: bool = True) -> FastAPI:
    store = Store(data_dir or Path(os.getenv("IRIS_DATA_DIR", ".iris")))
    jobs = JobManager(store)
    active_project = ContextVar("iris_request_project", default=DEFAULT_PROJECT_ID)
    workspace_operations = WorkspaceOperations(store)
    import_lock = threading.Lock()
    export_lock = threading.Lock()
    experiment_export_lock = threading.Lock()
    video_preview_lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app):
        if run_jobs:
            jobs.start()
        try:
            workspace_operations.recover()
            yield
        finally:
            try:
                workspace_operations.close()
            finally:
                if run_jobs:
                    jobs.close()

    app = FastAPI(
        title="IRIS", version=__version__, lifespan=lifespan, docs_url=None, redoc_url=None
    )
    app.state.store = store
    app.state.jobs = jobs
    app.state.workspace_operations = workspace_operations
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
        workspace_upload = (
            request.method == "POST" and request.url.path == "/api/workspace/restore-inspections"
        )
        if workspace_upload and (length is None or request.headers.get("transfer-encoding")):
            return JSONResponse(
                {
                    "detail": (
                        "Workspace archive uploads require Content-Length; "
                        "chunked uploads are unsupported"
                    )
                },
                411,
            )
        limit = (
            MAX_ARCHIVE_BYTES
            if workspace_upload
            else MAX_DATASET_UPLOAD_BYTES
            if dataset_upload
            else MAX_UPLOAD_BYTES
        )
        if length and (not length.isdecimal() or int(length) > limit + 1024**2):
            description = (
                "Workspace ZIP exceeds the 64 GiB limit"
                if workspace_upload
                else "Dataset ZIP exceeds the 64 MiB limit"
                if dataset_upload
                else ("Upload exceeds the 2 GiB limit")
            )
            return JSONResponse({"detail": description}, 413)
        if workspace_upload and length:
            size = int(length)
            temporary_root = Path(tempfile.gettempdir())
            same_volume = temporary_root.stat().st_dev == store.root.stat().st_dev
            temporary_free = shutil.disk_usage(temporary_root).free
            workspace_free = shutil.disk_usage(store.root).free
            reserve = 16 * 1024**2
            if (
                temporary_free < size + reserve
                or workspace_free < size * (2 if same_volume else 1) + reserve
            ):
                return JSONResponse(
                    {"detail": "Not enough local disk space to receive and inspect this archive"},
                    507,
                )
        project_id = request.query_params.get("project_id", DEFAULT_PROJECT_ID)
        global_request = request.url.path in {
            "/api/system",
            "/api/annotation-providers",
            "/api/assistance-status",
        } or request.url.path.startswith(("/api/projects", "/api/workspace/"))
        if request.url.path.startswith("/api/") and not global_request:
            if len(request.query_params.getlist("project_id")) > 1:
                return JSONResponse({"detail": "Choose one project per request"}, 422)
            if not store.get("projects", project_id):
                return JSONResponse({"detail": "Project not found"}, 404)
        mutation = request.method not in {
            "GET",
            "HEAD",
            "OPTIONS",
        } and not request.url.path.startswith("/api/workspace/")
        if mutation:
            try:
                workspace_operations.gate.enter()
            except WorkspaceBusy as exc:
                return JSONResponse({"detail": str(exc)}, 409)
        token = active_project.set(project_id)
        try:
            response = await call_next(request)
        finally:
            active_project.reset(token)
            if mutation:
                workspace_operations.gate.leave()
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers.setdefault(
            "Content-Security-Policy",
            (
                "default-src 'self'; img-src 'self' blob:; media-src 'self' blob:; "
                "style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; "
                "frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
            ),
        )
        return response

    def require(table: str, record_id: str):
        record = store.get(table, record_id)
        if record is None or record_project(store, table, record) != active_project.get():
            raise HTTPException(404, "Record not found in this project")
        return record

    def require_models(model_ids):
        for model_id in model_ids:
            trained = store.get("trained_models", model_id)
            if trained and record_project(store, "trained_models", trained) != active_project.get():
                raise HTTPException(404, "Trained model not found in this project")

    @app.get("/api/projects")
    def projects():
        return store.list("projects")

    @app.post("/api/projects", status_code=201)
    def add_project(payload: ProjectInput):
        try:
            return create_project(store, **payload.model_dump())
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/projects/{project_id}")
    def project(project_id: str):
        record = store.get("projects", project_id)
        if record is None:
            raise HTTPException(404, "Project not found")
        return record

    @app.get("/api/projects/{project_id}/taxonomies")
    def project_taxonomies(project_id: str):
        record = project(project_id)
        return {
            "current_taxonomy_id": record["taxonomy_id"],
            "versions": list_taxonomies(store, project_id),
        }

    @app.post("/api/projects/{project_id}/taxonomies", status_code=201)
    def add_taxonomy(project_id: str, payload: TaxonomyInput):
        project(project_id)
        try:
            return publish_taxonomy(store, project_id, **payload.model_dump())
        except TaxonomyConflict as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/system")
    def system():
        return {
            "version": __version__,
            "data_dir": str(store.root),
            "capabilities": {
                "projects": True,
                "custom_classes": True,
                "media_import": True,
                "frame_extraction": True,
                "video_sampling_preview": True,
                "video_passage_review": True,
                "frame_selection": True,
                "selection_insights": True,
                "dataset_split_planning": True,
                "durable_job_recovery": True,
                "inference": True,
                "comparison_replay": True,
                "annotation": True,
                "assisted_annotation": True,
                "local_batch_assistance": True,
                "dataset_versions": True,
                "custom_dataset_versions": True,
                "dataset_export": True,
                "evaluation_analysis": True,
                "experiment_reports": True,
                "workspace_backup": True,
                "coco_import": True,
                "training": True,
                "custom_class_training": True,
                "evaluation": True,
                "custom_class_evaluation": True,
                "review_queue": True,
            },
        }

    def workspace_action(action):
        try:
            return action()
        except KeyError as exc:
            raise HTTPException(404, "Workspace operation not found") from exc
        except WorkspaceBusy as exc:
            raise HTTPException(409, str(exc)) from exc
        except ArchiveLimitError as exc:
            raise HTTPException(413, str(exc)) from exc
        except ArchiveError as exc:
            raise HTTPException(422, str(exc)) from exc
        except OSError as exc:
            raise HTTPException(
                507, "Cannot access transfer files. Check local permissions and free disk space."
            ) from exc

    @app.get("/api/workspace/backup-preview")
    def workspace_backup_preview():
        return workspace_action(workspace_operations.preview)

    @app.get("/api/workspace/operations")
    def workspace_operation_list():
        return workspace_action(workspace_operations.list)

    @app.get("/api/workspace/operations/{identifier}")
    def workspace_operation(identifier: str):
        return workspace_action(lambda: workspace_operations.get(identifier))

    @app.post("/api/workspace/backups", status_code=202)
    def workspace_backup():
        return workspace_action(workspace_operations.backup)

    @app.post("/api/workspace/restore-inspections", status_code=202)
    def workspace_inspect(file: UploadFile):
        try:
            return workspace_action(
                lambda: workspace_operations.inspect_upload(
                    file.file, file.filename or "workspace.zip"
                )
            )
        finally:
            file.file.close()

    @app.post("/api/workspace/restores", status_code=202)
    def workspace_restore(payload: WorkspaceRestoreInput):
        return workspace_action(
            lambda: workspace_operations.restore(payload.inspection_id, payload.folder_name)
        )

    @app.post("/api/workspace/operations/{identifier}/cancel")
    def workspace_operation_cancel(identifier: str):
        return workspace_action(lambda: workspace_operations.cancel(identifier))

    @app.get("/api/workspace/operations/{identifier}/archive")
    def workspace_archive_download(identifier: str):
        path, filename = workspace_action(lambda: workspace_operations.archive(identifier))
        return _WorkspaceArchiveResponse(workspace_operations, identifier, path, filename)

    @app.delete("/api/workspace/operations/{identifier}", status_code=204)
    def workspace_operation_delete(identifier: str):
        workspace_action(lambda: workspace_operations.delete(identifier))
        return Response(status_code=204)

    @app.get("/api/sessions")
    def sessions():
        return project_records(store, "sessions", active_project.get())

    @app.get("/api/models")
    def models():
        owned = {
            row["id"] for row in project_records(store, "trained_models", active_project.get())
        }
        trained = {row["id"] for row in store.list("trained_models")}
        return [
            row for row in catalog(store.root) if row["id"] not in trained or row["id"] in owned
        ]

    @app.get("/api/dataset-candidates")
    def candidates(taxonomy_id: str | None = Query(default=None, min_length=1, max_length=128)):
        try:
            return dataset_candidates(
                store, project_id=active_project.get(), taxonomy_id=taxonomy_id
            )
        except (OSError, ValueError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/datasets/plan")
    def dataset_plan(payload: DatasetPlanInput):
        try:
            return preview_dataset_plan(
                store, project_id=active_project.get(), **payload.model_dump()
            )
        except KeyError as exc:
            raise HTTPException(404, "Class version not found") from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except (OSError, RuntimeError) as exc:
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
                return preview_import(
                    store, staged, file.filename or "dataset.zip", project_id=active_project.get()
                )
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
            for row in project_records(store, "dataset_imports", active_project.get())
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
        try:
            return [
                public(dataset_brief(store, row))
                for row in project_records(store, "dataset_versions", active_project.get())
            ]
        except (OSError, ValueError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/datasets", status_code=201)
    def freeze_dataset(payload: DatasetInput):
        try:
            return public(
                create_dataset(store, project_id=active_project.get(), **payload.model_dump())
            )
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

    @app.get("/api/datasets/{dataset_id}/export/coco")
    def export_dataset(dataset_id: str):
        require("dataset_versions", dataset_id)
        if not export_lock.acquire(blocking=False):
            raise HTTPException(409, "Another dataset export is in progress. Try again shortly.")
        archive = None
        try:
            try:
                archive = build_coco_export(store, dataset_id)
            except ExportLimitError as exc:
                raise HTTPException(413, str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(409, str(exc)) from exc
            except OSError as exc:
                raise HTTPException(
                    409,
                    "Could not prepare the dataset export. Check local files and free disk space.",
                ) from exc
            return _DatasetExportResponse(archive, dataset_id, export_lock)
        except BaseException:
            try:
                if archive is not None:
                    archive.unlink(missing_ok=True)
            finally:
                export_lock.release()
            raise

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

    @app.get("/api/training/devices")
    def training_devices():
        from iris.training_device import available_devices

        return available_devices()

    @app.get("/api/trainings")
    def trainings():
        return [
            training_detail(store, row["id"])
            for row in project_records(store, "training_runs", active_project.get())
        ]

    @app.post("/api/trainings", status_code=202)
    def train(payload: TrainingCreateInput):
        require("dataset_versions", payload.dataset_id)
        require_models([payload.parent_model_id])
        try:
            return create_training(store, jobs, **payload.model_dump())
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except (OSError, RuntimeError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/trainings/preview")
    def training_preview(payload: TrainingInput):
        require("dataset_versions", payload.dataset_id)
        require_models([payload.parent_model_id])
        try:
            return preview_training(store, **payload.model_dump())
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except (OSError, RuntimeError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/trainings/{training_id}")
    def training(training_id: str):
        require("training_runs", training_id)
        return training_detail(store, training_id)

    @app.post("/api/trainings/{training_id}/resume-preview")
    def training_resume_preview(training_id: str):
        from iris.training_recovery import preview_resume

        require("training_runs", training_id)
        try:
            return preview_resume(store, training_id)
        except (ValueError, RuntimeError, OSError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/trainings/{training_id}/resume", status_code=202)
    def training_resume(training_id: str, payload: TrainingResumeInput):
        from iris.training_recovery import resume_training

        require("training_runs", training_id)
        try:
            return resume_training(store, jobs, training_id, **payload.model_dump())
        except (ValueError, RuntimeError, OSError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/evaluations")
    def evaluations():
        return [
            evaluation_summary(store, row)
            for row in project_records(store, "evaluations", active_project.get())
        ]

    @app.post("/api/evaluations/preview")
    def evaluation_preview(payload: EvaluationInput):
        require("dataset_versions", payload.dataset_id)
        require_models(payload.model_ids)
        if payload.validation_evaluation_id:
            require("evaluations", payload.validation_evaluation_id)
        try:
            return preview_evaluation(store, **payload.model_dump())
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except (OSError, RuntimeError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/evaluations", status_code=202)
    def evaluate(payload: EvaluationInput):
        require("dataset_versions", payload.dataset_id)
        require_models(payload.model_ids)
        if payload.validation_evaluation_id:
            require("evaluations", payload.validation_evaluation_id)
        try:
            return create_evaluation(store, jobs, **payload.model_dump())
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except (OSError, RuntimeError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/evaluations/{evaluation_id}/analysis")
    def evaluation_errors(evaluation_id: str):
        require("evaluations", evaluation_id)
        try:
            return analyze_evaluation(store, evaluation_id)
        except (ValueError, OSError, RuntimeError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/evaluations/{evaluation_id}")
    def evaluation(evaluation_id: str):
        require("evaluations", evaluation_id)
        try:
            return evaluation_detail(store, evaluation_id)
        except (OSError, ValueError) as exc:
            raise HTTPException(409, str(exc)) from exc

    def experiment_action(action, *, invalid_status: int = 409):
        try:
            return action()
        except KeyError as exc:
            raise HTTPException(404, "Experiment or source record not found") from exc
        except ExperimentConflict as exc:
            raise HTTPException(409, str(exc)) from exc
        except ExperimentExportLimitError as exc:
            raise HTTPException(413, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(invalid_status, str(exc)) from exc
        except OSError as exc:
            raise HTTPException(
                409,
                "Could not access the experiment artifacts. Check local files and disk space.",
            ) from exc

    @app.get("/api/evaluations/{evaluation_id}/experiment-preview")
    def experiment_preview(evaluation_id: str):
        require("evaluations", evaluation_id)
        return experiment_action(lambda: preview_experiment(store, evaluation_id))

    @app.get("/api/experiments")
    def experiments():
        return experiment_action(lambda: list_experiments(store, project_id=active_project.get()))

    @app.post("/api/experiments", status_code=201)
    def experiment_create(payload: ExperimentInput):
        require("evaluations", payload.evaluation_id)
        return experiment_action(
            lambda: create_experiment(store, **payload.model_dump()), invalid_status=422
        )

    @app.get("/api/experiments/{experiment_id}")
    def experiment(experiment_id: str):
        require("experiment_reports", experiment_id)
        return experiment_action(lambda: experiment_detail(store, experiment_id))

    @app.patch("/api/experiments/{experiment_id}")
    def experiment_update(experiment_id: str, payload: ExperimentUpdateInput):
        require("experiment_reports", experiment_id)
        return experiment_action(
            lambda: update_experiment(store, experiment_id, **payload.model_dump()),
            invalid_status=422,
        )

    @app.get("/api/experiments/{experiment_id}/images/{frame_id}")
    def experiment_image(experiment_id: str, frame_id: str):
        require("experiment_reports", experiment_id)
        content = experiment_action(lambda: read_experiment_image(store, experiment_id, frame_id))
        return Response(content, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/experiments/{experiment_id}/export")
    def experiment_export(
        experiment_id: str,
        expected_revision: int = Query(ge=1),
        include_images: bool = Query(default=False),
    ):
        require("experiment_reports", experiment_id)
        if not experiment_export_lock.acquire(blocking=False):
            raise HTTPException(
                409, "Another experiment export is being prepared. Try again shortly."
            )
        try:
            content = experiment_action(
                lambda: render_experiment_html(
                    store,
                    experiment_id,
                    include_images=include_images,
                    expected_revision=expected_revision,
                )
            )
            return Response(
                content,
                media_type="text/html",
                headers={
                    "Content-Disposition": (
                        f'attachment; filename="iris-experiment-{experiment_id}.html"'
                    ),
                    "Cache-Control": "no-store",
                    "Content-Security-Policy": (
                        "default-src 'none'; img-src data:; style-src 'unsafe-inline'; "
                        "script-src 'none'; connect-src 'none'; base-uri 'none'; "
                        "form-action 'none'; object-src 'none'; frame-ancestors 'none'"
                    ),
                },
            )
        finally:
            experiment_export_lock.release()

    @app.get("/api/model-references")
    def references():
        return reference_history(store, project_id=active_project.get())

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

    @app.post("/api/sessions/{session_id}/comparisons/preview")
    def compare_preview(session_id: str, payload: ComparisonInput):
        require("sessions", session_id)
        require_models(payload.model_ids)
        try:
            return preview_comparison(store, session_id, **payload.model_dump())
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/sessions/{session_id}/comparisons", status_code=202)
    def compare(session_id: str, payload: ComparisonInput):
        require("sessions", session_id)
        require_models(payload.model_ids)
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
            "sessions",
            {
                "id": new_id(),
                **payload.model_dump(),
                "project_id": active_project.get(),
                "created_at": now(),
            },
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
                return public(
                    import_asset(
                        store,
                        session_id,
                        staged,
                        file.filename or "upload",
                        include_import_status=True,
                    )
                )
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

    @app.get("/api/sessions/{session_id}/selection-insights")
    def frame_insights(session_id: str):
        require("sessions", session_id)
        try:
            return selection_insights(store, session_id, project_id=active_project.get())
        except KeyError as exc:
            raise HTTPException(404, "Session not found") from exc
        except (OSError, ValueError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/sessions/{session_id}/selection")
    def select_frames(session_id: str, payload: BatchSelectionInput):
        require("sessions", session_id)
        try:
            return set_selection(
                store, session_id, project_id=active_project.get(), **payload.model_dump()
            )
        except KeyError as exc:
            raise HTTPException(404, "Session not found") from exc
        except SelectionConflict as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

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

    @app.get("/api/preannotation-providers")
    def preannotation_providers():
        return {
            "providers": [
                {
                    "id": provider,
                    "label": label,
                    "local": provider != "alibaba",
                    "capabilities": provider_capabilities(provider),
                    "models": models() if provider == "local_detector" else [],
                }
                for provider, label in (
                    ("local_detector", "Local detectors"),
                    ("ollama", "Local candidate review"),
                    ("alibaba", "API candidate review"),
                )
            ]
        }

    @app.post("/api/sessions/{session_id}/preannotations/preview")
    def preannotation_preview(session_id: str, payload: PreannotationPreviewInput):
        require("sessions", session_id)
        return batch_action(
            lambda: preview_preannotation(store, session_id, **payload.model_dump())
        )

    @app.post("/api/sessions/{session_id}/preannotations", status_code=202)
    def preannotation_create(session_id: str, payload: PreannotationInput):
        require("sessions", session_id)
        return batch_action(
            lambda: create_preannotation(store, jobs, session_id, **payload.model_dump())
        )

    @app.get("/api/sessions/{session_id}/preannotations")
    def preannotation_list(session_id: str):
        require("sessions", session_id)
        return batch_action(lambda: list_preannotations(store, session_id))

    @app.get("/api/preannotations/{comparison_id}")
    def preannotation_get(comparison_id: str):
        require("comparisons", comparison_id)
        return batch_action(lambda: preannotation_detail(store, comparison_id))

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
        return {
            **revisions[0],
            "taxonomy": get_taxonomy(store, revisions[0]["taxonomy_id"], active_project.get()),
        }

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

    @app.post("/api/frames/{frame_id}/annotation/taxonomy")
    def update_annotation_taxonomy(frame_id: str, payload: AnnotationTaxonomyInput):
        return annotation_action(
            lambda **fields: adopt_taxonomy(store, frame_id, **fields), frame_id, payload
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

    def batch_action(function):
        try:
            return function()
        except KeyError as exc:
            raise HTTPException(404, "Assistance batch or session not found") from exc
        except (AnnotationConflict, RuntimeError) as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except OSError as exc:
            raise HTTPException(409, "The source image is missing or unreadable") from exc

    @app.post("/api/sessions/{session_id}/assistance-batches/preview")
    def batch_preview(session_id: str, payload: AssistanceBatchPreviewInput):
        require("sessions", session_id)
        return batch_action(lambda: preview_batch(store, session_id, **payload.model_dump()))

    @app.post("/api/sessions/{session_id}/assistance-batches", status_code=202)
    def batch_create(session_id: str, payload: AssistanceBatchInput):
        require("sessions", session_id)
        return batch_action(lambda: create_batch(store, jobs, session_id, **payload.model_dump()))

    @app.get("/api/sessions/{session_id}/assistance-batches")
    def batch_list(session_id: str):
        require("sessions", session_id)
        return batch_action(lambda: list_batches(store, session_id))

    @app.get("/api/assistance-batches/{batch_id}")
    def batch_get(batch_id: str):
        require("assistance_batches", batch_id)
        return batch_action(lambda: batch_detail(store, batch_id))

    @app.post("/api/assistance-batches/{batch_id}/cancel")
    def batch_cancel(batch_id: str):
        require("assistance_batches", batch_id)
        return batch_action(lambda: cancel_batch(store, jobs, batch_id))

    @app.post("/api/assistance-batches/{batch_id}/retry-preview")
    def batch_retry_preview(batch_id: str):
        require("assistance_batches", batch_id)
        return batch_action(lambda: preview_retry_batch(store, batch_id))

    @app.post("/api/assistance-batches/{batch_id}/retry", status_code=202)
    def batch_retry(batch_id: str, payload: BatchRetryInput):
        require("assistance_batches", batch_id)
        return batch_action(lambda: retry_batch(store, jobs, batch_id, **payload.model_dump()))

    @app.get("/api/assets/{asset_id}/media")
    def asset_media(asset_id: str):
        record = require("assets", asset_id)
        status, path = local_media_file(store, record["path"], size_bytes=record["size_bytes"])
        if status == "unsafe":
            raise HTTPException(422, "The original media path is unsafe")
        if status == "size_mismatch":
            raise HTTPException(409, "The original media size changed since import")
        if status != "available":
            raise HTTPException(404, "The original media is missing or unreadable")
        image_types = {
            "JPEG": "image/jpeg",
            "PNG": "image/png",
            "WEBP": "image/webp",
            "BMP": "image/bmp",
            "TIFF": "image/tiff",
        }
        metadata = record["metadata"] if isinstance(record["metadata"], dict) else {}
        image_format = metadata.get("format")
        media_type = (
            image_types.get(image_format, "application/octet-stream")
            if isinstance(image_format, str)
            else "application/octet-stream"
        )
        if record["kind"] == "video":
            video_type = metadata.get("media_type")
            media_type = (
                video_type
                if isinstance(video_type, str) and video_type in VIDEO_TYPES
                else "application/octet-stream"
            )
        return FileResponse(
            path,
            filename=record["filename"],
            media_type=media_type,
            content_disposition_type="inline",
        )

    def extraction_config(payload: ExtractionInput) -> dict:
        config = payload.model_dump()
        # Calls made before sampling modes existed keep their exact configuration
        # and frame provenance. The worker also defaults missing modes to interval.
        if "sampling_mode" not in payload.model_fields_set:
            config.pop("sampling_mode")
        return config

    def extraction_preview(asset_id: str, payload: ExtractionInput, *, images: bool = False):
        require("assets", asset_id)
        try:
            return preview_extraction(
                store, asset_id, extraction_config(payload), include_images=images
            )
        except (ValueError, OSError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/assets/{asset_id}/extract/preview")
    def preview_extraction_plan(asset_id: str, payload: ExtractionInput):
        return JSONResponse(
            extraction_preview(asset_id, payload), headers={"Cache-Control": "no-store"}
        )

    @app.post("/api/assets/{asset_id}/extract/preview-images")
    def preview_extraction_images(asset_id: str, payload: ExtractionInput):
        require("assets", asset_id)
        if not video_preview_lock.acquire(blocking=False):
            raise HTTPException(409, "A video preview is already being decoded. Try again shortly.")
        try:
            return JSONResponse(
                extraction_preview(asset_id, payload, images=True),
                headers={"Cache-Control": "no-store"},
            )
        finally:
            video_preview_lock.release()

    @app.post("/api/assets/{asset_id}/extract", status_code=202)
    def extract(asset_id: str, payload: ExtractionInput):
        # Validate the same bounded plan as the preview without decoding media or
        # opening the source. Source integrity is checked by the local worker.
        extraction_preview(asset_id, payload)
        try:
            return jobs.submit(asset_id, extraction_config(payload))
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    def video_review_action(action):
        try:
            return action()
        except KeyError as exc:
            raise HTTPException(404, "Video review not found") from exc
        except (ValueError, OSError) as exc:
            raise HTTPException(422, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/assets/{asset_id}/video-reviews/preview", status_code=201)
    def video_review_preview(asset_id: str, payload: VideoReviewInput):
        require("assets", asset_id)
        if not video_preview_lock.acquire(blocking=False):
            raise HTTPException(409, "A video preview is already being decoded. Try again shortly.")
        try:
            return video_review_action(
                lambda: prepare_review(store, asset_id, **payload.model_dump())
            )
        finally:
            video_preview_lock.release()

    @app.get("/api/assets/{asset_id}/video-reviews")
    def video_review_history(asset_id: str):
        require("assets", asset_id)
        return video_review_action(lambda: list_reviews(store, asset_id))

    @app.get("/api/video-reviews/{review_id}")
    def video_review_detail(review_id: str):
        require("video_reviews", review_id)
        return video_review_action(lambda: get_review(store, review_id))

    @app.get("/api/video-reviews/{review_id}/images/{index}")
    def video_review_image(review_id: str, index: int):
        record = require("video_reviews", review_id)
        if not 0 <= index < len(record["images"]):
            raise HTTPException(404, "Preview image not found")
        content = video_review_action(lambda: read_review_images(store, record)[index])
        return Response(content, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.post("/api/video-reviews/{review_id}/run", status_code=202)
    def video_review_run(review_id: str, payload: VideoReviewRunInput):
        require("video_reviews", review_id)
        return video_review_action(
            lambda: queue_review(store, jobs, review_id, **payload.model_dump())
        )

    @app.post("/api/video-reviews/{review_id}/extract/preview")
    def passage_extraction_preview(review_id: str, payload: PassageExtractionInput):
        require("video_reviews", review_id)
        return video_review_action(
            lambda: preview_passage_extraction(store, review_id, **payload.model_dump())
        )

    @app.post("/api/video-reviews/{review_id}/extract", status_code=202)
    def passage_extract(review_id: str, payload: PassageExtractionInput):
        require("video_reviews", review_id)
        plan = video_review_action(
            lambda: preview_passage_extraction(store, review_id, **payload.model_dump())
        )
        try:
            return jobs.submit(
                plan["asset_id"], {"sampling_mode": "passages", "passages_plan": plan}
            )
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/jobs")
    def list_jobs():
        return project_records(store, "jobs", active_project.get())

    @app.get("/api/jobs/{job_id}")
    def job_get(job_id: str):
        require("jobs", job_id)
        return job_detail(store, job_id, project_id=active_project.get())

    def recovery_action(function):
        try:
            return function()
        except KeyError as exc:
            raise HTTPException(404, "Job or source not found") from exc
        except (ValueError, RuntimeError, OSError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/jobs/{job_id}/recovery")
    def job_recovery_preview(job_id: str):
        from iris.job_recovery import preview_job_recovery

        require("jobs", job_id)
        return recovery_action(
            lambda: preview_job_recovery(store, job_id, project_id=active_project.get())
        )

    @app.post("/api/jobs/{job_id}/recover", status_code=202)
    def job_recover(job_id: str, payload: RecoveryInput):
        from iris.job_recovery import recover_job

        require("jobs", job_id)
        return recovery_action(
            lambda: recover_job(
                store,
                jobs,
                job_id,
                fingerprint=payload.fingerprint,
                project_id=active_project.get(),
            )
        )

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

    from iris.benchmark_api import install_benchmark_routes

    install_benchmark_routes(app, store, jobs, require, active_project)
    from iris.model_export_api import install_model_export_routes

    install_model_export_routes(app, store, require, active_project)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")

    return app
