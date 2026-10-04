"""Project-scoped HTTP boundary for independent annotation benchmarks."""

import hashlib
from typing import Literal

from fastapi import HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field

from iris.annotations import MAX_BOXES
from iris.benchmark import (
    benchmark_candidates,
    benchmark_detail,
    benchmark_frame,
    create_benchmark,
    create_benchmark_config,
    list_benchmarks,
    lock_benchmark,
    preview_benchmark,
    preview_benchmark_config,
)
from iris.benchmark_corrections import correction_document, save_correction, timer_action
from iris.benchmark_runs import (
    benchmark_trial_detail,
    create_benchmark_trial,
    preview_benchmark_trial,
)


class StrictInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class ReferencePreview(StrictInput):
    frame_ids: list[str] = Field(min_length=2, max_length=50)
    roles: dict[str, Literal["tuning", "evaluation"]]
    reviewer: str = Field(min_length=1, max_length=120)
    independence_notes: str = Field(min_length=1, max_length=4000)
    independent_reference: bool
    taxonomy_id: str | None = None


class ReferenceCreate(ReferencePreview):
    name: str = Field(min_length=1, max_length=160)
    expected_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


class MultimodalSettings(StrictInput):
    image_long_edge: Literal[512, 1024, 1536, 2048] = 1536
    reasoning_effort: Literal["low", "medium", "high", "xhigh", "max"] = "low"
    max_output_tokens: int = Field(default=4096, ge=1024, le=8192)


class SegmentationSettings(StrictInput):
    class_prompts: dict[str, str] | None = None
    threshold: float = Field(default=0.5, ge=0, le=1)
    device: Literal["cuda"] = "cuda"


class ConfigPreview(StrictInput):
    model_id: str = Field(min_length=1, max_length=128)
    threshold: float = Field(default=0.5, ge=0, le=1)
    device: Literal["cpu", "cuda"] = "cpu"
    inference_mode: Literal["full", "tiled"] = "full"
    tile_size: int = Field(default=640, ge=64, le=4096)
    overlap: float = Field(default=0.2, ge=0, le=0.5)
    approach: Literal["local_detector", "multimodal", "segmentation", "combined"] = "local_detector"
    multimodal: MultimodalSettings | None = None
    segmentation: SegmentationSettings | None = None


class ConfigCreate(ConfigPreview):
    name: str = Field(min_length=1, max_length=160)
    expected_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


class LockInput(StrictInput):
    expected_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


class TrialPreview(StrictInput):
    config_id: str = Field(min_length=1, max_length=128)
    role: Literal["tuning", "evaluation"]


class TrialCreate(TrialPreview):
    expected_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    approve_external: bool = False
    max_cost_usd: float | None = Field(default=None, ge=0, le=1000)
    preview_token: str | None = Field(default=None, max_length=512)


class CorrectionBox(StrictInput):
    id: str = Field(min_length=1, max_length=128)
    label: str = Field(min_length=1, max_length=128)
    box: list[float] = Field(min_length=4, max_length=4)
    proposal_id: str | None = Field(default=None, max_length=128)


class CorrectionInput(StrictInput):
    expected_revision: int = Field(ge=0)
    boxes: list[CorrectionBox] = Field(max_length=MAX_BOXES)
    status: Literal["draft", "reviewed"] = "draft"
    reviewer: str = Field(default="", max_length=120)
    notes: str = Field(default="", max_length=4000)
    timer_revision: int | None = Field(default=None, ge=0)
    timer_token: str | None = Field(default=None, min_length=16, max_length=128)


class TimerInput(StrictInput):
    action: Literal["start", "heartbeat", "pause"]
    expected_revision: int = Field(ge=0)
    token: str = Field(min_length=16, max_length=128)
    operation_id: str = Field(min_length=16, max_length=128)
    reviewer: str = Field(default="", max_length=120)
    discard_unconfirmed: bool = False


def install_benchmark_routes(app, store, jobs, require, active_project):
    def action(function):
        try:
            return function()
        except KeyError as exc:
            raise HTTPException(404, "Benchmark or source not found") from exc
        except (ValueError, RuntimeError, OSError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/benchmark-providers")
    def providers():
        from iris.benchmark_multimodal import provider_catalog
        from iris.benchmark_segmentation import provider_catalog as sam_catalog

        return action(lambda: {**provider_catalog(), **sam_catalog(store)})

    @app.get("/api/benchmark-configs/{config_id}/frames/{frame_id}/input-image")
    def external_input_image(config_id: str, frame_id: str):
        from iris.benchmark_multimodal import input_image

        require("benchmark_configs", config_id)
        return Response(
            action(lambda: input_image(store, config_id, frame_id)),
            media_type="image/png",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/benchmark-candidates")
    def candidates(taxonomy_id: str | None = None):
        return action(
            lambda: benchmark_candidates(
                store, project_id=active_project.get(), taxonomy_id=taxonomy_id
            )
        )

    @app.get("/api/benchmarks")
    def listing():
        return action(lambda: list_benchmarks(store, project_id=active_project.get()))

    @app.post("/api/benchmarks/preview")
    def preview(payload: ReferencePreview):
        return action(
            lambda: preview_benchmark(
                store, project_id=active_project.get(), **payload.model_dump()
            )
        )

    @app.post("/api/benchmarks", status_code=201)
    def create(payload: ReferenceCreate):
        return action(
            lambda: create_benchmark(store, project_id=active_project.get(), **payload.model_dump())
        )

    @app.get("/api/benchmarks/{benchmark_id}")
    def detail(benchmark_id: str):
        require("benchmarks", benchmark_id)
        return action(lambda: benchmark_detail(store, benchmark_id))

    @app.post("/api/benchmarks/{benchmark_id}/configs/preview")
    def config_preview(benchmark_id: str, payload: ConfigPreview):
        require("benchmarks", benchmark_id)
        return action(lambda: preview_benchmark_config(store, benchmark_id, **payload.model_dump()))

    @app.post("/api/benchmarks/{benchmark_id}/configs", status_code=201)
    def config_create(benchmark_id: str, payload: ConfigCreate):
        require("benchmarks", benchmark_id)
        return action(lambda: create_benchmark_config(store, benchmark_id, **payload.model_dump()))

    @app.post("/api/benchmarks/{benchmark_id}/lock")
    def lock(benchmark_id: str, payload: LockInput):
        require("benchmarks", benchmark_id)
        return action(lambda: lock_benchmark(store, benchmark_id, **payload.model_dump()))

    @app.post("/api/benchmarks/{benchmark_id}/trials/preview")
    def trial_preview(benchmark_id: str, payload: TrialPreview):
        require("benchmarks", benchmark_id)
        require("benchmark_configs", payload.config_id)
        return action(lambda: preview_benchmark_trial(store, benchmark_id, **payload.model_dump()))

    @app.post("/api/benchmarks/{benchmark_id}/trials", status_code=202)
    def trial_create(benchmark_id: str, payload: TrialCreate):
        require("benchmarks", benchmark_id)
        require("benchmark_configs", payload.config_id)
        return action(
            lambda: create_benchmark_trial(store, jobs, benchmark_id, **payload.model_dump())
        )

    @app.get("/api/benchmark-trials/{trial_id}")
    def trial_detail(trial_id: str):
        require("benchmark_trials", trial_id)
        return action(lambda: benchmark_trial_detail(store, trial_id))

    @app.get("/api/benchmarks/{benchmark_id}/frames/{frame_id}/image")
    def frame_image(benchmark_id: str, frame_id: str):
        require("benchmarks", benchmark_id)

        def read():
            frame = benchmark_frame(store, benchmark_id, frame_id)
            path = (store.root / frame["image_path"]).resolve()
            if not path.is_relative_to(store.root.resolve() / "benchmarks" / benchmark_id):
                raise ValueError("The frozen image path is outside this benchmark")
            payload = path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != frame["image_file_sha256"]:
                raise ValueError("The frozen benchmark image has changed")
            return Response(payload, media_type="image/png", headers={"Cache-Control": "no-store"})

        return action(read)

    @app.get("/api/benchmark-outputs/{output_id}/correction")
    def correction_get(output_id: str):
        require("benchmark_outputs", output_id)
        return action(lambda: correction_document(store, output_id))

    @app.put("/api/benchmark-outputs/{output_id}/correction")
    def correction_put(output_id: str, payload: CorrectionInput):
        require("benchmark_outputs", output_id)
        return action(lambda: save_correction(store, output_id, **payload.model_dump()))

    @app.post("/api/benchmark-outputs/{output_id}/timer")
    def timer(output_id: str, payload: TimerInput):
        require("benchmark_outputs", output_id)
        return action(lambda: timer_action(store, output_id, **payload.model_dump()))

    @app.get("/api/benchmark-outputs/{output_id}/corrections/{revision}")
    def correction_revision(output_id: str, revision: int):
        require("benchmark_outputs", output_id)
        rows = store.list("benchmark_corrections", output_id=output_id)
        row = next((row for row in rows if row["revision"] == revision), None)
        if row is None:
            raise HTTPException(404, "Correction revision not found")
        return row
