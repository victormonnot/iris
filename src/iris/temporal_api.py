"""Project-scoped HTTP contracts for immutable temporal evidence.

These routes publish sequences, reviewed references and dataset manifests only.
They do not extract media, execute models or launch tracking jobs.
"""

from typing import Annotated, Any, Literal

from fastapi import HTTPException, Path
from pydantic import BaseModel, ConfigDict, Field

from iris import temporal

Identifier = Annotated[str, Field(min_length=1, max_length=128)]
RecordId = Annotated[str, Path(min_length=1, max_length=128)]
Metadata = Annotated[dict[str, Any], Field(max_length=64)]


class StrictInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class SequenceCreate(StrictInput):
    name: str = Field(min_length=1, max_length=160)
    asset_id: Identifier
    frame_ids: list[Identifier] = Field(min_length=1, max_length=10000)
    take_group: str | None = Field(default=None, min_length=1, max_length=160)
    parent_id: Identifier | None = None
    clip: Metadata | None = None
    clock: Metadata | None = None
    timestamps: dict[Identifier, float] | None = Field(default=None, max_length=10000)
    gaps: list[Metadata] | None = Field(default=None, max_length=10001)


class ReferenceCreate(StrictInput):
    expected_revision: int = Field(ge=0)
    payload: Metadata


class DatasetEntry(StrictInput):
    sequence_id: Identifier
    split: Literal["train", "val", "test"]
    reference_id: Identifier | None = None


class DatasetCreate(StrictInput):
    name: str = Field(min_length=1, max_length=160)
    entries: list[DatasetEntry] = Field(min_length=1, max_length=1000)
    parent_id: Identifier | None = None
    notes: str = Field(default="", max_length=4000)


def install_temporal_routes(app, store, require, active_project):
    def action(function):
        try:
            return function()
        except KeyError as exc:
            raise HTTPException(404, "Temporal source or record not found") from exc
        except OSError as exc:
            raise HTTPException(409, "Temporal source media is missing or unreadable") from exc
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/temporal/sequences")
    def sequences():
        return action(lambda: temporal.list_sequences(store, active_project.get()))

    @app.post("/api/temporal/sequences", status_code=201)
    def create_sequence(payload: SequenceCreate):
        require("assets", payload.asset_id)
        for frame_id in payload.frame_ids:
            require("frames", frame_id)
        if payload.parent_id is not None:
            require("temporal_sequences", payload.parent_id)
        return action(
            lambda: temporal.create_sequence(
                store, project_id=active_project.get(), **payload.model_dump()
            )
        )

    @app.get("/api/temporal/sequences/{sequence_id}")
    def sequence(sequence_id: RecordId):
        require("temporal_sequences", sequence_id)
        return action(lambda: temporal.sequence_detail(store, sequence_id))

    @app.get("/api/temporal/sequences/{sequence_id}/references")
    def references(sequence_id: RecordId):
        require("temporal_sequences", sequence_id)
        return action(lambda: temporal.list_references(store, sequence_id))

    @app.post("/api/temporal/sequences/{sequence_id}/references", status_code=201)
    def save_reference(sequence_id: RecordId, payload: ReferenceCreate):
        require("temporal_sequences", sequence_id)
        return action(lambda: temporal.save_reference(store, sequence_id, **payload.model_dump()))

    @app.get("/api/temporal/references/{reference_id}")
    def reference(reference_id: RecordId):
        require("temporal_references", reference_id)
        return action(lambda: temporal.reference_detail(store, reference_id))

    @app.get("/api/temporal/datasets")
    def datasets():
        return action(lambda: temporal.list_temporal_datasets(store, active_project.get()))

    @app.post("/api/temporal/datasets", status_code=201)
    def create_dataset(payload: DatasetCreate):
        for entry in payload.entries:
            require("temporal_sequences", entry.sequence_id)
            if entry.reference_id is not None:
                require("temporal_references", entry.reference_id)
        if payload.parent_id is not None:
            require("temporal_datasets", payload.parent_id)
        return action(
            lambda: temporal.create_temporal_dataset(
                store, project_id=active_project.get(), **payload.model_dump()
            )
        )

    @app.get("/api/temporal/datasets/{dataset_id}")
    def dataset(dataset_id: RecordId):
        require("temporal_datasets", dataset_id)
        return action(lambda: temporal.temporal_dataset_detail(store, dataset_id))
