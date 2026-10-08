"""Project-scoped, explicitly requested quality reports over frozen evidence."""

from typing import Annotated

from fastapi import HTTPException
from pydantic import Field

from iris import tracking_quality
from iris.temporal_api import Identifier, RecordId, StrictInput


class QualityCreate(StrictInput):
    reference_id: Identifier
    class_mapping: dict[Annotated[str, Field(min_length=1, max_length=16)], Identifier | None] = (
        Field(min_length=1, max_length=100)
    )
    iou_threshold: float = Field(default=0.5, gt=0, le=1)


def install_tracking_quality_routes(app, store, require, active_project):
    def action(function):
        try:
            return function()
        except KeyError as exc:
            raise HTTPException(
                404, "Tracking comparison, reference or quality report not found"
            ) from exc
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/temporal/tracking-quality-status")
    def status():
        return tracking_quality.status()

    @app.get("/api/temporal/tracking-comparisons/{comparison_id}/quality-reports")
    def reports(comparison_id: RecordId):
        require("jobs", comparison_id)
        return action(
            lambda: tracking_quality.list_quality_reports(
                store, comparison_id, project_id=active_project.get()
            )
        )

    @app.post("/api/temporal/tracking-comparisons/{comparison_id}/quality-reports", status_code=201)
    def create(comparison_id: RecordId, payload: QualityCreate):
        require("jobs", comparison_id)
        require("temporal_references", payload.reference_id)
        return action(
            lambda: tracking_quality.create_quality_report(
                store, comparison_id, project_id=active_project.get(), **payload.model_dump()
            )
        )

    @app.get("/api/temporal/tracking-quality-reports/{report_id}")
    def detail(report_id: RecordId):
        require("tracking_quality_reports", report_id)
        return action(
            lambda: tracking_quality.get_quality_report(
                store, report_id, project_id=active_project.get()
            )
        )
