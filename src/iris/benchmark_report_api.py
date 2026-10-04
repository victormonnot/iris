"""Project-scoped comparison and immutable report HTTP endpoints."""

from typing import Literal

from fastapi import HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field


class ReportPreview(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    role: Literal["tuning", "evaluation"]
    title: str = Field(min_length=1, max_length=160)
    objective: str = Field(default="", max_length=4000)
    conclusion: str = Field(default="", max_length=4000)
    evidence_kind: Literal["not_declared", "simulation", "real_data"] = "not_declared"


class ReportCreate(ReportPreview):
    expected_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


def install_benchmark_report_routes(app, store, require):
    from iris.benchmark_analysis import build_comparison
    from iris.benchmark_report_export import export_html, export_json
    from iris.benchmark_reports import create_report, get_report, list_reports, preview_report

    def action(function):
        try:
            return function()
        except KeyError as exc:
            raise HTTPException(404, "Benchmark or report not found") from exc
        except (ValueError, RuntimeError, OSError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/benchmarks/{benchmark_id}/comparison")
    def comparison(benchmark_id: str, role: Literal["tuning", "evaluation"] = "evaluation"):
        require("benchmarks", benchmark_id)
        return action(lambda: build_comparison(store, benchmark_id, role=role))

    @app.get("/api/benchmarks/{benchmark_id}/reports")
    def listing(benchmark_id: str):
        require("benchmarks", benchmark_id)
        return action(lambda: list_reports(store, benchmark_id))

    @app.post("/api/benchmarks/{benchmark_id}/reports/preview")
    def preview(benchmark_id: str, payload: ReportPreview):
        require("benchmarks", benchmark_id)
        return action(lambda: preview_report(store, benchmark_id, **payload.model_dump()))

    @app.post("/api/benchmarks/{benchmark_id}/reports", status_code=201)
    def create(benchmark_id: str, payload: ReportCreate):
        require("benchmarks", benchmark_id)
        return action(lambda: create_report(store, benchmark_id, **payload.model_dump()))

    @app.get("/api/benchmark-reports/{report_id}")
    def detail(report_id: str):
        require("benchmark_reports", report_id)
        return action(lambda: get_report(store, report_id))

    def exported(report_id, renderer, extension, media_type):
        require("benchmark_reports", report_id)
        row = action(lambda: get_report(store, report_id))
        return Response(
            action(lambda: renderer(row)),
            media_type=media_type,
            headers={
                "Content-Disposition": (
                    f'attachment; filename="iris-benchmark-{row["id"]}.{extension}"'
                ),
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
                "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; "
                "base-uri 'none'; form-action 'none'; frame-ancestors 'none'",
            },
        )

    @app.get("/api/benchmark-reports/{report_id}/export.json")
    def json_download(report_id: str):
        return exported(report_id, export_json, "json", "application/json")

    @app.get("/api/benchmark-reports/{report_id}/export.html")
    def html_download(report_id: str):
        return exported(report_id, export_html, "html", "text/html")
