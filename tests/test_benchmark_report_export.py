"""External usage receipts stay distinct from budget and incomplete delivery."""

import json

import pytest
from test_benchmark import workspace as workspace
from test_benchmark_combined import configured, launch, run
from test_benchmark_combined import providers as providers

from iris import multimodal_provider, sam_runtime
from iris.benchmark_report_export import export_html, export_json
from iris.benchmark_reports import create_report, preview_report


@pytest.mark.parametrize("timeout", [False, True])
def test_export_preserves_known_and_unknown_usage_without_rerunning(
    workspace, providers, monkeypatch, timeout
):
    store = workspace[0]
    reference, candidate, preview = configured(workspace)
    if timeout:
        providers["fail"] = "review"
    trial = launch(workspace, reference, candidate, preview)
    result = run(workspace, trial)
    store.update(
        "jobs", trial["job_id"], {"status": "failed" if timeout else "succeeded", "result": result}
    )

    def forbidden(*args, **kwargs):
        pytest.fail("No model or transport is needed for report export")

    monkeypatch.setattr(multimodal_provider, "_request", forbidden)
    monkeypatch.setattr(sam_runtime, "SamRuntime", forbidden)
    values = {
        "role": "tuning",
        "title": "Simulated combined receipts",
        "evidence_kind": "simulation",
    }
    prepared = preview_report(store, reference["id"], **values)
    report = create_report(
        store, reference["id"], **values, expected_fingerprint=prepared["fingerprint"]
    )
    assert json.loads(export_json(report)) == report
    html = export_html(report).decode()
    assert "Planning allowance:" in html and "Approved budget:" in html
    if timeout:
        assert "Total unknown; known usage subtotal USD 0.015000" in html
        assert "1 unknown delivery outcomes" in html
        assert "0/1 complete trials" in html
    else:
        assert "USD 0.060000 estimated from recorded usage" in html
        assert "already included in processing time" in html
        assert "1/1 complete trials" in html
