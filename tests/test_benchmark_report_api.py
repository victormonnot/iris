"""Read-only comparisons, confirmed snapshots and scoped standalone downloads."""

import json
from html.parser import HTMLParser

import pytest
from test_benchmark_api import completed_trial, config, freeze
from test_benchmark_api import workspace as workspace

from iris import benchmark_runs, multimodal_provider, sam_provider, sam_runtime
from iris.projects import create_project


def prepared(workspace):
    reference = freeze(workspace)
    candidate = config(workspace, reference)
    trial = completed_trial(workspace, reference, candidate)
    return reference, candidate, trial


def payload(**changes):
    return {
        "role": "tuning",
        "title": "SIMULATION: candidate comparison",
        "objective": "Verify report software using synthetic images and outputs.",
        "conclusion": "No claim about real model quality.",
        "evidence_kind": "simulation",
        **changes,
    }


def save(client, reference, **changes):
    path = f"/api/benchmarks/{reference['id']}/reports"
    data = payload(**changes)
    preview = client.post(path + "/preview", json=data)
    assert preview.status_code == 200, preview.text
    created = client.post(
        path, json={**data, "expected_fingerprint": preview.json()["fingerprint"]}
    )
    assert created.status_code == 201, created.text
    return created.json(), preview.json()


def test_comparison_and_report_never_probe_or_run_models(workspace, monkeypatch):
    client, store, _ = workspace
    reference, _, trial = prepared(workspace)

    def forbidden(*args, **kwargs):
        pytest.fail("Reporting must not probe, load or request a model")

    monkeypatch.setattr(multimodal_provider, "_request", forbidden)
    monkeypatch.setattr(multimodal_provider, "provider_status", forbidden)
    monkeypatch.setattr(sam_provider, "provider_status", forbidden)
    monkeypatch.setattr(sam_runtime, "SamRuntime", forbidden)
    monkeypatch.setattr(benchmark_runs, "catalog", forbidden)
    baseline = {
        name: store.list(name) for name in ("jobs", "annotation_revisions", "benchmark_outputs")
    }
    response = client.get(
        f"/api/benchmarks/{reference['id']}/comparison", params={"role": "tuning"}
    )
    assert response.status_code == 200, response.text
    comparison = response.json()
    assert comparison["configs"][0]["trials"][0]["id"] == trial["id"]
    assert comparison["configs"][0]["trials"][0]["quality"]["complete"]
    report, preview = save(client, reference)
    assert report["snapshot"]["comparison"] == comparison
    assert report["snapshot_sha256"] == preview["fingerprint"]
    listing = client.get(f"/api/benchmarks/{reference['id']}/reports").json()
    assert listing[0]["id"] == report["id"] and "snapshot" not in listing[0]
    assert client.get(f"/api/benchmark-reports/{report['id']}").json() == report
    for name, rows in baseline.items():
        assert store.list(name) == rows


def test_saved_report_survives_new_corrections_and_is_not_current_view(workspace):
    client, store, _ = workspace
    reference, _, trial = prepared(workspace)
    report, _ = save(client, reference)
    original = json.dumps(report, sort_keys=True)
    output = trial["outputs"][0]
    corrected = client.put(
        f"/api/benchmark-outputs/{output['id']}/correction",
        json={
            "expected_revision": 0,
            "boxes": [{"id": "new", "label": "person", "box": [2, 3, 30, 35]}],
            "status": "reviewed",
            "reviewer": "Synthetic reviewer",
        },
    )
    assert corrected.status_code == 200, corrected.text
    current = client.get(
        f"/api/benchmarks/{reference['id']}/comparison", params={"role": "tuning"}
    ).json()
    assert current["configs"][0]["trials"][0]["corrections"]["reviewed_count"] == 1
    assert (
        report["snapshot"]["comparison"]["configs"][0]["trials"][0]["corrections"]["reviewed_count"]
        == 0
    )
    assert (
        json.dumps(client.get(f"/api/benchmark-reports/{report['id']}").json(), sort_keys=True)
        == original
    )
    assert store.list("annotation_suggestions") == []


def test_exports_are_saved_data_and_escape_all_author_text(workspace):
    client = workspace[0]
    reference, _, _ = prepared(workspace)
    hostile = '</title><script src="https://example.invalid/leak">bad()</script>&'
    report, _ = save(client, reference, title=hostile, objective=hostile, conclusion=hostile)
    prefix = f"/api/benchmark-reports/{report['id']}"
    exported = client.get(prefix + "/export.json")
    assert exported.status_code == 200 and exported.json() == report
    assert "attachment;" in exported.headers["content-disposition"]
    assert exported.headers["cache-control"] == "no-store"
    html = client.get(prefix + "/export.html")
    assert html.status_code == 200, html.text
    assert "default-src 'none'" in html.headers["content-security-policy"]
    assert html.headers["x-content-type-options"] == "nosniff"
    assert "&lt;script" in html.text and report["snapshot_sha256"] in html.text
    assert "Simulation" in html.text and "Local execution cost not measured" in html.text
    assert "Unmeasured" in html.text and "100.0%" not in html.text

    class Tags(HTMLParser):
        unsafe = []

        def handle_starttag(self, tag, attrs):
            if tag in {"script", "iframe", "img", "link", "object", "embed", "form"}:
                self.unsafe.append(tag)

    parser = Tags()
    parser.feed(html.text)
    assert parser.unsafe == []


def test_every_report_endpoint_enforces_benchmark_project_scope(workspace):
    client, store, _ = workspace
    reference, _, _ = prepared(workspace)
    report, _ = save(client, reference)
    project = create_project(store, name="Another project")
    params = {"project_id": project["id"]}
    base = f"/api/benchmarks/{reference['id']}"
    for path in (
        base + "/comparison",
        base + "/reports",
        f"/api/benchmark-reports/{report['id']}",
        f"/api/benchmark-reports/{report['id']}/export.json",
        f"/api/benchmark-reports/{report['id']}/export.html",
    ):
        assert client.get(path, params=params).status_code == 404
    for path, data in (
        (base + "/reports/preview", payload()),
        (base + "/reports", payload(expected_fingerprint=report["snapshot_sha256"])),
    ):
        assert client.post(path, params=params, json=data).status_code == 404


@pytest.mark.parametrize(
    "changes",
    [
        {"role": "both"},
        {"title": ""},
        {"title": True},
        {"objective": "x" * 4001},
        {"evidence_kind": "verified"},
        {"trial_ids": []},
        {"run_models": True},
    ],
)
def test_report_inputs_are_bounded_and_cannot_select_successful_attempts_only(workspace, changes):
    client = workspace[0]
    reference = freeze(workspace)
    response = client.post(
        f"/api/benchmarks/{reference['id']}/reports/preview", json=payload(**changes)
    )
    assert response.status_code == 422


def test_empty_role_can_be_inspected_but_not_saved_as_measured_report(workspace):
    client, store, _ = workspace
    reference = freeze(workspace)
    base = f"/api/benchmarks/{reference['id']}"
    comparison = client.get(base + "/comparison").json()
    assert comparison["role"] == "evaluation"
    assert comparison["coverage"]["trial_count"] == 0
    preview = client.post(base + "/reports/preview", json=payload()).json()
    response = client.post(
        base + "/reports", json=payload(expected_fingerprint=preview["fingerprint"])
    )
    assert response.status_code == 409
    assert store.list("benchmark_reports") == []
    assert client.get(base + "/comparison", params={"role": "all"}).status_code == 422


def test_stale_report_preview_is_rejected_and_same_snapshot_is_idempotent(workspace):
    client = workspace[0]
    reference, candidate, _ = prepared(workspace)
    report, preview = save(client, reference)
    path = f"/api/benchmarks/{reference['id']}/reports"
    retried = client.post(path, json=payload(expected_fingerprint=preview["fingerprint"]))
    assert retried.status_code == 201 and retried.json()["id"] == report["id"]
    completed_trial(workspace, reference, candidate)
    stale = client.post(path, json=payload(expected_fingerprint=preview["fingerprint"]))
    assert stale.status_code == 409
    assert len(client.get(path).json()) == 1
