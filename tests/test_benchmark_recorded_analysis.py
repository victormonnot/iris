"""Recorded evidence stays distinct from new inference, spending and repeated execution."""

from copy import deepcopy

import pytest
from test_benchmark_api import workspace as workspace
from test_benchmark_recorded import bundle, configured, imported

from iris.benchmark_analysis import (
    build_comparison,
    validate_comparison_snapshot,
    validate_snapshot_sources,
)
from iris.benchmark_report_export import _cost
from iris.benchmark_runs import run_benchmark_trial
from iris.store import now


@pytest.mark.parametrize("transform", ["identity", "threshold", "review"])
def test_recorded_comparison_cost_scope_and_sources(workspace, transform):
    _, store, _ = workspace
    reference, _, source, _, trial, _ = imported(workspace, transform)
    report = build_comparison(store, reference["id"], role="tuning")
    candidate = report["configs"][0]
    entry = candidate["trials"][0]
    assert entry["quality"]["complete"]
    assert entry["cost"]["recorded"] and not entry["cost"]["external"]
    assert entry["cost"]["usage_cost_usd"] is None
    assert entry["cost"]["source_costs"]["dinox_estimate_cny"] == sum(
        frame["dinox"]["receipt"]["estimated_cost_cny"] for frame in source["frames"]
    )
    assert "not provider inference" in entry["latency"]["includes"]
    assert "no new provider requests" in _cost(entry["cost"])
    assert "shared source costs are not additive" in _cost(entry["cost"])
    assert entry["corrections"]["reviewed_count"] == 0
    assert candidate["repeatability"]["measured"] is False
    assert validate_comparison_snapshot(report) == report
    with store.connect() as conn:
        assert validate_snapshot_sources(conn, report, store=store) == report
    if transform == "review":
        assert entry["identity"]["returned_models"] == ["gpt-6-astra"]
    output = store.list("benchmark_outputs", trial_id=trial["id"])[0]
    changed = deepcopy(output["metadata"])
    changed["source"]["dinox"]["estimated_cost_cny"] = 10
    store.update("benchmark_outputs", output["id"], {"metadata": changed})
    with pytest.raises(ValueError):
        build_comparison(store, reference["id"], role="tuning")


def test_reimported_source_never_measures_provider_repeatability(workspace):
    client, store, _ = workspace
    reference, config, source, _, original, _ = imported(workspace)
    path = f"/api/benchmarks/{reference['id']}/recorded-trials"
    # A different source receipt is another submitted bundle, not proof of another call.
    source["frames"][0]["dinox"]["receipt"]["elapsed_ms"] += 1
    body = {"config_id": config["id"], "role": "tuning", "bundle": source}
    preview = client.post(path + "/preview", json=body)
    assert preview.status_code == 200
    created = client.post(
        path, json={**body, "expected_fingerprint": preview.json()["fingerprint"]}
    )
    assert created.status_code == 202
    trial = created.json()
    assert trial["id"] != original["id"]
    store.update("jobs", trial["job_id"], {"status": "running"})
    result = run_benchmark_trial(store, trial["id"], lambda *_: None, lambda: False)
    store.update(
        "jobs", trial["job_id"], {"status": "succeeded", "result": result, "finished_at": now()}
    )
    report = build_comparison(store, reference["id"], role="tuning")
    repeat = report["configs"][0]["repeatability"]
    assert repeat["complete_count"] == 2
    assert repeat["measured"] is False and repeat["identical_geometry"] is None
    assert "do not establish repeated provider execution" in repeat["note"]
    assert validate_comparison_snapshot(report) == report


@pytest.mark.parametrize(
    "damage",
    ["missing_attempt", "foreign_attempt", "result_owner", "result_operation", "params_operation"],
)
def test_recorded_comparison_requires_owning_job_and_attempt(workspace, damage):
    _, store, _ = workspace
    reference, _, _, _, trial, _ = imported(workspace)
    job = store.get("jobs", trial["job_id"])
    if damage == "params_operation":
        store.update(
            "jobs",
            job["id"],
            {"params": {"trial_id": trial["id"], "operation": "unrelated_operation"}},
        )
    else:
        result = deepcopy(job["result"])
        if damage == "missing_attempt":
            result.pop("benchmark_attempt_id")
        elif damage == "foreign_attempt":
            result["benchmark_attempt_id"] = "another-attempt"
        elif damage == "result_owner":
            result["trial_id"] = "another-trial"
        else:
            result["operation"] = "unrelated_operation"
        store.update("jobs", job["id"], {"result": result})
    with pytest.raises(ValueError, match="Recorded"):
        build_comparison(store, reference["id"], role="tuning")


@pytest.mark.parametrize("status", ["queued", "cancelled", "interrupted", "failed"])
def test_recorded_comparison_keeps_preimport_jobs_without_an_attempt(workspace, status):
    client, store, _ = workspace
    reference, candidate = configured(workspace, "identity")
    source = bundle(store, reference, candidate)
    path = f"/api/benchmarks/{reference['id']}/recorded-trials"
    body = {"config_id": candidate["id"], "role": "tuning", "bundle": source}
    preview = client.post(path + "/preview", json=body)
    assert preview.status_code == 200
    created = client.post(
        path, json={**body, "expected_fingerprint": preview.json()["fingerprint"]}
    )
    assert created.status_code == 202
    trial = created.json()
    store.update("jobs", trial["job_id"], {"status": status})
    report = build_comparison(store, reference["id"], role="tuning")
    entry = report["configs"][0]["trials"][0]
    assert entry["status"] == status and entry["quality"]["complete"] is False
    assert entry["quality"]["metrics"] is None
    assert entry["coverage"] == {"planned": 1, "ready": 0, "failed": 0, "missing": 1}
    assert entry["latency"]["measured_count"] == 0 and entry["latency"]["total_ms"] is None
    assert validate_comparison_snapshot(report) == report
    with store.connect() as connection:
        assert validate_snapshot_sources(connection, report, store=store) == report
