"""Reports restore offline with historical revisions and reject forged evidence."""

import zipfile
from copy import deepcopy

import pytest
import test_benchmarks_archive as archive_fixtures
from test_benchmark_reports import later_revision, later_trial, save

from iris import combined_provider, multimodal_provider, sam_provider, sam_runtime
from iris.benchmark_analysis import _coverage, _repeatability, validate_comparison_snapshot
from iris.benchmark_reports import _digest, get_report
from iris.store import SCHEMA_VERSION, Store, new_id, now
from iris.workspace_archive import ArchiveError, create_archive
from iris.workspace_restore import inspect_archive, restore_archive

benchmark_workspace = archive_fixtures.benchmark_workspace


def test_report_archive_retains_historical_revision_and_every_original_byte_offline(
    benchmark_workspace, tmp_path, monkeypatch
):
    case = benchmark_workspace
    store = case[0]
    report = save(case)
    later_revision(case)
    later_trial(case)
    expected = archive_fixtures.records(store)

    def forbidden(*_args, **_kwargs):
        pytest.fail("Reports must be inspected and restored without any provider or runtime")

    monkeypatch.setattr(multimodal_provider, "_api_key", forbidden)
    monkeypatch.setattr(multimodal_provider.http.client, "HTTPSConnection", forbidden)
    monkeypatch.setattr(combined_provider, "CombinedOpenAI", forbidden)
    monkeypatch.setattr(sam_provider, "provider_status", forbidden)
    monkeypatch.setattr(sam_runtime, "runtime_status", forbidden)
    monkeypatch.setattr(sam_runtime, "SamRuntime", forbidden)
    archived = create_archive(store.root, tmp_path / "reports.zip")
    assert archived["manifest"]["schema_version"] == SCHEMA_VERSION
    assert archived["manifest"]["counts"]["benchmark_reports"] == 1
    restored = tmp_path / "restored"
    with monkeypatch.context() as context:
        context.setattr(Store, "__init__", forbidden)
        inspected = inspect_archive(archived["path"])
        restore_archive(
            archived["path"], restored, expected_archive_sha256=inspected["archive_sha256"]
        )
    with zipfile.ZipFile(archived["path"]) as archive:
        for item in archived["manifest"]["files"]:
            assert (restored / item["path"]).read_bytes() == archive.read(item["path"])
    reopened = Store(restored)
    assert archive_fixtures.records(reopened) == expected
    assert get_report(reopened, report["id"]) == report


@pytest.mark.parametrize(
    "damage",
    ["hash", "shape", "aggregate", "latency", "cost", "reference", "correction", "raw_source"],
)
def test_archive_rejects_corrupt_or_rehashed_fabricated_report_evidence(
    benchmark_workspace, tmp_path, damage
):
    case = benchmark_workspace
    store = case[0]
    report = deepcopy(save(case))
    comparison = report["snapshot"]["comparison"]
    trial = comparison["configs"][0]["trials"][0]
    if damage == "hash":
        report["snapshot_sha256"] = "a" * 64
    elif damage == "shape":
        report["snapshot"]["unsupported"] = True
    elif damage == "aggregate":
        comparison["coverage"]["complete_trial_count"] += 1
    elif damage == "latency":
        trial["latency"]["total_ms"] = 123456
    elif damage == "cost":
        trial["cost"]["usage_cost_usd"] = 0
    elif damage == "reference":
        comparison["benchmark"]["independence"]["independence_notes"] = "Fabricated declaration"
    elif damage == "correction":
        trial["frames"][0]["correction"]["reviewer"] = "Fabricated reviewer"
    else:
        store.update("benchmark_outputs", case[5]["id"], {"raw_response": {"forged": True}})
    if damage not in {"hash", "raw_source"}:
        report["snapshot_sha256"] = _digest(report["snapshot"])
    store.update(
        "benchmark_reports",
        report["id"],
        {key: report[key] for key in ("snapshot", "snapshot_sha256")},
    )
    target = tmp_path / "corrupted.zip"
    with pytest.raises(ArchiveError, match="report"):
        create_archive(store.root, target)
    assert not target.exists()


@pytest.mark.parametrize("omitted", ["failed_trial", "configuration"])
def test_archive_rejects_rehashed_report_that_omits_existing_trial_or_configuration(
    benchmark_workspace, tmp_path, omitted
):
    case = benchmark_workspace
    store = case[0]
    if omitted == "failed_trial":
        hidden = later_trial(case, status="failed")
    else:
        hidden = store.insert(
            "benchmark_configs",
            {**case[3], "id": new_id(), "name": "Other saved candidate", "created_at": now()},
        )
    report = deepcopy(save(case))
    comparison = report["snapshot"]["comparison"]
    if omitted == "failed_trial":
        config = comparison["configs"][0]
        config["trials"] = [trial for trial in config["trials"] if trial["id"] != hidden["id"]]
        config["repeatability"] = _repeatability(config["trials"])
    else:
        comparison["configs"] = [c for c in comparison["configs"] if c["id"] != hidden["id"]]
    comparison["coverage"] = _coverage(comparison["configs"])
    # A forged subset can be internally consistent and checksummed correctly.
    # Only checking source coverage as of the report time catches the omission.
    validate_comparison_snapshot(comparison)
    store.update(
        "benchmark_reports",
        report["id"],
        {"snapshot": report["snapshot"], "snapshot_sha256": _digest(report["snapshot"])},
    )
    target = tmp_path / "omitted.zip"
    with pytest.raises(ArchiveError, match="report"):
        create_archive(store.root, target)
    assert not target.exists()
