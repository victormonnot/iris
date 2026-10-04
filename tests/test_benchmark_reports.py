"""Atomic report confirmation, immutable evidence and historical source validation."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Barrier

import pytest
import test_benchmarks_archive as archive_fixtures

from iris.benchmark import BenchmarkConflict
from iris.benchmark_reports import (
    _digest,
    create_report,
    get_report,
    list_reports,
    preview_report,
    validate_report_row,
)
from iris.projects import create_project, project_records, record_project
from iris.store import new_id, now

benchmark_workspace = archive_fixtures.benchmark_workspace


def settings(**changes):
    return {
        "role": "evaluation",
        "title": "Fixture comparison",
        "objective": "Exercise storage with simulated evidence.",
        "conclusion": "No real candidate quality claim.",
        "evidence_kind": "simulation",
        **changes,
    }


def save(case, **changes):
    store, benchmark = case[:2]
    values = settings(**changes)
    preview = preview_report(store, benchmark["id"], **values)
    return create_report(
        store, benchmark["id"], expected_fingerprint=preview["fingerprint"], **values
    )


def later_revision(case):
    store, *_, correction, _timer = case
    return store.insert(
        "benchmark_corrections",
        {**correction, "id": new_id(), "revision": correction["revision"] + 1, "notes": "Later"},
    )


def later_trial(case, *, role="evaluation", status="failed"):
    store, benchmark, _manifest, config, *_ = case
    trial_id = new_id()
    job = store.insert(
        "jobs",
        dict(
            id=new_id(),
            kind="benchmark",
            status=status,
            params={"trial_id": trial_id},
            created_at=now(),
        ),
    )
    original = case[4]
    return store.insert(
        "benchmark_trials",
        {
            **original,
            "id": trial_id,
            "benchmark_id": benchmark["id"],
            "config_id": config["id"],
            "split": role,
            "job_id": job["id"],
            "created_at": now(),
        },
    )


def test_preview_is_deterministic_read_only_and_report_has_compact_validated_listing(
    benchmark_workspace,
):
    case = benchmark_workspace
    store, benchmark = case[:2]
    original = archive_fixtures.records(store)
    first = preview_report(store, benchmark["id"], **settings())
    assert preview_report(store, benchmark["id"], **settings()) == first
    assert archive_fixtures.records(store) == original
    assert first["fingerprint"] == _digest(first["snapshot"])
    assert first["snapshot"]["protocol"] == "iris-benchmark-report-v1"
    report = save(case)
    assert report["snapshot"] == first["snapshot"]
    assert get_report(store, report["id"]) == report
    assert get_report(store, "missing") is None
    assert list_reports(store, benchmark["id"]) == [
        {key: report[key] for key in ("id", "benchmark_id", "snapshot_sha256", "created_at")}
        | {"title": settings()["title"], "role": "evaluation", "evidence_kind": "simulation"}
    ]


def test_concurrent_confirmation_creates_one_immutable_report(benchmark_workspace):
    store, benchmark = benchmark_workspace[:2]
    preview = preview_report(store, benchmark["id"], **settings())
    barrier = Barrier(2)

    def create(_):
        barrier.wait(timeout=10)
        return create_report(
            store, benchmark["id"], expected_fingerprint=preview["fingerprint"], **settings()
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        reports = list(pool.map(create, range(2)))
    assert reports[0] == reports[1]
    assert len(store.list("benchmark_reports")) == 1


@pytest.mark.parametrize("change", ["text", "correction", "trial", "job"])
def test_confirmation_refuses_new_evidence_or_changed_text(benchmark_workspace, change):
    case = benchmark_workspace
    store, benchmark = case[:2]
    values = settings()
    preview = preview_report(store, benchmark["id"], **values)
    if change == "text":
        values["conclusion"] = "A changed conclusion."
    elif change == "correction":
        later_revision(case)
    elif change == "trial":
        later_trial(case)
    else:
        store.update("jobs", case[4]["job_id"], {"status": "failed"})
    with pytest.raises(BenchmarkConflict, match="changed"):
        create_report(store, benchmark["id"], expected_fingerprint=preview["fingerprint"], **values)
    assert store.list("benchmark_reports") == []


@pytest.mark.parametrize("status", ["queued", "running"])
def test_active_trial_in_selected_role_blocks_saving_but_preview_stays_available(
    benchmark_workspace, status
):
    case = benchmark_workspace
    later_trial(case, status=status)
    with pytest.raises(BenchmarkConflict, match="finish"):
        save(case)
    assert case[0].list("benchmark_reports") == []


def test_failed_trial_is_reportable_and_other_role_activity_is_separate(benchmark_workspace):
    case = benchmark_workspace
    store = case[0]
    store.update("jobs", case[4]["job_id"], {"status": "failed"})
    # Retain the actual tuning partition in the trial's immutable definition.
    trial = later_trial(case, role="tuning", status="running")
    frozen = deepcopy(trial["config"])
    frozen["role"] = "tuning"
    frozen["frame_ids"] = [f["frame_id"] for f in case[2]["frames"] if f["role"] == "tuning"]
    store.update("benchmark_trials", trial["id"], {"config": frozen})
    assert save(case)["snapshot"]["comparison"]["role"] == "evaluation"


def test_role_without_trials_cannot_be_saved(benchmark_workspace):
    with pytest.raises(BenchmarkConflict, match="at least one"):
        save(benchmark_workspace, role="tuning")


def test_saved_report_keeps_original_revisions_after_later_results(benchmark_workspace):
    case = benchmark_workspace
    store, benchmark = case[:2]
    report = save(case)
    later_revision(case)
    later_trial(case)
    store.insert(
        "benchmark_configs",
        {**case[3], "id": new_id(), "name": "A later candidate", "created_at": now()},
    )
    store.update("benchmarks", benchmark["id"], {"name": "A renamed benchmark"})
    assert get_report(store, report["id"]) == report
    with pytest.raises(BenchmarkConflict, match="changed"):
        create_report(
            store, benchmark["id"], expected_fingerprint=report["snapshot_sha256"], **settings()
        )
    newer = save(case)
    assert newer["id"] != report["id"]
    assert len(list_reports(store, benchmark["id"])) == 2


@pytest.mark.parametrize(
    "changes",
    [
        {"title": " "},
        {"title": "x" * 161},
        {"title": 123},
        {"objective": "x" * 4001},
        {"conclusion": None},
        {"evidence_kind": "verified_real"},
        {"evidence_kind": []},
        {"role": "test"},
    ],
)
def test_invalid_report_parameters_do_not_write(benchmark_workspace, changes):
    store, benchmark = benchmark_workspace[:2]
    with pytest.raises(ValueError):
        preview_report(store, benchmark["id"], **settings(**changes))
    assert store.list("benchmark_reports") == []


def test_report_canonical_text_and_size_limit(benchmark_workspace, monkeypatch):
    store, benchmark = benchmark_workspace[:2]
    preview = preview_report(store, benchmark["id"], **settings(title="  A title  "))
    assert preview["snapshot"]["title"] == "A title"
    monkeypatch.setattr("iris.benchmark_reports.MAX_SNAPSHOT_BYTES", 100)
    with pytest.raises(ValueError, match="16 MiB"):
        preview_report(store, benchmark["id"], **settings())


@pytest.mark.parametrize("damage", ["hash", "owner", "protocol", "source", "correction"])
def test_saved_report_validation_rejects_corrupted_history(benchmark_workspace, damage):
    case = benchmark_workspace
    store = case[0]
    report = save(case)
    if damage == "hash":
        report["snapshot_sha256"] = "a" * 64
    elif damage == "owner":
        report["benchmark_id"] = new_id()
    elif damage == "protocol":
        report["snapshot"]["protocol"] = "untrusted-v2"
        report["snapshot_sha256"] = _digest(report["snapshot"])
    elif damage == "source":
        store.update("benchmark_outputs", case[5]["id"], {"raw_response": {"forged": True}})
    else:
        store.update("benchmark_corrections", case[6]["id"], {"reviewer": "Forged reviewer"})
    with store.connect() as connection, pytest.raises(ValueError):
        validate_report_row(report, connection=connection)


def test_report_ownership_follows_its_benchmark(benchmark_workspace):
    case = benchmark_workspace
    store, benchmark = case[:2]
    report = save(case)
    other = create_project(store, name="Other project")
    assert record_project(store, "benchmark_reports", report) == benchmark["project_id"]
    assert project_records(store, "benchmark_reports", other["id"]) == []
    assert project_records(store, "benchmark_reports", benchmark["project_id"]) == [report]


def test_report_failure_rolls_back_before_insert(benchmark_workspace, monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise ValueError("Invalid original evidence")

    monkeypatch.setattr("iris.benchmark_reports.validate_report_row", forbidden)
    with pytest.raises(ValueError, match="original evidence"):
        save(benchmark_workspace)
    assert benchmark_workspace[0].list("benchmark_reports") == []
