"""Terminal worker status reflects proposal failures while preserving per-image evidence."""

from types import SimpleNamespace

import pytest

from iris import worker
from iris.store import Store, new_id, now


@pytest.mark.parametrize(
    "ready,issues,cancelled,expected",
    [
        (0, 2, False, "failed"),
        (1, 1, False, "succeeded"),
        (2, 0, False, "succeeded"),
        (0, 1, True, "cancelled"),
    ],
)
@pytest.mark.parametrize("kind", ["infer", "benchmark"])
def test_preannotation_terminal_status(
    tmp_path, monkeypatch, ready, issues, cancelled, expected, kind
):
    store = Store(tmp_path / "workspace")
    identifier = new_id()
    store.insert(
        "jobs",
        {
            "id": identifier,
            "kind": kind,
            "status": "running",
            "params": {"comparison_id": "synthetic-output-only", "trial_id": "synthetic-trial"},
            "created_at": now(),
            "cancel_requested": cancelled,
        },
    )
    result = {
        "preannotation": {
            "frames": [{"state": "invalid_output"}] if issues else [],
            "frames_ready": ready,
            "frames_issues": issues,
            "suggestions_created": ready,
        }
    }
    if kind == "benchmark":
        result = result["preannotation"]
        monkeypatch.setattr("iris.benchmark_runs.run_benchmark_trial", lambda *args: result)
    monkeypatch.setattr(worker, "run_comparison", lambda *args: result)
    monkeypatch.setattr(worker.signal, "signal", lambda *args: None)
    monkeypatch.setattr(
        worker.ctypes, "CDLL", lambda *args, **kwargs: SimpleNamespace(prctl=lambda *args: 0)
    )
    monkeypatch.setattr(worker.os, "getppid", lambda: 123)
    worker.run(store.root, identifier, 123)
    job = store.get("jobs", identifier)
    assert job["status"] == expected
    assert job["result"] == result
    if not cancelled:
        assert f"{ready} images ready for review" in job["message"]
        assert f"{issues} need attention" in job["message"]
    assert bool(job["error"]) == (expected == "failed")
