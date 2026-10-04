"""Pure schema-15 archival verification of all three combined candidate stages."""

import zipfile
from copy import deepcopy

import pytest
import test_benchmarks_archive as archive_fixtures
from test_benchmark_combined_dispatch import advance, make_trial
from test_benchmarks_archive import records

from iris import benchmark_combined_dispatch as dispatch
from iris import combined_provider, multimodal_provider, sam_provider, sam_runtime
from iris.store import SCHEMA_VERSION, Store
from iris.workspace_archive import ArchiveError, create_archive
from iris.workspace_restore import inspect_archive, restore_archive

benchmark_workspace = archive_fixtures.benchmark_workspace


@pytest.fixture
def combined_archive(benchmark_workspace):
    store, benchmark, manifest, *_ = benchmark_workspace
    frames = [frame for frame in manifest["frames"] if frame["role"] == "evaluation"]
    case = make_trial(store, benchmark, frames, manifest["taxonomy"], role="evaluation")
    yield case
    for image in case["images"].values():
        image.close()


@pytest.mark.parametrize(
    "state", ["not_started", "planning_dispatching", "grounding_running", "review_unknown", "ready"]
)
def test_combined_archive_is_offline_byte_exact_and_never_resumes(
    combined_archive, tmp_path, monkeypatch, state
):
    case = combined_archive
    store, trial, output_id = case["store"], case["trial"], case["outputs"][0]
    if state != "not_started":
        attempt = dispatch.claim_trial(store, trial["id"])
        if state == "planning_dispatching":
            dispatch.begin_stage(store, output_id, attempt, "planning")
        elif state == "grounding_running":
            advance(case, attempt, through="planning")
            dispatch.begin_stage(store, output_id, attempt, "grounding")
        elif state == "review_unknown":
            advance(case, attempt, through="review_started")
            dispatch.fail_stage(
                store,
                output_id,
                attempt,
                "review",
                "Lost response",
                raw={"body": "partial", "truncated": True},
            )
        else:
            advance(case, attempt, through="publish")
        dispatch.record_image_timing(store, output_id, attempt, 120)
    store.update("jobs", trial["job_id"], {"status": "interrupted"})
    expected = records(store)
    saved_output = store.get("benchmark_outputs", output_id)

    def forbidden(*args, **kwargs):
        pytest.fail("Archival validation must not probe credentials, runtimes or providers")

    monkeypatch.setattr(multimodal_provider, "_api_key", forbidden)
    monkeypatch.setattr(multimodal_provider.http.client, "HTTPSConnection", forbidden)
    monkeypatch.setattr(combined_provider, "CombinedOpenAI", forbidden)
    monkeypatch.setattr(sam_provider, "provider_status", forbidden)
    monkeypatch.setattr(sam_runtime, "runtime_status", forbidden)
    monkeypatch.setattr(sam_runtime, "SamRuntime", forbidden)
    archive = create_archive(store.root, tmp_path / "combined.zip")
    assert archive["manifest"]["schema_version"] == SCHEMA_VERSION
    restored = tmp_path / "restored"
    with monkeypatch.context() as context:
        context.setattr(Store, "__init__", forbidden)
        checked = inspect_archive(archive["path"])
        restore_archive(
            archive["path"], restored, expected_archive_sha256=checked["archive_sha256"]
        )
    with zipfile.ZipFile(archive["path"]) as saved:
        for item in archive["manifest"]["files"]:
            assert (restored / item["path"]).read_bytes() == saved.read(item["path"])
    reopened = Store(restored)
    assert records(reopened) == expected
    dispatch.recover_combined_dispatches(reopened)
    recovered = reopened.get("benchmark_outputs", output_id)
    assert recovered["raw_response"] == saved_output["raw_response"]
    assert recovered["result"] == saved_output["result"]
    stages = recovered["metadata"]["pipeline"]["stages"]
    assert stages["planning"]["state"] == (
        "outcome_unknown"
        if state == "planning_dispatching"
        else saved_output["metadata"]["pipeline"]["stages"]["planning"]["state"]
    )
    if state == "grounding_running":
        assert stages["grounding"]["state"] == "interrupted"
        assert stages["review"]["state"] == "not_started"


@pytest.mark.parametrize(
    "damage",
    [
        "consent",
        "request_coverage",
        "attempt_owner",
        "missing_raw",
        "stage_order",
        "reservation",
        "review_input",
        "planning_result",
        "grounding_runtime",
        "final_box",
        "runtime_plan",
        "source_pixels",
    ],
)
def test_archive_rejects_corrupt_combined_cross_stage_evidence(combined_archive, tmp_path, damage):
    case = combined_archive
    store, trial, output_id = case["store"], deepcopy(case["trial"]), case["outputs"][0]
    attempt = dispatch.claim_trial(store, trial["id"])
    advance(case, attempt, through="publish")
    store.update("jobs", trial["job_id"], {"status": "succeeded"})
    row = store.get("benchmark_outputs", output_id)
    stages = row["metadata"]["pipeline"]["stages"]
    if damage == "consent":
        trial["config"]["external_plan"]["approval"]["allow_external"] = False
    elif damage == "request_coverage":
        with store.connect() as conn:
            conn.execute("DELETE FROM benchmark_outputs WHERE id=?", (output_id,))
    elif damage == "attempt_owner":
        store.update("jobs", trial["job_id"], {"result": {"benchmark_attempt_id": "foreign"}})
    elif damage == "missing_raw":
        row["raw_response"]["planning"] = None
    elif damage == "stage_order":
        stages["planning"]["result"] = None
        stages["planning"]["completed_at"] = None
    elif damage == "reservation":
        stages["review"]["budget"]["reserved_microusd"] = 0
    elif damage == "review_input":
        stages["review"]["request"]["input_sha256"] = "f" * 64
    elif damage == "planning_result":
        stages["planning"]["result"][0]["text"] = "fabricated prompt"
    elif damage == "grounding_runtime":
        row["raw_response"]["grounding"]["metadata"]["runtime_identity"]["cuda"]["device"] = (
            "Foreign GPU"
        )
    elif damage == "final_box":
        row["result"]["proposals"][0]["box"] = [0, 0, 1, 1]
        stages["review"]["result"] = deepcopy(row["result"])
    elif damage == "runtime_plan":
        trial["config"]["external_plan"]["runtime_identity"]["code_revision"] = "f" * 40
    else:
        trial["config"]["external_plan"]["requests"][0]["planning"]["input"]["image"][
            "source_pixel_sha256"
        ] = "f" * 64
    if damage in {"consent", "runtime_plan", "source_pixels"}:
        store.update("benchmark_trials", trial["id"], {"config": trial["config"]})
    elif damage not in {"request_coverage", "attempt_owner"}:
        store.update(
            "benchmark_outputs",
            output_id,
            {key: row[key] for key in ("metadata", "raw_response", "result")},
        )
    target = tmp_path / "corrupt.zip"
    with pytest.raises(ArchiveError):
        create_archive(store.root, target)
    assert not target.exists()
