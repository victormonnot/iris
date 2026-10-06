"""Portable saved-proposal imports retain their evidence without a provider call."""

import hashlib
import socket
import zipfile
from copy import deepcopy
from datetime import UTC, datetime, timedelta

import pytest
import test_workspace_restore as restore_fixtures
from test_benchmark_api import freeze
from test_benchmark_api import workspace as workspace
from test_benchmarks_archive import records
from test_dinox_review_provider import decisions, response

from iris import benchmark_corrections as corrections
from iris import dinox_provider as dinox
from iris import dinox_review_provider as review
from iris import multimodal_provider as openai
from iris.benchmark import _digest, open_benchmark_image
from iris.benchmark_recorded import BUNDLE_PROTOCOL
from iris.benchmark_runs import run_benchmark_trial
from iris.jobs import JobManager
from iris.store import SCHEMA_VERSION, Store, new_id, now
from iris.workspace_archive import ArchiveError, create_archive
from iris.workspace_restore import inspect_archive, restore_archive


def prepared(workspace, transform="identity"):
    client, store, _ = workspace
    reference = freeze(workspace)
    manifest = reference["manifest"]
    settings = {"dinox_config": dinox.freeze_config(manifest["taxonomy"]), "transform": transform}
    if transform == "threshold":
        settings["threshold"] = 0.5
    elif transform == "review":
        settings["review_config"] = review.freeze_config(manifest["taxonomy"])
    model_id = dinox.MODEL + ("+gpt-6-astra" if transform == "review" else "")
    values = {"approach": "recorded_proposals", "model_id": model_id, "recorded": settings}
    path = f"/api/benchmarks/{reference['id']}/configs"
    preview = client.post(path + "/preview", json=values)
    assert preview.status_code == 200, preview.text
    candidate = client.post(
        path,
        json={
            **values,
            "name": "Saved fixture",
            "expected_fingerprint": preview.json()["fingerprint"],
        },
    )
    assert candidate.status_code == 201, candidate.text
    candidate = candidate.json()
    entries = []
    for frame in manifest["frames"]:
        if frame["role"] != "tuning":
            continue
        raw = {"objects": [{"category": "person", "bbox": [2, 3, 30, 35], "score": 0.85}]}
        entry = {
            "frame_id": frame["frame_id"],
            "image_file_sha256": frame["image_file_sha256"],
            "source_pixel_sha256": frame["sha256"],
            "width": frame["width"],
            "height": frame["height"],
            "dinox": {
                "raw_result": raw,
                "receipt": {
                    "task_id": "synthetic-task-" + frame["frame_id"],
                    "status": "succeeded",
                    "recorded_at": now(),
                    "elapsed_ms": None,
                    "estimated_cost_cny": None,
                },
            },
        }
        if transform == "review":
            native = dinox.normalize(raw, settings["dinox_config"], frame["width"], frame["height"])
            with open_benchmark_image(store, frame) as image:
                request = review.prepare_request(image, settings["review_config"], native)
            saved_input = review.safe_request(request)
            raw_review = response({"decisions": decisions(native)})
            raw_review["id"] = "resp-synthetic-" + frame["frame_id"]
            entry["review"] = {
                "input": saved_input,
                "raw_response": raw_review,
                "receipt": {
                    "response_id": raw_review["id"],
                    "status": "completed",
                    "recorded_at": now(),
                    "elapsed_ms": 2500.0,
                    "usage_cost_usd": openai._usage_metadata(
                        raw_review, settings["review_config"]["openai_config"]
                    )["usage_cost_usd"],
                    "request_sha256": saved_input["request_sha256"],
                },
            }
        entries.append(entry)
    bundle = {"protocol": BUNDLE_PROTOCOL, "frames": entries}
    path = f"/api/benchmarks/{reference['id']}/recorded-trials"
    values = {"config_id": candidate["id"], "role": "tuning", "bundle": bundle}
    preview = client.post(path + "/preview", json=values)
    assert preview.status_code == 200, preview.text
    values["expected_fingerprint"] = preview.json()["fingerprint"]
    created = client.post(path, json=values)
    assert created.status_code == 202, created.text
    trial = store.get("benchmark_trials", created.json()["id"])
    repeated = client.post(path, json=values)
    assert repeated.status_code == 202 and repeated.json()["id"] == trial["id"]
    assert len(store.list("benchmark_trials")) == len(store.list("jobs")) == 1
    return store, reference, candidate, trial


def complete(preparation, *, worker=False):
    store, _, _, trial = preparation
    if worker:
        store.update("jobs", trial["job_id"], {"status": "running", "started_at": now()})
        JobManager(store)._execute(store.get("jobs", trial["job_id"]))
        job = store.get("jobs", trial["job_id"])
        assert job["status"] == "succeeded", job
    else:
        result = run_benchmark_trial(store, trial["id"], lambda *_: None, lambda: False)
        store.update("jobs", trial["job_id"], {"status": "succeeded", "result": result})
    return store.list("benchmark_outputs", trial_id=trial["id"])[0]


def forbid_external(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("Saved evidence recovery must not consult credentials or contact a provider")

    for module, names in (
        (dinox, ("_credential", "_disk_credentials", "provider_status", "submit", "poll")),
        (openai, ("_api_key", "provider_status", "OpenAIPreannotator", "_request")),
        (review, ("DinoXReviewer",)),
        (socket, ("socket", "create_connection")),
    ):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    monkeypatch.setattr(openai.http.client, "HTTPSConnection", forbidden)


@pytest.mark.parametrize("transform", ["identity", "threshold", "review"])
def test_recorded_round_trip_retains_sources_and_real_correction_intervals_offline(
    workspace, tmp_path, monkeypatch, transform
):
    preparation = prepared(workspace, transform)
    store, _, _, trial = preparation
    reference_revisions = store.list("annotation_revisions")
    output = complete(preparation, worker=transform == "review")
    clock = [10.0]
    monkeypatch.setattr(
        corrections,
        "_clock",
        lambda: (datetime(2026, 10, 7, tzinfo=UTC) + timedelta(seconds=clock[0]), clock[0]),
    )
    token = "fixture-review-owner"
    timer = corrections.timer_action(
        store,
        output["id"],
        action="start",
        expected_revision=0,
        token=token,
        operation_id=new_id(),
        reviewer="Synthetic reviewer",
    )
    clock[0] += 2.5
    correction = corrections.save_correction(
        store,
        output["id"],
        expected_revision=0,
        boxes=[],
        status="reviewed",
        reviewer="Synthetic reviewer",
        timer_revision=timer["revision"],
        timer_token=token,
    )
    assert correction["timing"]["elapsed_ms"] == 2500
    assert correction["timing"]["fully_timed"] is True
    assert store.get("benchmark_outputs", output["id"]) == output
    assert store.list("annotation_revisions") == reference_revisions
    before = records(store)
    forbid_external(monkeypatch)
    archive = create_archive(store.root, tmp_path / "recorded.zip")
    assert archive["manifest"]["schema_version"] == SCHEMA_VERSION
    restored = tmp_path / "restored"
    with monkeypatch.context() as context:
        context.setattr(
            Store, "__init__", lambda *_a, **_k: pytest.fail("Offline restore opened Store")
        )
        checked = inspect_archive(archive["path"])
        restore_archive(
            archive["path"], restored, expected_archive_sha256=checked["archive_sha256"]
        )
    with zipfile.ZipFile(archive["path"]) as saved:
        for item in archive["manifest"]["files"]:
            assert (restored / item["path"]).read_bytes() == saved.read(item["path"])
    reopened = Store(restored)
    assert records(reopened) == before
    assert reopened.get("benchmark_outputs", output["id"]) == output
    assert reopened.get("benchmark_trials", trial["id"]) == trial
    assert output["metadata"]["source"]["dinox"]["elapsed_ms"] is None
    assert output["metadata"]["source"]["dinox"]["estimated_cost_cny"] is None


@pytest.mark.parametrize(
    "damage",
    [
        "config",
        "bundle_hash",
        "bundle_pixel_hash",
        "bundle_file_hash",
        "bundle_frame",
        "raw",
        "result",
        "source_receipt",
        "frame_evidence_hash",
        "output_bundle_hash",
        "attempt",
        "missing_attempt",
        "result_owner",
        "operation",
        "coverage",
        "state",
        "partition",
    ],
)
def test_recorded_archive_rejects_inconsistent_sources_and_ownership(workspace, tmp_path, damage):
    preparation = prepared(workspace)
    store, reference, candidate, trial = preparation
    output = complete(preparation)
    if damage == "config":
        config = deepcopy(candidate["config"])
        config["recorded"]["dinox_config"]["model"] = "another-model"
        store.update("benchmark_configs", candidate["id"], {"config": config})
    elif damage.startswith("bundle_"):
        frozen = deepcopy(trial["config"])
        if damage == "bundle_hash":
            frozen["recorded_bundle_sha256"] = "a" * 64
        else:
            field = {
                "bundle_pixel_hash": "source_pixel_sha256",
                "bundle_file_hash": "image_file_sha256",
                "bundle_frame": "frame_id",
            }[damage]
            frozen["recorded_bundle"]["frames"][0][field] = "a" * 64
            frozen["recorded_bundle_sha256"] = _digest(frozen["recorded_bundle"])
        store.update("benchmark_trials", trial["id"], {"config": frozen})
    elif damage == "raw":
        raw = deepcopy(output["raw_response"])
        raw["dinox"]["raw_result"]["objects"][0]["bbox"][0] += 1
        store.update("benchmark_outputs", output["id"], {"raw_response": raw})
    elif damage == "result":
        result = deepcopy(output["result"])
        result["proposals"][0]["box"][0] += 1
        store.update("benchmark_outputs", output["id"], {"result": result})
    elif damage in {
        "source_receipt",
        "frame_evidence_hash",
        "output_bundle_hash",
        "attempt",
        "state",
    }:
        metadata = deepcopy(output["metadata"])
        if damage == "source_receipt":
            metadata["source"]["dinox"]["task_id"] = "different-task"
        elif damage == "state":
            metadata["state"] = "not_started"
        else:
            field = {
                "frame_evidence_hash": "frame_evidence_sha256",
                "output_bundle_hash": "bundle_sha256",
                "attempt": "attempt_id",
            }[damage]
            metadata["recorded"][field] = "a" * 64
        store.update("benchmark_outputs", output["id"], {"metadata": metadata})
    elif damage == "missing_attempt":
        store.update("jobs", trial["job_id"], {"result": {}})
    elif damage == "result_owner":
        result = store.get("jobs", trial["job_id"])["result"]
        result["trial_id"] = "another-import"
        store.update("jobs", trial["job_id"], {"result": result})
    elif damage == "operation":
        store.update("jobs", trial["job_id"], {"params": {"trial_id": trial["id"]}})
    elif damage == "coverage":
        with store.connect() as connection:
            connection.execute("DELETE FROM benchmark_outputs WHERE id=?", (output["id"],))
    else:
        other = next(
            frame for frame in reference["manifest"]["frames"] if frame["role"] == "evaluation"
        )
        store.update("benchmark_outputs", output["id"], {"frame_id": other["frame_id"]})
    with pytest.raises(ArchiveError):
        create_archive(store.root, tmp_path / "rejected.zip")
    assert not (tmp_path / "rejected.zip").exists()


def test_rehashed_archive_still_refuses_modified_recorded_canonical_output(workspace, tmp_path):
    preparation = prepared(workspace, "review")
    store = preparation[0]
    output = complete(preparation)
    result = deepcopy(output["result"])
    result["proposals"][0]["label"] = "car"
    store.update("benchmark_outputs", output["id"], {"result": result})
    archive = restore_fixtures._write_archive(
        tmp_path / "rehashed.zip", restore_fixtures._payload(store)
    )
    with pytest.raises(ArchiveError, match="Recorded"):
        inspect_archive(archive)
    destination = tmp_path / "not-restored"
    with pytest.raises(ArchiveError, match="Recorded"):
        restore_archive(
            archive,
            destination,
            expected_archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        )
    assert not destination.exists()


def test_interrupted_recorded_import_keeps_frozen_bundle_without_claiming_success(
    workspace, tmp_path
):
    store, _, _, trial = prepared(workspace)
    store.update("jobs", trial["job_id"], {"status": "interrupted"})
    expected = records(store)
    archive = create_archive(store.root, tmp_path / "interrupted.zip")
    checked = inspect_archive(archive["path"])
    restored = tmp_path / "restored"
    restore_archive(archive["path"], restored, expected_archive_sha256=checked["archive_sha256"])
    assert records(Store(restored)) == expected
    assert not Store(restored).list("benchmark_outputs")
