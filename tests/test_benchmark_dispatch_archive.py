"""Schema-15 archives preserve external request journals without sending anything."""

import json
import zipfile
from copy import deepcopy
from decimal import ROUND_CEILING, Decimal

import pytest
import test_benchmarks_archive as archive_fixtures
from test_benchmarks_archive import canonical, digest, records

from iris import benchmark_dispatch as dispatch
from iris import multimodal_provider as provider
from iris.benchmark import PROTOCOL, open_benchmark_image
from iris.benchmark_multimodal import MULTIMODAL_SCORING, prepared_input
from iris.store import Store, _encode, new_id, now
from iris.workspace_archive import ArchiveError, create_archive
from iris.workspace_restore import inspect_archive, restore_archive

benchmark_workspace = archive_fixtures.benchmark_workspace


@pytest.fixture
def external_archive(benchmark_workspace):
    store, benchmark, manifest, *_ = benchmark_workspace
    frame = next(frame for frame in manifest["frames"] if frame["role"] == "evaluation")
    frozen_provider = provider.freeze_config(manifest["taxonomy"])
    candidate = {
        "protocol": PROTOCOL,
        "approach": "multimodal",
        "model_id": frozen_provider["model"],
        "model_name": "Offline archival fixture",
        "provider_config": frozen_provider,
        "reference_manifest_sha256": benchmark["manifest_sha256"],
        "scoring": MULTIMODAL_SCORING,
    }
    config = store.insert(
        "benchmark_configs",
        {
            "id": new_id(),
            "benchmark_id": benchmark["id"],
            "name": "Frozen external fixture",
            "approach": "multimodal",
            "config": candidate,
            "fingerprint": digest(canonical(candidate)),
            "created_at": now(),
        },
    )
    with open_benchmark_image(store, frame) as image:
        prepared = provider.prepare_request(image, frozen_provider)
    ceiling = int(
        (Decimal(str(prepared["estimate"]["upper_bound_usd"])) * 1_000_000).to_integral_value(
            rounding=ROUND_CEILING
        )
    )
    plan = {
        "protocol": dispatch.PLAN_PROTOCOL,
        "provider": "openai",
        "model": frozen_provider["model"],
        "requests": [
            {
                "frame_id": frame["frame_id"],
                "input": prepared_input(prepared),
                "estimate": prepared["estimate"],
            }
        ],
        "estimate": {
            "currency": "USD",
            "upper_bound_usd": ceiling / 1_000_000,
            "estimated_ceiling_microusd": ceiling,
        },
        "approval": {
            "approved_at": now(),
            "allow_external": True,
            "fingerprint": "d" * 64,
            "budget_usd": ceiling / 1_000_000,
            "budget_microusd": ceiling,
            "estimated_ceiling_microusd": ceiling,
        },
    }
    trial_id, job_id = new_id(), new_id()
    trial = {
        "id": trial_id,
        "benchmark_id": benchmark["id"],
        "config_id": config["id"],
        "split": "evaluation",
        "job_id": job_id,
        "created_at": now(),
        "config": {
            "protocol": PROTOCOL,
            "fingerprint": "d" * 64,
            "source_config_fingerprint": config["fingerprint"],
            "benchmark_manifest_sha256": benchmark["manifest_sha256"],
            "frame_ids": [frame["frame_id"]],
            "role": "evaluation",
            "candidate_config": candidate,
            "work": {"image_count": 1, "request_count": 1},
            "warnings": [],
            "external_plan": plan,
        },
    }
    with store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "INSERT INTO jobs(id,kind,status,params,created_at) VALUES(?,?,?,?,?)",
            (job_id, "benchmark", "queued", json.dumps({"trial_id": trial_id}), now()),
        )
        encoded = _encode(trial)
        conn.execute(
            f"INSERT INTO benchmark_trials({','.join(encoded)}) "
            f"VALUES({','.join('?' for _ in encoded)})",
            tuple(encoded.values()),
        )
        output_id = dispatch.initialize_outputs(conn, trial, [frame], plan)[0]
    return store, trial, output_id


@pytest.mark.parametrize(
    "state", ["not_started", "dispatching", "response_received", "outcome_unknown"]
)
def test_external_journal_round_trip_is_offline_and_preserves_every_byte(
    external_archive, tmp_path, monkeypatch, state
):
    store, trial, output_id = external_archive
    if state != "not_started":
        attempt = dispatch.claim_trial(store, trial["id"])
        dispatch.begin_dispatch(store, output_id, attempt)
        if state == "response_received":
            dispatch.save_response(
                store,
                output_id,
                attempt,
                {"id": "response-fixture", "output": [], "usage": {"input_tokens": 123}},
                {
                    "request_id": "response-fixture",
                    "http_request_id": "http-fixture",
                    "usage_cost_usd": 0.001,
                },
            )
            dispatch.publish_output(store, output_id, attempt, {"proposals": []}, {})
        elif state == "outcome_unknown":
            dispatch.fail_output(
                store,
                output_id,
                attempt,
                "Lost response after dispatch",
                raw={"http_status": 200, "body": "incomplete bytes", "truncated": True},
            )
    store.update("jobs", trial["job_id"], {"status": "interrupted"})
    expected = records(store)
    saved_output = store.get("benchmark_outputs", output_id)

    def forbidden(*_args, **_kwargs):
        pytest.fail("Archives must not inspect credentials, contact a provider or open Store")

    # Pure frozen configuration validation remains available even with no credential/runtime path.
    monkeypatch.setattr(provider, "_api_key", forbidden)
    monkeypatch.setattr(provider, "provider_status", forbidden)
    monkeypatch.setattr(provider, "OpenAIPreannotator", forbidden)
    monkeypatch.setattr(provider.http.client, "HTTPSConnection", forbidden)
    archive = create_archive(store.root, tmp_path / "external.zip")
    assert archive["manifest"]["schema_version"] == 15
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
    assert reopened.get("benchmark_outputs", output_id) == saved_output
    assert saved_output["metadata"]["dispatch"]["state"] == state
    # Restoration never reconciles an in-flight receipt. Runtime recovery is explicit.
    dispatch.recover_benchmark_dispatches(reopened)
    recovered = reopened.get("benchmark_outputs", output_id)
    assert recovered["metadata"]["dispatch"]["state"] == (
        "outcome_unknown" if state == "dispatching" else state
    )
    assert recovered["metadata"]["budget"] == saved_output["metadata"]["budget"]
    assert recovered["raw_response"] == saved_output["raw_response"]


@pytest.mark.parametrize(
    "damage",
    [
        "consent",
        "consent_fingerprint",
        "budget",
        "reserved",
        "input",
        "request_coverage",
        "attempt_owner",
        "missing_response",
        "source_config",
    ],
)
def test_archive_rejects_inconsistent_external_consent_and_dispatch_provenance(
    external_archive, tmp_path, damage
):
    store, trial, output_id = external_archive
    attempt = dispatch.claim_trial(store, trial["id"])
    dispatch.begin_dispatch(store, output_id, attempt)
    dispatch.save_response(store, output_id, attempt, {"id": "saved-response"}, {})
    store.update("jobs", trial["job_id"], {"status": "interrupted"})
    config = deepcopy(trial["config"])
    output = store.get("benchmark_outputs", output_id)
    if damage == "consent":
        config["external_plan"]["approval"]["allow_external"] = False
    elif damage == "consent_fingerprint":
        config["external_plan"]["approval"]["fingerprint"] = "f" * 64
    elif damage == "budget":
        config["external_plan"]["approval"]["budget_microusd"] = 0
    elif damage == "reserved":
        output["metadata"]["budget"]["reserved_microusd"] = 0
    elif damage == "input":
        output["metadata"]["input"]["request_sha256"] = "f" * 64
    elif damage == "request_coverage":
        with store.connect() as conn:
            conn.execute("DELETE FROM benchmark_outputs WHERE id=?", (output_id,))
    elif damage == "attempt_owner":
        store.update(
            "jobs", trial["job_id"], {"result": {"benchmark_attempt_id": "foreign-attempt"}}
        )
    elif damage == "missing_response":
        store.update("benchmark_outputs", output_id, {"raw_response": None})
    else:
        config["candidate_config"]["provider_config"]["model"] = "another-model"
    if damage in {"consent", "consent_fingerprint", "budget", "source_config"}:
        store.update("benchmark_trials", trial["id"], {"config": config})
    elif damage in {"reserved", "input"}:
        store.update("benchmark_outputs", output_id, {"metadata": output["metadata"]})
    with pytest.raises(ArchiveError):
        create_archive(store.root, tmp_path / "rejected.zip")
    assert not (tmp_path / "rejected.zip").exists()
