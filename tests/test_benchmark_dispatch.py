"""Durable paid-request accounting using SQLite only, with no HTTP or provider key."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import pytest
from PIL import Image

from iris import benchmark_dispatch as dispatch
from iris.job_dispatch import DispatchConflict
from iris.media import import_asset
from iris.store import Store, _encode, new_id, now


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@pytest.fixture
def external(tmp_path):
    store = Store(tmp_path / "workspace")
    frames = []
    session = store.insert(
        "sessions",
        {
            "id": new_id(),
            "name": "Generated fixture",
            "scene_group": "fixture",
            "created_at": now(),
        },
    )
    for index in range(2):
        path = tmp_path / f"{index}.png"
        Image.new("RGB", (40, 30), (index * 100, 20, 30)).save(path)
        asset = import_asset(store, session["id"], path, path.name)
        frame = store.list("frames", asset_id=asset["id"])[0]
        frames.append({"frame_id": frame["id"], "width": 40, "height": 30})
    benchmark = store.insert(
        "benchmarks",
        {
            "id": new_id(),
            "project_id": "default",
            "name": "Private fixture",
            "path": "fixture",
            "manifest_sha256": "a" * 64,
            "summary": {},
            "created_at": now(),
        },
    )
    candidate = {
        "approach": "multimodal",
        "provider_config": {"provider": "openai", "model": "gpt-6-astra"},
    }
    config = store.insert(
        "benchmark_configs",
        {
            "id": new_id(),
            "benchmark_id": benchmark["id"],
            "name": "Fixture only",
            "approach": "multimodal",
            "config": candidate,
            "fingerprint": sha(candidate),
            "created_at": now(),
        },
    )
    requests = [
        {
            "frame_id": frame["frame_id"],
            "input": {
                "image": {
                    "sha256": str(index) * 64,
                    "width": 40,
                    "height": 30,
                    "sent_width": 40,
                    "sent_height": 30,
                    "bytes": 100,
                    "encoding": "png",
                    "transform": {"scale": [1, 1], "offset": [0, 0]},
                },
                "prompt": "Fixture class descriptions only; no reference annotation.",
                "request_sha256": str(index + 2) * 64,
            },
            "estimate": {
                "currency": "USD",
                "upper_bound_usd": 0.0000011,
                "guaranteed_billing_cap": False,
                "basis": "Synthetic allowance",
            },
        }
        for index, frame in enumerate(frames)
    ]
    plan = {
        "protocol": dispatch.PLAN_PROTOCOL,
        "provider": "openai",
        "model": "gpt-6-astra",
        "requests": requests,
        "estimate": {
            "currency": "USD",
            "upper_bound_usd": 0.000004,
            "estimated_ceiling_microusd": 4,
        },
        "approval": {
            "allow_external": True,
            "fingerprint": "e" * 64,
            "approved_at": now(),
            "budget_usd": 0.000004,
            "budget_microusd": 4,
            "estimated_ceiling_microusd": 4,
        },
    }
    trial_id, job_id = new_id(), new_id()
    trial = {
        "id": trial_id,
        "benchmark_id": benchmark["id"],
        "config_id": config["id"],
        "split": "tuning",
        "job_id": job_id,
        "created_at": now(),
        "config": {
            "fingerprint": "e" * 64,
            "candidate_config": candidate,
            "external_plan": plan,
            "source_config_fingerprint": config["fingerprint"],
            "frame_ids": [frame["frame_id"] for frame in frames],
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
        identifiers = dispatch.initialize_outputs(conn, trial, frames, plan)
    return store, trial, identifiers, frames


def rows(external):
    store, _, identifiers, _ = external
    return [store.get("benchmark_outputs", identifier) for identifier in identifiers]


def test_all_receipts_are_known_unsent_and_microdollar_rounding_is_per_image(external):
    store, trial, _, _ = external
    summary = dispatch.dispatch_summary(store, trial["id"])
    assert summary["counts"]["not_started"] == 2
    assert (
        summary["estimated_ceiling_microusd"] == 4
    )  # Two separately rounded 1.1-micro allowances.
    assert summary["reserved_microusd"] == 0
    assert summary["usage_cost_usd"] is None and summary["known_usage_cost_usd"] is None
    for row in rows(external):
        assert row["raw_response"] is row["result"] is row["error"] is None
        assert row["metadata"]["budget"]["ceiling_microusd"] == 2
        assert dispatch.validate_output_row(row, trial)["state"] == "not_started"


def test_trial_claim_and_dispatch_are_atomic_single_attempts(external):
    store, trial, identifiers, _ = external

    def claim():
        try:
            return dispatch.claim_trial(store, trial["id"])
        except DispatchConflict:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        attempts = list(pool.map(lambda _: claim(), range(2)))
    assert sum(attempt is not None for attempt in attempts) == 1
    attempt = next(attempt for attempt in attempts if attempt)

    def begin():
        try:
            return dispatch.begin_dispatch(store, identifiers[0], attempt)
        except DispatchConflict:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: begin(), range(2)))
    assert sum(result is not None for result in results) == 1
    # This separate read stands at the transport boundary, after the durable commit.
    saved = store.get("benchmark_outputs", identifiers[0])
    assert saved["metadata"]["dispatch"]["state"] == "dispatching"
    assert saved["metadata"]["budget"]["reserved_microusd"] == 2
    dispatch.begin_dispatch(store, identifiers[1], attempt)
    assert dispatch.dispatch_summary(store, trial["id"])["reserved_microusd"] == 4


@pytest.mark.parametrize("before_claim", [True, False])
def test_cancellation_before_transport_remains_unsent(external, before_claim):
    store, trial, identifiers, _ = external
    attempt = None if before_claim else dispatch.claim_trial(store, trial["id"])
    store.update("jobs", trial["job_id"], {"status": "cancelled", "cancel_requested": True})
    with pytest.raises(DispatchConflict):
        if before_claim:
            dispatch.claim_trial(store, trial["id"])
        else:
            dispatch.begin_dispatch(store, identifiers[0], attempt)
    dispatch.recover_benchmark_dispatches(store)
    assert all(row["metadata"]["dispatch"]["state"] == "not_started" for row in rows(external))
    assert dispatch.dispatch_summary(store, trial["id"])["reserved_microusd"] == 0


def test_preflight_failure_does_not_claim_a_send_or_allow_later_transport(external):
    store, trial, identifiers, _ = external
    attempt = dispatch.claim_trial(store, trial["id"])
    failed = dispatch.fail_output(store, identifiers[0], attempt, RuntimeError())
    assert failed["error"] == "RuntimeError"
    assert failed["metadata"]["dispatch"]["state"] == "not_started"
    assert failed["metadata"]["budget"]["reserved_microusd"] == 0
    with pytest.raises(DispatchConflict):
        dispatch.begin_dispatch(store, identifiers[0], attempt)


def test_unknown_outcome_keeps_reservation_and_late_response_after_stop(external):
    store, trial, identifiers, _ = external
    attempt = dispatch.claim_trial(store, trial["id"])
    dispatch.begin_dispatch(store, identifiers[0], attempt)
    failed = dispatch.fail_output(
        store,
        identifiers[0],
        attempt,
        "Socket lost",
        raw={"http_status": 200, "body": "partial", "truncated": True},
    )
    assert failed["metadata"]["dispatch"]["state"] == "outcome_unknown"
    assert failed["metadata"]["budget"]["reserved_microusd"] == 2
    with pytest.raises(DispatchConflict):
        dispatch.begin_dispatch(store, identifiers[0], attempt)
    store.update("jobs", trial["job_id"], {"status": "interrupted"})
    received = dispatch.save_response(
        store,
        identifiers[0],
        attempt,
        {"id": "response-1"},
        {"request_id": "response-1", "usage": {"input_tokens": 12}},
    )
    assert received["metadata"]["dispatch"]["state"] == "response_received"
    published = dispatch.publish_output(store, identifiers[0], attempt, {"proposals": []}, {})
    assert published["metadata"]["state"] == "cancelled"
    assert published["result"] is None and published["raw_response"] == {"id": "response-1"}
    assert published["metadata"]["budget"]["reserved_microusd"] == 2


def test_complete_invalid_response_is_received_with_raw_evidence_and_known_usage(external):
    store, trial, identifiers, _ = external
    attempt = dispatch.claim_trial(store, trial["id"])
    dispatch.begin_dispatch(store, identifiers[0], attempt)
    raw = {"http_status": 200, "body": "null", "truncated": False}
    output = dispatch.fail_output(
        store,
        identifiers[0],
        attempt,
        "Invalid provider JSON",
        raw=raw,
        response_received=True,
        metadata={
            "request_id": "request-x",
            "usage": {"input_tokens": 100},
            "usage_cost_usd": 0.000001,
        },
    )
    assert output["raw_response"] == raw
    assert output["metadata"]["dispatch"]["state"] == "response_received"
    assert output["result"] is None
    summary = dispatch.dispatch_summary(store, trial["id"])
    assert summary["usage_cost_usd"] == 0.000001
    assert summary["outputs"][0]["request_id"] == "request-x"


@pytest.mark.parametrize("complete", [False, True])
def test_bounded_http_body_retains_raw_evidence_after_json_escaping(external, complete):
    store, trial, identifiers, _ = external
    attempt = dispatch.claim_trial(store, trial["id"])
    dispatch.begin_dispatch(store, identifiers[0], attempt)
    # A permitted 2 MiB HTTP body expands to 12 MiB in its saved JSON envelope.
    raw = {"http_status": 200, "body": "\0" * (2 * 1024 * 1024), "truncated": not complete}
    if complete:
        dispatch.save_response(store, identifiers[0], attempt, raw, {})
    output = dispatch.fail_output(
        store,
        identifiers[0],
        attempt,
        "Invalid provider JSON" if complete else "Incomplete HTTP body",
        raw=raw,
        response_received=complete,
    )
    assert output["raw_response"] == raw
    assert store.get("benchmark_outputs", identifiers[0])["raw_response"] == raw
    assert output["metadata"]["dispatch"]["state"] == (
        "response_received" if complete else "outcome_unknown"
    )
    assert output["metadata"]["budget"]["reserved_microusd"] == 2


def test_partial_unknown_cost_is_not_zero_or_the_known_usage_subtotal(external):
    store, trial, identifiers, _ = external
    attempt = dispatch.claim_trial(store, trial["id"])
    for identifier in identifiers:
        dispatch.begin_dispatch(store, identifier, attempt)
    dispatch.save_response(
        store, identifiers[0], attempt, {"id": "r1"}, {"usage_cost_usd": 0.000001}
    )
    dispatch.publish_output(store, identifiers[0], attempt, {"proposals": []}, {})
    dispatch.fail_output(store, identifiers[1], attempt, "Response lost")
    summary = dispatch.dispatch_summary(store, trial["id"])
    assert summary["state"] == "outcome_unknown" and summary["unknown_outcome_count"] == 1
    assert summary["known_usage_cost_usd"] == 0.000001 and summary["usage_cost_usd"] is None
    assert summary["usage_missing_count"] == 1
    assert summary["reserved_microusd"] == 4 and summary["remaining_planned_microusd"] == 0


def test_response_and_published_output_are_immutable_and_owned(external):
    store, trial, identifiers, _ = external
    attempt = dispatch.claim_trial(store, trial["id"])
    identifier = identifiers[0]
    with pytest.raises(DispatchConflict):
        dispatch.begin_dispatch(store, identifier, "wrong-attempt")
    dispatch.begin_dispatch(store, identifier, attempt)
    with pytest.raises(DispatchConflict):
        dispatch.publish_output(store, identifier, attempt, {"proposals": []}, {})
    with pytest.raises(ValueError):
        dispatch.save_response(store, identifier, attempt, None, {})
    dispatch.save_response(store, identifier, attempt, {"id": "r1"}, {})
    with pytest.raises(DispatchConflict):
        dispatch.save_response(store, identifier, attempt, {"id": "replaced"}, {})
    with pytest.raises(DispatchConflict):
        dispatch.fail_output(store, identifier, "another-attempt", "Bad claim")
    ready = dispatch.publish_output(store, identifier, attempt, {"proposals": []}, {})
    assert dispatch.publish_output(store, identifier, attempt, {"proposals": []}, {}) == ready
    with pytest.raises(DispatchConflict):
        dispatch.fail_output(store, identifier, attempt, "Too late")
    assert store.get("benchmark_outputs", identifier) == ready


def test_provider_metadata_cannot_replace_dispatch_budget_or_request(external):
    store, trial, identifiers, _ = external
    attempt = dispatch.claim_trial(store, trial["id"])
    saved = dispatch.begin_dispatch(store, identifiers[0], attempt)
    for field in ("dispatch", "budget", "input", "state"):
        with pytest.raises(ValueError):
            dispatch.save_response(store, identifiers[0], attempt, {"id": "r1"}, {field: {}})
        assert store.get("benchmark_outputs", identifiers[0]) == saved


def test_interruption_preserves_all_receipts_and_never_claims_a_new_attempt(external):
    store, trial, identifiers, _ = external
    attempt = dispatch.claim_trial(store, trial["id"])
    dispatch.begin_dispatch(store, identifiers[0], attempt)
    store.update("jobs", trial["job_id"], {"status": "interrupted"})
    assert dispatch.dispatch_summary(store, trial["id"])["state"] == "outcome_unknown"
    dispatch.recover_benchmark_dispatches(store)
    saved = rows(external)
    assert saved[0]["metadata"]["dispatch"]["state"] == "outcome_unknown"
    assert saved[1]["metadata"]["dispatch"]["state"] == "not_started"
    dispatch.recover_benchmark_dispatches(store)
    assert rows(external) == saved
    store.update("jobs", trial["job_id"], {"status": "running"})
    with pytest.raises(DispatchConflict):
        dispatch.claim_trial(store, trial["id"])


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p["approval"].update(budget_microusd=True),
        lambda p: p["approval"].update(allow_external=False),
        lambda p: p["approval"].update(fingerprint="a" * 64),
        lambda p: p["estimate"].update(estimated_ceiling_microusd=3),
        lambda p: p["approval"].update(estimated_ceiling_microusd=3),
        lambda p: p["approval"].update(budget_usd=0.000003, budget_microusd=3),
        lambda p: p["requests"][0]["estimate"].update(upper_bound_usd=float("nan")),
        lambda p: p["requests"][0]["estimate"].update(upper_bound_usd=True),
        lambda p: p["requests"][0]["input"].update(payload={"reference": "must not be stored"}),
        lambda p: p["requests"][0]["input"]["image"].update(sha256="invalid"),
        lambda p: p["requests"][0]["input"]["image"].update(width=True),
        lambda p: p["requests"][0].update(frame_id="foreign"),
    ],
)
def test_invalid_budget_or_request_plan_is_rejected_without_mutation(external, mutation):
    store, trial, _, frames = external
    original = rows(external)
    frozen = deepcopy(trial["config"])
    mutation(frozen["external_plan"])
    with pytest.raises(ValueError):
        dispatch.validate_external_trial(frozen, frozen["candidate_config"], frames)
    assert rows(external) == original


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row["metadata"]["budget"].update(reserved_microusd=True),
        lambda row: row["metadata"]["budget"].update(ceiling_microusd=0),
        lambda row: row["metadata"]["dispatch"].update(state="response_received"),
        lambda row: row["metadata"]["input"].update(request_sha256="f" * 64),
    ],
)
def test_corrupt_receipt_cannot_reach_transport(external, mutation):
    store, trial, identifiers, _ = external
    attempt = dispatch.claim_trial(store, trial["id"])
    row = rows(external)[0]
    mutation(row)
    store.update("benchmark_outputs", identifiers[0], {"metadata": row["metadata"]})
    with pytest.raises(ValueError):
        dispatch.begin_dispatch(store, identifiers[0], attempt)


def test_failed_initialization_rolls_back_all_placeholders(external):
    store, trial, _, frames = external
    before = rows(external)
    with pytest.raises(RuntimeError, match="rollback fixture"):
        with store.connect() as conn:
            conn.execute("DELETE FROM benchmark_outputs WHERE trial_id=?", (trial["id"],))
            dispatch.initialize_outputs(conn, trial, frames, trial["config"]["external_plan"])
            raise RuntimeError("rollback fixture")
    assert rows(external) == before
