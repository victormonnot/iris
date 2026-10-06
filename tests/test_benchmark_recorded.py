"""Recorded provider evidence fixtures: offline import, never paid requests."""

from copy import deepcopy

import pytest
from test_benchmark_api import freeze
from test_benchmark_api import workspace as workspace
from test_dinox_review_provider import decisions, response

from iris import benchmark_recorded as recorded
from iris import benchmark_runs, dinox_provider, dinox_review_provider, multimodal_provider
from iris.benchmark import load_benchmark_manifest, open_benchmark_image
from iris.projects import create_project
from iris.store import now


def configured(workspace, transform="identity"):
    client, store, _ = workspace
    reference = freeze(workspace)
    manifest = load_benchmark_manifest(store, reference["id"])
    settings = {
        "dinox_config": dinox_provider.freeze_config(manifest["taxonomy"]),
        "transform": transform,
    }
    if transform == "threshold":
        settings["threshold"] = 0.5
    elif transform == "review":
        settings["review_config"] = dinox_review_provider.freeze_config(manifest["taxonomy"])
    values = {
        "approach": "recorded_proposals",
        "model_id": recorded._model(settings),
        "recorded": settings,
    }
    path = f"/api/benchmarks/{reference['id']}/configs"
    preview = client.post(path + "/preview", json=values)
    assert preview.status_code == 200, preview.text
    saved = client.post(
        path,
        json={
            **values,
            "name": "Synthetic recorded " + transform,
            "expected_fingerprint": preview.json()["fingerprint"],
        },
    )
    assert saved.status_code == 201, saved.text
    return reference, saved.json()


def bundle(store, reference, config, role="tuning", *, empty=False):
    manifest = load_benchmark_manifest(store, reference["id"])
    result = {"protocol": recorded.BUNDLE_PROTOCOL, "frames": []}
    settings = config["config"]["recorded"]
    for frame in [f for f in manifest["frames"] if f["role"] == role]:
        raw = {
            "objects": []
            if empty
            else [
                {"category": "person", "bbox": [2, 3, 30, 35], "score": 0.8},
                {"category": "person", "bbox": [40, 10, 50, 25], "score": 0.3},
            ]
        }
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
                    "elapsed_ms": 1234.0,
                    "estimated_cost_cny": 0.15,
                },
            },
        }
        if settings["transform"] == "review":
            native = dinox_provider.normalize(
                raw, settings["dinox_config"], frame["width"], frame["height"]
            )
            with open_benchmark_image(store, frame) as image:
                prepared = dinox_review_provider.prepare_request(
                    image, settings["review_config"], native
                )
            decision_list = decisions(native)
            if decision_list:
                decision_list[-1]["action"] = "reject"
            raw_review = response({"decisions": decision_list})
            raw_review["id"] = "synthetic-response-" + frame["frame_id"]
            entry["review"] = {
                "input": dinox_review_provider.safe_request(prepared),
                "raw_response": raw_review,
                "receipt": {
                    "response_id": raw_review["id"],
                    "status": "completed",
                    "recorded_at": now(),
                    "elapsed_ms": 2345.0,
                    "usage_cost_usd": multimodal_provider._usage_metadata(
                        raw_review, settings["review_config"]["openai_config"]
                    )["usage_cost_usd"],
                    "request_sha256": prepared["request_sha256"],
                },
            }
        result["frames"].append(entry)
    return result


def imported(workspace, transform="identity", *, empty=False):
    client, store, _ = workspace
    reference, config = configured(workspace, transform)
    source = bundle(store, reference, config, empty=empty)
    path = f"/api/benchmarks/{reference['id']}/recorded-trials"
    payload = {"config_id": config["id"], "role": "tuning", "bundle": source}
    preview = client.post(path + "/preview", json=payload)
    assert preview.status_code == 200, preview.text
    preview = preview.json()
    created = client.post(path, json={**payload, "expected_fingerprint": preview["fingerprint"]})
    assert created.status_code == 202, created.text
    trial = created.json()
    store.update("jobs", trial["job_id"], {"status": "running"})
    result = benchmark_runs.run_benchmark_trial(store, trial["id"], lambda *_: None, lambda: False)
    store.update(
        "jobs", trial["job_id"], {"status": "succeeded", "result": result, "finished_at": now()}
    )
    return reference, config, source, preview, trial, result


@pytest.mark.parametrize("transform,count", [("identity", 2), ("threshold", 1), ("review", 1)])
def test_import_canonical_source_and_real_correction_editor(
    workspace, monkeypatch, transform, count
):
    def forbidden(*args, **kwargs):
        pytest.fail("Recorded import must never authenticate or call a provider")

    monkeypatch.setattr(dinox_provider, "_credential", forbidden)
    monkeypatch.setattr(dinox_provider, "submit", forbidden)
    monkeypatch.setattr(dinox_review_provider.DinoXReviewer, "request", forbidden)
    client, store, _ = workspace
    annotations = store.list("annotation_revisions")
    reference, config, source, preview, trial, result = imported(workspace, transform)
    detail = client.get(f"/api/benchmark-trials/{trial['id']}").json()
    assert result["frames_ready"] == 1
    assert detail["counts"]["ready"] == 1 and "external_dispatch" not in detail
    assert detail["latency"]["includes"] == recorded.LATENCY_SCOPE
    assert detail["config"]["work"]["request_count"] == 0
    output = detail["outputs"][0]
    assert len(output["result"]["proposals"]) == count
    assert output["raw_response"] == source["frames"][0]
    assert output["metadata"]["source"]["dinox"]["elapsed_ms"] == 1234
    assert output["metadata"]["timing"]["scope"] == "local_import_validation"
    recorded.validate_output_row(output, detail, attempt=result["benchmark_attempt_id"])
    corrected = client.get(f"/api/benchmark-outputs/{output['id']}/correction").json()
    assert len(corrected["boxes"]) == count
    assert "reference" not in corrected and corrected["timer"]["elapsed_ms"] is None
    assert store.list("annotation_revisions") == annotations
    assert not store.list("benchmark_corrections") and not store.list("benchmark_timers")
    path = f"/api/benchmarks/{reference['id']}/recorded-trials"
    repeat = client.post(
        path,
        json={
            "config_id": config["id"],
            "role": "tuning",
            "bundle": source,
            "expected_fingerprint": preview["fingerprint"],
        },
    )
    assert repeat.status_code == 202 and repeat.json()["id"] == trial["id"]
    assert len(store.list("benchmark_trials")) == 1
    normal = client.post(
        f"/api/benchmarks/{reference['id']}/trials/preview",
        json={"config_id": config["id"], "role": "tuning"},
    )
    assert normal.status_code == 409 and "import preview" in normal.text


@pytest.mark.parametrize("transform", ["identity", "threshold", "review"])
def test_explicit_empty_evidence_is_reviewable(workspace, transform):
    _, _, _, _, trial, result = imported(workspace, transform, empty=True)
    assert result["frames_ready"] == 1
    output = workspace[1].list("benchmark_outputs", trial_id=trial["id"])[0]
    assert output["result"]["proposals"] == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("frame_id", "wrong"),
        ("image_file_sha256", "0" * 64),
        ("source_pixel_sha256", "0" * 64),
        ("width", 79),
    ],
)
def test_wrong_image_or_incomplete_role_fails_without_writes(workspace, field, value):
    client, store, _ = workspace
    reference, config = configured(workspace)
    source = bundle(store, reference, config)
    source["frames"][0][field] = value
    result = client.post(
        f"/api/benchmarks/{reference['id']}/recorded-trials/preview",
        json={"config_id": config["id"], "role": "tuning", "bundle": source},
    )
    assert result.status_code == 409
    assert not store.list("benchmark_trials") and not store.list("benchmark_outputs")


def test_config_lock_and_evaluation_partition(workspace):
    client, store, _ = workspace
    reference, config = configured(workspace)
    source = bundle(store, reference, config, "evaluation")
    path = f"/api/benchmarks/{reference['id']}/recorded-trials"
    payload = {"config_id": config["id"], "role": "evaluation", "bundle": source}
    assert client.post(path + "/preview", json=payload).status_code == 409
    detail = client.get(f"/api/benchmarks/{reference['id']}").json()
    locked = client.post(
        f"/api/benchmarks/{reference['id']}/lock",
        json={"expected_fingerprint": detail["lock_fingerprint"]},
    )
    assert locked.status_code == 200, locked.text
    assert client.post(path + "/preview", json=payload).status_code == 200
    assert (
        client.post(
            f"/api/benchmarks/{reference['id']}/configs/preview",
            json={
                "model_id": config["config"]["model_id"],
                "approach": "recorded_proposals",
                "recorded": config["config"]["recorded"],
            },
        ).status_code
        == 409
    )


@pytest.mark.parametrize(
    "mutation", ["raw", "result", "source", "attempt", "cost", "review_request"]
)
def test_saved_output_tampering_rejected(workspace, mutation):
    _, _, _, _, trial, result = imported(workspace, "review")
    output = deepcopy(workspace[1].list("benchmark_outputs", trial_id=trial["id"])[0])
    if mutation == "raw":
        output["raw_response"]["dinox"]["raw_result"]["objects"][0]["bbox"][0] += 1
    elif mutation == "result":
        output["result"]["proposals"][0]["box"][0] += 1
    elif mutation == "source":
        output["metadata"]["source"]["dinox"]["task_id"] = "different"
    elif mutation == "attempt":
        output["metadata"]["recorded"]["attempt_id"] = "different"
    elif mutation == "cost":
        output["metadata"]["source"]["review"]["usage_cost_usd"] = 0
    else:
        output["raw_response"]["review"]["input"]["request_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        recorded.validate_output_row(output, trial, attempt=result["benchmark_attempt_id"])


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "extra",
        "source_score",
        "receipt",
        "review_receipt",
        "model",
        "decision",
        "nonobject",
    ],
)
def test_invalid_provider_evidence_is_rejected_before_queue(workspace, mutation):
    client, store, _ = workspace
    reference, config = configured(workspace, "review")
    source = bundle(store, reference, config)
    frame = source["frames"][0]
    if mutation == "missing":
        source["frames"] = []
    elif mutation == "extra":
        source["frames"].append(deepcopy(frame))
    elif mutation == "source_score":
        frame["dinox"]["raw_result"]["objects"][0]["score"] = 1.1
    elif mutation == "receipt":
        frame["dinox"]["receipt"]["status"] = "pending"
    elif mutation == "review_receipt":
        frame["review"]["receipt"]["usage_cost_usd"] = 0
    elif mutation == "model":
        frame["review"]["raw_response"]["model"] = "unapproved-model"
    elif mutation == "decision":
        frame["review"]["raw_response"]["output"][0]["content"][0]["text"] = '{"decisions":[]}'
    else:
        frame["review"]["raw_response"] = []
    response = client.post(
        f"/api/benchmarks/{reference['id']}/recorded-trials/preview",
        json={"config_id": config["id"], "role": "tuning", "bundle": source},
    )
    assert response.status_code == 409, response.text
    assert not store.list("jobs") and not store.list("benchmark_trials")


def test_import_stale_preview_cancel_and_late_worker_do_not_publish(workspace):
    client, store, _ = workspace
    reference, config = configured(workspace)
    source = bundle(store, reference, config)
    path = f"/api/benchmarks/{reference['id']}/recorded-trials"
    payload = {"config_id": config["id"], "role": "tuning", "bundle": source}
    preview = client.post(path + "/preview", json=payload).json()
    stale = client.post(path, json={**payload, "expected_fingerprint": "0" * 64})
    assert stale.status_code == 409 and not store.list("jobs")
    created = client.post(path, json={**payload, "expected_fingerprint": preview["fingerprint"]})
    assert created.status_code == 202, created.text
    trial = created.json()
    result = benchmark_runs.run_benchmark_trial(store, trial["id"], lambda *_: None, lambda: True)
    assert result["cancelled"] is True and not store.list("benchmark_outputs")
    store.update("jobs", trial["job_id"], {"status": "cancelled"})
    with pytest.raises(recorded.BenchmarkConflict):
        benchmark_runs.run_benchmark_trial(store, trial["id"], lambda *_: None, lambda: False)
    assert not store.list("benchmark_outputs")


def test_size_and_huge_receipt_numbers_fail_safely(workspace, monkeypatch):
    client, store, _ = workspace
    reference, config = configured(workspace)
    source = bundle(store, reference, config)
    frames = load_benchmark_manifest(store, reference["id"])["frames"][:1]
    source["frames"][0]["dinox"]["receipt"]["elapsed_ms"] = 10**400
    with pytest.raises(ValueError, match="receipt"):
        recorded.validate_bundle(source, config["config"], frames)
    source["frames"][0]["dinox"]["receipt"]["elapsed_ms"] = None
    monkeypatch.setattr(recorded, "MAX_BUNDLE_BYTES", 64)
    response = client.post(
        f"/api/benchmarks/{reference['id']}/recorded-trials/preview",
        json={"config_id": config["id"], "role": "tuning", "bundle": source},
    )
    assert response.status_code == 409 and "16 MiB" in response.text


def test_recorded_api_project_scope_and_unknown_fields(workspace):
    client, store, _ = workspace
    reference, config = configured(workspace)
    source = bundle(store, reference, config)
    payload = {"config_id": config["id"], "role": "tuning", "bundle": source}
    path = f"/api/benchmarks/{reference['id']}/recorded-trials"
    project = create_project(store, name="Other project")
    assert (
        client.post(
            path + "/preview", json=payload, params={"project_id": project["id"]}
        ).status_code
        == 404
    )
    assert (
        client.post(
            path,
            json={**payload, "expected_fingerprint": "0" * 64},
            params={"project_id": project["id"]},
        ).status_code
        == 404
    )
    assert (
        client.post(path + "/preview", json={**payload, "approve_external": True}).status_code
        == 422
    )
    assert not store.list("benchmark_trials") and not store.list("jobs")
