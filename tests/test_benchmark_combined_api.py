"""HTTP integration for C, with simulated local and external model responses."""

import json

import pytest
from test_benchmark_api import freeze
from test_benchmark_api import workspace as workspace
from test_benchmark_combined import providers as providers

from iris import benchmark_multimodal, multimodal_provider
from iris.benchmark_runs import run_benchmark_trial
from iris.projects import create_project


def configured(workspace):
    client = workspace[0]
    reference = freeze(workspace)
    path = f"/api/benchmarks/{reference['id']}/configs"
    values = {
        "approach": "combined",
        "model_id": "gpt-6-astra+sam3",
        "combined": {
            "multimodal": {"image_long_edge": 512, "max_output_tokens": 1024},
            "segmentation": {"threshold": 0.7, "device": "cuda"},
        },
    }
    preview = client.post(path + "/preview", json=values)
    assert preview.status_code == 200, preview.text
    created = client.post(
        path,
        json={
            **values,
            "name": "Combined HTTP fixture",
            "expected_fingerprint": preview.json()["fingerprint"],
        },
    )
    assert created.status_code == 201, created.text
    candidate = created.json()
    trial_preview = client.post(
        f"/api/benchmarks/{reference['id']}/trials/preview",
        json={"config_id": candidate["id"], "role": "tuning"},
    )
    assert trial_preview.status_code == 200, trial_preview.text
    return reference, candidate, trial_preview.json()


def approval(candidate, preview):
    return {
        "config_id": candidate["id"],
        "role": "tuning",
        "expected_fingerprint": preview["fingerprint"],
        "preview_token": preview["preview_token"],
        "approve_external": True,
        "max_cost_usd": preview["external_plan"]["estimate"]["upper_bound_usd"],
    }


def test_config_and_generated_data_disclosure_without_installation(workspace, monkeypatch):
    client, store, _ = workspace
    for name in (
        multimodal_provider.KEY_ENV,
        multimodal_provider.FALLBACK_KEY_ENV,
        "IRIS_SAM_PYTHON",
    ):
        monkeypatch.delenv(name, raising=False)
    reference, candidate, preview = configured(workspace)
    catalog = client.get("/api/benchmark-providers")
    assert catalog.status_code == 200 and catalog.json()["combined"]["ready"] is False
    assert preview["launch_allowed"] is False and preview["work"]["request_count"] == 2
    item = preview["external_plan"]["requests"][0]
    assert "request_sha256" in item["planning"]["input"]
    assert item["review"]["template"]["dynamic_fields"] == ["planning_prompts", "candidates"]
    assert client.get(item["image_url"]).headers["content-type"] == "image/png"
    rejected = client.post(
        f"/api/benchmarks/{reference['id']}/trials", json=approval(candidate, preview)
    )
    assert rejected.status_code == 409 and not store.list("jobs")


@pytest.mark.parametrize(
    "change,status",
    [
        ({"combined": {"segmentation": {"class_prompts": {"person": "helmet"}}}}, 422),
        ({"combined": {"segmentation": {"device": "cpu"}}}, 422),
        ({"combined": {"segmentation": {"threshold": True}}}, 422),
        ({"combined": {"iterations": 2}}, 422),
        ({"combined": {"multimodal": {"provider": "another"}}}, 422),
        ({"model_id": "sam3"}, 409),
        ({"threshold": 0.9}, 409),
        ({"inference_mode": "tiled"}, 409),
        ({"segmentation": {}}, 409),
        ({"multimodal": {}}, 409),
        ({"approach": "multimodal"}, 409),
    ],
)
def test_protocol_does_not_accept_silent_settings_or_unbounded_iterations(
    workspace, change, status
):
    client, store, _ = workspace
    reference = freeze(workspace)
    result = client.post(
        f"/api/benchmarks/{reference['id']}/configs/preview",
        json={
            "approach": "combined",
            "model_id": "gpt-6-astra+sam3",
            "combined": {},
            **change,
        },
    )
    assert result.status_code == status, result.text
    assert not store.list("benchmark_configs")


def test_consent_replay_stage_history_and_cross_project_access(workspace, providers):
    client, store, _ = workspace
    reference, candidate, preview = configured(workspace)
    path = f"/api/benchmarks/{reference['id']}/trials"
    values = approval(candidate, preview)
    for change in ({"approve_external": False}, {"max_cost_usd": 0}, {"preview_token": "invalid"}):
        rejected = client.post(path, json={**values, **change})
        assert rejected.status_code == 409, rejected.text
    created = client.post(path, json=values)
    assert created.status_code == 202, created.text
    trial = created.json()
    replay = client.post(path, json=values)
    assert replay.status_code == 202 and replay.json()["id"] == trial["id"]
    assert len(store.list("jobs")) == 1 and not providers["requests"]
    result = run_benchmark_trial(store, trial["id"], lambda *_: None, lambda: False)
    assert result["frames_ready"] == 1
    detail = client.get(f"/api/benchmark-trials/{trial['id']}")
    assert detail.status_code == 200, detail.text
    data = detail.json()
    assert [row["stage"] for row in data["external_dispatch"]["outputs"]] == ["planning", "review"]
    assert set(data["outputs"][0]["metadata"]["pipeline"]["stages"]) == {
        "planning",
        "grounding",
        "review",
    }
    assert "offline-combined-fixture-key" not in json.dumps(data)
    job = client.get(f"/api/jobs/{trial['job_id']}")
    assert job.status_code == 200, job.text
    assert job.json()["dispatch"]["counts"]["response_received"] == 2
    assert client.get("/api/jobs").status_code == 200
    other = create_project(store, name="Other fixture project")
    for url in (
        f"/api/benchmark-trials/{trial['id']}",
        preview["external_plan"]["requests"][0]["image_url"],
    ):
        forbidden = client.get(url, params={"project_id": other["id"]})
        assert forbidden.status_code == 404, forbidden.text


def test_expired_approval_never_creates_trial(workspace, providers, monkeypatch):
    monkeypatch.setattr(benchmark_multimodal, "PREVIEW_LIFETIME_SECONDS", -1)
    reference, candidate, preview = configured(workspace)
    result = workspace[0].post(
        f"/api/benchmarks/{reference['id']}/trials", json=approval(candidate, preview)
    )
    assert result.status_code == 409 and "expired" in result.text
    assert not workspace[1].list("jobs") and not providers["requests"]
