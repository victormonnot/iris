"""SAM HTTP preparation and admission; no runtime probing or model execution."""

from copy import deepcopy

import pytest
from test_benchmark_api import freeze
from test_benchmark_api import workspace as workspace

from iris import sam_provider, sam_runtime
from iris.projects import create_project


@pytest.fixture(autouse=True)
def no_runtime(monkeypatch):
    monkeypatch.delenv("IRIS_SAM_PYTHON", raising=False)
    monkeypatch.delenv("IRIS_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def absent(config=None, *, force=False):
        return {
            "ready": False,
            "status": "missing_runtime",
            "reason": "Isolated fixture: runtime deliberately absent",
            "identity": None,
        }

    def unexpected(*args, **kwargs):
        pytest.fail("HTTP preparation must not load or invoke SAM")

    monkeypatch.setattr(sam_runtime, "runtime_status", absent)
    monkeypatch.setattr(sam_provider, "Sam3Preannotator", unexpected)


def configured(workspace):
    reference = freeze(workspace)
    client = workspace[0]
    path = f"/api/benchmarks/{reference['id']}/configs"
    settings = {
        "approach": "segmentation",
        "model_id": "sam3",
        "segmentation": {
            "threshold": 0.6,
            "device": "cuda",
            "class_prompts": {"person": "person", "car": "passenger car"},
        },
    }
    preview = client.post(path + "/preview", json=settings)
    assert preview.status_code == 200, preview.text
    payload = {
        **settings,
        "name": "SAM prepared without installed runtime",
        "expected_fingerprint": preview.json()["fingerprint"],
    }
    response = client.post(path, json=payload)
    assert response.status_code == 201, response.text
    return reference, response.json(), preview.json(), payload


def test_http_preview_freeze_and_replay_need_no_runtime_or_weights(workspace):
    client, store, _ = workspace
    revisions = deepcopy(store.list("annotation_revisions"))
    reference, config, preview, payload = configured(workspace)
    frozen = config["config"]["provider_config"]
    assert config["approach"] == "segmentation"
    assert frozen["prompts"] == [
        {"class_id": "person", "text": "person"},
        {"class_id": "car", "text": "passenger car"},
    ]
    assert frozen["settings"]["threshold"] == 0.6
    assert frozen["weights"]["sha256"] == sam_provider.CHECKPOINT_SHA256
    assert frozen["code_revision"] == sam_provider.CODE_REVISION
    assert frozen["taxonomy"] == reference["manifest"]["taxonomy"]
    assert preview["provider_status"]["ready"] is False
    assert preview["work"]["tuning"]["prompt_evaluations"] == 2
    path = f"/api/benchmarks/{reference['id']}/configs"
    replay = client.post(path, json=payload)
    assert replay.status_code == 201 and replay.json()["id"] == config["id"]
    assert len(store.list("benchmark_configs")) == 1
    assert not store.list("jobs")
    assert store.list("annotation_revisions") == revisions


def test_http_trial_preview_explains_missing_installation_and_queue_is_blocked(workspace):
    client, store, _ = workspace
    reference, config, _, _ = configured(workspace)
    path = f"/api/benchmarks/{reference['id']}/trials"
    payload = {"config_id": config["id"], "role": "tuning"}
    response = client.post(path + "/preview", json=payload)
    assert response.status_code == 200, response.text
    preview = response.json()
    assert preview["launch_allowed"] is False
    assert preview["launch_reason"]
    assert preview["local_plan"]["runtime_identity"] is None
    assert preview["local_plan"]["weight_sha256"] == sam_provider.CHECKPOINT_SHA256
    assert "external_plan" not in preview
    queued = client.post(
        path,
        json={**payload, "expected_fingerprint": preview["fingerprint"]},
    )
    assert queued.status_code == 409, queued.text
    assert not store.list("jobs")
    assert not store.list("benchmark_trials")
    assert not store.list("benchmark_outputs")


def test_sam_catalog_describes_local_readiness_and_limits_without_probe(workspace):
    response = workspace[0].get("/api/benchmark-providers")
    assert response.status_code == 200, response.text
    data = response.json()
    status = data["segmentation"]
    assert status["local_only"] and not status["external"]
    assert status["model_id"] == "sam3" and status["ready"] is False
    assert status["runtime"]["status"] == "missing_runtime"
    assert status["weights"]["available"] is False
    assert status["weights"]["size_bytes"] == sam_provider.CHECKPOINT_SIZE
    assert data["segmentation_settings"]["devices"] == ["cuda"]
    assert data["segmentation_settings"]["prompt_limits"]["max_tokens"] == 30
    assert not workspace[1].list("jobs")


@pytest.mark.parametrize(
    "settings,expected_status",
    [
        ({"threshold": True}, 422),
        ({"threshold": -0.1}, 422),
        ({"threshold": 1.1}, 422),
        ({"device": "cpu"}, 422),
        ({"precision": "float32"}, 422),
        ({"class_prompts": {"person": 3, "car": "car"}}, 422),
        ({"class_prompts": {}}, 409),
        ({"class_prompts": {"person": "person"}}, 409),
        ({"class_prompts": {"person": "person", "car": "car", "unknown": "thing"}}, 409),
        ({"class_prompts": {"person": "", "car": "car"}}, 409),
        ({"class_prompts": {"person": "person", "car": "x" * 121}}, 409),
        ({"class_prompts": {"person": "person\n", "car": "car"}}, 409),
    ],
)
def test_invalid_sam_settings_never_publish_configuration(workspace, settings, expected_status):
    reference = freeze(workspace)
    client, store, _ = workspace
    response = client.post(
        f"/api/benchmarks/{reference['id']}/configs/preview",
        json={"model_id": "sam3", "approach": "segmentation", "segmentation": settings},
    )
    assert response.status_code == expected_status, response.text
    assert not store.list("benchmark_configs")
    assert not store.list("jobs")


@pytest.mark.parametrize(
    "changes",
    [
        {"model_id": "sam3.1"},
        {"model_id": "gpt-6-astra"},
        {"multimodal": {"reasoning_effort": "low"}},
        {"approach": "combined"},
        {"approach": "local_detector"},
    ],
)
def test_sam_model_and_approach_are_explicit_without_fallback(workspace, changes):
    reference = freeze(workspace)
    response = workspace[0].post(
        f"/api/benchmarks/{reference['id']}/configs/preview",
        json={"model_id": "sam3", "approach": "segmentation", "segmentation": {}, **changes},
    )
    assert response.status_code == 409, response.text
    assert not workspace[1].list("benchmark_configs")


def test_preview_fingerprint_binds_the_exact_text_and_threshold(workspace):
    client, store, _ = workspace
    reference = freeze(workspace)
    path = f"/api/benchmarks/{reference['id']}/configs"
    payload = {"model_id": "sam3", "approach": "segmentation", "segmentation": {"threshold": 0.5}}
    response = client.post(path + "/preview", json=payload)
    assert response.status_code == 200, response.text
    changed = {
        **payload,
        "segmentation": {"threshold": 0.7},
        "name": "Changed after preview",
        "expected_fingerprint": response.json()["fingerprint"],
    }
    refused = client.post(path, json=changed)
    assert refused.status_code == 409, refused.text
    assert not store.list("benchmark_configs")


def test_sam_configuration_and_trial_operations_cannot_cross_projects(workspace):
    client, store, _ = workspace
    reference, config, _, payload = configured(workspace)
    project = create_project(store, name="Foreign benchmark project")
    params = {"project_id": project["id"]}
    prefix = f"/api/benchmarks/{reference['id']}"
    for suffix, body in [
        ("/configs/preview", {"model_id": "sam3", "approach": "segmentation"}),
        ("/configs", payload),
        ("/trials/preview", {"config_id": config["id"], "role": "tuning"}),
        (
            "/trials",
            {"config_id": config["id"], "role": "tuning", "expected_fingerprint": "a" * 64},
        ),
    ]:
        response = client.post(prefix + suffix, params=params, json=body)
        assert response.status_code == 404, response.text
    assert not store.list("jobs")
    assert not store.list("benchmark_trials")
