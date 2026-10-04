"""Multimodal HTTP consent and isolation with an intercepted, offline HTTPS fixture."""

import base64
import hashlib
import io
import json
from copy import deepcopy

import pytest
from PIL import Image, PngImagePlugin
from test_benchmark_api import freeze
from test_benchmark_api import workspace as workspace
from test_multimodal_provider import KEY
from test_multimodal_provider import transport as transport

from iris import benchmark_multimodal, benchmark_runs, multimodal_provider
from iris.annotations import save_annotation
from iris.media import import_asset
from iris.projects import create_project


def configured(workspace, *, settings=None):
    client = workspace[0]
    reference = freeze(workspace)
    path = f"/api/benchmarks/{reference['id']}/configs"
    values = dict(approach="multimodal", model_id="gpt-6-astra", multimodal=settings or {})
    preview = client.post(path + "/preview", json=values)
    assert preview.status_code == 200, preview.text
    created = client.post(
        path,
        json={
            **values,
            "name": "Offline multimodal API fixture",
            "expected_fingerprint": preview.json()["fingerprint"],
        },
    )
    assert created.status_code == 201, created.text
    return reference, created.json()


def preview_trial(client, reference, config):
    response = client.post(
        f"/api/benchmarks/{reference['id']}/trials/preview",
        json=dict(config_id=config["id"], role="tuning"),
    )
    assert response.status_code == 200, response.text
    return response.json()


def approval(config, preview):
    return dict(
        config_id=config["id"],
        role="tuning",
        expected_fingerprint=preview["fingerprint"],
        approve_external=True,
        max_cost_usd=preview["external_plan"]["estimate"]["upper_bound_usd"],
        preview_token=preview["preview_token"],
    )


def test_missing_key_allows_offline_config_and_exact_preview_but_not_trial(
    workspace, transport, monkeypatch
):
    client, store, _ = workspace
    monkeypatch.delenv(multimodal_provider.KEY_ENV)
    monkeypatch.delenv(multimodal_provider.FALLBACK_KEY_ENV, raising=False)
    provider = client.get("/api/benchmark-providers")
    assert provider.status_code == 200, provider.text
    assert provider.json()["multimodal"]["status"] == "missing_key"
    assert provider.json()["multimodal"]["connection_verified"] is False
    assert provider.json()["multimodal_settings"]["defaults"]["image_long_edge"] == 1536
    reference, config = configured(workspace)
    before = {
        table: store.list(table)
        for table in ("benchmark_configs", "benchmark_trials", "benchmark_outputs", "jobs")
    }
    preview = preview_trial(client, reference, config)
    assert preview["launch_allowed"] is False
    assert preview["preview_token"] and preview["expires_at"]
    request = preview["external_plan"]["requests"][0]
    image = client.get(request["image_url"])
    assert image.status_code == 200
    assert hashlib.sha256(image.content).hexdigest() == request["input"]["image"]["sha256"]
    response = client.post(
        f"/api/benchmarks/{reference['id']}/trials", json=approval(config, preview)
    )
    assert response.status_code == 409, response.text
    assert "KEY" in response.json()["detail"]
    assert {table: store.list(table) for table in before} == before
    assert transport["connections"] == []


def test_ready_catalog_and_previews_remain_read_only_and_hide_key(workspace, transport):
    client, store, payload = workspace
    reference, config = configured(
        workspace,
        settings=dict(image_long_edge=512, reasoning_effort="high", max_output_tokens=1024),
    )
    baseline = {
        table: store.list(table)
        for table in (
            "benchmarks",
            "benchmark_configs",
            "benchmark_trials",
            "benchmark_outputs",
            "annotation_revisions",
            "annotation_suggestions",
            "jobs",
        )
    }
    catalog = client.get("/api/benchmark-providers").json()
    assert catalog["multimodal"]["status"] == "ready"
    assert catalog["multimodal"]["connection_verified"] is False
    preview = preview_trial(client, reference, config)
    again = preview_trial(client, reference, config)
    assert again["fingerprint"] == preview["fingerprint"]
    assert preview["launch_allowed"] is True
    assert {table: store.list(table) for table in baseline} == baseline
    saved = config["config"]["provider_config"]
    assert saved["image_encoding"]["long_edge"] == 512
    assert saved["settings"]["reasoning"]["effort"] == "high"
    assert saved["settings"]["max_output_tokens"] == 1024
    request = preview["external_plan"]["requests"][0]
    assert set(request["input"]) == {"image", "prompt", "request_sha256"}
    prompt = json.loads(request["input"]["prompt"])
    candidate_text = json.loads(prompt["input_text"])
    assert set(candidate_text) == {"classes", "image_size", "coordinate_space"}
    assert candidate_text["classes"] == [
        {key: item[key] for key in ("id", "name", "definition")}
        for item in reference["manifest"]["taxonomy"]["classes"]
    ]
    assert payload["reviewer"] not in request["input"]["prompt"]
    assert payload["independence_notes"] not in request["input"]["prompt"]
    assert "human-box" not in request["input"]["prompt"]
    assert KEY not in json.dumps([catalog, config, preview])
    assert transport["connections"] == []


def test_input_png_is_exact_resized_metadata_free_and_project_scoped(
    workspace, transport, tmp_path
):
    client, store, payload = workspace
    old_frame = store.get("frames", payload["frame_ids"][0])
    source = tmp_path / "wide-private-name.png"
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("private-reference", "HUMAN METADATA SENTINEL")
    Image.new("RGB", (1600, 800), (10, 60, 90)).save(source, pnginfo=metadata)
    asset = import_asset(store, old_frame["session_id"], source, source.name)
    frame = store.list("frames", asset_id=asset["id"])[0]
    store.update("frames", frame["id"], {"selected": True})
    save_annotation(
        store,
        frame["id"],
        expected_revision=0,
        boxes=[dict(id="private-human-box", label="person", box=[20, 30, 400, 600])],
        decisions={},
        status="validated",
        reviewer="Private independent reviewer",
    )
    payload["frame_ids"][0] = frame["id"]
    reference, config = configured(workspace, settings={"image_long_edge": 512})
    preview = preview_trial(client, reference, config)
    request = preview["external_plan"]["requests"][0]
    assert request["frame_id"] == frame["id"]
    response = client.get(request["image_url"])
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.headers["cache-control"] == "no-store"
    descriptor = request["input"]["image"]
    assert (descriptor["width"], descriptor["height"]) == (1600, 800)
    assert (descriptor["sent_width"], descriptor["sent_height"]) == (512, 256)
    assert descriptor["transform"]["scale"] == [3.125, 3.125]
    assert hashlib.sha256(response.content).hexdigest() == descriptor["sha256"]
    with Image.open(io.BytesIO(response.content)) as png:
        assert png.size == (512, 256) and png.mode == "RGB" and not png.info
    assert b"HUMAN METADATA SENTINEL" not in response.content
    assert source.name not in request["input"]["prompt"]
    project = create_project(store, name="Isolated image reader")
    assert client.get(request["image_url"], params={"project_id": project["id"]}).status_code == 404
    assert transport["connections"] == []


@pytest.mark.parametrize(
    "change,status",
    [
        ({"api_key": "not-accepted-in-ui"}, 422),
        ({"endpoint": "https://other.invalid"}, 422),
        ({"multimodal": {"reasoning_effort": "none"}}, 422),
        ({"multimodal": {"max_output_tokens": True}}, 422),
        ({"multimodal": {"max_output_tokens": 8193}}, 422),
        ({"multimodal": {"image_long_edge": 1600}}, 422),
        ({"multimodal": {"system_prompt": "Use reference boxes"}}, 422),
        ({"model_id": "gpt-other"}, 409),
        ({"approach": "segmentation"}, 409),
        ({"approach": "combined"}, 409),
    ],
)
def test_config_api_rejects_unsupported_or_secret_payloads(workspace, transport, change, status):
    client, store, _ = workspace
    reference = freeze(workspace)
    response = client.post(
        f"/api/benchmarks/{reference['id']}/configs/preview",
        json={"approach": "multimodal", "model_id": "gpt-6-astra", **change},
    )
    assert response.status_code == status, response.text
    assert not store.list("benchmark_configs") and not store.list("jobs")
    assert transport["connections"] == []


def test_api_consent_requires_exact_token_budget_and_strict_types(workspace, transport):
    client, store, _ = workspace
    reference, config = configured(workspace)
    preview = preview_trial(client, reference, config)
    values = approval(config, preview)
    path = f"/api/benchmarks/{reference['id']}/trials"
    for change, status in [
        ({"approve_external": False}, 409),
        ({"approve_external": "true"}, 422),
        ({"max_cost_usd": True}, 422),
        ({"max_cost_usd": "1"}, 422),
        ({"max_cost_usd": 0}, 409),
        ({"max_cost_usd": 1000.01}, 422),
        ({"preview_token": None}, 409),
        ({"preview_token": "invalid"}, 409),
        ({"expected_fingerprint": "0" * 64}, 409),
        ({"reference_boxes": []}, 422),
    ]:
        response = client.post(path, json={**values, **change})
        assert response.status_code == status, (change, response.text)
    assert not store.list("benchmark_trials") and not store.list("jobs")
    assert transport["connections"] == []


def test_expired_or_restarted_preview_creates_no_job(workspace, transport, monkeypatch):
    client, store, _ = workspace
    reference, config = configured(workspace)
    monkeypatch.setattr(benchmark_multimodal, "PREVIEW_LIFETIME_SECONDS", -1)
    expired = preview_trial(client, reference, config)
    monkeypatch.setattr(benchmark_multimodal, "PREVIEW_LIFETIME_SECONDS", 600)
    response = client.post(
        f"/api/benchmarks/{reference['id']}/trials", json=approval(config, expired)
    )
    assert response.status_code == 409 and "expired" in response.text
    current = preview_trial(client, reference, config)
    monkeypatch.setattr(benchmark_multimodal, "_PREVIEW_SECRET", b"new-fixture-process")
    response = client.post(
        f"/api/benchmarks/{reference['id']}/trials", json=approval(config, current)
    )
    assert response.status_code == 409 and "expired" in response.text
    assert not store.list("benchmark_trials") and not store.list("jobs")
    assert transport["connections"] == []


def test_trial_receipt_reconciles_once_then_offline_output_retains_usage_and_isolation(
    workspace, transport
):
    client, store, payload = workspace
    reference, config = configured(workspace)
    preview = preview_trial(client, reference, config)
    values = approval(config, preview)
    path = f"/api/benchmarks/{reference['id']}/trials"
    response = client.post(path, json=values)
    assert response.status_code == 202, response.text
    trial = response.json()
    assert trial["external_dispatch"]["counts"]["not_started"] == 1
    assert transport["connections"] == [], "Queueing cannot dispatch with run_jobs=False"
    replay = client.post(path, json=values)
    assert replay.status_code == 202 and replay.json()["id"] == trial["id"]
    listed = client.get(f"/api/benchmarks/{reference['id']}").json()["trials"]
    assert len(listed) == 1 and listed[0]["config"]["fingerprint"] == preview["fingerprint"]
    assert len(store.list("jobs")) == len(store.list("benchmark_trials")) == 1
    before = deepcopy(store.list("annotation_revisions"))
    result = benchmark_runs.run_benchmark_trial(store, trial["id"], lambda *_: None, lambda: False)
    store.update("jobs", trial["job_id"], {"status": "succeeded", "result": result})
    detail = client.get(f"/api/benchmark-trials/{trial['id']}").json()
    assert detail["quality"]["complete"] is True
    assert detail["external_dispatch"]["counts"]["response_received"] == 1
    assert detail["external_dispatch"]["usage_cost_usd"] > 0
    assert detail["external_dispatch"]["unknown_outcome_count"] == 0
    output = detail["outputs"][0]
    assert output["raw_response"] == transport["response"]
    assert output["result"]["proposals"][0]["score"] is None
    assert len(transport["requests"]) == 1
    sent = transport["requests"][0][2]
    assert payload["reviewer"].encode() not in sent
    assert payload["independence_notes"].encode() not in sent
    assert b"human-box" not in sent
    assert (
        hashlib.sha256(sent).hexdigest()
        == preview["external_plan"]["requests"][0]["input"]["request_sha256"]
    )
    sent_png = base64.b64decode(
        json.loads(sent)["input"][0]["content"][1]["image_url"].split(",")[1]
    )
    assert sent_png == client.get(preview["external_plan"]["requests"][0]["image_url"]).content
    assert store.list("annotation_revisions") == before
    assert not store.list("annotation_suggestions")
    job = client.get(f"/api/jobs/{trial['job_id']}").json()
    assert job["dispatch"]["counts"]["response_received"] == 1
    project = create_project(store, name="Other trial reader")
    for read in (
        f"/api/benchmarks/{reference['id']}",
        f"/api/benchmark-trials/{trial['id']}",
        f"/api/benchmark-outputs/{output['id']}/correction",
        f"/api/jobs/{trial['job_id']}",
    ):
        assert client.get(read, params={"project_id": project["id"]}).status_code == 404
    assert (
        client.post(
            path + "/preview",
            params={"project_id": project["id"]},
            json={"config_id": config["id"], "role": "tuning"},
        ).status_code
        == 404
    )
    assert client.post(path, params={"project_id": project["id"]}, json=values).status_code == 404
    assert len(transport["requests"]) == 1


def test_changed_frozen_pixels_block_new_preview_and_input_image(workspace, transport):
    client, store, _ = workspace
    reference, config = configured(workspace)
    preview = preview_trial(client, reference, config)
    request = preview["external_plan"]["requests"][0]
    frame = next(
        row for row in reference["manifest"]["frames"] if row["frame_id"] == request["frame_id"]
    )
    (store.root / frame["image_path"]).write_bytes(b"changed frozen pixels")
    assert client.get(request["image_url"]).status_code == 409
    response = client.post(
        f"/api/benchmarks/{reference['id']}/trials/preview",
        json={"config_id": config["id"], "role": "tuning"},
    )
    assert response.status_code == 409
    assert not store.list("jobs") and transport["connections"] == []
