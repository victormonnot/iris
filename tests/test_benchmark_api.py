"""Independent benchmark HTTP workflow and project boundaries (synthetic detector)."""

from copy import deepcopy

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from iris import benchmark, benchmark_runs, inference
from iris.annotations import save_annotation
from iris.app import create_app
from iris.media import import_asset
from iris.projects import create_project
from iris.store import new_id, now

MODEL = "ssdlite320_mobilenet_v3_large"


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    app = create_app(tmp_path / "workspace", run_jobs=False)
    store = app.state.store
    frames = []
    roles = {}
    for i, role in enumerate(("tuning", "evaluation")):
        session = store.insert(
            "sessions", dict(id=new_id(), name=role, scene_group=role, created_at=now())
        )
        source = tmp_path / f"{role}.png"
        Image.new("RGB", (80, 60), (i * 60, 30, 50)).save(source)
        asset = import_asset(store, session["id"], source, source.name)
        frame = store.list("frames", asset_id=asset["id"])[0]
        store.update("frames", frame["id"], {"selected": True})
        save_annotation(
            store,
            frame["id"],
            expected_revision=0,
            boxes=[dict(id="human-box", label="person", box=[2, 3, 30, 35])],
            decisions={},
            status="validated",
            reviewer="Synthetic independent reference",
        )
        frames.append(frame)
        roles[role] = role
    model = {**inference.get_spec(MODEL), "status": "ready", "weight_sha256": "a" * 64}
    for module in (benchmark, benchmark_runs):
        monkeypatch.setattr(module, "catalog", lambda _: [deepcopy(model)])
    payload = dict(
        frame_ids=[f["id"] for f in frames],
        roles=roles,
        reviewer="Synthetic reference author",
        independence_notes=(
            "Generated fixture, manually specified reference before detector invocation."
        ),
        independent_reference=True,
    )
    with TestClient(app, base_url="http://127.0.0.1") as client:
        yield client, store, payload


def freeze(workspace):
    client, _, payload = workspace
    preview = client.post("/api/benchmarks/preview", json=payload)
    assert preview.status_code == 200, preview.text
    created = client.post(
        "/api/benchmarks",
        json={
            **payload,
            "name": "Independent fixture",
            "expected_fingerprint": preview.json()["fingerprint"],
        },
    )
    assert created.status_code == 201, created.text
    return created.json()


def config(workspace, reference):
    client = workspace[0]
    path = f"/api/benchmarks/{reference['id']}/configs"
    preview = client.post(path + "/preview", json={"model_id": MODEL})
    assert preview.status_code == 200, preview.text
    created = client.post(
        path,
        json={
            "name": "Fixture detector",
            "model_id": MODEL,
            "expected_fingerprint": preview.json()["fingerprint"],
        },
    )
    assert created.status_code == 201, created.text
    return created.json()


class FixtureDetector:
    metadata = {"weight_sha256": "a" * 64}

    def __init__(self, root, model_id, *, device):
        assert model_id == MODEL and device == "cpu"

    def warmup(self, image):
        assert isinstance(image, Image.Image)

    def predict(self, image):
        assert isinstance(image, Image.Image)
        return {
            "input_size": list(image.size),
            "timing": {"preprocess_ms": 1, "inference_ms": 2, "postprocess_ms": 1, "total_ms": 4},
            "detections": [],
        }


def completed_trial(workspace, reference, configuration, role="tuning"):
    client, store, _ = workspace
    path = f"/api/benchmarks/{reference['id']}/trials"
    payload = dict(config_id=configuration["id"], role=role)
    preview = client.post(path + "/preview", json=payload)
    assert preview.status_code == 200, preview.text
    response = client.post(
        path, json={**payload, "expected_fingerprint": preview.json()["fingerprint"]}
    )
    assert response.status_code == 202, response.text
    trial = response.json()
    store.update("jobs", trial["job_id"], {"status": "running"})
    result = benchmark_runs.run_benchmark_trial(
        store, trial["id"], lambda *_: None, lambda: False, detector_factory=FixtureDetector
    )
    store.update("jobs", trial["job_id"], {"status": "succeeded", "result": result})
    response = client.get(f"/api/benchmark-trials/{trial['id']}")
    assert response.status_code == 200, response.text
    return response.json()


def test_full_http_protocol_corrects_without_reference_leakage(workspace):
    client, store, payload = workspace
    reference = freeze(workspace)
    configuration = config(workspace, reference)
    before = store.list("annotation_revisions")
    trial = completed_trial(workspace, reference, configuration)
    assert trial["counts"] == dict(total=1, outputs=1, ready=1, issues=0)
    assert trial["quality"]["complete"]
    output = trial["outputs"][0]
    path = f"/api/benchmark-outputs/{output['id']}"
    doc = client.get(path + "/correction").json()
    assert doc["boxes"] == []
    assert doc["frame"]["image_url"]
    assert "reference" not in doc and "annotation" not in doc["frame"]
    image = client.get(doc["frame"]["image_url"])
    assert image.status_code == 200 and image.content.startswith(b"\x89PNG")
    started = client.post(
        path + "/timer",
        json=dict(
            action="start",
            expected_revision=0,
            token="fixture-owner-abcdefghijkl",
            operation_id=new_id(),
            reviewer="Fixture reviewer",
        ),
    )
    assert started.status_code == 200, started.text
    saved = client.put(
        path + "/correction",
        json=dict(
            expected_revision=0,
            boxes=[dict(id="added", label="person", box=[3, 4, 30, 35])],
            status="reviewed",
            reviewer="Fixture reviewer",
            timer_revision=started.json()["revision"],
            timer_token="fixture-owner-abcdefghijkl",
        ),
    )
    assert saved.status_code == 200, saved.text
    assert saved.json()["timing"]["elapsed_ms"] > 0
    assert client.get(path + "/corrections/1").json()["status"] == "reviewed"
    assert store.list("annotation_revisions") == before
    assert store.get("benchmark_outputs", output["id"]) == output
    assert (
        client.get(f"/api/jobs/{trial['job_id']}").json()["next_action"]["workspace"] == "benchmark"
    )
    # Evaluation requires explicit config-set lock, tuning is closed afterward.
    assert (
        client.post(
            f"/api/benchmarks/{reference['id']}/trials/preview",
            json=dict(config_id=configuration["id"], role="evaluation"),
        ).status_code
        == 409
    )
    detail = client.get(f"/api/benchmarks/{reference['id']}").json()
    locked = client.post(
        f"/api/benchmarks/{reference['id']}/lock",
        json={"expected_fingerprint": detail["lock_fingerprint"]},
    )
    assert locked.status_code == 200, locked.text
    assert (
        client.post(
            f"/api/benchmarks/{reference['id']}/configs/preview", json={"model_id": MODEL}
        ).status_code
        == 409
    )
    evaluation = completed_trial(workspace, reference, configuration, "evaluation")
    assert evaluation["quality"]["complete"]
    assert (
        client.post(
            f"/api/benchmarks/{reference['id']}/trials/preview",
            json=dict(config_id=configuration["id"], role="tuning"),
        ).status_code
        == 409
    )
    assert len(payload["frame_ids"]) == 2


def test_project_scope_and_strict_request_boundaries(workspace):
    client, store, _ = workspace
    reference = freeze(workspace)
    configuration = config(workspace, reference)
    trial = completed_trial(workspace, reference, configuration)
    output = trial["outputs"][0]
    project = create_project(store, name="Other project")
    params = {"project_id": project["id"]}
    for path in [
        f"/api/benchmarks/{reference['id']}",
        f"/api/benchmark-trials/{trial['id']}",
        f"/api/benchmark-outputs/{output['id']}/correction",
        f"/api/benchmarks/{reference['id']}/frames/{output['frame_id']}/image",
    ]:
        assert client.get(path, params=params).status_code == 404
    assert client.get("/api/benchmarks", params=params).json() == []
    assert (
        client.post(
            f"/api/benchmark-outputs/{output['id']}/timer",
            json=dict(
                action="start",
                expected_revision=True,
                token="abcdefghijklmnop",
                operation_id=new_id(),
                reviewer="Test",
            ),
        ).status_code
        == 422
    )
    assert (
        client.put(
            f"/api/benchmark-outputs/{output['id']}/correction",
            json=dict(expected_revision=0, boxes=[], reference=[]),
        ).status_code
        == 422
    )


def test_frozen_image_bytes_are_checked_on_read(workspace):
    client, store, _ = workspace
    reference = freeze(workspace)
    frame = reference["manifest"]["frames"][0]
    path = store.root / frame["image_path"]
    path.write_bytes(b"changed")
    response = client.get(f"/api/benchmarks/{reference['id']}/frames/{frame['frame_id']}/image")
    assert response.status_code == 409
    assert "changed" in response.json()["detail"]


def test_damaged_frozen_image_cannot_start_or_complete_human_review(workspace):
    client, store, _ = workspace
    reference = freeze(workspace)
    trial = completed_trial(workspace, reference, config(workspace, reference))
    output = trial["outputs"][0]
    frame = next(
        frame
        for frame in reference["manifest"]["frames"]
        if frame["frame_id"] == output["frame_id"]
    )
    (store.root / frame["image_path"]).write_bytes(b"damaged")
    path = f"/api/benchmark-outputs/{output['id']}"
    assert client.get(path + "/correction").status_code == 409
    assert (
        client.post(
            path + "/timer",
            json=dict(
                action="start",
                expected_revision=0,
                token="abcdefghijklmnop",
                operation_id=new_id(),
                reviewer="Test",
            ),
        ).status_code
        == 409
    )
    assert (
        client.put(
            path + "/correction",
            json=dict(expected_revision=0, boxes=[], status="reviewed", reviewer="Test"),
        ).status_code
        == 409
    )
    assert store.list("benchmark_corrections") == []
    assert store.list("benchmark_timers") == []
