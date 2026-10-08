"""HTTP cache creation, durable reads, project boundaries and activity evidence."""

from copy import deepcopy

import pytest
from fastapi.testclient import TestClient
from test_temporal_api import publish_sequence, sequence_source
from test_temporal_detector import frozen_config, runtime_metadata

from iris import temporal_detections as caches
from iris import temporal_detector as adapter
from iris.app import create_app
from iris.projects import record_project
from iris.store import new_id, now
from iris.training_architectures import SSDLITE


@pytest.fixture
def client(tmp_path, monkeypatch):
    def prepare(root, model_id, **settings):
        config = frozen_config(
            model_id,
            device=settings.get("device", "cpu"),
            inference_mode=settings.get("inference_mode", "full"),
        )
        config["min_score"] = settings.get("min_score", 0.001)
        if config["inference"]["mode"] == "tiled":
            config["inference"]["tiling"] = adapter.validate_tiling_config(
                settings.get("tile_size", 640), settings.get("overlap", 0.2)
            )
        return adapter.validate_detector_config(config)

    monkeypatch.setattr(caches, "prepare_detector", prepare)
    monkeypatch.setattr(
        adapter,
        "_runtime",
        lambda architecture, mode: frozen_config(architecture, inference_mode=mode)["runtime"],
    )
    with TestClient(
        create_app(tmp_path / "workspace", run_jobs=False), base_url="http://127.0.0.1"
    ) as api:
        yield api


def sequence(client, tmp_path, project_id="default"):
    return publish_sequence(
        client, sequence_source(client, tmp_path, project_id=project_id), project_id
    )


def endpoint(sequence):
    return f"/api/temporal/sequences/{sequence['id']}/detection-caches"


def create(client, sequence, *, project_id="default", **settings):
    response = client.post(
        endpoint(sequence),
        params={"project_id": project_id},
        json={"name": "Synthetic local cache", "model_id": SSDLITE, **settings},
    )
    assert response.status_code == 201, response.text
    return response.json()


def execute(client, cache, *, stop_after=None, job_id=None):
    store = client.app.state.store
    identifier = job_id or cache["job_id"]
    store.update("jobs", identifier, {"status": "running", "started_at": now()})

    class Detector:
        def __init__(self, root, settings):
            self.metadata = runtime_metadata(settings)

        def warmup(self, image):
            pass

        def predict(self, image):
            return {
                "input_size": list(image.size),
                "detections": []
                if image.getpixel((0, 0))[0] == 13
                else [
                    {"label_id": 1, "label": "person", "score": 0.8, "box": [2, 3, 20, 40]},
                    {"label_id": 3, "label": "car", "score": 0.2, "box": [30, 5, 65, 50]},
                ],
                "timing": {
                    "preprocess_ms": 1,
                    "inference_ms": 2,
                    "postprocess_ms": 1,
                    "total_ms": 4,
                },
            }

    result = caches.run_detection_cache(
        store,
        identifier,
        lambda *_args: None,
        lambda: (
            stop_after is not None
            and len(store.list("temporal_detection_frames", cache_id=cache["id"])) >= stop_after
        ),
        detector_factory=Detector,
    )
    store.update(
        "jobs",
        identifier,
        {
            "status": "interrupted" if result["cancelled"] else "succeeded",
            "finished_at": now(),
            "result": result,
        },
    )
    return store.get("jobs", identifier)


def test_preview_is_read_only_and_creation_reuses_exact_cache_unless_explicitly_fresh(
    client, tmp_path
):
    source = sequence(client, tmp_path)
    path = endpoint(source)
    preview = client.post(path + "/preview", json={"model_id": SSDLITE})
    assert preview.status_code == 200, preview.text
    assert preview.json()["work"] == {
        "frames": 3,
        "forward_passes": 3,
        "warmup_forward_passes_per_attempt": 1,
        "source_gap_frames": 0,
    }
    assert preview.json()["existing_cache_id"] is None
    assert client.app.state.store.list("jobs") == []
    assert client.get(path).json() == []
    first = create(client, source)
    assert first["reused"] is False
    reused = client.post(path, json={"name": "Same recipe", "model_id": SSDLITE})
    assert reused.status_code == 200 and reused.json()["reused"] is True
    assert reused.json()["id"] == first["id"]
    assert len(client.app.state.store.list("jobs")) == 1
    current = client.post(path + "/preview", json={"model_id": SSDLITE}).json()
    assert current["existing_cache_id"] == first["id"]
    assert current["coverage"]["state"] == "empty"
    fresh = create(client, source, force_new=True)
    assert fresh["id"] != first["id"] and fresh["fingerprint"] != first["fingerprint"]
    assert len(client.get(path).json()) == 2
    assert len(client.app.state.store.list("jobs")) == 2


def test_completed_cache_reads_filter_without_reexecution_and_preserve_empty_frames(
    client, tmp_path, monkeypatch
):
    source = sequence(client, tmp_path)
    cache = create(client, source, min_score=0.1)
    assert execute(client, cache)["status"] == "succeeded"
    store = client.app.state.store
    original = deepcopy(store.list("temporal_detection_frames", cache_id=cache["id"]))
    path = f"/api/temporal/detection-caches/{cache['id']}"

    def unexpected(*args, **kwargs):
        pytest.fail("Saved cache reads must not prepare or execute a detector")

    monkeypatch.setattr(caches, "prepare_detector", unexpected)
    monkeypatch.setattr(adapter, "_runtime", unexpected)
    monkeypatch.setattr(adapter.models, "TorchvisionDetector", unexpected)
    complete = client.get(path + "/frames")
    assert complete.status_code == 200, complete.text
    frames = complete.json()["frames"]
    assert [frame["frame_index"] for frame in frames] == [0, 1, 2]
    assert [frame["timestamp_seconds"] for frame in frames] == [0, 0.1, 0.2]
    assert frames[1]["detections"] == []
    filtered = client.post(path + "/read", json={"min_score": 0.5, "class_ids": [1]})
    assert filtered.status_code == 200, filtered.text
    assert filtered.json()["result_sha256"] == complete.json()["result_sha256"]
    assert [len(frame["detections"]) for frame in filtered.json()["frames"]] == [1, 0, 1]
    assert filtered.json()["frames"][0]["detections"][0]["detection_index"] == 0
    assert client.post(path + "/read", json={"min_score": 0.05}).status_code == 409
    assert client.post(path + "/read", json={"class_ids": [1, 1]}).status_code == 409
    assert client.post(path + "/read", json={"class_ids": [99]}).status_code == 409
    assert client.get(path).json()["coverage"]["state"] == "complete"
    assert store.list("temporal_detection_frames", cache_id=cache["id"]) == original
    assert len(store.list("jobs")) == 1


def test_partial_cache_exposes_coverage_and_activity_then_common_recovery_continues(
    client, tmp_path
):
    source = sequence(client, tmp_path)
    cache = create(client, source)
    original_job = execute(client, cache, stop_after=1)
    path = f"/api/temporal/detection-caches/{cache['id']}"
    detail = client.get(path).json()
    assert detail["coverage"]["state"] == "partial"
    assert detail["coverage"]["remaining_count"] == 2
    assert client.get(path + "/frames").status_code == 409
    assert client.post(path + "/read", json={}).status_code == 409
    activity = client.get(f"/api/jobs/{original_job['id']}")
    assert activity.status_code == 200, activity.text
    assert activity.json()["context"]["session_id"] == source["manifest"]["asset"]["session_id"]
    assert activity.json()["context"]["name"] == cache["name"]
    assert activity.json()["recovery"]["can_check"] is True
    assert activity.json()["next_action"]["workspace"] == "tracking"
    assert activity.json()["context"]["sequence_id"] == source["id"]
    assert {row["kind"]: row["count"] for row in activity.json()["artifacts"]} == {
        "temporal_detection_frames": 1,
        "temporal_detection_attempt_frames": 1,
    }
    preview = client.get(f"/api/jobs/{original_job['id']}/recovery")
    assert preview.status_code == 200 and preview.json()["available"], preview.text
    response = client.post(
        f"/api/jobs/{original_job['id']}/recover",
        json={"fingerprint": preview.json()["fingerprint"]},
    )
    assert response.status_code == 202, response.text
    child = response.json()
    assert child["params"]["recovery_of"] == original_job["id"]
    execute(client, cache, job_id=child["id"])
    assert client.get(path).json()["coverage"]["state"] == "complete"
    assert client.app.state.store.get("jobs", original_job["id"]) == original_job
    activity = client.get(f"/api/jobs/{child['id']}").json()
    assert activity["lineage"]["parent_job_id"] == original_job["id"]
    assert {row["kind"]: row["count"] for row in activity["artifacts"]} == {
        "temporal_detection_frames": 3,
        "temporal_detection_attempt_frames": 2,
    }


def test_cache_results_and_jobs_remain_owned_by_the_sequence_project(client, tmp_path):
    project = client.post("/api/projects", json={"name": "Temporal other"}).json()["id"]
    source = sequence(client, tmp_path, project)
    cache = create(client, source, project_id=project)
    execute(client, cache)
    path = f"/api/temporal/detection-caches/{cache['id']}"
    for url in (endpoint(source), path, path + "/frames", f"/api/jobs/{cache['job_id']}"):
        assert client.get(url, params={"project_id": project}).status_code == 200
        assert client.get(url).status_code == 404
    assert client.get("/api/jobs").json() == []
    assert len(client.get("/api/jobs", params={"project_id": project}).json()) == 1
    assert (
        client.post(endpoint(source), json={"model_id": SSDLITE, "name": "Foreign"}).status_code
        == 404
    )
    assert client.post(endpoint(source) + "/preview", json={"model_id": SSDLITE}).status_code == 404
    assert client.post(path + "/read", json={}).status_code == 404
    assert (
        client.post(
            f"/api/jobs/{cache['job_id']}/recover", json={"fingerprint": "a" * 64}
        ).status_code
        == 404
    )
    store = client.app.state.store
    row = store.list("temporal_detection_frames", cache_id=cache["id"])[0]
    assert record_project(store, "temporal_detection_frames", row) == project
    # Reverse ownership also serves old/imported job receipts without params.
    store.update("jobs", cache["job_id"], {"params": {}})
    assert (
        client.get(f"/api/jobs/{cache['job_id']}", params={"project_id": project}).status_code
        == 200
    )
    assert client.get(f"/api/jobs/{cache['job_id']}").status_code == 404


def test_cross_project_trained_model_is_rejected_before_detector_preparation(
    client, tmp_path, monkeypatch
):
    source = sequence(client, tmp_path)
    other = client.post("/api/projects", json={"name": "Foreign training"}).json()["id"]
    store = client.app.state.store
    dataset_id, job_id, training_id, model_id = (new_id() for _ in range(4))
    store.insert(
        "dataset_versions",
        {
            "id": dataset_id,
            "project_id": other,
            "name": "Synthetic foreign dataset",
            "path": "datasets/unused/manifest.json",
            "manifest_sha256": "a" * 64,
            "summary": {},
            "created_at": now(),
        },
    )
    store.insert(
        "jobs",
        {
            "id": job_id,
            "kind": "train",
            "status": "succeeded",
            "params": {"training_id": training_id},
            "created_at": now(),
        },
    )
    store.insert(
        "training_runs",
        {
            "id": training_id,
            "name": "Synthetic foreign training",
            "dataset_id": dataset_id,
            "parent_model_id": SSDLITE,
            "config": {},
            "job_id": job_id,
            "created_at": now(),
        },
    )
    store.insert(
        "trained_models",
        {
            "id": model_id,
            "name": "Synthetic foreign detector",
            "training_id": training_id,
            "parent_model_id": SSDLITE,
            "architecture": SSDLITE,
            "path": "models/fixture.pth",
            "weight_sha256": "a" * 64,
            "metadata": {},
            "created_at": now(),
        },
    )
    monkeypatch.setattr(
        caches, "prepare_detector", lambda *a, **k: pytest.fail("Foreign model loaded")
    )
    for suffix, body in (("/preview", {}), ("", {"name": "Rejected"})):
        response = client.post(endpoint(source) + suffix, json={"model_id": model_id, **body})
        assert response.status_code == 404
    assert store.list("temporal_detection_caches") == []


@pytest.mark.parametrize(
    "change",
    [
        {"project_id": "default"},
        {"model_id": "x" * 129},
        {"name": "x" * 161},
        {"device": "cuda:0"},
        {"inference_mode": "paired"},
        {"tile_size": True},
        {"tile_size": 127},
        {"overlap": 0.6},
        {"min_score": 0.0001},
        {"min_score": True},
        {"min_score": "0.1"},
        {"force_new": 1},
        {"class_ids": [1]},
    ],
)
def test_creation_contract_rejects_extra_fields_coercion_and_out_of_range_values(client, change):
    response = client.post(
        "/api/temporal/sequences/missing/detection-caches",
        json={"model_id": SSDLITE, "name": "Cache", **change},
    )
    assert response.status_code == 422
    assert client.app.state.store.list("jobs") == []


@pytest.mark.parametrize(
    "body",
    [
        {"class_ids": []},
        {"class_ids": [True]},
        {"class_ids": ["1"]},
        {"class_ids": [1] * 101},
        {"min_score": 0.0001},
        {"min_score": True},
        {"model_id": SSDLITE},
    ],
)
def test_cache_read_filter_is_bounded_and_strict(client, body):
    assert client.post("/api/temporal/detection-caches/missing/read", json=body).status_code == 422


def test_preview_rejects_creation_fields_and_media_errors_do_not_expose_local_paths(
    client, tmp_path, monkeypatch
):
    source = sequence(client, tmp_path)
    for field, value in (("name", "No creation"), ("force_new", True)):
        response = client.post(
            endpoint(source) + "/preview", json={"model_id": SSDLITE, field: value}
        )
        assert response.status_code == 422

    def missing(*args, **kwargs):
        raise FileNotFoundError("/private/workspace/secret/model.pth")

    monkeypatch.setattr(caches, "prepare_detector", missing)
    response = client.post(endpoint(source) + "/preview", json={"model_id": SSDLITE})
    assert response.status_code == 409
    assert "/private" not in response.text
    assert client.app.state.store.list("jobs") == []
