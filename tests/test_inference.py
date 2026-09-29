"""Comparison persistence and failure semantics with explicitly synthetic detectors."""

import sqlite3
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from iris.app import create_app
from iris.jobs import JobManager
from iris.media import import_asset
from iris.store import Store, new_id, now

BASE_URL = "http://127.0.0.1"
MODEL_IDS = ["ssdlite320_mobilenet_v3_large", "fasterrcnn_mobilenet_v3_large_320_fpn"]
TIMING = {"preprocess_ms": 1.25, "inference_ms": 2.5, "postprocess_ms": 0.75, "total_ms": 4.5}
DETECTIONS = [
    {"box": [2.25, 3.5, 20.75, 19.5], "label_id": 1, "label": "person", "score": 0.0125},
    {"box": [0.0, 0.0, 32.0, 24.0], "label_id": 3, "label": "car", "score": 0.875},
]

# This frozen historical schema ensures the migration test starts with an actual
# first-increment database, independently of future additions to Store.SCHEMA.
V1_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, scene_group TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS assets (
    id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id), filename TEXT NOT NULL,
    kind TEXT NOT NULL CHECK(kind IN ('image','video')), sha256 TEXT NOT NULL,
    size_bytes INTEGER NOT NULL, path TEXT NOT NULL, metadata TEXT NOT NULL,
    created_at TEXT NOT NULL, UNIQUE(session_id,sha256)
);
CREATE TABLE IF NOT EXISTS frames (
    id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id),
    asset_id TEXT NOT NULL REFERENCES assets(id), frame_index INTEGER, timestamp_seconds REAL,
    width INTEGER NOT NULL, height INTEGER NOT NULL, sha256 TEXT NOT NULL,
    perceptual_hash TEXT NOT NULL, path TEXT NOT NULL,
    selected INTEGER NOT NULL DEFAULT 0 CHECK(selected IN (0,1)),
    extraction TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL,
    UNIQUE(asset_id,frame_index)
);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY, kind TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN
      ('queued','running','succeeded','failed','cancelled','interrupted')),
    params TEXT NOT NULL, result TEXT, progress REAL NOT NULL DEFAULT 0,
    message TEXT NOT NULL DEFAULT '', logs TEXT NOT NULL DEFAULT '[]', error TEXT,
    created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT,
    cancel_requested INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS frames_session ON frames(session_id);
CREATE INDEX IF NOT EXISTS frames_hash ON frames(sha256);
CREATE INDEX IF NOT EXISTS assets_session ON assets(session_id);
CREATE INDEX IF NOT EXISTS jobs_status ON jobs(status);
"""


def make_session(store, name="Synthetic flight"):
    return store.insert(
        "sessions",
        {"id": new_id(), "name": name, "scene_group": name, "created_at": now()},
    )


def make_frame(store, session_id, value):
    source = store.root / f"fixture-{value}.png"
    Image.new("RGB", (32, 24), (value, 70, 110)).save(source)
    asset = import_asset(store, session_id, source, source.name)
    source.unlink()
    (frame,) = store.list("frames", asset_id=asset["id"])
    return store.update("frames", frame["id"], {"selected": True})


@pytest.fixture
def workspace(tmp_path):
    store = Store(tmp_path / "workspace")
    flight = make_session(store)
    frames = [make_frame(store, flight["id"], value) for value in (10, 20, 30)]
    return store, flight, frames


@pytest.fixture
def ready_catalog(monkeypatch):
    from iris import inference

    value = [{"id": model_id, "status": "ready", "reason": None} for model_id in MODEL_IDS]
    monkeypatch.setattr(inference, "catalog", lambda _root: value)
    return value


def queue_comparison(store, flight, frames, *, model_ids=None):
    from iris.inference import create_comparison

    return create_comparison(
        store,
        JobManager(store),
        flight["id"],
        name="Synthetic comparison",
        frame_ids=[frame["id"] for frame in frames],
        model_ids=model_ids or MODEL_IDS,
        device="cpu",
    )


def synthetic_factory(calls, *, empty=False, fail_at=None):
    class SyntheticDetector:
        def __init__(self, root, model_id, *, device):
            assert isinstance(root, Path)
            self.model_id = model_id
            self.metadata = {
                "model_id": model_id,
                "device": device,
                "checkpoint": "synthetic-test-only",
                "checkpoint_sha256": "a" * 64,
                "runtime": {"fixture": True},
            }
            calls.append(("create", model_id, device))

        def warmup(self, image):
            assert image.mode == "RGB"
            calls.append(("warmup", self.model_id, image.getpixel((0, 0))))

        def predict(self, image):
            assert image.mode == "RGB"
            if fail_at is not None and sum(call[0] == "predict" for call in calls) == fail_at:
                raise RuntimeError("Synthetic detector failure")
            calls.append(("predict", self.model_id, image.getpixel((0, 0))))
            return {
                "detections": [] if empty else DETECTIONS,
                "timing": TIMING,
                "input_size": list(image.size),
            }

    return SyntheticDetector


def test_v1_migration_preserves_sources_selection_and_job_outcomes(tmp_path):
    root = tmp_path / "old-workspace"
    root.mkdir()
    db = root / "iris.sqlite3"
    created = "2026-09-29T08:00:00+00:00"
    with sqlite3.connect(db) as connection:
        connection.executescript(V1_SCHEMA)
        connection.execute("PRAGMA user_version=1")
        connection.execute(
            "INSERT INTO sessions VALUES (?,?,?,?)", ("flight", "Existing flight", "field", created)
        )
        connection.execute(
            "INSERT INTO assets VALUES (?,?,?,?,?,?,?,?,?)",
            (
                "asset",
                "flight",
                "original.png",
                "image",
                "source-hash",
                17,
                "assets/source",
                "{}",
                created,
            ),
        )
        connection.execute(
            "INSERT INTO frames VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "frame",
                "flight",
                "asset",
                None,
                None,
                32,
                24,
                "pixel-hash",
                "0000",
                "frames/old.png",
                1,
                "{}",
                created,
            ),
        )
        connection.execute(
            "INSERT INTO jobs (id,kind,status,params,result,created_at,finished_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (
                "job",
                "extract",
                "succeeded",
                '{"asset_id":"asset"}',
                '{"created":1}',
                created,
                created,
            ),
        )
        before = {
            table: connection.execute(f"SELECT * FROM {table}").fetchall()
            for table in ("sessions", "assets", "frames", "jobs")
        }
    migrated = Store(root)
    with migrated.connect() as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 4
        after = {
            table: [tuple(row) for row in connection.execute(f"SELECT * FROM {table}")]
            for table in before
        }
    assert after == before
    assert migrated.get("frames", "frame")["selected"] is True
    assert migrated.get("jobs", "job")["result"] == {"created": 1}
    for table in ("comparisons", "runs", "predictions"):
        assert migrated.list(table) == []


def test_comparison_freezes_selected_frame_ids_hashes_and_settings(workspace, ready_catalog):
    store, flight, frames = workspace
    comparison = queue_comparison(store, flight, [frames[1], frames[0]])
    store.update("frames", frames[0]["id"], {"selected": False})
    store.update("frames", frames[2]["id"], {"selected": True})
    persisted = Store(store.root).get("comparisons", comparison["id"])
    assert persisted["frame_ids"] == [frames[1]["id"], frames[0]["id"]]
    assert persisted["model_ids"] == MODEL_IDS
    assert persisted["config"]["frame_hashes"] == {
        frame["id"]: frame["sha256"] for frame in frames[:2]
    }
    assert persisted["config"]["device"] == "cpu"
    assert persisted["config"]["warmup"] == 1
    assert persisted["config"]["taxonomy"] == "coco-2017-v1"
    assert persisted["config"]["class_mapping"] == {"person": 1, "car": 3}
    job = store.get("jobs", persisted["job_id"])
    assert job["status"] == "queued"
    assert job["params"]["comparison_id"] == persisted["id"]


def test_both_models_process_same_snapshot_and_preserve_raw_outputs(workspace, ready_catalog):
    from iris.inference import run_comparison

    store, flight, frames = workspace
    comparison = queue_comparison(store, flight, frames[:2])
    store.update("frames", frames[0]["id"], {"selected": False})
    calls = []
    updates = []
    result = run_comparison(
        store,
        comparison["id"],
        lambda value, message: updates.append((value, message)),
        lambda: False,
        detector_factory=synthetic_factory(calls),
    )
    assert result == {
        "comparison_id": comparison["id"],
        "frames_total": 2,
        "models_total": 2,
        "predictions_created": 4,
        "cancelled": False,
    }
    assert updates and updates[-1][0] == 1
    assert [value for value, _message in updates] == sorted(value for value, _message in updates)
    for model_id in MODEL_IDS:
        assert [call[2] for call in calls if call[:2] == ("predict", model_id)] == [
            (10, 70, 110),
            (20, 70, 110),
        ]
        assert sum(call[:2] == ("warmup", model_id) for call in calls) == 1
    reopened = Store(store.root)
    runs = reopened.list("runs", comparison_id=comparison["id"])
    predictions = reopened.list("predictions", comparison_id=comparison["id"])
    assert len(runs) == 2 and len(predictions) == 4
    assert {run["model_id"] for run in runs} == set(MODEL_IDS)
    for run in runs:
        assert run["metadata"]["checkpoint_sha256"] == "a" * 64
        assert run["metadata"]["runtime"] == {"fixture": True}
        rows = [prediction for prediction in predictions if prediction["run_id"] == run["id"]]
        assert {row["frame_id"] for row in rows} == {frame["id"] for frame in frames[:2]}
        for row in rows:
            assert row["model_id"] == run["model_id"]
            assert row["detections"] == DETECTIONS
            for phase in ("preprocess_ms", "inference_ms", "postprocess_ms"):
                assert row["timing"][phase] == TIMING[phase]
            assert row["timing"]["total_ms"] >= row["timing"]["decode_ms"] >= 0
            assert row["input_size"] == [32, 24]


def test_cancellation_keeps_explicit_empty_predictions_and_missing_work_distinct(
    workspace, ready_catalog
):
    from iris.inference import comparison_detail, run_comparison

    store, flight, frames = workspace
    comparison = queue_comparison(store, flight, frames)
    cancelled = False

    def progress(_value, _message):
        nonlocal cancelled
        if store.list("predictions", comparison_id=comparison["id"]):
            cancelled = True

    result = run_comparison(
        store,
        comparison["id"],
        progress,
        lambda: cancelled,
        detector_factory=synthetic_factory([], empty=True),
    )
    assert result["cancelled"] is True
    assert result["predictions_created"] == 1
    detail = comparison_detail(store, comparison["id"])
    assert len(detail["predictions"]) == 1
    assert detail["predictions"][0]["detections"] == []
    assert {row["frame_id"] for row in detail["predictions"]} == {frames[0]["id"]}
    assert detail["job"]["status"] != "succeeded"


def test_detector_failure_preserves_completed_predictions(workspace, ready_catalog):
    from iris.inference import run_comparison

    store, flight, frames = workspace
    comparison = queue_comparison(store, flight, frames)
    with pytest.raises(RuntimeError, match="Synthetic detector failure"):
        run_comparison(
            store,
            comparison["id"],
            lambda _value, _message: None,
            lambda: False,
            detector_factory=synthetic_factory([], fail_at=1),
        )
    assert len(store.list("predictions", comparison_id=comparison["id"])) == 1
    assert store.get("jobs", comparison["job_id"])["status"] != "succeeded"


def test_cancelled_before_start_never_initializes_a_model(workspace, ready_catalog):
    from iris.inference import run_comparison

    store, flight, frames = workspace
    comparison = queue_comparison(store, flight, frames)
    calls = []
    result = run_comparison(
        store,
        comparison["id"],
        lambda _value, _message: None,
        lambda: True,
        detector_factory=synthetic_factory(calls),
    )
    assert result["cancelled"] is True
    assert result["predictions_created"] == 0
    assert calls == []
    assert store.list("runs", comparison_id=comparison["id"]) == []


def test_saved_comparison_cannot_be_overwritten_by_reexecution(workspace, ready_catalog):
    from iris.inference import run_comparison

    store, flight, frames = workspace
    comparison = queue_comparison(store, flight, frames[:1])
    arguments = (store, comparison["id"], lambda _value, _message: None, lambda: False)
    run_comparison(*arguments, detector_factory=synthetic_factory([]))
    before = store.list("predictions", comparison_id=comparison["id"])
    with pytest.raises(ValueError, match="immutable"):
        run_comparison(*arguments, detector_factory=synthetic_factory([], empty=True))
    assert store.list("predictions", comparison_id=comparison["id"]) == before


@pytest.mark.parametrize(
    "invalid_prediction",
    [
        {"input_size": [24, 32]},
        {"detections": [{**DETECTIONS[0], "box": [-1, 1, 20, 21]}]},
        {"detections": [{**DETECTIONS[0], "box": [3, 1, 2, 21]}]},
        {"detections": [{**DETECTIONS[0], "box": [2, 1, 2, 21]}]},
        {"detections": [{**DETECTIONS[0], "box": [2, 1, 40, 21]}]},
        {"detections": [{**DETECTIONS[0], "box": [2, 1, float("nan"), 21]}]},
        {"detections": [{**DETECTIONS[0], "score": 1.01}]},
        {"timing": {**TIMING, "inference_ms": float("inf")}},
    ],
)
def test_invalid_detector_outputs_are_never_persisted(workspace, ready_catalog, invalid_prediction):
    from iris.inference import run_comparison

    store, flight, frames = workspace
    comparison = queue_comparison(store, flight, frames[:1])

    class InvalidDetector(synthetic_factory([])):
        def predict(self, image):
            return {**super().predict(image), **invalid_prediction}

    with pytest.raises(ValueError):
        run_comparison(
            store,
            comparison["id"],
            lambda _value, _message: None,
            lambda: False,
            detector_factory=InvalidDetector,
        )
    assert store.list("predictions", comparison_id=comparison["id"]) == []


@pytest.mark.parametrize("change", ["pixels", "dimensions", "missing"])
def test_changed_or_missing_frame_never_produces_predictions(workspace, ready_catalog, change):
    from iris.inference import run_comparison

    store, flight, frames = workspace
    comparison = queue_comparison(store, flight, frames[:1])
    path = store.artifact_path(frames[0]["path"])
    if change == "missing":
        path.unlink()
    else:
        Image.new("RGB", (32, 24) if change == "pixels" else (16, 12), "red").save(path)
    calls = []
    with pytest.raises((ValueError, RuntimeError, OSError)):
        run_comparison(
            store,
            comparison["id"],
            lambda _value, _message: None,
            lambda: False,
            detector_factory=synthetic_factory(calls),
        )
    assert not any(call[0] == "predict" for call in calls)
    assert store.list("predictions", comparison_id=comparison["id"]) == []


@pytest.mark.parametrize(
    "override",
    [
        {"name": " "},
        {"name": "x" * 161},
        {"frame_ids": []},
        {"frame_ids": ["missing"]},
        {"frame_ids": [str(number) for number in range(101)]},
        {"model_ids": []},
        {"model_ids": ["unknown-model"]},
        {"model_ids": [MODEL_IDS[0], MODEL_IDS[0]]},
        {"model_ids": MODEL_IDS + [MODEL_IDS[0]]},
        {"device": "auto"},
        {"device": True},
        {"unexpected": True},
    ],
)
def test_invalid_api_requests_never_create_comparisons_or_jobs(workspace, ready_catalog, override):
    store, flight, frames = workspace
    payload = {
        "name": "Synthetic comparison",
        "frame_ids": [frames[0]["id"]],
        "model_ids": MODEL_IDS,
        "device": "cpu",
        **override,
    }
    with TestClient(create_app(store.root, run_jobs=False), base_url=BASE_URL) as client:
        response = client.post(f"/api/sessions/{flight['id']}/comparisons", json=payload)
        assert response.status_code == 422, response.text
    assert store.list("comparisons") == []
    assert store.list("jobs") == []


@pytest.mark.parametrize("invalid_frame", ["foreign", "duplicate"])
def test_frame_validation_rejects_ambiguous_inputs(workspace, ready_catalog, invalid_frame):
    store, flight, frames = workspace
    selected = frames[:1]
    if invalid_frame == "foreign":
        other_flight = make_session(store, "Other scene")
        selected = [make_frame(store, other_flight["id"], 40)]
    else:
        selected *= 2
    with pytest.raises(ValueError):
        queue_comparison(store, flight, selected)
    assert store.list("comparisons") == []
    assert store.list("jobs") == []


def test_unavailable_model_is_visible_and_never_queued(workspace, ready_catalog):
    store, flight, frames = workspace
    ready_catalog[1].update(status="missing_weights", reason="Local weights are missing")
    with TestClient(create_app(store.root, run_jobs=False), base_url=BASE_URL) as client:
        response = client.post(
            f"/api/sessions/{flight['id']}/comparisons",
            json={
                "name": "Synthetic comparison",
                "frame_ids": [frames[0]["id"]],
                "model_ids": MODEL_IDS,
                "device": "cpu",
            },
        )
        assert response.status_code == 409, response.text
        assert "missing" in response.json()["detail"].lower()
    assert store.list("comparisons") == []
    assert store.list("jobs") == []


def test_api_comparison_keeps_provenance_and_hides_local_artifact_paths(workspace, ready_catalog):
    from iris.inference import run_comparison

    store, flight, frames = workspace
    with TestClient(create_app(store.root, run_jobs=False), base_url=BASE_URL) as client:
        response = client.post(
            f"/api/sessions/{flight['id']}/comparisons",
            json={
                "name": "  Compare fixture  ",
                "frame_ids": [frames[0]["id"]],
                "model_ids": [MODEL_IDS[0]],
                "device": "cpu",
            },
        )
        assert response.status_code == 202, response.text
        submitted = response.json()
        comparison = submitted
        assert comparison["name"] == "Compare fixture"
        run_comparison(
            store,
            comparison["id"],
            lambda _value, _message: None,
            lambda: False,
            detector_factory=synthetic_factory([]),
        )
        response = client.get(f"/api/comparisons/{comparison['id']}")
        assert response.status_code == 200, response.text
        detail = response.json()
        assert detail["id"] == comparison["id"]
        assert detail["job"]["id"] == comparison["job_id"]
        assert detail["frames"][0]["asset_id"] == frames[0]["asset_id"]
        assert detail["frames"][0]["sha256"] == frames[0]["sha256"]
        assert "path" not in detail["frames"][0]
        assert detail["predictions"][0]["detections"] == DETECTIONS
        assert len(client.get(f"/api/sessions/{flight['id']}/comparisons").json()) == 1
        assert client.get("/api/comparisons/missing").status_code == 404
        assert client.get("/api/sessions/missing/comparisons").status_code == 404
    with TestClient(create_app(store.root, run_jobs=False), base_url=BASE_URL) as reopened:
        assert reopened.get(f"/api/comparisons/{comparison['id']}").json() == detail


def test_queued_comparison_does_not_break_extraction_submission(workspace, ready_catalog):
    store, flight, frames = workspace
    comparison = queue_comparison(store, flight, frames[:1])
    job = JobManager(store).submit(frames[0]["asset_id"], {"max_frames": 1})
    assert job["kind"] == "extract"
    assert job["status"] == "queued"
    assert job["id"] != comparison["job_id"]


def test_real_worker_records_missing_checkpoint_as_failure(workspace, ready_catalog):
    store, flight, frames = workspace
    # Readiness can disappear between queue submission and worker execution.
    # Only that submit-time check is patched; the worker has no test detector,
    # no model files, and uses the actual offline production loader.
    with TestClient(create_app(store.root), base_url=BASE_URL) as client:
        response = client.post(
            f"/api/sessions/{flight['id']}/comparisons",
            json={
                "name": "Unavailable runtime fixture",
                "frame_ids": [frames[0]["id"]],
                "model_ids": [MODEL_IDS[0]],
                "device": "cpu",
            },
        )
        assert response.status_code == 202, response.text
        submitted = response.json()
        comparison = submitted
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            current = store.get("jobs", comparison["job_id"])
            if current["status"] not in {"queued", "running"}:
                break
            time.sleep(0.05)
        else:
            pytest.fail(f"Worker did not finish within 20 seconds: {current}")
        assert current["status"] == "failed", current
        assert current["error"] and current["finished_at"]
        assert current["progress"] < 1
        assert store.list("predictions", comparison_id=comparison["id"]) == []
        log = client.get(f"/api/jobs/{current['id']}/log")
        assert log.status_code == 200
        assert current["error"] in log.text
