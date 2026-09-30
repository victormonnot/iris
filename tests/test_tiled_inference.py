"""Tiled comparison persistence and API behavior with synthetic detections only."""

import copy
import re
import sqlite3

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from iris.app import create_app
from iris.inference import (
    comparison_detail,
    comparison_lanes,
    create_comparison,
    preview_comparison,
    run_comparison,
)
from iris.jobs import JobManager
from iris.media import import_asset
from iris.store import SCHEMA, Store, _encode, new_id, now

MODEL = "ssdlite320_mobilenet_v3_large"
OTHER = "fasterrcnn_mobilenet_v3_large_320_fpn"


@pytest.fixture
def example(tmp_path, monkeypatch):
    store = Store(tmp_path / "workspace")
    session = store.insert(
        "sessions",
        {
            "id": new_id(),
            "name": "Synthetic tiles",
            "scene_group": "fixture",
            "created_at": now(),
        },
    )
    path = tmp_path / "synthetic.png"
    Image.new("RGB", (384, 256), (20, 40, 60)).save(path)
    asset = import_asset(store, session["id"], path, path.name)
    frame = store.list("frames", asset_id=asset["id"])[0]
    monkeypatch.setattr(
        "iris.inference.catalog",
        lambda root: [
            {"id": model_id, "status": "ready", "weight_sha256": "a" * 64}
            for model_id in (MODEL, OTHER)
        ],
    )
    payload = {
        "name": "Paired fixture",
        "frame_ids": [frame["id"]],
        "model_ids": [MODEL],
        "inference_mode": "paired",
        "tile_size": 256,
        "overlap": 0.2,
    }
    return store, session, frame, payload


def factory(calls, after_predict=None):
    class Detector:
        def __init__(self, root, model_id, device):
            self.metadata = {"model_id": model_id, "weight_sha256": "a" * 64, "fixture": True}
            calls.append(("load", model_id))

        def warmup(self, image):
            calls.append(("warmup", image.size))

        def predict(self, image):
            calls.append(("predict", image.size))
            if after_predict:
                after_predict(calls)
            return {
                "detections": [
                    {
                        "box": [1, 2, 30, 40],
                        "label_id": 3,
                        "native_label_id": 2,
                        "label": "car",
                        "score": 0.8,
                    }
                ],
                "input_size": list(image.size),
                "timing": {
                    "preprocess_ms": 1,
                    "inference_ms": 2,
                    "postprocess_ms": 3,
                    "total_ms": 6,
                },
            }

    return Detector


def queue(example, **overrides):
    store, session, _frame, payload = example
    return create_comparison(store, JobManager(store), session["id"], **(payload | overrides))


def execute(example, comparison, calls, **kwargs):
    return run_comparison(
        example[0],
        comparison["id"],
        kwargs.pop("progress", lambda *args: None),
        kwargs.pop("cancelled", lambda: False),
        detector_factory=factory(calls, **kwargs),
    )


def test_preview_counts_passes_and_warmups_without_writes_or_model_loading(example, monkeypatch):
    store, session, frame, payload = example
    monkeypatch.setattr("iris.inference.catalog", lambda root: pytest.fail("No weights in preview"))
    plan = preview_comparison(store, session["id"], **payload)
    assert plan["lanes"] == [{"model_id": MODEL, "variant": value} for value in ("full", "tiled")]
    assert plan["forward_passes"] == 3
    assert plan["warmup_passes"] == 2
    assert plan["total_forward_passes"] == 5
    assert plan["tiles"] == [{"frame_id": frame["id"], "input_size": [384, 256], "tile_count": 2}]
    assert not store.list("jobs") and not store.list("comparisons")


def test_paired_run_loads_one_checkpoint_and_saves_original_coordinates_and_raw_tiles(example):
    store, _session, frame, _payload = example
    comparison = queue(example)
    assert [lane["run_id"] for lane in comparison["lanes"]] == [None, None]
    calls, progress = [], []
    result = execute(
        example, comparison, calls, progress=lambda value, message: progress.append(value)
    )
    assert result["predictions_created"] == 2
    assert not result["cancelled"]
    assert [call for call in calls if call[0] == "load"] == [("load", MODEL)]
    assert [call[1] for call in calls if call[0] == "warmup"] == [(384, 256), (256, 256)]
    assert [call[1] for call in calls if call[0] == "predict"] == [
        (384, 256),
        (256, 256),
        (256, 256),
    ]
    assert progress == sorted(progress) and progress[-1] == 1
    detail = comparison_detail(Store(store.root), comparison["id"])
    assert detail["model_ids"] == [MODEL]
    assert len({lane["run_id"] for lane in detail["lanes"]}) == 2
    runs = {run["variant"]: run for run in detail["runs"]}
    assert runs["tiled"]["metadata"]["inference"]["tile_boxes"][frame["id"]] == [
        [0, 0, 256, 256],
        [128, 0, 384, 256],
    ]
    predictions = {
        run["variant"]: next(pred for pred in detail["predictions"] if pred["run_id"] == run["id"])
        for run in detail["runs"]
    }
    full, tiled = predictions["full"], predictions["tiled"]
    assert full["input_size"] == tiled["input_size"] == [384, 256]
    assert [box["box"] for box in tiled["detections"]] == [[1, 2, 30, 40], [129, 2, 158, 40]]
    assert all(box["native_label_id"] == 2 and box["label_id"] == 3 for box in tiled["detections"])
    assert full["timing"]["forward_passes"] == 1
    assert tiled["timing"]["forward_passes"] == 2
    assert tiled["timing"]["inference_ms"] == 4
    assert tiled["timing"]["total_ms"] >= tiled["timing"]["decode_ms"] >= 0
    assert len(tiled["metadata"]["tiles"]) == 2
    assert tiled["metadata"]["tiles"][1]["detections"][0]["box"] == [1, 2, 30, 40]
    assert not store.list("annotation_revisions") and not store.list("annotation_suggestions")


def test_two_checkpoints_can_both_run_tiled_without_identity_aliases(example):
    comparison = queue(example, inference_mode="tiled", model_ids=[MODEL, OTHER])
    calls = []
    assert execute(example, comparison, calls)["predictions_created"] == 2
    detail = comparison_detail(example[0], comparison["id"])
    assert [lane["model_id"] for lane in detail["lanes"]] == [MODEL, OTHER]
    assert all(run["variant"] == "tiled" for run in detail["runs"])
    assert len([call for call in calls if call[0] == "predict"]) == 4


@pytest.mark.parametrize("failure", ["cancel", "error"])
def test_partial_tile_image_is_never_published_but_completed_full_image_survives(example, failure):
    comparison = queue(example)
    cancelled = False

    def after_predict(calls):
        nonlocal cancelled
        if len([call for call in calls if call[0] == "predict"]) == 2:
            if failure == "error":
                raise RuntimeError("Synthetic interrupted tile")
            cancelled = True

    if failure == "error":
        with pytest.raises(RuntimeError, match="interrupted tile"):
            execute(example, comparison, [], after_predict=after_predict)
    else:
        result = execute(
            example, comparison, [], after_predict=after_predict, cancelled=lambda: cancelled
        )
        assert result["cancelled"] is True and result["predictions_created"] == 1
    detail = comparison_detail(example[0], comparison["id"])
    assert len(detail["predictions"]) == 1
    assert detail["predictions"][0]["run_id"] == detail["lanes"][0]["run_id"]
    assert detail["lanes"][1]["run_id"] is not None


@pytest.mark.parametrize(
    "overrides",
    [
        {"model_ids": [MODEL, OTHER]},
        {"inference_mode": "unknown"},
        {"tile_size": True},
        {"tile_size": 127},
        {"tile_size": 2049},
        {"overlap": True},
        {"overlap": -0.1},
        {"overlap": 0.51},
    ],
)
def test_api_rejects_invalid_plan_before_publishing_a_job(example, overrides):
    store, session, _frame, payload = example
    with TestClient(create_app(store.root, run_jobs=False), base_url="http://127.0.0.1") as client:
        for suffix in ("/preview", ""):
            response = client.post(
                f"/api/sessions/{session['id']}/comparisons{suffix}", json=payload | overrides
            )
            assert response.status_code == 422, response.text
    assert not store.list("jobs") and not store.list("comparisons")


@pytest.mark.parametrize("limit", ["frame", "total"])
def test_work_limits_reject_large_selections_without_loading_images(example, limit):
    store, session, frame, payload = example
    if limit == "frame":
        store.update("frames", frame["id"], {"width": 100000, "height": 100000})
        frame_ids = [frame["id"]]
    else:
        frame_ids = []
        for index in range(9):
            clone = {**frame, "id": new_id(), "frame_index": index, "width": 1024, "height": 1024}
            store.insert("frames", clone)
            frame_ids.append(clone["id"])
    with pytest.raises(ValueError, match="limit"):
        preview_comparison(
            store,
            session["id"],
            **(
                payload
                | {
                    "frame_ids": frame_ids,
                    "tile_size": 128,
                    "overlap": 0,
                }
            ),
        )
    assert not store.list("jobs")


@pytest.mark.parametrize("changed", ["dimensions", "plan", "protocol"])
def test_worker_revalidates_frozen_plan_before_loading_weights(example, changed):
    store, _session, frame, _payload = example
    comparison = queue(example)
    if changed == "dimensions":
        store.update("frames", frame["id"], {"width": 385})
    else:
        config = copy.deepcopy(comparison["config"])
        if changed == "plan":
            config["lanes"] = [{"model_id": MODEL, "variant": "full"}]
        else:
            config["inference"]["tiling"]["merge_iou"] = 0.8
        store.update("comparisons", comparison["id"], {"config": config})
    calls = []
    with pytest.raises(ValueError):
        execute(example, comparison, calls)
    assert calls == [] and not store.list("runs")


@pytest.mark.parametrize("config", [None, [], {"inference": []}, {"inference": {"mode": []}}])
def test_saved_lane_validation_rejects_malformed_configuration(config):
    with pytest.raises(ValueError):
        comparison_lanes({"model_ids": [MODEL], "config": config})


OLD_RUNS = """CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY, comparison_id TEXT NOT NULL REFERENCES comparisons(id),
    model_id TEXT NOT NULL, metadata TEXT NOT NULL, created_at TEXT NOT NULL,
    UNIQUE(comparison_id, model_id)
);"""
OLD_PREDICTIONS = """CREATE TABLE IF NOT EXISTS predictions (
    id TEXT PRIMARY KEY, comparison_id TEXT NOT NULL REFERENCES comparisons(id),
    run_id TEXT NOT NULL REFERENCES runs(id), frame_id TEXT NOT NULL REFERENCES frames(id),
    model_id TEXT NOT NULL, detections TEXT NOT NULL, timing TEXT NOT NULL,
    input_size TEXT NOT NULL, created_at TEXT NOT NULL, UNIQUE(run_id, frame_id)
);"""


def historical_database(example, path, *, broken=False):
    store = example[0]
    comparison = queue(example, inference_mode="full")
    execute(example, comparison, [])
    schema = re.sub(r"CREATE TABLE IF NOT EXISTS runs \(.*?\);", OLD_RUNS, SCHEMA, flags=re.S)
    schema = re.sub(
        r"CREATE TABLE IF NOT EXISTS predictions \(.*?\);", OLD_PREDICTIONS, schema, flags=re.S
    )
    path.mkdir()
    snapshot = {}
    with sqlite3.connect(path / "iris.sqlite3") as conn:
        conn.executescript(schema)
        for table in ("sessions", "assets", "frames", "jobs", "comparisons", "runs", "predictions"):
            snapshot[table] = []
            for row in store.list(table):
                row = dict(row)
                if table == "runs":
                    row.pop("variant")
                if table == "predictions":
                    row.pop("metadata")
                if table == "comparisons":
                    for key in ("inference", "lanes", "work"):
                        row["config"].pop(key)
                encoded = _encode(row)
                conn.execute(
                    f"INSERT INTO {table} ({','.join(encoded)}) "
                    f"VALUES ({','.join('?' for _ in encoded)})",
                    list(encoded.values()),
                )
                snapshot[table].append(row)
        if broken:
            conn.execute("UPDATE predictions SET run_id='missing'")
        conn.execute("PRAGMA user_version=8")
    return comparison, snapshot


def test_schema8_migration_keeps_saved_rows_and_foreign_keys_and_allows_second_variant(example):
    path = example[0].root.parent / "schema8"
    comparison, before = historical_database(example, path)
    store = Store(path)
    for table, expected in before.items():
        actual = store.list(table)
        for row in actual:
            if table == "runs":
                assert row.pop("variant") == "full"
            if table == "predictions":
                assert row.pop("metadata") == {}
        assert actual == expected
    detail = comparison_detail(store, comparison["id"])
    assert detail["lanes"][0]["variant"] == "full"
    assert detail["lanes"][0]["run_id"] == before["runs"][0]["id"]
    with store.connect() as conn:
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 12
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("UPDATE predictions SET run_id='missing'")
    run = before["runs"][0]
    tiled = store.insert("runs", {**run, "id": new_id(), "variant": "tiled"})
    with pytest.raises(sqlite3.IntegrityError):
        store.insert("runs", {**tiled, "id": new_id()})
    with pytest.raises(sqlite3.IntegrityError):
        store.insert("runs", {**run, "id": new_id(), "variant": "invalid"})
    reopened = Store(path)
    assert reopened.list("runs") == store.list("runs")
    assert reopened.list("predictions") == store.list("predictions")


def test_failed_migration_rolls_back_schema_and_version(example):
    path = example[0].root.parent / "broken-schema8"
    historical_database(example, path, broken=True)
    with pytest.raises(RuntimeError, match="foreign keys"):
        Store(path)
    with sqlite3.connect(path / "iris.sqlite3") as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 8
        assert "variant" not in {row[1] for row in conn.execute("PRAGMA table_info(runs)")}
        assert "metadata" not in {row[1] for row in conn.execute("PRAGMA table_info(predictions)")}
        assert conn.execute("SELECT count(*) FROM predictions").fetchone()[0] == 1
