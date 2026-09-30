"""Saved-sample replay, with generated video fixtures and no model execution."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from iris.app import create_app
from iris.comparison_replay import comparison_replay
from iris.inference import comparison_detail
from iris.media import extract_frames, import_asset
from iris.store import Store, new_id, now

MODELS = ["ssdlite320_mobilenet_v3_large", "fasterrcnn_mobilenet_v3_large_320_fpn"]


def _video(store, session, tmp_path, name, fps, channel):
    source = tmp_path / name
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), fps, (48, 32))
    assert writer.isOpened()
    try:
        for index in range(8):
            pixels = np.full((32, 48, 3), 30, dtype=np.uint8)
            pixels[:, :, channel] = index * 25
            writer.write(pixels)
    finally:
        writer.release()
    asset = import_asset(store, session["id"], source, name)
    extract_frames(
        store,
        asset["id"],
        {"interval_seconds": 0.5, "max_frames": 3},
        lambda *_: None,
        lambda: False,
    )
    frames = sorted(
        store.list("frames", asset_id=asset["id"]), key=lambda frame: frame["timestamp_seconds"]
    )
    return asset, frames


def _comparison(store, session, frames, *, paired=False, legacy=False, status="succeeded"):
    job = store.insert(
        "jobs",
        {"id": new_id(), "kind": "infer", "status": status, "params": {}, "created_at": now()},
    )
    config = {} if legacy else {"inference": {"mode": "paired" if paired else "full"}}
    result = store.insert(
        "comparisons",
        {
            "id": new_id(),
            "session_id": session["id"],
            "name": "Fixture saved comparison",
            "frame_ids": [frame["id"] for frame in frames],
            "model_ids": MODELS[:1] if paired else MODELS,
            "config": config,
            "job_id": job["id"],
            "created_at": now(),
        },
    )
    return result


def _run(store, comparison, model, variant="full"):
    return store.insert(
        "runs",
        {
            "id": new_id(),
            "comparison_id": comparison["id"],
            "model_id": model,
            "variant": variant,
            "metadata": {},
            "created_at": now(),
        },
    )


def _prediction(store, comparison, run, frame, *, model=None):
    return store.insert(
        "predictions",
        {
            "id": new_id(),
            "comparison_id": comparison["id"],
            "run_id": run["id"],
            "frame_id": frame["id"],
            "model_id": model or run["model_id"],
            "detections": [],
            "timing": {},
            "input_size": [frame["width"], frame["height"]],
            "created_at": now(),
        },
    )


@pytest.fixture
def workspace(tmp_path):
    store = Store(tmp_path / "workspace")
    session = store.insert(
        "sessions",
        {"id": new_id(), "name": "Fixture flight", "scene_group": "fixture", "created_at": now()},
    )
    first, first_frames = _video(store, session, tmp_path, "first.avi", 4, 0)
    second, second_frames = _video(store, session, tmp_path, "second.avi", 6, 1)
    image = tmp_path / "still.png"
    Image.new("RGB", (48, 32), "blue").save(image)
    still = import_asset(store, session["id"], image, image.name)
    still_frame = store.list("frames", asset_id=still["id"])[0]
    frames = [first_frames[2], still_frame, second_frames[1], first_frames[0], second_frames[0]]
    saved = _comparison(store, session, frames)
    return store, session, first, second, first_frames, second_frames, still_frame, saved


def test_groups_saved_video_frames_and_preserves_original_selection(workspace):
    store, _session, first, second, a, b, still, saved = workspace
    detail = comparison_detail(store, saved["id"])
    replay = detail["replay"]
    assert replay["version"] == "iris-comparison-replay-v1"
    assert replay["mode"] == "saved-samples"
    assert replay["continuous_inference"] is False
    assert replay["timestamps_approximate"] is True
    assert replay["media_check"] == "path_and_size_only"
    assert replay["still_frame_ids"] == [still["id"]]
    assert [source["asset_id"] for source in replay["sources"]] == [first["id"], second["id"]]
    source = replay["sources"][0]
    assert source["filename"] == "first.avi"
    assert source["fps"] == 4 and source["duration_seconds"] == 2
    assert source["timestamp_basis"] == "frame_index / nominal_fps"
    assert source["media_url"] == f"/api/assets/{first['id']}/media"
    assert source["media_status"] == "available" and source["media_available"]
    assert [sample["frame_id"] for sample in source["samples"]] == [a[0]["id"], a[2]["id"]]
    assert [sample["timestamp_seconds"] for sample in source["samples"]] == [0, 1]
    assert all(sample["image_available"] for sample in source["samples"])
    assert all(
        not sample["complete"] and sample["prediction_count"] == 0 for sample in source["samples"]
    )
    assert [frame["id"] for frame in detail["frames"]] == saved["frame_ids"]
    assert a[1]["id"] not in json.dumps(replay)
    assert b[2]["id"] not in json.dumps(replay)
    assert "path" not in json.dumps(replay).replace('"path_and_size_only"', "")
    assert str(store.root) not in json.dumps(replay)


@pytest.mark.parametrize("paired", [False, True])
def test_partial_empty_predictions_use_exact_planned_run_identity(workspace, paired):
    store, session, first, _second, frames, _b, _still, _saved = workspace
    saved = _comparison(store, session, frames, paired=paired, status="interrupted")
    runs = [
        _run(store, saved, MODELS[0]),
        _run(store, saved, MODELS[0] if paired else MODELS[1], "tiled" if paired else "full"),
    ]
    _prediction(store, saved, runs[0], frames[0])
    _prediction(store, saved, runs[0], frames[1])
    _prediction(store, saved, runs[1], frames[1])
    unrelated = _run(store, saved, "unplanned-model")
    _prediction(store, saved, unrelated, frames[2])
    _prediction(store, saved, runs[0], frames[2], model="wrong-recorded-model")
    source = comparison_detail(store, saved["id"])["replay"]["sources"][0]
    assert source["asset_id"] == first["id"]
    samples = source["samples"]
    assert [sample["prediction_count"] for sample in samples] == [1, 2, 0]
    assert [sample["complete"] for sample in samples] == [False, True, False]
    assert samples[0]["predicted_run_ids"] == [runs[0]["id"]]
    assert set(samples[1]["predicted_run_ids"]) == {run["id"] for run in runs}


def test_missing_lane_is_not_mistaken_for_complete_after_single_result(workspace):
    store, _session, _first, _second, frames, _b, _still, saved = workspace
    run = _run(store, saved, MODELS[0])
    _prediction(store, saved, run, frames[0])
    sample = comparison_detail(store, saved["id"])["replay"]["sources"][0]["samples"][0]
    assert sample["prediction_count"] == 1 and not sample["complete"]


def test_legacy_comparison_without_inference_contract_or_fps_still_replays_saved_times(workspace):
    store, session, first, _second, frames, _b, _still, _saved = workspace
    saved = _comparison(store, session, list(reversed(frames)), legacy=True)
    store.update("assets", first["id"], {"metadata": {}})
    store.update("frames", frames[1]["id"], {"timestamp_seconds": None})
    # A recorded timestamp must not be replaced with a newly calculated value.
    store.update("frames", frames[2]["id"], {"timestamp_seconds": 1.125})
    detail = comparison_detail(store, saved["id"])
    source = detail["replay"]["sources"][0]
    assert source["fps"] is None and source["duration_seconds"] is None
    assert source["media_type"] is None
    assert [sample["timestamp_seconds"] for sample in source["samples"]] == [0, 1.125, None]
    assert source["samples"][-1]["frame_id"] == frames[1]["id"]
    assert all(lane["variant"] == "full" for lane in detail["lanes"])


def test_still_only_comparison_has_no_video_timeline(workspace):
    store, session, _first, _second, _a, _b, still, _saved = workspace
    saved = _comparison(store, session, [still])
    replay = comparison_detail(store, saved["id"])["replay"]
    assert replay["sources"] == [] and replay["still_frame_ids"] == [still["id"]]


@pytest.mark.parametrize("timestamp", [None, -1, True, "1", float("nan"), float("inf")])
def test_invalid_or_missing_saved_timestamps_remain_unpositioned(workspace, timestamp):
    store, _session, first, _second, frames, _b, _still, _saved = workspace
    frame = {**frames[0], "timestamp_seconds": timestamp}
    replay = comparison_replay(
        store, [frame], {first["id"]: first}, [{"model_id": MODELS[0], "run_id": None}], []
    )
    assert replay["sources"][0]["samples"][0]["timestamp_seconds"] is None


@pytest.mark.parametrize(
    "kind,status,http",
    [
        ("missing", "missing", 404),
        ("directory", "missing", 404),
        ("resized", "size_mismatch", 409),
        ("symlink", "unsafe", 422),
        ("parent_symlink", "unsafe", 422),
        ("outside", "unsafe", 422),
        ("dot", "unsafe", 422),
    ],
)
def test_unavailable_source_keeps_saved_predictions_and_images(
    workspace, tmp_path, kind, status, http
):
    store, _session, first, _second, frames, _b, _still, saved = workspace
    path = store.root / first["path"]
    if kind in {"missing", "directory", "symlink"}:
        path.unlink()
    if kind == "directory":
        path.mkdir()
    elif kind == "resized":
        path.write_bytes(b"different size")
    elif kind == "symlink":
        target = tmp_path / "private.txt"
        target.write_bytes(b"outside private bytes")
        path.symlink_to(target)
    elif kind == "parent_symlink":
        parent = store.root / "assets" / "linked"
        parent.symlink_to(path.parent, target_is_directory=True)
        store.update("assets", first["id"], {"path": f"assets/linked/{path.name}"})
    elif kind in {"outside", "dot"}:
        store.update(
            "assets", first["id"], {"path": "../private.txt" if kind == "outside" else "."}
        )
    run = _run(store, saved, MODELS[0])
    prediction = _prediction(store, saved, run, frames[0])
    with TestClient(create_app(store.root, run_jobs=False), base_url="http://127.0.0.1") as api:
        response = api.get(f"/api/comparisons/{saved['id']}")
        assert response.status_code == 200
        detail = response.json()
        source = detail["replay"]["sources"][0]
        assert source["media_status"] == status
        assert not source["media_available"] and source["media_url"] is None
        assert all(sample["image_available"] for sample in source["samples"])
        assert detail["predictions"] == [prediction]
        media = api.get(f"/api/assets/{first['id']}/media")
        assert media.status_code == http
        assert str(store.root) not in response.text and str(store.root) not in media.text


def test_missing_png_is_explicit_without_hiding_the_original_video(workspace):
    store, _session, first, _second, frames, _b, _still, saved = workspace
    (store.root / frames[0]["path"]).unlink()
    source = comparison_detail(store, saved["id"])["replay"]["sources"][0]
    assert source["media_available"] and source["asset_id"] == first["id"]
    assert not source["samples"][0]["image_available"]
    assert source["samples"][1]["image_available"]


def test_api_range_streams_original_bytes_and_replay_polling_does_not_read_media(
    workspace, monkeypatch
):
    store, _session, first, _second, _frames, _b, _still, saved = workspace
    source_bytes = (store.root / first["path"]).read_bytes()
    with TestClient(create_app(store.root, run_jobs=False), base_url="http://127.0.0.1") as api:
        media = api.get(f"/api/assets/{first['id']}/media", headers={"Range": "bytes=8-31"})
        assert media.status_code == 206
        assert media.content == source_bytes[8:32]
        assert media.headers["content-range"] == f"bytes 8-31/{len(source_bytes)}"
        assert media.headers["content-type"] == "video/x-msvideo"
        assert media.headers["content-disposition"].startswith("inline")
        before = {
            table: store.list(table)
            for table in ("comparisons", "frames", "assets", "predictions", "runs", "jobs")
        }

        def forbidden(*_args, **_kwargs):
            pytest.fail("Replay polling must not open/decode media or load a model")

        monkeypatch.setattr(Path, "open", forbidden)
        monkeypatch.setattr(cv2, "VideoCapture", forbidden)
        monkeypatch.setattr("iris.inference.TorchvisionDetector", forbidden)
        for _ in range(3):
            response = api.get(f"/api/comparisons/{saved['id']}")
            assert response.status_code == 200
            assert response.json()["replay"]["sources"][0]["media_available"]
        assert before == {table: store.list(table) for table in before}


def test_same_size_source_replacement_does_not_claim_checksum_verification(workspace):
    store, _session, first, _second, _a, _b, _still, saved = workspace
    path = store.root / first["path"]
    path.write_bytes(b"x" * path.stat().st_size)
    replay = comparison_detail(store, saved["id"])["replay"]
    assert replay["media_check"] == "path_and_size_only"
    assert replay["sources"][0]["media_available"]
