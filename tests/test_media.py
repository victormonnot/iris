"""Media tests use generated image/video fixtures, never flight data."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import cv2
import numpy as np
import pytest
from PIL import Image

from iris.media import extract_frames, import_asset
from iris.store import Store, new_id, now


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "data")


@pytest.fixture
def session(store):
    return store.insert(
        "sessions",
        {"id": new_id(), "name": "Synthetic flight", "scene_group": "fixture", "created_at": now()},
    )


def make_image(path: Path, *, color=(10, 20, 30), size=(24, 16)) -> Path:
    Image.new("RGB", size, color).save(path)
    return path


def make_video(path: Path, colors=None) -> Path:
    # MJPEG encodes each frame independently, so identical fixture frames decode
    # identically and can exercise the exact-pixel deduplication contract.
    colors = colors or [(index * 20, 30, 150) for index in range(8)]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 4.0, (48, 32))
    assert writer.isOpened(), "The test environment requires the local MJPEG encoder"
    try:
        for color in colors:
            writer.write(np.full((32, 48, 3), color, dtype=np.uint8))
    finally:
        writer.release()
    return path


def extract(store, asset, **config):
    return extract_frames(store, asset["id"], config, lambda *_: None, lambda: False)


def test_image_import_preserves_original_and_normalizes_exif(store, session, tmp_path):
    source = tmp_path / "portrait.jpg"
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (12, 20), (10, 30, 50)).save(source, exif=exif)
    asset = import_asset(store, session["id"], source, "../../portrait.jpg")
    assert asset["kind"] == "image"
    assert asset["filename"] == "portrait.jpg"
    assert (store.root / asset["path"]).read_bytes() == source.read_bytes()
    assert asset["metadata"]["exif_orientation"] == 6
    (frame,) = store.list("frames", asset_id=asset["id"])
    assert (frame["width"], frame["height"]) == (20, 12)
    assert frame["session_id"] == session["id"]
    assert frame["frame_index"] is None
    assert frame["selected"] is False
    assert frame["extraction"] == {"method": "image_import"}
    with Image.open(store.root / frame["path"]) as normalized:
        assert normalized.size == (20, 12)
        assert normalized.mode == "RGB"
        assert normalized.getexif().get(274, 1) == 1


def test_identical_asset_import_is_idempotent_without_file_leaks(store, session, tmp_path):
    source = make_image(tmp_path / "image.png")
    first = import_asset(store, session["id"], source, source.name)
    original_files = set(store.root.glob("assets/*")) | set(store.root.glob("frames/*"))
    second = import_asset(store, session["id"], source, "renamed.png")
    assert second == first
    assert len(store.list("assets")) == len(store.list("frames")) == 1
    assert set(store.root.glob("assets/*")) | set(store.root.glob("frames/*")) == original_files


def test_concurrent_identical_imports_share_one_asset(store, session, tmp_path, monkeypatch):
    source = make_image(tmp_path / "image.png")
    original_insert = store.insert
    ready = Barrier(2, timeout=5)

    def concurrent_insert(table, record):
        if table == "assets":
            ready.wait()
        return original_insert(table, record)

    monkeypatch.setattr(store, "insert", concurrent_insert)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(import_asset, store, session["id"], source, source.name) for _ in range(2)
        ]
        results = [future.result() for future in futures]
    assert results[0] == results[1]
    assert len(store.list("assets")) == len(store.list("frames")) == 1
    assert len(list(store.root.glob("assets/*"))) == 1
    assert len(list(store.root.glob("frames/*"))) == 1


def test_pixel_hash_is_encoding_independent_and_preserves_flight_provenance(
    store, session, tmp_path
):
    png = make_image(tmp_path / "image.png")
    bmp = make_image(tmp_path / "image.bmp")
    first = import_asset(store, session["id"], png, png.name)
    second = import_asset(store, session["id"], bmp, bmp.name)
    another_session = store.insert(
        "sessions",
        {"id": new_id(), "name": "Another flight", "scene_group": "fixture", "created_at": now()},
    )
    third = import_asset(store, another_session["id"], png, png.name)
    assert first["sha256"] != second["sha256"]
    frames = [store.list("frames", asset_id=asset["id"])[0] for asset in (first, second, third)]
    assert len({frame["sha256"] for frame in frames}) == 1
    assert frames[2]["session_id"] == another_session["id"]
    assert len({frame["id"] for frame in frames}) == 3


def test_invalid_media_creates_no_records_or_artifacts(store, session, tmp_path):
    source = tmp_path / "fake.mp4"
    source.write_bytes(b"this is not a media file")
    with pytest.raises(ValueError, match="Unsupported video container"):
        import_asset(store, session["id"], source, source.name)
    assert store.list("assets") == store.list("frames") == []
    assert not list(store.root.glob("assets/*"))
    assert not list(store.root.glob("frames/*"))


def test_playlist_is_rejected_before_opencv_opens_it(store, session, tmp_path, monkeypatch):
    source = tmp_path / "playlist.mp4"
    source.write_text("#EXTM3U\n#EXTINF:4,\nhttps://example.com/flight.ts\n")

    def forbidden_decoder(*args, **kwargs):
        raise AssertionError("A playlist must not be handed to the video decoder")

    monkeypatch.setattr(cv2, "VideoCapture", forbidden_decoder)
    with pytest.raises(ValueError, match="Unsupported video container"):
        import_asset(store, session["id"], source, source.name)
    assert store.list("assets") == []


def test_failed_image_frame_creation_rolls_back_asset_and_files(
    store, session, tmp_path, monkeypatch
):
    source = make_image(tmp_path / "image.png")
    original_insert = store.insert

    def fail_frame_insert(table, record):
        if table == "frames":
            raise OSError("simulated write failure")
        return original_insert(table, record)

    monkeypatch.setattr(store, "insert", fail_frame_insert)
    with pytest.raises(OSError, match="simulated write failure"):
        import_asset(store, session["id"], source, source.name)
    assert store.list("assets") == store.list("frames") == []
    assert not list(store.root.glob("assets/*"))
    assert not list(store.root.glob("frames/*"))


def test_video_sampling_has_provenance_bounds_and_repeatability(store, session, tmp_path):
    source = make_video(tmp_path / "synthetic.avi")
    asset = import_asset(store, session["id"], source, source.name)
    assert asset["kind"] == "video"
    assert asset["metadata"]["fps"] == pytest.approx(4)
    assert asset["metadata"]["duration_seconds"] == pytest.approx(2)
    assert asset["metadata"]["frame_count"] == 8
    assert asset["metadata"]["media_type"] == "video/x-msvideo"
    assert (asset["metadata"]["width"], asset["metadata"]["height"]) == (48, 32)
    config = dict(interval_seconds=0.5, start_seconds=0.5, end_seconds=1.6, job_id="fixture-job")
    first = extract(store, asset, **config)
    assert first["created"] == 3
    frames = sorted(store.list("frames", asset_id=asset["id"]), key=lambda row: row["frame_index"])
    assert [frame["frame_index"] for frame in frames] == [2, 4, 6]
    assert [frame["timestamp_seconds"] for frame in frames] == [0.5, 1.0, 1.5]
    assert all(frame["session_id"] == session["id"] for frame in frames)
    assert all(frame["extraction"] == config for frame in frames)
    assert all(frame["selected"] is False for frame in frames)
    second = extract(store, asset, interval_seconds=0.5, start_seconds=0.5, end_seconds=1.6)
    assert second["created"] == 0
    assert second["skipped_existing"] == 3
    assert len(list((store.root / "frames").glob("*.png"))) == 3


def test_video_exact_duplicates_are_removed_within_asset(store, session, tmp_path):
    source = make_video(tmp_path / "repeated.avi", colors=[(10, 20, 30)] * 4 + [(80, 90, 100)] * 4)
    asset = import_asset(store, session["id"], source, source.name)
    result = extract(store, asset, interval_seconds=0.25)
    assert result["sampled"] == 8
    assert result["created"] == 2
    assert result["skipped_exact"] == 6
    assert result["skipped_similar"] == 0


def test_perceptual_filter_is_explicit_and_sample_count_is_bounded(store, session, tmp_path):
    source = make_video(tmp_path / "flat_colors.avi")
    asset = import_asset(store, session["id"], source, source.name)
    # Different flat colors have identical difference hashes. The optional
    # filter deliberately treats them alike; default exact filtering does not.
    result = extract(store, asset, interval_seconds=0.001, max_frames=3, dedup_hamming=0)
    assert result["sampled"] == 3
    assert result["created"] == 1
    assert result["skipped_similar"] == 2


def test_cancelled_extraction_keeps_completed_frames_and_can_resume(store, session, tmp_path):
    source = make_video(tmp_path / "cancel.avi")
    asset = import_asset(store, session["id"], source, source.name)
    updates = []
    result = extract_frames(
        store,
        asset["id"],
        {"interval_seconds": 0.25},
        lambda *update: updates.append(update),
        lambda: len(updates) == 2,
    )
    assert result["cancelled"] is True
    assert result["created"] == len(store.list("frames")) == 2
    resumed = extract(store, asset, interval_seconds=0.25)
    assert resumed["created"] == 6
    assert resumed["skipped_existing"] == 2
    assert len(store.list("frames")) == 8


def test_already_cancelled_job_does_not_open_video(store, session, tmp_path, monkeypatch):
    source = make_video(tmp_path / "cancel.avi")
    asset = import_asset(store, session["id"], source, source.name)

    def forbidden_decoder(*args, **kwargs):
        raise AssertionError("An already-cancelled job must not open the video")

    monkeypatch.setattr(cv2, "VideoCapture", forbidden_decoder)
    result = extract_frames(store, asset["id"], {}, lambda *_: None, lambda: True)
    assert result["cancelled"] is True
    assert result["created"] == 0
    assert store.list("frames") == []


@pytest.mark.parametrize(
    "config",
    [
        {"interval_seconds": 0},
        {"interval_seconds": float("nan")},
        {"start_seconds": -1},
        {"start_seconds": 2},
        {"start_seconds": 1, "end_seconds": 0.5},
        {"max_frames": 0},
        {"max_frames": 501},
        {"max_frames": True},
        {"max_frames": 1.5},
        {"dedup_hamming": -1},
        {"dedup_hamming": 17},
    ],
)
def test_invalid_extraction_parameters_do_not_create_frames(store, session, tmp_path, config):
    source = make_video(tmp_path / "parameters.avi")
    asset = import_asset(store, session["id"], source, source.name)
    with pytest.raises(ValueError):
        extract(store, asset, **config)
    assert store.list("frames") == []
