"""Local media ingestion and bounded, repeatable video frame extraction."""

from __future__ import annotations

import base64
import hashlib
import io
import math
import shutil
import sqlite3
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
from PIL import Image, ImageOps, UnidentifiedImageError

from iris.store import new_id, now
from iris.video_sampling import ALGORITHM, evenly_spaced_indices, plan_extraction

if TYPE_CHECKING:
    from iris.store import Store


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pixel_hash(image: Image.Image) -> str:
    digest = hashlib.sha256(f"RGB:{image.width}:{image.height}:".encode())
    digest.update(image.tobytes())
    return digest.hexdigest()


def _perceptual_hash(image: Image.Image) -> str:
    """64-bit difference hash; optional near-duplicate filtering uses Hamming distance."""
    values = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS).tobytes()
    bits = 0
    for row in range(8):
        for col in range(8):
            bits = (bits << 1) | int(values[row * 9 + col] > values[row * 9 + col + 1])
    return f"{bits:016x}"


def _image_source(path: Path) -> tuple[Image.Image, dict] | None:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as original:
                if getattr(original, "n_frames", 1) != 1:
                    raise ValueError("Animated images are unsupported; import a video instead.")
                metadata = {
                    "format": original.format,
                    "original_width": original.width,
                    "original_height": original.height,
                    "exif_orientation": original.getexif().get(274, 1),
                }
                image = ImageOps.exif_transpose(original).convert("RGB")
                image.load()
                metadata.update(width=image.width, height=image.height)
                return image, metadata
    except UnidentifiedImageError:
        return None
    except (OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError("Invalid, truncated, or oversized image.") from exc


def _video_media_type(path: Path) -> str:
    # OpenCV delegates to FFmpeg, which also understands playlists and URLs.
    # Only pass known binary video containers to that decoder, never arbitrary
    # text that could trigger network reads of a remote playlist entry.
    with path.open("rb") as source:
        header = source.read(4096)
    if header[:4] == b"RIFF" and header[8:12] == b"AVI ":
        return "video/x-msvideo"
    if header[4:8] == b"ftyp":
        return "video/quicktime" if header[8:12] == b"qt  " else "video/mp4"
    if header[:4] == b"\x1a\x45\xdf\xa3":
        # Read the EBML DocType element's variable-length size, including legal
        # nonminimal encodings, instead of trusting the file extension.
        position = header.find(b"\x42\x82", 4)
        if position >= 0 and position + 2 < len(header):
            first_byte = header[position + 2]
            if first_byte:
                size_bytes = 9 - first_byte.bit_length()
                start = position + 2
                raw_size = header[start : start + size_bytes]
                length = int.from_bytes(raw_size, "big") & ((1 << (7 * size_bytes)) - 1)
                start += size_bytes
                doctype = header[start : start + length]
                if doctype == b"webm":
                    return "video/webm"
                if doctype == b"matroska":
                    return "video/x-matroska"
    raise ValueError("Unsupported video container; use AVI, MP4/MOV/M4V, or MKV/WebM.")


def _video_metadata(path: Path) -> dict:
    media_type = _video_media_type(path)
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError("The file is not a readable image or video.")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count_value = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, first_frame = capture.read()
        if (
            not ok
            or first_frame is None
            or not math.isfinite(fps)
            or fps <= 0
            or not math.isfinite(frame_count_value)
            or frame_count_value < 1
        ):
            raise ValueError("Unreadable video or unavailable timing metadata.")
        frame_count = int(frame_count_value)
        height, width = first_frame.shape[:2]
        return {
            "media_type": media_type,
            "fps": fps,
            "frame_count": frame_count,
            "duration_seconds": frame_count / fps,
            "width": width,
            "height": height,
            "timestamp_basis": "frame_index / nominal_fps",
        }
    finally:
        capture.release()


def _frame_record(
    store: Store,
    asset: dict,
    image: Image.Image,
    *,
    frame_index: int | None = None,
    timestamp_seconds: float | None = None,
    extraction: dict | None = None,
) -> dict:
    frame_id = new_id()
    relative_path = Path("frames") / f"{frame_id}.png"
    path = store.root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    try:
        image.save(temporary, format="PNG")
        temporary.replace(path)
        return store.insert(
            "frames",
            {
                "id": frame_id,
                "session_id": asset["session_id"],
                "asset_id": asset["id"],
                "frame_index": frame_index,
                "timestamp_seconds": timestamp_seconds,
                "width": image.width,
                "height": image.height,
                "sha256": _pixel_hash(image),
                "perceptual_hash": _perceptual_hash(image),
                "path": relative_path.as_posix(),
                "selected": False,
                "extraction": extraction or {},
                "created_at": now(),
            },
        )
    except BaseException:
        temporary.unlink(missing_ok=True)
        path.unlink(missing_ok=True)
        raise


def import_asset(store: Store, session_id: str, source: Path, filename: str) -> dict:
    """Preserve an original locally and create an EXIF-normalized frame for still images.

    Byte-identical imports are idempotent within a session. Distinct sessions retain
    their own provenance, even when their source files contain identical pixels.
    """
    if store.get("sessions", session_id) is None:
        raise ValueError("Session not found.")
    source = Path(source)
    if not source.is_file() or source.stat().st_size == 0:
        raise ValueError("The source file is missing or empty.")
    safe_filename = str(filename).replace("\\", "/").rsplit("/", 1)[-1]
    if not safe_filename or safe_filename in {".", ".."} or "\x00" in safe_filename:
        raise ValueError("Invalid filename.")
    source_hash = _file_hash(source)
    for existing in store.list("assets", session_id=session_id):
        if existing["sha256"] == source_hash:
            return existing

    decoded = _image_source(source)
    image, metadata = decoded if decoded else (None, _video_metadata(source))
    kind = "image" if image is not None else "video"
    asset_id = new_id()
    extension = Path(safe_filename).suffix.lower()
    if not extension or len(extension) > 10 or not extension[1:].isalnum():
        extension = ".media"
    relative_path = Path("assets") / f"{asset_id}{extension}"
    path = store.root / relative_path
    temporary = path.with_suffix(path.suffix + ".tmp")
    inserted = False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, temporary)
        temporary.replace(path)
        asset = store.insert(
            "assets",
            {
                "id": asset_id,
                "session_id": session_id,
                "filename": safe_filename,
                "kind": kind,
                "sha256": source_hash,
                "size_bytes": source.stat().st_size,
                "path": relative_path.as_posix(),
                "metadata": metadata,
                "created_at": now(),
            },
        )
        inserted = True
        if image is not None:
            _frame_record(store, asset, image, extraction={"method": "image_import"})
        return asset
    except BaseException as exc:
        if inserted:
            with store.connect() as connection:
                connection.execute("DELETE FROM assets WHERE id = ?", (asset_id,))
        temporary.unlink(missing_ok=True)
        path.unlink(missing_ok=True)
        # Concurrent HTTP uploads may both pass the initial lookup. The unique
        # session/hash constraint makes one win; remove this copy and reuse it.
        if not inserted and isinstance(exc, sqlite3.IntegrityError):
            for existing in store.list("assets", session_id=session_id):
                if existing["sha256"] == source_hash:
                    return existing
        raise
    finally:
        if image is not None:
            image.close()


def _verified_video_source(
    store: Store, asset: dict, cancelled: Callable[[], bool] | None = None
) -> Path | None:
    """Reject replaced sources before decoding; hashing can be interrupted."""
    source = store.artifact_path(asset["path"])
    if not source.is_file():
        raise ValueError("The original video is missing or unreadable.")
    digest = hashlib.sha256()
    with source.open("rb") as original:
        while True:
            if cancelled is not None and cancelled():
                return None
            chunk = original.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    if digest.hexdigest() != asset["sha256"]:
        raise ValueError(
            "The original video changed since import; import it again as a new source."
        )
    if cancelled is not None and cancelled():
        return None
    _video_media_type(source)
    return source


def _video_position_error(action: str, frame_index: int) -> ValueError:
    return ValueError(
        f"Cannot {action} video frame {frame_index}. "
        "Video timing metadata may be inaccurate; try a shorter range "
        "or a constant-frame-rate copy."
    )


def preview_extraction(
    store: Store, asset_id: str, config: dict, *, include_images: bool = False
) -> dict:
    """Preview the exact plan and at most twelve local thumbnails, without writes."""
    asset = store.get("assets", asset_id)
    if asset is None or asset["kind"] != "video":
        raise ValueError("Extraction requires an imported video.")
    plan = plan_extraction(asset["metadata"], config)
    result = {**plan, "asset_id": asset_id, "source_sha256": asset["sha256"]}
    if not include_images:
        return result
    source = _verified_video_source(store, asset)
    capture = cv2.VideoCapture(str(source))
    thumbnails = []
    try:
        if not capture.isOpened():
            raise ValueError("The original video is missing or unreadable.")
        for index in evenly_spaced_indices(0, plan["planned_count"] - 1, 12):
            position = plan["positions"][index]
            frame_index = position["frame_index"]
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index):
                raise _video_position_error("seek to", frame_index)
            ok, pixels = capture.read()
            if not ok or pixels is None:
                raise _video_position_error("decode", frame_index)
            with Image.fromarray(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)) as image:
                image.thumbnail((384, 216), Image.Resampling.LANCZOS)
                output = io.BytesIO()
                image.save(output, format="JPEG", quality=80)
                encoded = base64.b64encode(output.getvalue()).decode("ascii")
            thumbnails.append({**position, "image_data_url": f"data:image/jpeg;base64,{encoded}"})
        result["thumbnails"] = thumbnails
        return result
    finally:
        capture.release()


def extract_frames(
    store: Store,
    asset_id: str,
    config: dict,
    progress: Callable[[float, str], None],
    cancelled: Callable[[], bool],
    *,
    plan: dict | None = None,
) -> dict:
    """Sample a bounded set of video positions; keep completed work on cancellation.

    The limit applies to sampled positions, including existing/duplicate frames.
    Timestamps use the video's nominal FPS and are approximate for variable-FPS
    sources. Exact and optional perceptual deduplication stay within this asset.
    """
    asset = store.get("assets", asset_id)
    if asset is None or asset["kind"] != "video":
        raise ValueError("Extraction requires an imported video.")
    if plan is None:
        plan = plan_extraction(asset["metadata"], config)
    else:
        # A passage plan must be reproduced from saved model proposals and an
        # explicit human choice, never accepted as arbitrary frame indices.
        from iris.video_reviews import validate_passage_extraction

        plan = validate_passage_extraction(store, plan)
        if plan["asset_id"] != asset_id or plan["source_sha256"] != asset["sha256"]:
            raise ValueError("The extraction plan does not match this video source.")
    targets = [position["frame_index"] for position in plan["positions"]]
    fps = plan["fps"]
    dedup = config.get("dedup_hamming")
    provenance = dict(config)
    if plan["sampling_mode"] == "passages":
        provenance["sampling_algorithm"] = plan["algorithm"]
        provenance["video_review_id"] = plan["video_review_id"]
        provenance["passage_ids"] = plan["passage_ids"]
    if plan["sampling_mode"] == "uniform":
        provenance["sampling_algorithm"] = ALGORITHM
        provenance["sampling_plan"] = {
            key: plan[key]
            for key in (
                "planned_count",
                "start_seconds",
                "end_seconds",
                "first_timestamp_seconds",
                "last_timestamp_seconds",
            )
        }

    existing = store.list("frames", asset_id=asset_id)
    existing_indices = {frame["frame_index"] for frame in existing}
    exact_hashes = {frame["sha256"] for frame in existing}
    perceptual_hashes = [
        int(frame["perceptual_hash"], 16) for frame in existing if frame["perceptual_hash"]
    ]
    result = {
        "asset_id": asset_id,
        "sampled": 0,
        "created": 0,
        "skipped_existing": 0,
        "skipped_exact": 0,
        "skipped_similar": 0,
        "frame_ids": [],
        "cancelled": False,
        "timestamp_basis": "frame_index / nominal_fps",
        "plan": {**plan, "asset_id": asset_id, "source_sha256": asset["sha256"]},
    }
    if cancelled():
        result["cancelled"] = True
        return result
    source = _verified_video_source(store, asset, cancelled)
    if source is None:
        result["cancelled"] = True
        return result
    capture = cv2.VideoCapture(str(source))
    try:
        if not capture.isOpened():
            raise ValueError("The original video is missing or unreadable.")
        for position, frame_index in enumerate(targets):
            if cancelled():
                result["cancelled"] = True
                break
            result["sampled"] += 1
            if frame_index in existing_indices:
                result["skipped_existing"] += 1
            else:
                if not capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index):
                    raise _video_position_error("seek to", frame_index)
                ok, pixels = capture.read()
                if not ok or pixels is None:
                    raise _video_position_error("decode", frame_index)
                if cancelled():
                    result["cancelled"] = True
                    break
                image = Image.fromarray(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB))
                try:
                    pixel_hash = _pixel_hash(image)
                    perceptual_hash = int(_perceptual_hash(image), 16)
                    if pixel_hash in exact_hashes:
                        result["skipped_exact"] += 1
                    elif dedup is not None and any(
                        (perceptual_hash ^ previous).bit_count() <= dedup
                        for previous in perceptual_hashes
                    ):
                        result["skipped_similar"] += 1
                    else:
                        if cancelled():
                            result["cancelled"] = True
                            break
                        record = _frame_record(
                            store,
                            asset,
                            image,
                            frame_index=frame_index,
                            timestamp_seconds=frame_index / fps,
                            extraction=provenance,
                        )
                        exact_hashes.add(pixel_hash)
                        perceptual_hashes.append(perceptual_hash)
                        existing_indices.add(frame_index)
                        result["frame_ids"].append(record["id"])
                        result["created"] += 1
                finally:
                    image.close()
            progress((position + 1) / len(targets), f"{result['created']} frames extracted")
        return result
    finally:
        capture.release()
