"""Read-only navigation through saved comparison samples and their source videos."""

import math
import stat
from pathlib import PurePosixPath
from urllib.parse import quote

from iris.store import Store

TIMESTAMP_BASIS = "frame_index / nominal_fps"
VIDEO_TYPES = {"video/mp4", "video/quicktime", "video/webm", "video/x-matroska", "video/x-msvideo"}


def _number(value, *, positive=False):
    if type(value) not in {int, float}:
        return None
    try:
        if not math.isfinite(value):
            return None
    except OverflowError:
        return None
    return value if value > 0 or (not positive and value == 0) else None


def local_media_file(store: Store, relative, *, size_bytes=None):
    """Stat a managed regular file; this deliberately does not rehash source video.

    The returned path stays internal. Refuse symlinks, including parent directories,
    so the read endpoint and the replay availability indicator use the same rule.
    """
    if not isinstance(relative, str) or not relative or "\\" in relative:
        return "unsafe", None
    path = PurePosixPath(relative)
    if (
        not path.parts
        or path.is_absolute()
        or str(path) != relative
        or any(part in {".", ".."} for part in path.parts)
        or path.parts[0] not in {"assets", "imports", "frames"}
    ):
        return "unsafe", None
    current = store.root
    try:
        for part in path.parts:
            current = current / part
            if current.is_symlink():
                return "unsafe", None
        info = current.stat()
        if not stat.S_ISREG(info.st_mode):
            return "missing", None
    except (OSError, ValueError):
        return "missing", None
    if size_bytes is not None and (
        type(size_bytes) is not int or size_bytes < 0 or info.st_size != size_bytes
    ):
        return "size_mismatch", None
    return "available", current


def comparison_replay(
    store: Store, frames: list[dict], assets: dict, lanes: list[dict], predictions: list[dict]
) -> dict:
    """Build a small replay index from persisted records, without decoding or inference."""
    expected_runs = {lane["run_id"]: lane["model_id"] for lane in lanes if lane.get("run_id")}
    observed = {}
    for prediction in predictions:
        if expected_runs.get(prediction["run_id"]) == prediction["model_id"]:
            observed.setdefault(prediction["frame_id"], set()).add(prediction["run_id"])
    sources, still_frame_ids = {}, []
    for frame in frames:
        asset = assets[frame["asset_id"]]
        if asset["kind"] != "video":
            still_frame_ids.append(frame["id"])
            continue
        asset_id = asset["id"]
        if asset_id not in sources:
            metadata = asset["metadata"] if isinstance(asset["metadata"], dict) else {}
            status, _path = local_media_file(store, asset["path"], size_bytes=asset["size_bytes"])
            media_type = metadata.get("media_type")
            sources[asset_id] = {
                "asset_id": asset_id,
                "filename": asset["filename"],
                "media_url": f"/api/assets/{quote(asset_id, safe='')}/media"
                if status == "available"
                else None,
                "media_available": status == "available",
                "media_status": status,
                "media_type": media_type
                if isinstance(media_type, str) and media_type in VIDEO_TYPES
                else None,
                "duration_seconds": _number(metadata.get("duration_seconds"), positive=True),
                "fps": _number(metadata.get("fps"), positive=True),
                "timestamp_basis": TIMESTAMP_BASIS,
                "samples": [],
            }
        predicted_runs = observed.get(frame["id"], set())
        image_status, _path = local_media_file(store, frame["path"])
        frame_index = frame["frame_index"]
        sources[asset_id]["samples"].append(
            {
                "frame_id": frame["id"],
                "frame_index": frame_index
                if type(frame_index) is int and frame_index >= 0
                else None,
                # Keep the saved timestamp; never fabricate one for a legacy null value.
                "timestamp_seconds": _number(frame["timestamp_seconds"]),
                "image_url": f"/api/frames/{quote(frame['id'], safe='')}/image",
                "image_available": image_status == "available",
                "predicted_run_ids": sorted(predicted_runs),
                "prediction_count": len(predicted_runs),
                "complete": len(predicted_runs) == len(lanes),
            }
        )
    for source in sources.values():
        # Stable sorting preserves the saved selection order for coincident or
        # unavailable times; distinct video sources always have separate timelines.
        source["samples"].sort(
            key=lambda sample: (
                sample["timestamp_seconds"] is None,
                sample["timestamp_seconds"] or 0,
            )
        )
    return {
        "version": "iris-comparison-replay-v1",
        "mode": "saved-samples",
        "continuous_inference": False,
        "timestamps_approximate": True,
        "media_check": "path_and_size_only",
        "sources": list(sources.values()),
        "still_frame_ids": still_frame_ids,
    }
