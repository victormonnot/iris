"""Codec-free temporal fixtures with verified source bytes and real image pixels."""

import hashlib

from PIL import Image

from iris.media import _frame_record
from iris.store import new_id, now


def video_sequence(store, tmp_path, *, group="take-a", count=3, project_id="default", color=0):
    """Return ``(asset, frames)`` for one synthetic take, with 10 Hz source indices.

    The opaque video is never decoded: temporal publication checks its checksum
    and already extracted frames. PNG frames use the ordinary import writer.
    """
    session = store.insert(
        "sessions",
        {
            "id": new_id(),
            "project_id": project_id,
            "name": f"Synthetic {group}",
            "scene_group": group,
            "created_at": now(),
        },
    )
    identifier = new_id()
    data = f"Opaque temporal video fixture: {group}, {color}, {count}".encode()
    path = store.root / "assets" / f"{identifier}.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    asset = store.insert(
        "assets",
        {
            "id": identifier,
            "session_id": session["id"],
            "filename": f"{group}.mp4",
            "kind": "video",
            "path": path.relative_to(store.root).as_posix(),
            "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
            "metadata": {
                "media_type": "video/mp4",
                "width": 80,
                "height": 60,
                "fps": 10.0,
                "frame_count": count,
                "duration_seconds": count / 10,
                "timestamp_basis": "frame_index / nominal_fps",
            },
            "created_at": now(),
        },
    )
    frames = []
    for index in range(count):
        image = Image.new("RGB", (80, 60), ((color + index * 13) % 256, 60, 90))
        try:
            frames.append(
                _frame_record(
                    store,
                    asset,
                    image,
                    frame_index=index,
                    timestamp_seconds=index / 10,
                    extraction={"fixture": True},
                )
            )
        finally:
            image.close()
    return asset, frames
