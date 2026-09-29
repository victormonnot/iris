"""Local outbound-image previews bound to one explicit, budgeted API request."""

import base64
import hashlib
import math
from datetime import UTC, datetime, timedelta

from iris.assistance_provider import _encode
from iris.inference import _load_verified_frame
from iris.store import Store, new_id, now

PREVIEW_LIFETIME = timedelta(minutes=30)


def read_images(store: Store, preview: dict) -> list[bytes]:
    images = []
    for item in preview["images"]:
        path = store.artifact_path(item["path"])
        if path.stat().st_size != item["size_bytes"]:
            raise ValueError("Preview image changed; create a new preview")
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != item["sha256"]:
            raise ValueError("Preview image changed; create a new preview")
        images.append(content)
    if len(images) != 1 + len(preview["candidates"]):
        raise ValueError("Preview image count does not match the reviewed candidates")
    return images


def public_preview(preview: dict) -> dict:
    config = preview["config"]
    return {
        "id": preview["id"],
        "frame_id": preview["frame_id"],
        **config["provider"],
        "expires_at": preview["expires_at"],
        "candidate_count": len(preview["candidates"]),
        "cost": config["estimated_cost"],
        "instructions": config["instructions"],
        "base_revision": config["base_revision"],
        "deployment_scope": "Global",
        "data_notice": "Selected scene and crops will be sent to Alibaba Cloud. "
        "Frankfurt workspace uses Global deployment scope; "
        "this does not guarantee inference stays in the EU.",
        "images": [
            {
                **{key: value for key, value in item.items() if key != "path"},
                "url": f"/api/assist-previews/{preview['id']}/images/{index}",
            }
            for index, item in enumerate(preview["images"])
        ],
    }


def preview_assistance(store: Store, frame_id: str, **fields) -> dict:
    from iris.assistance import _prepare

    if fields.get("provider") != "alibaba":
        raise ValueError("An external preview requires the Alibaba API provider")
    frame, candidates, config = _prepare(store, frame_id, **fields)
    preview_id = new_id()
    directory = store.root / "assistance" / "previews" / preview_id
    directory.mkdir(parents=True)
    images = []
    with _load_verified_frame(store, frame, config["frame_sha256"]) as image:
        encoded = [("scene", "Selected scene", _encode(image, 1024))]
        for candidate in candidates:
            x1, y1, x2, y2 = candidate["box"]
            crop = image.crop((math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2)))
            encoded.append(
                ("crop", f"{candidate['id']} · {candidate['label']}", _encode(crop, 320))
            )
        for index, (kind, label, data) in enumerate(encoded):
            content = base64.b64decode(data, validate=True)
            path = directory / f"{index}.jpg"
            path.write_bytes(content)
            images.append(
                {
                    "kind": kind,
                    "label": label,
                    "path": str(path.relative_to(store.root)),
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "size_bytes": len(content),
                }
            )
    preview = store.insert(
        "assistance_previews",
        {
            "id": preview_id,
            "frame_id": frame_id,
            "config": config,
            "candidates": candidates,
            "images": images,
            "created_at": now(),
            "expires_at": (datetime.now(UTC) + PREVIEW_LIFETIME).isoformat(),
        },
    )
    return public_preview(preview)


def confirmed_preview(
    store: Store,
    frame_id: str,
    config: dict,
    candidates: list,
    preview_id: str | None,
    allow_external: bool,
    max_cost_usd,
) -> dict:
    if allow_external is not True or not preview_id:
        raise ValueError("Preview the outgoing images and explicitly approve external processing")
    preview = store.get("assistance_previews", preview_id)
    if preview is None or preview["frame_id"] != frame_id:
        raise ValueError("The preview does not belong to this frame")
    if preview["job_id"] is not None:
        raise ValueError("This preview has already been used; create a new preview")
    if datetime.fromisoformat(preview["expires_at"]) <= datetime.now(UTC):
        raise ValueError("The external preview expired; create a new preview")
    if preview["config"] != config or preview["candidates"] != candidates:
        raise ValueError("The review selection, model or pricing changed; create a new preview")
    if (
        type(max_cost_usd) not in (float, int)
        or not math.isfinite(max_cost_usd)
        or max_cost_usd < config["estimated_cost"]["upper_bound_usd"]
    ):
        raise ValueError("The approved budget is below the conservative request cost bound")
    read_images(store, preview)
    return preview
