"""Persisted, bounded video storyboards and human-selected extraction passages."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import shutil
from datetime import UTC, datetime, timedelta

import cv2
from PIL import Image

from iris import remote_provider
from iris.assistance_provider import ProviderConfig, provider_status
from iris.media import _verified_video_source, _video_position_error
from iris.store import Store, new_id, now
from iris.video_sampling import evenly_spaced_indices, plan_extraction

PREVIEW_LIFETIME = timedelta(minutes=30)
MAX_IMAGE_BYTES = 1024 * 1024
PASSAGE_ALGORITHM = "iris-video-passages-v1"


def _provider_status(provider: str, model=None, endpoint=None) -> dict:
    if provider == "ollama":
        defaults = ProviderConfig.from_env()
        config = {
            "endpoint": endpoint or defaults.endpoint,
            "model": defaults.model if model is None else model,
        }
        return provider_status(config)
    if provider == "alibaba":
        config = {"endpoint": endpoint or os.environ.get(remote_provider.ENDPOINT_ENV, "")}
        if model is not None:
            config["model"] = model
        return remote_provider.provider_status(config)
    raise ValueError("Unknown video review provider")


def _ready(status: dict):
    if status.get("status") != "ready":
        raise RuntimeError(status.get("reason") or "The video review provider is unavailable")


def _estimate(config: dict, count: int) -> dict:
    # This ceiling uses the documented maximum input, covering 1–12 samples.
    return remote_provider.conservative_estimate(config)


def _fingerprint(config: dict, images: list) -> str:
    return hashlib.sha256(
        json.dumps({"config": config, "images": images}, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def _samples(record: dict) -> list[dict]:
    return [
        {key: image[key] for key in ("id", "frame_index", "timestamp_seconds")}
        for image in record["images"]
    ]


def _record(store: Store, review_id: str) -> dict:
    record = store.get("video_reviews", review_id)
    if record is None:
        raise ValueError("Video review not found")
    return record


def _public(store: Store, record: dict) -> dict:
    job = store.get("jobs", record["job_id"]) if record["job_id"] else None
    status = (
        job["status"]
        if job
        else (
            "expired"
            if datetime.fromisoformat(record["expires_at"]) <= datetime.now(UTC)
            else "preview"
        )
    )
    return {
        **record,
        "status": status,
        "job": job,
        "images": [
            {
                **{key: value for key, value in sample.items() if key != "path"},
                "url": f"/api/video-reviews/{record['id']}/images/{index}",
            }
            for index, sample in enumerate(record["images"])
        ],
    }


def get_review(store: Store, review_id: str) -> dict:
    return _public(store, _record(store, review_id))


def list_reviews(store: Store, asset_id: str) -> list[dict]:
    return [_public(store, record) for record in store.list("video_reviews", asset_id=asset_id)]


def prepare_review(
    store: Store,
    asset_id: str,
    *,
    provider="ollama",
    model=None,
    start_seconds=0,
    end_seconds=None,
    sample_count=8,
    instructions="",
) -> dict:
    """Decode the exact outbound storyboard locally; never perform generation."""
    from iris.video_review_provider import PROMPT_VERSION

    if type(sample_count) is not int or not 2 <= sample_count <= 12:
        raise ValueError("sample_count must be an integer between 2 and 12")
    if not isinstance(instructions, str) or len(instructions) > 2000:
        raise ValueError("Additional instructions are limited to 2000 characters")
    asset = store.get("assets", asset_id)
    if asset is None or asset["kind"] != "video":
        raise ValueError("Video review requires an imported video")
    plan = plan_extraction(
        asset["metadata"],
        {
            "sampling_mode": "uniform",
            "start_seconds": start_seconds,
            "end_seconds": end_seconds,
            "max_frames": sample_count,
        },
    )
    status = _provider_status(provider, model)
    _ready(status)
    config = {
        "provider": {
            "provider": provider,
            "endpoint": status["endpoint"],
            "model": status["model"],
        },
        "model_digest": status.get("model_digest"),
        "model_version": status.get("version"),
        "source_sha256": asset["sha256"],
        "source_metadata": asset["metadata"],
        "plan": plan,
        "instructions": instructions.strip(),
        "prompt_version": PROMPT_VERSION,
    }
    if provider == "alibaba":
        config["estimated_cost"] = _estimate(config["provider"], plan["planned_count"])
    source = _verified_video_source(store, asset)
    review_id = new_id()
    directory = store.root / "video_reviews" / review_id
    directory.mkdir(parents=True)
    capture = cv2.VideoCapture(str(source))
    images = []
    try:
        if not capture.isOpened():
            raise ValueError("The original video is missing or unreadable")
        for index, position in enumerate(plan["positions"]):
            frame_index = position["frame_index"]
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index):
                raise _video_position_error("seek to", frame_index)
            ok, pixels = capture.read()
            if not ok or pixels is None:
                raise _video_position_error("decode", frame_index)
            with Image.fromarray(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)) as image:
                if image.size != (asset["metadata"]["width"], asset["metadata"]["height"]):
                    raise ValueError("Video dimensions changed while preparing the storyboard")
                image.thumbnail((512, 512), Image.Resampling.LANCZOS)
                output = io.BytesIO()
                image.save(output, format="JPEG", quality=85)
                content = output.getvalue()
                path = directory / f"{index}.jpg"
                path.write_bytes(content)
                images.append(
                    {
                        "id": f"s{index + 1}",
                        **position,
                        "width": image.width,
                        "height": image.height,
                        "path": path.relative_to(store.root).as_posix(),
                        "sha256": hashlib.sha256(content).hexdigest(),
                        "size_bytes": len(content),
                    }
                )
        # Decoding is not atomic with filesystem changes: verify the source again.
        _check_source(store, {"asset_id": asset_id, "config": config})
        record = store.insert(
            "video_reviews",
            {
                "id": review_id,
                "asset_id": asset_id,
                "config": config,
                "images": images,
                "metadata": {"preview_sha256": _fingerprint(config, images)},
                "created_at": now(),
                "expires_at": (datetime.now(UTC) + PREVIEW_LIFETIME).isoformat(),
            },
        )
        return _public(store, record)
    except BaseException:
        shutil.rmtree(directory, ignore_errors=True)
        raise
    finally:
        capture.release()


def read_review_images(store: Store, record: dict) -> list[bytes]:
    """Return only bounded, intact JPEGs matching the frozen sampling positions."""
    items = record.get("images")
    config = record.get("config", {})
    plan = config.get("plan", {})
    if not isinstance(items, list) or not 1 <= len(items) <= 12:
        raise ValueError("Video review requires 1–12 preview images")
    if len(items) != plan.get("planned_count") or len(items) != len(plan.get("positions", [])):
        raise ValueError("Storyboard image count changed; prepare a new preview")
    if record.get("metadata", {}).get("preview_sha256") != _fingerprint(config, items):
        raise ValueError("Storyboard settings or images changed; prepare a new preview")
    images = []
    for index, (item, position) in enumerate(zip(items, plan["positions"], strict=True)):
        if (
            item.get("id") != f"s{index + 1}"
            or item.get("frame_index") != position["frame_index"]
            or item.get("timestamp_seconds") != position["timestamp_seconds"]
            or type(item.get("size_bytes")) is not int
            or not 1 <= item["size_bytes"] <= MAX_IMAGE_BYTES
        ):
            raise ValueError("Invalid storyboard image metadata")
        expected = f"video_reviews/{record['id']}/{index}.jpg"
        if item.get("path") != expected:
            raise ValueError("Storyboard image path changed; prepare a new preview")
        path = store.artifact_path(item["path"])
        if not path.is_file() or path.stat().st_size != item["size_bytes"]:
            raise ValueError("Storyboard image changed; prepare a new preview")
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != item["sha256"]:
            raise ValueError("Storyboard image changed; prepare a new preview")
        try:
            with Image.open(io.BytesIO(content)) as image:
                if (
                    image.format != "JPEG"
                    or image.size != (item.get("width"), item.get("height"))
                    or not 1 <= image.width <= 512
                    or not 1 <= image.height <= 512
                ):
                    raise ValueError("Invalid storyboard JPEG dimensions or encoding")
                image.verify()
        except OSError as exc:
            raise ValueError("Invalid storyboard JPEG") from exc
        images.append(content)
    return images


def _check_source(store: Store, record: dict, cancelled=None) -> dict:
    asset = store.get("assets", record["asset_id"])
    config = record["config"]
    if (
        asset is None
        or asset["kind"] != "video"
        or asset["sha256"] != config["source_sha256"]
        or asset["metadata"] != config["source_metadata"]
    ):
        raise ValueError("Video source or timing metadata changed; prepare a new review")
    if _verified_video_source(store, asset, cancelled) is None:
        raise InterruptedError("Video review cancelled")
    return asset


def _check_source_row(conn, record: dict):
    asset = conn.execute(
        "SELECT kind,sha256,metadata FROM assets WHERE id=?", (record["asset_id"],)
    ).fetchone()
    if (
        asset is None
        or asset["kind"] != "video"
        or asset["sha256"] != record["config"]["source_sha256"]
        or json.loads(asset["metadata"]) != record["config"]["source_metadata"]
    ):
        raise ValueError("Video source or timing metadata changed; prepare a new review")


def _check_provider(config: dict, status=None):
    from iris.video_review_provider import PROMPT_VERSION

    status = status or _provider_status(**config["provider"])
    _ready(status)
    if (
        any(status.get(key) != value for key, value in config["provider"].items())
        or status.get("model_digest") != config.get("model_digest")
        or status.get("version") != config.get("model_version")
        or config.get("prompt_version") != PROMPT_VERSION
    ):
        raise ValueError("The video review model or provider changed; prepare a new preview")


def _budget(config: dict, max_cost_usd):
    cost = _estimate(config["provider"], config["plan"]["planned_count"])
    if (
        cost != config.get("estimated_cost")
        or type(max_cost_usd) not in (int, float)
        or not math.isfinite(max_cost_usd)
        or max_cost_usd < cost["upper_bound_usd"]
    ):
        raise ValueError("API pricing changed or the approved budget is below the request ceiling")


def queue_review(
    store: Store, jobs, review_id: str, *, allow_external=False, max_cost_usd=None
) -> dict:
    record = _record(store, review_id)
    if record["job_id"] is not None:
        raise ValueError("This video preview has already been used; prepare a new preview")
    if datetime.fromisoformat(record["expires_at"]) <= datetime.now(UTC):
        raise ValueError("Video preview expired; prepare a new preview")
    config = record["config"]
    external = config["provider"]["provider"] == "alibaba"
    if not external and (allow_external or max_cost_usd is not None):
        raise ValueError("External approval fields are only valid for API reviews")
    if external:
        if allow_external is not True:
            raise ValueError("Explicitly approve sending the displayed storyboard to the API")
        _budget(config, max_cost_usd)
    read_review_images(store, record)
    _check_source(store, record)
    _check_provider(config)
    metadata = dict(record["metadata"])
    if external:
        metadata["consent"] = {
            "allow_external": True,
            "max_cost_usd": max_cost_usd,
            "approved_at": now(),
            "preview_sha256": metadata["preview_sha256"],
            "image_hashes": [image["sha256"] for image in record["images"]],
        }
    job_id = new_id()
    with jobs.guard, store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        _check_source_row(conn, record)
        latest = conn.execute("SELECT * FROM video_reviews WHERE id=?", (review_id,)).fetchone()
        if (
            latest["job_id"] is not None
            or latest["expires_at"] <= now()
            or json.loads(latest["config"]) != config
            or json.loads(latest["images"]) != record["images"]
            or json.loads(latest["metadata"]) != record["metadata"]
        ):
            raise ValueError("Video preview changed, expired or already used")
        active = conn.execute(
            "SELECT 1 FROM video_reviews v JOIN jobs j ON j.id=v.job_id "
            "WHERE v.asset_id=? AND j.status IN ('queued','running')",
            (record["asset_id"],),
        ).fetchone()
        if active:
            raise RuntimeError("A video review is already queued or running for this video")
        conn.execute(
            "INSERT INTO jobs (id,kind,status,params,message,created_at) VALUES (?,?,?,?,?,?)",
            (
                job_id,
                "video_review",
                "queued",
                json.dumps({"asset_id": record["asset_id"], "video_review_id": review_id}),
                "Waiting for video storyboard review",
                now(),
            ),
        )
        conn.execute(
            "UPDATE video_reviews SET job_id=?,metadata=? WHERE id=?",
            (job_id, json.dumps(metadata, allow_nan=False), review_id),
        )
    return store.get("jobs", job_id)


def _job_active(store: Store, record: dict) -> bool:
    job = store.get("jobs", record["job_id"]) if record["job_id"] else None
    return bool(
        job
        and job["status"] == "running"
        and not job["cancel_requested"]
        and job["kind"] == "video_review"
        and job["params"] == {"asset_id": record["asset_id"], "video_review_id": record["id"]}
    )


def _published_result(record: dict, reviewed: dict) -> dict:
    from iris.video_review_provider import validate_review

    validated = validate_review(
        {"summary": reviewed["summary"], "passages": reviewed["passages"]}, _samples(record)
    )
    samples = {image["id"]: image for image in record["images"]}
    plan = record["config"]["plan"]
    passages = []
    for index, passage in enumerate(validated["passages"]):
        first = samples[passage["start_sample_id"]]
        last = samples[passage["end_sample_id"]]
        passages.append(
            {
                **passage,
                "id": f"p{index + 1}",
                "start_frame_index": first["frame_index"],
                "end_frame_index": last["frame_index"],
                "start_seconds": first["timestamp_seconds"],
                "end_seconds": min((last["frame_index"] + 1) / plan["fps"], plan["end_seconds"]),
            }
        )
    return {"summary": validated["summary"], "passages": passages}


def run_video_review(store: Store, review_id: str, progress, cancelled, reviewer_factory=None):
    from iris.video_review_provider import VideoReviewer

    record = _record(store, review_id)
    outcome = {"video_review_id": review_id, "asset_id": record["asset_id"], "cancelled": False}
    if cancelled() or not _job_active(store, record):
        return {**outcome, "cancelled": True}
    metadata = dict(record["metadata"])
    try:
        config = record["config"]
        if metadata.get("attempted_at"):
            raise ValueError("This video review was already attempted; prepare a new preview")
        if datetime.fromisoformat(record["expires_at"]) <= datetime.now(UTC):
            raise ValueError("Video preview expired before review; prepare a new preview")
        images = read_review_images(store, record)
        _check_source(store, record, cancelled)
        _check_provider(config)
        if config["provider"]["provider"] == "alibaba":
            consent = metadata.get("consent", {})
            if (
                consent.get("allow_external") is not True
                or consent.get("preview_sha256") != metadata["preview_sha256"]
                or consent.get("image_hashes") != [image["sha256"] for image in record["images"]]
            ):
                raise ValueError("Video review does not match an approved storyboard")
            _budget(config, consent.get("max_cost_usd"))
        reviewer = (reviewer_factory or VideoReviewer)(config=config["provider"])
        provider_metadata = reviewer.metadata
        if (
            provider_metadata.get("model_digest") != config.get("model_digest")
            or provider_metadata.get("version") != config.get("model_version")
            or any(provider_metadata.get(key) != value for key, value in config["provider"].items())
        ):
            raise ValueError("Video review model changed before generation")
        if cancelled():
            return {**outcome, "cancelled": True}
        metadata.update(attempted_at=now(), provider=provider_metadata)
        with store.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            latest = conn.execute(
                "SELECT metadata,config,images,expires_at FROM video_reviews WHERE id=?",
                (review_id,),
            ).fetchone()
            if json.loads(latest["metadata"]).get("attempted_at"):
                raise ValueError("This video review was already attempted")
            if (
                json.loads(latest["config"]) != config
                or json.loads(latest["images"]) != record["images"]
                or json.loads(latest["metadata"]) != record["metadata"]
            ):
                raise ValueError("Storyboard or approval changed before generation")
            if latest["expires_at"] <= now():
                raise ValueError("Video preview expired before generation")
            _check_source_row(conn, record)
            if cancelled() or not _job_active(store, record):
                return {**outcome, "cancelled": True}
            conn.execute(
                "UPDATE video_reviews SET metadata=? WHERE id=?",
                (json.dumps(metadata, allow_nan=False), review_id),
            )
        progress(0.2, "Reviewing the displayed storyboard; unsampled events may be missed")
        if cancelled() or not _job_active(store, record):
            return {**outcome, "cancelled": True}
        _check_source(store, record, cancelled)
        reviewed = reviewer.review(images, _samples(record), config["instructions"])
        metadata["provider"] = reviewed["metadata"]
        store.update(
            "video_reviews",
            review_id,
            {
                "metadata": metadata,
                "prompt": reviewed["prompt"],
                "raw_response": reviewed["raw_response"],
            },
        )
        if cancelled() or not _job_active(store, record):
            return {**outcome, "cancelled": True}
        result = _published_result(record, reviewed)
        _check_source(store, record, cancelled)
        outcome.update(passages_count=len(result["passages"]), result=result)
        with store.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            _check_source_row(conn, record)
            if cancelled() or not _job_active(store, record):
                return {**outcome, "cancelled": True, "result": None}
            latest = conn.execute(
                "SELECT config,images FROM video_reviews WHERE id=?", (review_id,)
            ).fetchone()
            if (
                json.loads(latest["config"]) != config
                or json.loads(latest["images"]) != record["images"]
            ):
                raise ValueError("Storyboard changed before publication")
            # Publish result and terminal job state together: cancellation cannot
            # race between a visible proposal and the worker's terminal update.
            conn.execute(
                "UPDATE video_reviews SET result=? WHERE id=?",
                (json.dumps(result, allow_nan=False), review_id),
            )
            conn.execute(
                "UPDATE jobs SET status='succeeded',result=?,progress=1,finished_at=?,"
                "message='Video passages ready for human selection' WHERE id=?",
                (json.dumps(outcome, allow_nan=False), now(), record["job_id"]),
            )
        return outcome
    except InterruptedError:
        return {**outcome, "cancelled": True}
    except Exception as exc:
        updates = {"error": str(exc)}
        for key in ("raw_response", "prompt"):
            if getattr(exc, key, None) is not None:
                updates[key] = getattr(exc, key)
        if getattr(exc, "metadata", None):
            metadata["provider"] = exc.metadata
            updates["metadata"] = metadata
        store.update("video_reviews", review_id, updates)
        raise


def preview_passage_extraction(
    store: Store,
    review_id: str,
    *,
    passage_ids: list[str],
    frames_per_passage=8,
    context_seconds=2,
    coverage_frames=8,
) -> dict:
    """Union human-selected passage samples with optional full-range coverage."""
    record = _record(store, review_id)
    job = store.get("jobs", record["job_id"]) if record["job_id"] else None
    if job is None or job["status"] != "succeeded" or record["result"] is None:
        raise ValueError("Passage extraction requires a successfully completed video review")
    if (
        not isinstance(passage_ids, list)
        or not 1 <= len(passage_ids) <= 6
        or any(not isinstance(item, str) for item in passage_ids)
        or len(set(passage_ids)) != len(passage_ids)
    ):
        raise ValueError("Choose 1–6 distinct passage IDs")
    if type(frames_per_passage) is not int or not 1 <= frames_per_passage <= 50:
        raise ValueError("frames_per_passage must be between 1 and 50")
    if type(coverage_frames) is not int or not 0 <= coverage_frames <= 32:
        raise ValueError("coverage_frames must be between 0 and 32")
    if (
        type(context_seconds) not in (int, float)
        or not math.isfinite(context_seconds)
        or not 0 <= context_seconds <= 30
    ):
        raise ValueError("context_seconds must be between 0 and 30")
    read_review_images(store, record)
    asset = _check_source(store, record)
    stored = record["result"]
    validated = _published_result(
        record,
        {
            "summary": stored.get("summary"),
            "passages": [
                {
                    key: passage.get(key)
                    for key in ("start_sample_id", "end_sample_id", "reason", "uncertainty")
                }
                for passage in stored.get("passages", [])
            ],
        },
    )
    if validated != stored:
        raise ValueError("Stored video passages changed; prepare a new review")
    by_id = {passage["id"]: passage for passage in validated["passages"]}
    if not set(passage_ids) <= by_id.keys():
        raise ValueError("A chosen passage does not belong to this video review")
    base = record["config"]["plan"]
    indices = set()
    ranges = []
    for passage_id in sorted(passage_ids, key=lambda item: by_id[item]["start_frame_index"]):
        passage = by_id[passage_id]
        start = max(base["start_seconds"], passage["start_seconds"] - context_seconds)
        end = min(base["end_seconds"], passage["end_seconds"] + context_seconds)
        segment = plan_extraction(
            asset["metadata"],
            {
                "sampling_mode": "uniform",
                "start_seconds": start,
                "end_seconds": end,
                "max_frames": frames_per_passage,
            },
        )
        observed = [
            image["frame_index"]
            for image in record["images"]
            if passage["start_frame_index"] <= image["frame_index"] <= passage["end_frame_index"]
        ]
        # Retain actual observations cited by the model. Context sampling must
        # not move a one-frame request away from the evidence the human chose.
        retained = (
            {observed[(len(observed) - 1) // 2]}
            if frames_per_passage == 1
            else {observed[0], observed[-1]}
        )
        candidates = [
            position["frame_index"]
            for position in segment["positions"]
            if position["frame_index"] not in retained
        ]
        retained.update(
            candidates[index]
            for index in evenly_spaced_indices(
                0, len(candidates) - 1, frames_per_passage - len(retained)
            )
        )
        indices.update(retained)
        ranges.append({"passage_id": passage_id, "start_seconds": start, "end_seconds": end})
    if coverage_frames:
        coverage = plan_extraction(
            asset["metadata"],
            {
                "sampling_mode": "uniform",
                "start_seconds": base["start_seconds"],
                "end_seconds": base["end_seconds"],
                "max_frames": coverage_frames,
            },
        )
        indices.update(position["frame_index"] for position in coverage["positions"])
    positions = [
        {"frame_index": index, "timestamp_seconds": index / base["fps"]}
        for index in sorted(indices)
    ]
    return {
        **base,
        "algorithm": PASSAGE_ALGORITHM,
        "sampling_mode": "passages",
        "asset_id": asset["id"],
        "source_sha256": asset["sha256"],
        "source_metadata": asset["metadata"],
        "video_review_id": review_id,
        "passage_ids": [item["passage_id"] for item in ranges],
        "frames_per_passage": frames_per_passage,
        "context_seconds": context_seconds,
        "coverage_frames": coverage_frames,
        "ranges": ranges,
        "positions": positions,
        "max_frames": len(passage_ids) * frames_per_passage + coverage_frames,
        "planned_count": len(positions),
        "first_timestamp_seconds": positions[0]["timestamp_seconds"],
        "last_timestamp_seconds": positions[-1]["timestamp_seconds"],
    }


def validate_passage_extraction(store: Store, plan: dict) -> dict:
    """Recompute a queued passage plan before the extraction worker uses it."""
    if not isinstance(plan, dict) or plan.get("algorithm") != PASSAGE_ALGORITHM:
        raise ValueError("Invalid video passage extraction plan")
    checked = preview_passage_extraction(
        store,
        plan.get("video_review_id", ""),
        **{
            key: plan.get(key)
            for key in ("passage_ids", "frames_per_passage", "context_seconds", "coverage_frames")
        },
    )
    if checked != plan:
        raise ValueError("The video passage extraction plan changed; prepare it again")
    return checked
