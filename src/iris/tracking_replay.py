"""Replay complete temporal detection caches without running their detector.

Reports are standalone evidence. No workspace jobs or tracker-result tables are
created, and a cancelled or failed pass never publishes a complete report.
"""

import json
import os
import sqlite3
import tempfile
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

from iris.store import SCHEMA_VERSION, TABLES, Store
from iris.temporal import _digest
from iris.temporal_detections import get_detection_cache, read_detection_cache
from iris.tracking_contracts import (
    make_profile,
    profile_hash,
    semantic_frame,
    validate_profile,
    validate_tracking_frame,
)

REPORT_SCHEMA = "iris-tracking-replay-v1"


class TrackingReplayCancelled(RuntimeError):
    """Replay stopped without publishing an incomplete report as a success."""


class ReadOnlyReplayStore(Store):
    """Use existing Store readers without initialization, migrations or writes."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.db_path = self.root / "iris.sqlite3"
        if not self.db_path.is_file():
            raise ValueError("The source is not an initialized IRIS workspace")
        with self.connect() as conn:
            if conn.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION:
                raise ValueError("Open this workspace with IRIS to upgrade it before replay")
            self.columns = {
                table: {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
                for table in TABLES
            }

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.db_path.as_uri() + "?mode=ro", uri=True, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA query_only=ON")
            with conn:
                yield conn
        finally:
            conn.close()


def _stop(cancelled):
    if cancelled is not None and cancelled():
        raise TrackingReplayCancelled("Tracking replay cancelled; no complete report published")


def _profile(cache, algorithm, profile, class_ids):
    known = {entry["id"] for entry in cache["config"]["detector"]["classes"]}
    if profile is not None:
        if algorithm is not None or class_ids is not None:
            raise ValueError("Choose a complete profile or tracker/class options, not both")
        checked = validate_profile(profile)
    else:
        if class_ids is not None and (
            not isinstance(class_ids, list)
            or not class_ids
            or any(type(value) is not int or value not in known for value in class_ids)
            or len(set(class_ids)) != len(class_ids)
        ):
            raise ValueError("Class IDs must be distinct native labels from this detector cache")
        checked = make_profile(
            algorithm or "bytetrack", class_ids=sorted(known if class_ids is None else class_ids)
        )
    if not set(checked["class_ids"]) <= known:
        raise ValueError("Profile classes are not native labels from this detector cache")
    if cache["config"]["detector"]["min_score"] > checked["low_threshold"]:
        raise ValueError(
            "The saved cache score floor is above this tracker's low threshold; "
            "explicitly calculate a new lower-floor detector cache where supported, "
            "or raise the tracker low threshold in an explicit profile"
        )
    return checked


def _source_image(store, frame):
    # These imports never load detector weights or import Torch. They are only
    # needed for a profile that explicitly requests image-based compensation.
    import numpy as np

    from iris.inference import _load_verified_frame
    from iris.media import _file_hash

    saved = store.get("frames", frame["frame_id"])
    if saved is None:
        raise ValueError(f"Source frame {frame['frame_id']} is missing")
    if [saved["width"], saved["height"]] != frame["input_size"] or _file_hash(
        store.artifact_path(saved["path"])
    ) != frame["file_sha256"]:
        raise ValueError("GMC source PNG no longer matches the frozen detection cache")
    with _load_verified_frame(store, saved, frame["frame_sha256"]) as image:
        return np.asarray(image, dtype=np.uint8)[:, :, ::-1].copy()


def _factory(profile):
    from iris.tracking import make_tracker

    return make_tracker(profile)


def replay_detection_cache(
    store,
    cache_id,
    *,
    algorithm=None,
    profile=None,
    class_ids=None,
    repeats=2,
    cancelled=None,
):
    """Produce a complete report using identical saved inputs in every pass.

    The tracker owns association validation. Timing-independent semantic hashes
    compare observed repeatability for this input and runtime, not a general
    deterministic guarantee or a tracking-quality score.
    """
    if type(repeats) is not int or not 1 <= repeats <= 5:
        raise ValueError("Replay repeats must be an integer between 1 and 5")
    _stop(cancelled)
    cache = get_detection_cache(store, cache_id)
    checked = _profile(cache, algorithm, profile, class_ids)
    # Keep all saved scores of the selected classes, including below the tracker
    # low threshold. The adapter must explicitly account for unassigned rows.
    view = read_detection_cache(store, cache_id, class_ids=checked["class_ids"])
    sequence = view["sequence"]
    profile_sha256 = profile_hash(checked)
    requires_images = checked["algorithm"] == "botsort" and checked["gmc_method"] != "none"
    passes = []
    runtime_sha256 = None
    for pass_index in range(repeats):
        _stop(cancelled)
        setup_start = time.perf_counter()
        tracker = _factory(deepcopy(checked))
        tracker.reset(sequence["id"])
        setup_ms = (time.perf_counter() - setup_start) * 1000
        metadata = deepcopy(tracker.metadata)
        current_runtime_sha256 = _digest(metadata)
        if runtime_sha256 is not None and current_runtime_sha256 != runtime_sha256:
            raise ValueError("Tracker runtime or source provenance changed between replay passes")
        runtime_sha256 = current_runtime_sha256
        frame_results, image_reads = [], []
        replay_start = time.perf_counter()
        for position, frame in enumerate(view["frames"]):
            _stop(cancelled)
            image, image_read_ms = None, 0.0
            if requires_images:
                image_start = time.perf_counter()
                image = _source_image(store, frame)
                image_read_ms = (time.perf_counter() - image_start) * 1000
                _stop(cancelled)
            result = validate_tracking_frame(
                tracker.update(deepcopy(frame), image=image), frame, checked
            )
            if result["sequence_id"] != sequence["id"] or result["update_index"] != position + 1:
                raise ValueError("Tracker did not reset to the requested sequence and update order")
            _stop(cancelled)
            frame_results.append(deepcopy(result))
            image_reads.append({"frame_id": frame["frame_id"], "image_read_ms": image_read_ms})
        replay_ms = (time.perf_counter() - replay_start) * 1000
        _stop(cancelled)
        tracker.verify_runtime()
        if _digest(tracker.metadata) != runtime_sha256:
            raise ValueError("Tracker runtime or source provenance changed during replay")
        semantics = {
            "profile_sha256": profile_sha256,
            "cache_fingerprint": view["cache_fingerprint"],
            "sequence_sha256": cache["config"]["sequence_sha256"],
            "frames": [semantic_frame(result) for result in frame_results],
        }
        passes.append(
            {
                "pass_index": pass_index,
                "semantic_sha256": _digest(semantics),
                "frames": frame_results,
                "metadata": metadata,
                "runtime_sha256": runtime_sha256,
                "image_reads": image_reads,
                "timing": {
                    "adapter_setup_ms": setup_ms,
                    "replay_ms": replay_ms,
                    "source_image_read_ms": sum(row["image_read_ms"] for row in image_reads),
                },
            }
        )
    hashes = [item["semantic_sha256"] for item in passes]
    _stop(cancelled)
    return {
        "schema": REPORT_SCHEMA,
        "complete": True,
        "cache": {
            "id": cache_id,
            "fingerprint": view["cache_fingerprint"],
            "result_sha256": view["result_sha256"],
            "config": deepcopy(cache["config"]),
            "limitations": view["limitations"],
        },
        "sequence": sequence,
        "profile": checked,
        "profile_sha256": profile_sha256,
        "input_filter": view["filter"],
        "source_payloads": [
            {
                "frame_id": frame["frame_id"],
                "payload_sha256": frame["stored_payload_sha256"],
                "producer_job_id": frame["producer_job_id"],
                "execution_signature_sha256": frame["execution_signature_sha256"],
            }
            for frame in view["frames"]
        ],
        "detector_executions": {
            attempt["result"]["execution"]["signature_sha256"]: deepcopy(
                attempt["result"]["execution"]["signature"]
            )
            for attempt in cache["attempts"]
            if attempt["result"]["execution"] is not None
        },
        "excluded_class_ids": sorted(
            {entry["id"] for entry in cache["config"]["detector"]["classes"]}
            - set(checked["class_ids"])
        ),
        "source_images_required": requires_images,
        "passes": passes,
        "repeatability": {
            "status": "not_checked"
            if repeats == 1
            else "observed_match"
            if len(set(hashes)) == 1
            else "observed_mismatch",
            "passes": repeats,
            "semantic_sha256": hashes,
            "scope": "Observed outputs for these saved inputs and this runtime; excludes timings",
        },
        "timing_scope": {
            "adapter_setup_ms": "Adapter creation and sequence reset, including first imports",
            "replay_ms": "Frame loop with optional verified image reads, tracker and result copies",
            "source_image_read_ms": "PNG/pixel verification, decode and BGR copy",
            "excluded": "Detector inference, cache/runtime verification and report writing",
        },
        "limitations": (
            "No temporal reference comparison or quality ranking. Native memory counts analyzed "
            "updates, not wall-clock seconds; source gaps are preserved without synthetic empty "
            "updates. Repeated matching outputs are observed evidence, not a global guarantee. "
            "Cached detector timings are not measured end-to-end tracking performance."
        ),
    }


def replay_to_file(store, cache_id, destination, **settings):
    """Publish a successful JSON report atomically, without replacing any path."""
    destination = Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Replay report already exists: {destination}")
    if not destination.parent.is_dir():
        raise ValueError("The replay report destination directory must already exist")
    report = replay_detection_cache(store, cache_id, **settings)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination.parent, prefix=".iris-replay-", delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _stop(settings.get("cancelled"))
        # Hard-link publication fails if a file/symlink appeared during replay.
        # Both paths share a directory and therefore a filesystem.
        os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return report
