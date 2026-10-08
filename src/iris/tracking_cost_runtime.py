"""Fresh detector and tracker cost measurements, with no workspace mutations."""

import hashlib
import os
import platform
import sys
import time
from copy import deepcopy
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

from iris.tracking_cost_contracts import (
    LIMITATIONS,
    PROTOCOL,
    REPORT_SCHEMA,
    detector_recipe,
    digest,
    frame_schedule,
    next_frame,
    source_binding,
    summarize,
    validate_cost_report,
    validate_cost_request,
)


class TrackingCostCancelled(RuntimeError):
    """A stopped run must not publish a successful partial cost report."""


def _stop(cancelled):
    if cancelled is not None and cancelled():
        raise TrackingCostCancelled("Cost measurement cancelled; no complete report published")


def _prepare_detector(root, frozen, device):
    from iris.temporal_detector import prepare_detector

    inference = frozen["inference"]
    options = {
        "device": device,
        "min_score": frozen["min_score"],
        "inference_mode": inference["mode"],
    }
    if inference["mode"] == "tiled":
        options.update(
            tile_size=inference["tiling"]["tile_size"], overlap=inference["tiling"]["overlap"]
        )
    current = prepare_detector(root, frozen["model_id"], **options)
    if digest(detector_recipe(current)) != digest(detector_recipe(frozen)):
        raise ValueError("The current detector no longer matches this comparison's frozen recipe")
    return current


def _detector_factory(root, config):
    from iris.temporal_detector import detector_factory

    return detector_factory(root, config)


def _tracker_factory(profile):
    from iris.tracking import make_tracker

    return make_tracker(profile)


def _load_source(store, frozen):
    """Decode exactly the byte snapshot that was checked, never reopen after hashing."""
    from PIL import Image

    from iris.media import _pixel_hash

    saved = store.get("frames", frozen["frame_id"])
    if saved is None:
        raise ValueError("A frozen measurement frame is no longer available")
    if saved["sha256"] != frozen["sha256"] or [saved["width"], saved["height"]] != [
        frozen["width"],
        frozen["height"],
    ]:
        raise ValueError("A measurement frame changed since the sequence was frozen")
    content = store.artifact_path(saved["path"]).read_bytes()
    if hashlib.sha256(content).hexdigest() != frozen["file_sha256"]:
        raise ValueError("Measurement source PNG bytes no longer match the frozen sequence")
    with Image.open(BytesIO(content)) as source:
        if source.format != "PNG":
            raise ValueError("Measurement source must be a frozen PNG")
        image = source.convert("RGB")
        image.load()
    if image.size != (frozen["width"], frozen["height"]) or _pixel_hash(image) != frozen["sha256"]:
        image.close()
        raise ValueError("Measurement source pixels no longer match the frozen sequence")
    return image


def _rss_bytes():
    if not sys.platform.startswith("linux"):
        return None
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                fields = line.split()
                if len(fields) == 3 and fields[2] == "kB":
                    return int(fields[1]) * 1024
    except (OSError, ValueError):
        pass
    return None


def _process_peak_bytes():
    if not (sys.platform.startswith("linux") or sys.platform == "darwin"):
        return None
    try:
        import resource

        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value) * (1 if sys.platform == "darwin" else 1024)
    except (ImportError, OSError, ValueError):
        return None


class _Memory:
    def __init__(self, detector, device, actual_device):
        self.detector, self.device = detector, device
        self.actual_device = actual_device
        self.rss_start = self.peak = _rss_bytes()
        self.cuda_start = None
        if device == "cuda":
            if str(getattr(detector, "device", None)) != actual_device:
                raise ValueError("CUDA memory instrumentation requires the actual detector device")
            cuda = detector.torch.cuda
            cuda.synchronize(detector.device)
            cuda.reset_peak_memory_stats(detector.device)
            self.cuda_start = {
                "device": actual_device,
                "allocated_start_bytes": cuda.memory_allocated(detector.device),
                "reserved_start_bytes": cuda.memory_reserved(detector.device),
            }

    def sample(self):
        current = _rss_bytes()
        if current is not None:
            self.peak = max(current, self.peak or 0)
        return current

    def finish(self):
        end = self.sample()
        cuda_memory = None
        if self.cuda_start is not None:
            cuda = self.detector.torch.cuda
            cuda.synchronize(self.detector.device)
            cuda_memory = {
                **self.cuda_start,
                "allocated_peak_bytes": cuda.max_memory_allocated(self.detector.device),
                "reserved_peak_bytes": cuda.max_memory_reserved(self.detector.device),
            }
        return {
            "rss_start_bytes": self.rss_start,
            "rss_end_bytes": end,
            "rss_sampled_peak_bytes": self.peak,
            "process_lifetime_peak_bytes": _process_peak_bytes(),
            "cuda": cuda_memory,
        }


def _host():
    from iris.models import _cpu_name

    try:
        affinity = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = None
    return {
        "cpu": _cpu_name(),
        "platform": platform.platform(),
        "machine": platform.machine() or "unknown",
        "python": platform.python_version(),
        "logical_cpus": os.cpu_count(),
        "affinity_cpus": affinity,
    }


def _process_frame(
    store, frozen, detector, tracker, config, profile, cancelled, sequence_id, update_index
):
    from iris.temporal_detection_worker import _raw_prediction
    from iris.tiling import TiledInferenceCancelled, tiled_predict
    from iris.tracking_contracts import semantic_frame, validate_tracking_frame

    _stop(cancelled)
    started = time.perf_counter()
    with _load_source(store, frozen) as image:
        decoded = time.perf_counter()
        inference = config["inference"]
        try:
            prediction = (
                tiled_predict(
                    detector,
                    image,
                    inference["tiling"],
                    cancelled=lambda: bool(cancelled and cancelled()),
                )
                if inference["mode"] == "tiled"
                else detector.predict(image)
            )
        except TiledInferenceCancelled as exc:
            raise TrackingCostCancelled(
                "Cost measurement cancelled during tiled detection"
            ) from exc
        predicted = time.perf_counter()
        _stop(cancelled)
        _raw_prediction(prediction, frozen, config)
        selected = set(profile["class_ids"])
        detections = [
            {
                "detection_index": index,
                **{key: detection[key] for key in ("label_id", "label", "score", "box")},
            }
            for index, detection in enumerate(prediction["detections"])
            if detection["score"] >= config["min_score"] and detection["label_id"] in selected
        ]
        source = {
            "frame_id": frozen["frame_id"],
            "frame_index": frozen["frame_index"],
            "timestamp_seconds": frozen["timestamp_seconds"],
            "input_size": [frozen["width"], frozen["height"]],
            "detections": detections,
        }
        filtered = time.perf_counter()
        tracking_image = None
        if profile["gmc_method"] != "none":
            import numpy as np

            tracking_image = np.asarray(image, dtype=np.uint8)[:, :, ::-1].copy()
            converted = time.perf_counter()
        else:
            converted = filtered
        tracked = tracker.update(source, image=tracking_image)
        updated = time.perf_counter()
        checked = validate_tracking_frame(tracked, source, profile)
        if checked["sequence_id"] != sequence_id or checked["update_index"] != update_index:
            raise ValueError("Tracker did not reset to the requested sequence and update order")
    finished = time.perf_counter()
    detector_timing, tracker_timing = prediction["timing"], checked["timing"]
    timing = {
        "pipeline_ms": (finished - started) * 1000,
        "verify_decode_ms": (decoded - started) * 1000,
        "detector_call_ms": (predicted - decoded) * 1000,
        "filter_ms": (filtered - predicted) * 1000,
        "tracking_image_ms": (converted - filtered) * 1000,
        "tracker_call_ms": (updated - converted) * 1000,
        **{
            f"detector_{key}": detector_timing[key]
            for key in ("preprocess_ms", "inference_ms", "postprocess_ms")
        },
        **{f"detector_{key}": detector_timing.get(key, 0.0) for key in ("crop_ms", "merge_ms")},
        "tracker_gmc_ms": tracker_timing["gmc_ms"],
        "tracker_association_ms": tracker_timing["association_ms"],
        "tracker_adapter_ms": tracker_timing["total_ms"],
    }
    return {
        **{
            key: source[key]
            for key in ("frame_id", "frame_index", "timestamp_seconds", "input_size")
        },
        "timing": timing,
        "schedule": None,
        "work": {
            "detection_count": len(detections),
            "observation_count": len(checked["observations"]),
            "prediction_count": len(checked["predictions"]),
            "unassigned_count": len(checked["unassigned"]),
            "forward_passes": detector_timing.get("forward_passes", 1),
            "tile_count": detector_timing.get("tile_count", 0),
        },
        "outputs_sha256": digest(semantic_frame(checked)),
    }


def _pipeline_sources():
    return {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ("tracking_cost_runtime.py", "tracking_cost_contracts.py")
    }


def measure_tracking_cost(
    store,
    comparison,
    *,
    lane_index=0,
    device="cpu",
    repeats=1,
    policy="offline_all",
    cadence_fps=None,
    progress=None,
    cancelled=None,
    detector_factory=None,
    tracker_factory=None,
):
    """Measure a fresh bounded pipeline, returning only a complete portable report."""
    from iris.temporal_detector import verify_detector_metadata

    request = validate_cost_request(
        {
            "lane_index": lane_index,
            "device": device,
            "repeats": repeats,
            "policy": policy,
            "cadence_fps": cadence_fps,
        }
    )
    binding = source_binding(comparison, lane_index)
    sequence = deepcopy(comparison["report"]["sequence"])
    lane = comparison["report"]["lanes"][lane_index]["report"]
    profile = deepcopy(lane["profile"])
    config = _prepare_detector(store.root, lane["cache"]["config"]["detector"], device)
    _stop(cancelled)
    started_at = datetime.now(UTC).isoformat()
    started = time.perf_counter()
    detector = (detector_factory or _detector_factory)(store.root, config)
    detector_load_ms = (time.perf_counter() - started) * 1000
    signature = verify_detector_metadata(config, detector.metadata)
    _stop(cancelled)
    started = time.perf_counter()
    tracker = (tracker_factory or _tracker_factory)(deepcopy(profile))
    tracker.reset(sequence["id"])
    tracker_setup_ms = (time.perf_counter() - started) * 1000
    tracker_metadata = deepcopy(tracker.metadata)
    if progress:
        progress(0.0, "Warming up a fresh detector and tracker; warmup is excluded from samples")
    started = time.perf_counter()
    _process_frame(
        store,
        sequence["frames"][0],
        detector,
        tracker,
        config,
        profile,
        cancelled,
        sequence["id"],
        1,
    )
    warmup_ms = (time.perf_counter() - started) * 1000
    report = {
        "schema": REPORT_SCHEMA,
        "complete": True,
        "request": request,
        "source": binding,
        "sequence": sequence,
        "profile": profile,
        "detector_config": config,
        "execution": {
            "started_at": started_at,
            "host": _host(),
            "detector_signature": signature,
            "tracker_metadata": tracker_metadata,
            "tracker_metadata_sha256": digest(tracker_metadata),
            "pipeline_sources": _pipeline_sources(),
            "setup_ms": {
                "detector_load_ms": detector_load_ms,
                "tracker_setup_ms": tracker_setup_ms,
                "warmup_ms": warmup_ms,
            },
            "warmup": {
                "frame_id": sequence["frames"][0]["frame_id"],
                "passes": 1,
                "tracker_reset": True,
            },
        },
        "passes": [],
        "summary": None,
        "protocol": deepcopy(PROTOCOL),
        "limitations": list(LIMITATIONS),
    }
    sources = sequence["frames"]
    indices = [frame["frame_index"] for frame in sources]
    for pass_index in range(repeats):
        _stop(cancelled)
        tracker.reset(sequence["id"])
        memory = _Memory(detector, device, signature["device"])
        position, virtual_ms, frames, dropped = 0, 0.0, [], []
        loop_started = time.perf_counter()
        while position < len(sources):
            selected, omitted, start_ms = next_frame(indices, position, virtual_ms, request)
            dropped.extend(omitted)
            sample = _process_frame(
                store,
                sources[selected],
                detector,
                tracker,
                config,
                profile,
                cancelled,
                sequence["id"],
                len(frames) + 1,
            )
            _stop(cancelled)
            sample["schedule"] = frame_schedule(
                indices[selected], indices[0], start_ms, sample["timing"]["pipeline_ms"], request
            )
            if sample["schedule"] is not None:
                virtual_ms = sample["schedule"]["finish_ms"]
            frames.append(sample)
            memory.sample()
            position = selected + 1
            if progress:
                progress(
                    (pass_index + position / len(sources)) / repeats,
                    f"Cost repetition {pass_index + 1}/{repeats}: {len(frames)} processed, "
                    f"{len(dropped)} simulated drops",
                )
        loop_ms = (time.perf_counter() - loop_started) * 1000
        report["passes"].append(
            {
                "pass_index": pass_index,
                "frames": frames,
                "dropped_frame_indices": dropped,
                "wall_ms": loop_ms,
                "memory": memory.finish(),
            }
        )
    _stop(cancelled)
    tracker.verify_runtime()
    if digest(tracker.metadata) != digest(tracker_metadata):
        raise ValueError("Tracker runtime changed during the cost measurement")
    if digest(verify_detector_metadata(config, detector.metadata)) != digest(signature):
        raise ValueError("Detector runtime changed during the cost measurement")
    if report["execution"]["pipeline_sources"] != _pipeline_sources():
        raise ValueError("Cost measurement implementation changed during execution")
    report["summary"] = summarize(report)
    result = validate_cost_report(report, comparison)
    _stop(cancelled)
    return result
