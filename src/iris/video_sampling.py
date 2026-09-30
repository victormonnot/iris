"""Deterministic, bounded plans for sampling imported videos by nominal time."""

from __future__ import annotations

import math

ALGORITHM = "iris-video-sampling-v1"
TIMESTAMP_BASIS = "frame_index / nominal_fps"


def _finite_number(values: dict, key: str, default: float | None = None) -> float:
    value = values.get(key, default)
    try:
        if isinstance(value, bool):
            raise ValueError
        number = float(value)
        if not math.isfinite(number):
            raise ValueError
        return number
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{key} must be a finite number.") from exc


def evenly_spaced_indices(first: int, last: int, count: int) -> list[int]:
    """Choose sorted unique indices, including both ends when there is room."""
    count = min(count, last - first + 1)
    if count < 1:
        return []
    if count == 1:
        return [(first + last) // 2]
    span = last - first
    denominator = count - 1
    # Nearest-integer spacing without floats; ties consistently round down.
    return [
        first + (2 * index * span + denominator - 1) // (2 * denominator) for index in range(count)
    ]


def _first_at_or_after(seconds: float, fps: float, frame_count: int) -> int:
    # Comparing the same division used by provenance avoids ceil(seconds * fps)
    # rounding a boundary onto a frame whose displayed timestamp precedes it.
    low, high = 0, frame_count
    while low < high:
        middle = (low + high) // 2
        if middle / fps < seconds:
            low = middle + 1
        else:
            high = middle
    return low


def plan_extraction(metadata: dict, config: dict) -> dict:
    """Return at most 500 positions, without decoding media or allocating a full grid.

    Missing sampling_mode retains the historical interval-prefix contract.
    Uniform sampling uses the original frame grid within [start, end).
    """
    if not isinstance(metadata, dict) or not isinstance(config, dict):
        raise ValueError("Video metadata and extraction settings must be objects.")
    mode = config.get("sampling_mode", "interval")
    if mode not in ("interval", "uniform"):
        raise ValueError("sampling_mode must be interval or uniform.")
    start = _finite_number(config, "start_seconds", 0.0)
    if start < 0:
        raise ValueError("The start must be nonnegative.")
    maximum = config.get("max_frames", 100)
    if isinstance(maximum, bool) or not isinstance(maximum, int) or not 1 <= maximum <= 500:
        raise ValueError("max_frames must be an integer between 1 and 500.")
    dedup = config.get("dedup_hamming")
    if dedup is not None and (
        isinstance(dedup, bool) or not isinstance(dedup, int) or not 0 <= dedup <= 16
    ):
        raise ValueError("dedup_hamming must be null or an integer between 0 and 16.")
    fps = _finite_number(metadata, "fps")
    duration = _finite_number(metadata, "duration_seconds")
    total_frames = metadata.get("frame_count")
    if (
        fps <= 0
        or duration <= 0
        or isinstance(total_frames, bool)
        or not isinstance(total_frames, int)
        or total_frames < 1
    ):
        raise ValueError("Video FPS, duration and frame count must be positive.")
    try:
        if not math.isfinite(total_frames / fps):
            raise ValueError
    except (ValueError, OverflowError) as exc:
        raise ValueError("Video timing metadata is outside the supported numeric range.") from exc
    end = duration if config.get("end_seconds") is None else _finite_number(config, "end_seconds")
    if end <= start or start >= duration:
        raise ValueError("The time range must overlap a nonempty portion of the video.")
    end = min(end, duration)
    interval = None
    truncated = False
    if mode == "uniform":
        first = _first_at_or_after(start, fps, total_frames)
        last = _first_at_or_after(end, fps, total_frames) - 1
        targets = evenly_spaced_indices(first, last, maximum)
    else:
        interval = _finite_number(config, "interval_seconds", 1.0)
        if interval <= 0:
            raise ValueError("The interval must be positive.")
        interval = max(interval, 1.0 / fps)
        targets = []
        # Preserve the old floor/epsilon and interval cap exactly; inspect one
        # additional theoretical position only to report prefix truncation.
        for index in range(maximum + 1):
            seconds = start + index * interval
            if seconds >= end:
                break
            scaled = seconds * fps
            if not math.isfinite(scaled):
                raise ValueError("Video timing metadata is outside the supported numeric range.")
            frame_index = int(math.floor(scaled + 1e-7))
            if frame_index < total_frames and (not targets or frame_index != targets[-1]):
                if index == maximum:
                    truncated = True
                else:
                    targets.append(frame_index)
    if not targets:
        raise ValueError("The selected time range contains no sampleable video frames.")
    positions = [
        {"frame_index": frame_index, "timestamp_seconds": frame_index / fps}
        for frame_index in targets
    ]
    return {
        "algorithm": ALGORITHM,
        "sampling_mode": mode,
        "start_seconds": start,
        "end_seconds": end,
        "duration_seconds": duration,
        "fps": fps,
        "timestamp_basis": TIMESTAMP_BASIS,
        "max_frames": maximum,
        "planned_count": len(positions),
        "positions": positions,
        "first_timestamp_seconds": positions[0]["timestamp_seconds"],
        "last_timestamp_seconds": positions[-1]["timestamp_seconds"],
        "truncated": truncated,
        "interval_seconds": interval,
    }
