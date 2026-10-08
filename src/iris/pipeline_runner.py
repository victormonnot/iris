"""Bounded local inputs and explicit evidence for the standalone pipeline.

Copied into exported bundles. No application, database, network or UI dependency.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import stat
import tempfile
from copy import deepcopy
from pathlib import Path, PurePosixPath

MAX_FRAMES = 10_000
MAX_JSON = 128 * 1024 * 1024
MAX_IMAGE = 32 * 1024 * 1024
MAX_PIXELS = 64_000_000
RUN_SCHEMA = "iris-pipeline-run-v1"
SCOPE = "Execution evidence only; no independent quality or real-time qualification."


def _json_value(value, depth=0):
    if depth > 24:
        raise ValueError("Run JSON exceeds its supported nesting depth")
    if isinstance(value, dict):
        if len(value) > 1000 or any(type(key) is not str for key in value):
            raise ValueError("Run JSON requires bounded objects with text keys")
        for key, item in value.items():
            _json_value(key, depth + 1)
            _json_value(item, depth + 1)
    elif isinstance(value, list):
        if len(value) > MAX_FRAMES:
            raise ValueError("Run JSON exceeds its supported array limit")
        for item in value:
            _json_value(item, depth + 1)
    elif type(value) is str:
        if len(value) > 20_000:
            raise ValueError("Run JSON contains oversized text")
        try:
            value.encode("utf-8")
        except UnicodeError as exc:
            raise ValueError("Run JSON must be valid UTF-8") from exc
    elif type(value) in (int, float):
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError("Run JSON contains a nonfinite or oversized number")
    elif value is not None and type(value) is not bool:
        raise ValueError("Run JSON contains an unsupported value")


def canonical(value):
    """Use the bundle's canonical encoding with bounds for a complete run report."""
    _json_value(value)
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    if len(raw) > MAX_JSON:
        raise ValueError("Run JSON exceeds 128 MiB")
    return raw


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _integer(value, name, minimum=0, maximum=MAX_FRAMES):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _text(value, name):
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 200
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError(f"{name} must be nonempty text of at most 200 characters")
    try:
        value.encode("utf-8")
    except UnicodeError as exc:
        raise ValueError(f"{name} must be valid UTF-8 text") from exc
    return value


def _regular(path):
    path = Path(os.path.abspath(path))
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError(f"Symbolic links are not accepted: {path}")
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError(f"Expected a regular local file: {path}")
    return path


def _sha_file(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _json(path):
    path = _regular(path)
    if path.stat().st_size > MAX_JSON:
        raise ValueError("JSON input exceeds 128 MiB")

    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    raw = path.read_bytes()
    if len(raw) > MAX_JSON:
        raise ValueError("JSON input exceeds 128 MiB")

    def invalid_constant(value):
        raise ValueError(f"Nonfinite JSON token: {value}")

    try:
        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid_constant)
        _json_value(value)
    except (UnicodeError, RecursionError, OverflowError) as exc:
        raise ValueError("Expected bounded finite UTF-8 JSON") from exc
    return value, hashlib.sha256(raw).hexdigest()


def _outside(path, directory):
    path = Path(os.path.abspath(path))
    if path.resolve().is_relative_to(Path(directory).resolve()):
        raise ValueError("Inputs and outputs must be outside the immutable bundle")
    return path


def write_report(path, report):
    """Publish a complete report atomically, without replacing an existing file."""
    path = Path(os.path.abspath(path))
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise ValueError("Output parent must be an existing directory")
    if os.path.lexists(path):
        raise ValueError("Output already exists; choose a new filename")
    raw = canonical(report) + b"\n"
    if len(raw) > MAX_JSON:
        raise ValueError("Report exceeds 128 MiB; use a shorter sequence")
    fd, temporary = tempfile.mkstemp(prefix=".pipeline-report-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # Atomic no-replace, including a racing writer.
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return path


def _event(row):
    selected = row.get("select_detection_index")
    release = row.get("release", False)
    if selected is not None:
        _integer(selected, "select_detection_index", maximum=299)
    if type(release) is not bool or (release and selected is not None):
        raise ValueError("release must be boolean and cannot accompany a selection")
    return {"select_detection_index": selected, "release": release}


def _pixel_hash(image):
    prefix = canonical({"mode": "RGB", "size": list(image.size)})
    return hashlib.sha256(prefix + b"\n" + image.tobytes()).hexdigest()


def _image(raw):
    from PIL import Image, ImageOps

    with Image.open(io.BytesIO(raw)) as image:
        if image.width * image.height > MAX_PIXELS:
            raise ValueError("Decoded image exceeds 64 million pixels")
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError("Use the video command for animated or multi-page inputs")
        return ImageOps.exif_transpose(image).convert("RGB")


def frame_input(path, bundle_directory):
    """Validate the complete clock and explicit event plan before model execution."""
    path = _regular(_outside(path, bundle_directory))
    value, sha = _json(path)
    if (
        not isinstance(value, dict)
        or set(value) != {"schema", "sequence_id", "clock_kind", "frames"}
        or value["schema"] != "iris-pipeline-input-v1"
    ):
        raise ValueError("Expected an iris-pipeline-input-v1 manifest")
    _text(value["sequence_id"], "sequence_id")
    clock = value["clock_kind"]
    if clock not in {"provided", "nominal_fps", "unknown"}:
        raise ValueError("Invalid clock_kind")
    rows = value["frames"]
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_FRAMES:
        raise ValueError("Provide between 1 and 10000 frames")
    seen, last_index, last_time = set(), -1, -1.0
    parsed = []
    for row in rows:
        required = {
            "frame_id",
            "frame_index",
            "timestamp_seconds",
            "path",
            "file_sha256",
            "input_size",
        }
        if (
            not isinstance(row, dict)
            or not required <= set(row)
            or set(row) - required - {"select_detection_index", "release"}
        ):
            raise ValueError("Invalid frame input fields")
        frame_id = _text(row["frame_id"], "frame_id")
        index = _integer(row["frame_index"], "frame_index", maximum=2**53 - 1)
        if frame_id in seen or index <= last_index:
            raise ValueError("Frame IDs must be unique and source indices strictly increasing")
        seen.add(frame_id)
        last_index = index
        timestamp = row["timestamp_seconds"]
        if clock == "unknown":
            if timestamp is not None:
                raise ValueError("Unknown clock requires null timestamps")
        else:
            if (
                type(timestamp) not in {float, int}
                or not math.isfinite(timestamp)
                or timestamp < 0
                or timestamp <= last_time
            ):
                raise ValueError("Known clock requires finite, nonnegative, increasing timestamps")
            last_time = timestamp
        relative = row["path"]
        if not isinstance(relative, str) or "\\" in relative:
            raise ValueError("Image path must be a relative POSIX path")
        parts = PurePosixPath(relative)
        if parts.is_absolute() or not parts.parts or ".." in parts.parts:
            raise ValueError("Image paths must stay inside the input manifest directory")
        image_path = _regular(_outside(path.parent / relative, bundle_directory))
        sha_value = row["file_sha256"]
        if (
            not isinstance(sha_value, str)
            or len(sha_value) != 64
            or any(c not in "0123456789abcdef" for c in sha_value)
        ):
            raise ValueError("Invalid image SHA-256")
        size = row["input_size"]
        if not isinstance(size, dict) or set(size) != {"width", "height"}:
            raise ValueError("input_size requires width and height")
        for dimension in size.values():
            _integer(dimension, "image dimension", minimum=1, maximum=MAX_PIXELS)
        if size["width"] * size["height"] > MAX_PIXELS:
            raise ValueError("Decoded image exceeds 64 million pixels")
        parsed.append((row, image_path, _event(row)))
    return value, sha, parsed


def semantic_frame(frame, image_input):
    result = deepcopy(frame)
    result["detector"].pop("timing", None)
    result["tracking"].pop("timing", None)
    return {"input": deepcopy(image_input), "output": result}


def _semantic(report):
    return {
        "clock_kind": report["clock_kind"],
        "pipeline_contract": report["pipeline_contract"],
        "frames": [
            semantic_frame(frame, image_input)
            for frame, image_input in zip(report["frames"], report["input_frames"], strict=True)
        ],
    }


def _report(pipeline, source, clock):
    return {
        "pipeline_contract": {
            "tracker_profile": pipeline.manifest["tracker"]["profile"],
            "selection_policy": (
                pipeline.manifest["selection"]["policy"] if pipeline.manifest["selection"] else None
            ),
        },
        "schema": RUN_SCHEMA,
        "complete": True,
        "bundle_manifest_sha256": digest(pipeline.manifest),
        "source": source,
        "clock_kind": clock,
        "runtime": deepcopy(pipeline.metadata),
        "input_frames": [],
        "frames": [],
        "scope": SCOPE,
    }


def _append(report, pipeline, image, *, frame_id, frame_index, timestamp_seconds, event):
    frame = pipeline.update(
        image,
        frame_id=frame_id,
        frame_index=frame_index,
        timestamp_seconds=timestamp_seconds,
        **event,
    )
    identity = {
        "frame_id": frame_id,
        "frame_index": frame_index,
        "timestamp_seconds": timestamp_seconds,
        "pixel_sha256": _pixel_hash(image),
        "input_size": {"width": image.width, "height": image.height},
        **event,
    }
    # Charge actual serialized rows before accumulating the next frame.
    size = len(canonical(frame)) + len(canonical(identity))
    report["_bytes"] = report.get("_bytes", 0) + size
    if report["_bytes"] > MAX_JSON - 1024 * 1024:
        raise ValueError("Report limit reached; use a shorter sequence")
    report["input_frames"].append(identity)
    report["frames"].append(frame)


def _finish(report):
    if not report["frames"]:
        raise ValueError("Input contains no decodable frames")
    report.pop("_bytes", None)
    report["semantic_sha256"] = digest(_semantic(report))
    return report


def run_frames(pipeline, path, *, bundle_directory):
    value, sha, rows = frame_input(path, bundle_directory)
    pipeline.reset(value["sequence_id"], clock_kind=value["clock_kind"])
    report = _report(
        pipeline,
        {"kind": "frames", "input_sha256": sha, "coverage": "all_declared_frames"},
        value["clock_kind"],
    )
    for row, image_path, event in rows:
        if image_path.stat().st_size > MAX_IMAGE:
            raise ValueError("Image exceeds 32 MiB")
        raw = image_path.read_bytes()
        if len(raw) > MAX_IMAGE or hashlib.sha256(raw).hexdigest() != row["file_sha256"]:
            raise ValueError("Input image size or SHA-256 mismatch")
        image = _image(raw)
        if {"width": image.width, "height": image.height} != row["input_size"]:
            raise ValueError("Oriented image dimensions differ from the manifest")
        _append(
            report,
            pipeline,
            image,
            frame_id=row["frame_id"],
            frame_index=row["frame_index"],
            timestamp_seconds=row["timestamp_seconds"],
            event=event,
        )
    pipeline.verify_runtime()
    _same(report["runtime"], pipeline.metadata, "runtime at completed image run")
    return _finish(report)


def _video_events(path, maximum, bundle_directory):
    if path is None:
        return {}, None
    data, sha = _json(_outside(path, bundle_directory))
    if (
        not isinstance(data, dict)
        or set(data) != {"schema", "events"}
        or data["schema"] != "iris-pipeline-events-v1"
        or not isinstance(data["events"], list)
        or len(data["events"]) > maximum
    ):
        raise ValueError("Expected an iris-pipeline-events-v1 event list")
    result = {}
    for row in data["events"]:
        if (
            not isinstance(row, dict)
            or "frame_index" not in row
            or set(row) - {"frame_index", "select_detection_index", "release"}
        ):
            raise ValueError("Invalid video event fields")
        index = _integer(row["frame_index"], "event frame_index", maximum=maximum - 1)
        if index in result:
            raise ValueError("Duplicate video event index")
        result[index] = _event(row)
    return result, sha


def run_video(
    pipeline, path, *, bundle_directory, max_frames=MAX_FRAMES, clock="nominal_fps", events=None
):
    import cv2
    from PIL import Image

    _integer(max_frames, "max_frames", minimum=1)
    if clock not in {"nominal_fps", "unknown"}:
        raise ValueError("Video clock must be nominal_fps or unknown")
    path = _regular(_outside(path, bundle_directory))
    if path.suffix.lower() not in {".avi", ".mp4", ".mov", ".mkv", ".webm"}:
        raise ValueError("Use a local AVI, MP4, MOV, MKV or WebM file")
    # Reject text playlists renamed as video before the decoder can resolve URLs.
    with path.open("rb") as stream:
        header = stream.read(32)
    binary_container = (
        (header.startswith(b"RIFF") and header[8:12] == b"AVI ")
        or header.startswith(b"\x1aE\xdf\xa3")
        or header[4:8] in {b"ftyp", b"moov", b"mdat", b"wide", b"free"}
    )
    if not binary_container:
        raise ValueError("File does not have a supported binary video container header")
    event_map, event_sha = _video_events(events, max_frames, bundle_directory)
    original_sha = _sha_file(path)
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError("Local video could not be opened")
        capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
        rotation = capture.get(cv2.CAP_PROP_ORIENTATION_META)
        if rotation not in {0, 90, 180, 270}:
            raise ValueError("Unsupported video orientation metadata")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if clock == "nominal_fps" and (not math.isfinite(fps) or fps <= 0):
            raise ValueError("Video has no usable nominal FPS; explicitly choose --clock unknown")
        fps = fps if math.isfinite(fps) and fps > 0 else None
        sequence_id = f"video-{original_sha[:24]}"
        pipeline.reset(sequence_id, clock_kind=clock)
        source = {
            "kind": "video",
            "input_sha256": original_sha,
            "events_sha256": event_sha,
            "nominal_fps": fps,
            "orientation_degrees": rotation,
            "coverage": "bounded_prefix",
            "requested_max_frames": max_frames,
            "decoder": {
                "library": "OpenCV",
                "version": cv2.__version__,
                "backend": capture.getBackendName(),
            },
        }
        report = _report(pipeline, source, clock)
        consumed = set()
        for index in range(max_frames):
            available, array = capture.read()
            if not available:
                source["coverage"] = "decoder_end_of_stream"
                break
            if array.shape[0] * array.shape[1] > MAX_PIXELS:
                raise ValueError("Video frame exceeds 64 million pixels")
            image = Image.fromarray(cv2.cvtColor(array, cv2.COLOR_BGR2RGB))
            if rotation:
                image = image.rotate(-rotation, expand=True)
            event = event_map.get(index, {"select_detection_index": None, "release": False})
            if index in event_map:
                consumed.add(index)
            _append(
                report,
                pipeline,
                image,
                frame_id=f"frame-{index:08d}",
                frame_index=index,
                timestamp_seconds=index / fps if clock == "nominal_fps" else None,
                event=event,
            )
        if set(event_map) != consumed:
            raise ValueError("An event addresses a frame the decoder did not produce")
        if _sha_file(path) != original_sha:
            raise ValueError("Video changed while processing")
        pipeline.verify_runtime()
        _same(report["runtime"], pipeline.metadata, "runtime at completed video run")
        return _finish(report)
    finally:
        capture.release()


def _object(value, keys, name):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"Invalid {name} fields")
    return value


def _sha(value, name):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"Invalid {name} SHA-256")


def _timing(value, keys):
    _object(value, keys, "measured timing")
    if any(
        type(number) not in (int, float) or not math.isfinite(number) or number < 0
        for number in value.values()
    ):
        raise ValueError("Measured timings must be finite nonnegative numbers")


def _same(first, second, name):
    if canonical(first) != canonical(second):
        raise ValueError(f"Inconsistent {name}")


def _validate_report(report):
    """Check historical evidence without images, native trackers or ML imports.

    Selection transitions and observation provenance are reproducible JSON facts.
    Detector results, native association and runtime hardware remain declarations;
    a coherent report does not authenticate its author or prove model quality.
    """
    from .pipeline_selection import SelectionState
    from .tracking_contracts import validate_profile, validate_tracking_frame, validate_update_input
    from .tracking_selection_contracts import validate_policy

    _object(
        report,
        {
            "schema",
            "complete",
            "bundle_manifest_sha256",
            "source",
            "clock_kind",
            "runtime",
            "pipeline_contract",
            "input_frames",
            "frames",
            "scope",
            "semantic_sha256",
        },
        "pipeline run report",
    )
    if report["schema"] != RUN_SCHEMA or report["complete"] is not True or report["scope"] != SCOPE:
        raise ValueError("Comparison requires complete, unqualified pipeline run reports")
    for key in ("bundle_manifest_sha256", "semantic_sha256"):
        _sha(report[key], key)
    contract = _object(
        report["pipeline_contract"], {"tracker_profile", "selection_policy"}, "pipeline contract"
    )
    profile = validate_profile(contract["tracker_profile"])
    _same(profile, contract["tracker_profile"], "canonical tracker profile")
    policy = contract["selection_policy"]
    if policy is not None:
        _same(validate_policy(policy), policy, "canonical selected-object policy")
    selection = SelectionState(policy) if policy is not None else None
    clock = report["clock_kind"]
    if not isinstance(clock, str) or clock not in {"provided", "nominal_fps", "unknown"}:
        raise ValueError("Invalid run clock")
    runtime = _object(
        report["runtime"],
        {
            "schema",
            "bundle_manifest_sha256",
            "detector",
            "tracker",
            "selection",
            "code_sha256",
            "limits",
            "clock_kind",
            "independent_quality",
        },
        "runtime declaration",
    )
    if (
        runtime["schema"] != "iris-pipeline-runtime-v1"
        or runtime["bundle_manifest_sha256"] != report["bundle_manifest_sha256"]
        or runtime["clock_kind"] != clock
        or runtime["independent_quality"] != "not_qualified"
        or not isinstance(runtime["detector"], dict)
        or not isinstance(runtime["tracker"], dict)
        or runtime["tracker"].get("algorithm") != profile["algorithm"]
    ):
        raise ValueError("Runtime declaration differs from its report contract")
    if not isinstance(runtime["code_sha256"], dict) or not runtime["code_sha256"]:
        raise ValueError("Runtime code provenance is missing")
    for path, sha in runtime["code_sha256"].items():
        _text(path, "Runtime module path")
        _sha(sha, "Runtime module")
    _object(runtime["limits"], {"max_updates_per_reset", "max_image_pixels"}, "runtime limits")
    _integer(runtime["limits"]["max_updates_per_reset"], "Runtime update limit", 1, MAX_FRAMES)
    _integer(runtime["limits"]["max_image_pixels"], "Runtime pixel limit", 1, 64 * 1024**2)
    if policy is None:
        if runtime["selection"] is not None:
            raise ValueError("Runtime declares a policy absent from the pipeline contract")
    else:
        declared = _object(
            runtime["selection"],
            {"algorithm", "policy_sha256", "code_sha256", "identity_claim"},
            "runtime selection",
        )
        if (
            declared["algorithm"] != "guarded_geometry"
            or declared["policy_sha256"] != digest(policy)
            or declared["identity_claim"] != "association_evidence_not_physical_identity"
            or declared["code_sha256"]
            != runtime["code_sha256"].get("iris_bundle/pipeline_selection.py")
        ):
            raise ValueError("Runtime selection differs from its declared policy")
    rows, inputs = report["frames"], report["input_frames"]
    if (
        not isinstance(rows, list)
        or not isinstance(inputs, list)
        or not 1 <= len(rows) <= min(MAX_FRAMES, runtime["limits"]["max_updates_per_reset"])
        or len(rows) != len(inputs)
    ):
        raise ValueError("Invalid run report frame inventory")
    source = report["source"]
    if not isinstance(source, dict) or source.get("kind") not in {"frames", "video"}:
        raise ValueError("Invalid run source")
    if source["kind"] == "frames":
        _object(source, {"kind", "input_sha256", "coverage"}, "image source")
        if source["coverage"] != "all_declared_frames":
            raise ValueError("Image runs must cover every declared frame")
    else:
        _object(
            source,
            {
                "kind",
                "input_sha256",
                "events_sha256",
                "nominal_fps",
                "orientation_degrees",
                "coverage",
                "requested_max_frames",
                "decoder",
            },
            "video source",
        )
        if clock not in {"unknown", "nominal_fps"}:
            raise ValueError("Video reports cannot claim provided source timestamps")
        maximum = _integer(source["requested_max_frames"], "Requested video limit", 1)
        if len(rows) > maximum or source["coverage"] not in {
            "bounded_prefix",
            "decoder_end_of_stream",
        }:
            raise ValueError("Video coverage exceeds its declared limit")
        if source["coverage"] == "bounded_prefix" and len(rows) != maximum:
            raise ValueError("A bounded video prefix must reach the requested limit")
        if source["events_sha256"] is not None:
            _sha(source["events_sha256"], "Video events")
        fps = source["nominal_fps"]
        if fps is not None and (
            type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0
        ):
            raise ValueError("Video nominal FPS must be positive or explicitly unknown")
        if clock == "nominal_fps" and fps is None:
            raise ValueError("A nominal video clock requires known FPS")
        rotation = source["orientation_degrees"]
        if type(rotation) not in (int, float) or rotation not in {0, 90, 180, 270}:
            raise ValueError("Invalid source video orientation")
        decoder = _object(source["decoder"], {"library", "version", "backend"}, "video decoder")
        if decoder["library"] != "OpenCV":
            raise ValueError("Unknown video decoder")
        _text(decoder["version"], "Decoder version")
        _text(decoder["backend"], "Decoder backend")
    _sha(source["input_sha256"], "Source input")
    seen, last_index, last_time, dimensions, sequence_id = set(), -1, -1.0, None, None
    observed_history, label_names = {}, {}
    for position, (frame, image_input) in enumerate(zip(rows, inputs, strict=True), 1):
        _object(
            image_input,
            {
                "frame_id",
                "frame_index",
                "timestamp_seconds",
                "pixel_sha256",
                "input_size",
                "select_detection_index",
                "release",
            },
            "input frame",
        )
        identifier = _text(image_input["frame_id"], "Input frame ID")
        index = _integer(image_input["frame_index"], "Source frame index", maximum=2**53 - 1)
        if identifier in seen or index <= last_index:
            raise ValueError("Frame IDs must be unique and source indices strictly increasing")
        seen.add(identifier)
        last_index = index
        timestamp = image_input["timestamp_seconds"]
        if clock == "unknown":
            if timestamp is not None:
                raise ValueError("Unknown clocks require null timestamps")
        else:
            if (
                type(timestamp) not in (int, float)
                or not math.isfinite(timestamp)
                or timestamp < 0
                or timestamp <= last_time
            ):
                raise ValueError("Known timestamps must be finite, nonnegative and increasing")
            last_time = timestamp
        _sha(image_input["pixel_sha256"], "Oriented input pixels")
        size = _object(image_input["input_size"], {"width", "height"}, "Input image dimensions")
        for dimension in size.values():
            _integer(dimension, "Image dimension", 1, MAX_PIXELS)
        if size["width"] * size["height"] > min(MAX_PIXELS, runtime["limits"]["max_image_pixels"]):
            raise ValueError("Reported image exceeds its declared pixel limit")
        shape = [size["width"], size["height"]]
        if dimensions is not None and shape != dimensions:
            raise ValueError("Image dimensions cannot change within one pipeline sequence")
        dimensions = shape
        event = _event(image_input)
        if selection is None and (event["release"] or event["select_detection_index"] is not None):
            raise ValueError("An event requires a declared selected-object policy")
        _object(
            frame,
            {
                "schema",
                "sequence_id",
                "frame_id",
                "frame_index",
                "timestamp_seconds",
                "input_size",
                "update_index",
                "detector",
                "tracking",
                "selection",
            },
            "Pipeline frame",
        )
        if frame["schema"] != "iris-pipeline-frame-v1":
            raise ValueError("Unsupported pipeline frame schema")
        current_sequence = _text(frame["sequence_id"], "Pipeline sequence ID")
        if sequence_id is not None and current_sequence != sequence_id:
            raise ValueError("A run report cannot change sequence identity")
        sequence_id = current_sequence
        source_facts = {
            "frame_id": identifier,
            "frame_index": index,
            "timestamp_seconds": timestamp,
            "input_size": shape,
        }
        for key, value in {**source_facts, "update_index": position}.items():
            _same(frame[key], value, f"pipeline {key}")
        if source["kind"] == "video":
            if (
                index != position - 1
                or identifier != f"frame-{index:08d}"
                or sequence_id != f"video-{source['input_sha256'][:24]}"
            ):
                raise ValueError("Video frame provenance differs from its source")
            if clock == "nominal_fps":
                _same(timestamp, index / source["nominal_fps"], "nominal video timestamp")
        detected = _object(
            frame["detector"],
            {"input_size", "detections", "native_detection_count", "timing"},
            "Detector frame",
        )
        _same(detected["input_size"], shape, "detector dimensions")
        native_count = _integer(
            detected["native_detection_count"], "Native detection count", maximum=100
        )
        _timing(detected["timing"], {"preprocess_ms", "inference_ms", "postprocess_ms", "total_ms"})
        detections = detected["detections"]
        if not isinstance(detections, list) or len(detections) > native_count:
            raise ValueError("Invalid detector count or detection inventory")
        classes = set(profile["class_ids"])
        for detection in detections:
            if not isinstance(detection, dict):
                raise ValueError("Detector outputs must be objects")
            label_id = _integer(detection.get("label_id"), "Detector output class", 1, 2**53 - 1)
            classes.add(label_id)
        full = validate_update_input(
            {**source_facts, "detections": detections}, {**profile, "class_ids": sorted(classes)}
        )
        for row in full["detections"]:
            if row["detection_index"] >= native_count:
                raise ValueError("Stored detection index exceeds the native output count")
            if row["label_id"] in label_names and label_names[row["label_id"]] != row["label"]:
                raise ValueError("Detector output names cannot change within one sequence")
            label_names[row["label_id"]] = row["label"]
        tracking_source = {
            **source_facts,
            "detections": [row for row in detections if row["label_id"] in profile["class_ids"]],
        }
        tracked = validate_tracking_frame(frame["tracking"], tracking_source, profile)
        for key, value in {
            **source_facts,
            "sequence_id": sequence_id,
            "update_index": position,
        }.items():
            _same(tracked[key], value, f"tracker {key}")
        originals = {row["detection_index"]: row for row in tracking_source["detections"]}
        for disposition in (*tracked["observations"], *tracked["unassigned"]):
            original = originals[disposition["detection_index"]]
            _same({key: disposition[key] for key in original}, original, "measured observation")
        for prediction in tracked["predictions"]:
            previous = observed_history.get(prediction["track_id"])
            if previous is None:
                raise ValueError("A prediction requires an actual earlier measured observation")
            expected = {
                "last_observed_frame_id": previous["frame_id"],
                "last_observed_frame_index": previous["frame_index"],
                "last_observed_timestamp_seconds": previous["timestamp_seconds"],
                "last_observed_update_index": previous["update_index"],
                "age_updates": position - previous["update_index"],
                "age_seconds": None
                if timestamp is None
                else timestamp - previous["timestamp_seconds"],
                **{key: previous["observation"][key] for key in ("label_id", "label", "confirmed")},
            }
            for key, value in expected.items():
                _same(prediction[key], value, f"prediction {key}")
        for observation in tracked["observations"]:
            previous = observed_history.get(observation["track_id"])
            if previous is not None:
                for key in ("label_id", "label"):
                    _same(observation[key], previous["observation"][key], "track class identity")
            observed_history[observation["track_id"]] = {
                **source_facts,
                "update_index": position,
                "observation": observation,
            }
        expected_selection = None if selection is None else selection.update(tracked, **event)
        _same(frame["selection"], expected_selection, "selected-object transition")
    if digest(_semantic(report)) != report["semantic_sha256"]:
        raise ValueError("Run semantic SHA-256 mismatch")
    return report


def compare_reports(reference, actual):
    """Exact comparison of inputs and outputs; hashes do not attest an author."""
    loaded = []
    for path in (reference, actual):
        report, file_sha = _json(path)
        try:
            _validate_report(report)
        except (KeyError, TypeError, OverflowError, RecursionError) as exc:
            raise ValueError("Malformed pipeline run evidence") from exc
        semantic = _semantic(report)
        loaded.append((report, file_sha, semantic))
    left, right = loaded
    mismatch = []
    for index in range(max(len(left[2]["frames"]), len(right[2]["frames"]))):
        a = left[2]["frames"][index] if index < len(left[2]["frames"]) else None
        b = right[2]["frames"][index] if index < len(right[2]["frames"]) else None
        if canonical(a) != canonical(b):
            mismatch.append(index)
    matched = canonical(left[2]) == canonical(right[2])
    return {
        "schema": "iris-pipeline-parity-v1",
        "status": "exact_match" if matched else "mismatch",
        "reference_sha256": left[1],
        "actual_sha256": right[1],
        "reference_semantic_sha256": left[0]["semantic_sha256"],
        "actual_semantic_sha256": right[0]["semantic_sha256"],
        "reference_bundle_manifest_sha256": left[0]["bundle_manifest_sha256"],
        "actual_bundle_manifest_sha256": right[0]["bundle_manifest_sha256"],
        "reference_frames": len(left[0]["frames"]),
        "actual_frames": len(right[0]["frames"]),
        "clock_match": left[0]["clock_kind"] == right[0]["clock_kind"],
        "contract_match": canonical(left[0]["pipeline_contract"])
        == canonical(right[0]["pipeline_contract"]),
        "mismatched_update_indices": mismatch,
        "tolerance": 0,
        "exclusions": ["per-frame timing", "runtime hardware", "container and image encoding"],
        "scope": SCOPE,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run a trusted IRIS pipeline bundle locally")
    parser.add_argument("--bundle", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--device", help="cpu or cuda[:index], within the bundle target family")
    actions = parser.add_subparsers(dest="command", required=True)
    actions.add_parser("check-runtime", help="Verify files, packages and model loading")
    frames = actions.add_parser("frames", help="Process a timestamped image manifest")
    frames.add_argument("input", type=Path)
    frames.add_argument("--output", type=Path, required=True)
    video = actions.add_parser("video", help="Process a bounded local video")
    video.add_argument("input", type=Path)
    video.add_argument("--output", type=Path, required=True)
    video.add_argument("--max-frames", type=int, default=MAX_FRAMES)
    video.add_argument("--clock", choices=("nominal_fps", "unknown"), default="nominal_fps")
    video.add_argument("--events", type=Path)
    compare = actions.add_parser("compare", help="Compare two explicit run reports without ML")
    compare.add_argument("reference", type=Path)
    compare.add_argument("actual", type=Path)
    compare.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "compare":
            report = compare_reports(args.reference, args.actual)
        else:
            if args.command != "check-runtime":
                _outside(args.output, args.bundle)
                if os.path.lexists(args.output):
                    raise ValueError("Output already exists; choose a new filename")
            from .pipeline_runtime import Pipeline

            pipeline = Pipeline(args.bundle, device=args.device)
            if args.command == "check-runtime":
                print(
                    json.dumps(
                        {"ready": True, "runtime": pipeline.metadata, "scope": SCOPE}, indent=2
                    )
                )
                return 0
            if args.command == "frames":
                report = run_frames(pipeline, args.input, bundle_directory=args.bundle)
            else:
                report = run_video(
                    pipeline,
                    args.input,
                    bundle_directory=args.bundle,
                    max_frames=args.max_frames,
                    clock=args.clock,
                    events=args.events,
                )
        _outside(args.output, args.bundle)
        write_report(args.output, report)
        print(
            json.dumps(
                {
                    "report": str(args.output.absolute()),
                    "status": report.get("status", "complete"),
                    "scope": SCOPE,
                }
            )
        )
        return 2 if report.get("status") == "mismatch" else 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"Pipeline failed: {exc}\n")
