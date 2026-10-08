"""Frozen temporal inputs and reference identities, independent of tracker runtimes.

Reference identities are local to a sequence and never stand for tracker output IDs.
An omitted or unreviewed frame is unknown evidence, not an empty negative frame.
These contracts describe source observations; predicted tracker positions belong to
a separate future output contract and cannot be saved as reference fields here.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from copy import deepcopy

from iris.dataset_manifest import taxonomy_mappings

SEQUENCE_SCHEMA = "iris-temporal-sequence-v1"
REFERENCE_SCHEMA = "iris-temporal-reference-v1"
REFERENCE_SCHEMA_V2 = "iris-temporal-reference-v2"
MAX_SEQUENCE_FRAMES = 10_000
MAX_CLIP_FRAMES = 1_000_000
MAX_IDENTITIES = 10_000
MAX_OBJECTS_PER_FRAME = 500
MAX_REFERENCE_OBJECTS = 100_000
_HASH = re.compile(r"[0-9a-f]{64}\Z")


def _object(value, fields: set[str], description: str) -> dict:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{description} must contain exactly its supported fields")
    return value


def _text(
    value, description: str, *, maximum: int = 128, empty: bool = False, multiline: bool = False
) -> str:
    if (
        not isinstance(value, str)
        or len(value) > maximum
        or not empty
        and not value.strip()
        or any(
            ord(character) < 32 and not (multiline and character in "\n\r\t") for character in value
        )
    ):
        raise ValueError(f"{description} must be bounded text{' (nonempty)' if not empty else ''}")
    try:
        value.encode("utf-8")
    except UnicodeError as exc:
        raise ValueError(f"{description} must be valid UTF-8 text") from exc
    return value


def _integer(value, description: str, *, minimum: int = 0, maximum: int = 2**53 - 1) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{description} must be an integer between {minimum} and {maximum}")
    return value


def _number(value, description: str, *, positive: bool = False) -> float:
    try:
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError
        number = float(value)
        if number < 0 or positive and number == 0:
            raise ValueError
        return 0.0 if number == 0 else number
    except (ValueError, OverflowError) as exc:
        raise ValueError(
            f"{description} must be a finite {'positive' if positive else 'nonnegative'} number"
        ) from exc


def _digest(value, description: str) -> str:
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise ValueError(f"{description} must be a lowercase SHA-256 digest")
    return value


def _choice(value, choices: set[str], description: str) -> str:
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"{description} must be one of {', '.join(sorted(choices))}")
    return value


def _items(value, description: str, *, maximum: int, minimum: int = 0) -> list:
    if not isinstance(value, list) or not minimum <= len(value) <= maximum:
        raise ValueError(f"{description} must contain between {minimum} and {maximum} items")
    return value


def _clock(value) -> dict:
    _object(value, {"basis", "fps", "provenance"}, "Sequence clock")
    basis = _choice(value["basis"], {"nominal_fps", "provided", "unknown"}, "Clock basis")
    provenance = _text(value["provenance"], "Clock provenance", maximum=2000)
    fps = value["fps"]
    if basis == "nominal_fps":
        fps = _number(fps, "Nominal FPS", positive=True)
    elif fps is not None:
        raise ValueError("Provided or unknown clocks require null FPS")
    return {"basis": basis, "fps": fps, "provenance": provenance}


def validate_sequence_manifest(manifest: dict) -> dict:
    """Check complete bounded frame/gap coverage without opening workspace files.

    Nominal times are source frame index divided by nominal FPS, not decoder PTS.
    Provided times are caller-declared values with provenance; validation does not
    establish that they were measured by a camera. Clip bounds are inclusive.
    """
    _object(
        manifest,
        {
            "schema",
            "id",
            "project_id",
            "name",
            "parent_id",
            "asset",
            "take_group",
            "taxonomy",
            "clock",
            "clip",
            "frames",
            "gaps",
        },
        "Temporal sequence",
    )
    if manifest["schema"] != SEQUENCE_SCHEMA:
        raise ValueError("Unsupported temporal sequence schema")
    identifier = _text(manifest["id"], "Sequence ID")
    _text(manifest["project_id"], "Project ID")
    _text(manifest["name"], "Sequence name", maximum=160)
    if manifest["parent_id"] is not None:
        _text(manifest["parent_id"], "Parent sequence ID")
        if manifest["parent_id"] == identifier:
            raise ValueError("A sequence cannot be its own parent")
    asset = _object(
        manifest["asset"], {"id", "sha256", "session_id", "scene_group"}, "Sequence asset"
    )
    _text(asset["id"], "Source asset ID")
    _digest(asset["sha256"], "Source asset hash")
    _text(asset["session_id"], "Source session ID")
    _text(asset["scene_group"], "Source scene group", maximum=160)
    _text(manifest["take_group"], "Take group", maximum=160)
    taxonomy_mappings(manifest["taxonomy"])
    clock = _clock(manifest["clock"])
    clip = _object(manifest["clip"], {"start_frame", "end_frame"}, "Sequence clip")
    start = _integer(clip["start_frame"], "Clip start frame")
    end = _integer(clip["end_frame"], "Clip end frame", minimum=start)
    if end - start + 1 > MAX_CLIP_FRAMES:
        raise ValueError(f"A temporal clip supports at most {MAX_CLIP_FRAMES} source frames")
    frames = _items(manifest["frames"], "Available frames", maximum=MAX_SEQUENCE_FRAMES, minimum=1)
    gaps = _items(manifest["gaps"], "Sequence gaps", maximum=MAX_SEQUENCE_FRAMES + 1)
    frame_ids, previous_index, previous_time, intervals, normalized_frames = (
        set(),
        None,
        None,
        [],
        [],
    )
    for frame in frames:
        _object(
            frame,
            {
                "frame_id",
                "frame_index",
                "timestamp_seconds",
                "width",
                "height",
                "sha256",
                "file_sha256",
            },
            "Sequence frame",
        )
        frame_id = _text(frame["frame_id"], "Frame ID")
        if frame_id in frame_ids:
            raise ValueError("Sequence frame IDs must be unique")
        frame_ids.add(frame_id)
        index = _integer(frame["frame_index"], "Frame index", minimum=start, maximum=end)
        if previous_index is not None and index <= previous_index:
            raise ValueError("Sequence frame indices must be strictly increasing")
        previous_index = index
        _integer(frame["width"], "Frame width", minimum=1, maximum=1_000_000)
        _integer(frame["height"], "Frame height", minimum=1, maximum=1_000_000)
        _digest(frame["sha256"], "Frame pixel hash")
        _digest(frame["file_sha256"], "Frame file hash")
        timestamp = frame["timestamp_seconds"]
        if clock["basis"] == "unknown":
            if timestamp is not None:
                raise ValueError("An unknown clock requires null timestamps")
        else:
            timestamp = _number(timestamp, "Frame timestamp")
            if clock["basis"] == "nominal_fps":
                expected = index / clock["fps"]
                if not math.isfinite(expected) or timestamp != expected:
                    raise ValueError("Nominal timestamps must equal source frame index / FPS")
            if previous_time is not None and timestamp <= previous_time:
                raise ValueError("Sequence timestamps must be strictly increasing")
            previous_time = timestamp
        normalized_frames.append({**frame, "timestamp_seconds": timestamp})
        intervals.append((index, index))
    previous_gap_end = None
    for gap in gaps:
        _object(gap, {"start_frame", "end_frame", "reason"}, "Sequence gap")
        first = _integer(gap["start_frame"], "Gap start frame", minimum=start, maximum=end)
        last = _integer(gap["end_frame"], "Gap end frame", minimum=first, maximum=end)
        _choice(gap["reason"], {"skipped", "unavailable", "unknown"}, "Gap reason")
        if previous_gap_end is not None and first <= previous_gap_end:
            raise ValueError("Sequence gaps must be ordered and nonoverlapping")
        previous_gap_end = last
        intervals.append((first, last))
    cursor = start
    for first, last in sorted(intervals):
        if first != cursor:
            raise ValueError("Frames and gaps must cover the clip exactly without overlap or holes")
        cursor = last + 1
    if cursor != end + 1:
        raise ValueError("Frames and gaps must cover the clip exactly without overlap or holes")
    result = deepcopy(manifest)
    result["clock"] = clock
    result["frames"] = normalized_frames
    return result


def sequence_hash(manifest: dict) -> str:
    """Hash the canonical validated manifest, not a mutable workspace row."""
    raw = json.dumps(
        validate_sequence_manifest(manifest),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _box(value, frame: dict) -> list[float]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("Reference boxes require four xyxy pixel coordinates")
    coordinates = [_number(item, "Reference box coordinate") for item in value]
    x1, y1, x2, y2 = coordinates
    if not (x1 < x2 <= frame["width"] and y1 < y2 <= frame["height"]):
        raise ValueError("Reference boxes must have positive area inside their source frame")
    return coordinates


def validate_reference_provenance(provenance, labels: set[str]) -> dict:
    """Validate a seed snapshot, independently of subsequently edited identities."""
    _object(provenance, {"author", "origin"}, "Reference provenance")
    _text(provenance["author"], "Reference author", maximum=200, empty=True)
    origin = provenance["origin"]
    if origin is None:
        return deepcopy(provenance)
    _object(
        origin,
        {
            "comparison_id",
            "lane_index",
            "cache_id",
            "cache_fingerprint",
            "result_sha256",
            "profile_sha256",
            "semantic_sha256",
            "class_mapping",
            "track_mapping",
        },
        "Reference seed origin",
    )
    for key in ("comparison_id", "cache_id"):
        _text(origin[key], f"Seed {key}")
    _integer(origin["lane_index"], "Seed lane index", maximum=1)
    for key in ("cache_fingerprint", "result_sha256", "profile_sha256", "semantic_sha256"):
        _digest(origin[key], f"Seed {key}")
    mapping = origin["class_mapping"]
    if not isinstance(mapping, dict) or not 1 <= len(mapping) <= 100:
        raise ValueError("Seed class mapping requires 1–100 explicit native classes")
    tracks = origin["track_mapping"]
    if not isinstance(tracks, dict) or len(tracks) > MAX_IDENTITIES:
        raise ValueError("Seed track mapping exceeds the reference identity limit")
    for name, values in (("class", mapping), ("track", tracks)):
        for native_id in values:
            if (
                not isinstance(native_id, str)
                or not re.fullmatch(r"[1-9][0-9]{0,15}", native_id)
                or int(native_id) > 2**53 - 1
            ):
                raise ValueError(f"Seed {name} IDs must be canonical positive native integers")
    for label in mapping.values():
        if label is not None:
            _choice(label, labels, "Seed taxonomy label")
    for identifier in tracks.values():
        _text(identifier, "Seed reference identity ID")
        if not identifier.startswith("ref_"):
            raise ValueError("Seed identities must use separate reference IDs")
    if len(set(tracks.values())) != len(tracks):
        raise ValueError("Seed tracks require distinct reference identity IDs")
    return deepcopy(provenance)


def validate_reference(reference: dict, sequence_manifest: dict) -> dict:
    """Validate a temporal reference while preserving declared review provenance.

    Visible objects need an observed box; fully occluded objects may have none.
    Out-of-view and unknown observations cannot carry inferred coordinates. Sparse
    or assistant-reviewed annotations never become dense human ground truth.
    """
    sequence = validate_sequence_manifest(sequence_manifest)
    schema = reference.get("schema") if isinstance(reference, dict) else None
    if schema not in (REFERENCE_SCHEMA, REFERENCE_SCHEMA_V2):
        raise ValueError("Unsupported temporal reference schema")
    _object(
        reference,
        {
            "schema",
            "sequence_id",
            "sequence_sha256",
            "taxonomy_id",
            "identities",
            "frames",
            "notes",
        }
        | ({"provenance"} if schema == REFERENCE_SCHEMA_V2 else set()),
        "Temporal reference",
    )
    if reference["sequence_id"] != sequence["id"]:
        raise ValueError("Reference must belong to the exact sequence")
    _digest(reference["sequence_sha256"], "Reference sequence hash")
    if reference["sequence_sha256"] != sequence_hash(sequence):
        raise ValueError("Reference sequence hash does not match its frozen manifest")
    if reference["taxonomy_id"] != sequence["taxonomy"]["id"]:
        raise ValueError("Reference taxonomy must match its frozen sequence")
    _text(reference["notes"], "Reference notes", maximum=4000, empty=True, multiline=True)
    identities = _items(reference["identities"], "Reference identities", maximum=MAX_IDENTITIES)
    labels = {item["id"] for item in sequence["taxonomy"]["classes"]}
    if schema == REFERENCE_SCHEMA_V2:
        validate_reference_provenance(reference["provenance"], labels)
    identity_labels = {}
    for identity in identities:
        _object(identity, {"id", "label"}, "Reference identity")
        identifier = _text(identity["id"], "Reference identity ID")
        if identifier in identity_labels:
            raise ValueError("Reference identity IDs must be unique")
        label = _choice(identity["label"], labels, "Reference identity label")
        identity_labels[identifier] = label
    source_frames = {item["frame_index"]: item for item in sequence["frames"]}
    frames = _items(reference["frames"], "Reference frames", maximum=len(source_frames))
    previous_index, object_count, normalized_frames = None, 0, []
    for frame in frames:
        _object(frame, {"frame_index", "coverage", "review", "objects"}, "Reference frame")
        index = _integer(frame["frame_index"], "Reference frame index")
        if index not in source_frames:
            raise ValueError("References may annotate only available sequence frames")
        if previous_index is not None and index <= previous_index:
            raise ValueError("Reference frame indices must be unique and strictly increasing")
        previous_index = index
        coverage = _choice(
            frame["coverage"], {"complete", "partial", "unreviewed"}, "Reference coverage"
        )
        review = _object(frame["review"], {"status", "reviewer"}, "Reference review")
        status = _choice(
            review["status"],
            {"unreviewed", "assistant_reviewed", "human_reviewed"},
            "Review status",
        )
        _text(review["reviewer"], "Reviewer", maximum=200, empty=status == "unreviewed")
        if (coverage == "unreviewed") != (status == "unreviewed"):
            raise ValueError("Unreviewed coverage and review status must agree")
        objects = _items(frame["objects"], "Reference objects", maximum=MAX_OBJECTS_PER_FRAME)
        object_count += len(objects)
        if object_count > MAX_REFERENCE_OBJECTS:
            raise ValueError(
                f"A temporal reference supports at most {MAX_REFERENCE_OBJECTS} objects"
            )
        seen_identities, normalized_objects = set(), []
        for obj in objects:
            _object(
                obj, {"identity_id", "label", "box", "visibility", "certainty"}, "Reference object"
            )
            identity_id = obj["identity_id"]
            label = _choice(obj["label"], labels, "Reference object label")
            if identity_id is not None:
                _text(identity_id, "Reference object identity ID")
                if identity_id not in identity_labels:
                    raise ValueError("Object identity must be declared in this reference")
                if identity_labels[identity_id] != label:
                    raise ValueError("Object labels must match their reference identity")
                if identity_id in seen_identities:
                    raise ValueError("An identity cannot occur more than once per reference frame")
                seen_identities.add(identity_id)
            visibility = _choice(
                obj["visibility"],
                {"visible", "occluded", "out_of_view", "unknown"},
                "Object visibility",
            )
            certainty = _choice(obj["certainty"], {"certain", "uncertain"}, "Object certainty")
            if certainty == "certain" and (identity_id is None or visibility == "unknown"):
                raise ValueError(
                    "Certain objects require a reference identity and known visibility"
                )
            if coverage == "complete" and certainty != "certain":
                raise ValueError("Complete coverage cannot contain uncertain objects")
            box = obj["box"]
            if visibility == "visible" and box is None:
                raise ValueError("Visible objects require an observed reference box")
            if visibility in {"out_of_view", "unknown"} and box is not None:
                raise ValueError("Out-of-view or unknown objects cannot have inferred boxes")
            if box is not None:
                box = _box(box, source_frames[index])
            normalized_objects.append({**obj, "box": box})
        normalized_frames.append(
            {**frame, "review": deepcopy(review), "objects": normalized_objects}
        )
    result = deepcopy(reference)
    result["frames"] = normalized_frames
    return result


def reference_summary(reference: dict, sequence_manifest: dict) -> dict:
    """Report evidence coverage; missing source frames prevent a dense reference.

    Annotated frames are those with a declared review, including partial reviews.
    Unreviewed counts include available frames omitted from the reference entirely.
    Human review is a declared provenance, not independently authenticated by IRIS.
    """
    sequence = validate_sequence_manifest(sequence_manifest)
    reference = validate_reference(reference, sequence)
    frames = reference["frames"]
    human = [frame for frame in frames if frame["review"]["status"] == "human_reviewed"]
    assistant = [frame for frame in frames if frame["review"]["status"] == "assistant_reviewed"]
    complete = [frame for frame in frames if frame["coverage"] == "complete"]
    human_complete = [frame for frame in human if frame["coverage"] == "complete"]
    objects = [obj for frame in frames for obj in frame["objects"]]
    available = len(sequence["frames"])
    clip_count = sequence["clip"]["end_frame"] - sequence["clip"]["start_frame"] + 1
    return {
        "available_frames": available,
        "annotated_frames": len(human) + len(assistant),
        "omitted_frames": available - len(frames),
        "human_reviewed_frames": len(human),
        "assistant_reviewed_frames": len(assistant),
        "complete_frames": len(complete),
        "human_complete_frames": len(human_complete),
        "partial_frames": sum(frame["coverage"] == "partial" for frame in frames),
        "unreviewed_frames": available - len(human) - len(assistant),
        "identity_count": len(reference["identities"]),
        "object_count": len(objects),
        "uncertain_objects": sum(obj["certainty"] == "uncertain" for obj in objects),
        "dense_human_reference": available == clip_count == len(human_complete),
    }
