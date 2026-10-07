"""Immutable temporal sources, reference revisions and leakage-aware dataset versions.

This layer does not run inference, assign tracker IDs or promote image annotations
to identity ground truth. All public writes publish a complete document atomically.
"""

import hashlib
import json
import math

from iris.store import DEFAULT_PROJECT_ID, Store, _decode, _encode, new_id, now
from iris.taxonomies import TAXONOMY
from iris.temporal_contracts import (
    MAX_SEQUENCE_FRAMES,
    reference_summary,
    sequence_hash,
    validate_reference,
    validate_sequence_manifest,
)

DATASET_SCHEMA = "iris-temporal-dataset-v1"
MAX_SEQUENCES = 1000
EVALUATION_POLICY = {
    "identity_scope": "sequence",
    "assistant_reviewed": "diagnostic_only",
    "unreviewed": "exclude",
    "uncertain": "exclude",
    "predicted": "exclude",
}
INDEPENDENCE_WARNING = (
    "Source videos, identical pixels and declared scene/take groups retain their split. "
    "Different files or group names do not prove independence: group related takes before "
    "freezing a dataset. Missing frames and unreviewed identities are not negative examples."
)


class TemporalConflict(RuntimeError):
    """An immutable source or the expected reference revision changed."""


def _digest(value):
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _text(value, field, limit=160):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"{field} must contain 1–{limit} characters")
    if any(ord(character) < 32 for character in value):
        raise ValueError(f"{field} must not contain control characters")
    return value.strip()


def _valid_fps(value):
    try:
        return type(value) in (int, float) and math.isfinite(value) and value > 0
    except OverflowError:
        return False


def _row(conn, table, identifier):
    # Table names are exclusively internal constants, never HTTP input.
    record = _decode(conn.execute(f"SELECT * FROM {table} WHERE id=?", (identifier,)).fetchone())
    if record is None:
        raise KeyError(identifier)
    return record


def _insert(conn, table, row):
    data = _encode(row)
    conn.execute(
        f"INSERT INTO {table} ({','.join(data)}) VALUES ({','.join('?' for _ in data)})",
        list(data.values()),
    )


def _taxonomy(conn, project_id, taxonomy_id):
    if taxonomy_id == TAXONOMY["id"]:
        return TAXONOMY
    row = _row(conn, "taxonomy_versions", taxonomy_id)
    if row["project_id"] != project_id:
        raise ValueError("Temporal taxonomy belongs to another project")
    return row["snapshot"]


def _sequence_record(conn, row):
    manifest = validate_sequence_manifest(row["manifest"])
    if (
        sequence_hash(manifest) != row["manifest_sha256"]
        or _digest(row["manifest"]) != row["manifest_sha256"]
    ):
        raise ValueError("Temporal sequence checksum does not match its manifest")
    if any(manifest[key] != row[key] for key in ("id", "project_id", "name", "parent_id")):
        raise ValueError("Temporal sequence metadata does not match its manifest")
    asset = _row(conn, "assets", row["asset_id"])
    session = _row(conn, "sessions", asset["session_id"])
    if (
        asset["kind"] != "video"
        or session["project_id"] != row["project_id"]
        or manifest["asset"]
        != {
            "id": asset["id"],
            "sha256": asset["sha256"],
            "session_id": session["id"],
            "scene_group": session["scene_group"],
        }
    ):
        raise ValueError("Temporal sequence source provenance or ownership changed")
    metadata = asset["metadata"]
    frame_count = metadata.get("frame_count")
    if type(frame_count) is not int or not 0 <= manifest["clip"]["end_frame"] < frame_count:
        raise ValueError("Temporal clip is outside its source video")
    clock = manifest["clock"]
    if clock["basis"] == "nominal_fps" and clock["fps"] != metadata.get("fps"):
        raise ValueError("Temporal nominal clock does not match source metadata")
    if manifest["taxonomy"] != _taxonomy(conn, row["project_id"], manifest["taxonomy"]["id"]):
        raise ValueError("Temporal taxonomy no longer matches its frozen definition")
    for frame in manifest["frames"]:
        saved = _row(conn, "frames", frame["frame_id"])
        if (
            saved["asset_id"] != row["asset_id"]
            or saved["session_id"] != session["id"]
            or any(saved[key] != frame[key] for key in ("frame_index", "sha256", "width", "height"))
        ):
            raise ValueError("Temporal frame provenance no longer matches its source")
    if row["parent_id"] is not None:
        parent = _row(conn, "temporal_sequences", row["parent_id"])
        if (
            parent["id"] == row["id"]
            or parent["project_id"] != row["project_id"]
            or parent["manifest"]["asset"]["sha256"] != asset["sha256"]
            or parent["manifest"]["take_group"] != manifest["take_group"]
        ):
            raise ValueError("Temporal sequence parent has a different source, project or take")
    return {**row, "manifest": manifest}


def _reference_record(conn, row, sequence=None):
    sequence = sequence or _sequence_record(
        conn, _row(conn, "temporal_sequences", row["sequence_id"])
    )
    payload = validate_reference(row["payload"], sequence["manifest"])
    if (
        payload["sequence_id"] != row["sequence_id"]
        or _digest(payload) != row["payload_sha256"]
        or _digest(row["payload"]) != row["payload_sha256"]
        or type(row["revision"]) is not int
        or row["revision"] < 1
    ):
        raise ValueError("Temporal reference revision or checksum is invalid")
    return {**row, "payload": payload, "summary": reference_summary(payload, sequence["manifest"])}


def _automatic_gaps(frames, clip):
    cursor, result = clip["start_frame"], []
    for frame in frames:
        if frame["frame_index"] > cursor:
            result.append(
                {"start_frame": cursor, "end_frame": frame["frame_index"] - 1, "reason": "unknown"}
            )
        cursor = frame["frame_index"] + 1
    if cursor <= clip["end_frame"]:
        result.append({"start_frame": cursor, "end_frame": clip["end_frame"], "reason": "unknown"})
    return result


def _verify_sequence_media(store, conn, sequence):
    """Bind new evidence to the bytes originally frozen, not only their DB metadata."""
    from iris.inference import _load_verified_frame
    from iris.media import _file_hash

    try:
        asset = _row(conn, "assets", sequence["asset_id"])
        if (
            _file_hash(store.artifact_path(asset["path"]))
            != sequence["manifest"]["asset"]["sha256"]
        ):
            raise ValueError("Source video bytes no longer match the frozen sequence")
        for frozen in sequence["manifest"]["frames"]:
            frame = _row(conn, "frames", frozen["frame_id"])
            if _file_hash(store.artifact_path(frame["path"])) != frozen["file_sha256"]:
                raise ValueError("Source frame bytes no longer match the frozen sequence")
            with _load_verified_frame(store, frame, frozen["sha256"]):
                pass
    except OSError as exc:
        raise ValueError("Temporal source media is missing or unreadable") from exc


def create_sequence(
    store: Store,
    *,
    name,
    asset_id,
    frame_ids,
    project_id=DEFAULT_PROJECT_ID,
    take_group=None,
    parent_id=None,
    clip=None,
    clock=None,
    timestamps=None,
    gaps=None,
):
    """Freeze already extracted video frames; never invent timestamps or identities."""
    from iris.media import _file_hash

    name = _text(name, "Sequence name")
    if (
        not isinstance(frame_ids, list)
        or not 1 <= len(frame_ids) <= MAX_SEQUENCE_FRAMES
        or any(not isinstance(value, str) for value in frame_ids)
        or len(set(frame_ids)) != len(frame_ids)
    ):
        raise ValueError("Select 1–10000 distinct extracted video frames")
    with store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        project = _row(conn, "projects", project_id)
        asset = _row(conn, "assets", asset_id)
        session = _row(conn, "sessions", asset["session_id"])
        if asset["kind"] != "video" or session["project_id"] != project_id:
            raise ValueError("Select a source video from this project")
        parent = (
            _sequence_record(conn, _row(conn, "temporal_sequences", parent_id))
            if parent_id
            else None
        )
        take_group = _text(
            take_group
            if take_group is not None
            else (parent["manifest"]["take_group"] if parent else session["scene_group"]),
            "Take group",
        )
        frames = [_row(conn, "frames", identifier) for identifier in frame_ids]
        if any(
            frame["asset_id"] != asset_id
            or frame["session_id"] != session["id"]
            or type(frame["frame_index"]) is not int
            for frame in frames
        ):
            raise ValueError("Every temporal frame must belong to the same source video")
        frames.sort(key=lambda frame: frame["frame_index"])
        if clock is None:
            fps = asset["metadata"].get("fps")
            if _valid_fps(fps):
                clock = {
                    "basis": "nominal_fps",
                    "fps": float(fps),
                    "provenance": "frame_index / nominal_fps (estimated, not capture timestamps)",
                }
            else:
                clock = {"basis": "unknown", "fps": None, "provenance": "No usable source clock"}
        if not isinstance(clock, dict):
            raise ValueError("Temporal clock must be an object")
        if clock.get("basis") == "provided":
            if not isinstance(timestamps, dict) or set(timestamps) != set(frame_ids):
                raise ValueError("Provided timestamps must cover exactly the selected frames")
        elif timestamps is not None:
            raise ValueError("Explicit timestamps require a provided clock with provenance")
        frozen = []
        for frame in frames:
            timestamp = None
            if clock.get("basis") == "nominal_fps":
                fps = clock.get("fps")
                if not _valid_fps(fps):
                    raise ValueError("Nominal FPS must be finite and positive")
                timestamp = frame["frame_index"] / fps
            elif clock.get("basis") == "provided":
                timestamp = timestamps[frame["id"]]
            frozen.append(
                {
                    "frame_id": frame["id"],
                    "frame_index": frame["frame_index"],
                    "timestamp_seconds": timestamp,
                    "file_sha256": _file_hash(store.artifact_path(frame["path"])),
                    **{key: frame[key] for key in ("width", "height", "sha256")},
                }
            )
        clip = (
            clip
            if clip is not None
            else {"start_frame": frozen[0]["frame_index"], "end_frame": frozen[-1]["frame_index"]}
        )
        if (
            not isinstance(clip, dict)
            or set(clip) != {"start_frame", "end_frame"}
            or any(type(value) is not int for value in clip.values())
        ):
            raise ValueError("Clip must contain integer start_frame and end_frame bounds")
        manifest = validate_sequence_manifest(
            {
                "schema": "iris-temporal-sequence-v1",
                "id": new_id(),
                "project_id": project_id,
                "name": name,
                "parent_id": parent_id,
                "asset": {
                    "id": asset_id,
                    "sha256": asset["sha256"],
                    "session_id": session["id"],
                    "scene_group": session["scene_group"],
                },
                "take_group": take_group,
                "taxonomy": _taxonomy(conn, project_id, project["taxonomy_id"]),
                "clock": clock,
                "clip": clip,
                "frames": frozen,
                "gaps": gaps if gaps is not None else _automatic_gaps(frozen, clip),
            }
        )
        row = {
            "id": manifest["id"],
            "project_id": project_id,
            "asset_id": asset_id,
            "parent_id": parent_id,
            "name": name,
            "manifest": manifest,
            "manifest_sha256": sequence_hash(manifest),
            "created_at": now(),
        }
        _sequence_record(conn, row)
        _verify_sequence_media(store, conn, row)
        _insert(conn, "temporal_sequences", row)
    return row


def sequence_detail(store, sequence_id):
    with store.connect() as conn:
        conn.execute("BEGIN")
        row = _sequence_record(conn, _row(conn, "temporal_sequences", sequence_id))
        latest = conn.execute(
            "SELECT * FROM temporal_references WHERE sequence_id=? ORDER BY revision DESC LIMIT 1",
            (sequence_id,),
        ).fetchone()
        return {
            **row,
            "latest_reference": _reference_record(conn, _decode(latest), row) if latest else None,
        }


def list_sequences(store, project_id=DEFAULT_PROJECT_ID):
    return [
        sequence_detail(store, row["id"])
        for row in store.list("temporal_sequences", project_id=project_id)
    ]


def save_reference(store, sequence_id, *, payload, expected_revision=0):
    if type(expected_revision) is not int or expected_revision < 0:
        raise ValueError("Expected revision must be a nonnegative integer")
    with store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        sequence = _sequence_record(conn, _row(conn, "temporal_sequences", sequence_id))
        actual = conn.execute(
            "SELECT COALESCE(MAX(revision),0) FROM temporal_references WHERE sequence_id=?",
            (sequence_id,),
        ).fetchone()[0]
        if actual != expected_revision:
            raise TemporalConflict(
                f"Temporal reference changed: expected revision {expected_revision}, found {actual}"
            )
        payload = validate_reference(payload, sequence["manifest"])
        _verify_sequence_media(store, conn, sequence)
        row = {
            "id": new_id(),
            "sequence_id": sequence_id,
            "revision": actual + 1,
            "payload": payload,
            "payload_sha256": _digest(payload),
            "created_at": now(),
        }
        _insert(conn, "temporal_references", row)
        return _reference_record(conn, row, sequence)


def reference_detail(store, reference_id):
    with store.connect() as conn:
        conn.execute("BEGIN")
        return _reference_record(conn, _row(conn, "temporal_references", reference_id))


def list_references(store, sequence_id):
    with store.connect() as conn:
        conn.execute("BEGIN")
        sequence = _sequence_record(conn, _row(conn, "temporal_sequences", sequence_id))
        return [
            _reference_record(conn, _decode(row), sequence)
            for row in conn.execute(
                "SELECT * FROM temporal_references WHERE sequence_id=? ORDER BY revision DESC",
                (sequence_id,),
            )
        ]


def _dataset_record(conn, row):
    manifest = row["manifest"]
    if (
        not isinstance(manifest, dict)
        or set(manifest)
        != {
            "schema",
            "id",
            "project_id",
            "parent_id",
            "name",
            "entries",
            "evaluation_policy",
            "notes",
        }
        or manifest["schema"] != DATASET_SCHEMA
    ):
        raise ValueError("Unsupported temporal dataset manifest")
    if (
        any(manifest[key] != row[key] for key in ("id", "project_id", "parent_id", "name"))
        or _digest(manifest) != row["manifest_sha256"]
        or manifest["evaluation_policy"] != EVALUATION_POLICY
        or not isinstance(manifest["notes"], str)
        or len(manifest["notes"]) > 4000
    ):
        raise ValueError("Temporal dataset metadata, policy or checksum is invalid")
    _text(manifest["name"], "Dataset name")
    _row(conn, "projects", row["project_id"])
    if row["parent_id"] is not None:
        parent = _row(conn, "temporal_datasets", row["parent_id"])
        if parent["project_id"] != row["project_id"] or parent["id"] == row["id"]:
            raise ValueError("Temporal dataset parent belongs to another project")
    entries, seen, taxonomy = manifest["entries"], set(), None
    if not isinstance(entries, list) or not 1 <= len(entries) <= MAX_SEQUENCES:
        raise ValueError("Select 1–1000 temporal sequences per dataset")
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {
            "sequence_id",
            "sequence_sha256",
            "reference_id",
            "reference_sha256",
            "split",
        }:
            raise ValueError("Temporal dataset entries have an unsupported format")
        if not isinstance(entry["sequence_id"], str) or entry["sequence_id"] in seen:
            raise ValueError("Temporal dataset sequences must be distinct")
        seen.add(entry["sequence_id"])
        if entry["split"] not in ("train", "val", "test"):
            raise ValueError("Temporal split must be train, val or test")
        sequence = _sequence_record(conn, _row(conn, "temporal_sequences", entry["sequence_id"]))
        if (
            sequence["project_id"] != row["project_id"]
            or sequence["manifest_sha256"] != entry["sequence_sha256"]
        ):
            raise ValueError("Temporal dataset sequence ownership or checksum changed")
        current = sequence["manifest"]["taxonomy"]
        if taxonomy is not None and taxonomy != current:
            raise ValueError("Temporal dataset sequences must use one frozen taxonomy")
        taxonomy = current
        if entry["reference_id"] is None:
            if entry["reference_sha256"] is not None:
                raise ValueError("A missing temporal reference cannot have a checksum")
        else:
            if not isinstance(entry["reference_id"], str):
                raise ValueError("Temporal reference ID must be a string or null")
            reference = _reference_record(
                conn, _row(conn, "temporal_references", entry["reference_id"]), sequence
            )
            if (
                reference["sequence_id"] != sequence["id"]
                or reference["payload_sha256"] != entry["reference_sha256"]
            ):
                raise ValueError("Temporal dataset reference owner or checksum changed")
    return row


def _assign(mapping, key, split, label):
    if key in mapping and mapping[key] != split:
        raise ValueError(f"Temporal split conflict: {label} is already reserved for {mapping[key]}")
    mapping[key] = split


def _reserve_sequence(manifest, split, groups, pixels, videos, *, in_project):
    if in_project:
        for group in (manifest["asset"]["scene_group"], manifest["take_group"]):
            _assign(groups, group, split, "scene/take group")
    digest = manifest["asset"]["sha256"]
    existing = videos.get(digest, set())
    if existing and existing != {split}:
        raise ValueError("Temporal split conflict: the source video is already reserved")
    videos.setdefault(digest, set()).add(split)
    for frame in manifest["frames"]:
        _assign(pixels, frame["sha256"], split, "identical image pixels")


def temporal_reservations(conn, project_id):
    """Share split reservations with image datasets; source/pixel identities are global."""
    groups, pixels, videos = {}, {}, {}
    for raw in conn.execute("SELECT * FROM temporal_datasets ORDER BY created_at,id"):
        dataset = _dataset_record(conn, _decode(raw))
        for entry in dataset["manifest"]["entries"]:
            sequence = _row(conn, "temporal_sequences", entry["sequence_id"])
            _reserve_sequence(
                sequence["manifest"],
                entry["split"],
                groups,
                pixels,
                videos,
                in_project=dataset["project_id"] == project_id,
            )
    return groups, pixels, videos


def create_temporal_dataset(
    store, *, name, entries, project_id=DEFAULT_PROJECT_ID, parent_id=None, notes=""
):
    from iris.datasets import _reservation_state

    name = _text(name, "Dataset name")
    if not isinstance(entries, list) or not 1 <= len(entries) <= MAX_SEQUENCES:
        raise ValueError("Select 1–1000 temporal sequences per dataset")
    with store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        frozen = []
        for entry in entries:
            if (
                not isinstance(entry, dict)
                or not {"sequence_id", "split"} <= set(entry)
                or set(entry) - {"sequence_id", "split", "reference_id"}
            ):
                raise ValueError("Each entry needs a sequence_id, split and optional reference_id")
            if not isinstance(entry["sequence_id"], str):
                raise ValueError("Sequence ID must be a string")
            sequence = _sequence_record(
                conn, _row(conn, "temporal_sequences", entry["sequence_id"])
            )
            reference_id = entry.get("reference_id")
            reference = (
                _reference_record(conn, _row(conn, "temporal_references", reference_id), sequence)
                if reference_id is not None
                else None
            )
            frozen.append(
                {
                    "sequence_id": sequence["id"],
                    "sequence_sha256": sequence["manifest_sha256"],
                    "reference_id": reference_id,
                    "reference_sha256": reference["payload_sha256"] if reference else None,
                    "split": entry["split"],
                }
            )
        identifier = new_id()
        manifest = {
            "schema": DATASET_SCHEMA,
            "id": identifier,
            "project_id": project_id,
            "parent_id": parent_id,
            "name": name,
            "entries": frozen,
            "evaluation_policy": EVALUATION_POLICY.copy(),
            "notes": notes,
        }
        row = {
            "id": identifier,
            "project_id": project_id,
            "parent_id": parent_id,
            "name": name,
            "manifest": manifest,
            "manifest_sha256": _digest(manifest),
            "created_at": now(),
        }
        _dataset_record(conn, row)
        groups, pixels, videos = _reservation_state(store, conn, project_id)
        for entry in frozen:
            sequence = _row(conn, "temporal_sequences", entry["sequence_id"])
            _verify_sequence_media(store, conn, sequence)
            _reserve_sequence(
                sequence["manifest"], entry["split"], groups, pixels, videos, in_project=True
            )
        _insert(conn, "temporal_datasets", row)
    return {**row, "independence_warning": INDEPENDENCE_WARNING}


def temporal_dataset_detail(store, dataset_id):
    with store.connect() as conn:
        conn.execute("BEGIN")
        row = _dataset_record(conn, _row(conn, "temporal_datasets", dataset_id))
        return {**row, "independence_warning": INDEPENDENCE_WARNING}


def list_temporal_datasets(store, project_id=DEFAULT_PROJECT_ID):
    return [
        temporal_dataset_detail(store, row["id"])
        for row in store.list("temporal_datasets", project_id=project_id)
    ]


def _acyclic(rows):
    parents, done = {row["id"]: row["parent_id"] for row in rows}, set()
    for identifier in parents:
        path, cursor = set(), identifier
        while cursor is not None and cursor not in done:
            if cursor in path or cursor not in parents:
                raise ValueError("Temporal version history contains a cycle or missing parent")
            path.add(cursor)
            cursor = parents[cursor]
        done.update(path)


def validate_temporal_records(connection):
    """Read-only archive validation; does not construct Store or touch the filesystem."""
    try:
        sequences = [
            _sequence_record(connection, _decode(row))
            for row in connection.execute("SELECT * FROM temporal_sequences")
        ]
        _acyclic(sequences)
        revisions = {}
        for raw in connection.execute(
            "SELECT * FROM temporal_references ORDER BY sequence_id,revision"
        ):
            row = _reference_record(connection, _decode(raw))
            expected = revisions.get(row["sequence_id"], 0) + 1
            if row["revision"] != expected:
                raise ValueError("Temporal reference history has missing revisions")
            revisions[row["sequence_id"]] = expected
        datasets = [
            _dataset_record(connection, _decode(row))
            for row in connection.execute("SELECT * FROM temporal_datasets")
        ]
        _acyclic(datasets)
        for project_id in {row["project_id"] for row in datasets}:
            temporal_reservations(connection, project_id)
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError("Temporal records contain invalid or missing source references") from exc


def validate_temporal_dataset_splits(connection, dataset_manifests):
    """Check classic frozen datasets/imports against temporal split reservations."""
    project_ids = {
        row[0] for row in connection.execute("SELECT DISTINCT project_id FROM temporal_datasets")
    }
    if not project_ids:
        return
    imported = list(
        connection.execute(
            "SELECT s.project_id,s.scene_group,f.sha256,a.kind,"
            "a.sha256 AS source_sha256,a.metadata "
            "FROM frames f JOIN sessions s ON s.id=f.session_id JOIN assets a ON a.id=f.asset_id"
        )
    )
    for project_id in project_ids:
        groups, pixels, videos = temporal_reservations(connection, project_id)

        def check(mapping, key, split, label):
            # Only compare with temporal reservations. Historical image datasets
            # may contain unrelated old conflicts; T1 must not reinterpret them.
            if key in mapping and mapping[key] != split:
                raise ValueError(f"Temporal dataset conflicts with {label}")

        for manifest in dataset_manifests:
            for frame in manifest["frames"]:
                if manifest.get("project_id", DEFAULT_PROJECT_ID) == project_id:
                    check(groups, frame["scene_group"], frame["split"], "a frozen scene group")
                check(pixels, frame["sha256"], frame["split"], "frozen image pixels")
                source = frame.get("source", {})
                if source.get("kind") == "video":
                    check(
                        videos,
                        source["sha256"],
                        {frame["split"]},
                        "a frozen image dataset source split",
                    )
        for raw in imported:
            row = _decode(raw)
            split = row["metadata"].get("dataset_import", {}).get("source_split")
            if split is None:
                continue
            if row["project_id"] == project_id:
                check(groups, row["scene_group"], split, "an imported scene group")
            check(pixels, row["sha256"], split, "imported image pixels")
            if (
                row["kind"] == "video"
                and row["source_sha256"] in videos
                and videos[row["source_sha256"]] != {split}
            ):
                raise ValueError("Temporal dataset conflicts with an imported source split")
