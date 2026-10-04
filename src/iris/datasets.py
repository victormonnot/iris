"""Immutable, reviewed dataset snapshots with persistent scene-group splits."""

import hashlib
import json
import shutil
from pathlib import Path

from iris.annotations import TAXONOMY, _coordinates, _latest
from iris.inference import _load_verified_frame
from iris.media import _file_hash
from iris.store import DEFAULT_PROJECT_ID, Store, _decode, new_id, now

SCHEMA_VERSION = 1
CLASS_MAPPING = {"person": 1, "car": 2}
SPLITS = {"train", "val", "test"}
MAX_FRAMES = 1000
SPLIT_POLICY = (
    "Assign complete scene groups to train, validation or test. Group assignments and exact "
    "image pixels retain their split across dataset versions and declared source splits "
    "of imported datasets. Scene groups belong to their project; exact image pixels retain "
    "their split across this workspace."
)
INDEPENDENCE_WARNING = (
    "Different scene groups are not proof of independent data. Group related sessions and "
    "visually similar scenes together before freezing a dataset."
)


def _canonical(data: dict) -> bytes:
    return json.dumps(
        data, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _manifest_from_row(store: Store, row: dict, *, verify_images: bool = False) -> dict:
    dataset_dir = store.artifact_path(f"datasets/{row['id']}")
    manifest_path = store.artifact_path(row["path"])
    if manifest_path != dataset_dir / "manifest.json":
        raise ValueError("Dataset manifest is outside its version directory")
    try:
        raw = manifest_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["manifest_sha256"]:
            raise ValueError("Dataset manifest no longer matches its recorded hash")
        manifest = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Dataset manifest is missing or unreadable") from exc
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("id") != row["id"]
        or manifest.get("project_id", DEFAULT_PROJECT_ID)
        != row.get("project_id", DEFAULT_PROJECT_ID)
        or manifest.get("taxonomy") != TAXONOMY
        or manifest.get("class_mapping") != CLASS_MAPPING
        or not isinstance(manifest.get("frames"), list)
        or not manifest["frames"]
    ):
        raise ValueError("Dataset manifest has an unsupported format or taxonomy")
    for frame in manifest["frames"]:
        image_path = store.artifact_path(frame["image_path"])
        if image_path != dataset_dir / "images" / f"{frame['frame_id']}.png":
            raise ValueError("Dataset image is outside its version directory")
        if verify_images:
            try:
                if _file_hash(image_path) != frame["image_file_sha256"]:
                    raise ValueError("Frozen dataset image no longer matches its recorded hash")
                with _load_verified_frame(
                    store,
                    {**frame, "id": frame["frame_id"], "path": frame["image_path"]},
                    frame["sha256"],
                ):
                    pass
            except OSError as exc:
                raise ValueError("Frozen dataset image is missing or unreadable") from exc
    return manifest


def load_manifest(store: Store, dataset_id: str, verify_images: bool = False) -> dict:
    """Verify a frozen manifest, optionally including its copied PNGs and pixel hashes."""
    row = store.get("dataset_versions", dataset_id)
    if row is None:
        raise KeyError(dataset_id)
    return _manifest_from_row(store, row, verify_images=verify_images)


def dataset_detail(store: Store, dataset_id: str) -> dict:
    row = store.get("dataset_versions", dataset_id)
    if row is None:
        raise KeyError(dataset_id)
    return {**row, "manifest": _manifest_from_row(store, row)}


def _reservations(store: Store, conn, project_id: str = DEFAULT_PROJECT_ID) -> tuple[dict, dict]:
    """Scope group names to a project while retaining workspace-wide pixel protection."""
    groups, pixels = {}, {}
    for raw_row in conn.execute(
        "SELECT s.scene_group, s.project_id, f.sha256, a.metadata FROM frames f "
        "JOIN sessions s ON s.id=f.session_id JOIN assets a ON a.id=f.asset_id"
    ):
        row = _decode(raw_row)
        split = row["metadata"].get("dataset_import", {}).get("source_split")
        if split is None:
            continue
        group, digest = row["scene_group"], row["sha256"]
        if split not in SPLITS:
            raise ValueError("An imported dataset has an invalid source split")
        in_project = row["project_id"] == project_id
        if (in_project and groups.get(group, split) != split) or pixels.get(digest, split) != split:
            raise ValueError("Imported datasets have conflicting source split assignments")
        if in_project:
            groups[group] = split
        pixels[digest] = split
    for raw_row in conn.execute("SELECT * FROM dataset_versions ORDER BY created_at,id"):
        row = _decode(raw_row)
        manifest = _manifest_from_row(store, row)
        in_project = row["project_id"] == project_id
        for frame in manifest["frames"]:
            group, digest, split = frame["scene_group"], frame["sha256"], frame["split"]
            if split not in SPLITS:
                raise ValueError("An existing dataset has an invalid split")
            if (in_project and groups.get(group, split) != split) or pixels.get(
                digest, split
            ) != split:
                raise ValueError("Existing dataset versions have conflicting split assignments")
            if in_project:
                groups[group] = split
            pixels[digest] = split
    return groups, pixels


def _eligibility(conn, frame: dict) -> tuple[dict | None, str | None]:
    if not frame["selected"]:
        return None, "unselected"
    latest = _latest(conn, frame["id"])
    if latest is None:
        return None, "unannotated"
    if latest["status"] != "validated":
        return latest, "draft"
    suggestion_ids = {
        row[0]
        for row in conn.execute(
            "SELECT id FROM annotation_suggestions WHERE frame_id=?", (frame["id"],)
        )
    }
    if suggestion_ids - latest["decisions"].keys():
        return latest, "pending_suggestions"
    return latest, None


def dataset_candidates(store: Store, project_id: str = DEFAULT_PROJECT_ID) -> dict:
    """Return only currently selected, fully reviewed images, grouped for split assignment."""
    if store.get("projects", project_id) is None:
        raise ValueError("Project does not exist")
    excluded = {
        reason: 0 for reason in ("unselected", "unannotated", "draft", "pending_suggestions")
    }
    groups = {}
    with store.connect() as conn:
        conn.execute("BEGIN")
        reserved_groups, reserved_pixels = _reservations(store, conn, project_id)
        sessions = {
            row["id"]: dict(row)
            for row in conn.execute(
                "SELECT * FROM sessions WHERE project_id=? ORDER BY created_at,id", (project_id,)
            )
        }
        for row in conn.execute(
            "SELECT f.* FROM frames f JOIN sessions s ON s.id=f.session_id "
            "WHERE s.project_id=? ORDER BY f.created_at,f.id",
            (project_id,),
        ):
            frame = _decode(row)
            annotation, reason = _eligibility(conn, frame)
            if reason:
                excluded[reason] += 1
                continue
            session = sessions[frame["session_id"]]
            group = groups.setdefault(
                session["scene_group"],
                {
                    "scene_group": session["scene_group"],
                    "sessions": [],
                    "frames": [],
                    "count": 0,
                    "reserved_split": reserved_groups.get(session["scene_group"]),
                },
            )
            public_session = {"id": session["id"], "name": session["name"]}
            if public_session not in group["sessions"]:
                group["sessions"].append(public_session)
            group["frames"].append(
                {
                    **{
                        key: frame[key] for key in ("id", "session_id", "width", "height", "sha256")
                    },
                    "revision": annotation["revision"],
                    "annotation_revision_id": annotation["id"],
                    "box_count": len(annotation["boxes"]),
                    "reserved_split": reserved_pixels.get(frame["sha256"]),
                }
            )
            group["count"] += 1
    return {
        "groups": sorted(groups.values(), key=lambda group: group["scene_group"]),
        "excluded": excluded,
        "split_policy": SPLIT_POLICY,
        "warnings": [INDEPENDENCE_WARNING],
    }


def _summary(frames: list[dict]) -> dict:
    counts = dict.fromkeys(sorted(SPLITS), 0)
    classes = dict.fromkeys(CLASS_MAPPING, 0)
    split_classes = {split: dict.fromkeys(CLASS_MAPPING, 0) for split in sorted(SPLITS)}
    for frame in frames:
        counts[frame["split"]] += 1
        for box in frame["boxes"]:
            classes[box["label"]] += 1
            split_classes[frame["split"]][box["label"]] += 1
    near_pairs = 0
    for index, frame in enumerate(frames):
        for other in frames[index + 1 :]:
            if frame["split"] == other["split"]:
                continue
            if (
                int(frame["perceptual_hash"], 16) ^ int(other["perceptual_hash"], 16)
            ).bit_count() <= 4:
                near_pairs += 1
    warnings = [INDEPENDENCE_WARNING]
    if near_pairs:
        warnings.append(
            f"{near_pairs} cross-split image pairs have similar perceptual hashes. This heuristic "
            "can miss duplicates or flag unrelated images; review their scene groups."
        )
    if not counts["test"]:
        warnings.append(
            "No test split: validation data must not be reported as an independent test."
        )
    return {
        "frame_count": len(frames),
        "box_count": sum(classes.values()),
        "negative_count": sum(not frame["boxes"] for frame in frames),
        "split_counts": counts,
        "class_counts": classes,
        "split_class_counts": split_classes,
        "scene_groups": sorted({frame["scene_group"] for frame in frames}),
        "near_duplicate_cross_split_pairs": near_pairs,
        "warnings": warnings,
    }


def create_dataset(
    store: Store,
    *,
    name: str,
    frame_ids: list[str],
    splits: dict[str, str],
    parent_id: str | None = None,
    project_id: str = DEFAULT_PROJECT_ID,
) -> dict:
    """Copy a reviewed snapshot and publish it atomically; later edits create new releases."""
    if not isinstance(name, str) or not name.strip() or len(name.strip()) > 160:
        raise ValueError("Dataset name must contain 1 to 160 characters")
    if (
        not isinstance(frame_ids, list)
        or not 1 <= len(frame_ids) <= MAX_FRAMES
        or any(not isinstance(value, str) or not value for value in frame_ids)
        or len(set(frame_ids)) != len(frame_ids)
    ):
        raise ValueError(f"Choose 1 to {MAX_FRAMES} distinct frame IDs")
    if (
        not isinstance(splits, dict)
        or any(not isinstance(group, str) or not group.strip() for group in splits)
        or any(not isinstance(split, str) or split not in SPLITS for split in splits.values())
    ):
        raise ValueError("Assign scene groups to train, val or test")
    if parent_id is not None and (not isinstance(parent_id, str) or not parent_id):
        raise ValueError("Parent dataset ID must identify an existing release")

    identifier, created_at = new_id(), now()
    relative_dir = Path("datasets") / identifier
    destination = store.artifact_path(str(relative_dir))
    staging = store.artifact_path(f"datasets/.staging-{identifier}")
    staging.mkdir(parents=True)
    (staging / "images").mkdir()
    published = False
    try:
        with store.connect() as conn:
            # This lock prevents annotation/suggestion changes and competing split
            # reservations while the exact reviewed snapshot is copied and published.
            conn.execute("BEGIN IMMEDIATE")
            if conn.execute("SELECT id FROM projects WHERE id=?", (project_id,)).fetchone() is None:
                raise ValueError("Project does not exist")
            if parent_id is not None:
                parent = conn.execute(
                    "SELECT * FROM dataset_versions WHERE id=?", (parent_id,)
                ).fetchone()
                if parent is None:
                    raise ValueError("Parent dataset version does not exist")
                if parent["project_id"] != project_id:
                    raise ValueError("Parent dataset version belongs to a different project")
                _manifest_from_row(store, _decode(parent))
            reserved_groups, reserved_pixels = _reservations(store, conn, project_id)
            snapshots, selected_groups, seen_pixels = [], set(), {}
            for frame_id in frame_ids:
                raw_frame = conn.execute("SELECT * FROM frames WHERE id=?", (frame_id,)).fetchone()
                if raw_frame is None:
                    raise ValueError(f"Frame {frame_id} does not exist")
                frame = _decode(raw_frame)
                session = dict(
                    conn.execute(
                        "SELECT * FROM sessions WHERE id=?", (frame["session_id"],)
                    ).fetchone()
                )
                if session["project_id"] != project_id:
                    raise ValueError(f"Frame {frame_id} belongs to a different project")
                annotation, reason = _eligibility(conn, frame)
                if reason:
                    raise ValueError(f"Frame {frame_id} is not eligible for freezing: {reason}")
                if annotation["taxonomy_id"] != TAXONOMY["id"]:
                    raise ValueError("Every annotation must use iris-objects-v1")
                if not annotation["reviewer"] or annotation["frame_sha256"] != frame["sha256"]:
                    raise ValueError("Validated annotation does not match the reviewed image")
                for box in annotation["boxes"]:
                    if box["label"] not in CLASS_MAPPING:
                        raise ValueError("Dataset annotations contain an unsupported class")
                    _coordinates(box["box"], frame)
                asset = _decode(
                    conn.execute("SELECT * FROM assets WHERE id=?", (frame["asset_id"],)).fetchone()
                )
                group = session["scene_group"]
                selected_groups.add(group)
                split = splits.get(group)
                if split is None:
                    raise ValueError(f"Assign a split to scene group {group!r}")
                if reserved_groups.get(group, split) != split:
                    raise ValueError(
                        f"Scene group {group!r} is already reserved for {reserved_groups[group]}"
                    )
                if reserved_pixels.get(frame["sha256"], split) != split:
                    raise ValueError("Image pixels are already reserved for a different split")
                if frame["sha256"] in seen_pixels:
                    if seen_pixels[frame["sha256"]] != split:
                        raise ValueError("Exact duplicate images cannot cross dataset splits")
                    raise ValueError("A dataset version cannot contain duplicate image pixels")
                seen_pixels[frame["sha256"]] = split
                image_path = staging / "images" / f"{frame_id}.png"
                with _load_verified_frame(store, frame, annotation["frame_sha256"]) as image:
                    image.save(image_path, format="PNG")
                snapshots.append(
                    {
                        "frame_id": frame_id,
                        "session_id": frame["session_id"],
                        "session_name": session["name"],
                        "scene_group": group,
                        "split": split,
                        "sha256": frame["sha256"],
                        "width": frame["width"],
                        "height": frame["height"],
                        "perceptual_hash": frame["perceptual_hash"],
                        "image_path": str(relative_dir / "images" / f"{frame_id}.png"),
                        "image_file_sha256": _file_hash(image_path),
                        "annotation_revision_id": annotation["id"],
                        "revision": annotation["revision"],
                        "boxes": annotation["boxes"],
                        "annotation": annotation,
                        "source": {
                            "asset_id": asset["id"],
                            "filename": asset["filename"],
                            "kind": asset["kind"],
                            "sha256": asset["sha256"],
                            "metadata": asset["metadata"],
                            "frame_index": frame["frame_index"],
                            "timestamp_seconds": frame["timestamp_seconds"],
                            "extraction": frame["extraction"],
                        },
                    }
                )
            if set(splits) != selected_groups:
                raise ValueError("Split assignments must exactly match the selected scene groups")
            summary = _summary(snapshots)
            if not summary["split_counts"]["train"] or not summary["split_counts"]["val"]:
                raise ValueError(
                    "A dataset requires nonempty train and val splits from different scene groups"
                )
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "id": identifier,
                "project_id": project_id,
                "name": name.strip(),
                "parent_id": parent_id,
                "created_at": created_at,
                "taxonomy": TAXONOMY,
                "class_mapping": CLASS_MAPPING,
                "splits": splits,
                "split_policy": SPLIT_POLICY,
                "summary": summary,
                "frames": snapshots,
            }
            raw_manifest = _canonical(manifest)
            (staging / "manifest.json").write_bytes(raw_manifest)
            manifest_sha256 = hashlib.sha256(raw_manifest).hexdigest()
            staging.rename(destination)
            published = True
            conn.execute(
                "INSERT INTO dataset_versions "
                "(id,project_id,name,parent_id,path,manifest_sha256,summary,created_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (
                    identifier,
                    project_id,
                    name.strip(),
                    parent_id,
                    str(relative_dir / "manifest.json"),
                    manifest_sha256,
                    json.dumps(summary, allow_nan=False),
                    created_at,
                ),
            )
    except BaseException:
        shutil.rmtree(destination if published else staging, ignore_errors=True)
        raise
    return dataset_detail(store, identifier)
