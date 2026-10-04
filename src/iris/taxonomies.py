"""Immutable project class definitions, independent of annotations and model runtimes."""

import json
import re
from copy import deepcopy
from datetime import datetime

from iris.models import COCO_CATEGORIES
from iris.store import Store, _decode, new_id, now

TAXONOMY = {
    "id": "iris-objects-v1",
    "box_format": "xyxy_pixels",
    "classes": [
        {
            "id": "person",
            "name": "Person",
            "definition": (
                "A visible human, including a rider. Enclose the visible extent of each person; "
                "do not infer a box for a fully occluded person."
            ),
            "coco_id": 1,
        },
        {
            "id": "car",
            "name": "Car",
            "definition": (
                "A passenger car, including an SUV or passenger minivan. Exclude buses, trucks, "
                "motorcycles and bicycles. Enclose the visible extent of each car."
            ),
            "coco_id": 3,
        },
    ],
    "review_guidance": (
        "Review the whole image for missing objects and imprecise boxes. Only validate when "
        "all visible target objects are annotated. A validated empty image is an explicit "
        "negative example. Automatic proposals are never reference annotations by themselves."
    ),
}

_CLASS_ID = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z")
_COCO_IDS = {index for index, name in enumerate(COCO_CATEGORIES) if index and name != "N/A"}


class TaxonomyConflict(ValueError):
    """The project's current definitions changed after the editor was opened."""


def _classes(classes):
    if not isinstance(classes, list) or not 1 <= len(classes) <= 100:
        raise ValueError("A taxonomy needs between 1 and 100 classes")
    result, identifiers, mappings = [], set(), set()
    for item in classes:
        if (
            not isinstance(item, dict)
            or not {"id", "name", "definition"} <= item.keys()
            or item.keys() - {"id", "name", "definition", "coco_id"}
        ):
            raise ValueError("Each class needs an ID, name and definition, with optional COCO ID")
        identifier = item["id"]
        if (
            not isinstance(identifier, str)
            or not _CLASS_ID.fullmatch(identifier)
            or identifier == "exclude"
        ):
            raise ValueError(
                "Class IDs must use 1–64 lowercase letters, digits, underscores or hyphens, "
                "start with a letter, and cannot be 'exclude'"
            )
        if identifier in identifiers:
            raise ValueError("Class IDs must be unique")
        identifiers.add(identifier)
        value = {"id": identifier}
        for key, limit in (("name", 120), ("definition", 2000)):
            field = item[key]
            if not isinstance(field, str) or not 1 <= len(field.strip()) <= limit:
                raise ValueError(f"Class {key} must contain 1–{limit} characters")
            value[key] = field.strip()
        coco_id = item.get("coco_id")
        if coco_id is not None:
            if type(coco_id) is not int or coco_id not in _COCO_IDS:
                raise ValueError("COCO mappings must identify an existing COCO object class")
            if coco_id in mappings:
                raise ValueError("Each COCO class can map to only one project class")
            mappings.add(coco_id)
            value["coco_id"] = coco_id
        result.append(value)
    return result


def _get(conn, taxonomy_id, project_id=None):
    if taxonomy_id == TAXONOMY["id"]:
        return deepcopy(TAXONOMY)
    row = _decode(
        conn.execute("SELECT * FROM taxonomy_versions WHERE id=?", (taxonomy_id,)).fetchone()
    )
    if row is None or project_id is not None and row["project_id"] != project_id:
        raise ValueError("Taxonomy does not exist in this project")
    return row["snapshot"]


def get_taxonomy(store: Store, taxonomy_id: str, project_id: str | None = None) -> dict:
    """Read an independent snapshot without changing the historical builtin payload."""
    with store.connect() as conn:
        if (
            project_id is not None
            and conn.execute("SELECT id FROM projects WHERE id=?", (project_id,)).fetchone() is None
        ):
            raise ValueError("Project does not exist")
        return _get(conn, taxonomy_id, project_id)


def current_taxonomy(store: Store, project_id: str) -> dict:
    with store.connect() as conn:
        conn.execute("BEGIN")
        project = conn.execute("SELECT * FROM projects WHERE id=?", (project_id,)).fetchone()
        if project is None:
            raise ValueError("Project does not exist")
        return _get(conn, project["taxonomy_id"], project_id)


def list_taxonomies(store: Store, project_id: str) -> list[dict]:
    with store.connect() as conn:
        conn.execute("BEGIN")
        project = conn.execute("SELECT * FROM projects WHERE id=?", (project_id,)).fetchone()
        if project is None:
            raise ValueError("Project does not exist")
        builtin = {
            **deepcopy(TAXONOMY),
            "version": 1,
            "parent_id": None,
            "created_at": project["created_at"],
        }
        return [builtin] + [
            _decode(row)["snapshot"]
            for row in conn.execute(
                "SELECT * FROM taxonomy_versions WHERE project_id=? ORDER BY version",
                (project_id,),
            )
        ]


def publish_taxonomy(
    store: Store, project_id: str, *, expected_taxonomy_id: str, classes: list[dict]
) -> dict:
    """Publish one immutable definition and switch the project pointer atomically.

    The first custom version may replace the starter classes. Subsequent versions
    keep all previously published IDs so historical labels retain their identity.
    """
    values = _classes(classes)
    with store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        project = conn.execute("SELECT * FROM projects WHERE id=?", (project_id,)).fetchone()
        if project is None:
            raise ValueError("Project does not exist")
        if project["taxonomy_id"] != expected_taxonomy_id:
            raise TaxonomyConflict("Project classes changed; reload them before publishing")
        previous = _get(conn, expected_taxonomy_id, project_id)
        if previous["id"] != TAXONOMY["id"] and not {
            item["id"] for item in previous["classes"]
        } <= {item["id"] for item in values}:
            raise ValueError("Published class IDs cannot be removed or renamed")
        if values == previous["classes"]:
            raise ValueError("Class definitions have not changed")
        identifier, created_at = f"taxonomy-{new_id()}", now()
        version = previous.get("version", 1) + 1
        snapshot = {
            "id": identifier,
            "version": version,
            "parent_id": previous["id"],
            "classes": values,
            "box_format": TAXONOMY["box_format"],
            "review_guidance": TAXONOMY["review_guidance"],
            "created_at": created_at,
        }
        conn.execute(
            "INSERT INTO taxonomy_versions (id,project_id,version,parent_id,snapshot,created_at) "
            "VALUES (?,?,?,?,?,?)",
            (
                identifier,
                project_id,
                version,
                previous["id"],
                json.dumps(snapshot, allow_nan=False),
                created_at,
            ),
        )
        conn.execute("UPDATE projects SET taxonomy_id=? WHERE id=?", (identifier, project_id))
    return snapshot


def validate_taxonomy_records(conn):
    """Check references that share a virtual builtin and cannot use a SQLite FK.

    Archive readers call this on their read-only connection, without constructing
    Store or migrating the database under inspection.
    """
    projects = {row["id"]: dict(row) for row in conn.execute("SELECT * FROM projects")}
    versions, latest = {}, {identifier: TAXONOMY["id"] for identifier in projects}
    expected_fields = {
        "id",
        "version",
        "parent_id",
        "classes",
        "box_format",
        "review_guidance",
        "created_at",
    }
    for raw in conn.execute("SELECT * FROM taxonomy_versions ORDER BY project_id,version"):
        row = _decode(raw)
        snapshot = row["snapshot"]
        project_id = row["project_id"]
        if (
            project_id not in projects
            or not isinstance(snapshot, dict)
            or set(snapshot) != expected_fields
            or row["id"] == TAXONOMY["id"]
            or not row["id"]
            or any(
                snapshot.get(key) != row[key]
                for key in ("id", "version", "parent_id", "created_at")
            )
            or type(snapshot["version"]) is not int
            or snapshot["box_format"] != TAXONOMY["box_format"]
            or snapshot["review_guidance"] != TAXONOMY["review_guidance"]
            or row["parent_id"] != latest[project_id]
        ):
            raise ValueError("Taxonomy version identity or project history is inconsistent")
        if (
            not isinstance(snapshot["created_at"], str)
            or datetime.fromisoformat(snapshot["created_at"]).tzinfo is None
        ):
            raise ValueError("Taxonomy creation time must include a timezone")
        classes = _classes(snapshot["classes"])
        if classes != snapshot["classes"]:
            raise ValueError("Taxonomy classes are not normalized")
        previous = TAXONOMY if row["parent_id"] == TAXONOMY["id"] else versions[row["parent_id"]][1]
        if row["version"] != previous.get("version", 1) + 1:
            raise ValueError("Taxonomy version numbers must preserve publication order")
        if previous["id"] != TAXONOMY["id"] and not {
            item["id"] for item in previous["classes"]
        } <= {item["id"] for item in classes}:
            raise ValueError("Taxonomy history removed or renamed a published class ID")
        versions[row["id"]] = (project_id, snapshot)
        latest[project_id] = row["id"]
    for identifier, project in projects.items():
        if project["taxonomy_id"] != latest[identifier]:
            raise ValueError("Project taxonomy pointer does not match its latest version")
    for query in (
        "SELECT f.taxonomy_id,s.project_id FROM frames f JOIN sessions s ON s.id=f.session_id",
        "SELECT r.taxonomy_id,s.project_id FROM annotation_revisions r "
        "JOIN frames f ON f.id=r.frame_id JOIN sessions s ON s.id=f.session_id",
    ):
        for row in conn.execute(query):
            identifier = row["taxonomy_id"]
            if identifier != TAXONOMY["id"] and (
                identifier not in versions or versions[identifier][0] != row["project_id"]
            ):
                raise ValueError(
                    "Saved annotation or frame taxonomy is missing or in another project"
                )
