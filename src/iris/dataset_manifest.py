"""Self-contained class contracts for frozen datasets, without live workspace lookups."""

import re

from iris.taxonomies import TAXONOMY, _classes

SCHEMA_VERSION = 2
LEGACY_CLASS_MAPPING = {"person": 1, "car": 2}
LEGACY_COCO_MAPPING = {"person": 1, "car": 3}
_TAXONOMY_ID = re.compile(r"taxonomy-[0-9a-f]{32}\Z")


def taxonomy_mappings(taxonomy: dict) -> tuple[dict, dict]:
    """Validate a frozen definition and derive its training-head and export category IDs."""
    if not isinstance(taxonomy, dict):
        raise ValueError("Frozen dataset taxonomy must be a complete class snapshot")
    if taxonomy.get("id") == TAXONOMY["id"]:
        if taxonomy != TAXONOMY or taxonomy["classes"] != _classes(taxonomy["classes"]):
            raise ValueError("Frozen builtin taxonomy must retain its original class definitions")
        return LEGACY_CLASS_MAPPING.copy(), LEGACY_COCO_MAPPING.copy()
    if (
        set(taxonomy)
        != {"id", "version", "parent_id", "classes", "box_format", "review_guidance", "created_at"}
        or not isinstance(taxonomy["id"], str)
        or not _TAXONOMY_ID.fullmatch(taxonomy["id"])
        or type(taxonomy["version"]) is not int
        or taxonomy["version"] < 2
        or not isinstance(taxonomy["parent_id"], str)
        or not (
            taxonomy["parent_id"] == TAXONOMY["id"] or _TAXONOMY_ID.fullmatch(taxonomy["parent_id"])
        )
        or taxonomy["parent_id"] == taxonomy["id"]
        or (taxonomy["version"] == 2) != (taxonomy["parent_id"] == TAXONOMY["id"])
        or taxonomy["box_format"] != TAXONOMY["box_format"]
        or taxonomy["review_guidance"] != TAXONOMY["review_guidance"]
        or not isinstance(taxonomy["created_at"], str)
        or not taxonomy["created_at"]
    ):
        raise ValueError("Frozen custom taxonomy has an unsupported snapshot format")
    if taxonomy["classes"] != _classes(taxonomy["classes"]):
        raise ValueError("Frozen class definitions must retain their normalized values")
    mapping = {item["id"]: index for index, item in enumerate(taxonomy["classes"], 1)}
    return mapping, mapping.copy()


def _matches_mapping(value, expected: dict) -> bool:
    return (
        isinstance(value, dict)
        and value == expected
        and all(type(identifier) is int for identifier in value.values())
    )


def manifest_mappings(manifest: dict) -> tuple[dict, dict, dict]:
    """Validate schema 1/2 class snapshots using only the manifest's frozen contents."""
    if (
        not isinstance(manifest, dict)
        or type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] not in {1, SCHEMA_VERSION}
    ):
        raise ValueError("Dataset manifest has an unsupported schema version")
    taxonomy = manifest.get("taxonomy")
    class_mapping, coco_mapping = taxonomy_mappings(taxonomy)
    if manifest["schema_version"] == 1 and taxonomy != TAXONOMY:
        raise ValueError("Legacy dataset manifests require the original builtin taxonomy")
    if not _matches_mapping(manifest.get("class_mapping"), class_mapping):
        raise ValueError("Frozen dataset class mapping does not match its class definitions")
    if manifest["schema_version"] == SCHEMA_VERSION or "coco_mapping" in manifest:
        if not _matches_mapping(manifest.get("coco_mapping"), coco_mapping):
            raise ValueError("Frozen dataset COCO mapping does not match its class definitions")
    return taxonomy, class_mapping, coco_mapping
