"""Frozen detector class contracts; legacy checkpoints retain their original COCO outputs."""

from copy import deepcopy


def class_contract(value: dict) -> dict:
    """Resolve a config/spec/metadata payload, never infer custom labels from numeric IDs."""
    # Import lazily: taxonomies uses the official category catalog from models.
    from iris.dataset_manifest import taxonomy_mappings
    from iris.taxonomies import TAXONOMY

    if not isinstance(value, dict):
        raise ValueError("Checkpoint class contract must be an object")
    taxonomy = value.get("taxonomy")
    if taxonomy is None:
        if (
            "taxonomy" in value
            or "output_class_mapping" in value
            or value.get("taxonomy_id", TAXONOMY["id"]) != TAXONOMY["id"]
        ):
            raise ValueError("Checkpoint has no complete frozen class definition")
        taxonomy = TAXONOMY
    internal, output = taxonomy_mappings(taxonomy)
    if value.get("taxonomy_id", taxonomy["id"]) != taxonomy["id"]:
        raise ValueError("Checkpoint taxonomy ID does not match its frozen class definition")
    for field, expected in (("class_mapping", internal), ("output_class_mapping", output)):
        actual = value.get(field, expected if "taxonomy" not in value else None)
        if (
            actual != expected
            or not isinstance(actual, dict)
            or any(type(identifier) is not int for identifier in actual.values())
        ):
            raise ValueError(f"Checkpoint {field} does not match its frozen class definition")
    return {
        "taxonomy": deepcopy(taxonomy),
        "taxonomy_id": taxonomy["id"],
        "class_mapping": internal,
        "output_class_mapping": output,
    }


def dataset_contract(manifest: dict) -> dict:
    """Read the training/output namespace from frozen dataset definitions and mappings."""
    from iris.dataset_manifest import taxonomy_mappings

    internal, output = taxonomy_mappings(manifest["taxonomy"])
    return class_contract(
        {
            "taxonomy": manifest["taxonomy"],
            "taxonomy_id": manifest["taxonomy"]["id"],
            "class_mapping": manifest["class_mapping"],
            "output_class_mapping": manifest.get("coco_mapping", output),
        }
    )


def compatible_parent(parent: dict, contract: dict) -> None:
    """A trained head may continue only with exactly the same frozen semantics and slots."""
    if parent.get("origin") == "trained" and class_contract(parent) != contract:
        raise ValueError(
            "The trained parent uses different class definitions or mappings. "
            "Choose a checkpoint with the exact same class version or start from official weights."
        )
