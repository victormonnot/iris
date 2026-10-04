"""Frozen output identities for comparison and annotation consumers."""

from iris.models import COCO_CATEGORIES

COCO_TAXONOMY = "coco-2017-v1"


def output_contract(spec: dict) -> dict:
    if spec.get("origin") != "trained":
        return {"taxonomy_id": COCO_TAXONOMY}
    from iris.model_taxonomy import class_contract

    return class_contract(spec)


def validate_output_labels(detections: list[dict], contract: dict) -> None:
    if contract["taxonomy_id"] == COCO_TAXONOMY:
        labels = {i: name for i, name in enumerate(COCO_CATEGORIES) if i and name != "N/A"}
    else:
        labels = {
            identifier: label for label, identifier in contract["output_class_mapping"].items()
        }
    for detection in detections:
        identifier = detection.get("label_id")
        if (
            type(identifier) is not int
            or identifier not in labels
            or detection.get("label") != labels[identifier]
            or detection.get("ignored")
            or detection.get("taxonomy_id", contract["taxonomy_id"]) != contract["taxonomy_id"]
        ):
            raise ValueError("Detector output does not match its saved class definitions")


def annotation_mapping(comparison: dict, run: dict, taxonomy: dict) -> tuple[dict, str]:
    """Resolve a saved source without looking up today's model or project definitions."""
    config = comparison["config"]
    contracts = config.get("model_class_contracts")
    if contracts is None:
        if config.get("taxonomy") != COCO_TAXONOMY:
            raise ValueError("The saved prediction has incompatible class definitions")
        return {
            item["coco_id"]: item["id"] for item in taxonomy["classes"] if "coco_id" in item
        }, COCO_TAXONOMY
    contract = contracts.get(run["model_id"])
    if not isinstance(contract, dict) or run["metadata"].get("class_contract") != contract:
        raise ValueError("The saved prediction's class definitions are inconsistent")
    if contract.get("taxonomy_id") == COCO_TAXONOMY:
        return {
            item["coco_id"]: item["id"] for item in taxonomy["classes"] if "coco_id" in item
        }, COCO_TAXONOMY
    from iris.model_taxonomy import class_contract

    checked = class_contract(contract)
    if checked["taxonomy"] != taxonomy:
        raise ValueError("The trained checkpoint uses different class definitions from this frame")
    return {
        identifier: label for label, identifier in checked["output_class_mapping"].items()
    }, checked["taxonomy_id"]
