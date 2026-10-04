"""Offline adapter contracts: synthetic outputs are not executions or quality evidence."""

from copy import deepcopy

import pytest

from iris import preannotation_contracts as contracts
from iris.dataset_manifest import taxonomy_mappings
from iris.models import get_spec
from iris.store import DEFAULT_PROJECT_ID, Store
from iris.taxonomies import TAXONOMY, publish_taxonomy

MODEL = "ssdlite320_mobilenet_v3_large"
CLASSES = [
    {"id": "all", "name": "Vehicle", "definition": "Fixture vehicle", "coco_id": 3},
    {"id": "helmet", "name": "Helmet", "definition": "Fixture helmet with no COCO mapping"},
    {"id": "bottle", "name": "Bottle", "definition": "Fixture bottle", "coco_id": 44},
]


@pytest.fixture
def workspace(tmp_path):
    store = Store(tmp_path / "workspace")
    taxonomy = publish_taxonomy(
        store,
        DEFAULT_PROJECT_ID,
        expected_taxonomy_id=TAXONOMY["id"],
        classes=CLASSES,
    )
    return store, taxonomy


def trained(taxonomy):
    internal, output = taxonomy_mappings(taxonomy)
    return {
        "id": "fixture-checkpoint",
        "origin": "trained",
        "taxonomy": deepcopy(taxonomy),
        "taxonomy_id": taxonomy["id"],
        "class_mapping": internal,
        "output_class_mapping": output,
    }


def detection(label_id=3, label="car", score=0.8, **extra):
    return {"label_id": label_id, "label": label, "box": [1, 2, 40, 60], "score": score, **extra}


def payload(taxonomy, *, normalized=False, source=None):
    return {
        "protocol": contracts.OUTPUT_PROTOCOL,
        "taxonomy_id": taxonomy["id"],
        "coordinates": {
            "format": "xyxy",
            "space": "normalized" if normalized else "original_pixels",
            "image_size": [200, 100],
            "to_original": {"scale": [200, 100] if normalized else [1, 1], "offset": [0, 0]},
        },
        "proposals": [
            {
                "id": "fixture-box",
                "label": "helmet",
                "box": [0.1, 0.2, 0.5, 0.8] if normalized else [20, 20, 100, 80],
                "uncertain": True,
                "reason": "Fixture only: inspect the box manually",
                "source": source or {"fixture": True},
            }
        ],
    }


def test_capabilities_describe_only_available_provider_operations():
    detector = contracts.provider_capabilities("local_detector")
    assert detector["execution"] == "local" and detector["creates_boxes"]
    assert detector["operations"] == ["propose_boxes"] and detector["custom_classes"]
    assert not detector["requires_candidates"] and not detector["creates_class_definitions"]
    for provider in ("ollama", "alibaba"):
        reviewer = contracts.provider_capabilities(provider)
        assert reviewer["operations"] == ["review_candidates"]
        assert reviewer["requires_candidates"] and reviewer["max_candidates"] == 8
        assert reviewer["taxonomy_ids"] == [TAXONOMY["id"]]
        assert not reviewer["creates_boxes"] and not reviewer["custom_classes"]
        assert not reviewer["scores_comparable"] and not reviewer["automatic_validation"]
        assert reviewer["execution"] == ("external" if provider == "alibaba" else "local")
    for future in ("sam", "multimodal", "combined", "fixture"):
        with pytest.raises(ValueError, match="implemented"):
            contracts.provider_capabilities(future)
    detector["operations"].append("invented")
    assert contracts.provider_capabilities("local_detector")["operations"] == ["propose_boxes"]


def test_official_mapping_reports_partial_and_zero_coverage_without_inventing_classes(workspace):
    store, taxonomy = workspace
    contract = contracts.build_contract(store, MODEL, taxonomy)
    assert contract["label_mapping"] == {"3": "all", "44": "bottle"}
    assert contract["supported_class_ids"] == ["all", "bottle"]
    assert contract["unsupported_class_ids"] == ["helmet"]
    assert contract["coverage_complete"] is False
    assert "helmet" in contract["warnings"][-1]
    assert "comparable" in contract["warnings"][0]
    custom_only = {**taxonomy, "classes": [taxonomy["classes"][1]]}
    empty = contracts.build_contract(store, MODEL, custom_only)
    assert empty["label_mapping"] == {} and empty["supported_class_ids"] == []
    assert empty["unsupported_class_ids"] == ["helmet"]
    assert contracts.build_contract(store, MODEL, TAXONOMY)["label_mapping"] == {
        "1": "person",
        "3": "car",
    }


def test_trained_detector_requires_exact_frozen_semantics(workspace, monkeypatch):
    store, taxonomy = workspace
    spec = trained(taxonomy)
    monkeypatch.setattr(contracts, "get_spec", lambda *args: deepcopy(spec))
    frozen = contracts.build_contract(store, spec["id"], taxonomy)
    assert frozen["label_mapping"] == {"1": "all", "2": "helmet", "3": "bottle"}
    assert frozen["coverage_complete"] is True
    changed = deepcopy(taxonomy)
    changed["classes"][0]["definition"] = "Different fixture meaning"
    with pytest.raises(ValueError, match="exact same"):
        contracts.build_contract(store, spec["id"], changed)
    normalized = contracts.normalize_candidates(
        [detection(1, "all", taxonomy_id=taxonomy["id"])], frozen, 0.5, width=80, height=80
    )
    assert normalized["proposals"][0]["label"] == "all"
    with pytest.raises(ValueError, match="class definitions"):
        contracts.normalize_candidates([detection(1, "person")], frozen, 0.5, width=80, height=80)


@pytest.mark.parametrize("mutation", ["mapping", "coverage", "classes", "namespace"])
def test_saved_contract_cannot_change_mapping_coverage_or_semantics(workspace, mutation):
    store, taxonomy = workspace
    frozen = contracts.build_contract(store, MODEL, taxonomy)
    if mutation == "mapping":
        frozen["label_mapping"]["3"] = "helmet"
    elif mutation == "coverage":
        frozen["unsupported_class_ids"] = []
    elif mutation == "classes":
        frozen["taxonomy"]["classes"][0]["coco_id"] = 1
    else:
        frozen["source_contract"]["taxonomy_id"] = taxonomy["id"]
    with pytest.raises(ValueError):
        contracts.validate_contract(frozen)


def test_normalized_detector_proposals_keep_raw_indices_and_do_not_read_live_registry(
    workspace, monkeypatch
):
    store, taxonomy = workspace
    frozen = contracts.build_contract(store, MODEL, taxonomy)
    raw = [detection(1, "person"), detection(), detection(44, "bottle", score=0.2)]
    before = deepcopy(raw)
    monkeypatch.setattr(contracts, "get_spec", lambda *args: pytest.fail("Live registry read"))
    result = contracts.normalize_candidates(raw, frozen, 0.5, width=80, height=80)
    assert raw == before and result["raw_output"] == before
    assert result["unmapped_count"] == result["filtered_count"] == 1
    (proposal,) = result["proposals"]
    assert (proposal["source_index"], proposal["label"], proposal["original_label_id"]) == (
        1,
        "all",
        3,
    )
    assert proposal["source"]["raw_detection"] == raw[1]
    assert proposal["geometry"]["raw_box"] == raw[1]["box"]
    result["raw_output"][1]["label"] = "edited"
    assert raw == before


@pytest.mark.parametrize(
    "invalid",
    [
        {"box": [0, 0, 81, 20]},
        {"box": [0, 0, 0, 20]},
        {"box": [0, 0, float("nan"), 20]},
        {"score": float("inf")},
        {"score": 10**500},
        {"score": True},
        {"label_id": True},
        {"label": "wrong namespace"},
        {"taxonomy_id": "wrong taxonomy"},
        {"ignored": True},
        {"native_extra": float("nan")},
    ],
)
def test_invalid_detector_outputs_are_rejected_even_when_unmapped_or_below_threshold(
    workspace, invalid
):
    store, taxonomy = workspace
    frozen = contracts.build_contract(store, MODEL, taxonomy)
    with pytest.raises(ValueError):
        contracts.normalize_candidates(
            [{**detection(1, "person", score=0.01), **invalid}],
            frozen,
            0.9,
            width=80,
            height=80,
        )


@pytest.mark.parametrize("route", ["multimodal", "segmentation", "combined"])
@pytest.mark.parametrize("normalized", [False, True])
def test_future_adapter_fixtures_share_box_contract_without_claiming_live_integration(
    workspace, route, normalized
):
    _, taxonomy = workspace
    source = {"fixture": True, "route": route}
    if route != "multimodal":
        source.update(mask_id="fixture-mask", geometry_derivation="fixture mask bounds")
    if route == "combined":
        source["steps"] = ["fixture instruction", "fixture geometry", "fixture review"]
    raw = payload(taxonomy, normalized=normalized, source=source)
    before = deepcopy(raw)
    result = contracts.normalize_output(raw, taxonomy, width=200, height=100)
    (proposal,) = result["proposals"]
    assert proposal["box"] == [20, 20, 100, 80]
    assert proposal["score"] is None and proposal["uncertain"] is True
    assert (
        proposal["source"] == source
        and proposal["geometry"]["raw_box"] == raw["proposals"][0]["box"]
    )
    assert result["raw_output"] == raw == before
    assert contracts.OMISSION_WARNING in result["warnings"]


@pytest.mark.parametrize(
    "mutation", ["class", "transform", "size", "bool_scale", "duplicate", "space"]
)
def test_generic_contract_rejects_unknown_classes_and_ambiguous_geometry(workspace, mutation):
    _, taxonomy = workspace
    raw = payload(taxonomy)
    if mutation == "class":
        raw["proposals"][0]["label"] = "invented-new-class"
    elif mutation == "transform":
        del raw["coordinates"]["to_original"]
    elif mutation == "size":
        raw["coordinates"]["image_size"] = [100, 200]
    elif mutation == "bool_scale":
        raw["coordinates"]["to_original"]["scale"] = [True, True]
    elif mutation == "space":
        raw["coordinates"]["space"] = "crop_pixels"
    else:
        raw["proposals"].append(deepcopy(raw["proposals"][0]))
    with pytest.raises(ValueError):
        contracts.normalize_output(raw, taxonomy, width=200, height=100)


def test_empty_and_bounded_outputs_never_imply_human_negative(workspace):
    store, taxonomy = workspace
    frozen = contracts.build_contract(store, MODEL, taxonomy)
    result = contracts.normalize_candidates([], frozen, 0.5, width=80, height=80)
    assert result["proposals"] == [] and "negative" not in result
    assert contracts.OMISSION_WARNING in result["warnings"]
    with pytest.raises(ValueError, match="bounded"):
        contracts.normalize_candidates([detection()] * 301, frozen, 0.5, width=80, height=80)
    oversized = payload(taxonomy)
    oversized["proposals"][0]["source"]["fixture_blob"] = "x" * contracts.MAX_OUTPUT_BYTES
    with pytest.raises(ValueError, match="1 MiB"):
        contracts.normalize_output(oversized, taxonomy, width=200, height=100)


def test_build_contract_uses_metadata_only_without_model_runtime(workspace, monkeypatch):
    store, taxonomy = workspace
    spec = get_spec(MODEL)
    seen = []

    def metadata_only(model_id, root):
        seen.append((model_id, root))
        return deepcopy(spec)

    monkeypatch.setattr(contracts, "get_spec", metadata_only)
    assert contracts.build_contract(store, MODEL, taxonomy)["model_id"] == MODEL
    assert seen == [(MODEL, store.root)]
