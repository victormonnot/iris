"""SAM protocol fixtures: no weights, GPU, installation or real model execution."""

import hashlib
import math
import os
import sys
from copy import deepcopy
from types import SimpleNamespace

import pytest
from PIL import Image

from iris import sam_provider as provider
from iris.taxonomies import TAXONOMY


def native(config=None):
    config = config or provider.freeze_config(TAXONOMY)
    return {
        "protocol": provider.RAW_PROTOCOL,
        "image": {"width": 200, "height": 100},
        "coordinates": {
            "format": "xyxy",
            "space": "normalized",
            "image_size": [200, 100],
            "to_original": {"scale": [200, 100], "offset": [0, 0]},
        },
        "prompts": [
            {
                **prompt,
                "boxes": [[0.1, 0.2, 0.7, 0.9]],
                "scores": [0.8],
                "native_indices": [7],
                "error": None,
            }
            for prompt in config["prompts"]
        ],
        "complete": True,
        "metadata": {"runtime": "fixture"},
    }


@pytest.fixture
def runtime(monkeypatch):
    state = {"loads": [], "predicts": [], "closes": 0, "error": None, "ready": True}

    def status(config=None, *, force=False):
        return {
            "ready": state["ready"],
            "status": "ready" if state["ready"] else "missing_runtime",
            "reason": None if state["ready"] else "Fixture runtime unavailable",
            "identity": {"code_revision": provider.CODE_REVISION, "fixture": True},
        }

    class Runtime:
        def __init__(self, config, path, *, cancelled):
            state["loads"].append({"config": config, "path": path})
            self.metadata = {"runtime_identity": status()["identity"], "load_ms": 12}
            if state.get("identity_absent"):
                self.metadata.pop("runtime_identity")

        def predict(self, image, *, class_prompts, threshold, cancelled):
            state["predicts"].append(
                {"image": image, "prompts": class_prompts, "threshold": threshold}
            )
            if state["error"] is not None:
                raise state["error"]
            return native()

        def close(self):
            state["closes"] += 1

    monkeypatch.setitem(
        sys.modules, "iris.sam_runtime", SimpleNamespace(runtime_status=status, SamRuntime=Runtime)
    )
    return state


@pytest.fixture
def weights(tmp_path, monkeypatch):
    data = b"small offline SAM checkpoint identity fixture"
    monkeypatch.setattr(provider, "CHECKPOINT_SIZE", len(data))
    monkeypatch.setattr(provider, "CHECKPOINT_SHA256", hashlib.sha256(data).hexdigest())
    provider._WEIGHT_CACHE.clear()
    path = tmp_path / provider.CHECKPOINT_PATH
    path.parent.mkdir(parents=True)
    path.write_bytes(data)
    return path


def test_configuration_is_pure_and_preserves_class_definitions(monkeypatch):
    monkeypatch.delenv("IRIS_SAM_PYTHON", raising=False)
    config = provider.freeze_config(TAXONOMY)
    assert config["taxonomy"] == TAXONOMY
    assert config["prompts"] == [
        {"class_id": item["id"], "text": item["name"]} for item in TAXONOMY["classes"]
    ]
    assert config["local_only"] and not config["external"]
    assert config["settings"] == {"threshold": 0.5, "device": "cuda", "precision": "bfloat16"}
    assert config["limits"]["max_prompt_tokens"] == 30
    assert provider.validate_frozen_config(config) == config
    config["taxonomy"]["classes"][0]["name"] = "Changed independently"
    assert TAXONOMY["classes"][0]["name"] != "Changed independently"
    with pytest.raises(ValueError):
        provider.validate_frozen_config(config)


@pytest.mark.parametrize(
    "options",
    [
        {"model": "sam3.1"},
        {"device": "cpu"},
        {"threshold": True},
        {"threshold": math.nan},
        {"threshold": math.inf},
        {"threshold": -0.01},
        {"threshold": 1.01},
        {"class_prompts": {}},
        {"class_prompts": {"person": "person", "car": "\ncar"}},
        {"class_prompts": {"person": " ", "car": "car"}},
        {"class_prompts": {"person": "x" * 121, "car": "car"}},
        {
            "class_prompts": [
                {"class_id": "car", "text": "car"},
                {"class_id": "person", "text": "person"},
            ]
        },
    ],
)
def test_invalid_profiles_and_phrases_are_rejected(options):
    with pytest.raises(ValueError):
        provider.freeze_config(TAXONOMY, **options)


@pytest.mark.parametrize(
    "field", ["code_revision", "weights", "source", "runtime_required", "settings", "prompts"]
)
def test_historical_validation_rejects_changed_frozen_identity(field):
    config = provider.freeze_config(TAXONOMY)
    config[field] = "forged"
    with pytest.raises(ValueError):
        provider.validate_frozen_config(config)


def test_custom_phrases_are_not_definitions_and_classes_can_overlap():
    taxonomy = deepcopy(TAXONOMY)
    taxonomy.update(
        id="taxonomy-" + "a" * 32, version=2, parent_id=TAXONOMY["id"], created_at="2026-10-04"
    )
    taxonomy["classes"] = [
        {
            "id": "helmet",
            "name": "Casque",
            "definition": "Human review definition, never a SAM prompt",
        },
        {"id": "hat", "name": "Chapeau", "definition": "Another frozen human class definition"},
    ]
    config = provider.freeze_config(
        taxonomy, class_prompts={"helmet": "safety helmet", "hat": "hat"}
    )
    raw = native(config)
    result = provider.normalize_response(raw, config, width=200, height=100)
    assert [item["label"] for item in result["proposals"]] == ["helmet", "hat"]
    assert result["proposals"][0]["box"] == result["proposals"][1]["box"]
    assert result["proposals"][0]["source"]["prompt"] == "safety helmet"
    assert result["taxonomy_id"] == taxonomy["id"]


def test_native_boxes_are_scaled_clipped_and_preserved_without_mask_derivation():
    config = provider.freeze_config(TAXONOMY)
    raw = native()
    raw["prompts"][0]["boxes"] = [[-0.05, 0.1, 1.1, 0.8]]
    before = deepcopy(raw)
    result = provider.normalize_response(raw, config, width=200, height=100)
    box = result["proposals"][0]
    assert box["box"] == [0, 10, 200, 80]
    assert box["score"] == 0.8
    assert box["source"]["native_box"] == [-0.05, 0.1, 1.1, 0.8]
    assert box["source"]["native_index"] == 7
    assert box["source"]["clipped"] is True
    assert box["source"]["box_origin"] == "native_detector"
    assert result["clipped_count"] == 1
    assert raw == before


def test_strict_threshold_and_valid_empty_do_not_claim_negative_validation():
    raw = native()
    for row in raw["prompts"]:
        row["scores"] = [0.5]
    result = provider.normalize_response(
        raw, provider.freeze_config(TAXONOMY), width=200, height=100
    )
    assert result["proposals"] == []
    assert result["filtered_count"] == 2
    assert any("do not establish" in text for text in result["warnings"])
    for row in raw["prompts"]:
        row.update(boxes=[], scores=[], native_indices=[])
    assert (
        provider.normalize_response(raw, provider.freeze_config(TAXONOMY), width=200, height=100)[
            "proposals"
        ]
        == []
    )


@pytest.mark.parametrize(
    "corruption",
    [
        "partial",
        "error",
        "empty_error",
        "missing",
        "class",
        "phrase",
        "transform",
        "dimensions",
        "nan",
        "score",
        "outside",
        "reversed",
        "indices",
        "duplicate_indices",
        "oversized",
    ],
)
def test_invalid_and_partial_native_evidence_is_retained_without_proposals(corruption):
    raw = native()
    row = raw["prompts"][0]
    if corruption == "partial":
        raw["complete"] = False
    elif corruption in {"error", "empty_error"}:
        row["error"] = "failed" if corruption == "error" else ""
    elif corruption == "missing":
        raw["prompts"].pop()
    elif corruption == "class":
        row["class_id"] = "car"
    elif corruption == "phrase":
        row["text"] = "changed phrase"
    elif corruption == "transform":
        raw["coordinates"]["to_original"]["scale"] = [100, 200]
    elif corruption == "dimensions":
        raw["image"]["width"] = 201
    elif corruption == "nan":
        row["boxes"][0][0] = math.nan
    elif corruption == "score":
        row["scores"] = [True]
    elif corruption == "outside":
        row["boxes"] = [[-2, 0, -1, 1]]
    elif corruption == "reversed":
        row["boxes"] = [[0.9, 0, 0.1, 1]]
        row["scores"] = [0.01]  # Invalid geometry still fails below the cutoff.
    elif corruption == "indices":
        row["native_indices"] = [True]
    elif corruption == "duplicate_indices":
        row["boxes"] *= 2
        row["scores"] *= 2
        row["native_indices"] *= 2
    elif corruption == "oversized":
        raw["metadata"]["oversized"] = "x" * provider.MAX_RESPONSE_BYTES
    with pytest.raises(provider.ProviderResponseError) as error:
        provider.normalize_response(raw, provider.freeze_config(TAXONOMY), width=200, height=100)
    assert error.value.raw_response is raw


def test_excess_proposals_fail_without_silent_truncation():
    raw = native()
    for row in raw["prompts"]:
        row.update(boxes=[[0, 0, 1, 1]] * 51, scores=[0.8] * 51, native_indices=list(range(51)))
    with pytest.raises(provider.ProviderResponseError, match="more than 100"):
        provider.normalize_response(raw, provider.freeze_config(TAXONOMY), width=200, height=100)
    assert sum(len(row["boxes"]) for row in raw["prompts"]) == 102


def test_valid_outside_native_query_below_cutoff_does_not_fail_useful_output():
    raw = native()
    raw["prompts"][0].update(boxes=[[-2, 0, -1, 1]], scores=[0.01])
    result = provider.normalize_response(
        raw, provider.freeze_config(TAXONOMY), width=200, height=100
    )
    assert len(result["proposals"]) == 1
    assert result["proposals"][0]["label"] == "car"
    assert result["filtered_count"] == 1


def test_offline_status_reports_missing_weights_and_runtime_without_loading(tmp_path, runtime):
    runtime["ready"] = False
    status = provider.provider_status(tmp_path)
    assert not status["ready"] and not status["weights"]["available"]
    assert status["runtime"]["status"] == "missing_runtime"
    assert status["devices"] == [
        {"id": "cuda", "available": False, "reason": "Fixture runtime unavailable"}
    ]
    assert runtime["loads"] == []


def test_checkpoint_hash_cache_does_not_hide_same_size_replacement(tmp_path, runtime, weights):
    assert provider.provider_status(tmp_path)["ready"]
    previous = weights.stat()
    weights.write_bytes(b"x" * previous.st_size)
    os.utime(weights, ns=(previous.st_atime_ns, previous.st_mtime_ns))
    status = provider.provider_status(tmp_path)
    assert status["status"] == "invalid_weights"
    assert "SHA-256" in status["reason"]


def test_weight_symlinks_cannot_escape_workspace(tmp_path, runtime, weights):
    outside = tmp_path.parent / (tmp_path.name + "-outside-fixture")
    outside.write_bytes(weights.read_bytes())
    weights.unlink()
    weights.symlink_to(outside)
    assert provider.provider_status(tmp_path)["status"] == "invalid_weights"


def test_adapter_passes_pixels_and_phrases_to_runtime_only(tmp_path, runtime, weights):
    config = provider.freeze_config(TAXONOMY)
    adapter = provider.Sam3Preannotator(tmp_path, config)
    image = Image.new("RGB", (200, 100))
    raw = adapter.predict(image)
    assert raw["complete"]
    assert len(runtime["loads"]) == 1
    assert runtime["predicts"] == [{"image": image, "prompts": config["prompts"], "threshold": 0.5}]
    assert adapter.metadata["load_ms"] == 12
    adapter.close()
    adapter.close()
    assert runtime["closes"] == 1
    with pytest.raises(provider.ProviderResponseError, match="closed"):
        adapter.predict(image)


def test_cancelled_initialization_never_loads_runtime(tmp_path, runtime):
    with pytest.raises(provider.ProviderResponseError, match="cancelled"):
        provider.Sam3Preannotator(
            tmp_path, provider.freeze_config(TAXONOMY), cancelled=lambda: True
        )
    assert runtime["loads"] == []


def test_loaded_runtime_must_report_its_actual_identity(tmp_path, runtime, weights):
    runtime["identity_absent"] = True
    with pytest.raises(provider.ProviderResponseError, match="actual identity"):
        provider.Sam3Preannotator(tmp_path, provider.freeze_config(TAXONOMY))
    assert runtime["closes"] == 1


def test_cancelled_prediction_and_runtime_partial_errors_keep_evidence(tmp_path, runtime, weights):
    adapter = provider.Sam3Preannotator(tmp_path, provider.freeze_config(TAXONOMY))
    image = Image.new("RGB", (200, 100))
    with pytest.raises(provider.ProviderResponseError, match="cancelled"):
        adapter.predict(image, cancelled=lambda: True)
    assert runtime["predicts"] == []
    partial = native()
    partial["complete"] = False
    runtime["error"] = provider.ProviderResponseError(
        "Fixture failure", raw_response=partial, metadata={"state": "failed"}
    )
    with pytest.raises(provider.ProviderResponseError, match="Fixture failure") as error:
        adapter.predict(image)
    assert error.value.raw_response is partial
    assert error.value.metadata == {"state": "failed"}
    adapter.close()
