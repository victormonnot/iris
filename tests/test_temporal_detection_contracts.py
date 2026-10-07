"""Synthetic native outputs validate cache identity; no inference is performed."""

from copy import deepcopy

import pytest
from test_temporal_contracts import sequence, set_path

from iris import models, temporal_detector
from iris import temporal_detection_contracts as contracts
from iris.dataset_manifest import taxonomy_mappings
from iris.model_taxonomy import class_contract
from iris.prediction_taxonomy import COCO_TAXONOMY
from iris.temporal_contracts import sequence_hash
from iris.tiling import tile_boxes, validate_tiling_config
from iris.training_architectures import FRCNN, SSDLITE, YOLOX


def cache_config(manifest, *, tiled=False, architecture=SSDLITE):
    mode = "tiled" if tiled else "full"
    spec = models.get_spec(architecture)
    filtering, preprocessing = temporal_detector._native_profile(architecture)
    inference = {"mode": mode}
    if tiled:
        inference.update(algorithm="iris-tiling-v1", tiling=validate_tiling_config(tile_size=128))
    detector = {
        "schema": temporal_detector.SCHEMA,
        "model_id": architecture,
        "architecture": architecture,
        "origin": "official",
        "weight_sha256": "b" * 64,
        "classes": deepcopy(spec["classes"]),
        "class_contract": {"taxonomy_id": COCO_TAXONOMY},
        "device": "cpu",
        "min_score": 0.001,
        "inference": inference,
        "native_filtering": filtering,
        "preprocessing": preprocessing,
        "output_policy": temporal_detector._output_policy(architecture),
        "runtime": {
            "adapter_revision": temporal_detector.ADAPTER_REVISION,
            "python": "3.12.0-synthetic",
            "packages": {
                name: "historical-synthetic-version" for name in temporal_detector.PACKAGES
            },
            "source_sha256": {
                name: "c" * 64 for name in temporal_detector._sources(architecture, mode)
            },
        },
    }
    return {
        "schema": contracts.CACHE_SCHEMA,
        "sequence_id": manifest["id"],
        "sequence_sha256": sequence_hash(manifest),
        "detector": detector,
        "generation": None,
        "frame_ids": [frame["frame_id"] for frame in manifest["frames"]],
    }


def frame_payload(config, manifest, *, position=0):
    frame = manifest["frames"][position]
    size = [frame["width"], frame["height"]]
    inference = config["detector"]["inference"]
    tile_count = len(tile_boxes(*size, inference["tiling"])) if inference["mode"] == "tiled" else 0
    return {
        "schema": contracts.FRAME_SCHEMA,
        "cache_fingerprint": contracts.cache_fingerprint(config),
        "frame_id": frame["frame_id"],
        "frame_index": frame["frame_index"],
        "timestamp_seconds": frame["timestamp_seconds"],
        "frame_sha256": frame["sha256"],
        "file_sha256": frame["file_sha256"],
        "input_size": size,
        "detections": [
            {
                "detection_index": 0,
                "label_id": 1,
                "label": "person",
                "score": 0.0011,
                "box": [0, 0, 20, 30],
            },
            {
                "detection_index": 3,
                "label_id": 3,
                "label": "car",
                "score": 0.8,
                "box": [20, 30, 40, 60],
            },
        ],
        "native_detection_count": 5,
        "execution_signature_sha256": "d" * 64,
        "timing": {name: 1 for name in contracts.TIMING_FIELDS},
        "work": {"forward_passes": tile_count or 1, "tile_count": tile_count},
    }


def test_cache_binds_complete_ordered_sequence_and_historical_runtime_without_loading_models(
    monkeypatch,
):
    manifest = sequence(sparse=True)
    config = cache_config(manifest)

    def forbidden(*args, **kwargs):
        raise AssertionError("Pure contract validation cannot inspect the execution runtime")

    monkeypatch.setattr(temporal_detector, "_runtime", forbidden)
    result = contracts.validate_cache_config(config, manifest)
    assert result == config and result is not config
    result["detector"]["classes"][0]["name"] = "mutated"
    result["frame_ids"].reverse()
    assert config["detector"]["classes"][0]["name"] == "person"
    assert config["frame_ids"] == ["frame-0", "frame-2"]


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("schema",), "unknown-schema"),
        (("sequence_id",), "different-sequence"),
        (("sequence_sha256",), "f" * 64),
        (("sequence_sha256",), "not-a-hash"),
        (("generation",), "A" * 32),
        (("generation",), "a" * 31),
        (("generation",), False),
        (("frame_ids",), ["frame-0", "frame-2"]),
        (("frame_ids",), ["frame-2", "frame-1", "frame-0"]),
        (("frame_ids",), ["frame-0", "frame-0", "frame-2"]),
        (("frame_ids",), [True, "frame-1", "frame-2"]),
        (("detector", "min_score"), 0),
        (("detector", "min_score"), float("nan")),
        (("detector", "classes", 0, "name"), "invented class"),
    ],
)
def test_cache_rejects_wrong_sequence_incomplete_order_and_ambiguous_recipe(path, replacement):
    manifest = sequence()
    config = cache_config(manifest)
    set_path(config, path, replacement)
    with pytest.raises(ValueError):
        contracts.validate_cache_config(config, manifest)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("generation",), "0" * 32),
        (("detector", "min_score"), 0.5),
        (("detector", "device"), "cuda"),
        (("detector", "weight_sha256"), "f" * 64),
        (("detector", "runtime", "packages", "numpy"), "different-runtime"),
    ],
)
def test_cache_fingerprint_changes_for_execution_affecting_parameters_and_explicit_reruns(
    path, replacement
):
    config = cache_config(sequence())
    changed = deepcopy(config)
    set_path(changed, path, replacement)
    assert contracts.cache_fingerprint(changed) != contracts.cache_fingerprint(config)
    reordered = dict(reversed(list(config.items())))
    assert contracts.cache_fingerprint(reordered) == contracts.cache_fingerprint(config)


def test_frame_retains_native_output_indices_without_confusing_them_with_track_identities():
    manifest = sequence(sparse=True)
    config = cache_config(manifest)
    original = frame_payload(config, manifest)
    result = contracts.validate_frame_payload(original, config, manifest)
    assert result == original
    assert [d["detection_index"] for d in result["detections"]] == [0, 3]
    assert result["native_detection_count"] == 5
    assert all(type(value) is float for value in result["detections"][0]["box"])
    assert all(type(value) is float for value in result["timing"].values())
    result["detections"][0]["box"][0] = 3
    result["timing"]["decode_ms"] = 100
    result["work"]["forward_passes"] = 10
    assert original["detections"][0]["box"][0] == 0
    assert original["timing"]["decode_ms"] == 1
    assert original["work"]["forward_passes"] == 1


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("schema",), "iris-tracker-frame-v1"),
        (("cache_fingerprint",), "f" * 64),
        (("frame_id",), "foreign-frame"),
        (("frame_index",), True),
        (("frame_index",), 1),
        (("frame_sha256",), "f" * 64),
        (("file_sha256",), "f" * 64),
        (("timestamp_seconds",), True),
        (("timestamp_seconds",), 0.01),
        (("timestamp_seconds",), None),
        (("input_size",), [100, True]),
        (("input_size",), [100, 81]),
        (("input_size",), (100, 80)),
        (("native_detection_count",), True),
        (("native_detection_count",), -1),
        (("native_detection_count",), 101),
        (("native_detection_count",), 1),
        (("execution_signature_sha256",), "not-a-hash"),
        (("detections", 0, "detection_index"), True),
        (("detections", 0, "detection_index"), -1),
        (("detections", 1, "detection_index"), 5),
        (("detections", 1, "detection_index"), 0),
        (("detections", 0, "label_id"), True),
        (("detections", 0, "label_id"), 0),
        (("detections", 0, "label_id"), 999),
        (("detections", 0, "label"), "car"),
        (("detections", 0, "score"), True),
        (("detections", 0, "score"), 0.0009),
        (("detections", 0, "score"), 1.001),
        (("detections", 0, "score"), float("nan")),
        (("detections", 0, "box"), [0, 0, 0, 10]),
        (("detections", 0, "box"), [-1, 0, 10, 10]),
        (("detections", 0, "box"), [0, 0, 101, 10]),
        (("detections", 0, "box"), [0, 0, 10, float("inf")]),
        (("detections", 0, "box"), [0, True, 10, 10]),
        (("timing", "total_ms"), -1),
        (("timing", "filter_ms"), float("inf")),
        (("timing", "inference_ms"), True),
        (("work", "forward_passes"), True),
        (("work", "forward_passes"), 2),
        (("work", "tile_count"), 1),
    ],
)
def test_frame_rejects_changed_inputs_fabricated_identity_class_score_and_work(path, replacement):
    manifest = sequence()
    config = cache_config(manifest)
    payload = frame_payload(config, manifest)
    set_path(payload, path, replacement)
    with pytest.raises(ValueError):
        contracts.validate_frame_payload(payload, config, manifest)


def test_unknown_timestamps_remain_null_and_provided_timestamps_remain_exact():
    manifest = sequence(basis="unknown")
    config = cache_config(manifest)
    payload = frame_payload(config, manifest)
    assert contracts.validate_frame_payload(payload, config, manifest)["timestamp_seconds"] is None
    payload["timestamp_seconds"] = 0
    with pytest.raises(ValueError, match="unknown source clock"):
        contracts.validate_frame_payload(payload, config, manifest)
    manifest = sequence(basis="provided")
    manifest["frames"][1]["timestamp_seconds"] = 0.075
    config = cache_config(manifest)
    payload = frame_payload(config, manifest, position=1)
    assert contracts.validate_frame_payload(payload, config, manifest)["timestamp_seconds"] == 0.075
    payload["timestamp_seconds"] = 0.1
    with pytest.raises(ValueError, match="exact source time"):
        contracts.validate_frame_payload(payload, config, manifest)


def test_empty_cache_output_records_native_count_without_claiming_a_negative_annotation():
    manifest = sequence()
    config = cache_config(manifest)
    payload = frame_payload(config, manifest)
    payload["detections"] = []
    for count in [0, 5, 100]:
        payload["native_detection_count"] = count
        assert contracts.validate_frame_payload(payload, config, manifest)["detections"] == []


def test_cache_keeps_all_detector_classes_even_outside_project_annotation_taxonomy():
    manifest = sequence()
    config = cache_config(manifest)
    payload = frame_payload(config, manifest)
    payload["detections"][0].update(label_id=6, label="bus")
    assert "bus" not in {entry["id"] for entry in manifest["taxonomy"]["classes"]}
    result = contracts.validate_frame_payload(payload, config, manifest)
    assert result["detections"][0]["label"] == "bus"


@pytest.mark.parametrize("size", [(100, 80), (300, 200)])
def test_tiled_work_matches_real_tile_layout_even_when_only_one_tile(size):
    manifest = sequence()
    for frame in manifest["frames"]:
        frame["width"], frame["height"] = size
    config = cache_config(manifest, tiled=True)
    payload = frame_payload(config, manifest)
    payload["native_detection_count"] = 300
    expected = len(tile_boxes(*size, config["detector"]["inference"]["tiling"]))
    assert contracts.validate_frame_payload(payload, config, manifest)["work"] == {
        "forward_passes": expected,
        "tile_count": expected,
    }
    payload["work"]["tile_count"] -= 1
    with pytest.raises(ValueError, match="work must match"):
        contracts.validate_frame_payload(payload, config, manifest)
    payload["work"]["tile_count"] = expected
    payload["native_detection_count"] = 301
    with pytest.raises(ValueError, match="Native detection count"):
        contracts.validate_frame_payload(payload, config, manifest)


@pytest.mark.parametrize("level", ["cache", "payload", "detection", "work", "timing"])
def test_unknown_fields_cannot_introduce_track_predictions_or_change_persisted_semantics(level):
    manifest = sequence()
    config = cache_config(manifest)
    payload = frame_payload(config, manifest)
    targets = {
        "cache": config,
        "payload": payload,
        "detection": payload["detections"][0],
        "work": payload["work"],
        "timing": payload["timing"],
    }
    targets[level]["tracker_id"] = 123
    with pytest.raises(ValueError, match="supported fields"):
        contracts.validate_frame_payload(payload, config, manifest)


def test_output_hash_is_canonical_and_changes_with_observations_or_measurements():
    manifest = sequence()
    config = cache_config(manifest)
    payload = contracts.validate_frame_payload(frame_payload(config, manifest), config, manifest)
    original_hash = contracts.payload_hash(payload)
    assert contracts.payload_hash(dict(reversed(list(payload.items())))) == original_hash
    payload["detections"][0]["score"] = 0.25
    assert contracts.payload_hash(payload) != original_hash
    payload["timing"]["total_ms"] = float("nan")
    with pytest.raises(ValueError, match="finite UTF-8 JSON"):
        contracts.payload_hash(payload)


@pytest.mark.parametrize("tiled", [False, True])
def test_reused_validation_context_matches_standalone_results_without_revalidating_sources(
    monkeypatch, tiled
):
    manifest = sequence()
    config = cache_config(manifest, tiled=tiled)
    payloads = [frame_payload(config, manifest, position=index) for index in range(3)]
    expected = [contracts.validate_frame_payload(payload, config, manifest) for payload in payloads]
    context = contracts.frame_validation_context(config, manifest)

    def forbidden(*args, **kwargs):
        raise AssertionError("A prepared frame loop must not revalidate/hash all source inputs")

    for name in (
        "validate_sequence_manifest",
        "validate_cache_config",
        "cache_fingerprint",
        "tile_boxes",
    ):
        monkeypatch.setattr(contracts, name, forbidden)
    assert [
        contracts.validate_frame_payload(payload, config, manifest, context=context)
        for payload in payloads
    ] == expected
    payloads[0]["frame_index"] = 99
    with pytest.raises(ValueError, match="frame index"):
        contracts.validate_frame_payload(payloads[0], config, manifest, context=context)


def test_validation_context_is_internal_and_cannot_be_reused_for_another_input_pair():
    manifest = sequence()
    config = cache_config(manifest)
    payload = frame_payload(config, manifest)
    context = contracts.frame_validation_context(config, manifest)
    with pytest.raises(ValueError, match="context must belong"):
        contracts.validate_frame_payload(payload, deepcopy(config), manifest, context=context)
    with pytest.raises(ValueError, match="context must belong"):
        contracts.validate_frame_payload(payload, config, deepcopy(manifest), context=context)
    with pytest.raises(ValueError, match="context must belong"):
        contracts.validate_frame_payload(payload, config, manifest, context={})
    with pytest.raises(TypeError):
        context.frames["frame-0"]["width"] = 1
    with pytest.raises(TypeError):
        context.classes[1] = "invented label"


def trained_cache_config(manifest):
    config = cache_config(manifest)
    taxonomy = manifest["taxonomy"]
    internal, output = taxonomy_mappings(taxonomy)
    detector = config["detector"]
    detector.update(
        origin="trained",
        model_id="trained_fixture",
        class_contract=class_contract(
            {
                "taxonomy": taxonomy,
                "class_mapping": internal,
                "output_class_mapping": output,
            }
        ),
        classes=[{"id": output[item["id"]], "name": item["id"]} for item in taxonomy["classes"]],
    )
    return config


def test_trained_class_semantics_must_match_sequence_even_when_taxonomy_id_is_unchanged():
    manifest = sequence()
    builtin_id = manifest["taxonomy"]["id"]
    manifest["taxonomy"].update(
        id="taxonomy-" + "a" * 32,
        version=2,
        parent_id=builtin_id,
        created_at="2026-10-07T15:00:00Z",
    )
    config = trained_cache_config(manifest)
    assert contracts.validate_cache_config(config, manifest) == config
    changed = deepcopy(manifest)
    changed["taxonomy"]["classes"][0]["definition"] = "Different frozen person semantics."
    config["sequence_sha256"] = sequence_hash(changed)
    # Both snapshots independently validate, but numeric labels do not establish
    # equal class meaning. Archives must reject this relationship as services do.
    assert temporal_detector.validate_detector_config(config["detector"]) == config["detector"]
    with pytest.raises(ValueError, match="taxonomy exactly"):
        contracts.validate_cache_config(config, changed)
    with pytest.raises(ValueError, match="taxonomy exactly"):
        contracts.frame_validation_context(config, changed)


@pytest.mark.parametrize("architecture", [SSDLITE, FRCNN, YOLOX])
@pytest.mark.parametrize("tiled", [False, True])
def test_saved_scores_retain_native_strict_or_inclusive_threshold_semantics(architecture, tiled):
    manifest = sequence()
    config = cache_config(manifest, architecture=architecture, tiled=tiled)
    payload = frame_payload(config, manifest)
    payload["detections"][0]["score"] = config["detector"]["native_filtering"]["score_threshold"]
    if architecture == YOLOX:
        assert contracts.validate_frame_payload(payload, config, manifest) == payload
    else:
        with pytest.raises(ValueError, match="native threshold comparison"):
            contracts.validate_frame_payload(payload, config, manifest)
    config["detector"]["min_score"] = 0.2
    payload["cache_fingerprint"] = contracts.cache_fingerprint(config)
    payload["detections"][0]["score"] = 0.2
    assert contracts.validate_frame_payload(payload, config, manifest) == payload
