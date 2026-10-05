"""Enriched HTML uses frozen synthetic evidence, never target execution."""

import builtins
from copy import deepcopy
from html import escape

import pytest
from test_experiment_export import Document, render
from test_experiment_export import rendered as rendered

from iris import experiment_export as exporter


@pytest.fixture
def enriched(rendered):
    snapshot = rendered["report"]["snapshot"]
    snapshot["version"] = 2
    counts = {
        label: {
            "ground_truth_count": 0 if label == "car" else 2,
            "frame_count": 2,
            "runs": {
                "baseline": {"tp": 1, "fp": 2, "fn": 1, "error_frames": 2},
                "candidate": {"tp": 2, "fp": 1, "fn": 0, "error_frames": 1},
            },
            "changes": {"recovered": 1, "new_misses": 0, "fp_delta": -1},
        }
        for label in ("all", "person", "car")
    }
    snapshot["insights"] = {
        "protocol": "iris-experiment-insights-v1",
        "quality_delta": {"map": 0.1, "fp": -1},
        "scenes": [
            {
                "scene_group": "evening-loading-bay",
                "frame_count": 2,
                "negative_frame_count": 1,
                "counts": counts,
            }
        ],
        "frame_changes": {"frame-id": "improved"},
        "suggested_examples": [{"frame_id": "frame-id", "reason": "Recovered object"}],
        "sampling": {
            "continuous_inference": False,
            "video_sources": [
                {
                    "source_id": "video-id",
                    "filename": "loading-bay.mp4",
                    "frame_count": 2,
                    "timestamps_available": 1,
                    "first_timestamp_seconds": 1.25,
                    "last_timestamp_seconds": 1.25,
                    "timestamps_approximate": True,
                }
            ],
            "still_image_count": 0,
            "unknown_source_count": 0,
            "warning": "One sampled frame has no source timestamp.",
        },
        "timing": {
            "comparable": False,
            "reasons": ["Different saved execution devices."],
            "scope": "Only timing from the same recorded IRIS execution context is compared.",
        },
    }
    snapshot["deployments"] = {
        "protocol": "iris-experiment-deployments-v1",
        "measurements": [
            {
                "id": "measurement-id",
                "export_id": "export-id",
                "name": "Declared edge target",
                "model_id": "model_candidate",
                "lane_id": "candidate",
                "created_at": snapshot["captured_at"],
                "fingerprint": "d" * 64,
                "archive_sha256": "e" * 64,
                "model_sha256": "2" * 64,
                "profile": {
                    "id": "iris-torchvision-trained-native-v2",
                    "architecture": "fasterrcnn_mobilenet_v3_large_320_fpn",
                    "device": "cuda",
                    "precision": "float32",
                    "batch_size": 1,
                    "timing": {
                        "total_ms": (
                            "Preprocess + forward + postprocess; excludes decode/load/warmup."
                        ),
                        "decode_ms": "Read, checksum verification and decode.",
                        "load_ms": "Construction and checkpoint loading.",
                        "warmup_passes": 1,
                        "cuda_synchronization": "Before and after each measured stage.",
                    },
                },
                "source": {
                    "evaluation_id": "evaluation-id",
                    "evaluation_model_id": "candidate",
                    "dataset_manifest_sha256": "a" * 64,
                    "reference_device": "cpu",
                    "frame_ids": ["frame-id"],
                },
                "environment": {
                    "device": "cuda:0",
                    "processor": "Fixture processor",
                    "platform": "Fixture operating system",
                    "machine": "aarch64",
                    "torch": "2.10.0+cu128",
                    "torchvision": "0.25.0+cu128",
                    "python": "3.12.12",
                    "pillow": "12.3.0",
                    "threads": 4,
                    "interop_threads": 1,
                    "cuda": {
                        "name": "Synthetic GPU, not real hardware",
                        "runtime": "12.8",
                        "cudnn": 91002,
                        "capability": [8, 7],
                        "total_memory": 8 * 1024**3,
                        "index": 0,
                        "tf32_matmul": False,
                        "tf32_cudnn": False,
                        "cudnn_benchmark": False,
                    },
                },
                "summary": {
                    "parity_passed": False,
                    "frames": 1,
                    "repeats": 2,
                    "sample_count": 2,
                    "mismatched_samples": [{"frame_id": "frame-id", "repeat": 1}],
                    "timing_ms": {
                        key: {"min": value, "median": value + 0.5, "max": value + 1}
                        for key, value in (
                            ("preprocess_ms", 1),
                            ("inference_ms", 2),
                            ("postprocess_ms", 1),
                            ("total_ms", 4),
                        )
                    },
                    "decode_ms": {"min": 7, "median": 8, "max": 9},
                    "load_ms": 501,
                    "warmup_ms": 211,
                    "execution_verified": False,
                },
                "declaration": "external_execution",
            }
        ],
        "limitations": ["Selected target samples do not establish deployment-wide quality."],
    }
    return rendered


def test_scene_counts_and_negative_images_retain_operating_point_context(enriched):
    html = render(enriched)
    assert "Results by scene" in html and "evening-loading-bay" in html
    assert "Negative images: 1" in html and "Evaluated images: 2" in html
    assert "saved confidence and IoU operating point" in html
    assert "not scene AP scores" in html
    assert "Error images" in html and "FP change" in html and "<td>-1</td>" in html
    assert "All classes" in html and "Baseline · Full image" in html
    assert "Saved frame change</dt><dd>improved" in html
    assert "Example suggestion</dt><dd>Recovered object" in html


def test_scene_aggregate_and_custom_class_names_use_frozen_taxonomy(enriched):
    snapshot = enriched["report"]["snapshot"]
    snapshot["error_analysis"]["aggregate_filter"] = "__all__"
    snapshot["evaluation"]["config"]["taxonomy"] = {
        "classes": [
            {"id": "person", "name": "A person", "definition": "Person"},
            {"id": "car", "name": "A vehicle", "definition": "Vehicle"},
        ]
    }
    counts = snapshot["insights"]["scenes"][0]["counts"]
    counts["__all__"] = counts.pop("all")
    html = render(enriched)
    assert "All classes" in html and "A person" in html and "A vehicle" in html


def test_incomparable_local_timings_remain_visible_without_delta(enriched):
    html = render(enriched)
    assert "10.00 ms" in html and "8.00 ms" in html
    assert "Not comparable" in html and "Different saved execution devices." in html
    assert "-2.00 ms (-20.0%)" not in html
    assert "+10.0 pp" in html
    enriched["report"]["snapshot"]["insights"]["timing"].update(comparable=True, reasons=[])
    assert "-2.00 ms (-20.0%)" in render(enriched)


def test_sampled_video_does_not_claim_continuous_coverage_or_invent_timestamps(enriched):
    html = render(enriched)
    assert "loading-bay.mp4" in html and "video-id" in html
    assert "Timed frames" in html and "1.250 s" in html
    assert "not a continuous measured duration" in html
    assert "does not establish continuous video coverage" in html
    assert "One sampled frame has no source timestamp." in html
    source = enriched["report"]["snapshot"]["insights"]["sampling"]["video_sources"][0]
    source.update(timestamps_available=0, first_timestamp_seconds=None, last_timestamp_seconds=None)
    assert "1.250 s" not in render(enriched)


def test_declared_target_measurement_keeps_identity_environment_and_failed_parity(enriched):
    html = render(enriched)
    for value in (
        "measurement-id",
        "export-id",
        "model_candidate",
        "d" * 64,
        "e" * 64,
        "2" * 64,
        "a" * 64,
        "cuda:0",
        "2.10.0+cu128",
        "aarch64",
        "Synthetic GPU, not real hardware",
        "TF32 matrix multiplication",
    ):
        assert value in html
    assert "Exact parity: FAILED" in html and "Mismatched samples: 1" in html
    assert "Execution has not been independently verified" in html
    assert "no cross-context speedup or automatic winner is inferred" in html
    assert "Producer declaration: external_execution" in html
    assert "Selected target samples do not establish deployment-wide quality." in html
    assert "Measurement payload SHA-256</dt><dd>" + "d" * 64 in html
    assert "Export fingerprint" not in html


def test_local_cpu_runtime_settings_and_timing_boundaries_are_auditable(enriched):
    snapshot = enriched["report"]["snapshot"]
    snapshot["deployments"]["measurements"] = []
    runtime = snapshot["lanes"][0]["runtime"]
    runtime.update(
        platform="Synthetic local CPU platform",
        architecture="fasterrcnn_mobilenet_v3_large_320_fpn",
        threads=7,
        interop_threads=3,
        timing_protocol={
            "version": "torchvision-forward-v1",
            "total_ms": "Recorded local preprocessing, decode, forward and postprocessing.",
            "decode_ms": "Recorded local image checksum and decoding.",
            "warmup_in_timings": False,
            "batch_size": 1,
        },
    )
    html = render(enriched)
    assert "Platform</dt><dd>Synthetic local CPU platform" in html
    assert "Architecture</dt><dd>fasterrcnn_mobilenet_v3_large_320_fpn" in html
    assert "CPU threads</dt><dd>7" in html and "Interop threads</dt><dd>3" in html
    assert "Recorded local image checksum and decoding." in html
    assert "Warmup included in timings</dt><dd>False" in html


def test_local_cuda_context_uses_explicit_fields_and_escapes_saved_hardware(enriched):
    snapshot = enriched["report"]["snapshot"]
    cuda = deepcopy(snapshot["deployments"]["measurements"][0]["environment"]["cuda"])
    attack = '<img src="https://bad.invalid" onerror="evil()">Local GPU'
    cuda.update(name=attack, uuid="PRIVATE DEVICE UUID", path="/private/device")
    snapshot["lanes"][1]["runtime"].update(device="cuda:0", cuda=cuda)
    snapshot["deployments"]["measurements"] = []
    html = render(enriched)
    for label, value in (
        ("CUDA runtime", "12.8"),
        ("cuDNN", "91002"),
        ("GPU index", "0"),
        ("Compute capability", "[8, 7]"),
        ("GPU memory (bytes)", str(8 * 1024**3)),
        ("TF32 matrix multiplication", "False"),
        ("TF32 cuDNN", "False"),
        ("cuDNN benchmarking", "False"),
    ):
        assert f"{label}</dt><dd>{value}" in html
    assert escape(attack, quote=True) in html and attack not in html
    assert "PRIVATE DEVICE UUID" not in html and "/private/device" not in html
    assert not {"img", "script"} & {tag for tag, _ in Document(html).tags}


def test_target_processing_decode_load_and_warmup_are_separate(enriched):
    html = render(enriched)
    assert "<td>Total processing</td><td>4.00 ms</td><td>4.50 ms</td><td>5.00 ms</td>" in html
    assert "<td>Image decoding</td><td>7.00 ms</td><td>8.00 ms</td><td>9.00 ms</td>" in html
    assert "Model loading (excluded from processing)</dt><dd>501.00 ms" in html
    assert "Warmup (excluded from processing)</dt><dd>211.00 ms" in html
    assert "Before and after each measured stage." in html
    assert "-55.0%" not in html


def test_simulation_is_visibly_not_hardware_evidence(enriched):
    enriched["report"]["snapshot"]["deployments"]["measurements"][0]["declaration"] = "simulation"
    html = render(enriched)
    assert "SIMULATION" in html
    assert "this is not measured hardware performance" in html


def test_empty_target_selection_does_not_suggest_an_unperformed_measurement(enriched):
    enriched["report"]["snapshot"]["deployments"]["measurements"] = []
    html = render(enriched)
    assert "No target measurement was selected for this report." in html
    assert "Exact parity:" not in html and "Measured samples" not in html


def test_legacy_snapshot_keeps_its_recorded_evidence_without_invented_insights(rendered):
    html = render(rendered)
    assert "Results by scene" not in html and "Declared target measurements" not in html
    assert "Image and video sampling" not in html
    assert "-2.00 ms (-20.0%)" in html


@pytest.mark.parametrize(
    "field", ["name", "declaration", "scene", "video", "reason", "environment"]
)
def test_enriched_strings_remain_text_not_markup_or_external_resources(enriched, field):
    attack = '</dd><script>evil()</script><img src="https://evil.invalid" onerror="evil()">&'
    snapshot = enriched["report"]["snapshot"]
    measurement = snapshot["deployments"]["measurements"][0]
    if field in {"name", "declaration"}:
        measurement[field] = attack
    elif field == "scene":
        snapshot["insights"]["scenes"][0]["scene_group"] = attack
    elif field == "video":
        snapshot["insights"]["sampling"]["video_sources"][0]["filename"] = attack
    elif field == "reason":
        snapshot["insights"]["timing"]["reasons"] = [attack]
    else:
        measurement["environment"]["cuda"]["name"] = attack
    html = render(enriched, include_images=True)
    assert escape(attack, quote=True) in html and attack not in html
    document = Document(html)
    assert attack in "".join(document.text)
    assert not {"script", "img", "link", "a", "iframe", "form"} & {t for t, _ in document.tags}
    for _, attrs in document.tags:
        assert all(not key.startswith("on") for key in attrs)
        for key in ("href", "src", "action", "srcset"):
            assert key not in attrs or attrs[key].startswith("data:image/jpeg;base64,")
    assert exporter.CSP in next(
        attrs["content"]
        for tag, attrs in document.tags
        if tag == "meta" and attrs.get("http-equiv") == "Content-Security-Policy"
    )


def test_projection_omits_raw_predictions_paths_unknown_environment_and_declaration_keys(enriched):
    private = {
        "path": "/home/private/weights",
        "raw_predictions": "PRIVATE PREDICTIONS",
        "prompt": "PRIVATE PROMPT",
    }
    snapshot = enriched["report"]["snapshot"]
    measurement = snapshot["deployments"]["measurements"][0]
    for target in (
        measurement,
        measurement["environment"],
        measurement["environment"]["cuda"],
        measurement["source"],
        measurement["profile"],
        measurement["profile"]["timing"],
        measurement["summary"],
        snapshot["insights"],
        snapshot["insights"]["sampling"],
        snapshot["insights"]["sampling"]["video_sources"][0],
    ):
        target.update(private)
    html = render(enriched)
    assert all(value not in html for value in private.values())


def test_enriched_export_is_readonly_and_does_not_import_ml_or_read_images(enriched, monkeypatch):
    before = deepcopy(enriched["report"])
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"torch", "torchvision", "numpy", "requests", "httpx"}:
            pytest.fail("Export must use frozen evidence without model/runtime/network access")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    render(enriched)
    assert enriched["report"] == before
    assert enriched["reads"] == []


def test_enriched_sections_obey_exact_utf8_document_bound(enriched, monkeypatch):
    enriched["report"]["snapshot"]["deployments"]["measurements"][0]["name"] = "Évaluation GPU"
    size = len(render(enriched).encode("utf-8"))
    monkeypatch.setattr(exporter, "MAX_HTML_BYTES", size)
    assert len(render(enriched).encode("utf-8")) == size
    monkeypatch.setattr(exporter, "MAX_HTML_BYTES", size - 1)
    with pytest.raises(exporter.ExperimentExportLimitError):
        render(enriched)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, True, "<script>"])
def test_invalid_target_timing_fails_closed(enriched, value):
    measurement = enriched["report"]["snapshot"]["deployments"]["measurements"][0]
    measurement["summary"]["timing_ms"]["inference_ms"]["median"] = value
    with pytest.raises(ValueError, match="invalid numeric"):
        render(enriched)
