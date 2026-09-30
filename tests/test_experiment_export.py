"""Standalone report safety and semantics; measurements below are synthetic fixtures."""

import base64
import hashlib
import io
from copy import deepcopy
from html.parser import HTMLParser

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from test_evaluation_api import evaluate_fixture
from test_evaluation_api import ready_models as ready_models
from test_training_api import BASE_URL, prepare_dataset

from iris import experiment_export as exporter
from iris import experiments
from iris.app import create_app

STAMP = "2026-09-30T14:00:00+00:00"


class Document(HTMLParser):
    def __init__(self, text):
        super().__init__(convert_charrefs=True)
        self.tags = []
        self.text = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))

    def handle_data(self, data):
        self.text.append(data)


def fixture_report():
    lanes = []
    for index, (identifier, name) in enumerate(
        (("baseline", "Baseline"), ("candidate", "Candidate"))
    ):
        lanes.append(
            {
                "id": identifier,
                "model_id": "model_" + identifier,
                "variant": "full",
                "name": name + " · Full image",
                "weight_sha256": str(index + 1) * 64,
                "metrics": {
                    "protocol": {"id": "synthetic-protocol"},
                    "summary": {
                        "map": 0.4 + index / 10,
                        "map50": 0.6,
                        "map75": None,
                        "precision": 0.5,
                        "recall": 0.5,
                        "tp": 1,
                        "fp": 1,
                        "fn": 1,
                        "frame_count": 1,
                        "ground_truth_count": 2,
                    },
                    "per_class": [
                        {
                            "label": "person",
                            "support": 2,
                            "ap": 0.4 + index / 10,
                            "ap50": 0.6,
                            "precision": 1.0,
                            "recall": 0.5,
                            "fp": 0,
                            "fn": 1,
                        },
                        {
                            "label": "car",
                            "support": 0,
                            "ap": None,
                            "ap50": None,
                            "precision": 0.0,
                            "recall": None,
                            "fp": 1,
                            "fn": 0,
                        },
                    ],
                    "warnings": ["Synthetic fixture; no field accuracy claim."],
                },
                "timing": {"frame_count": 1, "mean_total_ms": 10.0 - index * 2},
                "runtime": {
                    "device": "cpu",
                    "hardware": "Fixture CPU",
                    "precision": "float32",
                    "torch_version": "fixture",
                    "torchvision_version": "fixture",
                },
                "training": None,
                "training_status": "pretrained",
            }
        )
    lanes[1]["training_status"] = "recorded"
    lanes[1]["training"] = {
        "id": "training-id",
        "name": "Fixture partial adaptation",
        "dataset_id": "dataset-id",
        "dataset_name": "Fixture release",
        "dataset_manifest_sha256": "a" * 64,
        "parent_model_id": "model_baseline",
        "parent_weight_sha256": "1" * 64,
        "config": {"scope": "partial_backbone", "steps": 2, "learning_rate": 0.001, "seed": 7},
        "history_summary": {"steps_completed": 2, "first_loss": 0.9, "last_loss": 0.8},
    }
    example_lanes = []
    for index, lane in enumerate(lanes):
        example_lanes.append(
            {
                "run_id": lane["id"],
                "model_id": lane["model_id"],
                "variant": "full",
                "detections": [
                    {"label": "bicycle", "label_id": 2, "score": 0.9, "box": [1, 1, 5, 5]},
                    {"label": "person", "label_id": 1, "score": 0.1, "box": [1, 1, 5, 5]},
                    {"label": "person", "label_id": 1, "score": 0.9, "box": [10, 10, 80, 80]},
                    {"label": "car", "label_id": 3, "score": 0.8, "box": [100, 90, 160, 140]},
                ],
                "errors": {
                    "tp": 1,
                    "fp": 1,
                    "fn": 1,
                    "false_positives": [3],
                    "false_negatives": [1 - index],
                    "matches": [
                        {
                            "label": "person",
                            "ground_truth_index": index,
                            "detection_index": 2,
                            "iou": 1.0,
                        }
                    ],
                },
            }
        )
    snapshot = {
        "version": 1,
        "captured_at": STAMP,
        "evaluation": {
            "id": "evaluation-id",
            "name": "Frozen fixture comparison",
            "dataset_id": "dataset-id",
            "split": "val",
            "created_at": STAMP,
            "config": {
                "inference": {"mode": "full"},
                "protocol": {
                    "id": "synthetic-protocol",
                    "engine": "Fixture engine",
                    "engine_version": "fixture",
                    "numpy_version": "fixture",
                },
            },
        },
        "dataset": {
            "id": "dataset-id",
            "name": "Fixture release",
            "manifest_sha256": "a" * 64,
            "summary": {"frame_count": 3},
            "source_groups": [{"scene_group": "synthetic-val", "frame_count": 1}],
            "sources": [
                {
                    "source_url": "https://example.invalid/dataset",
                    "license_name": "Fixture license",
                    "attribution": "Fixture author",
                }
            ],
        },
        "lanes": lanes,
        "error_analysis": {
            "protocol": "iris-error-analysis-v2",
            "confidence_threshold": 0.5,
            "iou_threshold": 0.5,
            "warnings": ["Synthetic fixture; no field accuracy claim."],
            "comparison": {"baseline_run_id": "baseline", "candidate_run_id": "candidate"},
            "summary": {
                label: {
                    "changes": {
                        "recovered": 0 if label == "car" else 1,
                        "new_misses": 0 if label == "car" else 1,
                        "fp_delta": 0,
                    }
                }
                for label in ("all", "person", "car")
            },
        },
        "reference_decisions": {
            "historical": True,
            "captured_at": STAMP,
            "current_reference_id": "reference-id",
            "decisions": [
                {
                    "id": "reference-id",
                    "model_id": "model_candidate",
                    "model_name": "Candidate",
                    "variant": "full",
                    "created_at": STAMP,
                    "reviewer": "Fixture reviewer",
                    "notes": "Persistence check only.",
                }
            ],
        },
        "examples": [
            {
                "frame_id": "frame-id",
                "width": 640,
                "height": 360,
                "scene_group": "synthetic-val",
                "source": {"filename": "fixture.jpg", "timestamp_seconds": 1.5, "frame_index": 45},
                "ground_truth": [
                    {"id": "a", "label": "person", "box": [10, 10, 80, 80]},
                    {"id": "b", "label": "person", "box": [200, 10, 280, 80]},
                ],
                "lanes": example_lanes,
            }
        ],
    }
    return {
        "id": "report-id",
        "evaluation_id": "evaluation-id",
        "title": "Synthetic experiment",
        "objective": "Inspect stored results.",
        "conclusion": "No quality conclusion from fixtures.",
        "revision": 1,
        "snapshot": snapshot,
        "snapshot_sha256": "c" * 64,
        "images": [],
        "created_at": STAMP,
        "updated_at": STAMP,
    }


@pytest.fixture
def rendered(monkeypatch):
    report = fixture_report()
    output = io.BytesIO()
    Image.new("RGB", (640, 360), "#b7c1ac").save(output, format="JPEG", quality=85)
    image_bytes = output.getvalue()
    state = {"report": report, "reads": [], "detail_reads": 0, "image": image_bytes}

    def detail(_store, identifier):
        assert identifier == report["id"]
        state["detail_reads"] += 1
        return deepcopy(state["report"])

    def image(_store, identifier, frame_id):
        assert identifier == report["id"] and frame_id == "frame-id"
        state["reads"].append(frame_id)
        return state["image"]

    monkeypatch.setattr(experiments, "experiment_detail", detail)
    monkeypatch.setattr(experiments, "read_experiment_image", image)
    return state


def render(state, **kwargs):
    return exporter.render_experiment_html(
        None, "report-id", expected_revision=1, **kwargs
    ).decode()


def test_default_export_is_text_only_and_reads_no_image_pixels(rendered):
    html = render(rendered)
    document = Document(html)
    assert rendered["reads"] == []
    assert rendered["detail_reads"] == 2
    assert not {"img", "image", "svg"} & {tag for tag, _ in document.tags}
    assert "data:image" not in html
    assert "No image pixels included" in html
    assert "partial_backbone" in html and "training-id" in html
    assert "1" * 64 in html and "a" * 64 in html
    assert "Fixture license" in html and "Fixture author" in html
    assert "Fixture engine" in html


def test_rates_use_percentage_points_and_timing_uses_absolute_and_relative_change(rendered):
    html = render(rendered)
    assert "+10.0 pp" in html
    assert "+25.0%" not in html
    assert "-2.00 ms (-20.0%)" in html
    assert "<td>N/A</td>" in html
    assert "N/A means undefined or unavailable, never a perfect score" in html
    assert "Recovered by candidate" in html and "Newly missed by candidate" in html


def test_recorded_metric_and_timing_definitions_are_included(rendered):
    snapshot = rendered["report"]["snapshot"]
    snapshot["evaluation"]["config"]["protocol"].update(
        max_dets=[1, 10, 100], ap_iou_thresholds=[0.5, 0.55, 0.6, 0.95]
    )
    for lane in snapshot["lanes"]:
        lane["runtime"]["timing_protocol"] = {
            "version": "fixture-timing-v1",
            "total_ms": "Saved decode, verification and forward; excludes warmup and weights",
            "inference_ms": "All detector passes",
            "warmup_frames": "One saved warmup, excluded",
            "batch_size": 1,
            "synchronize": "CPU fixture",
            "prompt": "PRIVATE TIMING EXTRA",
        }
    html = render(rendered)
    assert "<details open>" in html
    assert "1, 10, 100" in html and "0.50, 0.55, 0.60, 0.95" in html
    assert "fixture-timing-v1" in html and "excludes warmup and weights" in html
    assert "One saved warmup, excluded" in html
    assert "All detector passes" in html and "CPU fixture" in html
    assert "PRIVATE TIMING EXTRA" not in html
    assert "Evaluated images" in html and "Reviewed objects" in html


@pytest.mark.parametrize(
    "baseline,candidate,expected",
    [(0, 8, "+8.00 ms (relative N/A)"), (None, 8, "N/A"), (10, None, "N/A")],
)
def test_zero_or_unavailable_timings_do_not_invent_relative_change(
    rendered, baseline, candidate, expected
):
    lanes = rendered["report"]["snapshot"]["lanes"]
    lanes[0]["timing"]["mean_total_ms"] = baseline
    lanes[1]["timing"]["mean_total_ms"] = candidate
    html = render(rendered)
    assert f"<td>{expected}</td>" in html
    assert "nan" not in html.lower() and "infinity" not in html.lower()


def test_standalone_document_has_restrictive_csp_and_no_active_or_external_resources(rendered):
    html = render(rendered, include_images=True)
    document = Document(html)
    tags = {tag for tag, _ in document.tags}
    assert not {"script", "iframe", "object", "embed", "link", "form", "a"} & tags
    policies = [
        attrs["content"]
        for tag, attrs in document.tags
        if tag == "meta" and attrs.get("http-equiv") == "Content-Security-Policy"
    ]
    assert policies == [exporter.CSP]
    assert "default-src 'none'" in policies[0] and "img-src data:" in policies[0]
    assert "@page { size: A4" in html and "@media print" in html
    assert "@media(max-width:" in html and "@font-face" not in html and "url(" not in html
    for _, attrs in document.tags:
        assert all(not key.lower().startswith("on") for key in attrs)
        for key in ("href", "src", "srcset", "action"):
            if key in attrs:
                assert attrs[key].startswith("data:image/jpeg;base64,")


def test_image_export_embeds_exact_verified_jpegs_and_original_geometry(rendered):
    html = render(rendered, include_images=True)
    document = Document(html)
    images = [attrs for tag, attrs in document.tags if tag == "image"]
    assert rendered["reads"] == ["frame-id"]
    assert len(images) == 2
    assert all(
        base64.b64decode(item["href"].split(",", 1)[1]) == rendered["image"] for item in images
    )
    assert all(item["width"] == "640" and item["height"] == "360" for item in images)
    assert all(attrs["viewbox"] == "0 0 640 360" for tag, attrs in document.tags if tag == "svg")
    rectangles = [attrs for tag, attrs in document.tags if tag == "rect"]
    assert len(rectangles) == 8
    assert sum("stroke-dasharray" in item for item in rectangles) == 4
    assert sum(item["stroke"] == "#18744e" for item in rectangles) == 2
    assert sum(item["stroke"] == "#ad661b" for item in rectangles) == 2
    assert "bicycle" not in html and "10.0%" not in html


def test_tiny_source_images_do_not_create_oversized_or_offscreen_labels(rendered):
    example = rendered["report"]["snapshot"]["examples"][0]
    example.update(width=32, height=18)
    for box in example["ground_truth"]:
        box["box"] = [value / 20 for value in box["box"]]
    for lane in example["lanes"]:
        for box in lane["detections"]:
            box["box"] = [value / 20 for value in box["box"]]
    labels = [
        attrs
        for tag, attrs in Document(render(rendered, include_images=True)).tags
        if tag == "text"
    ]
    assert labels
    for label in labels:
        # Text stays below 1/16 of the source width, even when the image is
        # magnified into a report card; a 10-source-pixel minimum breaks this.
        assert float(label["font-size"]) <= example["width"] / 16
        assert 0 <= float(label["x"]) < example["width"]
        assert 0 < float(label["y"]) <= example["height"]


@pytest.mark.parametrize(
    "field",
    [
        "title",
        "objective",
        "conclusion",
        "model",
        "dataset",
        "filename",
        "scene_group",
        "reviewer",
        "notes",
        "warning",
        "license",
        "attribution",
    ],
)
def test_every_user_or_source_string_is_escaped(rendered, field):
    attack = '</title><script>alert("x")</script><img src="https://bad.invalid/" onerror="evil()">&'
    report = rendered["report"]
    snapshot = report["snapshot"]
    if field in {"title", "objective", "conclusion"}:
        report[field] = attack
    elif field == "model":
        snapshot["lanes"][0]["name"] = attack
    elif field == "dataset":
        snapshot["dataset"]["name"] = attack
    elif field == "filename":
        snapshot["examples"][0]["source"]["filename"] = attack
    elif field == "scene_group":
        snapshot["examples"][0]["scene_group"] = attack
    elif field in {"reviewer", "notes"}:
        snapshot["reference_decisions"]["decisions"][0][field] = attack
    elif field == "warning":
        snapshot["error_analysis"]["warnings"] = [attack]
    else:
        snapshot["dataset"]["sources"][0]["license_name" if field == "license" else field] = attack
    html = render(rendered, include_images=True)
    document = Document(html)
    assert attack in "".join(document.text)
    assert attack not in html
    assert not {"script", "img"} & {tag for tag, _ in document.tags}
    assert all(not key.startswith("on") for _, attrs in document.tags for key in attrs)


def test_export_allowlist_omits_paths_keys_raw_prompts_and_annotation_notes(rendered):
    private = {
        "path": "/home/private-user/secret-file",
        "api_key": "sk-private-fixture",
        "prompt": "PRIVATE RAW ASSISTANCE PROMPT",
        "notes": "PRIVATE ANNOTATION NOTE",
        "raw_response": "PRIVATE MODEL RESPONSE",
    }
    snapshot = rendered["report"]["snapshot"]
    rendered["report"].update(private)
    snapshot.update(private)
    snapshot["evaluation"]["config"].update(private)
    snapshot["dataset"].update(private)
    snapshot["examples"][0]["source"].update(private)
    for lane in snapshot["lanes"]:
        lane.update(private)
        lane["runtime"].update(private)
        lane["metrics"].update(private)
    html = render(rendered, include_images=True)
    assert all(value not in html for value in private.values())


def test_single_run_has_no_comparison_deltas(rendered):
    snapshot = rendered["report"]["snapshot"]
    snapshot["lanes"] = snapshot["lanes"][:1]
    snapshot["examples"][0]["lanes"] = snapshot["examples"][0]["lanes"][:1]
    snapshot["error_analysis"]["comparison"] = None
    html = render(rendered)
    assert "Candidate − baseline" not in html
    assert "One run: recovered objects and new misses are not compared" in html
    assert "+10.0 pp" not in html


def test_paired_same_checkpoint_runs_keep_variants_distinct(rendered):
    snapshot = rendered["report"]["snapshot"]
    snapshot["lanes"][1].update(model_id="model_baseline", variant="tiled", name="Baseline · Tiled")
    snapshot["evaluation"]["config"]["inference"] = {
        "mode": "paired",
        "tiling": {"tile_size": 640, "overlap": 0.2, "merge_iou": 0.5},
    }
    html = render(rendered)
    assert "Baseline · Full image" in html and "Baseline · Tiled" in html
    assert "Full image · Full image" not in html
    assert "Tile size" in html and "<dd>640</dd>" in html
    assert "Tile overlap" in html and "<dd>20.0%</dd>" in html


def test_test_split_is_labeled_as_audit_and_references_are_historical(rendered):
    rendered["report"]["snapshot"]["evaluation"]["split"] = "test"
    html = render(rendered)
    assert "Final test audit" in html
    assert "not a basis for choosing or promoting a model" in html
    assert "Historical decisions captured with this report" in html
    assert "They do not describe the workspace's current reference" in html


def test_stale_revision_fails_before_reading_images(rendered):
    rendered["report"]["revision"] = 2
    with pytest.raises(experiments.ExperimentConflict):
        render(rendered, include_images=True)
    assert rendered["reads"] == []


def test_revision_change_during_render_is_not_silently_downloaded(rendered, monkeypatch):
    original = experiments.read_experiment_image

    def changing_image(*args):
        rendered["report"]["revision"] = 2
        return original(*args)

    monkeypatch.setattr(experiments, "read_experiment_image", changing_image)
    with pytest.raises(experiments.ExperimentConflict, match="during export"):
        render(rendered, include_images=True)


@pytest.mark.parametrize(
    "field,value",
    [
        ("include_images", 1),
        ("include_images", "true"),
        ("expected_revision", True),
        ("expected_revision", 0),
    ],
)
def test_invalid_export_arguments_are_rejected_before_loading(rendered, field, value):
    values = {"include_images": False, "expected_revision": 1, field: value}
    with pytest.raises(ValueError):
        exporter.render_experiment_html(None, "report-id", **values)
    assert rendered["detail_reads"] == 0


def test_html_size_bound_is_enforced_even_without_images(rendered, monkeypatch):
    monkeypatch.setattr(exporter, "MAX_HTML_BYTES", 100)
    with pytest.raises(exporter.ExperimentExportLimitError):
        render(rendered)
    assert rendered["reads"] == []


def test_image_base64_budget_is_checked_before_encoding(rendered, monkeypatch):
    rendered["image"] = b"x" * exporter.MAX_HTML_BYTES

    def no_encoding(_content):
        pytest.fail("Oversized images must be rejected before base64 allocation")

    monkeypatch.setattr(exporter.base64, "b64encode", no_encoding)
    with pytest.raises(exporter.ExperimentExportLimitError):
        render(rendered, include_images=True)


@pytest.mark.parametrize(
    "box",
    [
        [0, 0, float("nan"), 5],
        [0, 0, float("inf"), 5],
        [False, 0, 5, 5],
        [-1, 0, 5, 5],
        [0, 0, 641, 5],
        ['1" onload="evil()', 0, 5, 5],
    ],
)
def test_nonfinite_or_injected_svg_coordinates_fail_closed(rendered, box):
    rendered["report"]["snapshot"]["examples"][0]["ground_truth"][0]["box"] = box
    with pytest.raises(ValueError):
        render(rendered, include_images=True)


def test_verified_saved_report_exports_without_original_data_or_model_runtime(
    tmp_path, ready_models, monkeypatch
):
    with TestClient(
        create_app(tmp_path / "workspace", run_jobs=False), base_url=BASE_URL
    ) as client:
        dataset, _ = prepare_dataset(client)
        evaluated = evaluate_fixture(client, dataset["id"])
        store = client.app.state.store
        report = experiments.create_experiment(
            store,
            evaluation_id=evaluated["id"],
            title="Real saved fixture report",
            objective="Export existing synthetic measurements only.",
            example_frame_ids=[evaluated["frames"][0]["frame_id"]],
        )
        before = {
            name: store.list(name) for name in ("jobs", "evaluations", "model_references", "frames")
        }
        expected_image = experiments.read_experiment_image(
            store, report["id"], report["images"][0]["frame_id"]
        )
        assert hashlib.sha256(expected_image).hexdigest() == report["images"][0]["sha256"]
        for frame in dataset["manifest"]["frames"]:
            store.artifact_path(frame["image_path"]).unlink()

        def forbidden(*_args, **_kwargs):
            pytest.fail("Export must read saved report evidence, not its live source or models")

        monkeypatch.setattr(experiments, "evaluation_detail", forbidden)
        monkeypatch.setattr(experiments, "load_manifest", forbidden)
        monkeypatch.setattr(experiments, "_analyze", forbidden)
        html = exporter.render_experiment_html(
            store, report["id"], expected_revision=1, include_images=True
        ).decode()
        embedded = [attrs["href"] for tag, attrs in Document(html).tags if tag == "image"]
        assert embedded and all(
            base64.b64decode(value.split(",", 1)[1]) == expected_image for value in embedded
        )
        assert {name: store.list(name) for name in before} == before
        frozen = store.get("experiment_reports", report["id"])
        image = store.artifact_path(frozen["images"][0]["path"])
        image.write_bytes(b"changed")
        # Text export still has no dependency on the image files.
        exporter.render_experiment_html(store, report["id"], expected_revision=1)
        with pytest.raises(ValueError, match="saved size"):
            exporter.render_experiment_html(
                store, report["id"], expected_revision=1, include_images=True
            )
        snapshot = frozen["snapshot"]
        snapshot["dataset"]["name"] = "tampered"
        store.update("experiment_reports", report["id"], {"snapshot": snapshot})
        with pytest.raises(ValueError, match="saved hash"):
            exporter.render_experiment_html(store, report["id"], expected_revision=1)
