"""Portable reports retain generic frozen semantics and historical report behavior."""

import json
from copy import deepcopy
from html.parser import HTMLParser

import pytest
from test_evaluation_analysis import saved as legacy_saved
from test_evaluation_taxonomy import OFFICIAL, queue, run
from test_evaluation_taxonomy import custom_workspace as custom_workspace

from iris import experiments
from iris.annotations import save_annotation
from iris.evaluation import evaluation_detail
from iris.experiment_export import render_experiment_html
from iris.experiments import create_experiment, experiment_detail, preview_experiment
from iris.store import DEFAULT_PROJECT_ID
from iris.taxonomies import TAXONOMY, publish_taxonomy

legacy_saved = legacy_saved


class SvgLabels(HTMLParser):
    def __init__(self, raw):
        super().__init__()
        self.in_svg = False
        self.labels = []
        self.feed(raw.decode())

    def handle_starttag(self, tag, attrs):
        if tag == "svg":
            self.in_svg = True

    def handle_endtag(self, tag):
        if tag == "svg":
            self.in_svg = False

    def handle_data(self, value):
        if self.in_svg:
            self.labels.append(value)


@pytest.mark.parametrize("official", [False, True])
def test_custom_report_keeps_all_frozen_definitions_mappings_and_analysis(
    custom_workspace, official
):
    workspace = custom_workspace(mapped=official)
    store, dataset, _, _, contract = workspace
    row = queue(workspace, **({"model_ids": [OFFICIAL]} if official else {}))
    run(workspace, row)
    detail = evaluation_detail(store, row["id"])
    preview = preview_experiment(store, row["id"])
    assert store.list("experiment_reports") == []
    assert len(preview["available_examples"]) == 2
    selected = [frame["frame_id"] for frame in detail["frames"]]
    report = create_experiment(
        store,
        evaluation_id=row["id"],
        title="Custom frozen semantics",
        example_frame_ids=selected,
    )
    snapshot = report["snapshot"]
    assert snapshot["dataset"]["taxonomy"] == contract["taxonomy"]
    assert snapshot["dataset"]["class_mapping"] == dataset["manifest"]["class_mapping"]
    assert snapshot["dataset"]["coco_mapping"] == {"helmet": 1, "vehicle": 2, "all": 3}
    config = snapshot["evaluation"]["config"]
    assert config["taxonomy"] == contract["taxonomy"]
    assert config["class_mapping"] == contract["output_class_mapping"]
    assert config["model_class_contracts"] == row["config"]["model_class_contracts"]
    analysis = snapshot["error_analysis"]
    assert analysis["aggregate_filter"] == "__all__"
    assert analysis["filters"] == ["__all__", "helmet", "vehicle", "all"]
    assert analysis["summary"]["__all__"]["ground_truth_count"] == 1
    assert analysis["summary"]["all"]["ground_truth_count"] == 0
    assert snapshot["dataset"]["summary"]["class_counts"]["all"] == 0
    assert {item["label"] for item in snapshot["lanes"][0]["metrics"]["per_class"]} == {
        "helmet",
        "vehicle",
        "all",
    }
    html = render_experiment_html(store, report["id"], include_images=True, expected_revision=1)
    assert b"Protective helmet" in html and b"All marker" in html
    assert b"A worn helmet." in html
    labels = " ".join(SvgLabels(html).labels)
    assert "TP: Protective helmet" in labels and "FP: Vehicle" in labels
    assert "Label: Protective helmet" in labels
    assert "bicycle" not in labels and "person" not in labels
    if official:
        for example in snapshot["examples"]:
            detections = example["lanes"][0]["detections"]
            ignored = next(item for item in detections if item.get("ignored"))
            assert ignored["label"] == "bicycle" and ignored["label_id"] == 2
            assert ignored["taxonomy_id"] == "coco-2017-v1"
            assert ignored["native_label_id"] == 2
            assert len(detections) == 2


def test_custom_report_and_html_remain_identical_after_live_sources_and_registry_change(
    custom_workspace, monkeypatch
):
    workspace = custom_workspace()
    store, dataset, frames, _, contract = workspace
    row = queue(workspace, inference_mode="paired", tile_size=256)
    run(workspace, row)
    report = create_experiment(
        store,
        evaluation_id=row["id"],
        title="Self-contained evidence",
        example_frame_ids=[
            frame["frame_id"] for frame in evaluation_detail(store, row["id"])["frames"]
        ],
    )
    report_before = json.dumps(report, sort_keys=True).encode()
    html_before = render_experiment_html(
        store, report["id"], include_images=True, expected_revision=1
    )
    classes = deepcopy(contract["taxonomy"]["classes"])
    classes[0].update(name="New display name", definition="A changed class definition")
    publish_taxonomy(
        store, DEFAULT_PROJECT_ID, expected_taxonomy_id=contract["taxonomy_id"], classes=classes
    )
    for frame in frames:
        save_annotation(store, frame["id"], expected_revision=1, boxes=[], decisions={})
        store.artifact_path(frame["path"]).unlink()
    for frame in dataset["manifest"]["frames"]:
        store.artifact_path(frame["image_path"]).unlink()
    store.artifact_path(dataset["path"]).unlink()
    monkeypatch.setattr(
        experiments, "evaluation_detail", lambda *_: pytest.fail("Frozen report reread evaluation")
    )
    monkeypatch.setattr(
        experiments, "load_manifest", lambda *_: pytest.fail("Frozen report reread dataset")
    )
    assert (
        json.dumps(experiment_detail(store, report["id"]), sort_keys=True).encode() == report_before
    )
    assert (
        render_experiment_html(store, report["id"], include_images=True, expected_revision=1)
        == html_before
    )


def test_historical_builtin_report_keeps_bytes_after_project_becomes_custom(legacy_saved):
    store, detail, _, _ = legacy_saved
    report = create_experiment(
        store,
        evaluation_id=detail["id"],
        title="Historical person/car report",
        example_frame_ids=[detail["frames"][0]["frame_id"]],
    )
    before = render_experiment_html(store, report["id"], include_images=True, expected_revision=1)
    assert "taxonomy" not in report["snapshot"]["evaluation"]["config"]
    publish_taxonomy(
        store,
        DEFAULT_PROJECT_ID,
        expected_taxonomy_id=TAXONOMY["id"],
        classes=[{"id": "helmet", "name": "Helmet", "definition": "Protective helmet"}],
    )
    assert experiment_detail(store, report["id"]) == report
    assert (
        render_experiment_html(store, report["id"], include_images=True, expected_revision=1)
        == before
    )
    assert set(report["snapshot"]["error_analysis"]["summary"]) == {"all", "person", "car"}
