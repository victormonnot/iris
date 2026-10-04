"""Generic evaluation contracts are frozen and exercised without loading model weights."""

from copy import deepcopy

import pytest
from PIL import Image

from iris import evaluation, models
from iris.annotations import save_annotation
from iris.datasets import create_dataset
from iris.evaluation_analysis import analyze_evaluation
from iris.jobs import JobManager
from iris.media import import_asset
from iris.model_taxonomy import dataset_contract
from iris.store import DEFAULT_PROJECT_ID, Store, new_id, now
from iris.taxonomies import TAXONOMY, publish_taxonomy

OFFICIAL = "ssdlite320_mobilenet_v3_large"
TRAINED = "trained_custom_fixture"
BOX = [2, 3, 14, 25]
TIMING = {"preprocess_ms": 1, "inference_ms": 2, "postprocess_ms": 1, "total_ms": 4}


@pytest.fixture
def custom_workspace(tmp_path, monkeypatch):
    def build(*, mapped=False):
        store = Store(tmp_path / "workspace")
        classes = [
            {"id": "helmet", "name": "Protective helmet", "definition": "A worn helmet."},
            {"id": "vehicle", "name": "Vehicle", "definition": "A passenger car.", "coco_id": 3},
            {"id": "all", "name": "All marker", "definition": "An uncommon marker."},
        ]
        if mapped:
            classes[0]["coco_id"] = 1
            classes[2]["coco_id"] = 90
        taxonomy = publish_taxonomy(
            store, DEFAULT_PROJECT_ID, expected_taxonomy_id=TAXONOMY["id"], classes=classes
        )
        frames = []
        for color, split in enumerate(("train", "val", "val", "test"), 1):
            session = store.insert(
                "sessions",
                {"id": new_id(), "name": split, "scene_group": split, "created_at": now()},
            )
            path = tmp_path / f"source-{color}.png"
            Image.new("RGB", (40, 30), (color, 20, 50)).save(path)
            asset = import_asset(store, session["id"], path, path.name)
            frame = store.list("frames", asset_id=asset["id"])[0]
            store.update("frames", frame["id"], {"selected": True})
            save_annotation(
                store,
                frame["id"],
                expected_revision=0,
                boxes=[] if color == 3 else [{"id": "helmet", "label": "helmet", "box": BOX}],
                decisions={},
                status="validated",
                reviewer="Synthetic reviewer",
            )
            frames.append(frame)
        dataset = create_dataset(
            store,
            name="Custom frozen release",
            frame_ids=[frame["id"] for frame in frames],
            splits={"train": "train", "val": "val", "test": "test"},
            taxonomy_id=taxonomy["id"],
        )
        official = {**models.get_spec(OFFICIAL), "status": "ready", "weight_sha256": "1" * 64}
        contract = dataset_contract(dataset["manifest"])
        trained = {
            "id": TRAINED,
            "name": "Custom checkpoint",
            "origin": "trained",
            "status": "ready",
            "weight_sha256": "2" * 64,
            "parent_model_id": OFFICIAL,
            **deepcopy(contract),
            "classes": [
                {"id": value, "name": key}
                for key, value in contract["output_class_mapping"].items()
            ],
            "provenance": {
                "training_scene_groups": ["train"],
                "training_frame_hashes": [frames[0]["sha256"]],
                "parent_weight_sha256": official["weight_sha256"],
            },
        }
        available = [official, trained]
        monkeypatch.setattr(evaluation, "catalog", lambda _root: available)
        return store, dataset, frames, available, contract

    return build


def queue(workspace, **overrides):
    store, dataset, *_ = workspace
    return evaluation.create_evaluation(
        store,
        JobManager(store),
        **{
            "name": "Frozen custom evaluation",
            "dataset_id": dataset["id"],
            "model_ids": [TRAINED],
            **overrides,
        },
    )


def detector_factory(workspace, calls=None, mutate=None):
    calls = [] if calls is None else calls
    contract = workspace[-1]

    class Detector:
        def __init__(self, root, model_id, *, device):
            self.model_id = model_id
            self.metadata = {
                "model_id": model_id,
                "device": device,
                "weight_sha256": ("1" if model_id == OFFICIAL else "2") * 64,
                **(deepcopy(contract) if model_id != OFFICIAL else {}),
            }
            calls.append(("load", model_id))

        def warmup(self, image):
            calls.append(("warmup", self.model_id))

        def predict(self, image):
            negative = image.getpixel((0, 0))[0] == 3
            label = "vehicle" if negative else "helmet"
            if self.model_id == OFFICIAL:
                detections = [
                    {
                        "label": "car" if negative else "person",
                        "label_id": 3 if negative else 1,
                        "score": 0.9,
                        "box": BOX,
                    },
                    # Native bicycle=2 collides with the custom vehicle=2 output ID.
                    {"label": "bicycle", "label_id": 2, "score": 0.99, "box": BOX},
                ]
            else:
                category = contract["output_class_mapping"][label]
                detections = [
                    {
                        "label": label,
                        "label_id": category,
                        "native_label_id": category,
                        "taxonomy_id": contract["taxonomy_id"],
                        "score": 0.9,
                        "box": BOX,
                    }
                ]
            prediction = {
                "input_size": list(image.size),
                "timing": dict(TIMING),
                "detections": detections,
            }
            if mutate:
                mutate(prediction)
            return prediction

    return Detector


def run(workspace, row, **overrides):
    return evaluation.run_evaluation(
        workspace[0],
        row["id"],
        lambda *_: None,
        lambda: False,
        detector_factory=overrides.pop("detector_factory", detector_factory(workspace)),
        **overrides,
    )


@pytest.mark.parametrize("mode", ["full", "paired"])
def test_custom_metrics_all_classes_negatives_and_analysis_filter_collision(custom_workspace, mode):
    workspace = custom_workspace()
    store, dataset, _, _, contract = workspace
    row = queue(workspace, inference_mode=mode, tile_size=256)
    assert row["config"]["taxonomy"] == dataset["manifest"]["taxonomy"]
    assert row["config"]["class_mapping"] == {"helmet": 1, "vehicle": 2, "all": 3}
    assert row["config"]["model_class_contracts"][TRAINED] == contract
    run(workspace, row)
    detail = evaluation.evaluation_detail(store, row["id"])
    for model in detail["models"]:
        metrics = model["metrics"]
        assert metrics["protocol"]["id"] == "coco-bbox-iris-v3"
        assert metrics["summary"]["tp"] == metrics["summary"]["fp"] == 1
        assert metrics["summary"]["fn"] == 0
        assert metrics["summary"]["map"] == pytest.approx(1)
        assert [item["label"] for item in metrics["per_class"]] == ["helmet", "vehicle", "all"]
        assert metrics["per_class"][1]["support"] == 0
        assert metrics["per_class"][1]["fp"] == 1
        assert metrics["per_class"][1]["ap"] is None
        assert metrics["per_class"][2]["predictions"] == 0
        assert metrics["per_class"][2]["ap"] is None
    analysis = analyze_evaluation(store, row["id"])
    assert analysis["protocol"] == "iris-error-analysis-v3"
    assert analysis["filters"] == ["__all__", "helmet", "vehicle", "all"]
    assert analysis["aggregate_filter"] == "__all__"
    assert analysis["summary"]["all"]["ground_truth_count"] == 0
    assert analysis["summary"]["__all__"]["ground_truth_count"] == 1
    assert analysis["taxonomy"] == contract["taxonomy"]
    assert analysis["class_mapping"] == contract["output_class_mapping"]


@pytest.mark.parametrize("mode", ["full", "paired"])
def test_explicit_official_coco_mapping_normalizes_and_preserves_ignored_collisions(
    custom_workspace, mode
):
    workspace = custom_workspace(mapped=True)
    row = queue(workspace, model_ids=[OFFICIAL], inference_mode=mode, tile_size=256)
    run(workspace, row)
    detail = evaluation.evaluation_detail(workspace[0], row["id"])
    for model in detail["models"]:
        summary = model["metrics"]["summary"]
        assert summary["tp"] == summary["fp"] == 1
        assert summary["ignored_prediction_count"] == 2
        assert summary["native_prediction_count"] == 4
        assert summary["project_prediction_count_before_threshold"] == 2
    for prediction in detail["predictions"]:
        mapped = next(item for item in prediction["detections"] if not item.get("ignored"))
        ignored = next(item for item in prediction["detections"] if item.get("ignored"))
        assert mapped["label"] in {"helmet", "vehicle"}
        assert mapped["native_label_id"] in {1, 3}
        assert mapped["taxonomy_id"] == workspace[-1]["taxonomy_id"]
        assert ignored["label"] == "bicycle" and ignored["label_id"] == 2
        assert ignored["taxonomy_id"] == "coco-2017-v1"
    assert (
        analyze_evaluation(workspace[0], row["id"])["summary"]["__all__"]["ground_truth_count"] == 1
    )


def test_missing_official_mapping_rejected_before_job_or_detector(custom_workspace):
    workspace = custom_workspace()
    with pytest.raises(ValueError, match="explicit COCO mapping for every"):
        queue(workspace, model_ids=[OFFICIAL])
    assert workspace[0].list("jobs") == []


@pytest.mark.parametrize("change", ["definition", "version", "mapping", "classes", "overlap"])
def test_incompatible_trained_checkpoint_never_queues(custom_workspace, change):
    workspace = custom_workspace()
    model = workspace[3][1]
    if change == "definition":
        model["taxonomy"]["classes"][0]["definition"] = "Changed semantics"
    elif change == "version":
        model["taxonomy"]["id"] = model["taxonomy_id"] = "taxonomy-" + "f" * 32
    elif change == "mapping":
        model["output_class_mapping"]["helmet"] = 3
    elif change == "classes":
        model["classes"][0]["name"] = "person"
    else:
        model["provenance"]["training_scene_groups"] = ["val"]
    with pytest.raises(ValueError, match="taxonomy|class|train"):
        queue(workspace)
    assert workspace[0].list("jobs") == []


def test_queued_contract_change_rejected_before_loading(custom_workspace):
    workspace = custom_workspace()
    row = queue(workspace)
    workspace[3][1]["taxonomy"]["classes"][0]["definition"] = "Changed after queue"
    calls = []
    with pytest.raises(ValueError, match="taxonomy|class"):
        run(workspace, row, detector_factory=detector_factory(workspace, calls))
    assert calls == []


def test_saved_analysis_independent_of_live_sources_taxonomy_and_registry(
    custom_workspace, monkeypatch
):
    workspace = custom_workspace()
    store, _, frames, _, contract = workspace
    row = queue(workspace)
    classes = deepcopy(contract["taxonomy"]["classes"])
    classes[0]["definition"] = "Future changed definition"
    publish_taxonomy(
        store, DEFAULT_PROJECT_ID, expected_taxonomy_id=contract["taxonomy_id"], classes=classes
    )
    for frame in frames:
        save_annotation(store, frame["id"], expected_revision=1, boxes=[], decisions={})
        store.artifact_path(frame["path"]).unlink()
    run(workspace, row)
    expected = analyze_evaluation(store, row["id"])
    monkeypatch.setattr(
        evaluation, "catalog", lambda _: pytest.fail("Analysis queried model registry")
    )
    with store.connect() as connection:
        connection.execute("DELETE FROM taxonomy_versions")
    assert analyze_evaluation(store, row["id"]) == expected
    assert expected["taxonomy"] == contract["taxonomy"]


@pytest.mark.parametrize(
    "corruption", ["namespace", "native", "ignored", "snapshot", "mapping", "contract"]
)
def test_analysis_rejects_corrupted_saved_custom_namespaces(custom_workspace, corruption):
    workspace = custom_workspace(mapped=True)
    store = workspace[0]
    row = queue(workspace, model_ids=[OFFICIAL])
    run(workspace, row)
    if corruption in {"namespace", "native", "ignored"}:
        prediction = store.list("evaluation_predictions", evaluation_id=row["id"])[0]
        detections = prediction["detections"]
        if corruption == "namespace":
            detections[0]["taxonomy_id"] = "coco-2017-v1"
        elif corruption == "native":
            detections[0]["native_label_id"] = 3
        else:
            detections[1].pop("ignored")
        store.update("evaluation_predictions", prediction["id"], {"detections": detections})
    else:
        config = row["config"]
        if corruption == "snapshot":
            config["taxonomy"]["classes"][0]["definition"] = "Corrupted"
        elif corruption == "mapping":
            config["class_mapping"]["helmet"] = 2
        else:
            config["model_class_contracts"][OFFICIAL] = {}
        store.update("evaluations", row["id"], {"config": config})
    with pytest.raises(ValueError):
        analyze_evaluation(store, row["id"])


def test_custom_validation_and_test_audit_keep_frozen_classes(custom_workspace):
    workspace = custom_workspace()
    validation = queue(workspace)
    run(workspace, validation)
    audit = queue(workspace, split="test", validation_evaluation_id=validation["id"])
    assert audit["config"]["taxonomy"] == validation["config"]["taxonomy"]
    assert audit["config"]["model_class_contracts"] == validation["config"]["model_class_contracts"]
    run(workspace, audit)
    assert (
        analyze_evaluation(workspace[0], audit["id"])["summary"]["__all__"]["ground_truth_count"]
        == 1
    )


@pytest.mark.parametrize(
    "mutation",
    [
        {"label": "person"},
        {"label_id": 3},
        {"taxonomy_id": "iris-objects-v1"},
        {"native_label_id": 3},
        {"ignored": True},
    ],
)
def test_custom_detector_mismapped_outputs_rejected_before_prediction_persistence(
    custom_workspace, mutation
):
    workspace = custom_workspace()
    row = queue(workspace)
    factory = detector_factory(
        workspace, mutate=lambda prediction: prediction["detections"][0].update(mutation)
    )
    with pytest.raises(ValueError):
        run(workspace, row, detector_factory=factory)
    assert workspace[0].list("evaluation_predictions", evaluation_id=row["id"]) == []


def test_reference_selection_records_custom_class_snapshot_and_output_mapping(custom_workspace):
    workspace = custom_workspace()
    row = queue(workspace)
    run(workspace, row)
    reference = evaluation.promote_reference(
        workspace[0],
        evaluation_id=row["id"],
        model_id=TRAINED,
        reviewer="Synthetic reviewer",
        notes="Custom validation checkpoint",
    )
    assert reference["metadata"]["taxonomy"] == workspace[-1]["taxonomy"]
    assert reference["metadata"]["class_mapping"] == workspace[-1]["output_class_mapping"]
    assert reference["metadata"]["class_contract"] == workspace[-1]
