"""Custom detector identities survive saved comparisons and human review proposals."""

from copy import deepcopy

import pytest
from test_datasets_custom_api import client as api_fixture
from test_datasets_custom_api import freeze
from test_datasets_custom_api import reviewed as reviewed_fixture

from iris import inference
from iris.annotations import add_detector_suggestions, get_annotation
from iris.jobs import JobManager
from iris.model_taxonomy import dataset_contract
from iris.prediction_taxonomy import validate_output_labels
from iris.taxonomies import publish_taxonomy

client = api_fixture
reviewed = reviewed_fixture
TIMING = {"preprocess_ms": 1, "inference_ms": 2, "postprocess_ms": 1, "total_ms": 4}


def setup_detector(client, reviewed, monkeypatch):
    dataset = freeze(client, reviewed[2])
    contract = dataset_contract(dataset["manifest"])
    model = {
        "id": "synthetic-custom-checkpoint",
        "name": "Synthetic custom detector",
        "origin": "trained",
        "status": "ready",
        "weight_sha256": "a" * 64,
        **contract,
    }
    monkeypatch.setattr(inference, "catalog", lambda _: [deepcopy(model)])
    monkeypatch.setattr(inference, "get_spec", lambda *_: deepcopy(model))

    class Detector:
        def __init__(self, *args, **kwargs):
            self.metadata = {"weight_sha256": model["weight_sha256"], **deepcopy(contract)}

        def warmup(self, image):
            pass

        def predict(self, image):
            return {
                "input_size": list(image.size),
                "timing": dict(TIMING),
                "detections": [
                    {
                        "label": "damaged_panel",
                        "label_id": 3,
                        "native_label_id": 3,
                        "taxonomy_id": contract["taxonomy_id"],
                        "box": [2, 3, 20, 22],
                        "score": 0.9,
                    }
                ],
            }

    return client.app.state.store, model, Detector


@pytest.mark.parametrize("mode", ["full", "tiled", "paired"])
def test_saved_custom_predictions_become_pending_proposals_with_frozen_semantics(
    client, reviewed, monkeypatch, mode
):
    store, model, detector = setup_detector(client, reviewed, monkeypatch)
    frame = reviewed[1][0]
    row = inference.create_comparison(
        store,
        JobManager(store),
        frame["session_id"],
        name="Synthetic identity check",
        frame_ids=[frame["id"]],
        model_ids=[model["id"]],
        inference_mode=mode,
        tile_size=128,
    )
    assert row["config"]["taxonomy"] == "model-specific-v1"
    assert row["config"]["model_class_contracts"][model["id"]]["taxonomy"] == reviewed[0]
    result = inference.run_comparison(
        store, row["id"], lambda *_: None, lambda: False, detector_factory=detector
    )
    assert result["predictions_created"] == (2 if mode == "paired" else 1)
    prediction = store.list("predictions", comparison_id=row["id"])[0]
    assert prediction["detections"][0]["label"] == "damaged_panel"
    # The live project can evolve; this frame and source still use their frozen version.
    publish_taxonomy(
        store,
        "default",
        expected_taxonomy_id=reviewed[0]["id"],
        classes=[
            {**item, "definition": item["definition"] + " Changed."}
            for item in reviewed[0]["classes"]
        ],
    )
    monkeypatch.setattr(inference, "get_spec", lambda *_: pytest.fail("Read saved evidence only"))
    annotation = get_annotation(store, frame["id"])
    assert any(source["id"] == prediction["id"] for source in annotation["prediction_sources"])
    result = add_detector_suggestions(
        store,
        frame["id"],
        prediction_id=prediction["id"],
        expected_revision=1,
    )
    proposals = store.list("annotation_suggestions", frame_id=frame["id"])
    assert len(proposals) == 1 and proposals[0]["label"] == "damaged_panel"
    assert proposals[0]["metadata"]["source_taxonomy"] == reviewed[0]["id"]
    assert result["revision"] == 1
    assert (
        store.list("annotation_revisions", frame_id=frame["id"])[0]["boxes"][0]["label"] == "helmet"
    )


def test_queued_custom_checkpoint_class_changes_fail_before_loading(client, reviewed, monkeypatch):
    store, model, _ = setup_detector(client, reviewed, monkeypatch)
    frame = reviewed[1][0]
    row = inference.create_comparison(
        store,
        JobManager(store),
        frame["session_id"],
        name="Synthetic stale model",
        frame_ids=[frame["id"]],
        model_ids=[model["id"]],
    )
    model["taxonomy"]["classes"][0]["definition"] += " Changed after queuing."
    with pytest.raises(ValueError, match="definitions changed"):
        inference.run_comparison(
            store,
            row["id"],
            lambda *_: None,
            lambda: False,
            detector_factory=lambda *_: pytest.fail("No model should load"),
        )
    assert store.list("runs") == []


def test_custom_output_id_never_silently_means_coco_car(client, reviewed, monkeypatch):
    _, model, _ = setup_detector(client, reviewed, monkeypatch)
    detection = {"label": "car", "label_id": 3}
    with pytest.raises(ValueError, match="class definitions"):
        validate_output_labels([detection], model)
