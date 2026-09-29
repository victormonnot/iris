"""Detector-assisted review and job publication, using synthetic fixtures only."""

from copy import deepcopy

import pytest
from PIL import Image
from test_annotations import prediction
from test_assistance import FixtureReviewer, run
from test_assistance import workspace as workspace

from iris.annotations import get_annotation
from iris.assistance import request_assistance


def queue_predictions(workspace, *, threshold=0.1):
    _, store, jobs, frame = workspace
    source = prediction(store, frame)
    job = request_assistance(
        store,
        jobs,
        frame["id"],
        expected_revision=0,
        prediction_id=source["id"],
        threshold=threshold,
        instructions="Synthetic detector review",
    )
    return source, job["params"]["assistance_id"]


def test_detector_review_uses_explicit_mapping_and_frozen_candidate_provenance(workspace):
    client, store, _, frame = workspace
    source, record_id = queue_predictions(workspace)
    record = store.get("assistance_records", record_id)
    assert [item["label"] for item in record["candidates"]] == ["person", "car"]
    assert [item["box"] for item in record["candidates"]] == [
        detection["box"] for detection in source["detections"][:2]
    ]
    assert record["config"]["base_revision"] == 0
    assert record["config"]["prediction_id"] == source["id"]
    assert record["config"]["frame_sha256"] == frame["sha256"]
    assert record["config"]["threshold"] == 0.1
    run_metadata = store.get("runs", source["run_id"])["metadata"]
    for index, candidate in enumerate(record["candidates"]):
        assert candidate["source"] == {
            "kind": "prediction",
            "prediction_id": source["id"],
            "detection_index": index,
            "score": source["detections"][index]["score"],
            "model_id": source["model_id"],
            "run_metadata": run_metadata,
        }
    result = run(store, record_id)
    assert result["suggestions_created"] == 2
    annotation = client.get(f"/api/frames/{frame['id']}/annotation").json()
    assert annotation["revision"] == 0
    assert annotation["status"] == "unannotated" and annotation["boxes"] == []
    assert {proposal["state"] for proposal in annotation["suggestions"]} == {"pending"}
    assert {proposal["metadata"]["recommendation"] for proposal in annotation["suggestions"]} == {
        "keep",
        "change",
    }
    for proposal in annotation["suggestions"]:
        original = next(
            item
            for item in record["candidates"]
            if item["id"] == proposal["metadata"]["candidate_id"]
        )
        assert proposal["metadata"]["source"] == original["source"]
        assert proposal["box"] == original["box"]
        assert "target_box_id" not in proposal["metadata"]


def test_api_can_request_detector_review_without_any_manual_annotation(workspace):
    client, store, _, frame = workspace
    source = prediction(store, frame)
    response = client.post(
        f"/api/frames/{frame['id']}/assist",
        json={"expected_revision": 0, "prediction_id": source["id"], "threshold": 0.5},
    )
    assert response.status_code == 202, response.text
    record = store.get("assistance_records", response.json()["params"]["assistance_id"])
    assert len(record["candidates"]) == 1
    assert record["candidates"][0]["label"] == "person"
    assert store.list("annotation_revisions") == []


@pytest.mark.parametrize(
    "change",
    ["taxonomy", "dimensions", "model", "frame_selection", "hash", "category", "confidence", "box"],
)
def test_incompatible_prediction_is_rejected_before_provider_or_job(workspace, monkeypatch, change):
    client, store, _, frame = workspace
    source = prediction(store, frame)
    comparison = store.get("comparisons", source["comparison_id"])
    if change in {"taxonomy", "hash"}:
        config = deepcopy(comparison["config"])
        if change == "taxonomy":
            config["taxonomy"] = "different-class-order"
        else:
            config["frame_hashes"][frame["id"]] = "b" * 64
        store.update("comparisons", comparison["id"], {"config": config})
    elif change == "frame_selection":
        store.update("comparisons", comparison["id"], {"frame_ids": []})
    elif change == "dimensions":
        store.update("predictions", source["id"], {"input_size": [60, 80]})
    elif change == "model":
        store.update("predictions", source["id"], {"model_id": "unrelated-model"})
    else:
        detections = deepcopy(source["detections"])
        if change == "category":
            detections[0]["label_id"] = True
        elif change == "confidence":
            detections[0]["score"] = 1.1
        else:
            detections[0]["box"] = [-1, 0, 30, 40]
        store.update("predictions", source["id"], {"detections": detections})
    monkeypatch.setattr("iris.assistance.provider_status", lambda: pytest.fail("No provider call"))
    response = client.post(
        f"/api/frames/{frame['id']}/assist",
        json={"expected_revision": 0, "prediction_id": source["id"]},
    )
    assert response.status_code == 422, response.text
    assert store.list("assistance_records") == []
    assert store.list("jobs", kind="assist") == []


@pytest.mark.parametrize("count", [0, 9])
def test_detector_candidates_obey_batch_bound_before_provider_lookup(workspace, monkeypatch, count):
    client, store, _, frame = workspace
    source = prediction(
        store,
        frame,
        detections=[{"label_id": 1, "label": "person", "box": [2, 3, 30, 45], "score": 0.9}]
        * count,
    )
    monkeypatch.setattr("iris.assistance.provider_status", lambda: pytest.fail("No provider call"))
    response = client.post(
        f"/api/frames/{frame['id']}/assist",
        json={"expected_revision": 0, "prediction_id": source["id"]},
    )
    assert response.status_code == 422, response.text
    assert store.list("assistance_records") == []


def test_none_and_uncertain_are_reviewable_recommendations_and_keep_candidate_geometry(workspace):
    _, store, _, frame = workspace
    _, record_id = queue_predictions(workspace)

    class RecommendationReviewer(FixtureReviewer):
        def review(self, *args, **kwargs):
            result = super().review(*args, **kwargs)
            for review, label in zip(result["reviews"], ["none", "uncertain"], strict=True):
                review["label"] = label
            return result

    run(store, record_id, factory=RecommendationReviewer)
    annotation = get_annotation(store, frame["id"])
    assert annotation["boxes"] == [] and annotation["status"] == "unannotated"
    candidates = store.get("assistance_records", record_id)["candidates"]
    for suggestion in annotation["suggestions"]:
        index = suggestion["metadata"]["source"]["detection_index"]
        assert suggestion["box"] == candidates[index]["box"]
        assert suggestion["label"] == candidates[index]["label"]
        assert suggestion["metadata"]["recommendation"] == ["reject", "uncertain"][index]
        assert suggestion["state"] == "pending"


@pytest.mark.parametrize("malformed", ["missing", "duplicate", "foreign", "unsupported_label"])
def test_malformed_review_never_partially_publishes_but_preserves_raw_response(
    workspace, malformed
):
    _, store, _, frame = workspace
    _, record_id = queue_predictions(workspace)

    class MalformedReviewer(FixtureReviewer):
        def review(self, *args, **kwargs):
            result = super().review(*args, **kwargs)
            if malformed == "missing":
                result["reviews"].pop()
            elif malformed == "duplicate":
                result["reviews"][1]["candidate_id"] = result["reviews"][0]["candidate_id"]
            elif malformed == "foreign":
                result["reviews"][1]["candidate_id"] = "not-this-request"
            else:
                # Valid first review is inserted, then the whole transaction must roll back.
                result["reviews"][1]["label"] = "unsupported"
            return result

    with pytest.raises(ValueError):
        run(store, record_id, factory=MalformedReviewer)
    assert store.list("annotation_suggestions") == []
    assert get_annotation(store, frame["id"])["status"] == "unannotated"
    record = store.get("assistance_records", record_id)
    assert record["raw_response"] == {"fixture": True, "done": True}
    assert record["prompt"] == "fixture prompt: Synthetic detector review"
    assert record["error"]


def test_frame_changed_during_review_blocks_publication_and_retains_response(workspace):
    _, store, _, frame = workspace
    _, record_id = queue_predictions(workspace)

    class FrameChangingReviewer(FixtureReviewer):
        def review(self, *args, **kwargs):
            result = super().review(*args, **kwargs)
            original = store.get("frames", frame["id"])
            Image.new("RGB", (80, 60), "red").save(store.artifact_path(original["path"]))
            return result

    with pytest.raises(ValueError, match="hash"):
        run(store, record_id, factory=FrameChangingReviewer)
    assert store.list("annotation_suggestions") == []
    record = store.get("assistance_records", record_id)
    assert record["raw_response"]["fixture"]
    assert record["error"]


def test_cancellation_during_final_frame_verification_prevents_publication(workspace, monkeypatch):
    from iris import assistance

    _, store, _, _ = workspace
    _, record_id = queue_predictions(workspace)
    verify = assistance._load_verified_frame
    calls = 0
    stopped = False

    def checked(*args, **kwargs):
        nonlocal calls, stopped
        result = verify(*args, **kwargs)
        calls += 1
        if calls == 2:
            stopped = True
        return result

    monkeypatch.setattr(assistance, "_load_verified_frame", checked)
    result = run(store, record_id, cancelled=lambda: stopped)
    assert result["cancelled"] is True
    assert result["suggestions_created"] == 0
    assert store.list("annotation_suggestions") == []
    assert store.get("assistance_records", record_id)["raw_response"]["fixture"]
