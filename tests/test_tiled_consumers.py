"""Saved full-image/tiled variants remain distinct throughout human review.

Images, detector outputs and the local reviewer status are explicit fixtures;
these checks never run a model or contact an annotation service.
"""

from copy import deepcopy

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from iris.annotations import AnnotationConflict, add_detector_suggestions, get_annotation
from iris.app import create_app
from iris.assistance import _candidates
from iris.assistance_batches import create_batch, preview_batch
from iris.media import import_asset
from iris.review_queue import review_queue
from iris.store import new_id, now

MODEL = "fixture-checkpoint"
READY = {
    "provider": "ollama",
    "endpoint": "http://127.0.0.1:11434",
    "model": "fixture-reviewer",
    "model_digest": "a" * 64,
    "status": "ready",
}


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr("iris.assistance_batches.provider_status", lambda *_: deepcopy(READY))
    app = create_app(tmp_path / "workspace", run_jobs=False)
    store = app.state.store
    session = store.insert(
        "sessions",
        {"id": new_id(), "name": "Fixture", "scene_group": "fixture", "created_at": now()},
    )
    source = tmp_path / "fixture.png"
    Image.new("RGB", (800, 400), (20, 60, 100)).save(source)
    asset = import_asset(store, session["id"], source, source.name)
    frame = store.list("frames", asset_id=asset["id"])[0]
    frame = store.update("frames", frame["id"], {"selected": True})
    job = store.insert(
        "jobs",
        {"id": new_id(), "kind": "infer", "status": "succeeded", "params": {}, "created_at": now()},
    )
    comparison = store.insert(
        "comparisons",
        {
            "id": new_id(),
            "name": "Same checkpoint, different inference",
            "session_id": session["id"],
            "frame_ids": [frame["id"]],
            "model_ids": [MODEL],
            "config": {
                "taxonomy": "coco-2017-v1",
                "frame_hashes": {frame["id"]: frame["sha256"]},
                "inference": {"mode": "paired"},
                "lanes": [
                    {"model_id": MODEL, "variant": "full"},
                    {"model_id": MODEL, "variant": "tiled"},
                ],
            },
            "job_id": job["id"],
            "created_at": now(),
        },
    )
    predictions = {}
    for variant in ("full", "tiled"):
        run = store.insert(
            "runs",
            {
                "id": new_id(),
                "comparison_id": comparison["id"],
                "model_id": MODEL,
                "variant": variant,
                "metadata": {
                    "model_id": MODEL,
                    "fixture": True,
                    "inference": {"variant": variant},
                },
                "created_at": now(),
            },
        )
        detections = [{"box": [20, 30, 40, 70], "label_id": 1, "score": 0.9}]
        if variant == "tiled":
            detections.append({"box": [610, 80, 630, 100], "label_id": 3, "score": 0.8})
        predictions[variant] = store.insert(
            "predictions",
            {
                "id": new_id(),
                "comparison_id": comparison["id"],
                "run_id": run["id"],
                "model_id": MODEL,
                "frame_id": frame["id"],
                "detections": detections,
                "timing": {},
                "input_size": [800, 400],
                "created_at": now(),
            },
        )
    with TestClient(app, base_url="http://127.0.0.1") as client:
        yield app, client, store, session, frame, comparison, predictions


def batch_options(workspace, **changes):
    _, _, _, _, frame, comparison, _ = workspace
    return {
        "frame_ids": [frame["id"]],
        "source": "comparison",
        "comparison_id": comparison["id"],
        "detector_model_id": MODEL,
        "model": READY["model"],
        **changes,
    }


def test_paired_review_uses_ordered_runs_without_duplicate_model_ids(workspace):
    _, _, store, session, frame, comparison, predictions = workspace
    result = review_queue(store, session["id"], comparison_id=comparison["id"])
    assert result["comparison"]["model_ids"] == [MODEL]
    assert [lane["variant"] for lane in result["comparison"]["lanes"]] == ["full", "tiled"]
    assert [lane["run_id"] for lane in result["comparison"]["lanes"]] == [
        predictions[variant]["run_id"] for variant in ("full", "tiled")
    ]
    signal = result["frames"][0]["signal"]
    assert signal["status"] == "disagreement"
    assert signal["counts"] == [1, 2]
    assert signal["prediction_ids"] == [predictions[key]["id"] for key in ("full", "tiled")]
    assert result["frames"][0]["id"] == frame["id"]
    assert not store.list("annotation_revisions")


def test_missing_tiled_prediction_is_not_replaced_by_full_prediction(workspace):
    _, _, store, session, _, comparison, predictions = workspace
    with store.connect() as conn:
        conn.execute("DELETE FROM predictions WHERE id=?", (predictions["tiled"]["id"],))
    result = review_queue(store, session["id"], comparison_id=comparison["id"])
    assert result["frames"][0]["signal"]["status"] == "unavailable"
    assert result["frames"][0]["signal"]["prediction_ids"] == []


def test_review_rejects_inconsistent_variant_provenance(workspace):
    _, _, store, session, _, comparison, predictions = workspace
    run = store.get("runs", predictions["tiled"]["run_id"])
    store.update(
        "runs", run["id"], {"metadata": {**run["metadata"], "inference": {"variant": "full"}}}
    )
    result = review_queue(store, session["id"], comparison_id=comparison["id"])
    assert result["frames"][0]["signal"]["status"] == "unavailable"
    assert "inference mode" in result["frames"][0]["signal"]["reason"]


def test_annotation_sources_and_proposals_retain_variant_and_original_coordinates(workspace):
    _, _, store, _, frame, _, predictions = workspace
    document = get_annotation(store, frame["id"])
    sources = {source["variant"]: source for source in document["prediction_sources"]}
    assert set(sources) == {"full", "tiled"}
    assert sources["tiled"]["id"] == predictions["tiled"]["id"]
    assert sources["tiled"]["model_id"] == sources["full"]["model_id"] == MODEL
    result = add_detector_suggestions(
        store, frame["id"], prediction_id=predictions["tiled"]["id"], expected_revision=0
    )
    assert result["revision"] == 0
    assert result["status"] == "unannotated"
    suggestions = store.list("annotation_suggestions", frame_id=frame["id"])
    car = next(suggestion for suggestion in suggestions if suggestion["label"] == "car")
    assert car["box"] == [610, 80, 630, 100]
    assert car["metadata"]["model_metadata"]["inference"]["variant"] == "tiled"
    assert car["metadata"]["run_id"] == predictions["tiled"]["run_id"]


def test_multimodal_candidates_use_mapped_tiled_boxes_and_provenance(workspace):
    _, _, store, _, frame, _, predictions = workspace
    candidates = _candidates(
        store, frame, {"revision": 0, "boxes": []}, predictions["tiled"]["id"], 0.5
    )
    assert [candidate["box"] for candidate in candidates] == [
        [20, 30, 40, 70],
        [610, 80, 630, 100],
    ]
    assert all(candidate["source"]["model_id"] == MODEL for candidate in candidates)
    assert all(
        candidate["source"]["run_metadata"]["inference"]["variant"] == "tiled"
        for candidate in candidates
    )


def test_batch_requires_explicit_variant_for_paired_source(workspace):
    _, _, store, session, _, _, _ = workspace
    with pytest.raises(ValueError, match="explicit.*mode"):
        preview_batch(store, session["id"], **batch_options(workspace))
    assert not store.list("assistance_batches")
    assert not store.list("assistance_records")


@pytest.mark.parametrize("variant,count", [("full", 1), ("tiled", 2)])
def test_batch_freezes_selected_variant_without_mixing_predictions(workspace, variant, count):
    app, _, store, session, _, _, predictions = workspace
    options = batch_options(workspace, detector_variant=variant)
    preview = preview_batch(store, session["id"], **options)
    assert preview["config"]["detector_variant"] == variant
    assert preview["eligible_count"] == 1
    assert preview["frames"][0]["prediction_id"] == predictions[variant]["id"]
    assert preview["frames"][0]["candidate_count"] == count
    batch = create_batch(
        store,
        app.state.jobs,
        session["id"],
        name="Variant review fixture",
        expected_fingerprint=preview["fingerprint"],
        **options,
    )
    assert batch["config"]["detector_model_id"] == MODEL
    assert batch["config"]["detector_variant"] == variant
    record = store.list("assistance_records")[0]
    assert record["config"]["prediction_id"] == predictions[variant]["id"]
    assert len(record["candidates"]) == count


def test_single_tiled_variant_resolves_implicitly(workspace):
    _, _, store, session, _, comparison, _ = workspace
    config = deepcopy(comparison["config"])
    config["inference"]["mode"] = "tiled"
    config["lanes"] = [{"model_id": MODEL, "variant": "tiled"}]
    store.update("comparisons", comparison["id"], {"config": config})
    preview = preview_batch(store, session["id"], **batch_options(workspace))
    assert preview["config"]["detector_variant"] == "tiled"
    assert preview["frames"][0]["candidate_count"] == 2


def test_stale_preview_cannot_switch_variant_silently(workspace):
    app, _, store, session, _, _, _ = workspace
    preview = preview_batch(
        store, session["id"], **batch_options(workspace, detector_variant="full")
    )
    with pytest.raises(AnnotationConflict, match="preview changed"):
        create_batch(
            store,
            app.state.jobs,
            session["id"],
            name="Wrong variant",
            expected_fingerprint=preview["fingerprint"],
            **batch_options(workspace, detector_variant="tiled"),
        )
    assert not store.list("assistance_records")


def test_removed_selected_variant_is_not_replaced_by_other_run(workspace):
    _, _, store, session, _, _, predictions = workspace
    with store.connect() as conn:
        conn.execute("DELETE FROM predictions WHERE id=?", (predictions["tiled"]["id"],))
    preview = preview_batch(
        store, session["id"], **batch_options(workspace, detector_variant="tiled")
    )
    assert preview["eligible_count"] == 0
    assert preview["frames"][0]["prediction_id"] is None


@pytest.mark.parametrize("invalid", ["crop", "", 1, True, ["full"]])
def test_invalid_batch_variant_rejected(workspace, invalid):
    _, _, store, session, _, _, _ = workspace
    with pytest.raises(ValueError, match="inference mode"):
        preview_batch(store, session["id"], **batch_options(workspace, detector_variant=invalid))


def test_annotation_source_rejects_irrelevant_variant(workspace):
    _, _, store, session, frame, _, _ = workspace
    with pytest.raises(ValueError, match="Comparison fields"):
        preview_batch(
            store,
            session["id"],
            frame_ids=[frame["id"]],
            source="annotations",
            model=READY["model"],
            detector_variant="full",
        )


def test_batch_http_accepts_explicit_variant_and_rejects_ambiguity(workspace):
    _, client, _, session, _, _, predictions = workspace
    endpoint = f"/api/sessions/{session['id']}/assistance-batches/preview"
    response = client.post(endpoint, json=batch_options(workspace, detector_variant="tiled"))
    assert response.status_code == 200, response.text
    assert response.json()["frames"][0]["prediction_id"] == predictions["tiled"]["id"]
    response = client.post(endpoint, json=batch_options(workspace))
    assert response.status_code == 422, response.text
    assert "explicit" in response.json()["detail"]
