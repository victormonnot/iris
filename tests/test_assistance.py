"""End-to-end annotation API and job persistence with explicit provider fixtures."""

import io
from copy import deepcopy

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from iris.annotations import AnnotationConflict, get_annotation, save_annotation
from iris.app import create_app
from iris.assistance import request_assistance, run_assistance
from iris.assistance_provider import ProviderResponseError
from iris.store import Store

READY = {
    "provider": "ollama",
    "endpoint": "http://127.0.0.1:11434",
    "model": "fixture-only",
    "model_digest": "a" * 64,
    "status": "ready",
}
BOX = {"id": "manual box with arbitrary ID", "label": "person", "box": [2, 3, 30, 35]}


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    app = create_app(tmp_path / "workspace", run_jobs=False)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        session = client.post(
            "/api/sessions", json={"name": "Fixture", "scene_group": "test"}
        ).json()
        image = io.BytesIO()
        Image.new("RGB", (80, 60), (25, 35, 45)).save(image, format="PNG")
        assert (
            client.post(
                f"/api/sessions/{session['id']}/assets",
                files={"file": ("synthetic.png", image.getvalue(), "image/png")},
            ).status_code
            == 201
        )
        (frame,) = client.get(f"/api/sessions/{session['id']}/frames").json()
        monkeypatch.setattr("iris.assistance.provider_status", lambda: deepcopy(READY))
        yield client, app.state.store, app.state.jobs, frame


class FixtureReviewer:
    """Fixture only: categorization is predetermined, never a model result."""

    def __init__(self, config):
        self.metadata = {**READY, "fixture": True}

    def review(self, image, candidates, instructions=""):
        assert image.size == (80, 60)
        assert len(candidates) <= 8
        return {
            "reviews": [
                {"candidate_id": c["id"], "label": "car", "reason": "Test fixture"}
                for c in candidates
            ],
            "scene_notes": "Synthetic test only",
            "prompt": "fixture prompt: " + instructions,
            "metadata": self.metadata,
            "raw_response": {"fixture": True, "done": True},
        }


def prepare(workspace, **changes):
    client, store, jobs, frame = workspace
    annotation = save_annotation(
        store,
        frame["id"],
        expected_revision=0,
        boxes=[BOX],
        decisions={},
        status="validated",
        reviewer="Test fixture",
    )
    job = request_assistance(store, jobs, frame["id"], expected_revision=1, **changes)
    return annotation, job, job["params"]["assistance_id"]


def run(store, record_id, *, factory=FixtureReviewer, cancelled=lambda: False):
    return run_assistance(store, record_id, lambda *_: None, cancelled, reviewer_factory=factory)


def test_api_manual_review_history_negative_and_conflict_survive_restart(workspace):
    client, store, _, frame = workspace
    endpoint = f"/api/frames/{frame['id']}/annotation"
    assert client.get(endpoint).json()["status"] == "unannotated"
    payload = {"expected_revision": 0, "boxes": [BOX], "decisions": {}}
    draft = client.put(endpoint, json=payload)
    assert draft.status_code == 200, draft.text
    assert draft.json()["status"] == "draft"
    assert client.put(endpoint, json=payload).status_code == 409
    payload.update(expected_revision=1, status="validated")
    assert client.put(endpoint, json=payload).status_code == 422  # reviewer is required
    payload.update(reviewer="Test fixture", boxes=[])
    negative = client.put(endpoint, json=payload).json()
    assert negative["status"] == "validated" and negative["boxes"] == []
    assert len(negative["history"]) == 2
    historic = client.get(endpoint + "/revisions/1").json()
    assert historic["boxes"][0]["box"] == BOX["box"]
    assert historic["status"] == "draft"
    assert client.get(endpoint + "/revisions/99").status_code == 404
    with TestClient(
        create_app(store.root, run_jobs=False), base_url="http://127.0.0.1"
    ) as reopened:
        assert reopened.get(endpoint).json() == negative
    assert client.get("/api/frames/missing/annotation").status_code == 404
    assert (
        client.post("/api/frames/missing/assist", json={"expected_revision": 0}).status_code == 404
    )


def test_assistance_keeps_validated_revision_and_requires_separate_human_review(workspace):
    client, store, _, frame = workspace
    original, job, record_id = prepare(workspace, instructions="Test notes")
    record = store.get("assistance_records", record_id)
    assert record["candidates"][0]["id"] == "annotation-0"
    assert record["config"]["model_digest"] == READY["model_digest"]
    assert record["config"]["base_revision"] == 1
    result = run(store, record_id)
    assert result["suggestions_created"] == 1
    after = get_annotation(store, frame["id"])
    assert after["boxes"] == original["boxes"]
    assert after["revision"] == 1 and after["status"] == "validated"
    (suggestion,) = after["suggestions"]
    assert suggestion["state"] == "pending"
    assert suggestion["label"] == "car" and suggestion["box"] == BOX["box"]
    assert suggestion["metadata"]["target_box_id"] == BOX["id"]
    assert suggestion["metadata"]["recommendation"] == "change"
    record = client.get(f"/api/frames/{frame['id']}/assistance").json()[0]
    assert record["job"]["id"] == job["id"]
    assert record["raw_response"]["fixture"] and record["prompt"].endswith("Test notes")
    assert Store(store.root).get("assistance_records", record_id) == {
        key: value for key, value in record.items() if key != "job"
    }
    with pytest.raises(ValueError, match="pending"):
        save_annotation(
            store,
            frame["id"],
            expected_revision=1,
            boxes=after["boxes"],
            decisions={},
            status="validated",
            reviewer="Test fixture",
        )
    # Explicit acceptance, using the original ID, replaces the manual box's origin.
    reviewed = save_annotation(
        store,
        frame["id"],
        expected_revision=1,
        boxes=[{**BOX, "label": "car", "suggestion_id": suggestion["id"]}],
        decisions={suggestion["id"]: "accepted"},
        status="validated",
        reviewer="Test fixture",
    )
    assert reviewed["revision"] == 2 and reviewed["boxes"][0]["source"]["kind"] == "multimodal"
    with pytest.raises(ValueError, match="immutable"):
        run(store, record_id)


def test_pending_request_is_unique_and_can_be_cancelled_then_retried(workspace):
    client, store, jobs, frame = workspace
    _, job, record_id = prepare(workspace)
    endpoint = f"/api/frames/{frame['id']}/assist"
    assert client.post(endpoint, json={"expected_revision": 1}).status_code == 409
    assert jobs.cancel(job["id"])["status"] == "cancelled"
    assert run(store, record_id, cancelled=lambda: True)["cancelled"] is True
    assert not store.list("annotation_suggestions")
    assert client.post(endpoint, json={"expected_revision": 1}).status_code == 202


def test_failed_provider_preserves_raw_output_and_never_writes_annotations(workspace):
    _, store, _, frame = workspace
    original, _, record_id = prepare(workspace)

    class BrokenReviewer(FixtureReviewer):
        def review(self, *args, **kwargs):
            raise ProviderResponseError(
                "Malformed fixture",
                raw_response={"bad": "fixture"},
                metadata=self.metadata,
                prompt="retained fixture prompt",
            )

    with pytest.raises(ProviderResponseError, match="Malformed"):
        run(store, record_id, factory=BrokenReviewer)
    record = store.get("assistance_records", record_id)
    assert record["raw_response"] == {"bad": "fixture"}
    assert record["error"] == "Malformed fixture"
    assert record["prompt"] == "retained fixture prompt"
    assert get_annotation(store, frame["id"]) == original


def test_cancelled_response_is_retained_without_publishing_proposals(workspace):
    _, store, _, _ = workspace
    _, _, record_id = prepare(workspace)
    stopped = False

    class CancellingReviewer(FixtureReviewer):
        def review(self, *args, **kwargs):
            nonlocal stopped
            result = super().review(*args, **kwargs)
            stopped = True
            return result

    assert (
        run(store, record_id, factory=CancellingReviewer, cancelled=lambda: stopped)["cancelled"]
        is True
    )
    assert store.get("assistance_records", record_id)["raw_response"]["fixture"]
    assert not store.list("annotation_suggestions")


def test_changed_checkpoint_or_pixels_blocks_request_before_review(workspace):
    _, store, _, frame = workspace
    _, _, record_id = prepare(workspace)

    class WrongModel(FixtureReviewer):
        def __init__(self, config):
            super().__init__(config)
            self.metadata["model_digest"] = "b" * 64

        def review(self, *args, **kwargs):
            pytest.fail("Changed model must not execute")

    with pytest.raises(ValueError, match="model changed"):
        run(store, record_id, factory=WrongModel)
    source = store.get("frames", frame["id"])
    Image.new("RGB", (80, 60), "red").save(store.artifact_path(source["path"]))
    with pytest.raises(ValueError, match="hash|pixels|changed"):
        run(store, record_id)
    assert not store.list("annotation_suggestions")


def test_provider_failure_does_not_block_manual_annotation(workspace, monkeypatch):
    client, store, jobs, frame = workspace
    save_annotation(store, frame["id"], expected_revision=0, boxes=[BOX], decisions={})
    monkeypatch.setattr(
        "iris.assistance.provider_status",
        lambda: {
            "status": "unavailable",
            "reason": "Fixture missing dependency",
        },
    )
    assert (
        client.post(f"/api/frames/{frame['id']}/assist", json={"expected_revision": 1}).status_code
        == 409
    )
    assert not store.list("jobs")
    updated = save_annotation(
        store,
        frame["id"],
        expected_revision=1,
        boxes=[BOX],
        decisions={},
        status="validated",
        reviewer="Test fixture",
    )
    assert updated["status"] == "validated"


@pytest.mark.parametrize("count", [0, 9])
def test_candidate_limit_checked_before_provider_or_job(workspace, count, monkeypatch):
    _, store, jobs, frame = workspace
    save_annotation(
        store,
        frame["id"],
        expected_revision=0,
        decisions={},
        boxes=[{**BOX, "id": f"box-{i}"} for i in range(count)],
    )
    monkeypatch.setattr("iris.assistance.provider_status", lambda: pytest.fail("No provider call"))
    with pytest.raises(ValueError, match="1–8"):
        request_assistance(store, jobs, frame["id"], expected_revision=1)
    assert not store.list("jobs")


def test_revision_changed_while_probing_provider_does_not_enqueue(workspace, monkeypatch):
    _, store, jobs, frame = workspace
    save_annotation(store, frame["id"], expected_revision=0, boxes=[BOX], decisions={})

    def changed():
        save_annotation(store, frame["id"], expected_revision=1, boxes=[BOX], decisions={})
        return READY

    monkeypatch.setattr("iris.assistance.provider_status", changed)
    with pytest.raises(AnnotationConflict, match="changed"):
        request_assistance(store, jobs, frame["id"], expected_revision=1)
    assert not store.list("jobs")
