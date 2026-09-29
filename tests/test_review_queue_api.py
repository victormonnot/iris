"""Read-only review queue contracts using synthetic images and saved detector fixtures."""

import io
import sqlite3

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from iris.app import create_app
from iris.store import new_id, now

BASE_URL = "http://127.0.0.1"


@pytest.fixture
def client(tmp_path):
    with TestClient(
        create_app(tmp_path / "workspace", run_jobs=False), base_url=BASE_URL
    ) as client:
        yield client


def session_with_frames(client, *, name="Synthetic review queue", count=3, color_base=40):
    response = client.post("/api/sessions", json={"name": name, "scene_group": name})
    assert response.status_code == 201, response.text
    session = response.json()
    for index in range(count):
        output = io.BytesIO()
        Image.new("RGB", (64, 48), (color_base + index, 70, 100)).save(output, format="PNG")
        response = client.post(
            f"/api/sessions/{session['id']}/assets",
            files={"file": (f"synthetic-{index}.png", output.getvalue(), "image/png")},
        )
        assert response.status_code == 201, response.text
    frames = client.get(f"/api/sessions/{session['id']}/frames").json()
    for frame in frames:
        client.patch(f"/api/frames/{frame['id']}", json={"selected": True})
    return session, frames


def detection(box, label_id=1, score=0.8):
    return {
        "box": box,
        "label_id": label_id,
        "label": "not trusted for class mapping",
        "score": score,
    }


def saved_comparison(store, session, frames, pairs):
    created = now()
    job_id, comparison_id = new_id(), new_id()
    model_ids = ["synthetic-detector-a", "synthetic-detector-b"]
    store.insert(
        "jobs",
        {
            "id": job_id,
            "kind": "infer",
            "status": "succeeded",
            "params": {"fixture": True},
            "created_at": created,
        },
    )
    comparison = store.insert(
        "comparisons",
        {
            "id": comparison_id,
            "session_id": session["id"],
            "name": "Synthetic saved predictions",
            "frame_ids": [frame["id"] for frame in frames],
            "model_ids": model_ids,
            "config": {
                "taxonomy": "coco-2017-v1",
                "frame_hashes": {frame["id"]: frame["sha256"] for frame in frames},
            },
            "job_id": job_id,
            "created_at": created,
        },
    )
    for side, model_id in enumerate(model_ids):
        run_id = new_id()
        store.insert(
            "runs",
            {
                "id": run_id,
                "comparison_id": comparison_id,
                "model_id": model_id,
                "metadata": {"name": model_id, "fixture": True},
                "created_at": created,
            },
        )
        for frame, pair in zip(frames, pairs, strict=True):
            if pair[side] is None:
                continue
            store.insert(
                "predictions",
                {
                    "id": new_id(),
                    "comparison_id": comparison_id,
                    "run_id": run_id,
                    "frame_id": frame["id"],
                    "model_id": model_id,
                    "detections": pair[side],
                    "timing": {},
                    "input_size": [frame["width"], frame["height"]],
                    "created_at": created,
                },
            )
    return comparison


def queue(client, session, **params):
    response = client.get(f"/api/sessions/{session['id']}/review-queue", params=params)
    assert response.status_code == 200, response.text
    return response.json()


def fixture_annotation(client, frame, status="validated", expected_revision=0):
    response = client.put(
        f"/api/frames/{frame['id']}/annotation",
        json={
            "expected_revision": expected_revision,
            "status": status,
            "reviewer": "Automated synthetic fixture; not human ground truth",
            "boxes": [],
            "decisions": {},
        },
    )
    assert response.status_code == 200, response.text


def dump(store):
    with sqlite3.connect(store.db_path) as conn:
        return list(conn.iterdump())


def test_review_counts_track_latest_revision_and_new_proposals(client):
    session, frames = session_with_frames(client)
    result = queue(client, session)
    assert result["counts"] == {
        "total": 3,
        "needs_review": 3,
        "unannotated": 3,
        "draft": 0,
        "pending": 0,
        "validated": 0,
    }
    fixture_annotation(client, frames[0])
    fixture_annotation(client, frames[1], "draft")
    result = queue(client, session)
    assert result["counts"]["validated"] == 1
    assert result["counts"]["draft"] == 1
    store = client.app.state.store
    store.insert(
        "annotation_suggestions",
        {
            "id": new_id(),
            "frame_id": frames[0]["id"],
            "kind": "imported",
            "label": "person",
            "box": [1, 2, 12, 20],
            "metadata": {"fixture": True},
            "created_at": now(),
        },
    )
    result = queue(client, session)
    assert result["counts"]["validated"] == 0
    assert result["counts"]["pending"] == 1
    item = next(item for item in result["frames"] if item["id"] == frames[0]["id"])
    assert item["annotation_status"] == "validated" and item["review_status"] == "pending"
    assert item["revision"] == 1 and item["pending_count"] == 1
    assert item["box_count"] == 0


def test_queue_selection_changes_do_not_change_review_records(client):
    session, frames = session_with_frames(client)
    fixture_annotation(client, frames[0])
    for frame in frames:
        client.patch(f"/api/frames/{frame['id']}", json={"selected": False})
    result = queue(client, session)
    assert result["frames"] == [] and result["counts"]["total"] == 0
    client.patch(f"/api/frames/{frames[0]['id']}", json={"selected": True})
    assert queue(client, session)["counts"]["validated"] == 1


def test_geometry_disagreement_empty_and_missing_are_distinct(client):
    session, frames = session_with_frames(client)
    comparison = saved_comparison(
        client.app.state.store,
        session,
        frames,
        [
            ([detection([1, 2, 10, 20])], [detection([30, 2, 40, 20])]),
            ([], []),
            ([], None),
        ],
    )
    result = queue(client, session, comparison_id=comparison["id"])
    by_id = {item["id"]: item for item in result["frames"]}
    signal = by_id[frames[0]["id"]]["signal"]
    assert signal["counts"] == [1, 1] and signal["unmatched_counts"] == [1, 1]
    assert signal["disagreement"] == 1
    assert signal["status"] == "disagreement"
    assert by_id[frames[1]["id"]]["signal"]["status"] == "no_detections"
    assert by_id[frames[1]["id"]]["signal"]["disagreement"] is None
    assert by_id[frames[2]["id"]]["signal"]["status"] == "unavailable"
    assert by_id[frames[2]["id"]]["signal"]["disagreement"] is None
    assert result["counts"]["validated"] == 0


def test_requested_thresholds_control_signal_and_original_indices(client):
    session, frames = session_with_frames(client, count=1)
    comparison = saved_comparison(
        client.app.state.store,
        session,
        frames,
        [
            (
                [detection([1, 2, 10, 20], score=0.2), detection([30, 2, 40, 20], 3, 0.7)],
                [detection([30, 2, 40, 20], 3, 0.7)],
            ),
        ],
    )
    result = queue(
        client, session, comparison_id=comparison["id"], confidence_threshold=0.5, iou_threshold=1
    )
    signal = result["frames"][0]["signal"]
    assert signal["status"] == "agreement"
    assert signal["matches"][0]["left_index"] == 1
    assert signal["matches"][0]["right_index"] == 0
    assert result["config"]["iou_threshold"] == 1
    lowered = queue(client, session, comparison_id=comparison["id"], confidence_threshold=0.1)
    assert lowered["frames"][0]["signal"]["disagreement"] == pytest.approx(1 / 3)


def test_saved_queue_is_read_only_and_survives_restart(client):
    session, frames = session_with_frames(client, count=1)
    store = client.app.state.store
    comparison = saved_comparison(store, session, frames, [([], [])])
    before = dump(store)
    result = queue(client, session, comparison_id=comparison["id"])
    assert dump(store) == before
    with TestClient(create_app(store.root, run_jobs=False), base_url=BASE_URL) as restarted:
        assert queue(restarted, session, comparison_id=comparison["id"]) == result
    assert dump(store) == before


def test_source_split_and_pixels_are_not_reassigned_by_queue(client):
    session, frames = session_with_frames(client, count=1)
    store = client.app.state.store
    asset = store.get("assets", frames[0]["asset_id"])
    store.update(
        "assets",
        asset["id"],
        {"metadata": {**asset["metadata"], "dataset_import": {"source_split": "test"}}},
    )
    before = dump(store)
    result = queue(client, session)
    assert result["frames"][0]["reserved_split"] == "test"
    assert dump(store) == before


@pytest.mark.parametrize(
    "params",
    [
        {"confidence_threshold": -0.1},
        {"confidence_threshold": 1.1},
        {"confidence_threshold": "NaN"},
        {"confidence_threshold": "inf"},
        {"iou_threshold": 0},
        {"iou_threshold": 1.1},
        {"iou_threshold": "NaN"},
        {"iou_threshold": "not-a-number"},
    ],
)
def test_invalid_thresholds_do_not_run_inference(client, params):
    session, _ = session_with_frames(client, count=0)
    response = client.get(f"/api/sessions/{session['id']}/review-queue", params=params)
    assert response.status_code == 422, response.text
    assert client.get("/api/jobs").json() == []


def test_missing_foreign_and_incomplete_comparisons_are_rejected(client):
    session, frames = session_with_frames(client, count=1)
    other, _ = session_with_frames(client, name="Other scene", count=0)
    comparison = saved_comparison(client.app.state.store, session, frames, [([], [])])
    assert client.get("/api/sessions/missing/review-queue").status_code == 404
    assert (
        client.get(
            f"/api/sessions/{session['id']}/review-queue", params={"comparison_id": "missing"}
        ).status_code
        == 404
    )
    assert (
        client.get(
            f"/api/sessions/{other['id']}/review-queue", params={"comparison_id": comparison["id"]}
        ).status_code
        == 422
    )
    client.app.state.store.update("jobs", comparison["job_id"], {"status": "failed"})
    assert (
        client.get(
            f"/api/sessions/{session['id']}/review-queue",
            params={"comparison_id": comparison["id"]},
        ).status_code
        == 422
    )
