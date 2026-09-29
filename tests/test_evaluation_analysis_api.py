"""Saved error-analysis HTTP contracts with explicitly synthetic detector outputs."""

from copy import deepcopy

import pytest
from fastapi.testclient import TestClient
from test_evaluation_api import MODEL_IDS, evaluate_fixture
from test_evaluation_api import ready_models as ready_models
from test_training_api import BASE_URL, prepare_dataset

from iris.app import create_app


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path / "workspace", run_jobs=False), base_url=BASE_URL) as api:
        yield api


def test_analysis_matches_saved_counts_and_preserves_all_records(client, ready_models):
    dataset, _ = prepare_dataset(client)
    detail = evaluate_fixture(client, dataset["id"])
    store = client.app.state.store
    tables = (
        "evaluations",
        "evaluation_models",
        "evaluation_predictions",
        "jobs",
        "model_references",
    )
    before = {name: store.list(name) for name in tables}
    response = client.get(f"/api/evaluations/{detail['id']}/analysis")
    assert response.status_code == 200, response.text
    analysis = response.json()
    assert analysis["evaluation_id"] == detail["id"]
    assert analysis["protocol"] == "iris-error-analysis-v1"
    assert analysis["comparison"] == {
        "baseline_model_id": MODEL_IDS[0],
        "candidate_model_id": MODEL_IDS[1],
    }
    assert analysis["summary"]["all"]["changes"] == {"new_misses": 2, "recovered": 0, "fp_delta": 0}
    assert analysis["summary"]["person"]["changes"]["new_misses"] == 1
    assert analysis["summary"]["car"]["changes"]["new_misses"] == 1
    for row in detail["models"]:
        counts = analysis["summary"]["all"]["models"][row["model_id"]]
        assert all(counts[key] == row["metrics"]["summary"][key] for key in ("tp", "fp", "fn"))
    assert {name: store.list(name) for name in tables} == before
    assert client.get("/api/system").json()["capabilities"]["evaluation_analysis"] is True


def test_analysis_unknown_evaluation_is_not_found(client):
    assert client.get("/api/evaluations/missing/analysis").status_code == 404


@pytest.mark.parametrize("status", ["queued", "running", "failed", "cancelled", "interrupted"])
def test_incomplete_evaluations_have_no_analysis(client, ready_models, status):
    dataset, _ = prepare_dataset(client)
    detail = evaluate_fixture(client, dataset["id"])
    client.app.state.store.update("jobs", detail["job_id"], {"status": status})
    response = client.get(f"/api/evaluations/{detail['id']}/analysis")
    assert response.status_code == 409
    assert "summary" not in response.json()


def test_single_model_has_errors_but_no_paired_claim(client, ready_models):
    dataset, _ = prepare_dataset(client)
    detail = evaluate_fixture(client, dataset["id"], model_ids=MODEL_IDS[1:])
    response = client.get(f"/api/evaluations/{detail['id']}/analysis")
    assert response.status_code == 200, response.text
    analysis = response.json()
    assert analysis["comparison"] is None
    assert analysis["summary"]["all"]["changes"] is None
    assert analysis["summary"]["all"]["models"][MODEL_IDS[1]]["fn"] == 2


def test_saved_analysis_survives_live_draft_and_workspace_reopen(client, ready_models):
    dataset, _ = prepare_dataset(client)
    detail = evaluate_fixture(client, dataset["id"])
    endpoint = f"/api/evaluations/{detail['id']}/analysis"
    response = client.get(endpoint)
    assert response.status_code == 200, response.text
    frame = detail["frames"][0]
    edit = client.put(
        f"/api/frames/{frame['frame_id']}/annotation",
        json={
            "expected_revision": frame["revision"],
            "status": "draft",
            "boxes": [],
            "decisions": {},
            "notes": "Automated synthetic edit after evaluation",
        },
    )
    assert edit.status_code == 200
    with TestClient(
        create_app(client.app.state.store.root, run_jobs=False), base_url=BASE_URL
    ) as reopened:
        assert reopened.get(endpoint).json() == response.json()


def test_missing_saved_prediction_is_not_an_empty_prediction(client, ready_models):
    dataset, _ = prepare_dataset(client)
    detail = evaluate_fixture(client, dataset["id"])
    with client.app.state.store.connect() as conn:
        conn.execute(
            "DELETE FROM evaluation_predictions WHERE id=?", (detail["predictions"][0]["id"],)
        )
    response = client.get(f"/api/evaluations/{detail['id']}/analysis")
    assert response.status_code == 409


def test_invalid_saved_error_index_returns_conflict(client, ready_models):
    dataset, _ = prepare_dataset(client)
    detail = evaluate_fixture(client, dataset["id"])
    second = next(row for row in detail["models"] if row["model_id"] == MODEL_IDS[1])
    metrics = deepcopy(second["metrics"])
    metrics["frames"][0]["false_negatives"][0] = 99999
    client.app.state.store.update("evaluation_models", second["id"], {"metrics": metrics})
    assert client.get(f"/api/evaluations/{detail['id']}/analysis").status_code == 409


def test_test_audit_analysis_does_not_promote_a_reference(client, ready_models):
    dataset, _ = prepare_dataset(client)
    validation = evaluate_fixture(client, dataset["id"])
    audit = evaluate_fixture(
        client, dataset["id"], split="test", validation_evaluation_id=validation["id"]
    )
    response = client.get(f"/api/evaluations/{audit['id']}/analysis")
    assert response.status_code == 200, response.text
    assert response.json()["split"] == "test"
    assert any("test" in warning.lower() for warning in response.json()["warnings"])
    assert client.get("/api/model-references").json()["current"] is None
