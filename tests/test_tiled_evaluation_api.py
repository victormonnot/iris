"""HTTP selection keeps checkpoint IDs distinct from full/tiled run identities."""

import pytest
import test_evaluation as evaluation_fixtures
from fastapi.testclient import TestClient
from test_evaluation import MODEL_IDS, fixture_detector

from iris.app import create_app
from iris.evaluation import run_evaluation
from iris.store import TABLES

BASE_URL = "http://127.0.0.1"
workspace = evaluation_fixtures.workspace


def payload(workspace, **overrides):
    return {
        "name": "Synthetic paired quality evaluation",
        "dataset_id": workspace[1]["id"],
        "model_ids": MODEL_IDS[:1],
        "inference_mode": "paired",
        "tile_size": 128,
        "overlap": 0.25,
        **overrides,
    }


def run(store, identifier):
    return run_evaluation(
        store, identifier, lambda *_: None, lambda: False, detector_factory=fixture_detector()
    )


def test_preview_is_read_only_and_queue_exposes_two_planned_runs(workspace):
    store = workspace[0]
    with TestClient(create_app(store.root, run_jobs=False), base_url=BASE_URL) as client:
        before = {table: store.list(table) for table in TABLES}
        response = client.post("/api/evaluations/preview", json=payload(workspace))
        assert response.status_code == 200, response.text
        plan = response.json()
        assert plan["frames_total"] == 2
        assert plan["forward_passes"] == 4 and plan["warmup_passes"] == 2
        assert plan["total_forward_passes"] == 6
        assert plan["limits"]["max_forward_passes"] == 4096
        assert {table: store.list(table) for table in TABLES} == before
        response = client.post("/api/evaluations", json=payload(workspace))
        assert response.status_code == 202, response.text
        result = response.json()
        assert result["model_ids"] == MODEL_IDS[:1]
        assert result["lanes"] == [
            {"model_id": MODEL_IDS[0], "variant": variant, "evaluation_model_id": None}
            for variant in ("full", "tiled")
        ]
        assert result["config"]["inference"] == plan["inference"]


@pytest.mark.parametrize(
    "changes",
    [
        {"inference_mode": "unsupported"},
        {"model_ids": MODEL_IDS},
        {"tile_size": True},
        {"tile_size": 0},
        {"tile_size": 2049},
        {"overlap": True},
        {"overlap": -0.01},
        {"overlap": 0.51},
        {"split": "train"},
        {"unexpected": "field"},
    ],
)
def test_invalid_pipeline_never_creates_job_or_evaluation(workspace, changes):
    store = workspace[0]
    with TestClient(create_app(store.root, run_jobs=False), base_url=BASE_URL) as client:
        for endpoint in ("/api/evaluations/preview", "/api/evaluations"):
            response = client.post(endpoint, json=payload(workspace, **changes))
            assert response.status_code == 422, response.text
    assert not store.list("evaluations") and not store.list("jobs")


def test_paired_analysis_reference_and_test_audit_preserve_selected_pipeline(workspace):
    store = workspace[0]
    with TestClient(create_app(store.root, run_jobs=False), base_url=BASE_URL) as client:
        created = client.post("/api/evaluations", json=payload(workspace)).json()
        run(store, created["id"])
        detail = client.get(f"/api/evaluations/{created['id']}").json()
        assert len(detail["models"]) == 2 and len(detail["predictions"]) == 4
        assert [lane["variant"] for lane in detail["lanes"]] == ["full", "tiled"]
        analysis = client.get(f"/api/evaluations/{created['id']}/analysis")
        assert analysis.status_code == 200, analysis.text
        value = analysis.json()
        assert value["protocol"] == "iris-error-analysis-v2"
        assert value["comparison"] == {
            "baseline_run_id": detail["lanes"][0]["evaluation_model_id"],
            "candidate_run_id": detail["lanes"][1]["evaluation_model_id"],
        }
        assert {item["model_id"] for item in value["runs"]} == {MODEL_IDS[0]}
        selection = {
            "evaluation_id": detail["id"],
            "model_id": MODEL_IDS[0],
            "reviewer": "Synthetic fixture reviewer",
            "notes": "Exercise exact pipeline choice",
            "expected_previous_id": None,
        }
        ambiguous = client.post("/api/model-references", json=selection)
        assert ambiguous.status_code == 422
        assert store.list("model_references") == []
        selected = client.post("/api/model-references", json=selection | {"variant": "tiled"})
        assert selected.status_code == 201, selected.text
        reference = selected.json()
        assert reference["model_id"] == MODEL_IDS[0]
        assert reference["metadata"]["variant"] == "tiled"
        assert (
            reference["metadata"]["evaluation_model_id"]
            == detail["lanes"][1]["evaluation_model_id"]
        )
        assert reference["metadata"]["inference"]["tile_size"] == 128
        audit_payload = payload(workspace, split="test", validation_evaluation_id=detail["id"])
        preview = client.post("/api/evaluations/preview", json=audit_payload)
        assert preview.status_code == 200, preview.text
        assert preview.json()["frames_total"] == 1
        assert preview.json()["total_forward_passes"] == 4
        for override in ({"inference_mode": "full"}, {"tile_size": 256}, {"overlap": 0.2}):
            changed = client.post("/api/evaluations", json=audit_payload | override)
            assert changed.status_code == 422, changed.text
        audit = client.post("/api/evaluations", json=audit_payload)
        assert audit.status_code == 202, audit.text
        run(store, audit.json()["id"])
        assert (
            client.get(f"/api/evaluations/{audit.json()['id']}").json()["job"]["status"]
            == "succeeded"
        )
        rejected = client.post(
            "/api/model-references",
            json=selection
            | {
                "evaluation_id": audit.json()["id"],
                "variant": "full",
                "expected_previous_id": reference["id"],
            },
        )
        assert rejected.status_code == 422
        assert client.get("/api/model-references").json()["current"] == reference
