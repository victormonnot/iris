"""Experiment HTTP contracts with saved synthetic evaluation outputs, no model calls."""

import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from test_evaluation_api import evaluate_fixture
from test_evaluation_api import ready_models as ready_models
from test_training_api import BASE_URL, prepare_dataset

from iris.app import create_app


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path / "workspace", run_jobs=False), base_url=BASE_URL) as api:
        yield api


@pytest.fixture
def completed(client, ready_models):
    dataset, _ = prepare_dataset(client)
    return evaluate_fixture(client, dataset["id"])


def create(client, completed, **changes):
    payload = {
        "evaluation_id": completed["id"],
        "title": "Synthetic experiment, no field quality claim",
        "objective": "Exercise report persistence on saved fixture results.",
        "conclusion": "Software check only.",
        "example_frame_ids": [completed["frames"][0]["frame_id"]],
    }
    payload.update(changes)
    return client.post("/api/experiments", json=payload)


def original_records(store):
    tables = (
        "sessions",
        "frames",
        "jobs",
        "dataset_versions",
        "training_runs",
        "trained_models",
        "evaluations",
        "evaluation_models",
        "evaluation_predictions",
        "model_references",
        "annotation_revisions",
    )
    return {table: store.list(table) for table in tables}


def test_preview_create_read_edit_image_export_and_reopen(client, completed):
    store = client.app.state.store
    before = original_records(store)
    preview = client.get(f"/api/evaluations/{completed['id']}/experiment-preview")
    assert preview.status_code == 200, preview.text
    assert client.get("/api/experiments").json() == []
    assert len(preview.json()["available_examples"]) == len(completed["frames"])
    response = create(client, completed)
    assert response.status_code == 201, response.text
    report = response.json()
    assert report["revision"] == 1
    assert report["snapshot"]["evaluation"]["id"] == completed["id"]
    assert len(report["snapshot"]["examples"]) == 1
    assert len(client.get("/api/experiments").json()) == 1
    assert client.get(f"/api/experiments/{report['id']}").json() == report
    image = client.get(report["images"][0]["url"])
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/jpeg"
    assert hashlib.sha256(image.content).hexdigest() == report["images"][0]["sha256"]
    assert "path" not in report["images"][0]
    endpoint = f"/api/experiments/{report['id']}"
    text_export = client.get(endpoint + "/export?expected_revision=1")
    assert text_export.status_code == 200, text_export.text
    assert text_export.headers["content-type"].startswith("text/html")
    assert text_export.headers["cache-control"] == "no-store"
    assert "attachment;" in text_export.headers["content-disposition"]
    assert "script-src 'none'" in text_export.headers["content-security-policy"]
    assert "img-src data:" in text_export.headers["content-security-policy"]
    assert "data:image/jpeg;base64," not in text_export.text
    picture_export = client.get(endpoint + "/export?expected_revision=1&include_images=true")
    assert picture_export.status_code == 200, picture_export.text
    assert "data:image/jpeg;base64," in picture_export.text
    edited = client.patch(
        endpoint,
        json={
            "expected_revision": 1,
            "title": "Revised interpretation",
            "objective": report["objective"],
            "conclusion": "Still a synthetic fixture, never a field benchmark.",
        },
    )
    assert edited.status_code == 200, edited.text
    changed = edited.json()
    assert changed["revision"] == 2
    assert changed["snapshot"] == report["snapshot"]
    assert changed["snapshot_sha256"] == report["snapshot_sha256"]
    assert client.get(endpoint + "/export?expected_revision=1").status_code == 409
    assert client.get(endpoint + "/export?expected_revision=2").status_code == 200
    assert original_records(store) == before
    with TestClient(create_app(store.root, run_jobs=False), base_url=BASE_URL) as reopened:
        assert reopened.get(endpoint).json() == changed
        assert reopened.get(report["images"][0]["url"]).content == image.content
    assert client.get("/api/system").json()["capabilities"]["experiment_reports"] is True


@pytest.mark.parametrize(
    "changes",
    [
        {"title": " "},
        {"title": "x" * 161},
        {"objective": "x" * 4001},
        {"conclusion": "x" * 4001},
        {"example_frame_ids": ["absent"]},
        {"example_frame_ids": ["x"] * 7},
        {"example_frame_ids": [True]},
        {"example_frame_ids": "invalid"},
        {"evaluation_id": None},
        {"unknown": "field"},
    ],
)
def test_invalid_create_does_not_write_a_report(client, completed, changes):
    before = original_records(client.app.state.store)
    response = create(client, completed, **changes)
    assert response.status_code == 422, response.text
    assert client.get("/api/experiments").json() == []
    assert original_records(client.app.state.store) == before


@pytest.mark.parametrize("status", ["queued", "running", "failed", "cancelled", "interrupted"])
def test_incomplete_evaluation_cannot_be_reported(client, completed, status):
    client.app.state.store.update("jobs", completed["job_id"], {"status": status})
    response = client.get(f"/api/evaluations/{completed['id']}/experiment-preview")
    assert response.status_code == 409, response.text
    assert create(client, completed).status_code in {409, 422}
    assert client.get("/api/experiments").json() == []


def test_conflicting_note_edits_and_readonly_result_fields(client, completed):
    report = create(client, completed).json()
    url = f"/api/experiments/{report['id']}"
    payload = {"expected_revision": 1, "title": "First editor", "objective": "", "conclusion": ""}
    assert client.patch(url, json=payload).status_code == 200
    assert client.patch(url, json={**payload, "title": "Stale editor"}).status_code == 409
    assert (
        client.patch(url, json={**payload, "expected_revision": 2, "snapshot": {}}).status_code
        == 422
    )
    assert client.patch(url, json={**payload, "expected_revision": True}).status_code == 422
    assert client.get(url).json()["title"] == "First editor"


def test_unknown_records_and_export_query_validation(client, completed):
    assert client.get("/api/evaluations/missing/experiment-preview").status_code == 404
    assert client.get("/api/experiments/missing").status_code == 404
    assert client.get("/api/experiments/missing/export?expected_revision=1").status_code == 404
    report = create(client, completed).json()
    url = f"/api/experiments/{report['id']}"
    assert client.get(url + "/images/missing").status_code == 404
    for query in ("", "?expected_revision=0", "?expected_revision=1&include_images=perhaps"):
        assert client.get(url + "/export" + query).status_code == 422


def test_create_rejects_cross_origin_requests(client, completed):
    response = client.post(
        "/api/experiments",
        headers={"Origin": "https://example.org"},
        json={
            "evaluation_id": completed["id"],
            "title": "Must not be created",
        },
    )
    assert response.status_code == 403
    assert client.get("/api/experiments").json() == []


def test_export_concurrency_limit_and_recovery(client, completed, monkeypatch):
    import iris.app as app_module

    report = create(client, completed).json()
    url = f"/api/experiments/{report['id']}/export?expected_revision=1"
    original = app_module.render_experiment_html
    started, finish = threading.Event(), threading.Event()

    def delayed(*args, **kwargs):
        started.set()
        assert finish.wait(10)
        return original(*args, **kwargs)

    monkeypatch.setattr(app_module, "render_experiment_html", delayed)
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(client.get, url)
        try:
            assert started.wait(5)
            assert client.get(url).status_code == 409
        finally:
            finish.set()
        assert first.result(timeout=10).status_code == 200
    assert client.get(url).status_code == 200


@pytest.mark.parametrize("kind,status", [("limit", 413), ("corrupt", 409), ("io", 409)])
def test_failed_export_releases_slot_and_hides_local_path(
    client, completed, monkeypatch, kind, status
):
    import iris.app as app_module
    from iris.experiment_export import ExperimentExportLimitError

    report = create(client, completed).json()
    url = f"/api/experiments/{report['id']}/export?expected_revision=1"
    original = app_module.render_experiment_html
    failure = {
        "limit": ExperimentExportLimitError("Synthetic limit"),
        "corrupt": ValueError("Synthetic corrupt record"),
        "io": OSError("private-file-path-must-not-leak"),
    }[kind]

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(app_module, "render_experiment_html", fail)
    response = client.get(url)
    assert response.status_code == status, response.text
    assert "private-file-path-must-not-leak" not in response.text
    monkeypatch.setattr(app_module, "render_experiment_html", original)
    assert client.get(url).status_code == 200
