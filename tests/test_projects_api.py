"""Project isolation through real HTTP handlers; synthetic media, no model calls."""

import io
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from test_datasets import add_frame
from test_evaluation_analysis import record_evaluation

from iris.app import create_app
from iris.datasets import create_dataset
from iris.experiments import create_experiment
from iris.store import DEFAULT_PROJECT_ID, new_id, now

BASE = "http://127.0.0.1"


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path / "workspace", run_jobs=False), base_url=BASE) as api:
        yield api


def project(client, name="Objects on a desk"):
    response = client.post("/api/projects", json={"name": name, "description": "Synthetic test"})
    assert response.status_code == 201, response.text
    return response.json()


def session(client, project_id, name="Session"):
    response = client.post(
        "/api/sessions",
        params={"project_id": project_id},
        json={"name": name, "scene_group": "scene-one"},
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_projects_creation_validation_persistence_and_legacy_default(client):
    initial = client.get("/api/projects").json()
    assert len(initial) == 1 and initial[0]["id"] == DEFAULT_PROJECT_ID
    p = project(client, "  Desk objects  ")
    assert p["name"] == "Desk objects" and p["taxonomy_id"] == "iris-objects-v1"
    assert client.get(f"/api/projects/{p['id']}").json() == p
    s = session(client, p["id"])
    assert s["project_id"] == p["id"]
    assert client.get("/api/sessions").json() == []
    for payload in ({"name": " "}, {"name": "x", "taxonomy_id": "other"}, {"name": 4}):
        assert client.post("/api/projects", json=payload).status_code == 422
    assert len(client.get("/api/projects").json()) == 2
    assert client.get("/api/sessions?project_id=missing").status_code == 404
    assert client.get("/api/sessions?project_id=default&project_id=missing").status_code == 422
    with TestClient(
        create_app(client.app.state.store.root, run_jobs=False), base_url=BASE
    ) as reopened:
        assert reopened.get(f"/api/sessions?project_id={p['id']}").json() == [s]
        assert reopened.get("/api/projects").json() == [*initial, p]


def test_project_media_and_mutations_cannot_cross_scope(client):
    a, b = project(client, "A"), project(client, "B")
    sa, sb = session(client, a["id"], "A session"), session(client, b["id"], "B session")
    image = io.BytesIO()
    Image.new("RGB", (32, 24), (21, 35, 69)).save(image, format="PNG")
    endpoint = f"/api/sessions/{sa['id']}/assets"
    wrong = client.post(
        endpoint, params={"project_id": b["id"]}, files={"file": ("x.png", image.getvalue())}
    )
    assert wrong.status_code == 404
    response = client.post(
        endpoint, params={"project_id": a["id"]}, files={"file": ("x.png", image.getvalue())}
    )
    assert response.status_code == 201, response.text
    asset = response.json()
    frame = client.app.state.store.list("frames", asset_id=asset["id"])[0]
    paths = [
        f"/api/sessions/{sa['id']}",
        endpoint,
        f"/api/sessions/{sa['id']}/frames",
        f"/api/sessions/{sa['id']}/review-queue",
        f"/api/frames/{frame['id']}/image",
        f"/api/frames/{frame['id']}/annotation",
        f"/api/assets/{asset['id']}/media",
    ]
    for path in paths:
        assert client.get(path, params={"project_id": a["id"]}).status_code == 200, path
        assert client.get(path, params={"project_id": b["id"]}).status_code == 404, path
        assert client.get(path).status_code == 404, path
    assert (
        client.patch(
            f"/api/frames/{frame['id']}?project_id={b['id']}", json={"selected": True}
        ).status_code
        == 404
    )
    assert client.app.state.store.get("frames", frame["id"])["selected"] is False
    assert client.get(f"/api/sessions?project_id={b['id']}").json() == [sb]


def fixture_release(client, tmp_path, project_id, color):
    store = client.app.state.store
    frames = [
        add_frame(store, tmp_path, group=group, color=(color, 20 + i, 30))
        for i, group in enumerate(("train", "val"))
    ]
    for frame in frames:
        store.update("sessions", frame["session_id"], {"project_id": project_id})
    return create_dataset(
        store,
        name="Synthetic release",
        frame_ids=[f["id"] for f in frames],
        splits={"train": "train", "val": "val"},
        project_id=project_id,
    )


def test_project_datasets_evaluations_reports_jobs_and_references_are_separate(client, tmp_path):
    store = client.app.state.store
    a, b = project(client, "A"), project(client, "B")
    data = {}
    for p, color in ((a, 50), (b, 60)):
        ds = fixture_release(client, tmp_path, p["id"], color)
        ev = record_evaluation(store, ds, {"fixture": [[]]}, model_ids=["fixture"])
        report = create_experiment(
            store, evaluation_id=ev["id"], title="Synthetic report", example_frame_ids=[]
        )
        data[p["id"]] = (ds, ev, report)
        store.insert(
            "model_references",
            {
                "id": new_id(),
                "evaluation_id": ev["id"],
                "model_id": "fixture",
                "reviewer": "Fixture reviewer",
                "notes": "Synthetic reference, no model execution",
                "metadata": {},
                "created_at": now(),
            },
        )
    for own, other in ((a, b), (b, a)):
        ds, ev, report = data[own["id"]]
        for path, expected in (
            ("datasets", ds["id"]),
            ("evaluations", ev["id"]),
            ("experiments", report["id"]),
            ("jobs", ev["job"]["id"]),
        ):
            rows = client.get(f"/api/{path}?project_id={own['id']}").json()
            assert [row["id"] for row in rows] == [expected]
        assert (
            client.get(f"/api/model-references?project_id={own['id']}").json()["current"][
                "evaluation_id"
            ]
            == ev["id"]
        )
        for path in (
            f"/api/datasets/{ds['id']}/manifest",
            f"/api/datasets/{ds['id']}/export/coco",
            f"/api/evaluations/{ev['id']}",
            f"/api/experiments/{report['id']}",
            f"/api/experiments/{report['id']}/export?expected_revision=1",
        ):
            assert (
                client.get(
                    path + ("&" if "?" in path else "?") + "project_id=" + other["id"]
                ).status_code
                == 404
            )
        assert (
            client.post(f"/api/jobs/{ev['job']['id']}/cancel?project_id={other['id']}").status_code
            == 404
        )


def test_trained_models_are_project_owned_and_official_weights_shared(
    client, tmp_path, monkeypatch
):
    store = client.app.state.store
    a, b = project(client, "A"), project(client, "B")
    ds_a = fixture_release(client, tmp_path, a["id"], 70)
    ds_b = fixture_release(client, tmp_path, b["id"], 80)
    job_id, run_id, model_id = new_id(), new_id(), new_id()
    store.insert(
        "jobs",
        {
            "id": job_id,
            "kind": "train",
            "status": "succeeded",
            "params": {"training_id": run_id},
            "created_at": now(),
        },
    )
    store.insert(
        "training_runs",
        {
            "id": run_id,
            "name": "Fixture",
            "dataset_id": ds_a["id"],
            "parent_model_id": "official",
            "config": {},
            "job_id": job_id,
            "created_at": now(),
        },
    )
    store.insert(
        "trained_models",
        {
            "id": model_id,
            "name": "Fixture",
            "training_id": run_id,
            "parent_model_id": "official",
            "architecture": "fixture",
            "path": "models/fixture",
            "weight_sha256": "a" * 64,
            "metadata": {},
            "created_at": now(),
        },
    )
    monkeypatch.setattr("iris.app.catalog", lambda root: [{"id": "official"}, {"id": model_id}])
    assert client.get(f"/api/models?project_id={a['id']}").json() == [
        {"id": "official"},
        {"id": model_id},
    ]
    assert client.get(f"/api/models?project_id={b['id']}").json() == [{"id": "official"}]
    for endpoint in ("/api/trainings", "/api/trainings/preview"):
        result = client.post(
            endpoint,
            params={"project_id": b["id"]},
            json={
                "name": "Invalid cross-project parent",
                "dataset_id": ds_b["id"],
                "parent_model_id": model_id,
            },
        )
        assert result.status_code == 404
    assert [r["id"] for r in client.get(f"/api/trainings?project_id={a['id']}").json()] == [run_id]
    assert client.get(f"/api/trainings?project_id={b['id']}").json() == []


def test_concurrent_requests_keep_their_own_project_scope(client):
    projects = [project(client, name) for name in ("A", "B")]
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(
            pool.map(lambda i: session(client, projects[i % 2]["id"], f"Session {i}"), range(12))
        )
    for p in projects:
        expected = {row["id"] for row in rows if row["project_id"] == p["id"]}
        actual = client.get("/api/sessions", params={"project_id": p["id"]}).json()
        assert {row["id"] for row in actual} == expected
    assert client.get("/api/sessions").json() == []
