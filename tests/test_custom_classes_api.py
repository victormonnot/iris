"""Custom classes through HTTP, using synthetic images and no model calls."""

import io
from copy import deepcopy

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from test_coco_import_api import config, package

from iris.app import create_app
from iris.taxonomies import TAXONOMY

HELMET = {"id": "helmet", "name": "Safety helmet", "definition": "A visible protective helmet."}


@pytest.fixture
def client(tmp_path):
    with TestClient(
        create_app(tmp_path / "workspace", run_jobs=False), base_url="http://127.0.0.1"
    ) as api:
        yield api


def publish(client, classes=None, expected=TAXONOMY["id"], project="default"):
    response = client.post(
        f"/api/projects/{project}/taxonomies",
        json={"expected_taxonomy_id": expected, "classes": classes or [HELMET]},
    )
    assert response.status_code == 201, response.text
    return response.json()


def intake(client, project="default", color=35):
    params = {"project_id": project}
    session = client.post(
        "/api/sessions", params=params, json={"name": "Synthetic image", "scene_group": str(color)}
    ).json()
    image = io.BytesIO()
    Image.new("RGB", (40, 30), (color, 60, 90)).save(image, format="PNG")
    response = client.post(
        f"/api/sessions/{session['id']}/assets",
        params=params,
        files={"file": ("synthetic.png", image.getvalue(), "image/png")},
    )
    assert response.status_code == 201, response.text
    (frame,) = client.get(f"/api/sessions/{session['id']}/frames", params=params).json()
    assert (
        client.patch(
            f"/api/frames/{frame['id']}", params=params, json={"selected": True}
        ).status_code
        == 200
    )
    return frame


def save(client, frame, taxonomy, revision=0, boxes=None, status="validated", decisions=None):
    return client.put(
        f"/api/frames/{frame['id']}/annotation",
        json={
            "expected_revision": revision,
            "taxonomy_id": taxonomy["id"],
            "status": status,
            "reviewer": "Synthetic test; not human ground truth",
            "boxes": boxes
            if boxes is not None
            else [{"id": "box", "label": "helmet", "box": [2, 3, 20, 22]}],
            "decisions": decisions or {},
        },
    )


def test_custom_class_annotation_definition_history_adoption_and_restart(client):
    first = publish(client)
    frame = intake(client)
    endpoint = f"/api/frames/{frame['id']}/annotation"
    assert client.get(endpoint).json()["taxonomy"] == first
    response = save(client, frame, first)
    assert response.status_code == 200, response.text
    original = deepcopy(client.app.state.store.list("annotation_revisions"))
    second = publish(
        client, [{**HELMET, "definition": "Only a helmet worn on a head."}], first["id"]
    )
    old = client.get(endpoint).json()
    assert old["status"] == "validated" and old["taxonomy"] == first
    assert old["taxonomy_outdated"] and old["current_taxonomy"] == second
    response = client.post(
        endpoint + "/taxonomy",
        json={
            "expected_revision": 1,
            "expected_taxonomy_id": first["id"],
            "target_taxonomy_id": second["id"],
        },
    )
    assert response.status_code == 200, response.text
    adopted = response.json()
    assert adopted["status"] == "draft" and adopted["reviewer"] == ""
    assert adopted["boxes"] == old["boxes"] and adopted["taxonomy"] == second
    assert client.app.state.store.get("annotation_revisions", original[0]["id"]) == original[0]
    assert client.get(endpoint + "/revisions/1").json()["taxonomy"] == first
    assert save(client, frame, first, revision=2).status_code == 409
    assert save(client, frame, second, revision=2).status_code == 200
    with TestClient(
        create_app(client.app.state.store.root, run_jobs=False), base_url="http://127.0.0.1"
    ) as restarted:
        assert restarted.get(endpoint).json()["taxonomy"] == second


def test_publish_compare_and_swap_and_project_isolation(client):
    custom = publish(client)
    stale = client.post(
        "/api/projects/default/taxonomies",
        json={
            "expected_taxonomy_id": TAXONOMY["id"],
            "classes": [HELMET],
        },
    )
    assert stale.status_code == 409
    other = client.post("/api/projects", json={"name": "Independent project"}).json()
    versions = client.get(f"/api/projects/{other['id']}/taxonomies").json()
    assert versions["current_taxonomy_id"] == TAXONOMY["id"]
    assert custom["id"] not in [version["id"] for version in versions["versions"]]
    assert client.get("/api/projects/missing/taxonomies").status_code == 404
    frame = intake(client, other["id"])
    endpoint = f"/api/frames/{frame['id']}/annotation"
    assert client.get(endpoint).status_code == 404
    assert client.post(
        endpoint + "/taxonomy",
        params={"project_id": other["id"]},
        json={
            "expected_revision": 0,
            "expected_taxonomy_id": TAXONOMY["id"],
            "target_taxonomy_id": custom["id"],
        },
    ).status_code in (409, 422)


def test_intake_pins_builtin_and_empty_adoption_requires_human_validation(client):
    frame = intake(client)
    custom = publish(client)
    endpoint = f"/api/frames/{frame['id']}/annotation"
    assert client.get(endpoint).json()["taxonomy"] == TAXONOMY
    adopted = client.post(
        endpoint + "/taxonomy",
        json={
            "expected_revision": 0,
            "expected_taxonomy_id": TAXONOMY["id"],
            "target_taxonomy_id": custom["id"],
        },
    )
    assert adopted.status_code == 200, adopted.text
    assert adopted.json()["status"] == "draft"
    response = save(client, frame, custom, revision=1, boxes=[])
    assert response.status_code == 200 and response.json()["status"] == "validated"


def test_custom_coco_mapping_is_frozen_at_preview_and_never_auto_validated(client):
    first = publish(client)
    preview = client.post(
        "/api/dataset-imports", files={"file": ("synthetic.zip", package(), "application/zip")}
    ).json()
    assert preview["taxonomy"] == first
    second = publish(client, [{**HELMET, "name": "Worn helmet"}], first["id"])
    payload = config()
    payload["category_mapping"] = {"42": "helmet", "8": "exclude"}
    endpoint = f"/api/dataset-imports/{preview['id']}/commit"
    response = client.post(endpoint, json=payload)
    assert response.status_code == 201, response.text
    result = response.json()
    assert client.post(endpoint, json=payload).json() == result
    frame = client.app.state.store.get("frames", result["frame_ids"][0])
    annotation = client.get(f"/api/frames/{frame['id']}/annotation").json()
    assert annotation["taxonomy"] == first and annotation["current_taxonomy"] == second
    assert annotation["status"] == "unannotated" and annotation["revision"] == 0
    (suggestion,) = annotation["suggestions"]
    assert suggestion["label"] == "helmet" and suggestion["state"] == "pending"
    assert suggestion["metadata"]["source_category"]["name"] == "Person"
    assert suggestion["metadata"]["target_taxonomy"] == first["id"]
    assert save(client, frame, first, boxes=[]).status_code == 422
    reviewed = save(
        client,
        frame,
        first,
        boxes=[
            {
                "id": "imported",
                "label": "helmet",
                "box": suggestion["box"],
                "suggestion_id": suggestion["id"],
            }
        ],
        decisions={suggestion["id"]: "accepted"},
    )
    assert reviewed.status_code == 200, reviewed.text
    candidates = client.get("/api/dataset-candidates").json()
    assert candidates["groups"] == []
    assert candidates["excluded"]["different_taxonomy"] == 1
    earlier = client.get("/api/dataset-candidates", params={"taxonomy_id": first["id"]}).json()
    assert sum(group["count"] for group in earlier["groups"]) == 1


def test_import_negative_image_and_legacy_preview_keep_their_saved_classes(client):
    legacy = client.post(
        "/api/dataset-imports", files={"file": ("synthetic.zip", package(22, negative=True))}
    ).json()
    custom = publish(client)
    response = client.post(f"/api/dataset-imports/{legacy['id']}/commit", json=config())
    assert response.status_code == 201, response.text
    frame_id = response.json()["frame_ids"][0]
    assert client.get(f"/api/frames/{frame_id}/annotation").json()["taxonomy"] == TAXONOMY
    preview = client.post(
        "/api/dataset-imports", files={"file": ("synthetic.zip", package(33, negative=True))}
    ).json()
    payload = config(group="negative-custom", split=None)
    payload["category_mapping"] = {"42": "helmet", "8": "exclude"}
    response = client.post(f"/api/dataset-imports/{preview['id']}/commit", json=payload)
    assert response.status_code == 201, response.text
    frame = {"id": response.json()["frame_ids"][0]}
    assert save(client, frame, custom, boxes=[]).status_code == 200


def test_custom_import_rejects_implicit_or_unknown_mappings_without_writes(client):
    publish(client)
    preview = client.post(
        "/api/dataset-imports", files={"file": ("synthetic.zip", package())}
    ).json()
    for mapping in (
        {"42": "person", "8": "exclude"},
        {"42": "helmet"},
        {"42": "missing", "8": "exclude"},
    ):
        response = client.post(
            f"/api/dataset-imports/{preview['id']}/commit",
            json={**config(), "category_mapping": mapping},
        )
        assert response.status_code == 422, response.text
    assert client.get("/api/sessions").json() == []
    assert client.app.state.store.list("annotation_suggestions") == []


def test_custom_assistance_fails_before_provider_or_job_creation(client, monkeypatch):
    custom = publish(client)
    frame = intake(client)
    assert save(client, frame, custom).status_code == 200

    def unexpected(*args, **kwargs):
        pytest.fail("Custom classes must fail before contacting a provider")

    monkeypatch.setattr("iris.assistance.provider_status", unexpected)
    monkeypatch.setattr("iris.remote_provider.provider_status", unexpected)
    for suffix, payload in (("assist", {}), ("assist/preview", {"provider": "alibaba"})):
        response = client.post(
            f"/api/frames/{frame['id']}/{suffix}", json={"expected_revision": 1, **payload}
        )
        assert response.status_code == 422, response.text
        assert "custom" in response.text.lower()
    assert client.app.state.store.list("jobs") == []
