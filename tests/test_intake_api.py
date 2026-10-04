"""Local intake batches preserve sources and never alter reviewed annotation state."""

import io
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
from PIL import Image
from test_datasets_custom_api import client as api_fixture
from test_datasets_custom_api import reviewed as reviewed_fixture

from iris.selection import SelectionConflict, set_selection

client = api_fixture
reviewed = reviewed_fixture


def upload(client, session, color=45, filename="synthetic.png"):
    image = io.BytesIO()
    Image.new("RGB", (32, 24), (color, 80, 120)).save(image, format="PNG")
    return client.post(
        f"/api/sessions/{session}/assets", files={"file": (filename, image.getvalue(), "image/png")}
    )


def session(client, name):
    response = client.post("/api/sessions", json={"name": name, "scene_group": name})
    assert response.status_code == 201, response.text
    return response.json()["id"]


def test_import_distinguishes_existing_source_and_recovers_after_invalid_file(client):
    identifier = session(client, "Synthetic mixed batch")
    first = upload(client, identifier)
    assert first.status_code == 201 and first.json()["import_status"] == "created"
    duplicate = upload(client, identifier, filename="copy-with-new-name.png")
    assert duplicate.status_code == 201 and duplicate.json()["import_status"] == "existing"
    assert duplicate.json()["id"] == first.json()["id"]
    failed = client.post(
        f"/api/sessions/{identifier}/assets", files={"file": ("broken.png", b"", "image/png")}
    )
    assert failed.status_code == 422
    assert upload(client, identifier, color=55).json()["import_status"] == "created"
    store = client.app.state.store
    assert len(store.list("assets", session_id=identifier)) == 2
    assert len(store.list("frames", session_id=identifier)) == 2
    assert all(not row["selected"] for row in store.list("frames", session_id=identifier))
    assert store.list("annotation_revisions") == []
    assert not list((store.root / "uploads").iterdir())
    assert all("import_status" not in row for row in store.list("assets"))


@pytest.fixture
def unselected(client):
    identifier = session(client, "Synthetic manual selection")
    for color in (15, 35):
        assert upload(client, identifier, color=color).status_code == 201
    frames = client.get(f"/api/sessions/{identifier}/frames").json()
    return identifier, frames


def batch(frames, selected=True):
    return {
        "frame_ids": [frame["id"] for frame in frames],
        "selected": selected,
        "expected_selection": {frame["id"]: bool(frame["selected"]) for frame in frames},
    }


def test_bulk_selection_is_atomic_and_stale_view_does_not_overwrite_other_tab(client, unselected):
    identifier, frames = unselected
    endpoint = f"/api/sessions/{identifier}/selection"
    assert (
        client.patch(f"/api/frames/{frames[0]['id']}", json={"selected": True}).status_code == 200
    )
    assert client.post(endpoint, json=batch(frames)).status_code == 409
    refreshed = client.get(f"/api/sessions/{identifier}/frames").json()
    assert [frame["selected"] for frame in refreshed] == [True, False]
    result = client.post(endpoint, json=batch(refreshed))
    assert result.status_code == 200 and result.json()["changed_count"] == 1
    selected = client.get(f"/api/sessions/{identifier}/frames").json()
    assert all(frame["selected"] for frame in selected)
    assert client.app.state.store.list("annotation_revisions") == []


def test_foreign_frame_rejects_entire_batch_and_project_scope_is_enforced(client, unselected):
    identifier, frames = unselected
    other_session = session(client, "Separate synthetic session")
    upload(client, other_session, color=75)
    foreign = client.get(f"/api/sessions/{other_session}/frames").json()[0]
    endpoint = f"/api/sessions/{identifier}/selection"
    assert client.post(endpoint, json=batch([frames[0], foreign])).status_code == 422
    assert all(
        not frame["selected"] for frame in client.get(f"/api/sessions/{identifier}/frames").json()
    )
    project = client.post("/api/projects", json={"name": "Separate project"}).json()
    assert (
        client.post(endpoint, params={"project_id": project["id"]}, json=batch(frames)).status_code
        == 404
    )
    assert (
        client.get(
            f"/api/sessions/{identifier}/selection-insights", params={"project_id": project["id"]}
        ).status_code
        == 404
    )


@pytest.mark.parametrize("change", ["missing", "duplicate", "foreign_expected", "not_boolean"])
def test_invalid_batch_selection_does_not_write(client, unselected, change):
    identifier, frames = unselected
    payload = batch(frames)
    if change == "missing":
        payload["expected_selection"].pop(frames[0]["id"])
    elif change == "duplicate":
        payload["frame_ids"].append(frames[0]["id"])
    elif change == "foreign_expected":
        payload["expected_selection"]["another-frame"] = False
    else:
        payload["expected_selection"][frames[0]["id"]] = "false"
    assert client.post(f"/api/sessions/{identifier}/selection", json=payload).status_code == 422
    assert all(
        not frame["selected"] for frame in client.get(f"/api/sessions/{identifier}/frames").json()
    )


def test_two_selection_writers_cannot_both_apply_a_stale_snapshot(client, unselected):
    identifier, frames = unselected
    barrier = Barrier(2)
    store = client.app.state.store

    def apply():
        barrier.wait()
        try:
            return set_selection(store, identifier, **batch(frames))
        except SelectionConflict:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _: apply(), range(2)))
    assert outcomes.count("conflict") == 1
    assert all(store.get("frames", frame["id"])["selected"] for frame in frames)


def test_insights_distinguish_reviewed_negatives_from_unreviewed_images(
    client, reviewed, unselected
):
    store = client.app.state.store
    before = {table: store.list(table) for table in store.columns}
    for index, frame in enumerate(reviewed[1]):
        response = client.get(f"/api/sessions/{frame['session_id']}/selection-insights")
        assert response.status_code == 200, response.text
        item = response.json()["frames"][0]
        assert item["review_status"] == "validated"
        assert item["negative"] is (index == 2)
    unknown = client.get(f"/api/sessions/{unselected[0]}/selection-insights").json()["frames"]
    assert all(
        item["review_status"] == "unannotated" and item["negative"] is None for item in unknown
    )
    assert {table: store.list(table) for table in store.columns} == before


def test_split_preview_is_read_only_and_its_revision_tokens_guard_freezing(client, reviewed):
    taxonomy, frames, _ = reviewed
    store = client.app.state.store
    before = {table: store.list(table) for table in store.columns}
    response = client.post("/api/datasets/plan", json={"taxonomy_id": taxonomy["id"], "seed": 7})
    assert response.status_code == 200, response.text
    plan = response.json()
    assert plan["can_freeze"], plan
    assert set(plan["frame_ids"]) == {frame["id"] for frame in frames}
    assert {table: store.list(table) for table in store.columns} == before
    assert (
        client.post("/api/datasets/plan", json={"taxonomy_id": taxonomy["id"], "seed": 7}).json()
        == plan
    )
    annotation = client.get(f"/api/frames/{frames[0]['id']}/annotation").json()
    changed = client.put(
        f"/api/frames/{frames[0]['id']}/annotation",
        json={
            "expected_revision": annotation["revision"],
            "taxonomy_id": taxonomy["id"],
            "boxes": [],
            "decisions": {},
            "status": "validated",
            "reviewer": "Synthetic changed review",
        },
    )
    assert changed.status_code == 200, changed.text
    result = client.post(
        "/api/datasets",
        json={
            "name": "Stale synthetic plan",
            "taxonomy_id": taxonomy["id"],
            "frame_ids": plan["frame_ids"],
            "splits": plan["splits"],
            "expected_revisions": plan["expected_revisions"],
        },
    )
    assert result.status_code == 409, result.text
    assert store.list("dataset_versions") == []


@pytest.mark.parametrize(
    "ratios",
    [
        {"train": 1, "val": 0, "test": 0},
        {"train": 0.8, "val": 0.3, "test": 0},
        {"train": 0.8, "val": 0.2, "other": 0},
    ],
)
def test_invalid_split_targets_are_rejected(client, reviewed, ratios):
    response = client.post(
        "/api/datasets/plan", json={"taxonomy_id": reviewed[0]["id"], "ratios": ratios}
    )
    assert response.status_code == 422, response.text
    assert client.app.state.store.list("dataset_versions") == []
