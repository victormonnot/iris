"""Versioned class definitions remain independent, immutable and race-safe."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Barrier

import pytest

from iris.projects import create_project, project_records, record_project
from iris.store import DEFAULT_PROJECT_ID, Store
from iris.taxonomies import (
    TAXONOMY,
    TaxonomyConflict,
    current_taxonomy,
    get_taxonomy,
    list_taxonomies,
    publish_taxonomy,
)

HELMET = {"id": "helmet", "name": "Helmet", "definition": "A visible protective helmet."}


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "workspace")


def publish(store, classes, expected=TAXONOMY["id"], project=DEFAULT_PROJECT_ID):
    return publish_taxonomy(store, project, expected_taxonomy_id=expected, classes=classes)


def test_builtin_payload_is_unchanged_and_returned_independently(store):
    from iris.annotations import TAXONOMY as annotation_taxonomy

    assert json.dumps(TAXONOMY) == json.dumps(annotation_taxonomy)
    # Pinned from the schema-13 builtin, independent of the compatibility reexport.
    assert (
        hashlib.sha256(
            json.dumps(TAXONOMY, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        == "1385bec7d06184fbc7cd7942d0f402cde20c6805f437084bcd1c54077b871484"
    )
    assert get_taxonomy(store, TAXONOMY["id"]) == TAXONOMY
    snapshot = current_taxonomy(store, DEFAULT_PROJECT_ID)
    snapshot["classes"][0]["definition"] = "Local edit"
    assert current_taxonomy(store, DEFAULT_PROJECT_ID) == TAXONOMY
    assert set(TAXONOMY) == {"id", "classes", "box_format", "review_guidance"}


def test_publish_preserves_versions_allows_first_replacement_and_keeps_stable_ids(store):
    old = get_taxonomy(store, TAXONOMY["id"])
    first = publish(store, [HELMET])
    assert first["version"] == 2 and first["parent_id"] == TAXONOMY["id"]
    edited = {**HELMET, "name": "Protective helmet", "definition": "Only worn helmets."}
    extra = {"id": "bicycle", "name": "Bicycle", "definition": "A bicycle.", "coco_id": 2}
    second = publish(store, [edited, extra], first["id"])
    assert second["version"] == 3 and second["parent_id"] == first["id"]
    assert current_taxonomy(store, DEFAULT_PROJECT_ID) == second
    assert get_taxonomy(store, first["id"]) == first
    assert get_taxonomy(store, TAXONOMY["id"]) == old
    assert [item["version"] for item in list_taxonomies(store, DEFAULT_PROJECT_ID)] == [1, 2, 3]
    assert list_taxonomies(Store(store.root), DEFAULT_PROJECT_ID)[1:] == [first, second]
    with pytest.raises(ValueError, match="removed or renamed"):
        publish(store, [extra], second["id"])
    with pytest.raises(ValueError, match="removed or renamed"):
        publish(store, [{**edited, "id": "new-helmet"}, extra], second["id"])
    with pytest.raises(ValueError, match="immutable"):
        store.update("taxonomy_versions", first["id"], {"snapshot": second})
    first["classes"][0]["definition"] = "Mutating return values does not alter storage"
    assert get_taxonomy(store, first["id"])["classes"] == [HELMET]


def test_versions_and_optimistic_edit_tokens_are_scoped_to_each_project(store):
    other = create_project(store, name="Other project")["id"]
    first = publish(store, [HELMET])
    second = publish(store, [{**HELMET, "definition": "Different task."}], project=other)
    assert first["version"] == second["version"] == 2
    assert len(list_taxonomies(store, other)) == 2
    assert list_taxonomies(store, other)[-1] == second
    with pytest.raises(ValueError, match="this project"):
        get_taxonomy(store, first["id"], other)
    with pytest.raises(TaxonomyConflict):
        publish(store, [HELMET], first["id"], project=other)
    assert current_taxonomy(store, other) == second
    rows = project_records(store, "taxonomy_versions", other)
    assert [row["id"] for row in rows] == [second["id"]]
    assert record_project(store, "taxonomy_versions", rows[0]) == other
    for action in (
        lambda: current_taxonomy(store, "missing"),
        lambda: list_taxonomies(store, "missing"),
        lambda: get_taxonomy(store, TAXONOMY["id"], "missing"),
        lambda: publish(store, [HELMET], project="missing"),
    ):
        with pytest.raises(ValueError, match="Project does not exist"):
            action()


@pytest.mark.parametrize(
    "classes",
    [
        [],
        [HELMET] * 101,
        [HELMET, HELMET],
        [{**HELMET, "id": "exclude"}],
        [{**HELMET, "id": "Helmet"}],
        [{**HELMET, "id": "1helmet"}],
        [{**HELMET, "id": "h" * 65}],
        [{**HELMET, "id": "helmet\n"}],
        [{**HELMET, "name": " "}],
        [{**HELMET, "name": "h" * 121}],
        [{**HELMET, "definition": " "}],
        [{**HELMET, "definition": "h" * 2001}],
        [{**HELMET, "extra": True}],
        [{"id": "helmet", "name": "Helmet"}],
        [{**HELMET, "coco_id": True}],
        [{**HELMET, "coco_id": 12}],
        [{**HELMET, "coco_id": 91}],
        [{**HELMET, "coco_id": 1}, {**HELMET, "id": "other", "coco_id": 1}],
        deepcopy(TAXONOMY["classes"]),
    ],
)
def test_invalid_or_unchanged_definitions_publish_nothing(store, classes):
    with pytest.raises(ValueError):
        publish(store, classes)
    assert store.list("taxonomy_versions") == []
    assert current_taxonomy(store, DEFAULT_PROJECT_ID) == TAXONOMY


def test_concurrent_publication_has_one_winner_and_no_orphan_version(store):
    barrier = Barrier(2)

    def edit(index):
        barrier.wait(timeout=5)
        try:
            return publish(store, [{**HELMET, "definition": f"Definition {index}."}])
        except TaxonomyConflict as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(edit, (0, 1)))
    winners = [result for result in results if isinstance(result, dict)]
    assert len(winners) == 1
    assert sum(isinstance(result, TaxonomyConflict) for result in results) == 1
    assert current_taxonomy(store, DEFAULT_PROJECT_ID) == winners[0]
    assert [row["snapshot"] for row in store.list("taxonomy_versions")] == winners
