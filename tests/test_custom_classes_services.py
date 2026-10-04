"""Assistance compatibility with class adoption, using a local synthetic reviewer only."""

from copy import deepcopy

import pytest
from test_assistance import prepare, run
from test_assistance import workspace as assistance_workspace

from iris.annotations import adopt_taxonomy, get_annotation
from iris.store import DEFAULT_PROJECT_ID
from iris.taxonomies import TAXONOMY, publish_taxonomy

workspace = assistance_workspace


def publish_custom(store):
    return publish_taxonomy(
        store,
        DEFAULT_PROJECT_ID,
        expected_taxonomy_id=TAXONOMY["id"],
        classes=[
            *deepcopy(TAXONOMY["classes"]),
            {"id": "helmet", "name": "Helmet", "definition": "A visible protective helmet."},
        ],
    )


def test_queued_assistance_stops_before_provider_if_image_adopts_custom_classes(workspace):
    _, store, _, frame = workspace
    _, _, record_id = prepare(workspace)
    custom = publish_custom(store)
    adopted = adopt_taxonomy(
        store,
        frame["id"],
        expected_revision=1,
        expected_taxonomy_id=TAXONOMY["id"],
        target_taxonomy_id=custom["id"],
    )

    def forbidden_provider(*args, **kwargs):
        pytest.fail("A queued review for an obsolete class version must not load a provider")

    with pytest.raises(ValueError, match="class definitions changed"):
        run(store, record_id, factory=forbidden_provider)
    assert store.list("annotation_suggestions") == []
    assert get_annotation(store, frame["id"]) == adopted
    record = store.get("assistance_records", record_id)
    assert "class definitions changed" in record["error"]
    assert record["raw_response"] is None


def test_project_definition_change_keeps_queued_legacy_image_review_valid(workspace):
    _, store, _, frame = workspace
    _, _, record_id = prepare(workspace)
    custom = publish_custom(store)
    result = run(store, record_id)
    assert result["suggestions_created"] == 1
    annotation = get_annotation(store, frame["id"])
    assert annotation["taxonomy"] == TAXONOMY and annotation["current_taxonomy"] == custom
    assert annotation["taxonomy_outdated"] is True
    assert annotation["revision"] == 1 and annotation["status"] == "validated"
    (suggestion,) = annotation["suggestions"]
    assert suggestion["state"] == "pending"
    assert suggestion["taxonomy_outdated"] is False
    assert store.get("assistance_records", record_id)["config"]["taxonomy_id"] == TAXONOMY["id"]


def test_unsupported_persisted_assistance_taxonomy_stops_before_provider(workspace):
    _, store, _, frame = workspace
    before, _, record_id = prepare(workspace)
    custom = publish_custom(store)
    record = store.get("assistance_records", record_id)
    store.update(
        "assistance_records",
        record_id,
        {"config": {**record["config"], "taxonomy_id": custom["id"]}},
    )

    def forbidden_provider(*args, **kwargs):
        pytest.fail("An unsupported saved assistance configuration must not load a provider")

    with pytest.raises(ValueError, match="does not support custom"):
        run(store, record_id, factory=forbidden_provider)
    after = get_annotation(store, frame["id"])
    assert after["revision"] == before["revision"] and after["boxes"] == before["boxes"]
    assert store.list("annotation_suggestions") == []
