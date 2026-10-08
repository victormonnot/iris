"""Quality reports pin exact reviewed revisions and survive offline archives."""

import json
import shutil
import sqlite3
import zipfile
from copy import deepcopy

import pytest
from test_temporal_api import publish_sequence, reference_payload, sequence_source
from test_temporal_detection_api import client as client
from test_temporal_detection_api import create, sequence
from test_temporal_detections import execute
from test_temporal_identities import comparison as comparison
from test_temporal_identities import completed as completed
from test_temporal_identities import edit, propose
from test_tracking_comparisons import launch, run_worker
from test_tracking_replay import forbidden, synthetic

from iris import temporal, tracking_quality
from iris.projects import record_project
from iris.store import SCHEMA_VERSION, TABLES, TRACKING_QUALITY_TABLES, Store
from iris.temporal import _digest
from iris.workspace_archive import ArchiveError, _inventory, create_archive, validate_database
from iris.workspace_restore import inspect_archive, restore_archive


@pytest.fixture
def reviewed(client, comparison):
    draft = propose(client, comparison).json()["payload"]
    for frame in draft["frames"]:
        for obj in frame["objects"]:
            obj["certainty"] = "certain"
    response = edit(
        client,
        draft,
        reviewed_frames=[
            {"frame_index": frame["frame_index"], "coverage": "complete"}
            for frame in draft["frames"]
        ],
    )
    assert response.status_code == 201, response.text
    return response.json()


def create_report(client, comparison, reference, **overrides):
    return client.post(
        f"/api/temporal/tracking-comparisons/{comparison['id']}/quality-reports",
        json={
            "reference_id": reference["id"],
            "class_mapping": {"1": "person", "3": None},
            **overrides,
        },
    )


def test_report_is_explicit_immutable_pinned_and_readable_without_ml(
    client, comparison, reviewed, monkeypatch
):
    store = client.app.state.store
    before = {table: store.list(table) for table in TABLES - TRACKING_QUALITY_TABLES}
    monkeypatch.setattr("iris.tracking.tracking_status", forbidden)
    monkeypatch.setattr("iris.tracking_replay._factory", forbidden)
    monkeypatch.setattr("iris.temporal_detections.prepare_detector", forbidden)
    assert client.get("/api/temporal/tracking-quality-status").status_code == 200
    assert (
        client.get(f"/api/temporal/tracking-comparisons/{comparison['id']}/quality-reports").json()
        == []
    )
    response = create_report(client, comparison, reviewed)
    assert response.status_code == 201, response.text
    record = response.json()
    assert record["sequence_id"] == comparison["sequence_id"]
    assert record["reference_id"] == reviewed["id"]
    assert record["config"] == {"class_mapping": {"1": "person", "3": None}, "iou_threshold": 0.5}
    assert record["report_sha256"] == _digest(record["report"])
    assert record["report"]["source"]["reference_sha256"] == reviewed["payload_sha256"]
    assert record["report"]["coverage"]["evaluated_frames"] == 3
    assert record_project(store, "tracking_quality_reports", record) == "default"
    assert {table: store.list(table) for table in before} == before
    detail_url = f"/api/temporal/tracking-quality-reports/{record['id']}"
    assert client.get(detail_url).json() == record
    history = client.get(
        f"/api/temporal/tracking-comparisons/{comparison['id']}/quality-reports"
    ).json()
    expected = deepcopy(record)
    for lane in expected["report"]["lanes"]:
        del lane["frames"]
    assert history == [expected]
    assert store.get("tracking_quality_reports", record["id"]) == record
    with pytest.raises(ValueError, match="immutable"):
        store.update("tracking_quality_reports", record["id"], {"report": {}})

    changed = deepcopy(reviewed["payload"])
    changed["frames"][0]["objects"][0]["box"] = [30, 4, 50, 41]
    revision = edit(client, changed, revision=1).json()
    assert revision["revision"] == 2
    assert client.get(detail_url).json() == record
    second = create_report(client, comparison, revision).json()
    assert second["id"] != record["id"]
    assert second["report"]["source"]["reference_revision"] == 2
    assert second["report"]["coverage"]["evaluated_frames"] == 2
    assert second["report"]["lanes"][0]["identity"]["available"] is False


@pytest.mark.parametrize(
    "options,status",
    [
        ({"class_mapping": {}}, 422),
        ({"class_mapping": {"1": "person"}}, 409),
        ({"class_mapping": {"1": None, "3": None}}, 409),
        ({"class_mapping": {"01": "person", "3": None}}, 409),
        ({"class_mapping": {"1": "missing", "3": None}}, 409),
        ({"class_mapping": {"1": True, "3": None}}, 422),
        ({"iou_threshold": 0}, 422),
        ({"iou_threshold": 1.1}, 422),
        ({"iou_threshold": True}, 422),
        ({"iou_threshold": "0.5"}, 422),
        ({"report": {}}, 422),
        ({"reference_id": "missing"}, 404),
    ],
)
def test_invalid_request_never_publishes_a_report(client, comparison, reviewed, options, status):
    response = create_report(client, comparison, reviewed, **options)
    assert response.status_code == status, response.text
    assert client.app.state.store.list("tracking_quality_reports") == []


def test_unfinished_comparison_and_another_sequence_reference_are_rejected(
    client, comparison, reviewed, tmp_path
):
    store = client.app.state.store
    other = publish_sequence(client, sequence_source(client, tmp_path))
    reference = temporal.save_reference(store, other["id"], payload=reference_payload(other))
    assert create_report(client, comparison, reference).status_code == 409
    store.update("jobs", comparison["id"], {"status": "failed", "result": None})
    response = create_report(client, comparison, reviewed)
    assert response.status_code == 409 and "complete" in response.text
    assert store.list("tracking_quality_reports") == []


def test_api_and_direct_reads_are_project_scoped(client, comparison, reviewed):
    store = client.app.state.store
    record = create_report(client, comparison, reviewed).json()
    foreign = client.post("/api/projects", json={"name": "Other project"}).json()["id"]
    url = f"/api/temporal/tracking-comparisons/{comparison['id']}/quality-reports"
    assert client.get(url, params={"project_id": foreign}).status_code == 404
    assert (
        client.get(
            f"/api/temporal/tracking-quality-reports/{record['id']}", params={"project_id": foreign}
        ).status_code
        == 404
    )
    assert (
        client.post(
            url,
            params={"project_id": foreign},
            json={"reference_id": reviewed["id"], "class_mapping": {"1": "person", "3": None}},
        ).status_code
        == 404
    )
    with pytest.raises(KeyError):
        tracking_quality.get_quality_report(store, record["id"], project_id=foreign)
    with pytest.raises(KeyError):
        tracking_quality.list_quality_reports(store, comparison["id"], project_id=foreign)
    with pytest.raises(KeyError):
        tracking_quality.create_quality_report(
            store,
            comparison["id"],
            reference_id=reviewed["id"],
            class_mapping={"1": "person", "3": None},
            project_id=foreign,
        )
    assert len(store.list("tracking_quality_reports")) == 1


def test_size_failure_and_publication_failure_roll_back(client, comparison, reviewed, monkeypatch):
    store = client.app.state.store
    with monkeypatch.context() as context:
        context.setattr(tracking_quality, "MAX_REPORT_BYTES", 100)
        response = create_report(client, comparison, reviewed)
        assert response.status_code == 409 and "too large" in response.text
    original = tracking_quality._insert

    def interrupted(conn, table, row):
        original(conn, table, row)
        raise RuntimeError("Interrupted before commit")

    monkeypatch.setattr(tracking_quality, "_insert", interrupted)
    response = create_report(client, comparison, reviewed)
    assert response.status_code == 409 and "Interrupted" in response.text
    assert store.list("tracking_quality_reports") == []


def test_nondefault_project_can_publish_and_read_its_own_report(
    client, comparison, reviewed, tmp_path, monkeypatch
):
    store = client.app.state.store
    project = client.post("/api/projects", json={"name": "Second project"}).json()["id"]
    source = sequence(client, tmp_path, project_id=project)
    cache = create(client, source, project_id=project)
    execute(store, cache)
    synthetic(monkeypatch)
    saved = launch(client, cache, project_id=project)
    assert run_worker(client, saved, monkeypatch)["status"] == "succeeded"
    reference = temporal.save_reference(store, source["id"], payload=reference_payload(source))
    url = f"/api/temporal/tracking-comparisons/{saved['id']}/quality-reports"
    request = {"reference_id": reference["id"], "class_mapping": {"1": "person", "3": None}}
    assert client.post(url, json=request).status_code == 404
    assert (
        client.post(
            url, params={"project_id": project}, json={**request, "reference_id": reviewed["id"]}
        ).status_code
        == 404
    )
    response = client.post(url, params={"project_id": project}, json=request)
    assert response.status_code == 201, response.text
    record = response.json()
    assert record_project(store, "tracking_quality_reports", record) == project
    detail = f"/api/temporal/tracking-quality-reports/{record['id']}"
    assert client.get(detail).status_code == 404
    assert client.get(detail, params={"project_id": project}).json() == record
    assert len(client.get(url, params={"project_id": project}).json()) == 1
    assert (
        client.get(f"/api/temporal/tracking-comparisons/{comparison['id']}/quality-reports").json()
        == []
    )


@pytest.mark.parametrize(
    "corruption", ["checksum", "rehashed_report", "config", "reference", "sequence"]
)
def test_read_history_and_archive_reject_corrupt_or_rehashed_evidence(
    client, comparison, reviewed, corruption
):
    store = client.app.state.store
    record = create_report(client, comparison, reviewed).json()
    report = record["report"]
    with store.connect() as conn:
        if corruption == "checksum":
            conn.execute("UPDATE tracking_quality_reports SET report_sha256=?", ("0" * 64,))
        elif corruption == "rehashed_report":
            report["lanes"][0]["counts"]["true_positives"] += 1
            conn.execute(
                "UPDATE tracking_quality_reports SET report=?,report_sha256=?",
                (json.dumps(report), _digest(report)),
            )
        elif corruption == "config":
            config = {**record["config"], "iou_threshold": 0.6}
            conn.execute("UPDATE tracking_quality_reports SET config=?", (json.dumps(config),))
        elif corruption == "reference":
            payload = deepcopy(reviewed["payload"])
            payload["notes"] = "Modified frozen revision"
            conn.execute(
                "UPDATE temporal_references SET payload=?,payload_sha256=? WHERE id=?",
                (json.dumps(payload), _digest(payload), reviewed["id"]),
            )
        else:
            conn.execute("PRAGMA foreign_keys=OFF")
            conn.execute("UPDATE tracking_quality_reports SET sequence_id='missing'")
    assert client.get(f"/api/temporal/tracking-quality-reports/{record['id']}").status_code in (
        404,
        409,
    )
    assert (
        client.get(
            f"/api/temporal/tracking-comparisons/{comparison['id']}/quality-reports"
        ).status_code
        == 409
    )
    inventory, _ = _inventory(store.root)
    with pytest.raises(ArchiveError):
        validate_database(store.root, inventory)


def test_quality_archive_round_trip_keeps_exact_report_and_v1_reference(
    client, comparison, tmp_path, monkeypatch
):
    store = client.app.state.store
    draft = propose(client, comparison).json()["payload"]
    draft.pop("provenance")
    draft["schema"] = "iris-temporal-reference-v1"
    reference = temporal.save_reference(store, comparison["sequence_id"], payload=draft)
    record = create_report(client, comparison, reference).json()
    assert record["report"]["coverage"]["evaluated_frames"] == 0
    assert record["report"]["lanes"][0]["identity"]["available"] is False
    # Historic JSON whitespace is part of saved DB bytes, even though contract
    # digests intentionally describe canonical JSON values.
    with store.connect() as conn:
        conn.execute(
            "UPDATE temporal_references SET payload=? WHERE id=?",
            (json.dumps(draft, indent=2), reference["id"]),
        )
        original_payload = conn.execute("SELECT payload FROM temporal_references").fetchone()[0]
    monkeypatch.setattr("iris.tracking.tracking_status", forbidden)
    monkeypatch.setattr("iris.tracking_replay._factory", forbidden)
    monkeypatch.setattr("iris.temporal_detections.prepare_detector", forbidden)
    archive = create_archive(store.root, tmp_path / "quality.zip")
    assert archive["manifest"]["schema_version"] == 22
    inspected = inspect_archive(archive["path"])
    target = tmp_path / "restored"
    assert restore_archive(
        archive["path"], target, expected_archive_sha256=inspected["archive_sha256"]
    )["verified"]
    with zipfile.ZipFile(archive["path"]) as zipped:
        assert (target / "iris.sqlite3").read_bytes() == zipped.read("iris.sqlite3")
    restored = Store(target)
    assert tracking_quality.get_quality_report(restored, record["id"]) == record
    assert (
        temporal.reference_detail(restored, reference["id"])["payload_sha256"]
        == reference["payload_sha256"]
    )
    with restored.connect() as conn:
        assert (
            conn.execute("SELECT payload FROM temporal_references").fetchone()[0]
            == original_payload
        )


def test_schema21_archive_restores_without_migration_then_additive_migration_preserves_rows(
    client, comparison, reviewed, tmp_path, monkeypatch
):
    source = client.app.state.store
    root = tmp_path / "schema21"
    root.mkdir()
    for path in source.root.iterdir():
        if path.is_dir():
            shutil.copytree(path, root / path.name)
    with source.connect() as old, sqlite3.connect(root / "iris.sqlite3") as saved:
        old.backup(saved)
        saved.execute("DROP TABLE tracking_quality_reports")
        saved.execute("PRAGMA user_version=21")

    def rows(path):
        with sqlite3.connect(path / "iris.sqlite3") as conn:
            return {
                table: conn.execute(f"SELECT * FROM {table} ORDER BY id").fetchall()
                for table in TABLES - TRACKING_QUALITY_TABLES
            }

    original = rows(root)
    archive = create_archive(root, tmp_path / "schema21.zip")
    assert archive["manifest"]["schema_version"] == 21
    assert set(archive["manifest"]["counts"]) == TABLES - TRACKING_QUALITY_TABLES
    inspected = inspect_archive(archive["path"])
    target = tmp_path / "restored21"
    with monkeypatch.context() as context:
        context.setattr(Store, "__init__", forbidden)
        assert restore_archive(
            archive["path"], target, expected_archive_sha256=inspected["archive_sha256"]
        )["verified"]
    with zipfile.ZipFile(archive["path"]) as zipped:
        assert (target / "iris.sqlite3").read_bytes() == zipped.read("iris.sqlite3")
    assert rows(target) == original
    for _ in range(2):
        restored = Store(target)
        assert restored.list("tracking_quality_reports") == []
        assert rows(target) == original
        with restored.connect() as conn:
            assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION == 22
            assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
