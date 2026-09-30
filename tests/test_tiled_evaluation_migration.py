"""Upgrade populated full-image evaluations without changing historical artifacts."""

import re
import shutil
import sqlite3

import pytest
import test_evaluation_analysis as analysis_fixtures

from iris.evaluation import evaluation_detail, reference_history
from iris.evaluation_analysis import analyze_evaluation
from iris.store import SCHEMA, Store, _encode, new_id, now

saved = analysis_fixtures.saved
OLD_MODELS = """CREATE TABLE IF NOT EXISTS evaluation_models (
    id TEXT PRIMARY KEY, evaluation_id TEXT NOT NULL REFERENCES evaluations(id),
    model_id TEXT NOT NULL, metadata TEXT NOT NULL, metrics TEXT,
    created_at TEXT NOT NULL, UNIQUE(evaluation_id,model_id)
);"""
OLD_PREDICTIONS = """CREATE TABLE IF NOT EXISTS evaluation_predictions (
    id TEXT PRIMARY KEY, evaluation_id TEXT NOT NULL REFERENCES evaluations(id),
    evaluation_model_id TEXT NOT NULL REFERENCES evaluation_models(id),
    frame_id TEXT NOT NULL REFERENCES frames(id), model_id TEXT NOT NULL,
    detections TEXT NOT NULL, timing TEXT NOT NULL, input_size TEXT NOT NULL,
    created_at TEXT NOT NULL, UNIQUE(evaluation_model_id,frame_id)
);"""


def make_schema9(saved, path, *, broken=False):
    source, detail, _dataset, _outputs = saved
    source.insert(
        "model_references",
        {
            "id": new_id(),
            "evaluation_id": detail["id"],
            "model_id": detail["model_ids"][0],
            "reviewer": "Synthetic historical fixture",
            "notes": "Preserve legacy decision",
            "metadata": {
                "model_name": "Old full-image reference",
                "metrics": detail["models"][0]["metrics"],
            },
            "created_at": now(),
        },
    )
    schema = re.sub(
        r"CREATE TABLE IF NOT EXISTS evaluation_models \(.*?\);", OLD_MODELS, SCHEMA, flags=re.S
    )
    schema = re.sub(
        r"CREATE TABLE IF NOT EXISTS evaluation_predictions \(.*?\);",
        OLD_PREDICTIONS,
        schema,
        flags=re.S,
    )
    path.mkdir()
    # The schema migration must leave the manifest and frozen images readable too.
    for directory in source.root.iterdir():
        if directory.is_dir():
            shutil.copytree(directory, path / directory.name)
    snapshot = {}
    with source.connect() as conn:
        tables = [
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        ]
    with sqlite3.connect(path / "iris.sqlite3") as conn:
        conn.executescript(schema)
        for table in tables:
            snapshot[table] = []
            for row in source.list(table):
                row = dict(row)
                if table == "evaluation_models":
                    row.pop("variant")
                if table == "evaluation_predictions":
                    row.pop("metadata")
                snapshot[table].append(row)
                encoded = _encode(row)
                conn.execute(
                    f"INSERT INTO {table} ({','.join(encoded)}) "
                    f"VALUES ({','.join('?' for _ in encoded)})",
                    list(encoded.values()),
                )
        if broken:
            conn.execute(
                "UPDATE evaluation_predictions SET evaluation_model_id='missing' "
                "WHERE id=(SELECT id FROM evaluation_predictions LIMIT 1)"
            )
        conn.execute("PRAGMA user_version=9")
    return snapshot


def test_migration_keeps_ids_metrics_predictions_reference_and_legacy_analysis(saved, tmp_path):
    original_analysis = analyze_evaluation(saved[0], saved[1]["id"])
    path = tmp_path / "schema9"
    before = make_schema9(saved, path)
    store = Store(path)
    for table, expected in before.items():
        rows = store.list(table)
        for row in rows:
            if table == "evaluation_models":
                assert row.pop("variant") == "full"
            if table == "evaluation_predictions":
                assert row.pop("metadata") == {}
        assert rows == expected
    assert analyze_evaluation(store, saved[1]["id"]) == original_analysis
    detail = evaluation_detail(store, saved[1]["id"])
    assert [lane["variant"] for lane in detail["lanes"]] == ["full", "full"]
    assert {lane["evaluation_model_id"] for lane in detail["lanes"]} == {
        row["id"] for row in before["evaluation_models"]
    }
    assert reference_history(store)["current"] == before["model_references"][0]
    with store.connect() as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 10
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("UPDATE evaluation_predictions SET evaluation_model_id='missing'")
    model = before["evaluation_models"][0]
    tiled = store.insert("evaluation_models", {**model, "id": new_id(), "variant": "tiled"})
    with pytest.raises(sqlite3.IntegrityError):
        store.insert("evaluation_models", {**tiled, "id": new_id()})
    with pytest.raises(sqlite3.IntegrityError):
        store.insert("evaluation_models", {**model, "id": new_id(), "variant": "unknown"})
    reopened = Store(path)
    assert reopened.list("evaluation_models") == store.list("evaluation_models")
    assert reopened.list("evaluation_predictions") == store.list("evaluation_predictions")
    assert reopened.list("model_references") == store.list("model_references")


def test_failed_rebuild_rolls_back_version_tables_and_existing_rows(saved, tmp_path):
    path = tmp_path / "broken-schema9"
    before = make_schema9(saved, path, broken=True)
    with pytest.raises(RuntimeError, match="foreign keys"):
        Store(path)
    with sqlite3.connect(path / "iris.sqlite3") as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 9
        assert "variant" not in {
            row[1] for row in conn.execute("PRAGMA table_info(evaluation_models)")
        }
        assert "metadata" not in {
            row[1] for row in conn.execute("PRAGMA table_info(evaluation_predictions)")
        }
        assert conn.execute("SELECT count(*) FROM evaluation_predictions").fetchone()[0] == len(
            before["evaluation_predictions"]
        )
        assert conn.execute("SELECT count(*) FROM model_references").fetchone()[0] == 1
