"""SQLite metadata and local artifact storage, shared with the worker process."""

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path


def new_id() -> str:
    return uuid.uuid4().hex


def now() -> str:
    return datetime.now(UTC).isoformat()


SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, scene_group TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS assets (
    id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id), filename TEXT NOT NULL,
    kind TEXT NOT NULL CHECK(kind IN ('image','video')), sha256 TEXT NOT NULL,
    size_bytes INTEGER NOT NULL, path TEXT NOT NULL, metadata TEXT NOT NULL,
    created_at TEXT NOT NULL, UNIQUE(session_id,sha256)
);
CREATE TABLE IF NOT EXISTS frames (
    id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id),
    asset_id TEXT NOT NULL REFERENCES assets(id), frame_index INTEGER, timestamp_seconds REAL,
    width INTEGER NOT NULL, height INTEGER NOT NULL, sha256 TEXT NOT NULL,
    perceptual_hash TEXT NOT NULL, path TEXT NOT NULL,
    selected INTEGER NOT NULL DEFAULT 0 CHECK(selected IN (0,1)),
    extraction TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL,
    UNIQUE(asset_id,frame_index)
);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY, kind TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN
      ('queued','running','succeeded','failed','cancelled','interrupted')),
    params TEXT NOT NULL, result TEXT, progress REAL NOT NULL DEFAULT 0,
    message TEXT NOT NULL DEFAULT '', logs TEXT NOT NULL DEFAULT '[]', error TEXT,
    created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT,
    cancel_requested INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS frames_session ON frames(session_id);
CREATE INDEX IF NOT EXISTS frames_hash ON frames(sha256);
CREATE INDEX IF NOT EXISTS assets_session ON assets(session_id);
CREATE INDEX IF NOT EXISTS jobs_status ON jobs(status);
CREATE TABLE IF NOT EXISTS comparisons (
    id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id), name TEXT NOT NULL,
    frame_ids TEXT NOT NULL, model_ids TEXT NOT NULL, config TEXT NOT NULL,
    job_id TEXT NOT NULL UNIQUE REFERENCES jobs(id), created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY, comparison_id TEXT NOT NULL REFERENCES comparisons(id),
    model_id TEXT NOT NULL, variant TEXT NOT NULL DEFAULT 'full'
    CHECK(variant IN ('full','tiled')), metadata TEXT NOT NULL, created_at TEXT NOT NULL,
    UNIQUE(comparison_id, model_id, variant)
);
CREATE TABLE IF NOT EXISTS predictions (
    id TEXT PRIMARY KEY, comparison_id TEXT NOT NULL REFERENCES comparisons(id),
    run_id TEXT NOT NULL REFERENCES runs(id), frame_id TEXT NOT NULL REFERENCES frames(id),
    model_id TEXT NOT NULL, detections TEXT NOT NULL, timing TEXT NOT NULL,
    input_size TEXT NOT NULL, metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL, UNIQUE(run_id, frame_id)
);
CREATE INDEX IF NOT EXISTS comparisons_session ON comparisons(session_id);
CREATE INDEX IF NOT EXISTS predictions_comparison ON predictions(comparison_id);
CREATE TABLE IF NOT EXISTS annotation_revisions (
    id TEXT PRIMARY KEY, frame_id TEXT NOT NULL REFERENCES frames(id), revision INTEGER NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('draft','validated')), taxonomy_id TEXT NOT NULL,
    frame_sha256 TEXT NOT NULL, boxes TEXT NOT NULL, decisions TEXT NOT NULL,
    reviewer TEXT NOT NULL DEFAULT '', notes TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL,
    UNIQUE(frame_id, revision)
);
CREATE TABLE IF NOT EXISTS annotation_suggestions (
    id TEXT PRIMARY KEY, frame_id TEXT NOT NULL REFERENCES frames(id),
    job_id TEXT REFERENCES jobs(id),
    kind TEXT NOT NULL CHECK(kind IN ('detector','multimodal','imported')),
    label TEXT NOT NULL, box TEXT NOT NULL, metadata TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS assistance_records (
    id TEXT PRIMARY KEY, frame_id TEXT NOT NULL REFERENCES frames(id),
    job_id TEXT NOT NULL UNIQUE REFERENCES jobs(id), config TEXT NOT NULL, candidates TEXT NOT NULL,
    prompt TEXT NOT NULL DEFAULT '', metadata TEXT NOT NULL DEFAULT '{}', raw_response TEXT,
    error TEXT, created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS annotation_revisions_frame ON annotation_revisions(frame_id,revision);
CREATE INDEX IF NOT EXISTS annotation_suggestions_frame ON annotation_suggestions(frame_id);
CREATE INDEX IF NOT EXISTS assistance_records_frame ON assistance_records(frame_id);
CREATE TABLE IF NOT EXISTS assistance_batches (
    id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id), name TEXT NOT NULL,
    frame_ids TEXT NOT NULL, job_ids TEXT NOT NULL, config TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS assistance_batches_session ON assistance_batches(session_id);
CREATE TABLE IF NOT EXISTS assistance_previews (
    id TEXT PRIMARY KEY, frame_id TEXT NOT NULL REFERENCES frames(id), config TEXT NOT NULL,
    candidates TEXT NOT NULL, images TEXT NOT NULL, created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL, job_id TEXT UNIQUE REFERENCES jobs(id)
);
CREATE TABLE IF NOT EXISTS dataset_versions (
    id TEXT PRIMARY KEY, name TEXT NOT NULL,
    parent_id TEXT REFERENCES dataset_versions(id), path TEXT NOT NULL,
    manifest_sha256 TEXT NOT NULL, summary TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS training_runs (
    id TEXT PRIMARY KEY, name TEXT NOT NULL,
    dataset_id TEXT NOT NULL REFERENCES dataset_versions(id), parent_model_id TEXT NOT NULL,
    config TEXT NOT NULL, metadata TEXT NOT NULL DEFAULT '{}',
    history TEXT NOT NULL DEFAULT '[]', checkpoint_id TEXT,
    job_id TEXT NOT NULL UNIQUE REFERENCES jobs(id), created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS trained_models (
    id TEXT PRIMARY KEY, name TEXT NOT NULL,
    training_id TEXT NOT NULL UNIQUE REFERENCES training_runs(id),
    parent_model_id TEXT NOT NULL, architecture TEXT NOT NULL, path TEXT NOT NULL,
    weight_sha256 TEXT NOT NULL, metadata TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS training_runs_dataset ON training_runs(dataset_id);
CREATE TABLE IF NOT EXISTS evaluations (
    id TEXT PRIMARY KEY, name TEXT NOT NULL,
    dataset_id TEXT NOT NULL REFERENCES dataset_versions(id),
    split TEXT NOT NULL CHECK(split IN ('val','test')), model_ids TEXT NOT NULL,
    config TEXT NOT NULL, job_id TEXT NOT NULL UNIQUE REFERENCES jobs(id), created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS evaluation_models (
    id TEXT PRIMARY KEY, evaluation_id TEXT NOT NULL REFERENCES evaluations(id),
    model_id TEXT NOT NULL, metadata TEXT NOT NULL, metrics TEXT,
    created_at TEXT NOT NULL, UNIQUE(evaluation_id,model_id)
);
CREATE TABLE IF NOT EXISTS evaluation_predictions (
    id TEXT PRIMARY KEY, evaluation_id TEXT NOT NULL REFERENCES evaluations(id),
    evaluation_model_id TEXT NOT NULL REFERENCES evaluation_models(id),
    frame_id TEXT NOT NULL REFERENCES frames(id), model_id TEXT NOT NULL,
    detections TEXT NOT NULL, timing TEXT NOT NULL, input_size TEXT NOT NULL,
    created_at TEXT NOT NULL, UNIQUE(evaluation_model_id,frame_id)
);
CREATE TABLE IF NOT EXISTS model_references (
    id TEXT PRIMARY KEY, evaluation_id TEXT NOT NULL REFERENCES evaluations(id),
    model_id TEXT NOT NULL, reviewer TEXT NOT NULL, notes TEXT NOT NULL,
    metadata TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS evaluations_dataset ON evaluations(dataset_id);
CREATE INDEX IF NOT EXISTS evaluation_predictions_evaluation
    ON evaluation_predictions(evaluation_id);
CREATE TABLE IF NOT EXISTS dataset_imports (
    id TEXT PRIMARY KEY, path TEXT NOT NULL, sha256 TEXT NOT NULL,
    summary TEXT NOT NULL, metadata TEXT NOT NULL DEFAULT '{}', result TEXT,
    created_at TEXT NOT NULL
);
"""

IMPORTED_SUGGESTIONS_MIGRATION = """
CREATE TABLE annotation_suggestions_v7 (
    id TEXT PRIMARY KEY, frame_id TEXT NOT NULL REFERENCES frames(id),
    job_id TEXT REFERENCES jobs(id),
    kind TEXT NOT NULL CHECK(kind IN ('detector','multimodal','imported')),
    label TEXT NOT NULL, box TEXT NOT NULL, metadata TEXT NOT NULL, created_at TEXT NOT NULL
);
INSERT INTO annotation_suggestions_v7
    (id,frame_id,job_id,kind,label,box,metadata,created_at)
SELECT id,frame_id,job_id,kind,label,box,metadata,created_at FROM annotation_suggestions;
DROP TABLE annotation_suggestions;
ALTER TABLE annotation_suggestions_v7 RENAME TO annotation_suggestions;
CREATE INDEX annotation_suggestions_frame ON annotation_suggestions(frame_id);
"""

RUN_VARIANTS_MIGRATION = """
CREATE TABLE runs_v9 (
    id TEXT PRIMARY KEY, comparison_id TEXT NOT NULL REFERENCES comparisons(id),
    model_id TEXT NOT NULL, variant TEXT NOT NULL DEFAULT 'full'
    CHECK(variant IN ('full','tiled')), metadata TEXT NOT NULL, created_at TEXT NOT NULL,
    UNIQUE(comparison_id, model_id, variant)
);
INSERT INTO runs_v9 (id,comparison_id,model_id,metadata,created_at)
SELECT id,comparison_id,model_id,metadata,created_at FROM runs;
DROP TABLE runs;
ALTER TABLE runs_v9 RENAME TO runs;
"""

JSON_FIELDS = {
    "metadata",
    "params",
    "result",
    "logs",
    "extraction",
    "frame_ids",
    "job_ids",
    "model_ids",
    "config",
    "detections",
    "timing",
    "input_size",
    "boxes",
    "decisions",
    "box",
    "candidates",
    "raw_response",
    "images",
    "summary",
    "history",
    "metrics",
}
BOOL_FIELDS = {"selected", "cancel_requested"}
TABLES = {
    "sessions",
    "assets",
    "frames",
    "jobs",
    "comparisons",
    "runs",
    "predictions",
    "annotation_revisions",
    "annotation_suggestions",
    "assistance_records",
    "assistance_batches",
    "assistance_previews",
    "dataset_versions",
    "training_runs",
    "trained_models",
    "evaluations",
    "evaluation_models",
    "evaluation_predictions",
    "model_references",
    "dataset_imports",
}


def _decode(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    result = dict(row)
    for key in result.keys() & JSON_FIELDS:
        if result[key] is not None:
            result[key] = json.loads(result[key])
    for key in result.keys() & BOOL_FIELDS:
        result[key] = bool(result[key])
    return result


def _encode(data: dict) -> dict:
    return {
        key: json.dumps(value, allow_nan=False)
        if key in JSON_FIELDS and value is not None
        else value
        for key, value in data.items()
    }


class Store:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "iris.sqlite3"
        with self.connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9):
                raise RuntimeError(f"Unsupported database version: {version}")
            old_suggestions = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='annotation_suggestions'"
            ).fetchone()
            # SQLite requires a table rebuild to extend the suggestion kind CHECK.
            # No tables reference suggestions; revisions retain their IDs in JSON.
            migration = (
                IMPORTED_SUGGESTIONS_MIGRATION
                if old_suggestions and "'imported'" not in old_suggestions[0]
                else ""
            )
            run_columns = {row[1] for row in conn.execute("PRAGMA table_info(runs)")}
            rebuild_runs = bool(run_columns and "variant" not in run_columns)
            if rebuild_runs:
                migration += RUN_VARIANTS_MIGRATION
                # Predictions reference runs. Keep the table name and IDs intact,
                # then check every foreign key before committing the rebuild.
                conn.execute("PRAGMA foreign_keys=OFF")
            prediction_columns = {row[1] for row in conn.execute("PRAGMA table_info(predictions)")}
            if prediction_columns and "metadata" not in prediction_columns:
                migration += (
                    "ALTER TABLE predictions ADD COLUMN metadata TEXT NOT NULL DEFAULT '{}';"
                )
            try:
                conn.executescript(
                    "BEGIN IMMEDIATE;\n" + SCHEMA + migration + "\nPRAGMA user_version=9;"
                )
                if conn.execute("PRAGMA foreign_key_check").fetchone():
                    raise RuntimeError("Database migration found broken foreign keys")
                conn.commit()
            except BaseException:
                conn.rollback()
                raise
            finally:
                conn.execute("PRAGMA foreign_keys=ON")
            self.columns = {
                table: {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
                for table in TABLES
            }

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _check(self, table: str, fields=()):
        if table not in TABLES or not set(fields) <= self.columns[table]:
            raise ValueError("Unknown table or column")

    def insert(self, table: str, data: dict) -> dict:
        self._check(table, data)
        encoded = _encode(data)
        with self.connect() as conn:
            conn.execute(
                f"INSERT INTO {table} ({','.join(encoded)}) "
                f"VALUES ({','.join('?' for _ in encoded)})",
                list(encoded.values()),
            )
        return self.get(table, data["id"])

    def get(self, table: str, record_id: str) -> dict | None:
        self._check(table)
        with self.connect() as conn:
            return _decode(
                conn.execute(f"SELECT * FROM {table} WHERE id=?", (record_id,)).fetchone()
            )

    def list(self, table: str, **filters) -> list[dict]:
        self._check(table, filters)
        where = " WHERE " + " AND ".join(f"{key}=?" for key in filters) if filters else ""
        with self.connect() as conn:
            return [
                _decode(row)
                for row in conn.execute(
                    f"SELECT * FROM {table}{where} ORDER BY created_at, id", list(filters.values())
                )
            ]

    def update(self, table: str, record_id: str, data: dict) -> dict:
        self._check(table, data)
        if not data or "id" in data:
            raise ValueError("An update must contain fields and cannot change the ID")
        encoded = _encode(data)
        with self.connect() as conn:
            cursor = conn.execute(
                f"UPDATE {table} SET {','.join(f'{key}=?' for key in encoded)} WHERE id=?",
                [*encoded.values(), record_id],
            )
            if cursor.rowcount != 1:
                raise KeyError(record_id)
        return self.get(table, record_id)

    def artifact_path(self, relative: str) -> Path:
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Artifact path is outside the workspace")
        return path
