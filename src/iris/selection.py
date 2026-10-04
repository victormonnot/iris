"""Atomic manual selection changes, with optimistic concurrency for visible batches."""

from iris.store import DEFAULT_PROJECT_ID, Store

MAX_SELECTION_FRAMES = 1000


class SelectionConflict(ValueError):
    """A displayed selection changed in another tab before this batch was applied."""


def set_selection(
    store: Store,
    session_id: str,
    *,
    frame_ids: list[str],
    selected: bool,
    expected_selection: dict[str, bool],
    project_id: str = DEFAULT_PROJECT_ID,
) -> dict:
    if (
        not isinstance(frame_ids, list)
        or not 1 <= len(frame_ids) <= MAX_SELECTION_FRAMES
        or any(not isinstance(identifier, str) or not identifier for identifier in frame_ids)
        or len(set(frame_ids)) != len(frame_ids)
        or type(selected) is not bool
        or not isinstance(expected_selection, dict)
        or set(expected_selection) != set(frame_ids)
        or any(type(value) is not bool for value in expected_selection.values())
    ):
        raise ValueError("Choose 1–1000 distinct frames with their displayed selection states")
    with store.connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        session = connection.execute(
            "SELECT project_id FROM sessions WHERE id=?", (session_id,)
        ).fetchone()
        if session is None or session["project_id"] != project_id:
            raise KeyError(session_id)
        records = {}
        # Keep parameter counts portable across SQLite builds.
        for start in range(0, len(frame_ids), 400):
            batch = frame_ids[start : start + 400]
            placeholders = ",".join("?" for _ in batch)
            records.update(
                (row["id"], bool(row["selected"]))
                for row in connection.execute(
                    f"SELECT id,selected FROM frames WHERE session_id=? AND id IN ({placeholders})",
                    (session_id, *batch),
                )
            )
        if set(records) != set(frame_ids):
            raise ValueError("Every chosen frame must belong to this session")
        if any(records[identifier] != expected_selection[identifier] for identifier in frame_ids):
            raise SelectionConflict("Selection changed in another tab; refresh before applying it")
        connection.executemany(
            "UPDATE frames SET selected=? WHERE id=?",
            [(int(selected), identifier) for identifier in frame_ids],
        )
    return {
        "session_id": session_id,
        "frames": [{"id": identifier, "selected": selected} for identifier in frame_ids],
        "changed_count": sum(value != selected for value in records.values()),
    }
