"""Background workspace transfers with durable receipts and a short mutation gate."""

from __future__ import annotations

import copy
import json
import re
import shlex
import shutil
import sys
import threading
import time
from pathlib import Path

from iris.store import Store, new_id, now
from iris.workspace_archive import (
    MAX_ARCHIVE_BYTES,
    ArchiveCancelled,
    ArchiveError,
    ArchiveLimitError,
    create_archive,
    preview_workspace,
)
from iris.workspace_restore import inspect_archive, restore_archive

ACTIVE = {"queued", "running"}
IDENTIFIER = re.compile(r"[0-9a-f]{32}\Z")
FOLDER_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}\Z")


class WorkspaceBusy(RuntimeError):
    """A local write or another workspace transfer is still active."""


class MutationGate:
    """Admission control, not a lock held while an HTTP request is awaited."""

    def __init__(self):
        self.lock = threading.Lock()
        self.writers = 0
        self.frozen = False

    def enter(self):
        with self.lock:
            if self.frozen:
                raise WorkspaceBusy("A workspace backup is running. Save again when it finishes.")
            self.writers += 1

    def leave(self):
        with self.lock:
            self.writers -= 1

    def freeze(self):
        with self.lock:
            if self.frozen or self.writers:
                raise WorkspaceBusy("A workspace change is still in progress. Try again shortly.")
            self.frozen = True

    def release(self):
        with self.lock:
            self.frozen = False


def _summary(manifest):
    return {
        key: manifest[key]
        for key in (
            "protocol",
            "app_version",
            "schema_version",
            "created_at",
            "file_count",
            "total_bytes",
            "counts",
        )
    }


class WorkspaceOperations:
    def __init__(self, store: Store):
        self.store = store
        self.root = store.root / "workspace_operations"
        self.gate = MutationGate()
        self.lock = threading.RLock()
        self.active_id = None
        self.thread = None
        self.cancel_event = threading.Event()
        self.downloads = {}
        self.volatile_receipts = {}

    def _directory(self, identifier):
        if not isinstance(identifier, str) or not IDENTIFIER.fullmatch(identifier):
            raise KeyError(identifier)
        directory = self.root / identifier
        if self.root.is_symlink() or directory.is_symlink():
            raise ArchiveError("Workspace transfer storage must not use symbolic links")
        return directory

    def _read(self, identifier):
        if identifier in self.volatile_receipts:
            return copy.deepcopy(self.volatile_receipts[identifier])
        path = self._directory(identifier) / "operation.json"
        if path.is_symlink():
            raise ArchiveError("Workspace operation receipt must be a regular local file")
        try:
            if path.stat().st_size > 1024 * 1024:
                raise ArchiveError("Workspace operation receipt exceeds its size limit")
            record = json.loads(path.read_text())
        except FileNotFoundError:
            raise KeyError(identifier) from None
        except (ValueError, UnicodeError):
            raise ArchiveError("Workspace operation receipt is unreadable") from None
        if not isinstance(record, dict) or record.get("id") != identifier:
            raise ArchiveError("Workspace operation receipt has an invalid identity")
        return record

    def _write(self, record):
        directory = self._directory(record["id"])
        directory.mkdir(parents=True, mode=0o700, exist_ok=True)
        target = directory / "operation.json"
        temporary = directory / "operation.json.tmp"
        temporary.write_text(json.dumps(record, allow_nan=False))
        temporary.chmod(0o600)
        temporary.replace(target)

    def _public(self, record):
        return copy.deepcopy(
            {key: value for key, value in record.items() if not key.startswith("_")}
        )

    def _terminal(self, record):
        try:
            self._write(record)
        except OSError:
            record["warning"] = (
                "The transfer receipt could not be saved. This status is available only until "
                "IRIS restarts. Keep any completed download or restored destination shown here."
            )
            self.volatile_receipts[record["id"]] = copy.deepcopy(record)

    def get(self, identifier):
        with self.lock:
            return self._public(self._read(identifier))

    def list(self):
        with self.lock:
            if not self.root.exists():
                return []
            if self.root.is_symlink():
                raise ArchiveError("Workspace transfer storage must not use symbolic links")
            records = [
                self._public(self._read(path.name))
                for path in self.root.iterdir()
                if IDENTIFIER.fullmatch(path.name) and (path / "operation.json").exists()
            ]
            return sorted(records, key=lambda record: record["created_at"], reverse=True)

    def recover(self):
        """An interrupted transfer is never resumed or presented as verified."""
        with self.lock:
            for public in self.list():
                if public["status"] not in ACTIVE:
                    continue
                record = self._read(public["id"])
                record.update(
                    status="interrupted",
                    updated_at=now(),
                    error="IRIS stopped before this transfer finished. Start a new transfer.",
                )
                if record.get("_destination"):
                    record["error"] += (
                        " Check the intended destination before retrying: " + record["_destination"]
                    )
                self._write(record)

    def close(self):
        with self.lock:
            self.cancel_event.set()
            thread = self.thread
        if thread:
            thread.join(timeout=10)

    def preview(self):
        result = preview_workspace(self.store.root)
        with self.lock:
            if self.active_id:
                result["can_create"] = False
                result["blocking_issues"].append("Another workspace transfer is in progress")
            return {
                **result,
                "workspace_path": str(self.store.root),
                "restore_parent": str(self.store.root.parent),
                "max_upload_bytes": MAX_ARCHIVE_BYTES,
                "active_operation_id": self.active_id,
                "write_locked": self.gate.frozen,
            }

    def _reserve(self, kind):
        with self.lock:
            if self.active_id:
                raise WorkspaceBusy("Another workspace transfer is in progress")
            identifier = new_id()
            record = {
                "id": identifier,
                "kind": kind,
                "status": "queued",
                "created_at": now(),
                "updated_at": now(),
                "progress": {
                    "phase": "queued",
                    "message": "Preparing the local transfer",
                    "bytes_done": 0,
                    "bytes_total": 0,
                    "files_done": 0,
                    "files_total": 0,
                },
                "error": None,
                "result": None,
            }
            self._write(record)
            self.active_id = identifier
            self.cancel_event = threading.Event()
            return record

    def _launch(self, record, action, *, unfreeze=False):
        identifier = record["id"]
        cancellation = self.cancel_event
        last_progress = [0.0, None]

        def progress(update):
            clock = time.monotonic()
            phase = update.get("phase")
            if phase == last_progress[1] and clock - last_progress[0] < 0.2:
                return
            last_progress[:] = [clock, phase]
            with self.lock:
                current = self._read(identifier)
                current["progress"].update(update)
                current["updated_at"] = now()
                self._write(current)

        def execute():
            try:
                with self.lock:
                    current = self._read(identifier)
                    current.update(status="running", updated_at=now())
                    self._write(current)
                result = action(progress, cancellation.is_set)
                with self.lock:
                    current = self._read(identifier)
                    current.update(status="succeeded", result=result, updated_at=now())
                    current["progress"].update(phase="complete", message="Transfer verified")
                    self._terminal(current)
            except Exception as exc:
                with self.lock:
                    current = self._read(identifier)
                    current.update(
                        status="cancelled" if isinstance(exc, ArchiveCancelled) else "failed",
                        error=(
                            "Transfer cancelled. No incomplete result was published."
                            if isinstance(exc, ArchiveCancelled)
                            else str(exc)
                            if isinstance(exc, (ArchiveError, WorkspaceBusy))
                            else "Transfer could not finish. Check local files and free disk space."
                        ),
                        updated_at=now(),
                    )
                    self._terminal(current)
            finally:
                with self.lock:
                    self.active_id = None
                    if unfreeze:
                        self.gate.release()

        with self.lock:
            self.thread = threading.Thread(
                target=execute, name=f"iris-workspace-{record['kind']}", daemon=True
            )
            try:
                self.thread.start()
            except BaseException as exc:
                failed = copy.deepcopy(record)
                failed.update(
                    status="failed",
                    updated_at=now(),
                    error="The transfer worker could not start. Try again after restarting IRIS.",
                )
                try:
                    self._terminal(failed)
                finally:
                    self.active_id = None
                    self.thread = None
                    if unfreeze:
                        self.gate.release()
                raise WorkspaceBusy(failed["error"]) from exc
            return self._public(record)

    def backup(self):
        with self.lock:
            if self.active_id:
                raise WorkspaceBusy("Another workspace transfer is in progress")
            self.gate.freeze()
            try:
                preview = preview_workspace(self.store.root)
                if not preview["can_create"]:
                    raise ArchiveError("; ".join(preview["blocking_issues"]))
                record = self._reserve("backup")
            except BaseException:
                self.gate.release()
                raise
            archive = self._directory(record["id"]) / "workspace.zip"

            def action(progress, cancelled):
                receipt = create_archive(
                    self.store.root, archive, progress=progress, cancelled=cancelled
                )
                return {
                    "summary": _summary(receipt["manifest"]),
                    "archive_size_bytes": receipt["archive_size_bytes"],
                    "archive_sha256": receipt["archive_sha256"],
                    "filename": f"iris-workspace-{record['id']}.zip",
                    "download_url": f"/api/workspace/operations/{record['id']}/archive",
                }

            return self._launch(record, action, unfreeze=True)

    def inspect_upload(self, source, filename):
        """Spool a completed multipart upload, then validate it in the background."""
        with self.lock:
            record = self._reserve("inspection")
        archive = self._directory(record["id"]) / "workspace.zip"
        try:
            written = 0
            with archive.open("xb") as target:
                archive.chmod(0o600)
                while chunk := source.read(1024 * 1024):
                    if self.cancel_event.is_set():
                        raise ArchiveCancelled()
                    written += len(chunk)
                    if written > MAX_ARCHIVE_BYTES:
                        raise ArchiveLimitError("Workspace ZIP exceeds the 64 GiB limit")
                    if shutil.disk_usage(archive.parent).free < len(chunk) + 16 * 1024 * 1024:
                        raise OSError("Insufficient disk space")
                    target.write(chunk)
            if not written:
                raise ArchiveError("The uploaded workspace archive is empty")
        except BaseException:
            with self.lock:
                archive.unlink(missing_ok=True)
                record.update(status="failed", updated_at=now(), error="Upload could not be saved")
                self._write(record)
                self.active_id = None
            raise

        def action(progress, cancelled):
            receipt = inspect_archive(archive, progress=progress, cancelled=cancelled)
            return {
                "summary": _summary(receipt["manifest"]),
                "archive_size_bytes": receipt["archive_size_bytes"],
                "archive_sha256": receipt["archive_sha256"],
                "filename": Path(filename.replace("\\", "/")).name[:200] or "workspace.zip",
            }

        return self._launch(record, action)

    def restore(self, inspection_id, folder_name):
        if not isinstance(folder_name, str) or not FOLDER_NAME.fullmatch(folder_name):
            raise ArchiveError(
                "Use 1–80 letters, digits, hyphens or underscores for the new folder"
            )
        destination = self.store.root.parent / folder_name
        if destination == self.store.root or destination.exists() or destination.is_symlink():
            raise WorkspaceBusy("The destination already exists. Choose a new folder name.")
        with self.lock:
            inspection = self._read(inspection_id)
            if inspection["kind"] != "inspection" or inspection["status"] != "succeeded":
                raise ArchiveError("Inspect and verify the uploaded archive before restoring it")
            if self.active_id:
                raise WorkspaceBusy("Another workspace transfer is in progress")
            archive = self._directory(inspection_id) / "workspace.zip"
            expected = inspection["result"]["archive_sha256"]
            record = self._reserve("restore")
            record["_inspection_id"] = inspection_id
            record["_destination"] = str(destination)
            self._write(record)

        def action(progress, cancelled):
            receipt = restore_archive(
                archive,
                destination,
                expected_archive_sha256=expected,
                progress=progress,
                cancelled=cancelled,
            )
            return {
                "summary": _summary(receipt["manifest"]),
                "archive_sha256": receipt["archive_sha256"],
                "destination": str(destination),
                "launch_command": shlex.join(
                    [
                        str(Path(sys.executable).parent / "iris"),
                        "--data-dir",
                        str(destination),
                        "--port",
                        "8011",
                    ]
                ),
            }

        return self._launch(record, action)

    def cancel(self, identifier):
        with self.lock:
            record = self._read(identifier)
            if record["status"] in ACTIVE and self.active_id == identifier:
                self.cancel_event.set()
                record["progress"]["message"] = "Cancellation requested; finishing the current file"
                record["updated_at"] = now()
                self._write(record)
            return self._public(record)

    def archive(self, identifier):
        with self.lock:
            record = self._read(identifier)
            if record["kind"] != "backup" or record["status"] != "succeeded":
                raise ArchiveError("This operation has no completed backup to download")
            path = self._directory(identifier) / "workspace.zip"
            if (
                path.is_symlink()
                or not path.is_file()
                or path.stat().st_size != record["result"]["archive_size_bytes"]
            ):
                raise ArchiveError("The saved backup archive is missing or has changed")
            self.downloads[identifier] = self.downloads.get(identifier, 0) + 1
            return path, record["result"]["filename"]

    def release_download(self, identifier):
        with self.lock:
            self.downloads[identifier] -= 1
            if not self.downloads[identifier]:
                del self.downloads[identifier]

    def delete(self, identifier):
        with self.lock:
            record = self._read(identifier)
            if record["status"] in ACTIVE:
                raise WorkspaceBusy(
                    "Cancel the transfer and wait for it to finish before removing it"
                )
            if record["kind"] not in {"backup", "inspection"}:
                raise ArchiveError("Restored workspaces cannot be removed from this dialog")
            if self.downloads.get(identifier):
                raise WorkspaceBusy("This backup is being downloaded. Wait before removing it.")
            if self.active_id:
                active = self._read(self.active_id)
                if active.get("_inspection_id") == identifier:
                    raise WorkspaceBusy("This archive is currently being restored")
            shutil.rmtree(self._directory(identifier))
            self.volatile_receipts.pop(identifier, None)
