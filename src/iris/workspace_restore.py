"""Verify private workspace archives and restore them into new directories only."""

import ctypes
import errno
import hashlib
import json
import os
import shutil
import stat
import struct
import tempfile
import unicodedata
import zipfile
from contextlib import contextmanager
from pathlib import Path

from iris.workspace_archive import (
    CHUNK_BYTES,
    MAX_ARCHIVE_BYTES,
    MAX_FILES,
    MAX_MANIFEST_BYTES,
    MAX_REFERENCE_BYTES,
    MAX_TOTAL_BYTES,
    ArchiveCancelled,
    ArchiveError,
    ArchiveLimitError,
    allowed_artifact_path,
    is_reference_document,
    safe_member_path,
    validate_database,
    validate_manifest,
)


def _cancel(cancelled):
    if cancelled is not None and cancelled():
        raise ArchiveCancelled("Workspace operation cancelled")


def _progress(callback, phase, message, done, total, files_done=0, files_total=0):
    if callback is not None:
        callback(
            {
                "phase": phase,
                "message": message,
                "bytes_done": done,
                "bytes_total": total,
                "files_done": files_done,
                "files_total": files_total,
            }
        )


def _signature(stream):
    value = os.fstat(stream.fileno())
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


@contextmanager
def _open_archive(path):
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "rb") as stream:
            value = os.fstat(stream.fileno())
            if not stat.S_ISREG(value.st_mode):
                raise ArchiveError("The archive must be a regular local file")
            if not 0 < value.st_size <= MAX_ARCHIVE_BYTES:
                raise ArchiveLimitError("The archive is empty or exceeds the archive size limit")
            yield stream
    except OSError as exc:
        raise ArchiveError("The archive cannot be read as a regular local file") from exc


def _digest(stream, *, progress, cancelled, phase):
    stream.seek(0)
    digest = hashlib.sha256()
    total, count = os.fstat(stream.fileno()).st_size, 0
    _progress(progress, phase, "Verifying archive identity", 0, total)
    while True:
        _cancel(cancelled)
        chunk = stream.read(CHUNK_BYTES)
        if not chunk:
            break
        count += len(chunk)
        if count > MAX_ARCHIVE_BYTES:
            raise ArchiveLimitError("The archive exceeds the archive size limit")
        digest.update(chunk)
        _progress(progress, phase, "Verifying archive identity", count, total)
    return digest.hexdigest()


def _directory_bounds(stream):
    """Bound central-directory allocation before zipfile constructs its member list."""
    size = os.fstat(stream.fileno()).st_size
    if size < 22:
        raise ArchiveError("The archive is not a complete ZIP file")
    stream.seek(size - 22)
    end = struct.unpack("<4s4H2LH", stream.read(22))
    signature, disk, directory_disk, disk_count, count, directory_size, offset, comment = end
    if signature != b"PK\x05\x06" or disk or directory_disk or comment:
        raise ArchiveError("ZIP comments, appended data and multipart archives are unsupported")
    directory_end = size - 22
    zip64 = False
    if size >= 42:
        stream.seek(size - 42)
        locator = stream.read(20)
        if locator[:4] == b"PK\x06\x07":
            zip64 = True
            _, zip64_disk, zip64_offset, disks = struct.unpack("<4sLQL", locator)
            if zip64_disk or disks != 1 or zip64_offset + 56 != size - 42:
                raise ArchiveError("The ZIP64 directory has an unsupported structure")
            stream.seek(zip64_offset)
            record = stream.read(56)
            if len(record) != 56:
                raise ArchiveError("The ZIP64 directory is truncated")
            record = struct.unpack("<4sQ2H2L4Q", record)
            if record[0] != b"PK\x06\x06" or record[1] != 44 or record[4] or record[5]:
                raise ArchiveError("The ZIP64 directory has an unsupported structure")
            disk_count, count, directory_size, offset = record[6:]
            directory_end = zip64_offset
    if not zip64 and (count == 65535 or offset == 0xFFFFFFFF or directory_size == 0xFFFFFFFF):
        raise ArchiveError("The ZIP64 directory is missing")
    if disk_count != count or not 1 < count <= MAX_FILES + 1:
        raise ArchiveLimitError("The archive has an invalid or excessive file count")
    # Names also appear in the bounded manifest. Allow fixed headers and small ZIP64
    # extras per file, without allocating a huge attacker-controlled directory first.
    if directory_size > MAX_MANIFEST_BYTES + (MAX_FILES + 1) * 192:
        raise ArchiveLimitError("The archive directory exceeds its size limit")
    if offset < 0 or directory_size < 46 * count or offset + directory_size != directory_end:
        raise ArchiveError("The ZIP directory is inconsistent")
    return count


def _members(archive, expected_count):
    entries = archive.infolist()
    if len(entries) != expected_count:
        raise ArchiveError("The ZIP directory file count is inconsistent")
    members, folded, total = {}, set(), 0
    for entry in entries:
        original = entry.orig_filename
        if "\x00" in original or original != entry.filename:
            raise ArchiveError("Archive member names must not contain NUL characters")
        name = safe_member_path(original)
        key = unicodedata.normalize("NFC", name).casefold()
        if name in members or key in folded:
            raise ArchiveError("Archive member names are duplicated or ambiguous")
        folded.add(key)
        if name != "manifest.json" and not allowed_artifact_path(name):
            raise ArchiveError("The archive contains an unsupported workspace artifact")
        mode = entry.external_attr >> 16
        if (
            entry.is_dir()
            or entry.external_attr & 0x10
            or stat.S_IFMT(mode) not in (0, stat.S_IFREG)
        ):
            raise ArchiveError("Only regular files are permitted in a workspace archive")
        if entry.flag_bits & ~0x808 or entry.compress_type != zipfile.ZIP_STORED:
            raise ArchiveError("Only unencrypted ZIP_STORED workspace archives are supported")
        if entry.compress_size != entry.file_size or entry.file_size < 0:
            raise ArchiveError("An archive member has inconsistent sizes")
        if entry.file_size > MAX_TOTAL_BYTES:
            raise ArchiveLimitError("An archive member exceeds the workspace size limit")
        if entry.comment or len(entry.extra) > 128:
            raise ArchiveError("Archive member comments or oversized metadata are unsupported")
        total += entry.file_size
        if total > MAX_TOTAL_BYTES + MAX_MANIFEST_BYTES:
            raise ArchiveLimitError("The expanded archive exceeds the workspace size limit")
        members[name] = entry
    if "manifest.json" not in members:
        raise ArchiveError("The workspace archive manifest is missing")
    for name in members:
        parents = Path(name).parents
        if any(parent.as_posix() in members for parent in parents if parent != Path(".")):
            raise ArchiveError("An archive member is both a file and a parent directory")
    return members


def _read_manifest(archive, members):
    entry = members["manifest.json"]
    if not 0 < entry.file_size <= MAX_MANIFEST_BYTES:
        raise ArchiveLimitError("The workspace archive manifest exceeds its size limit")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ArchiveError("The archive manifest contains duplicate JSON keys")
            result[key] = value
        return result

    def invalid_constant(_value):
        raise ArchiveError("The archive manifest contains a nonfinite JSON number")

    try:
        value = json.loads(
            archive.read(entry), object_pairs_hook=unique_object, parse_constant=invalid_constant
        )
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ArchiveError("The archive manifest is not valid JSON") from exc
    manifest = validate_manifest(value)
    inventory = {item["path"]: item for item in manifest["files"]}
    if set(inventory) != set(members) - {"manifest.json"} or "iris.sqlite3" not in inventory:
        raise ArchiveError("The archive file inventory is incomplete or contains extra members")
    for name, item in inventory.items():
        if members[name].file_size != item["size_bytes"]:
            raise ArchiveError("An archive member size differs from its manifest")
    return manifest, inventory


def _extract(archive, members, manifest, root, *, restore, progress, cancelled):
    count, done, total = 0, 0, manifest["total_bytes"]
    phase = "restoring" if restore else "inspecting"
    retained = [
        item
        for item in manifest["files"]
        if restore or item["path"] == "iris.sqlite3" or is_reference_document(item["path"])
    ]
    required = sum(((item["size_bytes"] + 4095) // 4096 + 1) * 4096 for item in retained)
    if shutil.disk_usage(root).free < required + 1024**2:
        raise ArchiveLimitError("Not enough free disk space to verify or restore this workspace")
    _progress(progress, phase, "Verifying workspace files", 0, total, 0, manifest["file_count"])
    for item in manifest["files"]:
        _cancel(cancelled)
        name = item["path"]
        reference = is_reference_document(name)
        if reference and item["size_bytes"] > MAX_REFERENCE_BYTES:
            raise ArchiveLimitError("A workspace reference document exceeds its size limit")
        keep = restore or name == "iris.sqlite3" or reference
        target = root / name
        destination = None
        try:
            if keep:
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                destination = target.open("xb")
                os.chmod(target, 0o600)
            digest, size = hashlib.sha256(), 0
            with archive.open(members[name]) as source:
                while True:
                    _cancel(cancelled)
                    chunk = source.read(CHUNK_BYTES)
                    if not chunk:
                        break
                    size += len(chunk)
                    done += len(chunk)
                    if size > item["size_bytes"] or done > total:
                        raise ArchiveLimitError("An archive member exceeds its declared size")
                    digest.update(chunk)
                    if destination is not None:
                        destination.write(chunk)
                    _progress(
                        progress,
                        phase,
                        "Verifying workspace files",
                        done,
                        total,
                        count,
                        manifest["file_count"],
                    )
            if size != item["size_bytes"] or digest.hexdigest() != item["sha256"]:
                raise ArchiveError("An archive member does not match its recorded SHA-256")
            if destination is not None:
                destination.flush()
                os.fsync(destination.fileno())
        finally:
            if destination is not None:
                destination.close()
        count += 1
        _progress(
            progress,
            phase,
            "Verifying workspace files",
            done,
            total,
            count,
            manifest["file_count"],
        )


def _database(root, inventory, manifest):
    checked = validate_database(root, inventory, verify_hashes=False)
    if checked["counts"] != manifest["counts"]:
        raise ArchiveError("Database row counts differ from the archive manifest")
    if checked["schema_version"] != manifest["schema_version"]:
        raise ArchiveError("Database schema differs from the archive manifest")
    if checked["active_jobs"]:
        raise ArchiveError("An archive must not contain queued or running jobs")
    for name, expected in checked["expected_hashes"].items():
        if name not in inventory or inventory[name]["sha256"] != expected:
            raise ArchiveError("A database artifact hash differs from the archive inventory")


def _verify_into(archive_path, root, *, expected, restore, progress, cancelled):
    try:
        with _open_archive(archive_path) as stream:
            signature = _signature(stream)
            identity = _digest(
                stream, progress=progress, cancelled=cancelled, phase="checking_archive"
            )
            if expected is not None and identity != expected:
                raise ArchiveError("The archive has changed since its verified preview")
            if _signature(stream) != signature:
                raise ArchiveError("The archive changed while its identity was checked")
            count = _directory_bounds(stream)
            with zipfile.ZipFile(stream, "r", allowZip64=True) as archive:
                members = _members(archive, count)
                manifest, inventory = _read_manifest(archive, members)
                _extract(
                    archive,
                    members,
                    manifest,
                    root,
                    restore=restore,
                    progress=progress,
                    cancelled=cancelled,
                )
            _cancel(cancelled)
            _progress(progress, "checking_database", "Verifying database and file references", 0, 0)
            _database(root, inventory, manifest)
            final_identity = _digest(
                stream, progress=progress, cancelled=cancelled, phase="checking_archive"
            )
            if final_identity != identity or _signature(stream) != signature:
                raise ArchiveError("The archive changed during verification")
            _cancel(cancelled)
            return {
                "manifest": manifest,
                "archive_size_bytes": signature[2],
                "archive_sha256": identity,
                "verified": True,
            }
    except (zipfile.BadZipFile, zipfile.LargeZipFile, EOFError, struct.error) as exc:
        raise ArchiveError("The ZIP archive is corrupt or unsupported") from exc


def inspect_archive(archive: Path, *, progress=None, cancelled=None) -> dict:
    """Verify every byte and database reference without retaining extracted media."""
    _cancel(cancelled)
    with tempfile.TemporaryDirectory(prefix="iris-workspace-inspect-") as directory:
        return _verify_into(
            Path(archive),
            Path(directory),
            expected=None,
            restore=False,
            progress=progress,
            cancelled=cancelled,
        )


def _publish(staging: Path, destination: Path):
    """Atomically publish with Linux renameat2; an existing target is never replaced."""
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, "renameat2", None)
    if rename is None:
        raise ArchiveError("Safe workspace publication requires Linux renameat2 support")
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    descriptor = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        result = rename(
            descriptor, os.fsencode(staging.name), descriptor, os.fsencode(destination.name), 1
        )
        if result:
            error = ctypes.get_errno()
            if error in (errno.EEXIST, errno.ENOTEMPTY):
                raise ArchiveError("The destination already exists; nothing was overwritten")
            if error in (errno.ENOSYS, errno.EINVAL, errno.ENOTSUP):
                raise ArchiveError("This filesystem does not support safe workspace publication")
            raise ArchiveError("The restored workspace could not be published")
        try:
            os.fsync(descriptor)
        except OSError:
            # Publication has already succeeded. Some filesystems cannot fsync a
            # directory; reporting failure here would hide an existing restored copy.
            pass
    finally:
        os.close(descriptor)


def restore_archive(
    archive: Path,
    destination: Path,
    *,
    expected_archive_sha256: str,
    progress=None,
    cancelled=None,
    forbidden_root: Path | None = None,
) -> dict:
    """Restore exactly a verified archive, creating a private, previously absent directory."""
    if (
        not isinstance(expected_archive_sha256, str)
        or len(expected_archive_sha256) != 64
        or any(value not in "0123456789abcdef" for value in expected_archive_sha256)
    ):
        raise ArchiveError("Restore requires the SHA-256 from a verified archive preview")
    _cancel(cancelled)
    requested = Path(destination).absolute()
    if os.path.lexists(requested):
        raise ArchiveError("The destination already exists; nothing was overwritten")
    try:
        parent = requested.parent.resolve(strict=True)
    except OSError as exc:
        raise ArchiveError("The destination parent directory must already exist") from exc
    if not parent.is_dir() or requested.name in ("", ".", ".."):
        raise ArchiveError("Choose a new workspace directory inside an existing parent")
    target = parent / requested.name
    if forbidden_root is not None:
        forbidden = Path(forbidden_root).resolve()
        if target.is_relative_to(forbidden) or forbidden.is_relative_to(target):
            raise ArchiveError("Restore must create a separate workspace directory")
    staging = Path(tempfile.mkdtemp(prefix=".iris-restore-", dir=parent))
    try:
        result = _verify_into(
            Path(archive),
            staging,
            expected=expected_archive_sha256,
            restore=True,
            progress=progress,
            cancelled=cancelled,
        )
        _progress(progress, "publishing", "Publishing the verified workspace", 0, 0)
        _cancel(cancelled)
        _publish(staging, target)
        return {**result, "path": target}
    finally:
        if staging.exists():
            shutil.rmtree(staging)
