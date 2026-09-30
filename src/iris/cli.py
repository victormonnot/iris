import argparse
import fcntl
import json
import os
from pathlib import Path

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Run the local IRIS workbench")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--data-dir", type=Path, default=Path(os.getenv("IRIS_DATA_DIR", ".iris")))
    commands = parser.add_subparsers(dest="command")
    models = commands.add_parser(
        "models", help="Inspect or explicitly download local detector weights"
    )
    actions = models.add_subparsers(dest="model_action", required=True)
    listing = actions.add_parser("list", help="List model readiness without network access")
    download = actions.add_parser("download", help="Download official weights (no user data sent)")
    download.add_argument("model_ids", nargs="*")
    download.add_argument("--all", action="store_true", help="Download both official checkpoints")
    for action in (listing, download):
        action.add_argument("--data-dir", type=Path, default=argparse.SUPPRESS)
    workspace = commands.add_parser(
        "workspace", help="Back up, verify or restore a local workspace"
    )
    transfers = workspace.add_subparsers(dest="workspace_action", required=True)
    backup = transfers.add_parser(
        "backup", help="Create a verified ZIP after stopping this workspace"
    )
    backup.add_argument("destination", type=Path)
    backup.add_argument("--data-dir", type=Path, default=argparse.SUPPRESS)
    inspect = transfers.add_parser(
        "inspect", help="Verify an IRIS workspace archive without restoring"
    )
    inspect.add_argument("archive", type=Path)
    restore = transfers.add_parser("restore", help="Restore to a new folder, without starting IRIS")
    restore.add_argument("archive", type=Path)
    restore.add_argument("--to", type=Path, required=True, dest="destination")
    args = parser.parse_args()
    if args.command == "workspace":
        from iris.workspace_archive import ArchiveError, create_archive
        from iris.workspace_restore import inspect_archive, restore_archive

        try:
            if args.workspace_action == "backup":
                root = args.data_dir.resolve()
                if not (root / "iris.sqlite3").is_file():
                    raise ArchiveError("The source is not an initialized IRIS workspace")
                with (root / ".server.lock").open("a") as lock:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except OSError:
                        raise ArchiveError(
                            "This workspace is open. Use Workspace backup in the app, "
                            "or stop its server before using the command line."
                        ) from None
                    receipt = create_archive(root, args.destination.absolute())
            else:
                receipt = inspect_archive(args.archive.absolute())
                if args.workspace_action == "restore":
                    receipt = restore_archive(
                        args.archive.absolute(),
                        args.destination.absolute(),
                        expected_archive_sha256=receipt["archive_sha256"],
                    )
            manifest = receipt.pop("manifest")
            receipt["summary"] = {
                key: manifest[key]
                for key in ("app_version", "schema_version", "file_count", "total_bytes", "counts")
            }
            print(json.dumps(receipt, indent=2, default=str))
        except (ArchiveError, OSError) as exc:
            parser.exit(1, f"Workspace transfer failed: {exc}\n")
        return
    if args.command == "models":
        from iris.models import catalog, download_model, get_spec

        root = args.data_dir.resolve()
        if args.model_action == "list":
            print(json.dumps(catalog(root), indent=2))
        else:
            if args.all and args.model_ids:
                parser.error("Choose --all or explicit model IDs, not both")
            model_ids = (
                [model["id"] for model in catalog(root) if model.get("origin") != "trained"]
                if args.all
                else args.model_ids
            )
            if not model_ids:
                parser.error("Specify model IDs or --all")
            try:
                specs = [get_spec(model_id) for model_id in dict.fromkeys(model_ids)]
                for spec in specs:
                    print(
                        f"Preparing {spec['name']} ({spec['download_bytes']:,} bytes)", flush=True
                    )
                    receipt = download_model(root, spec["id"])
                    print(f"Verified SHA-256: {receipt['weight_sha256']}", flush=True)
            except (ValueError, OSError) as exc:
                parser.exit(1, f"Model setup failed: {exc}\n")
        return
    os.environ["IRIS_DATA_DIR"] = str(args.data_dir.resolve())
    uvicorn.run("iris.app:create_app", factory=True, host="127.0.0.1", port=args.port)
