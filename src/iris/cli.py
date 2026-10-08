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
    download.add_argument("--all", action="store_true", help="Download all official checkpoints")
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
    tracking = commands.add_parser(
        "tracking", help="Replay saved temporal detections with an optional local tracker"
    )
    tracking_actions = tracking.add_subparsers(dest="tracking_action", required=True)
    tracking_actions.add_parser("status", help="Inspect optional tracker package readiness")
    replay = tracking_actions.add_parser("replay", help="Write a complete standalone replay report")
    replay.add_argument("--cache-id", required=True)
    replay.add_argument("--output", type=Path, required=True)
    choice = replay.add_mutually_exclusive_group(required=True)
    choice.add_argument("--tracker", choices=("bytetrack", "botsort"))
    choice.add_argument("--profile", type=Path, help="Complete versioned tracker profile JSON")
    replay.add_argument("--class-id", type=int, action="append", dest="class_ids")
    replay.add_argument("--repeats", type=int, default=2, choices=range(1, 6))
    replay.add_argument("--data-dir", type=Path, default=argparse.SUPPRESS)
    measure = tracking_actions.add_parser(
        "measure", help="Measure a fresh detector and tracker pipeline"
    )
    measure.add_argument("--comparison-id", required=True)
    measure.add_argument("--lane-index", type=int, choices=(0, 1), default=0)
    measure.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    measure.add_argument("--repeats", type=int, default=1, choices=range(1, 6))
    measure.add_argument(
        "--policy", choices=("offline_all", "simulated_latest"), default="offline_all"
    )
    measure.add_argument("--cadence-fps", type=float)
    measure.add_argument("--output", type=Path, required=True)
    measure.add_argument("--data-dir", type=Path, default=argparse.SUPPRESS)
    pipeline = commands.add_parser("pipeline", help="Inspect a portable experimental pipeline ZIP")
    pipeline_actions = pipeline.add_subparsers(dest="pipeline_action", required=True)
    pipeline_inspect = pipeline_actions.add_parser(
        "inspect", help="Verify contracts and hashes without executing bundled code"
    )
    pipeline_inspect.add_argument("archive", type=Path)
    pipeline_extract = pipeline_actions.add_parser(
        "extract", help="Verify and extract to a new directory without executing bundled code"
    )
    pipeline_extract.add_argument("archive", type=Path)
    pipeline_extract.add_argument("--to", type=Path, required=True, dest="destination")
    args = parser.parse_args()
    if args.command == "pipeline":
        from iris.pipeline_bundle_contracts import extract_bundle, inspect_bundle

        try:
            result = (
                extract_bundle(args.archive, args.destination)
                if args.pipeline_action == "extract"
                else inspect_bundle(args.archive)
            )
            print(json.dumps(result, indent=2))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            parser.exit(1, f"Pipeline inspection/extraction failed: {exc}\n")
        return
    if args.command == "tracking":
        if args.tracking_action == "status":
            from iris.tracking import tracking_status

            print(json.dumps(tracking_status(), indent=2))
            return
        if args.tracking_action == "measure":
            from iris.tracking_costs import measure_to_file
            from iris.tracking_replay import ReadOnlyReplayStore

            try:
                report = measure_to_file(
                    ReadOnlyReplayStore(args.data_dir),
                    args.comparison_id,
                    args.output,
                    lane_index=args.lane_index,
                    device=args.device,
                    repeats=args.repeats,
                    policy=args.policy,
                    cadence_fps=args.cadence_fps,
                )
                print(
                    json.dumps(
                        {
                            "report": str(args.output.absolute()),
                            "complete": report["complete"],
                            "summary": report["summary"],
                        },
                        indent=2,
                    )
                )
            except (ValueError, KeyError, OSError, RuntimeError, ImportError) as exc:
                parser.exit(1, f"Tracking measurement failed: {exc}\n")
            return
        from iris.tracking_replay import ReadOnlyReplayStore, replay_to_file

        if args.profile is not None and args.class_ids is not None:
            parser.error("--class-id cannot override a complete --profile")
        try:
            profile = (
                json.loads(args.profile.read_text(encoding="utf-8"))
                if args.profile is not None
                else None
            )
            report = replay_to_file(
                ReadOnlyReplayStore(args.data_dir),
                args.cache_id,
                args.output,
                algorithm=args.tracker,
                profile=profile,
                class_ids=args.class_ids,
                repeats=args.repeats,
            )
            print(
                json.dumps(
                    {
                        "report": str(args.output.absolute()),
                        "complete": report["complete"],
                        "frames": len(report["sequence"]["frames"]),
                        "repeatability": report["repeatability"],
                    },
                    indent=2,
                )
            )
        except (ValueError, KeyError, OSError, RuntimeError, ImportError) as exc:
            parser.exit(1, f"Tracking replay failed: {exc}\n")
        return
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
