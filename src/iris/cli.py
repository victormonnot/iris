import argparse
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
    args = parser.parse_args()
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
