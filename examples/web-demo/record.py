#!/usr/bin/env python3
"""Capture genuine public-example predictions with locally provisioned CPU weights.

The output must be a new JSON file. This optional maintenance command executes
real models; the normal fixture build reuses recording.json without inference.
No installed workspace or annotation review is imported. Nothing is downloaded.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from build import SOURCE, import_source, request, source_files
from fastapi.testclient import TestClient
from recording import MODEL_IDS, make_recording

from iris import __version__
from iris.app import create_app
from iris.inference import run_comparison
from iris.models import get_spec
from iris.store import now

REPOSITORY = Path(__file__).resolve().parents[2]


def file_digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def deny_network(event, _args):
    if event == "socket.connect":
        raise RuntimeError("The public recording command does not permit network connections")


def record(weights_dir, output):
    """Run the two official detectors in a disposable workspace, retaining measured values."""
    if output.exists():
        raise FileExistsError(
            "Choose a new recording file; existing captures are never overwritten"
        )
    # Fail before execution if the public source bundle changed or a weight is missing.
    archive, _source = source_files()
    verified_weights = []
    for model_id in MODEL_IDS:
        spec = get_spec(model_id)
        path = weights_dir / spec["weight_filename"]
        digest = file_digest(path)
        if path.stat().st_size != spec["download_bytes"] or not digest.startswith(
            spec["expected_hash_prefix"]
        ):
            raise ValueError(f"Weight does not match the official artifact: {model_id}")
        verified_weights.append((path, spec["weight_filename"]))

    started_at = now()
    with tempfile.TemporaryDirectory(prefix="iris-demo-recording-") as directory:
        workspace = Path(directory)
        (workspace / "models").mkdir()
        for source, filename in verified_weights:
            shutil.copyfile(source, workspace / "models" / filename)
        app = create_app(workspace, run_jobs=False)
        store = app.state.store
        with TestClient(app, base_url="http://127.0.0.1") as client:
            imported = import_source(client)
            frames = sorted(
                imported["frames"],
                key=lambda row: row["extraction"]["dataset_import"]["coco_image"]["id"],
            )
            assets = {asset["id"]: asset for asset in imported["assets"]}
            bindings = []
            for frame in frames:
                asset = assets[frame["asset_id"]]
                bindings.append(
                    {
                        "frame_id": frame["id"],
                        "asset_id": asset["id"],
                        "coco_image_id": asset["metadata"]["dataset_import"]["coco_image"]["id"],
                        "source_filename": asset["filename"],
                        "source_sha256": asset["sha256"],
                        "frame_pixel_sha256": frame["sha256"],
                        "width": frame["width"],
                        "height": frame["height"],
                    }
                )
            summary = request(
                client,
                "POST",
                f"/api/sessions/{imported['session']['id']}/comparisons",
                json={
                    "name": "Street scenes · SSDLite320 vs Faster R-CNN 320 · CPU",
                    "frame_ids": [frame["id"] for frame in frames],
                    "model_ids": list(MODEL_IDS),
                    "device": "cpu",
                    "inference_mode": "full",
                },
            )
            job_id = summary["job_id"]
            store.update("jobs", job_id, {"status": "running", "started_at": now()})

            def progress(value, message):
                store.update("jobs", job_id, {"progress": value, "message": message})
                print(message, flush=True)

            result = run_comparison(store, summary["id"], progress, lambda: False)
            if result["predictions_created"] != 4 or result["cancelled"]:
                raise ValueError("The comparison did not finish all four model/image results")
            store.update(
                "jobs",
                job_id,
                {
                    "status": "succeeded",
                    "progress": 1.0,
                    "result": result,
                    "message": "Comparison complete",
                    "finished_at": now(),
                },
            )
            if store.list("annotation_revisions"):
                raise ValueError("The public capture must not contain annotation reviews")
            detail = request(client, "GET", f"/api/comparisons/{summary['id']}")
            models = [
                model for model in request(client, "GET", "/api/models") if model["id"] in MODEL_IDS
            ]
            provenance = {
                "started_at": started_at,
                "completed_at": now(),
                "iris_version": __version__,
                "iris_git_commit": subprocess.check_output(
                    ["git", "-C", str(REPOSITORY), "rev-parse", "HEAD"],
                    text=True,
                ).strip(),
                "runtime_versions": {
                    name: importlib.metadata.version(name)
                    for name in (
                        "torch",
                        "torchvision",
                        "pillow",
                        "numpy",
                    )
                },
                "inference_source_sha256": {
                    name: file_digest(REPOSITORY / "src" / "iris" / name)
                    for name in (
                        "inference.py",
                        "models.py",
                        "media.py",
                        "store.py",
                        "coco_import.py",
                    )
                },
                "archive_sha256": hashlib.sha256(archive).hexdigest(),
                "source_provenance_sha256": file_digest(SOURCE / "provenance.json"),
                "network_used": False,
                "human_annotation_reviews": 0,
                "model_filtering": (
                    "All native model detections are saved; UI filtering has not been applied."
                ),
                "timing_scope": (
                    "Real CPU measurements from this capture, with the recorded thread counts; "
                    "not browser or user-device performance promises."
                ),
            }
            captured = make_recording(detail, bindings, models, provenance)
    with output.open("x") as stream:
        json.dump(captured, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights-dir",
        type=Path,
        required=True,
        help="Directory containing both official .pth files",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New recording JSON file; never overwritten"
    )
    args = parser.parse_args()
    sys.addaudithook(deny_network)
    try:
        path = record(args.weights_dir, args.output)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Cannot record the public example: {error}\n")
    print(f"Recorded genuine CPU predictions in {path}")


if __name__ == "__main__":
    main()
