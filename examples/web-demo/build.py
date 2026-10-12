#!/usr/bin/env python3
"""Build public demo records through Iris's API in a disposable workspace.

Requires the project's development dependencies. No network, model execution,
existing workspace, provider configuration or deployment is needed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

from fastapi.testclient import TestClient
from public_contract import (
    ANNOTATION,
    ASSET,
    FRAME,
    INSIGHTS,
    PROJECT,
    REVIEW_QUEUE,
    SESSION,
    TAXONOMIES,
    project,
)
from recording import DETAIL, MODEL, SUMMARY, load_recording, restore_recording

from iris import __version__
from iris.app import create_app
from iris.store import SCHEMA_VERSION

DIRECTORY = Path(__file__).resolve().parent
SOURCE = DIRECTORY.parent / "street-scenes"
PROJECT_ID = "web-demo"
SESSION_ID = "street-scenes"
SEED_TIME = "2026-10-12T00:00:00+00:00"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value) -> bytes:
    return (
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def request(client, method, path, **kwargs):
    response = client.request(method, path, **kwargs)
    response.raise_for_status()
    return response.json()


def source_files():
    """Read only the named, hash-verified members of the public source bundle."""
    provenance = json.loads((SOURCE / "provenance.json").read_text())
    payload = (SOURCE / "street-scenes.zip").read_bytes()
    with zipfile.ZipFile(SOURCE / "street-scenes.zip") as archive:
        expected = {**provenance["files"]}
        expected.update({item["archive_path"]: item for item in provenance["images"]})
        if sorted(archive.namelist()) != sorted(expected):
            raise ValueError("Unexpected members in the public street-scenes archive")
        for name, record in expected.items():
            data = archive.read(name)
            if len(data) != record["size_bytes"] or digest(data) != record["sha256"]:
                raise ValueError(f"Changed public source: {name}")
            if name in provenance["files"] and data != (SOURCE / name).read_bytes():
                raise ValueError(f"Public source and ZIP disagree: {name}")
    return payload, provenance


def import_source(client):
    """Create only the public example, using the same import handlers as the UI."""
    payload, provenance = source_files()
    public_project = request(
        client,
        "POST",
        "/api/projects",
        json={
            "name": "Street scenes",
            "description": "Public annotation example and recorded model comparison.",
        },
    )
    client.params = {"project_id": public_project["id"]}
    preview = request(
        client,
        "POST",
        "/api/dataset-imports",
        files={"file": ("street-scenes.zip", payload, "application/zip")},
    )
    result = request(
        client,
        "POST",
        f"/api/dataset-imports/{preview['id']}/commit",
        json={
            "name": "Street scenes",
            "scene_group": "Web demo",
            "source_url": provenance["annotation_source"]["mirror"],
            "license_name": "Photos CC BY-SA 2.0; COCO annotations CC BY 4.0.",
            "attribution": (
                "COCO Consortium; Clongriffin by infomatique; Damaged Traffic Light - "
                "Junction of High Road & White Hart Lane by Alan Stanton. "
                "See the included CREDITS.txt for photo and license links."
            ),
            "source_split": "val",
            "category_mapping": provenance["default_import_mapping"],
        },
    )
    if result["proposal_count"] != 11 or result["excluded_annotation_count"] != 8:
        raise ValueError("The public sample's import contract changed")
    sid = result["session_id"]
    return {
        "project": public_project,
        "session": request(client, "GET", f"/api/sessions/{sid}"),
        "assets": request(client, "GET", f"/api/sessions/{sid}/assets"),
        "frames": request(client, "GET", f"/api/sessions/{sid}/frames"),
        "import_id": preview["id"],
        "import_created_at": preview["created_at"],
    }


def normalize(value, identities, timestamps):
    """Replace temporary identities, retaining COCO IDs and all measured values."""
    if isinstance(value, dict):
        return {
            identities.get(key, key): normalize(item, identities, timestamps)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [normalize(item, identities, timestamps) for item in value]
    if isinstance(value, str):
        if value in identities:
            return identities[value]
        if value in timestamps:
            return SEED_TIME
        if value.startswith("/api/"):
            return "/".join(identities.get(part, part) for part in value.split("/"))
    return value


def build() -> dict[str, bytes]:
    """Return a deterministic, explicit inventory; never inspect a user's workspace."""
    archive, source_provenance = source_files()
    recording = load_recording()
    files = {}
    responses = {}
    resources = {}
    with tempfile.TemporaryDirectory(prefix="iris-public-demo-") as directory:
        app = create_app(Path(directory), run_jobs=False)
        with TestClient(app, base_url="http://127.0.0.1") as client:
            imported = import_source(client)
            public_project, session = imported["project"], imported["session"]
            assets, frames = imported["assets"], imported["frames"]
            frames.sort(key=lambda row: row["extraction"]["dataset_import"]["coco_image"]["id"])
            assets.sort(key=lambda row: row["metadata"]["dataset_import"]["coco_image"]["id"])
            frames_by_coco = {
                row["extraction"]["dataset_import"]["coco_image"]["id"]: row for row in frames
            }
            cid, _ = restore_recording(app.state.store, recording, session["id"], frames_by_coco)
            identities = {
                public_project["id"]: PROJECT_ID,
                session["id"]: SESSION_ID,
                imported["import_id"]: "street-scenes-import",
            }
            timestamps = {
                public_project["created_at"],
                session["created_at"],
                imported["import_created_at"],
            }
            for coco_id, frame in frames_by_coco.items():
                identities[frame["id"]] = f"frame-{coco_id}"
                identities[frame["asset_id"]] = f"asset-{coco_id}"
                timestamps.add(frame["created_at"])
            for asset in assets:
                timestamps.add(asset["created_at"])

            def get(path):
                return request(client, "GET", path)

            pid, sid = public_project["id"], session["id"]
            responses["/api/projects"] = project([public_project], [PROJECT])
            responses[f"/api/projects/{pid}"] = project(public_project, PROJECT)
            responses[f"/api/projects/{pid}/taxonomies"] = project(
                get(f"/api/projects/{pid}/taxonomies"), TAXONOMIES
            )
            responses["/api/sessions"] = project([session], [SESSION])
            responses[f"/api/sessions/{sid}"] = project(session, SESSION)
            responses[f"/api/sessions/{sid}/assets"] = project(assets, [ASSET])
            responses[f"/api/sessions/{sid}/frames"] = project(frames, [FRAME])
            for part, schema in (("selection-insights", INSIGHTS), ("review-queue", REVIEW_QUEUE)):
                value = get(f"/api/sessions/{sid}/{part}")
                # Equal-priority imported frames otherwise tie on fresh random IDs.
                key = "frame_id" if part == "selection-insights" else "id"
                value["frames"].sort(key=lambda row: identities[row[key]])
                responses[f"/api/sessions/{sid}/{part}"] = project(value, schema)
            for coco_id, frame in frames_by_coco.items():
                fid, aid = frame["id"], frame["asset_id"]
                document = get(f"/api/frames/{fid}/annotation")
                document["suggestions"].sort(
                    key=lambda row: row["metadata"]["source_annotation"]["id"]
                )
                for suggestion in document["suggestions"]:
                    source_id = suggestion["metadata"]["source_annotation"]["id"]
                    identities[suggestion["id"]] = f"suggestion-coco-{source_id}"
                    timestamps.add(suggestion["created_at"])
                responses[f"/api/frames/{fid}/annotation"] = project(document, ANNOTATION)
                responses[f"/api/frames/{fid}/assistance"] = project(
                    get(f"/api/frames/{fid}/assistance"), []
                )
                for endpoint, filename, media_type in (
                    (f"/api/frames/{fid}/image", f"media/{coco_id}.png", "image/png"),
                    (f"/api/assets/{aid}/media", f"media/{coco_id}.jpg", "image/jpeg"),
                ):
                    response = client.get(endpoint)
                    response.raise_for_status()
                    if response.headers["content-type"] != media_type:
                        raise ValueError(f"Unexpected media type for {filename}")
                    files[filename] = response.content
                    resources[endpoint] = filename

            def comparison_record(value, schema):
                result = project(value, schema)
                # The only dynamic object keys allowed here are these verified frame IDs.
                result["config"]["frame_hashes"] = project(
                    value["config"]["frame_hashes"], {frame["id"]: str for frame in frames}
                )
                return result

            responses[f"/api/sessions/{sid}/comparisons"] = [
                comparison_record(value, SUMMARY)
                for value in get(f"/api/sessions/{sid}/comparisons")
            ]
            responses[f"/api/comparisons/{cid}"] = comparison_record(
                get(f"/api/comparisons/{cid}"), DETAIL
            )
            responses["/api/models"] = project(recording["models"], [MODEL])

            # Paths are normalized separately because response-map keys are full URLs.
            responses = {
                normalize(path, identities, timestamps): normalize(value, identities, timestamps)
                for path, value in responses.items()
            }
            resources = {
                normalize(path, identities, timestamps): value for path, value in resources.items()
            }

    files["fixture.json"] = json_bytes(
        {
            "format": "iris-web-demo-v1",
            "fixture_version": 1,
            "project_id": PROJECT_ID,
            "session_id": SESSION_ID,
            "comparison_id": cid,
            "responses": responses,
            "resources": resources,
            "credits": "CREDITS.txt",
            "provenance": "provenance.json",
        }
    )
    files["CREDITS.txt"] = (SOURCE / "CREDITS.txt").read_bytes() + (
        b"\nWEB DEMO DERIVATIVES\n\n"
        b"The media/*.jpg files retain the original JPEG bytes. The corresponding\n"
        b"PNG files are Iris's RGB-normalized display images, at original dimensions.\n"
        b"No crop, drawn box or retouch is baked into these photographs. The PNGs\n"
        b"carry the same photo credits and CC BY-SA 2.0 license as the originals.\n"
        b"COCO labels remain imported proposals, without a human review. Recorded\n"
        b"detector outputs are separate from these labels. See provenance.json.\n"
    )
    files["provenance.json"] = json_bytes(
        {
            "format": "iris-web-demo-provenance-v1",
            "iris_version": __version__,
            "store_schema_version": SCHEMA_VERSION,
            "source_archive_sha256": digest(archive),
            "source": source_provenance,
            "recording_sha256": digest((DIRECTORY / "recording.json").read_bytes()),
            "recording_provenance": recording["provenance"],
            "normalization": {
                "identities": "Temporary import identities replaced with stable public sample IDs.",
                "import_created_at": SEED_TIME,
                "timestamps": (
                    "Import timestamps are fixture placeholders. Recorded inference dates and "
                    "timings are retained from the actual run."
                ),
                "order": "Imported frames, assets and proposals use stable source-ID ordering.",
                "predictions": "All recorded native detections, coordinates and scores retained.",
            },
            "files": {
                name: {"sha256": digest(data), "size_bytes": len(data)}
                for name, data in sorted(files.items())
            },
        }
    )
    return files


def write_output(files, output):
    """An existing directory is only accepted when it is already byte-identical."""
    if output.exists():
        actual = {
            path.relative_to(output).as_posix(): path.read_bytes()
            for path in output.rglob("*")
            if path.is_file()
        }
        if actual != files or any(path.is_symlink() for path in output.rglob("*")):
            raise FileExistsError("Output differs; choose a new directory and review the changes")
        return
    output.mkdir(parents=True)
    for name, data in files.items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DIRECTORY / "seed")
    parser.add_argument(
        "--check", action="store_true", help="Compare with the output; write nothing"
    )
    args = parser.parse_args()
    files = build()
    if args.check and not args.output.is_dir():
        raise FileNotFoundError("Output directory is missing")
    write_output(files, args.output)
    print(
        f"Verified {len(files)} public files" if args.check else f"Built {len(files)} public files"
    )


if __name__ == "__main__":
    main()
