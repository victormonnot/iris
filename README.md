# IRIS

A local computer vision workbench for drone imagery. IRIS is being built around
the full improvement loop: flight data → frame selection → assisted annotation →
human validation → versioned dataset → fine-tuning → model comparison.

The first increment provides **working data intake and frame selection**.
Inference, annotation, dataset releases, training, and evaluation are planned
parts of V1; they are not available yet. See [the architecture and V1 plan](docs/architecture.md).

## Run locally

Requires Linux or WSL, Python 3.12 or 3.13, and [uv](https://docs.astral.sh/uv/).
No GPU, model weights, API key, drone connection, or external account is needed
for data intake.

```sh
uv sync --locked
uv run iris
```

Open <http://127.0.0.1:8000>. The server binds to loopback only. Run one instance
per data directory; a workspace lock prevents concurrent worker managers.

```sh
uv run iris --port 8001 --data-dir /path/to/private/iris-data
```

By default, media, the SQLite database, and worker logs stay in `.iris/`, which
is excluded from Git. `IRIS_DATA_DIR` also sets the data location. Original
files are copied into this workspace; keep enough disk space for both originals
and extracted PNGs. Back up the whole data directory while the server is stopped.

## First workflow

1. Create a flight session and give related flights the same **scene group**.
   These groups will be used to separate training, validation, and test data.
2. Import images or videos. Each upload is limited to 2 GiB. Sources are kept
   with a SHA-256 checksum; importing an identical file into the same session
   is idempotent. No data leaves this machine.
3. For a video, choose a time interval, sampling step, and maximum frame count
   (up to 500 per extraction). Extraction runs in a separate process.
4. Browse frames, inspect their source and timestamp, and select useful images.
   Selection is persisted; it does **not** mean the image has been annotated
   or human-validated.
5. Inspect job progress, errors, and logs. Cancel extraction if needed; frames
   already produced remain available. After a server interruption, unfinished
   jobs are marked interrupted. Re-run extraction explicitly to recover.

Exact duplicate frames are skipped during extraction. Optional perceptual
deduplication is only a heuristic: inspect the retained frames before relying
on a selection. Identical images across sessions remain visible as duplicate
warnings; this is not yet a dataset split validator.

Supported video containers are AVI, MP4/M4V, modern MOV, MKV, and WebM, subject
to OpenCV's bundled codecs. Playlists, MPEG containers, and older MOV files
without a file-type header are rejected. Browser playback depends on the
browser's own codec support; a video can be extractable without being playable
in the browser. Timestamps currently derive from frame index and reported FPS;
they are approximate for variable-frame-rate recordings. There is no telemetry
alignment or live capture.

## Development and verification

```sh
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Tests generate small synthetic images and videos in temporary directories.
They exercise ingestion, provenance, extraction, deduplication, selection,
job lifecycle, cancellation, and persistence. They establish software behavior,
not detection quality on real drone data. There are no bundled datasets,
model predictions, or claimed training results.

The web UI and API are served by FastAPI; metadata lives in SQLite and media
in local files. The browser uses plain JavaScript without external assets.
The API schema is available at `/openapi.json`. The application works offline
after installation and does not load documentation assets from a CDN.

Future ML dependencies and model downloads will be optional. Model and dataset
licenses must be checked independently from framework licenses.
