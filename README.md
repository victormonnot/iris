# IRIS

A local computer vision workbench for drone imagery. IRIS is being built around
the full improvement loop: flight data → frame selection → assisted annotation →
human validation → versioned dataset → fine-tuning → model comparison.

IRIS currently provides **data intake, frame selection, saved detector comparisons,
and human annotation with optional local multimodal assistance**. Dataset releases,
training, and quality evaluation remain planned parts of V1. See
[the architecture and V1 plan](docs/architecture.md).

## Run locally

Requires Linux or WSL, Python 3.12 or 3.13, and [uv](https://docs.astral.sh/uv/).
No GPU, model weights, API key, drone connection, or external account is needed
for data intake or manual annotation.

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

## Compare detectors

The optional CPU runtime uses PyTorch 2.10.0 and Torchvision 0.25.0. Install it
explicitly; the base installation never downloads model weights or ML packages:

```sh
uv sync --locked --extra ml
uv run --extra ml iris models download --all
uv run --extra ml iris
```

These commands download roughly 300 MB in total (CPU packages and two official
COCO checkpoints). The checkpoints alone total 91,914,162 bytes. No user images
are sent to a provider. Downloads go into the chosen workspace's `models/`
directory and are verified against the published SHA-256 prefixes. Setup is
atomic and repeatable. Keep using `--extra ml` with `uv run` / `uv sync` to retain
the optional runtime in the managed environment.

```sh
uv run --extra ml iris models list
uv run --extra ml iris models download --all --data-dir /path/to/private/iris-data
uv run --extra ml iris --data-dir /path/to/private/iris-data
```

1. Select 1–100 frames in **Data intake**, then open **Model comparison**.
2. Choose one or both ready models, name the comparison, and start it. CPU is the
   default. CUDA is available only with a separately provisioned compatible
   runtime; the optional `ml` installation intentionally contains CPU wheels.
3. Inspect the same frame side by side. Changing the displayed confidence or
   class filters saved predictions; it does not rerun the models. Raw saved
   outputs include all returned COCO categories, before UI filtering.
4. Reopen the comparison from its history after a restart. It retains its frame
   selection, frame hashes, checkpoint hashes, model configuration, runtime,
   device, and timing protocol. Subsequent selection changes do not change it.

Inference executes in the local job worker. Each completed model/frame result
is saved independently, including an explicit empty detection list. A cancelled
or failed job keeps its partial outputs; **Not processed** is distinct from
**No detections**. Start a new comparison to retry. Saved results remain readable
when weights or the optional runtime are unavailable.

Timings use one excluded warmup per model, batch size one, float32, and at most
four CPU threads. Decode, tensor preparation, full model forward, and result
serialization are recorded separately. **Model forward includes Torchvision's
internal normalization, resizing, proposal filtering, NMS, and coordinate
restoration**; it is not a backbone-only measurement. Total time includes decode
and hash verification but excludes loading weights, warmup, and database writes.
CUDA runs synchronize at the measurement boundaries. Inspect the saved metadata
for the exact resolution, native thresholds, and hardware.

Both models use a native score cutoff of 0.001, NMS IoU 0.5, and at most 100
detections per image. These settings define what can be saved; lowering the UI
threshold cannot recover outputs below that native cutoff. Models still have
different internal proposal algorithms. Confidence, count differences, and
runtime are not quality metrics. Precision, recall, and mAP require a validated
reference dataset and are not reported by this increment.

## Annotate and review

Open **Annotation** for a selected frame. The initial taxonomy, `iris-objects-v1`,
contains **person** and **car**; the interface shows their exact definitions.
Draw boxes, move or resize them, or edit their pixel coordinates. Saved detector
outputs can be imported as proposals without running inference again.

Accept or reject proposals individually, correct labels and geometry, and inspect
the whole image for missed objects. **Save draft** records work in progress.
**Validate frame** requires a reviewer and a decision for every pending proposal;
an empty validated frame explicitly records that no target objects are visible.
Every save creates a revision with provenance. Concurrent edits produce a reload
conflict instead of silently replacing another revision. Editing a validated frame
requires a fresh human validation.

### Optional local multimodal assistance

Install [Ollama](https://docs.ollama.com/linux) separately to use a local vision
model. IRIS never installs runtimes or pulls a model automatically. The default
[Qwen3-VL 4B Instruct model](https://ollama.com/library/qwen3-vl:4b-instruct) is
approximately 3.3 GB, in addition to Ollama's runtime. Memory requirements depend
on the model and local hardware; CPU execution may be slow.

For a manually managed server, run this command from the project directory in
one terminal. It keeps models in the ignored workspace and disables Ollama Cloud:

```sh
OLLAMA_HOST=127.0.0.1:11434 OLLAMA_NO_CLOUD=1 \
  OLLAMA_MODELS="$PWD/.iris/ollama" ollama serve
```

If Ollama already runs as a service, configure and restart that service instead
of starting a second server. See [Ollama's configuration guide](https://docs.ollama.com/faq).
In another terminal, explicitly download the model:

```sh
OLLAMA_HOST=127.0.0.1:11434 ollama pull qwen3-vl:4b-instruct
```

Refresh availability in **Local multimodal review**. Choose the current saved
labels or a saved detector output containing 1–8 person/car boxes, then request a
review. Each request processes one selected frame and its candidate crops. The
frame is resized to at most 1024 pixels on its longest edge, crops to 320 pixels.
The model can propose a class correction, rejection or uncertainty; it supplies
no box coordinates. Missing objects and final geometry remain part of human
review. Its proposals never validate a frame automatically.

IRIS accepts only loopback HTTP endpoints and locally installed vision models.
It rejects cloud model names and remote aliases, ignores HTTP proxy settings,
and follows no redirects. Images are sent only when a review is explicitly
requested. Raw responses, prompts, candidate provenance, model digests and
generation settings remain in the local workspace. Invalid or incomplete output
fails the job without publishing proposals. Manual annotation and detector
proposals remain available when Ollama is absent.

The defaults can be changed when starting IRIS:

```sh
IRIS_OLLAMA_URL=http://127.0.0.1:11434 \
  IRIS_OLLAMA_MODEL=qwen3-vl:4b-instruct uv run iris
```

## Development and verification

```sh
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Tests generate small synthetic images and videos in temporary directories.
They exercise ingestion, provenance, extraction, selection, model availability,
comparison snapshots, raw outputs, annotation revisions, human validation,
multimodal response validation, job lifecycle, cancellation, migration, and
persistence. Detector and multimodal doubles are confined to tests and are never
exposed as models in the application. Tests establish software behavior, not
detection or annotation quality on real drone data. No datasets or model weights
are bundled.

After explicitly installing the runtime and both checkpoints, opt into the live
adapter smoke checks on generated images:

```sh
IRIS_TEST_MODEL_DIR=/absolute/path/to/iris-data uv run --extra ml pytest tests/test_models.py
```

Without this variable, those two live checks are skipped. They perform real
forwards through the models, but synthetic images provide no accuracy benchmark.

After separately provisioning and starting local Ollama, opt into a real
multimodal protocol check on a generated image:

```sh
IRIS_TEST_OLLAMA=1 uv run pytest tests/test_assistance_provider.py -k real_ollama
```

This check uses `IRIS_OLLAMA_URL` and `IRIS_OLLAMA_MODEL` when set. It never
downloads anything and fails if the configured local provider is unavailable.
Without `IRIS_TEST_OLLAMA=1`, it is skipped. It verifies the response structure
and provenance, not semantic accuracy.

The web UI and API are served by FastAPI; metadata lives in SQLite and media
in local files. The browser uses plain JavaScript without external assets.
The API schema is available at `/openapi.json`. The application works offline
after installation and does not load documentation assets from a CDN.

Model and dataset licenses must be checked independently from framework licenses;
catalog entries and download receipts retain the official source references.
