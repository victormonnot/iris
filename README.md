# IRIS

A local computer vision workbench for drone imagery. IRIS is being built around
the full improvement loop: flight data → frame selection → assisted annotation →
human validation → versioned dataset → fine-tuning → model comparison.

IRIS currently provides **data intake, COCO dataset import, detector comparison, assisted annotation,
human review, dataset versions and COCO export, local detector fine-tuning, held-out evaluation,
and explicit model reference selection**, with a review queue for tracking
annotation progress and inspecting detector disagreements. Trained checkpoints return to the visual
comparator and can be measured against their parents. See
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
   is idempotent. Import and extraction stay on this machine.
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
warnings; dataset freezing separately rejects identical pixels crossing splits.

Supported video containers are AVI, MP4/M4V, modern MOV, MKV, and WebM, subject
to OpenCV's bundled codecs. Playlists, MPEG containers, and older MOV files
without a file-type header are rejected. Browser playback depends on the
browser's own codec support; a video can be extractable without being playable
in the browser. Timestamps currently derive from frame index and reported FPS;
they are approximate for variable-frame-rate recordings. There is no telemetry
alignment or live capture.

## Import an annotated dataset

Use **Import annotated dataset** in the sidebar to upload a ZIP containing one COCO
bounding-box JSON and its images. Preview the images and boxes, explicitly map
each source category to `person`, `car`, or `exclude`, and record the source,
license and attribution. Assign one scene group to the package and preserve its
original train/validation/test split when known. Related scenes must stay in the
same group; do not manufacture independent splits from neighboring frames.

Import creates selected frames and reviewable proposals. **External labels are
not automatically human-validated**, including images with no imported boxes.
Open **Annotation** to accept, correct or reject the proposals and check missing
objects, then use the existing dataset/training/evaluation workflow. The archive,
original classes, coordinates and mapping remain available as provenance.
Declared source splits reserve both scene groups and exact image pixels, even
before review, and cannot be reassigned when freezing a dataset.

The initial importer handles small batches: at most 100 images and a 64 MiB ZIP.
Crowd and ignore annotations are rejected because their evaluation semantics are
not supported. Segmentations may be preserved as source metadata, but only
bounding boxes are imported. Previewing and importing use local files and never
fetch image URLs or transmit images to an annotation provider.
See [the COCO import format](docs/coco-import.md) for packaging, limits and a
public aerial-data example.

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
   outputs include all returned categories, before UI filtering. Trained person/car
   models retain their native IDs and an explicit mapping to COCO IDs 1 and 3.
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
runtime are not quality metrics. Use **Evaluation** with a frozen, validated
dataset to measure precision, recall and mAP separately from visual comparisons.

## Annotate and review

Open **Annotation** for a selected frame. The initial taxonomy, `iris-objects-v1`,
contains **person** and **car**; the interface shows their exact definitions.
Draw boxes, move or resize them, or edit their pixel coordinates. Saved detector
outputs can be imported as proposals without running inference again.

Use **Fit**, **1:1**, zoom controls and **Focus selected** to inspect small objects.
The hand tool or **Space + drag** moves the view; the mouse wheel zooms when the
canvas has focus. Labels and resize handles keep a readable size, and all saved
coordinates remain in original image pixels.

**Undo** and **Redo** cover up to 100 local edits, including proposal decisions,
box geometry, classes and review notes. They do not undo saved revisions or
validate labels. Saving, reloading a revision or switching frames starts a new
local history. See the [editor controls and shortcuts](docs/annotation-editor.md).

Accept or reject proposals individually, correct labels and geometry, and inspect
the whole image for missed objects. **Save draft** records work in progress.
**Validate frame** requires a reviewer and a decision for every pending proposal;
an empty validated frame explicitly records that no target objects are visible.
Every save creates a revision with provenance. Concurrent edits produce a reload
conflict instead of silently replacing another revision. Editing a validated frame
requires a fresh human validation.

### Local review batches

Expand **Local batch review** in Annotation to choose up to 25 selected frames
from the current session. Choose an installed local vision model and use either
saved labels or one detector's results from a completed comparison as candidates.
Preview the batch to see which images contain 1–8 eligible person/car boxes and
why other images will be skipped, then explicitly start the eligible reviews.

Images run sequentially through the existing local worker. The batch records
progress, errors and proposal counts for each image, survives reopening the app,
and can be cancelled while retaining proposals already saved. A failed image does
not stop the other queued images. Open each result in the review queue to accept,
correct or reject its proposals; a batch never validates labels.

This batch workflow uses **local Ollama models only**. It does not download models
or run a new detector, and it does not locate objects in an image without candidate
boxes. See [batch controls, sources and recovery](docs/annotation-batches.md).

### Local multimodal assistance

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

Refresh availability in **Multimodal review**, choose **Local**, and select an
installed vision model from the list. Choose the current saved
labels or a saved detector output containing 1–8 person/car boxes, then request a
review. Each request processes one selected frame and its candidate crops. The
frame is resized to at most 1024 pixels on its longest edge, crops to 320 pixels.
The model can propose a class correction, rejection or uncertainty; it supplies
no box coordinates. Missing objects and final geometry remain part of human
review. Its proposals never validate a frame automatically.

The Ollama adapter accepts only loopback HTTP endpoints and locally installed vision models.
It rejects cloud model names and remote aliases, ignores HTTP proxy settings,
and follows no redirects. Images are sent only when a review is explicitly
requested. Raw responses, prompts, candidate provenance, model digests and
generation settings remain in the local workspace. Invalid or incomplete output
fails the job without publishing proposals. Manual annotation and detector
proposals remain available when Ollama is absent.

The local defaults can be changed when starting IRIS:

```sh
IRIS_OLLAMA_URL=http://127.0.0.1:11434 \
  IRIS_OLLAMA_MODEL=qwen3-vl:4b-instruct uv run iris
```

### Hosted Qwen models through an API

Choose **API** in the annotation panel to use **Qwen3-VL 32B Instruct** or
**Qwen3-VL 235B-A22B Instruct** through Alibaba Cloud Model Studio. No local
model download is needed for these reviews. Manual annotation and saved local
results remain available without an API account.

Create a Model Studio workspace and API key separately. These presets use the
[Frankfurt workspace endpoint](https://www.alibabacloud.com/help/en/model-studio/regions).
Configure the IRIS server environment before starting it:

```sh
export IRIS_DASHSCOPE_BASE_URL="https://YOUR_WORKSPACE_ID.eu-central-1.maas.aliyuncs.com/compatible-mode/v1"
# Supply IRIS_DASHSCOPE_API_KEY through your shell or secret manager.
# Keep the key out of source files, Git, browser storage and screenshots.
.venv/bin/iris
```

IRIS reads these environment variables; it does not automatically load `.env`
files. **Configured** means the settings are present, not that the credentials
have been tested. Catalog refresh and preview generation never contact Alibaba.
The access region is Frankfurt, but these model presets use **Global deployment
scope**: inference is not guaranteed to remain in the EU.

1. Select the API provider/model and saved candidate source.
2. Generate a preview. Inspect the exact resized scene and crops to be sent,
   the endpoint, review focus and conservative cost ceiling in USD.
3. Explicitly authorize this request. Changing the selection or configuration
   invalidates the preview; previews expire after 30 minutes and can be used once.
4. Inspect the returned proposals, correct them, and validate the frame yourself.

The cost ceiling uses the documented maximum input tokens and a 1,024-token
output limit at recorded list prices, **not** a predicted token count. It excludes
taxes and later provider price changes. Model IDs and dated price sources are
saved with the request; reported usage is retained when available. A provider
bill is authoritative. Neither previewing nor saving annotations incurs API cost.

Hosted requests use JSON Object mode and strict local validation. A malformed,
incomplete or wrong-model response fails without creating labels. IRIS makes
one attempt, with no automatic retry or switch to another provider. Cancellation
stops the local worker; the provider may still finish and bill a request it has
already received. Retrying requires a new preview and approval.

Exact outgoing image hashes, the approved ceiling, provider/model, prompt,
response and usage remain traceable. The hosted model ID does not provide an
immutable weight digest; local Ollama requests retain their actual model digest.
API keys are read only by the server/worker and are never returned to the UI.

## Freeze a dataset and fine-tune

**Dataset & training** works across all flight sessions. It requires no API key.

1. Select frames and validate their annotations, including empty negative images.
   Drafts, unselected frames, and images with unresolved proposals are excluded.
2. Refresh dataset candidates and assign scene groups to **Train**, **Validation**,
   or **Test**. At least two distinct groups are required for train and validation;
   test is optional, and its absence is reported. Related flights belong in the
   same group. Never distribute neighboring frames randomly across splits.
3. Name and freeze the version, optionally linking a previous release as its parent.
   IRIS copies normalized PNGs and records image hashes, full annotation revisions,
   reviewers, source footage, timestamps, taxonomy, and splits in a checksummed
   manifest. Download the manifest or inspect it in the interface. Later label
   edits and selection changes leave that version intact.
4. Choose this dataset and a ready **Faster R-CNN MobileNetV3-Large 320 FPN** parent,
   either the official checkpoint or a previous IRIS checkpoint. Start a bounded
   CPU run: 20 optimizer steps by default, configurable from 1 to 200, batch size
   one, with a recorded learning rate and seed. Runtime depends on the CPU and
   images; these limits bound steps, not wall-clock time. No weights are downloaded.
5. Follow progress, individual loss components and logs. A completed run registers
   its checkpoint and SHA-256, parent, dataset version and settings. Use it in
   **Model comparison** alongside its parent on the same held-out frames.

Scene-group assignments and exact pixel hashes retain their split across all
versions in a workspace. Crossing these reservations is rejected, including when
an image is imported again under another session name. Duplicate pixels within
one release are also rejected. Perceptual-hash warnings help review similar
images; they do not establish independence. There is currently no operation to
retire or reassign a reserved test group. A new version contains a full snapshot;
linking a parent does not automatically add its images. A version holds at most
1,000 images and makes its own image copies, so allow additional disk space.

The first training engine adapts only the final classification and box regression
layers. It initializes background/person/car from the parent, freezes feature
extraction and proposal layers, and uses SGD on the frozen **train** split only.
Validation and test images are not opened by the training worker. Negative
training images are supported, but at least one positive annotation is required.
With fewer steps than training images, only part of the training set is visited.
This small scope makes a CPU experiment practical; it may be insufficient for
small distant objects or substantial domain changes.

Loss measures optimization on training examples, **not detection quality**.
Use the separate evaluation workflow on independent, reviewed imagery to measure
gains and regressions. Reference selection is explicit. Old checkpoints remain
available. Cancellation/failure preserves saved step history and logs, but publishes
no incomplete checkpoint. Restart marks unfinished jobs interrupted. Launch a new
run to retry; exact optimizer-state resume is not implemented.

### Export a reviewed release

In **Dataset releases**, choose a version and click **Download COCO ZIP**. The
local archive contains its frozen PNGs, COCO bounding-box annotations for each
train/validation/test split, the original manifest and an export inventory with
checksums. Validated negative images are included. Image integrity is checked
before the download; later edits in the annotation editor do not alter the release.

COCO category IDs are **1 = person, 3 = car**. They differ from the native training
mapping, where car is 2. Each split has its own `annotations.json` and `images/`
directory. Recorded source attribution, reviewer names and notes are retained in
the manifest. The download stays local and includes no model weights or original
videos. Archives are limited to 256 MiB; see [the export format and limits](docs/dataset-export.md).
This is a training interchange package, not a workspace backup or a ZIP that the
current single-JSON COCO importer can directly restore.

## Work through the review queue

Open **Annotation** to see the selected frames in the current session and how
many still need review. Filter unannotated images, drafts, pending proposals or
validated frames. New proposals make a previously validated image need review
again. An explicitly validated empty image counts as reviewed.

Optionally choose a completed comparison of two models and **Model disagreement
first**. The queue compares saved person/car boxes by class and overlap at the
displayed confidence and IoU thresholds. Each image explains its unmatched
detections or class conflicts. Equal detection counts can still disagree about
positions. Missing predictions and two empty outputs have distinct explanations;
neither establishes that an image contains no objects.

Open a frame to correct its labels. **Validate and next** saves a human-validated
revision, then opens the next image needing review. It requires the same reviewer
name and resolved proposals as **Validate frame**. Failed saves do not advance,
and filtering or refreshing preserves the frame being edited. Sorting changes
neither selected frames nor dataset splits. Test-reserved images remain in source
order and are marked for review only.

The queue uses existing local results: it starts no inference, training or API
request. Disagreement is an inspection aid, not an error rate or measured quality.
Both models can miss the same object. See [review queue behavior and matching](docs/review-queue.md).

## Evaluate and select a reference

Open **Evaluation**, choose a dataset release and one or two ready models, and
run them on the release's complete **validation** split. The models use the same
frozen images and reviewed labels. Choose the confidence and IoU thresholds for
precision/recall before launching; the default is 0.5 for both. Runs record the
dataset manifest hash, checkpoint hashes, training ancestry, class mapping,
metric implementation, thresholds, device and timing protocol.

Results include COCO bbox mAP at IoU 0.50:0.95, AP50, AP75, per-class AP, and
precision/recall with true/false positives and missed objects at the chosen
thresholds. AP uses all saved native scores, independently of the precision/recall
confidence threshold. Missing reference classes have **N/A** AP and are excluded
from macro averages; undefined precision/recall also display **N/A**. These are
not perfect scores. Inspect per-image errors, reviewed boxes and predictions,
plus inference and total processing times. Saved partial predictions remain
inspectable after cancellation or failure; an incomplete comparison cannot
support reference selection.

After a complete validation run, select a model as the workspace reference with
your reviewer name and reason. The choice and its evidence are appended to a
history; previous references and checkpoints remain available. Nothing selects
the latest training or the highest score automatically. A new training run does
not replace the reference.

If the dataset has a **test** split, launch a final audit from the completed
validation run. The audit reuses the same models, checkpoint hashes, thresholds,
device and metric protocol. Test results cannot directly promote a reference.
Repeatedly inspecting test results and then changing models can still bias
human decisions; preserve the test for final reporting. IRIS rejects local
training overlap by both scene group and exact image pixels, including ancestor
checkpoints. This does not establish independence from COCO pretraining data or
unrecognized related scenes.

See [the evaluation protocol](docs/evaluation.md) for metric definitions,
filtering limits, handling of absent classes, and verification scope.

### Find errors and regressions

Open a completed run in **Evaluation** to explore its saved errors. Filter by
person or car, find missed objects and false positives, and open a row to inspect
the same frozen image with both models' overlays. With two models, the first is
the baseline and the second is the candidate: see objects the candidate recovers,
objects it newly misses, and changes in false-positive counts.

This comparison follows individual reference objects, so an unchanged miss total
can still reveal both recoveries and regressions. Filters change the examples
shown, not the saved metrics or thresholds. No inference is rerun. Incomplete or
inconsistent results are shown as unavailable, and test audits remain for
reporting rather than model selection.

## Development and verification

```sh
uv run pytest
uv run ruff check .
uv run ruff format --check .
node --test tests/js/annotation-tools.test.cjs
```

The JavaScript tests use Node's built-in test runner; Node is only needed for
development, not for running IRIS. They check viewport geometry, zoom anchoring,
pixel scale, image bounds and local history branching.

Tests generate small synthetic images and videos in temporary directories.
They exercise ingestion, provenance, extraction, selection, model availability,
comparison snapshots, raw outputs, annotation revisions, human validation,
multimodal response validation, exact outgoing previews, explicit API consent,
local batch eligibility, atomic queueing, cancellation and interrupted history,
budget checks, immutable dataset snapshots, split leakage, checkpoint provenance,
COCO archive validation, imported-label review and source split preservation,
frozen COCO exports, negative images, checksums and interrupted-download cleanup,
review progress, saved-prediction disagreement and read-only queue persistence,
COCO metrics, error matching, fixed test audits, reference history,
saved error analysis, class filters and paired recovered/newly missed objects,
job lifecycle, cancellation, migration, and
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

To explicitly run the real training/worker check using already installed weights:

```sh
IRIS_TEST_TRAINING=1 IRIS_TEST_MODEL_DIR=/absolute/path/to/iris-data \
  uv run --extra ml pytest tests/test_training_live.py
```

It performs three CPU optimizer steps across two generations, verifies changed
prediction-head weights and unchanged frozen layers, reloads the checkpoint into
a comparison with its parent, evaluates both on frozen validation and test
splits, records an explicit reference, and checks persistence after restart. All images
and review records are generated fixtures, not human-validated flight data. It
downloads nothing. Without `IRIS_TEST_TRAINING=1`, this test is skipped.

After separately provisioning and starting local Ollama, opt into a real
multimodal protocol check on a generated image:

```sh
IRIS_TEST_OLLAMA=1 uv run pytest tests/test_assistance_provider.py -k real_ollama
```

This check uses `IRIS_OLLAMA_URL` and `IRIS_OLLAMA_MODEL` when set. It never
downloads anything and fails if the configured local provider is unavailable.
Without `IRIS_TEST_OLLAMA=1`, it is skipped. It verifies the response structure
and provenance, not semantic accuracy. Qwen3-VL 4B has also completed the real
local worker/UI workflow on a synthetic fixture using an RTX 4060. Hosted API
behavior is covered by offline transport and workflow fixtures; no paid request
or real drone annotation quality is claimed by those tests.

The web UI and API are served by FastAPI; metadata lives in SQLite and media
in local files. The browser uses plain JavaScript without external assets.
The API schema is available at `/openapi.json`. The application works offline
after installation and does not load documentation assets from a CDN.

Model and dataset licenses must be checked independently from framework licenses;
catalog entries and download receipts retain the official source references.
