# IRIS

A local computer vision workbench for improving object detectors from your own
images and videos. IRIS follows the full improvement loop: sources → frame selection → assisted annotation →
human validation → versioned dataset → fine-tuning → model comparison.

IRIS currently provides **projects, data intake, COCO dataset import, detector comparison, assisted annotation,
human review, dataset versions and COCO export, local detector fine-tuning, held-out evaluation,
explicit model reference selection, saved experiment reports, independent annotation benchmarks,
and workspace backup/restoration**, with a review queue for tracking
annotation progress and inspecting detector disagreements. Trained checkpoints return to the visual
comparator and can be measured against their parents. See
[the architecture](docs/architecture.md).

## Annotation benchmark

Open **Benchmark** to freeze an independent human reference and compare saved
preannotation configurations. Assign whole scenes to **tuning** or **evaluation**,
record the reference reviewer and independence declaration, then freeze candidate
settings. Lock the configuration set before evaluating the held-out images.

Choose an installed local detector as the control, **A · Multimodal · OpenAI**
using `gpt-6-astra`, or **B · SAM 3** for local text-prompted native boxes.
The combined approach remains a future integration.
Successful complete trials report proposal precision,
recall, false positives, misses, class conflicts and matched-box IoU at the
recorded operating point. Failures and missing outputs remain explicit.

Review candidate boxes in a separate editor with start/pause timing and an
append-only correction history. Neither these corrections nor the candidate
outputs modify the reference or ordinary annotations. Recorded review intervals
are separate from model latency, and missing time is never reported as zero.
See [the benchmark protocol and limitations](docs/benchmark.md).

The SAM path uses one short phrase per frozen class and an independent score
threshold. Configurations can be saved before setup; execution requires the
pinned checkpoint and a separate CUDA environment configured through
`IRIS_SAM_PYTHON`. Masks are disabled and no installation or download happens
automatically. Real SAM execution and hardware measurements remain deferred.
See [the SAM adapter and setup requirements](docs/sam-preannotation-adapter.md).

The OpenAI path is optional and sends images outside this computer. Configure
`IRIS_OPENAI_API_KEY` (or `OPENAI_API_KEY`) in the server environment, then restart
IRIS. Key presence does not verify account access. Every trial shows the exact
outgoing images and prompt, a planning estimate and a fresh consent/budget control.
The budget limits admitted requests; it is not a guaranteed provider invoice cap.
Failures stop the remaining requests, and ambiguous outcomes are never retried
automatically. API token usage and unknown charges remain visible in the receipts.
No key is needed to prepare configurations and inspect local previews.
See [the adapter contract and pricing sources](docs/openai-preannotation-adapter.md).

## Run locally

Requires Linux or WSL, Python 3.12 or 3.13, and [uv](https://docs.astral.sh/uv/).
No GPU, model weights, API key, or external account is needed
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
and extracted PNGs. Use **Workspace backup** to create and restore verified archives.

## Projects

Use the **Project** selector in the sidebar to switch projects, or **New project**
to create one with a name and optional description. Each project contains its own
sessions, dataset versions, trained models, evaluations, reference selection and
reports. Existing work appears in **Default project** after upgrading. Official
pretrained checkpoints are shared; workspace backup includes every project.

Use **Manage classes** to publish custom class IDs, names and definitions for
manual annotation and COCO imports. Each image retains its saved class version;
adopting newer definitions creates a draft that requires human review. The initial
`iris-objects-v1` Person / Car definitions remain available. Custom datasets can
be frozen, trained, evaluated and exported as COCO. Multimodal candidate review still requires the original definitions. Direct detector
preannotation and review signals support compatible custom definitions. See [class versions](docs/classes.md) for the supported workflow.
IRIS has no dependency on ARGOS, flight telemetry or a drone-specific file format;
recordings from any supported source can use the same local workflow.
See [projects and compatibility](docs/projects.md).

## First workflow

1. Choose a project, create a session, and give related sessions the same **scene group**.
   These groups will be used to separate training, validation, and test data.
2. Import images or videos. Each upload is limited to 2 GiB. Sources are kept
   with a SHA-256 checksum; importing an identical file into the same session
   is idempotent. Import and extraction stay on this machine.
3. For a video, choose a time range and image budget (up to 500 positions).
   **Across the whole range**, the default in the interface, distributes those
   positions across the entire range. Check the timeline and use **Preview images**
   to inspect up to 12 timestamped thumbnails before starting extraction.
   **Fixed interval** retains the original sampling behavior; its preview warns
   when the limit covers only the beginning of the range. Extraction runs in a
   separate process.
4. Browse frames, inspect their source and timestamp, and select useful images.
   Selection is persisted; it does **not** mean the image has been annotated
   or human-validated.
5. Inspect job progress, errors, and logs. Cancel extraction if needed; frames
   already produced remain available. After a server interruption, unfinished
   jobs are marked interrupted. New extractions can preview and continue their
   remaining frozen sampling positions. Older jobs require a new extraction.

The project job history provides filters, saved partial results and recovery
details. Continuation creates a linked extraction attempt; other model work needs
an explicit new run. Uncertain external requests are never resent automatically.
See [jobs and recovery](docs/job-recovery.md), including unfinished local review batches.

The import queue accepts multiple images and videos, reports each result and can
retry failed files. Gallery filters help inspect human-reviewed positives and
negatives, duplicate candidates and signals from saved compatible predictions.
Selection remains a manual decision. See [import and useful selection](docs/intake-selection.md)
for source navigation, batch selection and the partition assistant.

The budget limits sampled positions, including existing frames and duplicates;
it is not a guarantee of that many new images. Uniform sampling includes the
first and last eligible frame when the budget is at least two; a budget of one
chooses the middle frame. The end time is exclusive. If the range contains fewer
frames than the budget, every eligible position is considered. Fixed-interval
sampling preserves the legacy conversion from seconds to the preceding frame.
API requests without `sampling_mode` keep that legacy behavior.

Previews run locally, create no frames or jobs, and never change the selection.
The timeline uses metadata; thumbnail preview decodes at most 12 planned
positions after checking the source checksum. A thumbnail preview shows only a
subset when more positions are planned. The worker checks the same source
checksum before extraction and records the sampling method and resolved plan.
This sampling preview distributes positions in time; it does not judge which
events are interesting. For model-assisted selection, use **Suggest passages**
on a video: prepare a timestamped storyboard, review the exact images, and
explicitly start one local or API model request. Select the proposed passages
you want to extract, optionally keeping extra samples across the full range.
See [video passage review](docs/video-review.md) for the workflow and limits.

Exact duplicate frames are skipped within each video. Optional perceptual
deduplication is disabled by default and is only a heuristic: it can conflate
different content, including flat colors. Neither filter fills the vacated
positions, so retained images may no longer cover the whole range. The job result
shows sampled positions, new images, existing positions, exact duplicates and
similar-image skips. Inspect the retained frames before selecting them.
Identical images across sessions remain visible as duplicate warnings;
dataset freezing separately rejects identical pixels crossing splits.

Supported video containers are AVI, MP4/M4V, modern MOV, MKV, and WebM, subject
to OpenCV's bundled codecs. Playlists, MPEG containers, and older MOV files
without a file-type header are rejected. Browser playback depends on the
browser's own codec support; a video can be extractable without being playable
in the browser. Timestamps currently derive from frame index and reported FPS;
they are approximate for variable-frame-rate recordings. Some containers also
report an estimated frame count: a planned position near the end may not be
decodable. In that case preview or extraction reports an error instead of
silently substituting another frame. Try a shorter range or import a copy
converted to a constant frame rate; any frames already extracted remain
available. IRIS does not scan or transcode the entire video to repair timing
metadata. There is no telemetry alignment or live capture.

## Import an annotated dataset

Use **Import annotated dataset** in the sidebar to upload a ZIP containing one COCO
bounding-box JSON and its images. Preview the images and boxes, explicitly map
each source category to one of the preview's saved target classes or `exclude`, and record the source,
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
2. Choose one or both ready models and an inference mode, inspect the estimated
   detector passes, name the comparison, and start it. CPU is the
   default. CUDA is available only with a separately provisioned compatible
   runtime; the optional `ml` installation intentionally contains CPU wheels.
3. Inspect the same frame side by side. Changing the displayed confidence or
   class filters saved predictions; it does not rerun the models. Raw saved
   outputs include all returned categories, before UI filtering. Trained checkpoints
   retain their frozen class definitions and numeric mappings. Original Person / Car
   checkpoints retain output IDs 1 and 3; custom checkpoints use their own saved IDs.
4. Reopen the comparison from its history after a restart. It retains its frame
   selection, frame hashes, checkpoint hashes, model configuration, runtime,
   device, and timing protocol. Subsequent selection changes do not change it.

Inference executes in the local job worker. Each completed run/frame result
is saved independently, including an explicit empty detection list. A cancelled
or failed job keeps its partial outputs; **Not processed** is distinct from
**No detections**. Start a new comparison to retry. Saved results remain readable
when weights or the optional runtime are unavailable.

Timings use one excluded warmup per run, batch size one, float32, and at most
four CPU threads. Decode, tensor preparation, full model forward, and result
serialization are recorded separately. **Model forward includes Torchvision's
internal normalization, resizing, proposal filtering, NMS, and coordinate
restoration**; it is not a backbone-only measurement. Total time includes decode
and hash verification but excludes loading weights, warmup, and database writes.
CUDA runs synchronize at the measurement boundaries. Inspect the saved metadata
for the exact resolution, native thresholds, and hardware.

Both models use a native score cutoff of 0.001, NMS IoU 0.5, and at most 100
detections per detector call. These settings define what can be saved; lowering the UI
threshold cannot recover outputs below that native cutoff. Models still have
different internal proposal algorithms. Confidence, count differences, and
runtime are not quality metrics. Use **Evaluation** with a frozen, validated
dataset to measure precision, recall and mAP separately from visual comparisons.

### Replay a video comparison

Open a saved comparison containing video frames to review the original footage
alongside its saved model outputs. Choose a source video, use the timeline or the
sample selector to jump to an analysed image, and play the surrounding passage.
Playback follows the saved samples; the result cards always show the extracted
image and its own timestamp. Class and confidence filters still apply to saved
detections without running a model again.

Timeline markers distinguish samples processed by all runs, partial results and
samples with no saved prediction. The spaces between markers have no inference
results. A held image is explicitly labelled; its boxes are never drawn onto a
different video frame. An empty detection list is a processed result, whereas a
missing prediction remains **Not processed**. Full-image and tiled runs retain
separate results even when they use the same checkpoint.

Video positions are approximate: extraction records frame index divided by
nominal frame rate, so variable-frame-rate footage may not align exactly with
the browser's video clock. The extracted image is the reference for inspecting
boxes. Missing footage or an unsupported browser codec leaves saved images and
predictions available for review. No transcoding, model download or inference is
started by replay. Comparisons containing only still images keep their existing
image navigation.

### Compare full images with tiles

**Full image** runs each chosen checkpoint on the entire frame. **Tiled image**
runs it on overlapping regions, restores the boxes to the original frame and
suppresses duplicates. **Full image vs tiled** compares those two pipelines
side by side with one checkpoint. The source image and checkpoint are identical;
no camera zoom or retraining is involved. Smaller regions can preserve more of
a small object's pixels through the detector's internal resize. They cannot
recover details absent from the recording, and may lose context or cut objects
at region boundaries.

Tiles default to 640 pixels with 20% overlap. Sizes from 128 to 2048 and overlap
from 0 to 50% are supported. Edge regions are anchored to the image boundary, so
their overlap can be larger than requested. Small images produce one region,
without padding or enlargement. The preview counts all detector calls, including
one warmup for each run, before anything is queued. Each image is limited to
64 tiles and each comparison to 512 calls including warmups.

Tiled outputs use an additional, class-aware NMS at IoU greater than 0.5, ordered
by descending score with stable ties. At most 300 merged boxes are retained;
the interface reports truncation. All original per-tile outputs, crop coordinates
and timings remain in each saved prediction's `metadata.tiles`, accessible in
the comparison API. Run provenance includes the complete tile plan and merge
settings. Cropping and merging are timed separately; forward time is the sum
over all tiles. Total excludes warmup, weight loading, progress reporting and
database writes. These sequential local measurements are not an exported-model FPS benchmark.

Cancellation is checked between tiles and during merging. Only complete image
results are published; earlier completed results survive. In **Annotation** and
**Local review batch**, select the required full-image or tiled source explicitly.
The disagreement queue also accepts the two variants of the same checkpoint.
Proposals remain in original-image coordinates and require human review.

Use the same inference modes in **Quality evaluation** to score a frozen,
reviewed validation split. **Full image vs tiled** compares one checkpoint's
quality and processing time in the same evaluation. More boxes in the visual
comparator alone do not establish a quality gain.

## Annotate and review

Open **Annotation** for a selected frame. The interface shows the image's saved
class names and exact definitions, including custom classes created in **Manage classes**.
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

### Generate new proposals

In **Annotation → Generate proposals**, choose selected images and an installed
local detector. Preview the class coverage and work, then explicitly start the
run. No existing boxes or prior comparison are required. Official detectors use
explicit COCO mappings; a trained detector requires the image's exact saved class
version. Uncovered classes remain a manual task.

Completed image outputs retain their raw predictions and reviewable proposals. Inspect low-score
or uncertain proposals, correct classes and geometry, add missed objects and save
your review. An empty detector output never becomes a validated negative image.
No model downloads or external calls occur in this workflow. See
[preannotation and provider contracts](docs/preannotation.md).

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

**Dataset & training** works across all sessions in the selected project. It requires no API key.

1. Select frames and validate their annotations, including empty negative images.
   Drafts, unselected frames, and images with unresolved proposals are excluded.
2. Choose a class version, refresh dataset candidates and assign scene groups to **Train**, **Validation**,
   or **Test**. At least two distinct groups are required for train and validation;
   test is optional, and its absence is reported. Related scenes belong in the
   same group. Never distribute neighboring frames randomly across splits.
   **Suggest whole-group partitions** previews a repeatable allocation by target
   proportions, with coverage warnings and existing reservations. Review and apply
   the proposal before freezing. Copies of the same original video remain in one
   split, including across sessions and releases; reencoded footage still needs
   deliberate scene grouping.
   Every included image must use that exact class version. Other versions remain
   available through the selector; publishing classes never relabels a dataset.
3. Name and freeze the version, optionally linking a previous release with the same
   class version as its parent. A different class version starts an independent release.
   IRIS copies normalized PNGs and records image hashes, full annotation revisions,
   reviewers, source footage, timestamps, complete class definitions, numeric mappings,
   and splits in a checksummed
   manifest. Download the manifest or inspect it in the interface. Later label
   edits, class changes and selection changes leave that version intact. If reviews
   changed since the candidate list was loaded, refresh before freezing again.
4. Choose the frozen dataset and a ready
   **Faster R-CNN MobileNetV3-Large 320 FPN** parent,
   either the official checkpoint or an IRIS checkpoint with the exact same class version. Choose a training
   depth and preview the plan before starting a CPU run: 20 optimizer steps by
   default, configurable from 1 to 200, batch size one, with an explicit learning
   rate and seed. The preview shows the train image count, visits and complete
   passes. Editing a setting requires a new preview. Runtime depends on the CPU,
   images and depth; these limits bound steps, not wall-clock time. No weights
   are downloaded. Custom classes use the same workflow; see
   [custom training and compatibility](docs/custom-training.md).
5. Follow progress, individual loss components and logs. A completed run registers
   its checkpoint and SHA-256, parent, dataset version and settings. Use it in
   **Model comparison** alongside its parent on the same held-out frames.

Scene-group assignments retain their split across versions within a project.
Exact pixel hashes retain their split across the whole workspace, including other
projects. Crossing these reservations is rejected, including when
an image is imported again under another session name. Duplicate pixels within
one release are also rejected. Perceptual-hash warnings help review similar
images; they do not establish independence. There is currently no operation to
retire or reassign a reserved test group. A new version contains a full snapshot;
linking a parent does not automatically add its images. A version holds at most
1,000 images and makes its own image copies, so allow additional disk space.

Each run chooses its own depth; a trained checkpoint can be continued at another
depth without changing its parent:

| Depth | Parameters updated | Intended experiment |
| --- | --- | --- |
| Light (default) | Final classification and box regression layers | A small first adaptation with the feature extractor fixed. |
| Partial | Last MobileNet feature stage, feature pyramid, proposal network and detection heads | Adapt later visual features, for example when trying footage from a different camera. Earlier feature stages stay fixed. |
| Full | All learnable detector parameters | Also adapt early visual features; requires more computation and can overfit a small dataset. |

The head contains one slot per frozen class plus background. Official weights
initialize background and any explicitly mapped COCO classes; other classes use
the recorded seed for initialization. A compatible IRIS parent's head is retained. All depths keep the pretrained frozen
batch-normalization statistics fixed and use SGD on the frozen **train** split only.
The selected depth never silently changes the learning rate. Run metadata records
the exact trainable modules and parameter counts, plus which modules changed.
Validation and test images are not opened by the training worker. Negative
training images are supported, but at least one positive annotation is required.
With fewer steps than training images, only part of the training set is visited.
Deeper training is an option to measure, not a guarantee of better results on
analog imagery or small distant objects. It does not change inference resolution.

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

New releases record category IDs in their frozen `coco_mapping`; legacy manifests
use the export's category table. Original Person / Car releases retain
**1 = person, 3 = car**, while custom releases assign **1…N** in
saved class order. The optional official-detector COCO ID on a class is separate
from its export ID. Each split has its own `annotations.json` and `images/`
directory. Recorded source attribution, reviewer names and notes are retained in
the manifest. The download stays local and includes no model weights or original
videos. Archives are limited to 256 MiB; see [the export format and limits](docs/dataset-export.md).
This is a training interchange package, not a workspace backup or a ZIP that the
current single-JSON COCO importer can directly restore.

## Work through the review queue

Open **Annotation** to see the selected frames in the current session and how
many still need review. Filter unannotated images, drafts, pending proposals or
validated frames, or focus on uncertain proposals and possible omissions. These
filters are inspection hints, not accuracy measurements. New proposals make a previously validated image need review
again. An explicitly validated empty image counts as reviewed.

Optionally choose a completed comparison of two models or two inference variants
of the same model and **Model disagreement
first**. The queue compares saved boxes mapped to the image's saved class definitions by class and overlap at the
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

Open **Evaluation**, choose a dataset release, one or two ready models and an
inference mode, then inspect the estimated work before launching on the complete
**validation** split. The models use the same
frozen images and reviewed labels. Choose the confidence and IoU thresholds for
precision/recall before launching; the default is 0.5 for both. Runs record the
dataset manifest hash, checkpoint hashes, training ancestry, class mapping,
metric implementation, thresholds, device and timing protocol.

Choose **Full image vs tiled** with one checkpoint to measure the effect of
tiling on the same held-out data. The results keep distinct full-image and tiled
runs, including separate metric columns, per-frame errors, processing times and
provenance. **Tiled image** can also compare two checkpoints using the same tile
settings. Tiles use the comparator's implementation and limits of 64 per image;
an evaluation permits at most 4,096 detector passes including warmups. The entire
split is evaluated or the request is rejected; large splits are never silently sampled.

Results include COCO bbox mAP at IoU 0.50:0.95, AP50, AP75, per-class AP, and
precision/recall with true/false positives and missed objects at the chosen
thresholds. AP uses all saved native scores, independently of the precision/recall
confidence threshold. Missing reference classes have **N/A** AP and are excluded
from macro averages; undefined precision/recall also display **N/A**. These are
not perfect scores. Inspect per-image errors, reviewed boxes and predictions,
plus inference and total processing times. Saved partial predictions remain
inspectable after cancellation or failure; an incomplete comparison cannot
support reference selection.

After a complete validation run, select a model and inference mode as the project reference with
your reviewer name and reason. The choice and its evidence are appended to a
history; previous references and checkpoints remain available. Nothing selects
the latest training or the highest score automatically. A new training run does
not replace the reference. A tiled reference records its tile size, overlap and
merge settings; selecting the same weights in full-image mode is a different choice.

If the dataset has a **test** split, launch a final audit from the completed
validation run. The audit reuses the same models, checkpoint hashes, thresholds,
inference modes, tile/merge settings, device and metric protocol. Its pass count
is previewed separately because the test split can contain different images.
Test results cannot directly promote a reference.
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

## Keep an experiment report

Open **Experiments**, or start a report from a completed **Quality evaluation**.
Choose a saved evaluation, name the experiment, record its objective and your
conclusion, and optionally select up to six example images. Creating a report
does not run a model, train a checkpoint, recalculate metrics or change the reference.

The report brings together the frozen dataset, checkpoint identities, available
training settings, evaluation scores, per-class results and saved error changes.
Examples illustrate the results; scores still cover the entire evaluated split.
One-model reports do not imply a before/after comparison. Full-image and tiled
runs remain separate even when they use the same checkpoint.

Results and selected examples are fixed when the report is created. You can
revise the title, objective and conclusion; simultaneous edits are checked to
avoid overwriting a newer revision. Selected images are saved as bounded JPEG
copies, so an existing report remains readable independently of source images
and model weights.

Download a standalone **HTML report** to read offline or print. Images are
excluded by default; explicitly include the selected examples when needed.
The document embeds its styles and any included images, with no scripts,
external requests or required IRIS server. Text-only exports still contain
your written notes and the recorded experiment context. Exporting does not
publish or upload the report. See [the report format and limits](docs/experiments.md).

## Back up or restore a workspace

Open **Workspace backup** under **Storage** in the sidebar. This includes all
projects, regardless of the selected project. Review the included
data, installed model weights and disk space, then create a local ZIP archive.
Save pending edits first and wait for queued or running jobs to finish. IRIS
temporarily blocks changes while saving; reading saved results remains available.
Close and reopen the dialog to follow the operation or download its archive.
Downloads use the browser's normal file download mechanism.

The archive preserves original media, annotations, frozen datasets, checkpoints,
predictions, experiment reports and histories. It also includes detector and
Ollama model files stored inside the workspace. Temporary files, previous backups
and software environments are excluded. Environment-based API credentials must
be configured separately on another machine.

In **Restore**, choose an IRIS workspace ZIP and wait for its integrity check.
Review the contents, choose a new folder name and confirm restoration. The new
workspace is created beside the current one; an existing folder is never replaced.
IRIS shows its location and a command to open it separately. Restoring neither
switches the current app nor resumes training or other model jobs.

Command-line recovery also works with the web app stopped:

```sh
uv run iris workspace backup /path/to/iris-backup.zip --data-dir /path/to/workspace
uv run iris workspace inspect /path/to/iris-backup.zip
uv run iris workspace restore /path/to/iris-backup.zip --to /path/to/new-workspace
uv run iris --data-dir /path/to/new-workspace --port 8011
```

Stop the source workspace's server before using the backup CLI, or use the UI
while it is open. Archives contain private media, prompts and reviewer notes;
they are different from a shareable report or a COCO dataset export.
See [workspace backup and recovery](docs/workspace-backup.md) for compatibility,
storage limits and verification details.

## Development and verification

```sh
uv run pytest
uv run ruff check .
uv run ruff format --check .
node --test tests/js/*.test.cjs
```

The JavaScript tests use Node's built-in test runner; Node is only needed for
development, not for running IRIS. They check viewport geometry, zoom anchoring,
pixel scale, image bounds and local history branching.

Tests generate small synthetic images and videos in temporary directories.
They exercise ingestion, provenance, extraction, selection, model availability,
comparison snapshots, raw outputs, annotation revisions, human validation,
tiled coverage, coordinate restoration, merging, work limits and variant selection,
multimodal response validation, exact outgoing previews, explicit API consent,
SAM native-box validation, isolated runtime transport and offline recovery,
local batch eligibility, atomic queueing, cancellation and interrupted history,
budget checks, immutable dataset snapshots, split leakage, checkpoint provenance,
training-depth contracts, read-only workload previews and frozen-layer preservation,
COCO archive validation, imported-label review and source split preservation,
frozen COCO exports, negative images, checksums and interrupted-download cleanup,
review progress, saved-prediction disagreement and read-only queue persistence,
COCO metrics, error matching, fixed test audits, reference history,
saved error analysis, class filters and paired recovered/newly missed objects,
experiment snapshots, note revisions, saved images and standalone HTML reports,
workspace archive integrity, restoration, interrupted transfers and write admission,
job lifecycle, cancellation, migration, and
persistence. Detector and multimodal doubles are confined to tests and are never
exposed as models in the application. Tests establish software behavior, not
detection or annotation quality on real-world data. No datasets or model weights
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

The scope checks perform a few CPU optimizer steps across all three depths and
a continuation at a different depth. They inspect gradients, changed parameters,
unchanged frozen layers and normalization buffers, then reload the checkpoints.
The worker check performs three more steps across two generations, compares the
checkpoint with its parent, evaluates both on frozen validation and test splits,
records an explicit reference, and checks persistence after restart. All images
and review records are generated fixtures, not human-validated field data. It
downloads nothing. Without `IRIS_TEST_TRAINING=1`, these tests are skipped.

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
or real-world annotation quality is claimed by those tests.

The web UI and API are served by FastAPI; metadata lives in SQLite and media
in local files. The browser uses plain JavaScript without external assets.
The API schema is available at `/openapi.json`. The application works offline
after installation and does not load documentation assets from a CDN.

Model and dataset licenses must be checked independently from framework licenses;
catalog entries and download receipts retain the official source references.
