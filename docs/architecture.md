# Architecture

Workspace backup uses a streamed ZIP64 archive and a consistent SQLite snapshot.
An admission gate rejects new HTTP mutations while an idle workspace is copied;
database and file signatures also detect external changes. A background transfer
manager keeps operation receipts outside the archived inventory. Restoration
checks the manifest, all hashes, schema and file references, then atomically
publishes a new directory without replacing an existing destination. It preserves
the active workspace and starts no model jobs. The same archive services support
the UI and recovery CLI; see [backup and recovery](workspace-backup.md).

IRIS is a local workbench for improving object detectors from images and videos.
Projects organize import, model comparison, assisted annotation, human review,
dataset versions, fine-tuning, and comparison against earlier checkpoints. The
current executable model task is bounding-box detection. Temporal sequences,
reference identities, dataset versions and reusable detector outputs have separate
Python/JSON API services;
tracker execution, temporal editing and tracking metrics are not implemented yet.
Full segmentation workflows remain outside this version. No ARGOS or drone
integration is required.

## Application boundary

One Python application serves a FastAPI JSON API and a plain HTML/CSS/JavaScript
interface. SQLite stores metadata; a configurable local data directory stores
media and artifacts. A subprocess worker executes extraction, inference, annotation,
training and evaluation jobs without blocking requests. No Node build
step, database server, cloud account, or GPU is required for the data workspace.

Processing code is separate from the API and UI. Detection adapters expose
inference and declare whether training is supported. An annotation-provider
adapter consumes selected images or crops and returns reviewable suggestions.
Processing jobs share a persisted lifecycle: queued, running, succeeded,
failed, cancelled, or interrupted. Jobs retain configuration, logs, errors, and
useful partial artifacts; interrupted work is identified on restart, never
reported as success.

New extraction jobs carry a frozen sampling contract in `jobs.params` and durable
position checkpoints in `jobs.result`. Recovery verifies the source, retained
images, pinned taxonomy and deduplication inventory, then creates a linked attempt
with a compare-and-swap fingerprint. The old terminal job is preserved. This is
separate from the training continuation protocol described below, which also
restores the optimizer and random states.

Provider records retain a dispatch receipt in their metadata. Atomic claims prevent
two workers from executing one saved request. Dispatch is recorded before network
write, and confirmed receipt is distinguished from an uncertain transport outcome.
Startup reconciles unfinished receipts without submitting requests. Local batch
recovery creates a new, explicitly previewed batch for unfinished images without
saved proposals, excluding successes. These additions use schema 14; no old frozen
manifest or model contract is rewritten. See [jobs and recovery](job-recovery.md).

The server binds to loopback by default. A remote workstation can be reached
through an SSH tunnel; computation and storage remain on that workstation.
This is a single-user application, not an authenticated public service.

## Projects and compatibility

SQLite schema 21 adds `temporal_detection_caches` and `temporal_detection_frames`.
Immutable cache recipes pin a sequence, ordered frames, verified local detector
weights, classes, preprocessing, native filtering, CPU/CUDA selection and runtime
sources. The worker saves an execution receipt and commits each complete image
output together with its attempt checkpoint. Retained rows form an exact prefix;
explicit continuation creates a new job for the remaining frames. Read-only
score/class filters preserve source indices and never run inference. Archive
validation checks recipes, output hashes and attempt lineage without requiring
weights or the producing runtime. These caches do not implement association,
tracking metrics or a Studio tracking panel. See
[temporal detector caches](temporal-detections.md).

SQLite schema 20 adds `temporal_sequences`, `temporal_references` and
`temporal_datasets`. Sequence and dataset manifests are immutable hashed JSON;
reference revisions use optimistic concurrency and preserve their declared
review provenance. Sources retain project ownership, a frozen taxonomy, clock
provenance, file/pixel hashes and explicit missing-frame ranges. Temporal identity
labels never reuse image annotation IDs implicitly. Frozen image and temporal
datasets share source-video, pixel and declared group split reservations.
Archives preserve these records and verify their media references. No historical
annotation, model, dataset or experiment is rewritten. See
[the temporal contract](temporal-data.md).

SQLite schema 13 adds `projects` and a project foreign key on sessions, dataset
versions and COCO imports. Other ownership follows the existing session, dataset
and evaluation relationships; immutable saved payloads are not rewritten. Opening
an older workspace atomically attaches existing records to `default`. IDs, hashes,
annotation revisions, checkpoints and experiment snapshots retain their meaning.
Schema 14 adds immutable `taxonomy_versions` and a `frames.taxonomy_id` intake
version. Each project points to its current version; a frame's latest annotation
revision can explicitly adopt a newer version. The legacy definition remains
unchanged, including its serialized payload in old manifests.

The HTTP layer resolves a request-local project from `project_id` (defaulting to
`default` for existing clients). Lists are scoped before rendering, record reads
and mutations check ownership, and trained models cannot be selected from another
project. Reference selection is project-local. Official detector weights, provider
availability, the sequential worker and workspace transfers remain shared.

The browser keeps its project fixed for the lifetime of the page and changes
projects through local navigation. Existing unsaved-work protections apply, each
project remembers its session, and API, media and download URLs carry the project.
This avoids retaining another project's selection, modal or pending preview.

Scene-group reservations are project-local, while exact-pixel split reservations
span the workspace. Creating a project cannot turn an existing training image into
an independent test image. Related scenes still need human grouping and review.
Archive validation recognizes the exact structures of schemas 12 through 21. Restore
preserves archive payload bytes; opening an older restored workspace performs
the normal additive migration. See [project behavior](projects.md).

## Model and annotation choices

The detector backend is PyTorch/Torchvision, installed separately
from the lightweight workspace dependencies:

| Model | First capability | Official checkpoint size |
| --- | --- | --- |
| [SSDLite320 MobileNetV3-Large, COCO_V1](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.ssdlite320_mobilenet_v3_large.html) | Inference and fine-tuning | 13.4 MB |
| [Faster R-CNN MobileNetV3-Large 320 FPN, COCO_V1](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn.html) | Inference and fine-tuning | 74.2 MB |

They share an optional PyTorch runtime with CPU or NVIDIA CUDA execution. These are starting
points for a small experiment, not validated aerial detectors. Small distant
objects may require higher resolution or different models after measurement.
Torchvision supports a [standard detection fine-tuning workflow](https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html).
The training integration replaces the Faster R-CNN prediction head for the
project classes and records the frozen layers and optimizer configuration.
It copies background and explicitly mapped COCO rows into an N+1 class head,
initializes other rows using the saved seed, and optimizes only the final
classifier and box regressor by default, and preserves the feature extractor and
proposal network in this light mode. Each run can instead select partial or full
adaptation. Partial training unfreezes the last MobileNet stage (`backbone.body`
blocks 13–16), the FPN, RPN and ROI heads; full training unfreezes all learnable
parameters. Frozen batch-normalization statistics remain fixed in all modes.
Trained descendants can become parents at any depth when their frozen class
snapshot and mappings exactly match the dataset. Legacy native labels 1/2 retain
COCO output IDs 1/3; custom outputs use saved class IDs 1…N. Complete class
contracts travel with training, model specs, comparisons, evaluation and reports.
Checkpoints store model state and explicit architecture metadata. Installation
size, memory use, and runtime are larger than the weight files alone.

Training capabilities are selected from a pure architecture registry, without
importing the optional ML runtime in catalog requests. SSDLite uses its native
six-scale classifier layout: class projection rows are transferred per anchor,
while depthwise features and class-independent regression are preserved. The
training-only adapter supplies background loss for empty images and pins its
policy in the configuration. Ordinary SSDLite BatchNorm retains its running
statistics in all scopes; affine weights follow the selected scope. The existing
Faster R-CNN scope and recovery contracts remain valid. Completed models retain
the selected architecture and can be evaluated together on the same class version.
Native SSDLite exports have their own frozen profile; no model conversion or
schema migration is needed. See [trainable models](trainable-models.md).

Visual comparisons support full-image and tiled runs. Model IDs always identify
the original checkpoint; a separate `full` / `tiled` run variant identifies the
inference pipeline. A paired comparison contains one model and two ordered runs.
Predictions, annotation sources and disagreement signals are tied to run IDs,
so two pipelines cannot silently select each other's outputs. Legacy comparisons
retain their IDs and acquire the full-image variant through SQLite migration.

Comparison details also expose a read-only replay projection of their saved
frames, grouped by video source and ordered by recorded timestamp. The browser
uses the existing local media endpoint and byte-range responses for playback.
Sample coverage counts expected run identities, including separate full/tiled
variants; a saved empty prediction counts as processed. No database migration,
inference job, generated video or interpolated detection is involved.

The replay player and result cards have separate clocks: source-video position
and the timestamp of the extracted image currently displayed. Sparse coverage
is explicit between samples. Recorded positions use nominal FPS and are
approximate, especially for variable-frame-rate sources. Playback does not
certify frame-accurate alignment. Media availability checks the workspace path,
file type and recorded byte size; it does not rehash entire videos on each
comparison refresh or certify that same-sized files have not changed.

`tiling.py` plans deterministic, bounded crops over original pixels, invokes the
existing detector and maps outputs back before class-aware NMS. It adds no model
or framework dependency. Frame hashes, tile settings, merge protocol, region
coordinates and native per-region outputs are persisted. Partial tiled images
are never published. The preparation endpoint only counts work; it does not load
weights, read image pixels or create a job. The worker rechecks the frozen plan.
Held-out evaluations use the same inference implementation. Each model result
has a separate full/tiled variant, and predictions refer to that result's ID.
Reference decisions retain the selected checkpoint and its inference settings;
test audits must reuse validation's ordered variants, tile and merge settings,
and timing protocol. Historical evaluations remain full-image results.

The initial taxonomy, `iris-objects-v1`, contains `person` and `car`, with written
class definitions and an explicit COCO mapping (IDs 1 and 3). People include
riders; cars include passenger SUVs/minivans but exclude buses, trucks and
motorcycles. Boxes cover visible extents. Definition changes require a new
taxonomy version.
Custom classes and definitions are supported in the manual editor and COCO import.
Publication uses compare-and-swap on the project's current version. Adoption creates
a new draft revision, retains earlier revisions and requires human validation again.
Custom detector proposal mappings use explicitly configured COCO IDs. Frozen datasets
and COCO exports support custom classes, as do Faster R-CNN and SSDLite training, inference,
evaluation, disagreement review and reports. Multimodal candidate review retains
the original definitions. See [class version behavior](classes.md).
An unsupported class is not silently mapped to a superficially similar class.

Direct detector preannotation reuses comparison runs and the sequential inference
worker. `config.preannotation` freezes each frame's source hash, saved annotation
revision, class version, explicit output mapping and threshold. Preparation is
read-only; confirmation rechecks its fingerprint under a write transaction. Repeated
confirmation returns the same receipt, while a fresh preview permits another run.
Native predictions are saved before proposal validation. Each frame's publication
receipt and proposals are committed together; a changed annotation is a visible
conflict that preserves raw output and human work. Cancellation preserves partial
results and never automatically restarts the detector.

`preannotation_contracts.py` describes implemented capabilities and validates
bounded box outputs, source class identity and explicit coordinate transforms.
The real detector uses this normalization boundary. Fixtures exercise possible
future adapter shapes; they are not exposed as executable providers. Candidate
reviewers declare their existing geometry and class limits. See
[preannotation](preannotation.md). These records use schema 14 without migration.

DINO-X cloud proposals use the same annotation editor and review decisions. Schema
19 adds project-owned batches and request receipts, freezing image identity, taxonomy,
prompts, settings and a CNY allowance before queueing. Workers record submission intent
before sending pixels, persist the remote task ID before polling, and save native
responses before normalization. Successful responses can be reused without another
detection call; stable suggestion IDs preserve human decisions. Ambiguous submissions
are never replayed automatically. Archives validate and preserve these records offline,
with continued support for schemas 12–18. Credentials remain outside the workspace.
See [DINO-X preparation, review and recovery](dinox-preannotation.md).

Assisted annotation uses a configurable local Ollama endpoint, initially
[Qwen3-VL 4B Instruct](https://ollama.com/library/qwen3-vl:4b-instruct). The
published quantized artifact is approximately 3.3 GB and requires Ollama 0.12.7
or newer; model download and runtime provisioning are separate setup steps.
The [vision API](https://docs.ollama.com/capabilities/vision) accepts image bytes,
and [structured outputs](https://docs.ollama.com/capabilities/structured-outputs)
allow schema-constrained suggestions. Saved detector outputs or manually drawn
labels supply 1–8 candidate boxes. The model reviews their categories and
ambiguities, with possible omissions recorded as scene notes for a human. It
never generates coordinates: proposals retain the original candidate geometry.
Only `person`, `car`, `none`, and `uncertain` are accepted in the model response.
Manual annotation works without Ollama. The local adapter refuses remote endpoints,
redirects, proxies, cloud models and remote aliases. Model installation is
explicit; availability checks never send pixels or generate output.

The provider selector also offers Qwen3-VL 32B and 235B-A22B Instruct through
Alibaba Cloud Model Studio. Hosted configuration is checked offline. The
Frankfurt endpoint is workspace-specific and these models use Global deployment
scope. API access requires a server-side key and an explicit preview/confirmation
for each request; selecting the provider alone never transmits images.

An external preview freezes the review configuration, candidate geometry, exact
encoded scene/crop files and their hashes, and dated pricing assumptions. It
expires after 30 minutes. Confirmation rechecks revision, pixels, selection,
provider and cost ceiling, then consumes the preview atomically with job creation.
The worker verifies the approved snapshot and bytes again before transmission.
There are no automatic paid retries, external fallbacks or background API probes.

Hosted models support JSON Object mode; the same strict local output validator
is used for both providers. Hosted provenance records provider model IDs and
reported usage rather than claiming access to immutable weights. Cost bounds
use the documented maximum input and capped output at recorded list prices;
they exclude taxes and subsequent provider price changes.

Requests freeze the image hash, candidate coordinates, base annotation revision,
model digest, instructions and provider configuration before entering the job
queue. At execution the adapter verifies the local model identity, supplies a
resized scene and ordered crops, and requires one structured review per
candidate. Prompt messages, model settings, digest, raw response and failures
are saved. A failed or cancelled response cannot silently validate labels.
Cancellation stops the IRIS worker; the Ollama server may finish its current
generation before releasing memory.

Annotations use immutable full revisions with a compare-and-swap revision number.
An editor cannot overwrite a newer save. Automatic suggestions live separately
and require an explicit accept, correct or reject decision. Validation requires
a reviewer name and no unresolved proposals; it means the whole image was
reviewed, including missed objects. A later proposal does not alter an older
validated revision. Re-reviewing an existing box may replace its origin only if
the box still matches the saved revision that the model examined; earlier
decisions remain in history. Dataset releases freeze a specific validated revision
and require all current suggestions to be resolved.

Dataset publication holds a SQLite write transaction while copying verified
images to a staging directory. The completed directory is renamed before its
manifest hash and summary become visible in SQLite. Failures remove unpublished
artifacts. Releases copy their PNGs, full annotation revisions and source metadata;
later source edits cannot change their training inputs. Existing manifests reserve
scene groups, exact pixel hashes and original video hashes to one split across
versions. Scene groups are project-scoped; pixel and video reservations span the
workspace without exposing other projects' records. Read-only partition planning
links whole groups by shared sources, honors these reservations and returns
expected revision IDs. Historical manifests are not rewritten. A checksummed
manifest and image hashes are rechecked before consumption. These hashes detect
artifact changes; they do not replace backups of the complete workspace.

Intake selection is independent of annotation. The browser import queue captures
its target session and reports per-file outcomes; it is not a durable background
job. Bulk selection checks every expected boolean state in one SQLite transaction
before updating any frame. Bounded, read-only selection insights join human review
state with duplicate hashes and compatible saved predictions. No inference or
automatic selection runs as part of these signals. See [intake and selection](intake-selection.md).

New releases use manifest schema 2 with one complete, immutable class snapshot,
`class_mapping` and `coco_mapping`. The selected annotation revisions must all
use that same version. Candidate revision IDs provide a publication conflict check,
and parents must share the release's selected class version. Schema-1 manifests remain
readable without rewriting their bytes. These manifest formats are independent
of SQLite schema 14. Frozen readers resolve classes from the manifest itself,
without consulting current project definitions or annotations.

COCO export reads a frozen release independently of current annotations. It
verifies the original manifest and the exact PNG bytes written to a temporary ZIP,
converts boxes to COCO coordinates and category IDs, and preserves the split
assignments and original provenance. Preparation runs outside the event loop;
one export per application instance is allowed at a time. The archive is bounded
to 256 MiB and removed after the response, including interrupted downloads. No
dataset revision, job or model is created. See [the export contract](dataset-export.md).

Training snapshots the dataset and parent checkpoint hashes before queueing and
checks them again in the worker. A read-only preview validates the same inputs
and reports train images, planned visits and complete passes without loading a
model or creating a job. The interface requires another preview after an edit.
The scope and its versioned module contract are frozen in each run; the worker
rejects unsupported or changed contracts. Existing head-only runs remain valid.
Actual trainable/frozen parameter counts and changed modules are recorded with
the checkpoint; parameter and normalization checks preserve frozen model state.
Training uses a zero RPN score threshold so negative images still supply
background proposals; the inference preset's threshold can discard every
proposal on such images. Training proposal filtering is recorded separately
from inference filtering, which remains unchanged when loading a checkpoint.
CPU or single-GPU CUDA SGD runs for up to 10,000 steps with batch size one and a recorded seed; only
train images are read. Durable runs persist loss components and visited frame IDs
every ten steps or when saving a checkpoint. Completed state dictionaries are
published atomically with the model registry entry and loaded with `weights_only`.
Cancelled or failed runs retain their history and logs without registering a
partial model. The registry records
ancestor training groups/hashes and refuses parents that have consumed a new
dataset's held-out data. This cannot establish independence from the official
parent's pretraining corpus.

Schema 18 adds `training_checkpoints`, owned through the training run and its
dataset. Recovery states are separate from inference models: they contain full
model state, SGD momentum, module modes, gradient evidence, CPU RNG and the
remaining image order. Their metadata binds the original configuration, dataset,
parent, history prefix and runtime identity. The worker claims an attempt once;
state publication and its history prefix commit together, and late workers cannot
publish after interruption. The latest two registered states per attempt are
retained. New training previews bind creation to the displayed configuration.

Explicit continuation checks a stopped attempt and creates at most one successor,
preserving the source run. The successor repeats work beyond the latest durable
step and restores the saved optimizer only after compatibility checks. A final-step
state can retry model publication without another optimizer step. Legacy short
runs without recovery state remain readable. Archives preserve schema 12–17
databases and validate schema 18 lineage and state hashes without deserializing
Torch files. See [longer training and continuation](long-training.md).

Training-device previews pin the selected CUDA index, observed GPU identity and
runtime; a bounded subprocess inspects hardware and registered detection kernels
without loading weights. Workers recheck that identity, place model parameters,
replacement heads and targets on the same device, and synchronize recorded CUDA
step completion. CUDA recovery uses a separate protocol containing the selected
GPU RNG alongside CPU RNG. It preserves optimization state on that same device;
nondeterministic CUDA operations preclude a bitwise-repeatability promise.
CPU continuation records retain their original protocol and runtime identity.

Completed model and recovery tensor files are normalized to CPU storage. A
completed model can independently use CPU or CUDA for subsequent training,
comparison, evaluation or export. Hardware selection never changes the class,
dataset or frozen-layer contract. CUDA exhaustion preserves published recovery
states and fails visibly without an automatic CPU fallback. See
[compute targets and environment setup](compute-targets.md).

Experiment reports capture one successfully completed evaluation and its dataset,
checkpoint and available training lineage in a versioned, checksummed snapshot.
They reuse saved metrics and error matches without loading model weights or
recalculating quality scores. Selected examples are copied into bounded local
JPEG previews, retaining the original coordinate space for overlays. The snapshot
and examples stay fixed; title, objective and conclusion use optimistic revision
checks. An additive SQLite table stores report metadata. Standalone HTML exports
use explicit public fields, escaped text and embedded styles, with optional
images and no scripts or network dependencies. See [experiment reports](experiments.md).

Report snapshot v2 adds pure operating-point aggregation in `experiment_insights.py`:
per-scene counts, frame changes, bounded example suggestions, sampled-video context
and conservative timing comparability. `experiment_deployments.py` validates selected
saved export measurements against the exact full-image lane, metadata, dataset,
checkpoint and reference predictions. It freezes only explicitly selected summaries,
without loading models or including raw measurement detections. Preview enumerates
at most 100 recent measurements; creation reads only the zero to four selected IDs
and rechecks evidence under the publication lock. The preview source fingerprint
rejects stale evaluation evidence. Existing v1 snapshots remain unchanged and readable;
this report format does not require a schema change. Local evaluation timing and declared target timing have
different boundaries and are never combined into a cross-context speedup.

Evaluations are separate from session-based visual comparisons. They consume the
whole validation or test split of one immutable release, including frames from
multiple sessions. They freeze checkpoint hashes, ancestry, dataset hash and
protocol before queueing, and verify them again in the worker. Predictions and
timings persist per frame; each model receives metrics only after its complete
split has finished. Publishing the last model's metrics and job success is atomic.
Cancelled and failed runs preserve completed outputs without presenting a
partial model as a complete score.

The error explorer derives per-frame and per-class counts from completed saved
evaluations. Paired recoveries and new misses use the same frozen reference
indices; false-positive changes compare counts only. It checks saved record
identities and error partitions without rerunning detectors, rematching objects
or recomputing AP. Filters select examples for the existing image viewer and do
not mutate releases, metrics, reference decisions or test reservations.

`pycocotools` 2.0.11 implements COCO bbox AP; separate deterministic matching
provides confidence-specific precision/recall and per-frame errors. Class IDs
and definitions are explicit, including classes without reference instances.
The [evaluation protocol](evaluation.md) describes the limits. Test audits refer
to a successful validation run and cannot change its dataset, models, thresholds,
device or protocol. Reference decisions require complete validation results,
a reviewer, a reason and the current reference ID to prevent stale overwrites.
Decisions append to history; they never delete previous checkpoints or trigger
automatic training or deployment.

Torchvision's [code license is BSD-3-Clause](https://github.com/pytorch/vision/blob/main/LICENSE),
but its documentation [separates pretrained-model terms from the code license](https://docs.pytorch.org/vision/stable/models.html#general-information-on-pre-trained-weights).
The [Qwen model card specifies Apache-2.0](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct).
Record the source, exact revision or digest, and license reference for each
installed checkpoint. Imported media and third-party datasets require their own
usage and redistribution rights; neither code nor model licenses grant those.
Model files, datasets, private media, and credentials stay outside Git.

## Data and evaluation invariants

- Video sampling uses one bounded plan for metadata preview, thumbnail preview
  and worker execution. Explicit `sampling_mode: uniform` distributes up to
  500 original frame indices across a chosen half-open time range; an omitted
  mode retains historical fixed-interval sampling and its provenance. The
  interface defaults to uniform sampling. Planning allocates no full-video grid
  and decodes no pixels; explicit thumbnail preview decodes at most 12 planned
  positions without storing artifacts. Source checksums are verified before
  thumbnails or extraction. Completed jobs retain the plan and skip counts;
  cancellation preserves completed frames and retries reuse the same indices.
  Duplicate filtering may reduce temporal coverage. All extracted frames remain
  unselected until human selection; temporal sampling itself implies no semantic
  analysis. A separate [video passage review](video-review.md) prepares immutable
  JPEG storyboards and queues a single multimodal request. Model outputs identify
  observed sample IDs; the server resolves their timestamps. Passage extraction
  requires a separate human choice and recomputes the frozen extraction plan.
  Exact images, provider identity, prompts and raw responses remain recorded;
  external processing requires approval and a per-request budget. Review jobs
  never create frames, annotations or selection changes. Their result and final
  success state publish atomically, and interrupted requests are never retried.
- Every frame retains its source hash, session, extraction configuration, decoded
  dimensions, and source timestamp when available. Exact duplicates are detected
  using content hashes; related scenes need human grouping as well.
- Boxes use floating-point `xyxy` pixel coordinates in the stored image's
  orientation, with an exclusive right/bottom edge. Adapters own normalization,
  resizing, and COCO `xywh` conversion; conversion tests cover boundaries.
- Suggestions preserve their provider, checkpoint, prompt, parameters, and raw
  response. Proposed, corrected, rejected, and human-validated annotations are
  distinct. A validated empty image is an explicit negative, never a missing label.
- Dataset manifests freeze image hashes, annotation revisions, taxonomy, and
  train/validation/test assignments. Subsequent edits create new versions. Split
  by session or related scene group; reject duplicate content crossing splits and
  require review of near-duplicate scenes. Never split adjacent frames randomly.
- Validation supports model and threshold selection. A fixed test reference is
  excluded from training and tuning; reusing its examples for improvement retires
  its independent-test status. Insufficient independent groups are reported as a
  limitation, not repaired by distributing frames across splits.
- Evaluation uses human-validated labels and a recorded class mapping. Store
  predictions before display filtering. Use [COCO evaluation](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py)
  for AP at IoU 0.50:0.95, AP50, and per-class AP, plus precision/recall at stated
  confidence and IoU thresholds. Record prediction cutoffs, maximum detections,
  ignored regions, and classes without reference instances.
- Compare quality only on the same reference version and protocol. Record device,
  library versions, resolution, warmup, and batch size; distinguish preprocessing,
  inference, postprocessing, and total time. Confidence and model disagreement are
  inspection aids, not accuracy measurements. Promotion of a checkpoint is manual.

## Existing improvement loop

Annotated external data can enter the same loop through a bounded COCO ZIP
importer. Preview verifies images and geometry, then explicit class mapping and
source metadata produce reviewable proposals in a new session. Source split
reservations apply before review and survive dataset versions. Imported labels
never bypass human validation. See [the import contract](coco-import.md).

A read-only annotation queue summarizes the latest revisions and unresolved
proposals. Optional class-aware box matching between two saved detector outputs
helps order inspection without using reference labels as scoring input. The
queue preserves selection and split reservations; disagreement never validates
an image. See [the matching protocol and review behavior](review-queue.md).

Local annotation batches group up to 25 explicit frame requests into durable
`assist` child jobs. An eligibility preview fingerprints saved revisions, source
predictions, image hashes and the local model digest. SQLite schema 8 adds an
`assistance_batches` table; batch creation rechecks inputs and queues all eligible
children in one transaction. The existing sequential worker, per-frame provenance
and publication checks remain shared with single-image assistance. Aggregate
status is derived from persisted child jobs, so partial results, cancellation and
server interruption remain visible after reopening. See [batch behavior](annotation-batches.md).

All five increments are implemented on the initial person/car detection scope,
including quantitative evaluation and explicit reference selection. Live model verification depends on
explicit runtime provisioning. The README records setup commands and verification limits.

| Increment | Usable result | Acceptance check |
| --- | --- | --- |
| 1. Local data workspace | Import images/video into sessions, extract frames, inspect provenance, select images, inspect and cancel jobs. | Import a generated video fixture, extract and select frames, restart, and recover metadata and job outcomes. Fixtures exercise the pipeline; they are not field data. |
| 2. Saved model comparisons | Run both detectors on identical frames, persist raw predictions and timing, overlay results and inspect disagreements. | Execute both real checkpoints on a small authorized sample and reopen the comparison after restart; missing weights are a visible dependency. |
| 3. Assisted and manual annotation | Create, move, resize, reclassify, and delete boxes; request local multimodal suggestions and accept, correct, or reject them. | Complete one real Ollama request and human review; test invalid responses, coordinate conversions, and validated empty images. Test doubles are identified as fixtures. |
| 4. Dataset versions and training | Freeze reviewed labels and group-based splits; run bounded detector fine-tuning; register the resulting checkpoint. | Reject leakage and unreviewed labels, preserve old manifests after edits, and complete a real short training job that produces a reloadable checkpoint. |
| 5. Before/after evaluation | Reuse the comparator for parent and trained checkpoints, compute metrics on a common held-out reference, and inspect regressions. | Reproduce the full chain from new source data to a checkpoint and evaluation after restart. Report gains or regressions as measured; never require an improvement to declare the loop functional. |

The full demonstration on real data needs authorized recordings, independent scene groups,
human-reviewed labels, installed detector weights, a working multimodal runtime,
and suitable compute. Missing dependencies block the corresponding live
verification, not manual data handling or fixture tests. Large downloads and
heavy training are explicit setup actions, never startup side effects.

## Independent annotation benchmarks

Schema 15 adds benchmarks, immutable configuration records, trial jobs, raw and
normalized per-image outputs, correction revisions and acknowledged review
intervals. The frozen reference has its own copied images and checksum-verified
manifest. Reference boxes and review decisions do not enter the detector adapter;
quality is scored after candidate output has been saved.

The benchmark API is project scoped. A sequential `benchmark` worker evaluates
only the role and configuration confirmed by a preview fingerprint. The whole
configuration set locks before evaluation, while incomplete trials retain their
coverage and errors without a headline aggregate. Review intervals use server
monotonic time with short leases, ownership and revision checks. Startup and
shutdown preserve acknowledged time and interrupt unfinished intervals. Restored
workspace archives retain their original database bytes until normal startup.
See [the protocol](benchmark.md) for scene reservations, independence declarations,
metric definitions and timing limitations.

The optional multimodal benchmark adapter receives only copied pixels, frozen
class definitions and fixed model settings. It sends one OpenAI Responses request
per approved image. `benchmark_multimodal.py` builds local outgoing previews and
short-lived approval receipts; `benchmark_dispatch.py` journals per-image budget
reservations before transport and preserves raw responses separately from valid
proposals. Unknown outcomes retain their reservation, with no automatic retry.
The worker stops the remaining requests after any failure. Token-based costs use
the saved price schedule and are distinct from the provider invoice. These
records reuse the schema 15 tables and survive offline archive validation.

The local SAM path uses `benchmark_segmentation.py` to freeze per-class phrases,
native score threshold, checkpoint identity and execution protocol. Trial previews
bind the observed runtime identity, and admission checks input image bounds before
loading a model. `sam_provider.py` validates native outputs and converts normalized
boxes into original image pixels. `sam_runtime.py` communicates through bounded
JSON lines with the standalone `sam_runtime_worker.py`, launched by the separate
Python executable in `IRIS_SAM_PYTHON`. Only the worker imports the optional Meta
CUDA stack; NumPy version requirements remain isolated from the IRIS environment.

SAM loads once per trial, disables masks, encodes each image once and grounds each
class phrase independently. Runtime identity is checked again after loading.
Image timings include the cold first prediction and exclude separately recorded
model initialization. Cancellation stops the process group; a Linux parent-death
guard stops the worker if its owning job exits. Raw evidence is saved before
normalization, and partial images cannot publish successful proposals. Archives
validate the frozen protocol without probing SAM. They preserve the pinned
checkpoint when present, while histories can be restored without it or the
external runtime. See [the SAM adapter](sam-preannotation-adapter.md).

The combined candidate uses `benchmark_combined.py` to run a single three-stage
pipeline: Astra plans one phrase per class, SAM grounds those phrases, and Astra
reviews the identified SAM candidates. `combined_provider.py` freezes the prompt
and schema contracts, produces exact planning requests and bounded review
templates, and validates decisions without accepting generated geometry.
`combined_sam.py` reuses one loaded model. Only this path explicitly enables
phrase updates between images; the standalone SAM path retains fixed prompts.

`benchmark_combined_dispatch.py` stores the three stage receipts and raw responses
inside each existing benchmark output row, using schema 15. External stages have
separate budget reservations committed before transport. Each exact review input
is derived from the saved planning and grounding results and checked against the
approved template. A claimed trial or started stage cannot be automatically
resent. Cancellation preserves completed evidence and prevents later publication.
Recovery classifies interrupted external sends as unknown outcomes and never
restarts the pipeline. Total image timing includes all stages and the first model
load; initialization is also recorded as a subset of that elapsed time.

Archives reconstruct safe input digests and normalized stage results offline.
The full POST digest remains a recorded transport identity; verifying it from
metadata alone cannot reconstruct the encoded image bytes. Final boxes retain
SAM geometry, while their confidence is null and the original native score stays
in provenance. This avoids treating two provider outputs as a calibrated score.
See [the combined protocol](combined-preannotation-adapter.md).

## Standalone model packages

Schema 17 adds `model_exports` and `model_export_measurements`. Ownership follows
the trained model's training dataset; the source evaluation must be in the same
project. Versions 12–16 retain their original table layouts and database bytes
when restored, migrating only on opening.

`model_exports.py` verifies a completed CPU or CUDA full-image evaluation against the
trained checkpoint's frozen class, runtime and inference contracts. Preview binds
the chosen image bytes, saved native predictions, checkpoint and runner source to
a canonical digest. Creation compares that digest in one SQLite transaction and
queues an idempotent copy job. A bounded staging ZIP is verified before atomic
publication of the package record and terminal job success. Cancellation keeps
incomplete copies out of the published inventory; retries require a new preview.

`export_runner.py` is copied verbatim into the bundle and imports no IRIS module.
Inspection and measurement validation use the standard library; explicit predict
and measure commands load the pinned runtime and full state_dict onto the selected
CPU or CUDA target. Version-1 CPU bundles retain their original contract; version-2
bundles record a separate reference device and target. CUDA measurements synchronize
the selected device and preserve hardware/runtime evidence. Class order,
legacy native-to-output IDs, EXIF orientation, Torchvision transforms, native NMS
and prediction order remain frozen. Exact parity does not silently widen numeric
tolerances. Real acceptance trials exercised both architectures' light-scope
checkpoints across all four CPU/CUDA training-to-target paths in separate
environments without IRIS and with networking disabled on one CPU/RTX 4060 host.
Same-device references passed exact parity; cross-device references failed on
small numerical differences. Four separate control bundles with new target-device
references passed while preserving the original failures. These bounded
measurements do not certify other models or hardware; see
[the export validation evidence](model-export.md#what-is-verified).

External measurement JSON is bounded, linked to its manifest, checked for complete
ordered repeats and finite timings, and recomputed before saving. Results are
immutable and idempotent per payload hash; declared execution is not independently
verified. Archives validate nested file inventories and hashes, saved predictions,
source ownership and imported summaries without deserializing checkpoints.
See [the export contract](model-export.md).

An explicit standalone `check-runtime` command checks dependencies and CUDA
availability without model loading or inference. It does not certify an embedded
board: ARM and JetPack installations must satisfy the pinned Python and framework
requirements and still need real target parity, memory and timing measurements.

## Saved benchmark comparisons

Schema 16 adds only `benchmark_reports`. Its owner is an existing benchmark, so
project access follows the same foreign-key chain as configurations and trials.
Schema 12–15 archives retain their original database bytes on restoration and
migrate only when the workspace is opened. No benchmark outputs or correction
revisions are rewritten by this addition.

`benchmark_analysis.py` reads one SQLite transaction for a single reference role,
including every configuration and attempt. It recomputes box quality from saved
normalized outputs instead of trusting the job's cached summary. Only terminal
successful trials with complete image coverage contribute to quality ranges;
failed and missing outputs remain visible. Repetition comparisons preserve box
order and exclude IDs, scores and explanations from geometry equality. Local,
external and combined timings retain their different measurement scopes.

`benchmark_reports.py` freezes the comparison together with its evidence-origin
declaration and author interpretation. Preview/save compare a canonical digest
under a write transaction; new trials, corrections or edited text require a new
preview. Saving is idempotent for the same current snapshot. Reports do not run
models, and the evidence declaration is the author's statement, not automatic
verification of model execution. A report requires at least one terminal trial
and no active trials in its selected role.

Saved snapshots reference their exact correction revisions and normalized source
digests. Read and archive validation reconstruct their measurements against those
historical sources, allowing later corrections and trials without rewriting the
report. `benchmark_report_api.py` applies project ownership to previews, saved
reports and downloads. `benchmark_report_export.py` renders the saved values as
JSON or escaped, script-free standalone HTML; it performs no provider calls or
image requests. Image comparison stays in IRIS using the frozen reference images.
