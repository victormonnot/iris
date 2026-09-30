# Architecture and V1 delivery

Workspace backup uses a streamed ZIP64 archive and a consistent SQLite snapshot.
An admission gate rejects new HTTP mutations while an idle workspace is copied;
database and file signatures also detect external changes. A background transfer
manager keeps operation receipts outside the archived inventory. Restoration
checks the manifest, all hashes, schema and file references, then atomically
publishes a new directory without replacing an existing destination. It preserves
the active workspace and starts no model jobs. The same archive services support
the UI and recovery CLI; see [backup and recovery](workspace-backup.md).

IRIS is a local workbench for improving object detectors from flight recordings.
V1 covers import, model comparison, assisted annotation, human review, dataset
versions, fine-tuning, and comparison against earlier checkpoints. Onboard
deployment, drone control, tracking, and segmentation are outside this version.

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

The server binds to loopback by default. A remote workstation can be reached
through an SSH tunnel; computation and storage remain on that workstation.
This is a single-user application, not an authenticated public service.

## Model and annotation choices

The detector backend is PyTorch/Torchvision, installed separately
from the lightweight workspace dependencies:

| Model | First capability | Official checkpoint size |
| --- | --- | --- |
| [SSDLite320 MobileNetV3-Large, COCO_V1](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.ssdlite320_mobilenet_v3_large.html) | Inference baseline | 13.4 MB |
| [Faster R-CNN MobileNetV3-Large 320 FPN, COCO_V1](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn.html) | Inference and fine-tuning | 74.2 MB |

They share one optional CPU runtime and label vocabulary. These are starting
points for a small experiment, not validated aerial detectors. Small distant
objects may require higher resolution or different models after measurement.
Torchvision supports a [standard detection fine-tuning workflow](https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html).
The training integration replaces the Faster R-CNN prediction head for the
project classes and records the frozen layers and optimizer configuration.
It copies the parent's background/person/car rows, optimizes only the final
classifier and box regressor by default, and preserves the feature extractor and
proposal network in this light mode. Each run can instead select partial or full
adaptation. Partial training unfreezes the last MobileNet stage (`backbone.body`
blocks 13–16), the FPN, RPN and ROI heads; full training unfreezes all learnable
parameters. Frozen batch-normalization statistics remain fixed in all modes.
Trained descendants can become parents of subsequent runs at any depth. Native
labels 1/2 map explicitly to COCO IDs 1/3 for saved comparisons and annotation
proposals; raw native IDs remain recorded.
Checkpoints store model state and explicit architecture metadata. Installation
size, memory use, and runtime are larger than the weight files alone.

Visual comparisons support full-image and tiled runs. Model IDs always identify
the original checkpoint; a separate `full` / `tiled` run variant identifies the
inference pipeline. A paired comparison contains one model and two ordered runs.
Predictions, annotation sources and disagreement signals are tied to run IDs,
so two pipelines cannot silently select each other's outputs. Legacy comparisons
retain their IDs and acquire the full-image variant through SQLite migration.

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
An unsupported class is not silently mapped to a superficially similar class.

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
scene groups and exact pixel hashes to one split across versions. A checksummed
manifest and image hashes are rechecked before consumption. These hashes detect
artifact changes; they do not replace backups of the complete workspace.

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
CPU SGD runs for a bounded number of steps with
batch size one and a recorded seed; only train images are read. Loss components
and visited frame IDs are persisted each step. Completed state dictionaries are
published atomically with the model registry entry and loaded with `weights_only`.
Cancelled or failed runs retain their history and logs without registering a
partial model. Resume from optimizer state is not available. The registry records
ancestor training groups/hashes and refuses parents that have consumed a new
dataset's held-out data. This cannot establish independence from the official
parent's pretraining corpus.

Experiment reports capture one successfully completed evaluation and its dataset,
checkpoint and available training lineage in a versioned, checksummed snapshot.
They reuse saved metrics and error matches without loading model weights or
recalculating quality scores. Selected examples are copied into bounded local
JPEG previews, retaining the original coordinate space for overlays. The snapshot
and examples stay fixed; title, objective and conclusion use optimistic revision
checks. An additive SQLite table stores report metadata. Standalone HTML exports
use explicit public fields, escaped text and embedded styles, with optional
images and no scripts or network dependencies. See [experiment reports](experiments.md).

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
  by flight or related scene group; reject duplicate content crossing splits and
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

## Five testable increments

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
| 1. Flight data workspace | Import images/video into sessions, extract frames, inspect provenance, select images, inspect and cancel jobs. | Import a generated video fixture, extract and select frames, restart, and recover metadata and job outcomes. Fixtures exercise the pipeline; they are not flight data. |
| 2. Saved model comparisons | Run both detectors on identical frames, persist raw predictions and timing, overlay results and inspect disagreements. | Execute both real checkpoints on a small authorized sample and reopen the comparison after restart; missing weights are a visible dependency. |
| 3. Assisted and manual annotation | Create, move, resize, reclassify, and delete boxes; request local multimodal suggestions and accept, correct, or reject them. | Complete one real Ollama request and human review; test invalid responses, coordinate conversions, and validated empty images. Test doubles are identified as fixtures. |
| 4. Dataset versions and training | Freeze reviewed labels and group-based splits; run bounded Faster R-CNN fine-tuning; register the resulting checkpoint. | Reject leakage and unreviewed labels, preserve old manifests after edits, and complete a real short training job that produces a reloadable checkpoint. |
| 5. Before/after evaluation | Reuse the comparator for parent and trained checkpoints, compute metrics on a common held-out reference, and inspect regressions. | Reproduce the full chain from a new flight to a checkpoint and evaluation after restart. Report gains or regressions as measured; never require an improvement to declare the loop functional. |

The full demonstration on real flights needs authorized recordings, independent scene groups,
human-reviewed labels, installed detector weights, a working multimodal runtime,
and suitable compute. Missing dependencies block the corresponding live
verification, not manual data handling or fixture tests. Large downloads and
heavy training are explicit setup actions, never startup side effects.
