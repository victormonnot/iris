# Architecture and V1 delivery

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
classifier and box regressor, and preserves the feature extractor and proposal
network. Trained descendants can become parents of subsequent runs. Native
labels 1/2 map explicitly to COCO IDs 1/3 for saved comparisons and annotation
proposals; raw native IDs remain recorded.
Checkpoints store model state and explicit architecture metadata. Installation
size, memory use, and runtime are larger than the weight files alone.

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

Training snapshots the dataset and parent checkpoint hashes before queueing and
checks them again in the worker. CPU SGD runs for a bounded number of steps with
batch size one and a recorded seed; only train images are read. Loss components
and visited frame IDs are persisted each step. Completed state dictionaries are
published atomically with the model registry entry and loaded with `weights_only`.
Cancelled or failed runs retain their history and logs without registering a
partial model. Resume from optimizer state is not available. The registry records
ancestor training groups/hashes and refuses parents that have consumed a new
dataset's held-out data. This cannot establish independence from the official
parent's pretraining corpus.

Evaluations are separate from session-based visual comparisons. They consume the
whole validation or test split of one immutable release, including frames from
multiple sessions. They freeze checkpoint hashes, ancestry, dataset hash and
protocol before queueing, and verify them again in the worker. Predictions and
timings persist per frame; each model receives metrics only after its complete
split has finished. Publishing the last model's metrics and job success is atomic.
Cancelled and failed runs preserve completed outputs without presenting a
partial model as a complete score.

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
