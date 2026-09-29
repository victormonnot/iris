# Architecture and V1 delivery

IRIS is a local workbench for improving object detectors from flight recordings.
V1 covers import, model comparison, assisted annotation, human review, dataset
versions, fine-tuning, and comparison against earlier checkpoints. Onboard
deployment, drone control, tracking, and segmentation are outside this version.

## Application boundary

One Python application serves a FastAPI JSON API and a plain HTML/CSS/JavaScript
interface. SQLite stores metadata; a configurable local data directory stores
media and artifacts. A subprocess worker executes extraction, inference, and annotation jobs;
the same boundary will host training without blocking requests. No Node build
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
The planned training integration will replace the Faster R-CNN prediction head for the
project classes and records the frozen layers and optimizer configuration.
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
Manual annotation works without Ollama. The adapter refuses remote endpoints,
redirects, proxies, cloud models and remote aliases. Model installation is
explicit; availability checks never send pixels or generate output.

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
decisions remain in history. These revisions are not yet dataset releases.

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

The data workspace, saved comparisons and annotation editor/provider integration
(increments 1–3) are implemented. Dataset releases/training and quality evaluation
(increments 4–5) remain planned V1 work. Live model verification depends on
explicit runtime provisioning. The README records setup commands and verification limits.

| Increment | Usable result | Acceptance check |
| --- | --- | --- |
| 1. Flight data workspace | Import images/video into sessions, extract frames, inspect provenance, select images, inspect and cancel jobs. | Import a generated video fixture, extract and select frames, restart, and recover metadata and job outcomes. Fixtures exercise the pipeline; they are not flight data. |
| 2. Saved model comparisons | Run both detectors on identical frames, persist raw predictions and timing, overlay results and inspect disagreements. | Execute both real checkpoints on a small authorized sample and reopen the comparison after restart; missing weights are a visible dependency. |
| 3. Assisted and manual annotation | Create, move, resize, reclassify, and delete boxes; request local multimodal suggestions and accept, correct, or reject them. | Complete one real Ollama request and human review; test invalid responses, coordinate conversions, and validated empty images. Test doubles are identified as fixtures. |
| 4. Dataset versions and training | Freeze reviewed labels and group-based splits; run bounded Faster R-CNN fine-tuning; register the resulting checkpoint. | Reject leakage and unreviewed labels, preserve old manifests after edits, and complete a real short training job that produces a reloadable checkpoint. |
| 5. Before/after evaluation | Reuse the comparator for parent and trained checkpoints, compute metrics on a common held-out reference, and inspect regressions. | Reproduce the full chain from a new flight to a checkpoint and evaluation after restart. Report gains or regressions as measured; never require an improvement to declare the loop functional. |

The full demonstration needs authorized recordings, independent scene groups,
human-reviewed labels, installed detector weights, a working multimodal runtime,
and suitable compute. Missing dependencies block the corresponding live
verification, not manual data handling or fixture tests. Large downloads and
heavy training are explicit setup actions, never startup side effects.
