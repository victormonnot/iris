# Documentation

Start with the [project overview](../README.md) to see what Iris does, or
[try the included example](first-run.md) to import two photos, review labels
and compare detectors. [From Iris to Argos](iris-to-argos.md) follows one model
from reviewed recordings to an exported detector used in another application.

The guides below cover the current application. [Recorded experiments](acceptance-results.md)
describe what has actually been tried, on which data and with which limitations.

## Run Iris

| Guide | Use it to… |
| --- | --- |
| [Setup](setup.md) | Install, choose a workspace and run locally or through an SSH tunnel. |
| [First run](first-run.md) | Try a small, attributed example without preparing your own dataset. |
| [Projects and compatibility](projects.md) | Organize sessions, models and results by project. |
| [Compute targets](compute-targets.md) | Set up CPU or NVIDIA CUDA execution. |
| [Backup and restore](workspace-backup.md) | Move or preserve the complete workspace. |
| [Job recovery](job-recovery.md) | Understand interruptions, saved progress and supported continuations. |

## Prepare and review data

| Guide | Use it to… |
| --- | --- |
| [Intake and frame selection](intake-selection.md) | Import recordings and pick frames to work on. |
| [COCO import](coco-import.md) | Bring in an existing annotated dataset. |
| [Class definitions](classes.md) | Define the objects a project needs to recognize. |
| [Annotation editor](annotation-editor.md) | Draw, correct and validate boxes. |
| [Detector preannotation](preannotation.md) | Generate box suggestions from an installed detector. |
| [Multimodal review](multimodal-review.md) | Configure local or hosted Qwen to review existing person/car boxes. |
| [Annotation batches](annotation-batches.md) | Queue local candidate reviews across several frames. |
| [Video review](video-review.md) | Review a video storyboard and record proposed events. |
| [Review queue](review-queue.md) | Prioritize frames using review status and saved-model disagreements. |
| [Dataset releases and export](dataset-export.md) | Freeze reviewed data and export it as COCO. |

## Train, compare and export models

| Guide | Use it to… |
| --- | --- |
| [Model comparison](model-comparison.md) | Inspect predictions on the same images, including tiled inference. |
| [Trainable models](trainable-models.md) | Choose an architecture and training depth. |
| [Training custom classes](custom-training.md) | Adapt a detector's class head to your dataset. |
| [Longer training and recovery](long-training.md) | Plan a run, read its progress and resume compatible checkpoints. |
| [YOLOX-Nano training](yolox-training.md) | Use the YOLOX-specific training recipe and class mapping. |
| [Evaluation](evaluation.md) | Measure a frozen split, inspect errors and compare checkpoints. |
| [Experiments](experiments.md) | Save model comparisons, notes and standalone reports. |
| [Native model exports](model-export.md) | Package Torchvision weights, preprocessing and a standalone runner. |
| [YOLOX ONNX export](yolox-onnx.md) | Export YOLOX and check predictions through OpenCV DNN. |
| [Iris to Argos](iris-to-argos.md) | Read a concrete training and integration case study. |

## Work with video and tracking

Tracking has separate inputs and measurements: a temporal sequence, a fixed
detector output, tracker predictions and human reference identities.

| Guide | Use it to… |
| --- | --- |
| [Temporal data](temporal-data.md) | Build ordered sequences with source timestamps. |
| [Temporal detections](temporal-detections.md) | Save detector outputs for repeatable tracker comparisons. |
| [Reference identities](temporal-identities.md) | Review which observations belong to the same object. |
| [Tracking setup and replay](tracking.md) | Install and run ByteTrack or BoT-SORT. |
| [Tracking studio](tracking-studio.md) | Compare saved replays in the browser. |
| [Tracking quality](tracking-quality.md) | Measure results against explicit human review coverage. |
| [Execution cost](tracking-cost.md) | Measure detector and tracker runtime costs. |
| [Tracking studies](tracking-studies.md) | Combine quality and cost evidence in a saved study. |
| [Selected-object continuity](selected-object.md) | Inspect how a policy follows one object through saved observations. |
| [Pipeline bundles](pipeline-bundles.md) | Package a detector, tracker and optional selection policy. |
| [Standalone pipeline runtime](pipeline-runtime.md) | Execute a bundle outside Iris. |
| [Pipeline qualification](pipeline-qualification.md) | Plan independent checks on new footage and target hardware. |

## Assisted annotation experiments

These guides describe distinct workflows. Qwen candidate review checks existing
boxes; DINO-X can propose new ones. Benchmark adapters compare annotation methods
against a separate reviewed reference. Availability in code and measured quality
are documented separately.

| Guide | Use it to… |
| --- | --- |
| [Annotation benchmark](benchmark.md) | Freeze a reference, define candidates and inspect trials. |
| [DINO-X preannotation](dinox-preannotation.md) | Prepare and inspect a hosted box-proposal request. |
| [DINO-X with Astra review](dinox-astra-review.md) | Review proposals through the combined provider workflow. |
| [OpenAI adapter](openai-preannotation-adapter.md) | Configure its benchmark proposal contract. |
| [SAM adapter](sam-preannotation-adapter.md) | Understand the isolated runtime and supported box contract. |
| [Combined adapter](combined-preannotation-adapter.md) | Chain proposal and review stages with bounded requests. |

## Understand and develop Iris

- [Architecture](architecture.md): system diagram, code map, data flow and storage.
- [Development](development.md): routine checks and optional tests with real models.
- [Recorded experiments](acceptance-results.md): executed runs, measurements and limits.
- [README media credits](media/README.md) and [example provenance](../examples/street-scenes/README.md).
- [Software licence](../LICENSE) and [third-party notices](../THIRD_PARTY_NOTICES.md).
