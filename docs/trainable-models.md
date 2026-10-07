# Trainable detector choices

**YOLOX-Nano** is a third supported trainable family, with CPU/CUDA training,
the same three scope choices and durable recovery. Its portable export targets
OpenCV CPU through ONNX. See [its pinned training recipe](yolox-training.md) and
[export contract](yolox-onnx.md). The comparison below describes the original
two Torchvision families; export capabilities depend on the architecture.

IRIS can fine-tune **Faster R-CNN MobileNetV3-Large 320 FPN** and
**SSDLite320 MobileNetV3-Large**, including their compatible trained descendants.
Both accept frozen custom classes, CPU or NVIDIA CUDA training, durable recovery,
held-out evaluation and standalone native PyTorch CPU/CUDA export. Annotation
providers such as SAM and multimodal APIs remain separate from these deployable
detectors.

## Why these two candidates

SSDLite extends an existing inference adapter and uses the same pinned
PyTorch 2.10.0 / Torchvision 0.25.0 runtime. Its official checkpoint is about
13.4 MiB, compared with about 74.2 MiB for the existing Faster R-CNN checkpoint.
This makes it a useful smaller candidate to evaluate for constrained targets.
Weight-file size does not measure peak RAM, GPU memory, latency or trained quality.
IRIS does not select a winner automatically.

SSDLite resizes the image to a fixed 320 × 320 rectangle. Small objects can lose
detail, and aspect ratios are resized before boxes are restored to original-image
coordinates. Faster R-CNN uses its existing aspect-preserving resize and region
proposal pipeline. Compare both on exactly the same frozen validation images,
with the same output classes and operating thresholds. Confidence scores are not
calibrated across architectures. Keep the test split for the final independent
audit, and measure exported latency and memory on the intended target.

The [official SSDLite model description](https://docs.pytorch.org/vision/0.25/models/generated/torchvision.models.detection.ssdlite320_mobilenet_v3_large.html)
and [pinned implementation](https://github.com/pytorch/vision/blob/v0.25.0/torchvision/models/detection/ssdlite.py)
define the architecture and checkpoint. The
[Torchvision code license](https://github.com/pytorch/vision/blob/v0.25.0/LICENSE)
is BSD 3-Clause. Pretrained weights have separate usage considerations:
[Torchvision's weights notice](https://docs.pytorch.org/vision/0.25/models.html#general-information-on-pre-trained-weights)
directs users to the source dataset's terms. Fine-tuning and export do not erase
those conditions or the obligations associated with your own training data.

## Complete workflow

1. Freeze a dataset with human-reviewed boxes, useful negative images and scene
   groups separated into training, validation and test splits.
2. In **Dataset & training**, select a local parent from either architecture.
   Choose a training depth, CPU or NVIDIA GPU, steps, learning rate and seed.
   The preview lists the actual modules and negative-image policy for that model.
3. Run training, or explicitly continue a stopped attempt from its saved optimizer
   state. A completed checkpoint appears in the model catalog with its own class
   definitions and training provenance.
4. Train the other architecture using that same dataset. For classes outside
   COCO, compare the two trained checkpoints; an official COCO checkpoint cannot
   stand in for a missing custom category.
5. In **Quality evaluation**, select both checkpoints on the same validation
   split. Inspect metrics, errors and saved images. Training loss values from
   the two architectures are different objectives and cannot rank model quality.
6. Export each trained model with a completed full-image reference evaluation.
   Select the destination CPU/CUDA family independently of its training device.
   Run the bundle's runtime check and measurement procedure on the destination,
   then import its evidence. Parity failures remain failures.

Official weights are provisioned explicitly using the existing model setup
commands. Training, preview and export never download them. No additional ML
package is required for SSDLite. See [compute setup](compute-targets.md) and
[model export](model-export.md) for target requirements, including embedded
systems. These two profiles do not convert to ONNX or TensorRT. The separate
[YOLOX ONNX profile](yolox-onnx.md) targets OpenCV CPU; no profile establishes
blanket embedded-board support.

## Training contracts

| Depth | Faster R-CNN | SSDLite |
| --- | --- | --- |
| Light | ROI classifier and box predictor | Classification and regression heads at all six scales |
| Partial | Last MobileNet stage, FPN, RPN and ROI heads | Final MobileNet feature block, extra scales and detection heads |
| Full | All learnable parameters | All learnable parameters |

These names describe different module layouts. Configurations freeze the selected
architecture, scope and adapter policy; continuation cannot switch architectures.
Starting a new run from a completed model keeps its architecture and requires
exactly matching frozen class definitions. Legacy Faster R-CNN configurations
without an architecture field retain their original meaning.

SSDLite's classifier has one set of background/object slots per anchor at each
scale. Only explicitly mapped COCO rows are copied into the new class slots;
unmapped classes use seeded initialization. The existing depthwise features,
normalization state and class-independent box regression are retained. A trained
parent keeps its compatible head unchanged before optimization.

SSDLite uses ordinary BatchNorm layers. Their running means, variances and counters
remain frozen in every training depth, so batch-one 1 × 1 feature maps are valid.
Affine parameters can still be updated within the chosen scope. Final publication
checks all frozen parameters and buffers; recovery preserves these baselines and
module modes.

The pinned native SSD loss mines three background anchors per positive match.
For an empty training image it otherwise selects none. IRIS's versioned
`iris-ssdlite-training-v1` adapter uses the three hardest background anchors
(or all available when fewer than three) for that empty image, with summed
cross-entropy and a normalizer of one. Its box-regression loss stays zero and
connected to the computation graph. Positive images keep the native loss.
This is an explicit training policy, not a claim of improved quality; assess
false positives and recall on held-out data. Inference and exported runners
use the native architecture without this training-only loss adapter.

## Validation boundary

Software tests use synthetic datasets, fake detector engines, tiny CPU tensor
fixtures and mocked CUDA interfaces. They cover class/anchor initialization,
negative-image gradients, frozen normalization, training scopes, durable recovery,
architecture identity, evaluation, export profiles and archive preservation.
Real 40-step acceptance trials covered both architectures with light training on
CPU and all three depths on an RTX 4060 using PyTorch 2.10.0 / Torchvision 0.25.0
`cu128`. The trials used a small human-reviewed person dataset with negative images
and separate training and validation source contexts. Completed inference
checkpoints were reloaded and evaluated in separate workers. Light-scope runs
also continued after cancellation and forced worker termination on CPU and CUDA,
preserving the original device and runtime.

Copied standalone bundles of the light-scope models completed all four CPU/CUDA
training-to-inference paths on the same host, in separate environments without
IRIS and with networking disabled. Three repetitions of eight saved validation
images passed exact parity when the reference and target device matched.
Cross-device references failed exact parity on small numerical differences,
without changing detection counts or thresholded quality counts on this subset.
Four separate controls using new references evaluated on their target device
passed exact parity; original failures remain recorded.
See [export validation evidence and timing limits](model-export.md#what-is-verified).

This small pilot does not establish general quality gains. Full training is not
necessarily better than lighter scopes; using the same learning rate across
architectures and depths can cause regressions. Longer runs, partial/full training
on CPU, partial/full-scope recovery, server restart or power-loss recovery and
other hardware remain to be tested. Export parity, latency and memory need fresh
measurement for a different model or deployment environment.
