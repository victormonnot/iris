# Acceptance results

[Documentation](README.md)

This is the detailed record of experiments and software checks, with their dates,
settings and limits. Results describe the recorded data and hardware; later
features do not change what an earlier experiment established.

For a shorter account of training a detector and using it in another application,
start with [From Iris to Argos](iris-to-argos.md).

## R9: person preannotation comparison

On 2026-10-06, IRIS completed five configurations on the same **17 real 640 × 480
images, containing 16 human-reference people and one negative image**. The images
came from a single courtyard recording. They were reserved as evaluation scenes
for the annotation comparison, but had already been used for detector development
in R3–R5. This is a small development-validation result, **not an independent final
test or a general model ranking**.

The reference and candidate settings were frozen before these inference calls.
Matching used same-class, one-to-one boxes with IoU at least 0.5. Each configuration
produced all 17 outputs successfully; the resulting 85 outputs are retained in
Benchmark. Human reference boxes were not sent to either provider.

| Configuration | Correct detections | Extra boxes | Missed people | Mean matched-box IoU |
| --- | ---: | ---: | ---: | ---: |
| Astra alone | 16 | 0 | 0 | 0.902 |
| DINO-X, threshold 0.25 | 16 | 2 | 0 | 0.955 |
| DINO-X, threshold 0.50 | 16 | 0 | 0 | 0.955 |
| DINO-X 0.25 → Astra review | 16 | 2 | 0 | 0.955 |
| Local pretrained Faster R-CNN MobileNetV3-Large 320 FPN, CUDA, threshold 0.50 | 13 | 0 | 3 | 0.817 |

Mean IoU describes the overlap of matched boxes only. The local control's mean
covers 13 matches; each cloud configuration's mean covers 16. These are
single-operating-point results, not average precision. The local control used the
installed official pretrained checkpoint, not an IRIS fine-tuned checkpoint.

DINO-X used its fixed `person` prompt. The 0.50 threshold came from prior tuning,
not a search over these evaluation results. Both thresholds and the review variant
reuse the same 17 native DINO-X responses. They are alternative treatments of one
detector run, not independent repetitions. Astra reviewed every candidate from the
0.25 variant and accepted all 18, including both false positives. Retained box
geometry stayed unchanged, as required by the review contract.

For this person-annotation pilot, **DINO-X at 0.50 is the provisional first option**:
it retained all reference people without the two extra boxes or a second provider
stage. Astra alone also found all 16 people without extras. The review stage added
no measured quality benefit on this lot, so the evidence does not justify applying
it automatically to every image. This conclusion does not change any saved
configuration or application default. A higher detector threshold may miss weak
detections on harder images; a new set of distant people, analog noise, blur,
backlight and negative scenes is still needed.

### Requests and cost

The run made **51 unique provider submissions**: 17 to DINO-X, 17 to Astra alone
and 17 to Astra review. There were no automatic retries or top-ups.

| Work | Recorded cost basis | Amount for 17 images |
| --- | --- | ---: |
| Astra alone | Provider-reported tokens at the frozen USD rates | $0.18254 |
| Astra review of saved DINO-X candidates | Provider-reported tokens at the frozen USD rates | $0.18790 |
| Total OpenAI use across both stages | Sum of those usage calculations | **$0.37044** |
| DINO-X source run | Recorded per-call CNY estimate | **2.55 CNY** |

The $9 OpenAI planning allowance was an admission estimate, not the amount
reported by token usage. The amounts above are **not verified invoices or account
balance deductions**. Currencies remain separate, and local compute cost is
unmeasured. The DINO-X cost is counted once: threshold filtering and the three
recorded-evidence imports did not send another detector request. The review
workflow's attributable cost is the detector estimate plus $0.18790 for review;
Astra-alone cost is a separate comparison arm.

Recorded median request durations were 3.529 seconds for Astra alone, 2.290 seconds
for DINO-X and 3.178 seconds for the incremental Astra review. They include network
and provider processing; DINO-X includes submission and polling. The local CUDA
control recorded 31.104 milliseconds per image, including image decoding and
inference but excluding warm-up. These scopes differ and do not establish
deployment throughput. Import-job durations measure evidence processing, not
model inference, and review duration alone excludes the detector stage.

### Human review and interpretation

The planned correction-time comparison was **discontinued as uninterpretable**
after participant feedback. Four outputs were explicitly reviewed, each with one
box accepted unchanged and no additions, corrections or rejections. The reviewer
reported time spent finding controls and learning the interface. Only three of
those reviews belonged to the primary comparison, with neither DINO-only arm
covered. The different images and small number of observations cannot support a
fair comparison of correction effort.

The actual reviews and timing receipts remain saved. No duration was rewritten,
no guessed interface time was subtracted and no remaining output was automatically
marked reviewed. The remaining timed tasks are not required for this pilot.
**No annotation-time saving or method ranking is inferred from these timers.**
The feedback is useful evidence for improving the review interface.

Quality and cost results remain descriptive evidence for these 17 images. They
do not establish robustness to new scenes, tiny people or other cameras, nor the
stability of repeated provider runs. The source images had previously been
manually annotated. Neutral task labels offered limited masking; normal
application views remained accessible. SAM execution and the Astra → SAM → Astra
planning pipeline were not part of this adapted comparison and remain unmeasured.

### Retained evidence and software checks

An independent audit reconstructed boxes from native provider responses, reapplied
thresholds and review decisions, matched against the frozen reference, and
recalculated token costs. All five saved trial metrics matched. The audit also
checked the frozen protocol, image/request identities and preservation of earlier
annotations and evidence. Recorded imports establish internal consistency of the
supplied evidence, not independent authentication of provider execution. Native
responses, private media and participant records remain in the local workspace;
this public summary contains no source images or credentials.

The R9 workspace archive round trip passed with schema 19: **36 tables and 370
files** were verified after restoring a 2,708,815,631-byte archive into a separate
workspace. All five R9 trials and 85 outputs were preserved. The original
workspace rows and files remained unchanged. This verifies the tested snapshot;
it does not replace checking future backups. See [workspace backup and
restoration](workspace-backup.md).

R9 does not certify embedded devices, other GPUs or standalone deployment
performance. The distinct export measurements and their exact-parity limitations
are documented in [model exports](model-export.md); supported profiles are
described in [compute targets](compute-targets.md).

R10 below completed the second use case outside drones and addressed interface
friction through actual use. Difficult FPV quality validation remains a separate
follow-up using newly collected scenes; it does not require restarting the
abandoned timed-review schedule.

## R10: completed street-vehicle workflow, weak detector quality

On 2026-10-06, the second use case completed the real **cars and buses in street
scenes** workflow in a separate project. The participant validated all six photos:
20 source proposals were accepted, one was corrected and four boxes were added,
giving **23 cars and two buses**. The frozen release retains the exact saved human
revisions, source annotations, excluded categories, authors/licenses and image
identities. Four training photos contain 11 boxes; two validation photos contain
14. Dataset COCO export preserved the images, reviewed boxes and split assignments,
with custom export categories car=1 and bus=2 distinct from source COCO IDs 3 and 6.

SSDLite320 MobileNetV3-Large completed **40 light-scope CUDA training steps** on an
RTX 4060, using batch size one, float32, learning rate 0.001 and seed 42. Only the
four training images appeared in its history, for ten complete passes. Training
worker wall time was 8.128 seconds, including checkpoint loading and saving. The
checkpoint was then reloaded by a separate evaluation worker and compared with
its official pretrained parent on both validation photos, using full images,
confidence 0.5 and matching IoU 0.5. Settings were fixed before these outcomes.

| SSDLite configuration | Correct detections | Extra boxes | Missed objects | AP@[.50:.95] |
| --- | ---: | ---: | ---: | ---: |
| Official pretrained parent | 0 | 0 | 14 | 0.902% |
| Custom car/bus head after 40 steps | 0 | 0 | 14 | 0.519% |

Neither model retained a car/bus prediction at confidence 0.5. Precision is
undefined when no predictions remain, not 100%. AP uses the saved scores across
the confidence ranking, so low-score detections can yield a nonzero AP while the
chosen operating point retains nothing. This run **did not improve quality** and
does not provide a satisfactory street-vehicle detector. The before/after change
includes replacing the original COCO head with background/car/bus as well as
optimization; it does not isolate the effect of the 40 steps. No threshold was
retuned from these results and no new reference model was promoted.

The trained two-class checkpoint was exported and executed outside IRIS, using a
copied runner in a separate dependency-only CUDA environment, isolated Python and
disabled networking on the same physical host. Both validation images were run
three times: **all six samples passed strict exact parity** with the saved CUDA
reference. The comparison included all 100 native detections per image, including
low-score boxes; it was not a comparison of two empty thresholded lists. The real
measurement was imported into IRIS and attached to a saved experiment report,
which was also exported as standalone HTML. There were no cloud requests or new
API charges for this cycle.

### Reference and generalization limits

These photos come from the original COCO validation set, regrouped into a small
derived split solely to test the application workflow. This is not an official
COCO evaluation or evidence of unseen-image performance for pretrained models.
There are only two validation images and no validated negatives in this release.
The participant's added light van is treated as a car, whereas the original COCO
annotation calls it a truck: compatibility with the original category convention
is approximate for this instance. Two tiny, overlapping manual boxes in one
validation photo cannot reliably be resolved as distinct cars or a duplicate
from the available pixels. These limitations were recorded before training;
the original human decisions were preserved rather than changed after inference.
The scores describe this frozen reference, not a certified annotation standard.

The earlier R9 workspace backup/restoration check remains evidence for its original
snapshot; it was not repeated or relabeled as a backup of the new R10 project.
New difficult FPV footage remains a separate later quality evaluation.

Two interface fixes support this acceptance: an explicit shortcut from the save
controls to pending proposals, including proposals hidden by a filter, and a
guarded import-to-annotation handoff that restores keyboard focus to the workspace.
Neither action accepts proposals or validates images automatically. Disposable
browser checks covered filters, undo, loading, guarded navigation, and desktop and
mobile layouts in both themes. The participant subsequently used the actual
annotation workflow and completed all six reviews.

### Compatibility established so far

This table records the evidence available through R10 on 2026-10-06. The later
YOLOX and tracking measurements are recorded in the following sections.

| Capability | Real acceptance evidence | Limit |
| --- | --- | --- |
| CPU training | Both Torchvision detector architectures, 40 light-scope steps, checkpoint reload and evaluation | Longer runs and partial/full CPU training remain unmeasured |
| NVIDIA training | RTX 4060, both Torchvision architectures, light/partial/full scopes, 40 steps | Other GPUs untested; successful execution does not imply better quality |
| Interrupted training | Light-scope continuation after cancellation and forced worker termination on CPU/CUDA | Same runtime/device only; power loss and deeper-scope recovery untested |
| Standalone inference | Both architectures, all four CPU/CUDA training-to-target paths; R10 also exercised a custom car/bus SSDLite head on CUDA | Same-device exact parity passed; the four original cross-device comparisons failed. Separate target-reference controls passed without replacing those failures |
| DINO-X API | Integrated annotation worker and saved native Benchmark evidence | Small person-only quality pilot so far |
| Astra API | Eight tuning images and seventeen development-validation images | No independent final test or measured human time saving |
| DINO-X → Astra review | Real requests and imported Benchmark outputs | No incremental quality benefit in R9; no live combined Annotation option |
| SAM / Astra → SAM → Astra | Adapter and runtime preparation | Real model execution remains unmeasured |
| Backup/restoration | Separate restoration of 36 tables and 370 files | Verified snapshot, not a guarantee for every future backup |
| Jetson / other embedded devices | Runtime profiles documented | No physical-device acceptance; no TensorRT export. Later YOLOX ONNX workstation CPU results are recorded below |

Detailed conditions and results remain in [custom training](custom-training.md),
[training continuation](long-training.md), [model exports](model-export.md),
[compute targets](compute-targets.md), and the provider-specific documentation.

## YOLOX-Nano: custom detector accepted by an external application

On 2026-10-07, a person-detection cycle trained YOLOX-Nano in IRIS, exported its
raw ONNX graph and replayed it through ARGOS's actual detector and tracker. This
was offline work on the workstation; no flight service or default model changed.

The frozen human reference contains **57 images and 52 people**. Forty images
from bedroom and morning-park recordings supply training, including four
negatives; 17 courtyard images supply validation, including one negative.
Recordings are separated, but this small validation set was already used during
R3–R9. It is not a new independent test of distant-person or analog-video
robustness. No cloud annotation or paid API call was needed for this cycle.

### Training and detection quality

Two planned CUDA configurations used batch size one, learning rate 0.0001 and
seed 42 on an RTX 4060. The head-only configuration completed 400 steps. The first
partial-backbone attempt stopped after 15 updates because an unstable update on
a negative image led to nonfinite gradients in the next step. No checkpoint was
published from that attempt. The failure and original recipe remain recorded.

A training-only reproduction identified the numerical instability. A separately
versioned v2 recipe adds global L2 gradient clipping at 10 before SGD, retaining
the original learning rate, seed and data. The new partial-backbone attempt
completed 800 finite updates, with clipping applied on 728. Validation images
were not used to diagnose the failure or train either model. The v2 training
worker took 97.33 seconds; both published checkpoints reloaded and predicted on
CPU successfully.

Full-image IRIS evaluation used confidence 0.35 and matching IoU 0.5:

| YOLOX-Nano configuration | Correct | Extra | Missed | AP50 | AP@[.50:.95] |
| --- | ---: | ---: | ---: | ---: | ---: |
| Official parent | 15 | 0 | 1 | 93.07% | 57.05% |
| Head only, 400 steps | 15 | 0 | 1 | 95.59% | 56.80% |
| Partial backbone v2, 800 steps | 16 | 0 | 0 | 100% | 67.76% |

ARGOS independently confirmed **15/0/1 → 16/0/0** on the same 17 validation images
using its normal person-only filtering: confidence 0.35, NMS 0.45 and at most
16 detections. Its training-image result changed from 28/1/8 to 36/0/0; this is
training fit, not additional generalization evidence. The model remains a
candidate, with no automatic selection or deployment.

### Export and temporal replay

The final ONNX conversion passed its fixed raw-output numerical check on three
reference images (`rtol=0.001`, `atol=0.001`), with maximum absolute difference
0.0004352. The standalone OpenCV CPU runner then produced nine measured samples
(three images, three repetitions). Exact saved-prediction parity **failed** and
that failure was imported unchanged. Detection counts and order agreed; the
largest box-coordinate difference was 0.000184 pixels and largest score difference
was 0.000000179. Numerical conversion agreement and strict exact parity remain
separate claims. See [the export contract](yolox-onnx.md).

The temporal comparison used 300 original camera frames in three 100-frame clips,
with original receipt timestamps, identical inputs and per-clip tracker resets.
Two courtyard clips, totaling 200 frames, belong to the validation recording;
the 100-frame morning-park clip belongs to training and is diagnostic only.
The official and custom graph ran through the same ARGOS pipeline,
with alternating model order and two warmups each. Standalone viewers embed all
original frames and saved boxes, without rerunning inference.

On the 200 validation frames, frames with a retained detection increased from
**137 to 170**, while locally created track IDs increased from **20 to 23**.
Only two of those frames have human box references, and none have identity ground
truth. More frames with detections therefore do not establish fewer false alarms,
fewer true target losses or better identity continuity. Historical loss/ambiguity
markers locate useful passages; they are not verified target-loss labels.

Median inference time in this workstation replay was 7.19 ms for the official
model and 6.74 ms for the candidate; median full processing was 10.46 and 10.18 ms.
These are OpenCV CPU measurements on the workstation with four threads, not live
flight throughput, end-to-end latency or portable-computer measurements. The
portable computer was unavailable; its timing and a fresh difficult FPV set
remain follow-up work.

Independent checks reconstructed quality counts, verified original-image hashes
and preserved all 795 earlier workspace rows. Code checks include the full IRIS
suite (3,671 passed, eight skipped), subsequent targeted export/report checks and
real desktop/mobile browser inspection. The application bridge and viewer have
separate ARGOS tests. These checks establish the tested software cycle; new scenes
and temporal identity annotations are needed to qualify tracking improvements.

## Saved-frame tracking cost on CPU and CUDA

On 2026-10-08, T7 ran eight fresh detector/tracker measurement jobs on a workstation
with an Intel Core i5-12400F and NVIDIA RTX 4060. The source was an already known
eight-frame analog-camera passage at 640 × 480 pixels. The frozen detector was
official SSDLite320 MobileNet V3, full-image inference, batch one, native storage
score floor 0.001 and person-only tracker classes. Both tracker profiles retained
their existing thresholds; BoT-SORT used sparse optical-flow camera compensation
without learned re-identification.

Each job loaded a fresh detector and tracker, warmed up separately, and reset
tracker state before each repetition. All-frame runs used three repetitions,
giving 24 measured samples per configuration. Both CPU and CUDA inference used
the Torch 2.10.0+cu128 / Torchvision 0.25.0+cu128 environment, with four Torch CPU
threads. Native tracking remained on CPU for both detector devices.

| Detector device / tracker | Detector call, median | Tracker call, median | Saved-frame pipeline, median | Pipeline p95 |
| --- | ---: | ---: | ---: | ---: |
| CPU / ByteTrack | 37.19 ms | 1.05 ms | 48.30 ms | 55.31 ms |
| CPU / BoT-SORT + optical flow | 37.65 ms | 7.95 ms | 57.72 ms | 64.20 ms |
| CUDA / ByteTrack | 42.19 ms | 1.19 ms | 55.11 ms | 75.46 ms |
| CUDA / BoT-SORT + optical flow | 37.06 ms | 7.40 ms | 54.86 ms | 67.86 ms |

The complete measured pipeline includes saved-PNG verification and decoding,
fresh detector execution, filtering and tracker update. Stage medians need not
sum to the pipeline median. CUDA did not provide a clear speed advantage for
this small model and batch-one experiment. This is not a ranking of devices or
a measurement of ARGOS's separate YOLOX deployment pipeline.

Sampled process RSS peaks were about 748–752 MiB for CPU-detector jobs and
1,412–1,422 MiB for CUDA-detector jobs. PyTorch CUDA allocator peaks were
45.29 MiB allocated and 102 MiB reserved; these are not whole-board VRAM usage.
Process lifetime high-water and boundary-sampled RSS remain distinct counters.

At an assumed 30 FPS, the virtual latest-frame policy processed 18 of 24 available
frame opportunities on CPU and 16 of 24 on CUDA across three BoT-SORT repetitions.
At 120 FPS, CPU BoT-SORT processed six of 16 and dropped ten across two repetitions.
A separate sparse-source run preserved four missing source frames per repetition
and distinguished them from two simulated drops across both repetitions.
The simulation used measured per-frame service times; it captured no live camera.

The portable CLI also processed eight real frames while leaving its source
database unchanged. Import preserved that report as declared execution and
rejected altered summaries, scheduling, profiles and memory values. These trials
validate the [measurement workflow](tracking-cost.md) on one desktop. They do not
establish tracking accuracy, laptop or embedded performance, live flight latency,
or generalization to new FPV footage.
