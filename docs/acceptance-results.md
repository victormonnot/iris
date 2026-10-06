# Acceptance results

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

A real workspace archive round trip passed with schema 19: **36 tables and 370
files** were verified after restoring a 2,708,815,631-byte archive into a separate
workspace. All five R9 trials and 85 outputs were preserved. The original
workspace rows and files remained unchanged. This verifies the tested snapshot;
it does not replace checking future backups. See [workspace backup and
restoration](workspace-backup.md).

R9 does not certify embedded devices, other GPUs or standalone deployment
performance. The distinct export measurements and their exact-parity limitations
are documented in [model exports](model-export.md); supported profiles are
described in [compute targets](compute-targets.md).

The next acceptance step is an end-to-end second use case outside drones, with
remaining interface friction addressed through actual use. Difficult FPV quality
validation remains a separate follow-up using newly collected scenes; it does not
require restarting the abandoned timed-review schedule.

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

| Capability | Real acceptance evidence | Limit |
| --- | --- | --- |
| CPU training | Both detector architectures, 40 light-scope steps, checkpoint reload and evaluation | Longer runs and partial/full CPU training remain unmeasured |
| NVIDIA training | RTX 4060, both architectures, light/partial/full scopes, 40 steps | Other GPUs untested; successful execution does not imply better quality |
| Interrupted training | Light-scope continuation after cancellation and forced worker termination on CPU/CUDA | Same runtime/device only; power loss and deeper-scope recovery untested |
| Standalone inference | Both architectures, all four CPU/CUDA training-to-target paths; R10 also exercised a custom car/bus SSDLite head on CUDA | Same-device exact parity passed; the four original cross-device comparisons failed. Separate target-reference controls passed without replacing those failures |
| DINO-X API | Integrated annotation worker and saved native Benchmark evidence | Small person-only quality pilot so far |
| Astra API | Eight tuning images and seventeen development-validation images | No independent final test or measured human time saving |
| DINO-X → Astra review | Real requests and imported Benchmark outputs | No incremental quality benefit in R9; no live combined Annotation option |
| SAM / Astra → SAM → Astra | Adapter and runtime preparation | Real model execution remains unmeasured |
| Backup/restoration | Separate restoration of 36 tables and 370 files | Verified snapshot, not a guarantee for every future backup |
| Jetson / other embedded devices | Runtime profiles documented | No physical-device acceptance; no ONNX/TensorRT export |

Detailed conditions and results remain in [custom training](custom-training.md),
[training continuation](long-training.md), [model exports](model-export.md),
[compute targets](compute-targets.md), and the provider-specific documentation.
