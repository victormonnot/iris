# Qualifying a detector and tracking pipeline

[Documentation](README.md)

A [portable runtime](pipeline-runtime.md) can reproduce IRIS outputs without
establishing that those outputs are good enough for a new application. Evaluate
quality and deployment cost on independent footage before adopting a pipeline.
Keep execution parity, annotation coverage, tracking quality and target speed as
separate evidence.

## Reserve footage before running candidates

Choose several short recordings representative of the intended input: object
size, motion, occlusion, lighting, noise and camera movement. Include intervals
where objects leave the field of view. Empty intervals can reveal false positives
and incorrect recovery; they are not automatically annotation omissions.

Keep complete takes and related scenes in the same split. New frame indices,
other crops, exports, screen recordings and augmentations of an existing take do
not make an independent test. A previously unused take from a scene used during
tuning may still be unsuitable for the final test. Preserve source IDs, original
file hashes, parent recordings, acquisition conditions and split reservations.

A public dataset can supplement a missing domain, provided its annotation rules
match the question being measured. Digital road footage does not establish
robustness to analog FPV interference. A dataset's published `training` split can
be held out from a particular IRIS experiment, but that does not establish that
third-party pretrained models have never encountered it.

When importing a video assembled from published images, retain the image hashes
and the original-to-local frame-index mapping. Verify decoded pixels after
conversion and after import. A declared playback FPS supplies a nominal clock,
not measured camera exposure times. Preserve all frames in the chosen interval;
label sparse samples and missing intervals explicitly.

## Freeze the question and settings

Before inspecting candidate outputs, record:

- The source interval, selection rule, split and annotation revision.
- Detector architecture, checkpoint hash, class mapping, preprocessing and score floor.
- Each complete tracker profile, camera compensation setting and selection policy.
- Evaluation overlap threshold, eligible classes and treatment of occlusion,
  truncation, uncertain identities and ignored regions.
- Initial target identity and selection rule, repeat count and intended devices.

Include a baseline. Keep the same cached detections for tracker comparisons.
If testing a new detector, evaluate its effect separately from changed tracker
settings. Do not choose thresholds or replace difficult intervals after seeing
the held-out result. Further tuning makes that result development evidence and
requires a fresh independent test for the final decision.

## Preserve the reference's actual provenance

For local references, use the [temporal identity editor](temporal-identities.md)
to review identities and visibility, including disappearance and return. Mark
unknown passages explicitly. A tracker ID is not a ground-truth identity.

IRIS's [native quality report](tracking-quality.md) evaluates human-reviewed,
complete reference frames. Imported public annotations are not a local human
review. An automated import can preserve their boxes and identities for display
with `assistant_reviewed` status and explicit source/author notes; it must not
attribute a review to the local user.

The current reference schema has no typed external-ground-truth qualification
status. To score official external labels automatically, use a separate report
that binds the original annotation bytes, published conventions, conversion code,
frozen protocol and exact tracking outputs. Do not persist it as a native quality
report with fabricated human-review flags. If a person later reviews the imported
reference in Studio, save that review as a new revision.

Dataset ignore regions and class conventions matter. Dropping ignored regions
and treating the remaining annotations as exhaustive can inflate false positives.
Implement and document the declared exclusion rule, or select an eligible interval
using an annotation-only rule fixed before inference. State that restriction in
the results; such a subset is not the official dataset benchmark.

## Measure behavior and deployment separately

Run [fixed tracking comparisons](tracking-studio.md) on the frozen cache, then
repeat fresh tracker runs to check output stability. Score confirmed measured
observations under the declared reference protocol. Keep predictions and
unconfirmed detections visible but separate from measured matches.

Report matched and missed objects, extra observations, identity switches,
fragmentation and identity-score coverage. A zero switch count on a short clip
with few crossings is limited evidence. For [selected-object behavior](selected-object.md),
record correct observations, wrong-object substitutions, lost/ambiguous states
and expiry. If the predetermined target cannot be selected at the initial frame,
report initialization as unavailable rather than moving the anchor to an easier
moment without disclosing the change.

Use [fresh pipeline cost measurements](tracking-cost.md) for CPU/CUDA latency and
memory. Cached detector timings are not complete pipeline latency. Offline
throughput, simulated dropped frames and actual camera-to-application delay are
different measurements. A desktop result is not a laptop or embedded-board result.

Finally, export the exact candidate and compare its standalone outputs with an
explicit IRIS reference on the same frames and device family. Preserve numerical
mismatches; a tolerance change is a protocol change. Keep the source, run and
parity reports alongside the unchanged exported manifest.

## Record the decision and remaining coverage

Summarize each candidate against the baseline, including failures and unavailable
measurements. Keep the baseline when the evidence does not establish a useful
improvement. State which footage, devices and behaviors were actually checked,
and which remain untested. A successful software workflow, exact exported outputs
or a clean eight-second clip does not by itself qualify a full deployment.
