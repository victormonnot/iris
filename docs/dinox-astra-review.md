# DINO-X proposals reviewed by Astra

`dinox_review_provider.py` freezes `iris-dinox-astra-review-v1`, a separate
two-stage experiment: reuse
completed DINO-X detections, then ask GPT-6 Astra to review the existing boxes.
This tests whether visual review can remove false detections while retaining
the detector's geometry. It does not run a planner, SAM, a new DINO-X request or
an iterative correction loop.

The provider adapter runs independently. Completed native evidence can now enter
Benchmark through the explicit [recorded-evidence import](benchmark.md#import-recorded-provider-evidence)
for scoring and timed human correction. Import makes no provider request; it is
not a live combined option in Annotation. The initial acceptance experiment used
saved provider evidence outside the workspace database. The existing
[Astra → SAM → Astra adapter](combined-preannotation-adapter.md) keeps its own
unchanged protocol and has not been validated by this experiment.

## Frozen review contract

Before any request, the caller verifies the source image identity, frozen class
definitions, detector settings and completed native response. It re-normalizes
the DINO-X output and binds the saved candidates to the review request.

Astra receives a metadata-free image, class IDs/names/definitions and a list of
candidate IDs, class IDs and boxes normalized over the whole image. Detector
confidence scores, human reference annotations, previous Astra predictions,
filenames and evaluation results are withheld. Each request is independent and
uses Responses with strict structured output, no tools and `store: false`.

Every candidate must receive exactly one `accept`, `reject` or `relabel`
decision, a short reason and an uncertainty flag. Accept/reject preserve the
class; relabel requires another frozen class. A single-class experiment therefore
has no valid relabel target. Unknown, repeated or omitted candidates, invented
classes and returned coordinates are errors. An empty detector result still
receives one review request whose only valid result is an empty decision list.

Retained proposals preserve the detector's IDs and exact pixel coordinates.
Their final score is null; native scores, source geometry and all decisions,
including rejections, remain in the saved evidence. Review cannot recover an
object the detector missed or improve a box's localization. An empty result
does not establish that the image is a human-validated negative.

## Requests, accounting and reuse

All candidates are known before dispatch, so the preview estimates their actual
bounded prompt rather than reserving unknown future candidates. The caller owns
image selection, authorization, budgeting and a durable dispatch ledger.
Mandatory callbacks record dispatch intent before the single HTTPS POST and
save raw response evidence before output normalization. There is no automatic
retry, fallback model or repair request. Preparation and validation are offline.

The acceptance profile uses low reasoning effort, a 2048-token output limit and
a maximum image edge of 1536 pixels. Planned allowances are conservative
admission estimates, not guaranteed billing caps. Reported usage is priced using
the frozen [OpenAI adapter rates](openai-preannotation-adapter.md); this calculation
is not an invoice. The [official Astra model documentation](https://developers.openai.com/api/docs/models/gpt-6-astra)
and [structured output contract](https://developers.openai.com/api/docs/guides/structured-outputs)
were checked on 2026-10-06.

Reusing completed DINO-X outputs incurs no new DINO-X charge. Its historical
estimated cost remains separate from new OpenAI usage. Review latency measures
the incremental request; adding a previous detector measurement produces a
reconstructed pipeline estimate, not a fresh end-to-end timing.

The reviewer reads the OpenAI key from the process environment, using the
existing adapter's credential rules. The private acceptance launcher can load a
locally saved key into its own environment; that is not a new public credential
file lookup or a reason to put credentials in a workspace.

## Interpretation

The initial experiment reuses eight tuning images already inspected during
the separate DINO-X and Astra trials. It is not an independent generalization
test. The seventeen reserved evaluation images remain separate until the
later comparison protocol is frozen.

On this tuning set, the DINO-X false positive also has a lower score than every
true positive. A detector threshold chosen after inspecting the results can
therefore separate them without a reviewer. A successful combined result alone
cannot establish that paying for Astra is preferable to a validated detector
threshold. No human correction-time saving is claimed without a measured human
review session.

## Real acceptance result

On 2026-10-06, eight 640 × 480 tuning images completed eight unique Astra review
requests with the profile above. The source was the previously completed DINO-X
run at threshold 0.25 and fixed prompt `person`. There were no new detector calls,
retries, planner requests or SAM execution.

Astra accepted seven candidates and rejected one. At same-class, one-to-one IoU
matching with threshold 0.5, the combined output had **seven true positives,
zero false positives and zero misses**. The rejected box enclosed a household
object. All seven accepted IDs and pixel boxes exactly matched the source
detector output; matched mean IoU therefore remained **0.963066**. The image with
no source candidates returned an empty decision list. The previous DINO-X-only
result had seven true positives, one false positive and zero misses; Astra alone
had seven true positives, no false positives or misses, and mean IoU 0.924669.

The reviewer reported **6,205 input and 557 output tokens**, including 94 reasoning
tokens already counted in output. The frozen usage calculation is **$0.08990**
for all eight reviews, versus a **$1.4802875** planning allowance. The prior
detector run's **1.20 CNY estimated cost** was not incurred again. Median new
review request wall time was **7.427 seconds**, including network and provider
processing; it excludes the cached detector stage.

An independent audit reconstructed the native detections, outgoing image/request
hashes, raw review decisions, geometry, matching and token cost. The application
database, human annotation revisions and protected source files stayed unchanged.
No reserved evaluation image or human correction record was added or modified.
The comparison is available as a private, offline visual report; this initial
experiment did not create an integrated Benchmark trial or establish a general
quality or annotation-time improvement. Later recorded-evidence imports retain
their own local import history and do not rewrite the earlier provider execution.
