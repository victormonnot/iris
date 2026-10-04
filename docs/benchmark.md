# Preannotation benchmark

**Benchmark** measures candidate proposals against an independent human reference
and records the work required to correct them. It is separate from **Quality
evaluation** of trained detectors and **Experiments** reports. The reference,
candidate outputs and correction revisions have separate records; corrections
never update frame annotations or the benchmark reference.

## Freeze the reference and scene roles

1. In **Data intake**, select images from independently reviewed scenes. In
   **Annotation**, save explicit human validation, including empty negative images.
2. Open **Benchmark → Create an independent reference**. Choose the saved class
   version, enter a name and reviewer, and describe the independent review process.
   Confirm the independence declaration explicitly.
3. Assign different scene groups to **Tuning** and **Evaluation**. Expand a scene
   to choose its images. Both roles require 1–25 images. An excluded scene supplies
   no images. This selection does not change Data intake selection.
4. **Preview reference** checks current validated revisions, source pixels,
   classes, provenance and role reservations. Inspect counts and warnings, then
   choose **Freeze independent reference**.

Only validated frames without pending proposals are eligible. Reference boxes may
be manual or imported and human-reviewed; detector-derived and multimodal-derived
reference boxes are excluded. A human declaration records responsibility and the
review process. It cannot prove freedom from earlier model influence or pretraining.

Scene roles are independent from the names of dataset splits, but respect existing
reservations: **train → tuning**, **val/test → evaluation**. The opposite role is
rejected. Earlier benchmarks also reserve roles. A scene, identical pixels or one
original video cannot cross the two roles. Distinct scene names and source hashes
do not prove that visually related scenes are independent; review the real data
collection process as well.

Freezing copies the source images and saves reference boxes, original annotation
revision IDs, class definitions, scene roles, hashes and the review declaration.
Later annotation or taxonomy edits do not change the reference. Use a new benchmark
for a different reference or protocol. **Inspect frozen reference and protocol**
reopens the saved evidence without running a candidate.

## Tune candidates, then lock for evaluation

The three approaches are displayed separately:

- **A · Multimodal**: OpenAI `gpt-6-astra`, with an explicit external request preview
  and budget approval for each trial.
- **B · Segmentation**: local SAM 3 with one frozen text prompt per class. This
  adapter uses native box output; it does not calculate or save masks.
- **C · Combined**: Astra generates class prompts, local SAM 3 produces native
  boxes, then Astra accepts, rejects or relabels existing candidates. Each image
  uses at most two external calls and one local SAM pass, with explicit preview
  and budget approval.

The **local detector control** is executable using an already available checkpoint.
It exercises the benchmark workflow and does not substitute for evidence about
A, B or C. The local control does not download a model or call an external provider.

Prepare up to eight candidate configurations while the benchmark is in tuning.
Choose a local detector, proposal score threshold, CPU or CUDA, and full-image or
tiled inference. Preview class coverage, frozen checkpoint identity, resource work
and warnings before saving the configuration. Trained models require the exact
saved definitions. Official detectors use explicit COCO mappings; uncovered
classes remain visible in the protocol and the reference-based measurement.

Choose a configuration and **Tuning scenes**, preview the trial, then select
**Run checked trial**. Changing settings requires a separate frozen configuration;
an earlier configuration and its outputs are never rewritten. Inspect tuning
results before choosing **Lock configurations for evaluation**. This explicit lock
permanently ends new configurations and tuning trials for this benchmark. It does
not run anything automatically. The UI warns about configurations without a
complete tuning measurement; locking remains an explicit user decision.

After locking, preview and launch **Evaluation scenes** trials. Repeated evaluation
is recorded and warned about; it is not a fresh held-out dataset. Candidate adapters
receive only image pixels, class definitions and frozen settings. Human reference
boxes are not supplied to inference.

## Prepare a local SAM configuration

Choose **B · SAM 3 · local** in the configuration form. IRIS displays setup status
for the isolated runtime, official model weights and CUDA device. **Local setup
requirements** opens the local status and points to the
[SAM adapter guide](sam-preannotation-adapter.md). The interface does not install,
download, load or test a model automatically. Availability is not evidence of a
successful model run. This image adapter uses SAM 3; SAM 3.1's video tracking changes
are outside this benchmark's image protocol.

Configuration preparation and freezing work before local setup is complete: the
published code revision, weight identity and settings are saved independently of
their availability on this computer. Execution requires the configured isolated
Python runtime (`IRIS_SAM_PYTHON`), matching official weights at
`models/sam3/sam3.pt`, and a supported CUDA device. There is no automatic CPU fallback.

The form displays every class ID, name and definition from the **frozen benchmark
reference**, including custom classes. Each text prompt starts with that class's
name. Review or replace it with a descriptive phrase, between 1 and 120 characters.
The displayed definition helps the person preparing the prompt; it does not
automatically replace the entered text. Every class requires a prompt, and the same
prompt applies to every image in this configuration. Class names or prompts do not
guarantee that SAM will recognize the intended objects. Changing the reference
resets draft prompts; routine status refreshes preserve edits for the same reference.

Choose the native SAM score threshold (0–1, default 0.5) and CUDA device. Boxes must
score strictly above the threshold, matching the adapter's filtering rule. Native SAM
scores are not calibrated probabilities and are not interchangeable with detector
scores. **Preview configuration** shows image encodings and class-prompt evaluations
for both roles, plus availability and warnings. **Save frozen configuration** saves
the prompts, class snapshot, threshold, device, model/code identities and protocol;
it does not run SAM. Inspect a saved configuration to reopen its exact class prompts.

Select that frozen configuration and **Preview trial**. The local work plan includes
the runtime identity and requested work. If setup is missing or incompatible, the
reason stays visible and **Run checked local SAM trial** is disabled. Complete setup
separately and prepare a fresh preview before launching. SAM trials require no API
key, external-image consent or provider budget. They use the same explicit tuning
and evaluation lock as the other approaches. Prompt or threshold changes require a
new configuration before locking; evaluation cannot silently change either.

SAM measurements cover its native boxes, not mask quality. Model loading time is
displayed separately from image processing. There is no warm-up pass: measured
image processing includes the first pass. Invalid output or unavailable
runtime is a failure, never a successful empty prediction. Raw local output,
normalization evidence, timing coverage and saved settings remain inspectable.
Usable boxes open the same separate, timed human correction editor; they do not
alter the independent reference or frame annotations.

## Prepare and approve a multimodal trial

Choose **A · Multimodal · OpenAI** in the configuration form. The server reads
`IRIS_OPENAI_API_KEY`, falling back to `OPENAI_API_KEY`; keys are never entered or
displayed in this UI. The configuration status is an offline check. A configured
key does not prove that the account can access the model or that a connection works.
You can prepare and freeze a configuration without a key or external request.

The frozen settings include the exact model ID, class IDs/names/definitions,
prompt and structured output schema, image transform, reasoning effort, maximum
output tokens and pricing basis. The default maximum image long edge is 1536 pixels,
with 512/1024/1536/2048 available; images are not enlarged. Image detail is `original`.
Reasoning defaults to `low`; the available efforts come from the provider catalog.
The output limit defaults to 4096 tokens and accepts 1024–8192. Candidate scores are
unavailable, so multimodal proposals have no detector confidence threshold or AP.

After saving the configuration, choose its scene role and **Preview trial**. This
step remains local. For every image the preview displays the exact outgoing PNG,
original and sent dimensions, pixel/PNG hashes, coordinate transform, prompt, class
definitions, request hash and per-image estimate. The PNG is RGB and contains no
source metadata. Reference boxes, correction decisions, reviewer notes and
independence notes are excluded from provider requests. Text or sensitive content
visible in image pixels is still part of the outgoing image.

Inspect these inputs, enter the approved USD planning budget and check the explicit
approval for this provider, model, image set and budget. **Send approved external
trial** is available only after all outgoing images have loaded, the key is
configured, the preview is current, and the budget covers its estimate. The signed
preview expires after ten minutes. New previews, changed trial/configuration
settings, workspace or session changes and edited budgets clear approval; a changed
budget requires checking the declaration again. Every trial requires fresh approval.

The conservative estimate is an admission budget for the listed requests, **not a
guaranteed provider billing cap**. It includes the frozen output token allowance and
an offline input allowance, not an exact tokenizer measurement. The UI displays its
basis. Provider usage and invoices remain distinct; account-specific rates or
processing details can differ. Sending images does not establish a processing
region. The frozen request uses `store: false`, which alone does not establish zero
data retention. Review applicable provider/account settings before sending data.

IRIS records an attempted dispatch before sending each request. Cancelling a local
job cannot guarantee that an already submitted request stops or incurs no charge.
The first failed request stops the batch; remaining images are recorded as unsent.
IRIS does not automatically resend uncertain requests. If the creation response is
lost, the UI performs a read-only lookup for the exact saved trial fingerprint. A
matching receipt opens the existing trial. Otherwise it clears approval and directs
you to saved trials and Project jobs; preparing another trial can incur another
charge. Model aliases may change behind the same name; the raw response and returned
model identity remain part of the saved evidence.

## Prepare and approve a combined trial

Choose **C · Astra + SAM 3**. The configuration reuses A's image long edge,
reasoning effort and output-token limit, and B's SAM native-score threshold and
CUDA device. Class IDs and definitions come from the frozen benchmark reference.
There are no manually entered class prompts for C: Astra generates them for each
image. B's saved manual prompts do not carry over to C.

The frozen protocol has exactly three stages and no iterative refinement:

1. **Planning — Astra:** send the image and class definitions to generate one
   bounded text phrase per class. Reference boxes and human review notes remain
   excluded.
2. **Grounding — local SAM 3:** encode the original image once and evaluate the
   generated class prompts. Keep native boxes above the frozen SAM threshold.
   No masks are computed or saved.
3. **Review — Astra:** send the image again, together with the generated prompts
   and candidate IDs, coordinates, labels and native scores. Astra must accept,
   reject or relabel every candidate ID exactly once. It cannot invent a new
   candidate, move a box or request another model pass. Invalid decisions fail
   validation; they do not silently become an empty successful output.

**Preview configuration** shows work for each role: image count, the maximum of
two external calls per image, one SAM encoding per image and one prompt evaluation
per class. **Save frozen configuration** saves both provider profiles, settings,
class definitions, model/code/weight identities and the planning/review schemas.
Preparation does not require a server key or installed SAM runtime. Execution
requires both; missing setup remains visible and blocks launch. See the
[SAM setup guide](sam-preannotation-adapter.md) for local requirements. The UI
neither installs models nor tests provider access automatically.

**Preview trial** stays local and shows the exact first outgoing PNG and planning
prompt. The second request is deliberately displayed as a **review template**:
generated class prompts and SAM candidates do not exist yet. Its frozen prompt,
schema/settings, template hash and maximum **128 KiB of dynamic text** are
inspectable. The exact second request and its hash are retained after those data
exist. Both calls use the displayed transformed PNG; SAM uses the original frozen
image locally. The trial work plan also exposes the checked SAM runtime identity.

The explicit approval names the image set, both kinds of outgoing data, maximum
call count, provider/model and total USD planning budget. Approval requires every
image preview to load and a current signed preview, as for A. The total estimate
covers both planning and bounded review calls. It is a conservative admission
budget, not a guaranteed invoice cap. Changing the budget clears the checkbox;
each new trial requires fresh consent. A local cancellation cannot retract an
already submitted external call or prove zero charges.

The output retains separate **planning**, **grounding** and **review** records,
including raw responses, generated prompts, native SAM evidence, review decisions
and normalization errors. Open a stage's details from the saved trial to inspect
them. Planning and review also have separate external dispatch and usage receipts;
their counts refer to calls, not images. A saved planning response or a saved SAM
output is a partial result until the whole protocol succeeds. Failures,
cancellation and unknown delivery outcomes never trigger an automatic retry.

C's final proposals have no confidence score. Native SAM scores remain in source
provenance and are not comparable to calibrated probabilities or detector scores.
Final geometry is copied from SAM; only the retained class may change during
review. A usable output opens the same isolated human correction editor, without
changing its reference or model-stage records. Implementation tests and simulated
provider responses do not establish the quality or speed of real Astra/SAM runs.

## Read results and retained evidence

The table keeps configuration and scene role separate. It displays extra and missed
boxes, class conflicts, precision, recall and matched-box IoU. Local detectors and
SAM use their respective frozen native-score thresholds; multimodal trials include
all valid proposed boxes. Combined trials measure the boxes retained by the final
review after SAM thresholding, without a second confidence filter. SAM masks are
not part of this measurement.
This is operating-point geometry matching, not AP. Native provider scores
are not calibrated or comparable probabilities. The saved scoring protocol explains
one-to-one matching and the IoU threshold.

Headline quality metrics require a successful output for every image in that role.
An image with failed or invalid output is never treated as a successful empty
prediction. Partial outputs, raw responses, normalization errors, source identities
and work settings remain inspectable from the saved trial. **Job details and
cancellation** opens the durable processing record; no additional trial is launched.

Processing/API time and human correction time are distinct. Local processing timings
include decoding and inference for the measured images. Detector controls exclude
warm-up; SAM has no warm-up and includes the first pass. External
trials record observed image/request processing. The number of measured versus
planned images is shown. Failed attempts can also have recorded processing time.
Missing durations are unmeasured, not zero. Local monetary cost is unmeasured.
Combined image time covers planning, local SAM and review end to end, including
the first model load. Its displayed SAM loading duration is a **subset** of that
time and must not be added again. Approach B continues to report model loading
separately from its image-processing duration.

External trial details show each image's dispatch state: **Not sent**, **Request in
progress**, **Response received**, or **Delivery outcome unknown**. A received
response does not itself prove usable boxes or a final charge. Inspect the raw
response, normalization result/errors, returned usage and request identity alongside
the dispatch receipt. Invalid responses remain failures, never successful empty
images. An unknown delivery outcome is never assigned zero cost. When only some
usage is available, the displayed known subtotal is explicitly separate from the
unknown total. Usage-based estimates use the frozen price basis and are not invoices.
Reserved planning amounts are also distinct from recorded usage estimates.

## Measure corrections without changing the reference

Choose **Measure human correction** on a usable output. The separate editor loads
candidate boxes, frozen class definitions and any saved correction draft. It does
not load the human reference into its canvas.

Enter a correction reviewer and choose **Start review**. The image and editable
boxes appear only while the timer is running. You can draw missing boxes, select,
move, resize or remove existing boxes, change classes, apply exact coordinates,
zoom, pan and undo/redo. The correction record retains each box's originating
proposal ID. Saved decisions are derived as accepted, corrected or rejected;
newly drawn boxes are recorded separately.

- **Pause** masks the image and disables editing. Resume explicitly when ready.
- Timing pauses when the window loses focus, the tab is hidden, or the editor has
  no interaction for 60 seconds. Navigation pauses before leaving. Unsaved changes
  require saving or an explicit discard.
- **Save draft and pause** saves a correction revision and the final timing segment
  together. It does not complete human review.
- **Complete human review and pause** explicitly marks the correction reviewed,
  including when the human deliberately leaves the image empty.

The server records timing receipts. The browser sends a heartbeat every ten seconds;
its live clock includes the current interval awaiting a receipt. A lease expires
after thirty seconds without a confirmed continuation: an interruption gap is not
added to the measured duration. Restarted or restored timers retain confirmed
intervals and require explicit resume. An owner token and revisions guard concurrent
editors. The fully-timed flag means that no interruption is known to this protocol;
it does not prove continuous reviewer attention.

On a network failure the editor masks and checks the saved record with a GET. It
does not automatically resend a timer action or correction save. If a matching
new correction revision was already saved, the UI recovers that receipt. Otherwise
it keeps local edits and reports the conflict or uncertainty. After a lost start
receipt, a still-running owned timer must be paused before explicitly resuming.
This pause discards its unconfirmed interval and records an interruption, so time
spent behind the editor's mask is not added later. Reloading a running correction
uses the same conservative pause before resuming. Saving is blocked while a server
timer is running but the editor is masked.

Correction history preserves earlier revisions, reviewer, decisions and timing.
Only explicitly completed reviews contribute to completed correction summaries.
An incomplete or unmeasured timing record remains visible as such. These human
corrections are separate from both the immutable reference and original proposals.

## API and storage

The routes are project-scoped:

- `GET /api/benchmark-candidates`
- `GET /api/benchmark-providers` (offline configuration status and settings)
- `POST /api/benchmarks/preview`, `POST /api/benchmarks`, `GET /api/benchmarks`
- `GET /api/benchmarks/{id}`
- `POST /api/benchmarks/{id}/configs/preview` and `/configs`
- `POST /api/benchmarks/{id}/lock`
- `POST /api/benchmarks/{id}/trials/preview` and `/trials`
- `GET /api/benchmark-trials/{id}`
- `GET /api/benchmark-configs/{id}/frames/{frame_id}/input-image` (exact local PNG)
- `GET` and `PUT /api/benchmark-outputs/{id}/correction`
- `POST /api/benchmark-outputs/{id}/timer`
- `GET /api/benchmark-outputs/{id}/corrections/{revision}`

Creation uses preview fingerprints; external trials additionally require a signed,
unexpired preview token, `approve_external: true` and `max_cost_usd`. Correction and
timer changes use revision checks. Schema 15 stores independent benchmarks, configurations, trials, outputs,
correction revisions and timer receipts. Workspace backups include these records
and frozen benchmark images. No new background service is required.
