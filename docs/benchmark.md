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

The three planned approaches are displayed separately:

- **A · Multimodal**: not connected yet.
- **B · Segmentation**: not connected yet.
- **C · Combined**: not connected yet.

The **local detector control** is executable using an already available checkpoint.
It exercises the benchmark workflow and does not substitute for evidence about
A, B or C. No new model is downloaded and no external provider is called here.

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

## Read results and retained evidence

The table keeps configuration and scene role separate. It displays extra and missed
boxes, class conflicts, precision, recall and matched-box IoU at the frozen proposal
threshold. This is operating-point geometry matching, not AP. Native provider scores
are not calibrated or comparable probabilities. The saved scoring protocol explains
one-to-one matching and the IoU threshold.

Headline quality metrics require a successful output for every image in that role.
An image with failed or invalid output is never treated as a successful empty
prediction. Partial outputs, raw responses, normalization errors, source identities
and work settings remain inspectable from the saved trial. **Job details and
cancellation** opens the durable processing record; no additional trial is launched.

Local processing time and human correction time are distinct. Processing timings
include decoding and local inference for the measured images and exclude warm-up;
the number of measured versus planned images is shown. Failed attempts can also
have recorded processing time. Missing durations are unmeasured, not zero, and
monetary cost is not measured by this local workflow.

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
- `POST /api/benchmarks/preview`, `POST /api/benchmarks`, `GET /api/benchmarks`
- `GET /api/benchmarks/{id}`
- `POST /api/benchmarks/{id}/configs/preview` and `/configs`
- `POST /api/benchmarks/{id}/lock`
- `POST /api/benchmarks/{id}/trials/preview` and `/trials`
- `GET /api/benchmark-trials/{id}`
- `GET` and `PUT /api/benchmark-outputs/{id}/correction`
- `POST /api/benchmark-outputs/{id}/timer`
- `GET /api/benchmark-outputs/{id}/corrections/{revision}`

Creation uses preview fingerprints; correction and timer changes use revision
checks. Schema 15 stores independent benchmarks, configurations, trials, outputs,
correction revisions and timer receipts. Workspace backups include these records
and frozen benchmark images. No new background service is required.
