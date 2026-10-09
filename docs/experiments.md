# Experiment reports

[Documentation](README.md)

An experiment report is a saved reading of one completed evaluation. It joins
the evaluation to its frozen dataset, detector checkpoints and available local
training history. The title, objective and conclusion are written by the user;
measured results are kept separately. A report does not promote a model or start
inference, training, annotation or an external request.

## Create and read a report

Use **Experiments** or the report action in **Quality evaluation**. Select a
completed evaluation and review its scope before saving. Reports support one
detector, two detectors, or full-image and tiled runs of the same checkpoint.
All compared lanes belong to that evaluation and share its frozen split, class
definitions and metric settings. Custom reports retain the complete class snapshot,
per-class counts and checkpoint mappings; tables and overlays use those saved labels. Results from unrelated evaluations are not merged into a
comparison.

Select zero to six example images from the evaluated split. Each example retains
its source information, reviewed boxes and saved model outputs. The overlays use
the recorded confidence threshold; COCO AP continues to use all saved scores
according to the original metric protocol. Selecting illustrative examples does
not restrict or recalculate the aggregate metrics.

The example picker filters saved images by scene, operating-point changes,
negative images and recorded timestamps. **Use suggested examples** explicitly
adds up to six diverse examples with recovered objects, new misses or false
positives; it never saves a selection automatically. An improved or regressed
image describes those counts at the saved threshold, not all aspects of box
geometry or detector quality. A mixed image contains both gains and regressions.

Scene tables show labeled objects, true positives, false positives and misses
per class and pipeline. Recovered objects, new misses and false-positive changes
use the original matching evidence. Scene AP is not recomputed. Video source
summaries identify only the frozen sampled images and their approximate recorded
timestamps; their first and last timestamps do not establish continuous coverage,
tracking quality or live video performance.

The report separates:

- **Recorded evidence:** dataset identity, checkpoint hashes, available training
  configuration, measured scores, class breakdown, processing times and errors.
- **Written interpretation:** the user's objective and conclusion.
- **Historical decisions:** reference decisions known when the report was saved.

A reported reference is historical, not a claim that it is still the workspace's
current reference. A test-split report is a final-audit record, not a model-selection
action. Missing training history or timings are shown as unavailable. Official
pretrained models do not acquire an invented local training run.

Metric differences are candidate minus baseline. AP, precision and recall
differences are percentage points, not relative percent improvements. Undefined
metrics stay unavailable; they are not converted to zero. A single-model report
has no before/after delta or automatic winner. Speed measurements retain their
recorded environment and protocol and are not a general hardware benchmark.
For new reports, a timing difference is shown only when complete measurements,
hardware/runtime settings and compatible whole-pipeline timing protocols are
recorded. Otherwise absolute recorded times remain available with the reason a
comparison is unavailable.

## Include saved target measurements

Optionally select up to four measurements already imported through **Model
exports**. Nothing is selected by default. The picker considers the 100 most
recent saved measurements for this evaluation and omits missing or inconsistent
evidence. Each selected measurement must match the exact full-image evaluation
lane, checkpoint, class contract, dataset manifest and saved reference predictions.
Measures from unrelated evaluations or a tiled pipeline cannot be attached.

The report freezes the selected export and measurement identities, hashes,
declared target environment, exact parity result, reference-image identities and
timing summaries. Raw measurement detections, checkpoint paths and CUDA device
UUIDs are excluded. Simulation is labeled explicitly. An external execution
declaration is not authenticated by IRIS; a failed parity check remains visible.
Parity only covers the exported reference images.

Target processing time is displayed separately from decoding, model loading and
warmup. Export processing time excludes those costs, whereas IRIS evaluation
total time includes image decoding and verification. The report calculates no
speedup between those contexts and does not rank devices. Different training and
target devices remain independent, including CPU and NVIDIA CUDA targets.

## Persistence and edits

The report stores a checksummed, versioned snapshot and bounded copies of the
selected images. Images are normalized JPEG previews with a maximum edge of
1,024 pixels; their recorded boxes remain in original image coordinates. These
copies are for visual inspection, not replacement training data or a dataset backup.

Once saved, the evidence and example selection are fixed. Editing the title,
objective or conclusion increments the report revision and checks the revision
last read by the editor. A stale edit or export is rejected rather than silently
using another revision. Create another report to choose a different evaluation
or different examples. A saved report can be reopened without the original
dataset images, model weights or optional ML runtime. Its own snapshot and image
checksums are still verified.
Selected measurement summaries likewise remain readable without the original
export archive, measurement row or detector runtime. New snapshots use version 2;
existing version-1 reports retain their original evidence and do not acquire
scene analyses or measurements retrospectively. No database migration is needed.

The creation preview has a source fingerprint. If the evaluated evidence changes
before saving, refresh the preview and review the selection again. Selected
measurements are revalidated before publication; the saved report is immutable.

## Standalone export

Export downloads a single HTML file, limited to 16 MiB. The default excludes
image pixels. The optional image export includes only the examples already
selected for this report, with reviewed boxes and model overlays. Source and
experiment names, notes and recorded provenance are still present in a text-only
export, so review the document before sharing it.

The document contains inline styles, optional embedded JPEGs and static overlays.
It has no JavaScript, remote fonts, external images or server dependency. It can
be read offline and printed with the browser's print dialog. User-supplied text
is escaped, export fields are explicitly selected, and a restrictive content
security policy blocks scripts and network requests. Original workspace paths,
provider credentials and raw annotation prompts are not report fields.

Only one HTML export is prepared at a time per app instance. Preparation creates
no new job or model. Exporting does not publish a URL or send a document anywhere.

## Verification scope

Tests use synthetic reviewed images and saved evaluation fixtures to verify
source consistency, paired and single-model semantics, frozen evidence, note-edit
conflicts, image integrity, offline rendering and export escaping.
Synthetic CPU/CUDA measurement declarations also verify precise source linkage,
parity failures, explicit selection, scene counts, video sampling limits and old
snapshot compatibility. These simulations do not run a detector or a GPU.
Existing saved results can also be used to exercise the complete report workflow without new
model execution. These checks establish report behavior; they do not establish
detector quality on real flights.
