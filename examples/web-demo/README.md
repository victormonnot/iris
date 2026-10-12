# Iris browser demo

This example supplies one project, two street photographs, imported annotation
proposals and a saved comparison of two detectors. The records use Iris's actual
API shapes, for consumption by the shared browser interface. The browser adapter
lets visitors select images, review annotations and inspect recorded predictions
without an Iris API or model runtime.

## Try the browser demo locally

From the repository root, run the static preview with Python's standard library:

```sh
python3 examples/web-demo/serve.py
```

Open `http://127.0.0.1:8765/` (`--port` changes the port). This preview serves the
maintained Iris HTML, styles and editor with the demo adapter enabled. It creates
no local Iris workspace, runs no API and binds only to the loopback interface.
The normal `iris` application continues to use its local backend. This helper is
for previewing the browser runtime; it is not a production hosting server.

The shared workbench and both sample photos load behind a soft blur. **Try Iris**
reveals the already-loaded interface in place. The entry explains the demo's
limits, keeps the workbench inactive until entry, and offers a retry if the data,
images or interface cannot load. **Install Iris locally** opens concise Linux/WSL
instructions with copyable commands and links to the full setup and first-run
guides; model inference and CUDA setup are separate optional steps.

The preview loads the shared modules for intake, annotation and comparison.
Comparison records are read when that workspace opens. Links into processing
features explain that local Iris is required and leave the sample available.

The demo supports gallery selection and filters; drawing, moving, resizing and
deleting boxes; accepting, correcting and rejecting imported proposals; saving
drafts, validating, and viewing annotation history. Model comparison offers the
real saved outputs with interactive class and confidence filters. Annotation
edits never alter those recorded outputs or rerun a model.

Personal uploads, class changes, new projects, new inference, model assistance,
training, evaluation and the other processing workspaces require
[local Iris](../../docs/setup.md). Their controls explain this limit. Unsupported
adapter requests fail explicitly and cannot fall through to a server API.

### Browser storage and reset

Saved annotations, revisions and image selection live in IndexedDB for this
browser and site. They survive reloads, but clearing site data removes them.
Separate browser profiles have independent data; there are no hosted accounts.
If the browser initially denies storage, a visible notice explains that changes
live only in the current page and are lost on reload. A failure during an active
persistent save is reported without silently switching storage modes.

Two tabs in the same browser share saved data. A notification offers a reload
when another tab saves; unsaved editor changes are kept. Stale annotation saves
are rejected using revision and reset checks. Selection changes are atomic and
reject conflicting values. Save or discard current editor changes before using
the toolbar's export, reload or reset actions.

**Reset demo** asks for confirmation before removing saved annotations, history
and selection changes. A new reset identifier invalidates editors left open in
other tabs. When a seed or storage schema changes, incompatible saved data is
kept and the demo offers an explicit reset instead of silently deleting it.

### Export and continue locally

**Export saved annotations** downloads `iris-demo-annotations.json` as COCO JSON.
It contains the current saved boxes for both example images, regardless of gallery
selection. Pending proposals and recorded predictions are not exported as labels.
The `iris` extension records review states, unresolved proposals, image selection
and source attribution. Draft or unannotated images remain explicitly marked;
this export is not a frozen training dataset or a complete workspace backup.

To import the labels into local Iris:

1. Extract [street-scenes.zip](../street-scenes/street-scenes.zip).
2. Replace its `annotations.json` with the downloaded JSON, preserving the
   original `images/` directory, credits and provenance. Create a ZIP containing
   those files directly at its root.
3. Use **Import annotated dataset** in local Iris, map `person` and `car`, retain
   the original `val` split, and supply the photo and annotation credits from
   the example. Imported boxes require human review again in the local workspace.

Keep the downloaded JSON as well: the local COCO importer does not restore the
demo's revision history or its human-validation status from the `iris` extension.

## Build a static hosting artifact

The exporter uses Python's standard library and the checked-in public seed. It
requires no Iris server, application environment, Node build or model runtime:

```sh
python3 examples/web-demo/export.py --output dist/demo
```

`--output` is optional; its default is `dist/demo` under the repository root.
Use a dedicated directory beneath `dist/` or outside the repository. Rebuilding
an intact generated artifact is supported; modified artifacts and arbitrary
existing directories are refused. Exporting creates local files only. It does
not configure hosting, set up publication automation or publish the site.

The artifact contains:

```text
dist/demo/
  index.html
  manifest.json
  assets/<sha256>/
    static/
    seed/
    LICENSE
    THIRD_PARTY_NOTICES.md
```

The generated index uses the same maintained Iris interface and demo modules as
the local preview. Its referenced scripts, styles, bundled fonts and public
example data are packaged together under a content-addressed asset directory.
No backend, model weights or personal workspace are included. The manifest has
format `iris-web-demo-site-v1`, an `asset_version`, and a `files` map recording
file sizes and SHA-256 hashes.

Frontend changes reach the artifact when it is rebuilt from the updated
repository. An exported or published copy does not update merely because the
source checkout changed. There is one maintained interface, with the browser
demo supplying its own data adapter.

For a published release, add `--revision` with the full Git commit SHA. The
manifest records that source revision and the demo displays a small link to
the corresponding commit. The [deployment guide](../../deploy/web-demo/README.md)
describes the tested-artifact pipeline, Coolify hosting and rollback.

To inspect the actual artifact with an ordinary static server:

```sh
python3 -m http.server 8765 --bind 127.0.0.1 --directory dist/demo
```

Open `http://127.0.0.1:8765/`. This serves the exported files rather than the
development preview's live source files. Opening `index.html` through `file://`
is unsupported.

### Hosting paths, caching and browser data

The whole artifact can be served at a site root or beneath a path such as
`/iris/`, without rebuilding for that prefix. Keep its directory structure and
use a URL ending in `/` or `/index.html`. Configure a redirect from `/iris` to
`/iris/` if the host does not already normalize directory URLs. Asset and font
URLs are relative; `/api/` remains a logical adapter namespace and needs no
server route or proxy.

Use HTTPS when publishing. Browser integrity checks require a secure context;
HTTP on localhost works for local previews. An ordinary static host is enough:
the workbench's mutations stay in the visitor's browser, without a backend or
hosted account.

When configuring hosting, allow long-lived immutable caching for
`assets/<sha256>/`, and revalidate `index.html` and `manifest.json`. Upload the
new assets before replacing the index. Keep older asset directories available
while visitors may still have an older page open or cached; do not overwrite
files inside an existing content-addressed directory. These are hosting
requirements, not configuration applied by the exporter.

Saved browser data belongs to the site's origin and demo location, separately
from the asset version. Rebuilding only the interface preserves that namespace.
Changing the domain or demo path creates a different browser workspace. A
changed seed still triggers the explicit compatibility/reset flow described
above, rather than silently discarding edits.

Keep the bundled `LICENSE`, `THIRD_PARTY_NOTICES.md`, font licences and sample
credits with the artifact. The notices link to the accompanying font files,
photo credits and provenance. The image and annotation licences are separate
from Iris's software licence.

## Contents and attribution

The photographs and original labels come from the attributed
[street-scenes example](../street-scenes/README.md). The two original JPEGs are
unchanged. Iris also produces RGB PNG display images at their original dimensions.
Both formats retain the photo credits and CC BY-SA 2.0 licence. Original COCO
annotations retain their CC BY 4.0 attribution.

The example begins with **11 imported proposals: 3 people and 8 cars**. All 19
source annotations are retained in provenance, including the eight excluded by
the class mapping. Both frames remain unannotated, at revision zero, with no
reviewer, saved boxes or human validation. Their original split is `val`.

[recording.json](recording.json) contains an actual CPU comparison of
SSDLite320 MobileNetV3-Large and Faster R-CNN MobileNetV3-Large 320 FPN, recorded
on 12 October 2026 with PyTorch 2.10 and Torchvision 0.25. The four saved
predictions retain all 100 native detections per model/image, including low
confidence detections, without display filtering. Model identities, checkpoint
hashes, configuration, runtime versions, dates and timings accompany the results.
No weights are bundled.

These are small-object street scenes on which the two lightweight models miss
many objects at the default 0.5 display threshold. The example demonstrates
inspection of real outputs; it is not an independent model-quality benchmark,
a training dataset or a promise of performance on a visitor's computer. Editing
the example annotations must leave the recorded predictions and timings intact.

## Rebuild the public data

From the repository root, with the development environment installed:

```sh
uv sync --locked
uv run python examples/web-demo/build.py --check
uv run python examples/web-demo/build.py --output /tmp/iris-web-demo-seed
```

The normal build requires neither ML dependencies nor model weights. It imports
the hash-verified street-scenes ZIP through the real API in a new temporary
workspace, restores the immutable recorded results, and reads an explicit list
of API responses and media. It performs no network requests and never opens an
existing Iris workspace. Temporary storage is removed on completion.

An existing output is accepted only if every file is byte-identical. To update
the checked-in `seed/`, build into a new directory and review the differences.
`--check` writes nothing and fails when the output differs or is missing.

Public projections are defined in [public_contract.py](public_contract.py) and
[recording.py](recording.py). They select allowed fields recursively, reject
missing or incompatible fields, and exclude workstation paths, provider settings
and arbitrary job logs. Changes to the API or source images must pass the fixture
contract tests before regenerated data is accepted.

## Files and response contract

`seed/fixture.json` has format `iris-web-demo-v1` and a separate
`fixture_version`. Its `responses` object maps logical API paths to read-only
snapshots; `resources` maps image/media API paths to relative files in `media/`.
The project ID is `web-demo`, and the session ID is `street-scenes`.

Response keys omit `project_id` query parameters. A consumer must check the
project, interpret supported queries and apply browser edits itself. These
snapshots are the initial data, not a catch-all response for arbitrary requests.
System capabilities, provider availability and the live job queue belong to
the browser runtime and are deliberately absent. Model descriptors identify the
recorded detectors; they do not declare locally installed or runnable models.

Temporary import IDs and creation dates are normalized for reproducibility.
COCO IDs, original capture dates, annotation geometry, source hashes and all
measured inference values are retained. Sorting of imported records is stable.
The fixed import date is a fixture placeholder, not a claimed review date.

`seed/provenance.json` records source provenance, the recording checksum,
normalizations, and the hashes and sizes of every other generated file.
`asset.sha256` hashes the original JPEG bytes; `frame.sha256` hashes normalized
RGB pixels with dimensions. These differ from the PNG file hashes in provenance.
Keep [seed/CREDITS.txt](seed/CREDITS.txt) and provenance alongside deployed media.

## Record new model results explicitly

With Iris's optional ML dependencies and the two official checkpoints already
available, use the separate recorder:

```sh
uv run --extra ml python examples/web-demo/record.py \
  --weights-dir /path/to/official-checkpoints \
  --output /tmp/new-street-scenes-recording.json
```

The recorder creates another temporary public-only workspace and runs both
detectors on CPU. It does not download weights, validate annotations or read a
workspace database supplied by the caller. Review the new recording before
replacing `recording.json` and rebuilding the seed. A fresh ML run has its own
timestamps and timings and may produce numerical differences; ordinary fixture
rebuilds reuse the saved recording and are byte-for-byte reproducible.
