# Iris browser demo

This example supplies one project, two real FPV camera images, imported human
reference boxes and saved YOLOX-Nano results before and after training in Iris.
The records use Iris's actual API shapes in the shared browser interface. The
browser adapter lets visitors select images, review annotations and inspect
recorded predictions without an Iris API or model runtime.

The example project's current class definition is **Person** only, published
before importing the images. Imported proposals and annotation tools use this
same definition; recorded models retain their original class contracts.

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

The shared workbench and both FPV images load behind a soft blur. **Try IRIS**
reveals the already-loaded interface in place. The entry explains the demo's
limits, keeps the workbench inactive until entry, and offers a retry if the data,
images or interface cannot load. **Install IRIS locally** opens concise Linux/WSL
instructions with copyable commands and links to the full setup and first-run
guides; model inference and CUDA setup are separate optional steps.

The native Iris navigation stays in place, including its processing workspaces.
The saved comparison is preselected when **Model comparison** opens: the original
YOLOX-Nano is on the left and the version trained in Iris is on the right, on the
README example frame, filtered to people at confidence 0.35.

The demo supports gallery selection and filters; drawing, moving, resizing and
deleting boxes; accepting, correcting and rejecting imported proposals; saving
drafts, validating, and viewing annotation history. Class and confidence filters
operate on real saved predictions. Annotation edits never alter those outputs.

Personal uploads, import/export, class changes, new projects, inference, model
assistance, training and evaluation require [local Iris](../../docs/setup.md).
Their original controls remain visible but disabled with an explanation. Empty
processing lists do not invent training runs or datasets. Unsupported adapter
requests fail explicitly and cannot fall through to a server API.

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
the toolbar's reload or reset actions.

**Reset demo** asks for confirmation before removing saved annotations, history
and selection changes. A new reset identifier invalidates editors left open in
other tabs. When a seed or storage schema changes, incompatible saved data is
kept and the demo offers an explicit reset instead of silently deleting it.

### Continue locally

Use **Install IRIS locally** for setup instructions. The hosted sample lets you
explore the interface; importing your own recordings, exporting datasets and
running models are available in the installed tool. Browser edits remain local
to this demo and do not change the author's saved experiment.

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

Saved browser data belongs to the site's origin, demo location and sample,
separately from the asset version. Rebuilding only the interface preserves that namespace.
The person-only FPV sample uses the `fpv-yolox-person-v1` namespace. Earlier FPV
edits made with person/car classes and earlier street-scene edits remain in their
separate browser databases; they are not transferred to this class definition.
Changing the domain or demo path also creates a different browser workspace.
An incompatible update within the same sample still offers an explicit reset.

Keep the bundled `LICENSE`, `THIRD_PARTY_NOTICES.md`, font licences and sample
credits with the artifact. The notices link to the accompanying font files,
photo credits and provenance. The image and annotation licences are separate
from Iris's software licence.

## Contents and attribution

The [source bundle](source/fpv-yolox.zip) contains two unchanged camera JPEGs,
640 × 480 pixels, from Victor Monnot's 3 October 2026 courtyard recording.
Camera frame 1601 is the image in [the YOLOX example](../../docs/iris-to-argos.md);
frame 1681 follows in the same recording. RGB PNG display copies preserve the
complete images. The source images and manual annotations are published by their
author for this demonstration, with all rights reserved; they are separate from
Iris's MIT software licence. See [source credits](source/CREDITS.txt).

Each image starts with one imported person proposal, taken from the frozen human
reference used for the experiment. The original source was human-reviewed. In
this separate visitor workspace, both proposals start pending, with revision
zero, no saved boxes and no visitor validation. Their source split stays `val`.

[recording.json](recording.json) projects four actual saved prediction records
from the official and 800-step Iris-trained YOLOX-Nano evaluations of 7 October
2026. The comparison and job objects are display wrappers joining those two
existing evaluations; no joint comparison job or new inference was performed.
All 206 native detections retain their coordinates, scores, category metadata,
original record IDs, dates and per-image CUDA timings. Checkpoint hashes and the
training reference identify the exact models; no weights are bundled.

At confidence 0.35, the original model misses the person on frame 1601 while the
trained model detects them at 0.736884. Both detect a person on frame 1681.
These are selected development images from the reused 17-image validation set,
not independent evidence of generalization. The measurements came from the
original RTX 4060 evaluations and are not browser performance claims. The
[experiment record](../../docs/acceptance-results.md#yolox-nano-custom-detector-accepted-by-an-external-application)
explains the full split, quality results and limitations.

## Rebuild the public data

From the repository root, with the development environment installed:

```sh
uv sync --locked
uv run python examples/web-demo/build.py --check
uv run python examples/web-demo/build.py --output /tmp/iris-web-demo-seed
```

The normal build requires neither ML dependencies nor model weights. It imports
the hash-verified FPV ZIP through the real API in a new temporary
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
The project ID is `web-demo`, and the session ID is `fpv-yolox`.

Response keys omit `project_id` query parameters. A consumer must check the
project, interpret supported queries and apply browser edits itself. These
snapshots are the initial data, not a catch-all response for arbitrary requests.
System capabilities, provider availability and the live job queue belong to
the browser runtime and are deliberately absent. Model descriptors identify the
recorded detectors; they do not declare locally installed or runnable models.

Temporary import IDs and creation dates are normalized for reproducibility.
Public COCO image IDs equal camera frame numbers 1601 and 1681. Original source
record IDs, capture dates, annotation geometry, hashes and measured inference
values are retained in provenance. Sorting of imported records is stable.
The fixed import date is a fixture placeholder, not a claimed review date.

`seed/provenance.json` records source provenance, the recording checksum,
normalizations, and the hashes and sizes of every other generated file.
`asset.sha256` hashes the original JPEG bytes; `frame.sha256` hashes normalized
RGB pixels with dimensions. These differ from the PNG file hashes in provenance.
Keep [seed/CREDITS.txt](seed/CREDITS.txt) and provenance alongside deployed media.

## Reproject the saved source evaluations

`record.py` can reproduce the public projection from the original source
workspace when it is available:

```sh
uv run python examples/web-demo/record.py \
  --workspace /path/to/source-workspace \
  --output /tmp/iris-fpv-source-projection
```

This optional maintenance command opens SQLite in read-only mode, verifies the
selected frozen dataset and camera image hashes, and copies only the two named
images, their manual reference boxes and four saved model results. It never
executes a model or copies weights, unrelated records, private notes or local
paths. The output directory must not exist. Review the projection before
replacing `source/` and `recording.json`; ordinary builds need only those checked-in
public files and do not open the source workspace.
