# Public browser example data

This example supplies one project, two street photographs, imported annotation
proposals and a saved comparison of two detectors. The records use Iris's actual
API shapes, for consumption by the shared browser interface. It contains data
and generation tools; the interactive browser distribution is not bundled here.

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
