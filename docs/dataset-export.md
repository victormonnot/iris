# Frozen datasets and COCO export

In **Dataset & training**, select the saved class version for a release. Candidates
are selected images whose latest annotation is human-validated, with every proposal
resolved, using that exact class version. Images using other definitions remain
available by switching versions; they are never merged or silently relabeled.

Assign complete scene groups to train, validation or test. Train and validation
must each contain an image and use separate groups. Existing group reservations
apply within the project, while exact image pixels keep their split throughout
the workspace, including splits declared by COCO imports. Original video hashes
also retain one split across the workspace, including reimports into other
sessions. Historical conflicting reservations block reuse without rewriting old
releases. These protections do not prove independence between related scenes,
reencoded footage or visually similar images.

**Suggest whole-group partitions** previews an allocation by target proportions
and seed, with class coverage and similarity warnings. Applying it only fills the
builder; final publication rechecks current reviews and source reservations.
See [import and useful selection](intake-selection.md) for the workflow and limits.

Freezing copies each image and records its full reviewed revision, source provenance,
class definitions and mappings. The browser sends the revision IDs it displayed;
if a review changed meanwhile, publication returns a conflict and requires refresh.
An optional parent release must belong to the same project and use the same class
version. To use different definitions, start an independent release; existing split
reservations still apply. Empty images require human validation like positive images.

Custom releases can be inspected, trained, evaluated and exported. Training and
evaluation require compatible checkpoints and explicit launches; neither runs
as part of freezing or exporting a release. See [custom training](custom-training.md).

## Download

Choose a frozen version under **Dataset releases**, then **Download COCO ZIP**.
No GPU, checkpoint, API account or internet connection is needed. The same
operation is available at `GET /api/datasets/{dataset_id}/export/coco`.

Export reads the release snapshot, not current editor state or frame selection.
Every included image was recorded as validated when that release was created;
empty validated images remain explicit negative examples. Later drafts, new
proposals or changes to source files do not alter its contents. Export does not
validate any new image, change splits or run a model.

## Layout and labels

```text
train/annotations.json
train/images/<frame_id>.png
val/annotations.json
val/images/<frame_id>.png
test/annotations.json
test/images/<frame_id>.png
iris-manifest.json
export.json
README.txt
```

The test annotation document is present even when the release has no test images.
An empty test split does not establish an independent quality measurement. Files
remain assigned to their original splits and scene groups.

Each COCO `file_name` is `images/<frame_id>.png`, relative to its split directory.
Set an external loader's image root to the extracted `train`, `val` or `test`
directory matching its annotation document. Every split lists every frozen class,
including classes without boxes. COCO category names are stable class IDs; display
names and full definitions are retained in the manifest and version-2 export metadata.

New schema-2 manifests record two mappings:

| Mapping | Original Person / Car | Custom classes |
| --- | --- | --- |
| `class_mapping` | person=1, car=2 | 1…N in saved class order; 0 is reserved |
| `coco_mapping` | person=1, car=3 | 1…N in saved class order |

The optional `coco_id` in a class definition describes compatibility with an
official pretrained detector. It does not determine a custom export's category ID.
For example, a custom Bottle class may map from official detector category 44 but
use category 2 in this export. Consumers must use the COCO category table and,
for schema 2, the saved `coco_mapping` rather than inferring IDs from names or
detector mappings. Legacy schema-1 manifests have no `coco_mapping`; their
export category table retains person=1 and car=3.

Boxes use COCO `[x, y, width, height]` in image pixels, converted from the frozen
`[x1, y1, x2, y2]` coordinates without rounding. `area` is box width × height,
and `iscrowd` is 0. This export contains bounding boxes, not segmentation masks.
An image without objects is listed in `images` with no associated annotations.
Image and annotation numeric IDs are assigned deterministically; IRIS identifiers
retain the link to the original frame and reviewed box.

## Provenance and integrity

`iris-manifest.json` preserves the original manifest bytes and checksum. Its
image paths refer to the original workspace layout; `export.json` maps the
images to archive paths and records file checksums. New schema-2 manifests export
with `iris-coco-export-v2`, including the frozen class definitions and both mappings.
Existing schema-1 releases retain their original bytes and `iris-coco-export-v1`
behavior, including Person / Car IDs. No existing dataset manifest is rewritten
when the workspace schema is upgraded. Load images through the COCO paths when using the archive.

The manifest retains source identifiers, filenames, scene groups, timestamps,
annotation revisions, reviewer names and notes. Imported dataset attribution,
license and source metadata remain present as originally recorded; exporting
does not grant new rights or replace those licenses. User-authored metadata is
not redacted. Only the release files are packaged: the export does not collect
the workspace database, credentials from the environment, provider response
files, model checkpoints or original videos.

Before download, IRIS checks the manifest digest, review/split consistency and
each copied PNG's file digest, dimensions and pixel digest. It checks the same
image bytes that it writes to the archive. A missing or changed artifact fails
the whole export; a partial dataset is never returned as a successful download.
Archive entries use generated names and fixed ZIP timestamps. Unchanged releases
export to identical bytes under this export protocol.

## Limits and failures

- At most 1,000 images and 500 boxes per image, matching dataset release limits.
- At most 20 million pixels and 64 MiB of PNG data per image.
- At most 16 MiB for the frozen manifest and 256 MiB for the complete ZIP.
- One export at a time per running application, including the transfer.

PNG files are stored without recompression. Allow up to 256 MiB of temporary disk
space in the workspace and browser memory for the download. The server removes
the temporary archive after sending it or handling a disconnected client; a hard
process termination can leave a temporary file in the workspace's `exports/`
directory. The source release is preserved on failure. Oversized exports return
HTTP 413, invalid or unavailable artifacts and concurrent exports return 409,
and an unknown release returns 404. No archive is saved in the public repository.

The UI reports that the download has started; the browser determines where it
is saved. Changing releases during preparation cancels that UI download. This
does not guarantee immediate cancellation of server-side preparation.

This package is for external detection tools. The current IRIS importer accepts
one COCO document and one declared scene group/split per archive, so it cannot
directly restore this multi-split export. Use **Backup all projects** while the
workspace is idle to preserve jobs, models and all revision history. Stop IRIS
before using the CLI backup command; see [workspace backup](workspace-backup.md).

## Verification scope

Automated tests use generated images and explicitly synthetic reviewed labels.
They check COCO loading, fractional boxes, negatives, split and attribution
preservation, repeatability after reopening the workspace, changed-artifact
rejection, size bounds and temporary-file cleanup. Browser checks exercise the
download and its error states. These checks establish format and application
behavior; they do not measure detector quality on real data.

## Local API

`GET /api/dataset-candidates?taxonomy_id=<id>` selects a saved project class version;
without it, the project's current version is selected. The response includes the
selected `taxonomy`, available `taxonomies`, excluded counts and each candidate's
`annotation_revision_id`.

`POST /api/datasets` accepts `name`, `frame_ids`, `splits`, optional `parent_id`,
`taxonomy_id` and `expected_revisions` (frame ID to annotation revision ID). If provided,
the revision map must cover exactly the selected frames. Older clients may omit the
last two fields; the server then freezes the latest eligible revisions and requires
a single shared class version. Lists and detail responses expose `taxonomy_id`,
`taxonomy`, `class_mapping`, `coco_mapping`, `ml_supported` and `ml_limitation`.
Existing project-scoping query parameters still apply.
