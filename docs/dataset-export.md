# Frozen dataset export

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
directory matching its annotation document. Categories are **1 = person** and
**3 = car**, using the definitions in the frozen `iris-objects-v1` taxonomy.
The detector's native training mapping, `person=1, car=2`, is a separate mapping.

Boxes use COCO `[x, y, width, height]` in image pixels, converted from the frozen
`[x1, y1, x2, y2]` coordinates without rounding. `area` is box width × height,
and `iscrowd` is 0. This export contains bounding boxes, not segmentation masks.
An image without objects is listed in `images` with no associated annotations.
Image and annotation numeric IDs are assigned deterministically; IRIS identifiers
retain the link to the original frame and reviewed box.

## Provenance and integrity

`iris-manifest.json` preserves the original manifest bytes and checksum. Its
image paths refer to the original workspace layout; `export.json` maps the
images to archive paths and records file checksums and the export protocol
`iris-coco-export-v1`. Load images through the COCO paths when using the archive.

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
directly restore this multi-split export. Back up the complete workspace while
IRIS is stopped to preserve jobs, models and all revision history.

## Verification scope

Automated tests use generated images and explicitly synthetic reviewed labels.
They check COCO loading, fractional boxes, negatives, split and attribution
preservation, repeatability after reopening the workspace, changed-artifact
rejection, size bounds and temporary-file cleanup. Browser checks exercise the
download and its error states. These checks establish format and application
behavior; they do not measure detector quality on real flights.
