# COCO dataset import

[Documentation](README.md)

COCO means *Common Objects in Context*. It names a public computer vision
dataset, a common JSON annotation format, and an evaluation protocol. Importing
a dataset in COCO format does not mean downloading the entire COCO dataset or
using its category IDs as IRIS labels.

## Package and preview

Prepare a ZIP with exactly one `.json` file containing `images`, `annotations`
and `categories` arrays. Each image needs an integer ID, `file_name`, `width`
and `height`. Categories have integer IDs and names. Each annotation needs an
integer ID, image/category references, and a pixel box `[x, y, width, height]`.
The importer checks positive dimensions, finite coordinates and complete boxes
inside the image. IDs must be unique within each array.

```text
sample.zip
  annotations.json
  images/
    frame-001.jpg
    frame-002.jpg
```

Here `file_name` is `images/frame-001.jpg`. Files resolve relative to the ZIP
root or the JSON's directory; conflicting matches are rejected. Every referenced
image must be present. Remote URLs, traversal, absolute paths, symlinks and
duplicate ZIP entries are rejected. No image URL is fetched. EXIF orientations
other than 1 must be normalized together with their boxes before packaging.
Supported image formats are PNG, JPEG, WebP, BMP and TIFF, with one frame per
file. External document renderers are never invoked. IDs must fit a JavaScript
safe integer (0 through 9,007,199,254,740,991).

Limits per package are 64 MiB compressed, 256 MiB expanded, 16 MiB JSON, 100
images, 20 million pixels per image, 100 million pixels total and 500 annotations
per image. These bounds keep the preview practical on a local workstation.
The UI stays available while the server prepares the bounded preview.

Preview shows every image and its source boxes. Each source category requires
an explicit mapping to a target class or `exclude`, including unused categories.
The preview saves the project's class version and definitions. Publishing newer
classes later does not change that preview, its mapping choices or committed images;
upload a new preview to use the new version. Historical previews retain the original
Person / Car definitions. Imported positive and negative images both retain their
target version and require human review.
Check IRIS's class definitions before mapping; similar names do not establish
equivalent taxonomies. Exclusion removes proposals for that category, not its
source record. It is not an ignore region: a later prediction in that area can
still count as a false positive. Check that excluded classes do not contain
objects which the target taxonomy requires.

Any crowd or ignore annotation rejects the package. Do not convert such objects
to ordinary boxes or silently discard them to bypass this limitation. A curated
subset can omit entire affected images if that selection is documented; its
results would then describe that subset, not the official benchmark. Segmentation
and keypoint payloads are retained as source metadata but are not interpreted.

## Provenance and human review

Provide a name, scene group, source location, license description and attribution.
These fields record provenance; IRIS does not infer permission from a URL or
verify that a license applies. Keep original terms and attribution with any
dataset you redistribute.

The source location accepts an HTTP(S) URL without credentials or a local
`file:///absolute/path` URI, for example `file:///home/you/media/FPV%20session`.
File URIs must have no host (including `localhost`), query or fragment; encode
spaces as `%20`. The API retains the field name `source_url`. This location is
metadata only: IRIS never opens, resolves or fetches it. Images still come only
from the uploaded ZIP, and the recorded location need not exist on this machine.

One package creates one session and scene group. Split larger datasets into
packages for their existing scene groups and splits. COCO has no standard split
field; declare the source split as `train`, `val`, `test` or unknown. If supplied,
embedded split metadata must agree with that declaration. Unknown splits are
assigned later when freezing. Known source splits reserve groups and exact pixels
immediately, including unreviewed images. Related scenes across source splits
still require inspection; different IDs or dates do not prove independence.

Original archive bytes and SHA-256, raw annotations and categories, image pixels,
mapping and source metadata stay in the local workspace. Confirmation rechecks
integrity and publishes the session, frames and suggestions in one SQLite
transaction. Repeating the same confirmation returns the same imported session;
changing its configuration requires a new preview. Previews can be reopened
after restart.

Imported boxes appear as **Imported annotation** proposals. Accept, correct or
reject each one, inspect omissions and validate with a reviewer name. Empty
images also require explicit validation. A reviewed dataset release freezes the
annotation revisions and import provenance, including excluded source labels.
No import claims that a human has performed this review.

## Small public aerial pilot

[HIT-UAV](https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset) provides thermal aerial images
and annotations under its [CC BY 4.0 license](https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset/blob/669259659b21a2737ca9f5e4bc81892acbe6ecd8/LICENSE).
It is suitable for exercising import and review without private flight data.
Thermal imagery differs from RGB flight footage, and RGB COCO detector weights
are not validated for this domain.

The source's `normal_json` format uses `annotation` and image `filename` keys;
convert them explicitly to COCO's `annotations` and `file_name`, retaining the
original source and conversion record. Its normal boxes use `[x, y, width,
height]`, as documented in the [upstream format notes](https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset/blob/669259659b21a2737ca9f5e4bc81892acbe6ecd8/tools/notes/Structure%20of%20dataset.md). Preserve the original
train/validation split. Select whole images without `DontCare` regions rather
than deleting those labels. Map Person/Car explicitly and review all proposals
against IRIS's definitions.

A small curated sample can establish that the pipeline runs on real imagery.
It cannot establish an official HIT-UAV score, RGB flight performance, a gain
from fine-tuning, or scene independence. Those require reviewed labels and an
appropriate evaluation design. Keep your datasets and downloaded artifacts in
the ignored local workspace. The repository includes a small, attributed
[street-scene example](../examples/street-scenes/README.md) for the
[first-run guide](first-run.md).
