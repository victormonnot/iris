# Street scenes

Two real street photographs with their original COCO annotations. Use
[street-scenes.zip](street-scenes.zip) to try importing data and reviewing boxes
without downloading a model or using an API.

The ZIP contains two JPEGs, one `annotations.json`, and `CREDITS.txt`. It is ready
for the sidebar's **Import annotated dataset** action. The photos stay at their original sizes;
no model results or generated annotations are included.

## Import

Follow the [first-run guide](../../docs/first-run.md) for the complete walkthrough.
The existing default project has the **Person / Car** classes needed here. Choose
**Import annotated dataset** in the sidebar and upload the ZIP in the
**Import a COCO dataset** dialog. Map the categories as follows:

| COCO category | IRIS class |
| --- | --- |
| person (1) | Person |
| car (3) | Car |
| bus (6) | Exclude |
| truck (8) | Exclude |
| traffic light (10) | Exclude |

Keep the source split set to **Validation**. Use `First-run demo` as the scene
group. For source and attribution fields, use:

- Source: `https://huggingface.co/datasets/LibreYOLO/coco-val2017-mini500`
- License: `Photos CC BY-SA 2.0; COCO annotations CC BY 4.0.`
- Attribution: `COCO Consortium; Clongriffin by infomatique; Damaged Traffic Light - Junction of High Road & White Hart Lane by Alan Stanton. See the included CREDITS.txt for photo and license links.`

This mapping produces **11 imported proposals: 3 people and 8 cars**. The other
eight source annotations remain in the import's provenance. Review the boxes and
the whole image, then validate with your name. Imported labels are proposals,
not a claim that you have already reviewed the images.

## What is in it

| COCO image | Size | All source annotations |
| --- | --- | --- |
| 6723, *Clongriffin* | 640 × 361 | 3 cars, 1 bus, 2 trucks |
| 32941, *Damaged Traffic Light* | 458 × 640 | 3 people, 5 cars, 1 bus, 2 trucks, 2 traffic lights |

Both images come from **COCO val2017**. They are useful for learning the workflow,
but they do not form a training dataset or an independent test for COCO-pretrained
models. There is no training or test split in this sample.

Some boxes are small, overlapping or debatable. In image 6723, COCO has both bus
and truck annotations on one vehicle. Another van is labelled truck in COCO;
the default Car class excludes trucks. Excluding a category removes its proposals,
not the object from the image, so inspect the scene against your class definitions.
The sample keeps those source labels intact rather than quietly correcting them.

## Credits and provenance

The photographs are by **infomatique** and **Alan Stanton**, each under
**CC BY-SA 2.0**. COCO annotations are **CC BY 4.0**. Full photo titles, source pages,
license links and the packaging changes are in [CREDITS.txt](CREDITS.txt).
[provenance.json](provenance.json) records pinned download URLs and SHA-256 hashes.
The original JPEG bytes and all 19 selected annotation records are preserved.

## Rebuild

Python's standard library is enough. From the repository root:

```sh
python3 examples/street-scenes/build.py --output /tmp/street-scenes.zip
```

The script reuses the original JPEG members in the checked-in ZIP, along with
[annotations.json](annotations.json) and `CREDITS.txt`. It performs no download
and verifies the source hashes before writing a byte-for-byte reproducible ZIP.

To build without the existing ZIP, obtain the two JPEGs from the pinned URLs in
`provenance.json` and pass their directory:

```sh
python3 examples/street-scenes/build.py --image-dir /path/to/source-jpegs --output /tmp/street-scenes.zip
```

An existing output with different bytes is never overwritten.
