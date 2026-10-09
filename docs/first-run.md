# Try Iris on two street scenes

Start with two photos and their existing boxes. You will import them, review an
annotation and, optionally, compare two detectors on the same images.

The sample is included in the repository. Import and annotation work without a
GPU, model weights or an API key. The model comparison uses an optional CPU
installation and downloads about 22 MB of weights, plus the PyTorch runtime.

## 1. Open a separate workspace

Follow the [installation instructions](../README.md#getting-started), then stop
Iris with **Ctrl+C** if it is running. From the repository directory, start:

```sh
uv run iris --data-dir .iris-demo
```

Open **http://127.0.0.1:8000** and keep **Default project**. It already has
**Person** and **Car** classes. This tour saves its work in `.iris-demo/`,
separately from the usual `.iris/` workspace.

## 2. Import the sample

In the left sidebar, click **Import annotated dataset**. Choose
[`examples/street-scenes/street-scenes.zip`](../examples/street-scenes/street-scenes.zip)
from your clone, then **Preview archive**. Keep the ZIP intact; Iris opens it.

You should see **2 images and 19 source annotations**. The archive uses COCO,
a format for storing images and their boxes. Choose these mappings:

| Source category | Iris mapping |
| --- | --- |
| person | Person (person) |
| car | Car (car) |
| bus, truck, traffic light | Exclude this category, for each one |

This tour reviews people and passenger cars. Buses and trucks stay outside
those classes; excluding them does not turn them into cars. The original
source annotations remain recorded with the import.

Under **Source and grouping**, use:

| Field | Value |
| --- | --- |
| Session name | `Street scenes` |
| Scene group | `First-run demo` |
| Source location | `https://huggingface.co/datasets/LibreYOLO/coco-val2017-mini500` |
| License or usage terms | `Photos: CC BY-SA 2.0. COCO annotations: CC BY 4.0.` |
| Original dataset split | Validation (saved in the archive) |
| Attribution | `Photos by infomatique and Alan Stanton; annotations by COCO Consortium. Full titles, links and terms in examples/street-scenes/README.md.` |

The [sample credits](../examples/street-scenes/README.md) identify each photo.
The two scenes are grouped here for a short tour; this is not a training split.

Check the source-and-mapping confirmation, then click **Import for human review**.
The import creates the session and selects both images. You should have
**11 review proposals**, with 8 annotations excluded by the mapping.

## 3. Review an annotation

Click **Open in annotation**. The first image shows a street in Clongriffin.
Its three car boxes are **Imported annotation** proposals, waiting for review.

1. Click **Review pending proposals** to reach the proposal cards. Use
   **Accept proposal** for a useful box or **Reject proposal** for a wrong one.
2. Select an accepted box on the image and drag a corner to adjust it. **Focus
   view** and the zoom controls help with small cars. Use **Draw box** if a
   person or passenger car is missing.
3. Enter your name in **Reviewer**. **Save draft** keeps unfinished work. Use
   **Validate frame** only after checking the whole image and deciding on every
   proposal. **Validate & next** moves to the second image as well.

Imported labels can be incomplete or imprecise. On the first image, inspect
whether the leftmost car's box fits the car, and check the class definitions
before adding other vehicles. Validation records your review, not just an import.

You can stop here: the images, proposals and saved edits stay in this workspace.

## 4. Compare two models on CPU

Stop Iris with **Ctrl+C**. In the same repository directory:

```sh
uv sync --locked --extra ml
uv run --extra ml iris --data-dir .iris-demo models download \
  ssdlite320_mobilenet_v3_large yolox_nano
uv run --extra ml iris --data-dir .iris-demo
```

These commands install the CPU runtime and the two pretrained detectors. Keep
`--extra ml` on later `uv run` commands so the runtime remains installed. The
downloads do not send your images anywhere.

Reopen Iris and select **Street scenes**, then **Model comparison**:

1. Choose **SSDLite320 MobileNetV3-Large** and **YOLOX-Nano**. If their status
   has not updated, click **Refresh models**.
2. Keep **CPU** and **Full image**. Both imported images should be selected;
   if the selection is empty, select their checkboxes in **Data intake**.
3. Click **Run comparison** and wait for it to finish. The first run also loads
   the models, so its duration depends on your computer.
4. In **Saved comparisons**, choose **Class → car** and use the
   frame arrows to inspect both images. Move the confidence slider to see which
   boxes each model keeps or hides.

Look for small cars one model misses, boxes around the wrong object, and boxes
that fit poorly. More detections do not automatically mean a better model.
This view compares predictions visually; it does not score them against your
reviewed labels. Saved comparisons remain available after restarting Iris.

These photos come from COCO validation data. They are a small walkthrough, not
evidence of how either model will perform on your own recordings.

## Where to go next

Read [From Iris to Argos](iris-to-argos.md) for a real example of training a
model, exporting it and using it in another application. For your own data,
continue with [annotation](annotation-editor.md),
[dataset creation and export](dataset-export.md) and [evaluation](evaluation.md).
