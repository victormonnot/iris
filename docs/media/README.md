# README media

These visuals use the real Iris interface and saved project data. No model was
run to produce them, and no predictions or scores were invented.

## Model comparison

`iris-comparison.png` is a cropped screenshot of the current error viewer,
showing saved Faster R-CNN predictions on the same frame. The official model is
on the left; a version fine-tuned for 40 steps is on the right. Both find the
person, but the trained version also detects part of the camera overlay as a
person. This illustrates checking regressions, not an improvement.

The source is Victor Monnot's recorded Argos footage. Detection confidence and
matching IoU are both set to 0.5. The image belongs to the small development
validation set described in the [experiment notes](../acceptance-results.md).

## Annotation

`iris-annotation.gif` and `iris-annotation.mp4` show an annotation edit in the
real Iris interface. The demonstration was recorded in an isolated workspace
copy using a real street photo and imported annotations. Correcting a box and
adding a missing annotation were reenacted for the recording; the original saved
annotations were preserved. The starting proposals came from COCO annotations,
not AI predictions. A small pointer highlight makes the recorded mouse actions
easier to follow.

Photo: [**Clongriffin**](https://www.flickr.com/photos/infomatique/4575950642/)
by [infomatique](https://www.flickr.com/photos/infomatique/),
[CC BY-SA 2.0](https://creativecommons.org/licenses/by-sa/2.0/).
The photograph appears inside Iris with annotation overlays and viewport zoom.
The photograph and its adaptations in this demonstration are shared under
CC BY-SA 2.0.
Original COCO annotations: COCO Consortium,
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## YOLOX example

`iris-yolox.png` is a comparison figure assembled from the original image and
saved evaluation outputs. It is not an interface screenshot. Both panels show
the same complete frame from Victor Monnot's Argos recording. At a person
confidence threshold of 0.35, the original YOLOX-Nano retains no person detection
and the model trained in Iris retains one. The box comes from the saved trained
model output.

This image was already used during development and is not an independent test.
See the [YOLOX experiment](../acceptance-results.md#yolox-nano-custom-detector-accepted-by-an-external-application)
for the full data split, settings, results and limitations.
