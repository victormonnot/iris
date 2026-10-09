# From Iris to Argos

[Documentation](README.md)

I wanted to improve person detection on my analog camera recordings and try the
result in [Argos](https://victormonnot.com/projects/argos/). On 7 October 2026,
I used Iris to train a YOLOX-Nano model, compare it with the original, export it
and run it through Argos's detector and tracker.

This was an offline experiment on my workstation.

## Preparing the data

I used 57 manually reviewed images containing 52 people:

- **Training:** 40 images from bedroom and morning-park recordings, including
  four empty images.
- **Validation:** 17 images from a separate courtyard recording, containing
  16 people and one empty image.

I kept recordings in separate splits. Iris saved the images and reviewed boxes
as a dataset version, giving each model the same reference for comparison.

## Training and comparing

Starting from official YOLOX-Nano weights, I tried updating just the prediction
layers for 400 steps, then the last backbone stage and detection head for 800
steps. Both trained a person-only model on an RTX 4060.

The first partial-backbone attempt failed after 15 updates: an empty image
triggered a very large gradient update. I reproduced the failure on training
data and added gradient clipping to a new recipe. It completed 800 steps;
the failed attempt remained recorded.

At confidence **0.35** and matching IoU **0.5**, the results were:

| Model | People found | False detections | People missed |
| --- | ---: | ---: | ---: |
| Original YOLOX-Nano | 15 | 0 | 1 |
| Prediction layers only, 400 steps | 15 | 0 | 1 |
| Partial backbone, 800 steps with clipping | 16 | 0 | 0 |

I kept the 800-step version as the candidate. These **17 validation images had
already been used during development**; finding all 16 people here does not
establish performance on new scenes.

![Original YOLOX-Nano misses a person that the trained model finds](media/iris-yolox.png)

*The same recorded image and confidence threshold, using saved predictions.*

## Exporting and trying it in Argos

I exported the candidate to ONNX with its classes and preprocessing settings.
Argos loaded it through its normal person-detection path and confirmed the
change from 15 people found to 16 on the same images, with no extra detections.

The conversion check passed, but exact equality with saved predictions failed
on small numerical differences, which remain documented in the
[export notes](yolox-onnx.md).

I also replayed recorded clips through Argos's detector and tracker. That checked
the integration, but without reviewed identities it could not establish better
tracking. No flight service or default model was changed.

## What I took from it

I could take reviewed data through training and into another application, while
keeping the dataset, settings and results linked. The training failure also
gave me a concrete problem to reproduce and fix.

The next checks are new difficult footage, reviewed identities and measurements
on the intended computer. The [full experiment record](acceptance-results.md#yolox-nano-custom-detector-accepted-by-an-external-application)
and [training notes](yolox-training.md) contain the technical details.
