# Compare detectors on the same images

[Documentation](README.md)

Model comparison saves predictions from local detectors so you can inspect them
side by side. It accepts images and extracted video frames, including frames
that have not been annotated. For a short walkthrough with included photos,
use the [first-run guide](first-run.md).

Install the optional runtime and weights using [setup](setup.md) first.
The catalog includes SSDLite320, Faster R-CNN MobileNetV3-Large 320 FPN and
YOLOX-Nano, plus compatible trained checkpoints from the selected project.
Opening the comparison view does not install or download anything.

## Start a comparison

1. Select a session, then select **1–100 frames** in **Data intake**. Selection
   includes checked frames hidden by the current filters.
2. Open **Model comparison**. Under **Choose detectors**, select one or two
   ready models. **Refresh models** checks local availability after setup.
   Availability checks files and dependencies; the model loads when a run starts.
3. Choose **Device** and **Inference mode**. CPU is the default. CUDA requires
   separately installed compatible dependencies and hardware; choosing it does
   not install packages or fall back to CPU. See [compute targets](compute-targets.md).
4. Optionally enter a **Run name**. Check the displayed number of model passes
   and warmup passes, then click **Run comparison**. Work runs in the local job
   worker, and progress appears with the saved comparison and project jobs.
5. In **Saved comparisons**, use **Comparison history** to choose a run and the
   frame arrows to inspect its images. **Class** and **Display confidence** filter
   saved detections without running inference again.

The saved comparison freezes its frame selection, image hashes, checkpoint
hashes, model settings, class contracts, runtime, device and timing protocol.
Changing the session's selected frames later does not change that comparison.
Open **Run provenance and model settings** to inspect the recorded configuration.
History and saved predictions remain readable after restarting Iris, including
when the original weights or optional runtime are no longer available.

## Full images and tiles

| Inference mode | What runs |
| --- | --- |
| **Full image** | Each selected model processes the entire image. |
| **Tiled image** | Each selected model processes overlapping crops; boxes are mapped back to the original image and duplicates merged. |
| **Full image vs tiled · one model** | One checkpoint runs through both paths, producing two separate results per image. |

Tiles can preserve more pixels around a small object through a detector's input
resize. They cannot recover detail absent from the recording, and may lose
context or cut an object at a crop boundary. The paired mode changes how the
same checkpoint sees the same image; it does not retrain it or zoom the camera.

**Tile size · pixels** defaults to 640 and accepts integers from 128 to 2048.
**Overlap · fraction** defaults to 0.2 and accepts 0 to 0.5. Edge crops are
anchored to the image boundary, so their actual overlap can be larger. An image
smaller than one tile produces one crop without extra padding or enlargement;
the detector still applies its usual input transform.

The preview allows at most **64 tiles per image** and **512 detector passes per
comparison, including warmups**. Paired mode includes both the full-image and
tiled passes. Reduce the selection or increase tile size if the preview rejects
the work budget. Changing settings updates the preview before a job can start.

This comparison workflow merges boxes of the same class with non-maximum
suppression (NMS) at IoU greater than 0.5, then retains at most **300 merged
boxes**. Highest scores come first; ties preserve tile and output order.
Truncation is recorded and shown in the results. This is the comparison's merge
configuration, not a universal limit for every Iris tiled workflow: the shared
Python tiling service accepts saved limits from 1 to 1,000. The comparison HTTP
API exposes tile size and overlap, using the same merge defaults as the UI.

Each saved tiled prediction retains its original per-tile detections, crop
coordinates and timings in `metadata.tiles`, available through
`GET /api/comparisons/{id}?project_id=PROJECT_ID`. Saved boxes use original-image
pixel coordinates. Full and tiled results remain distinct even when they use
the same checkpoint. In [annotation](annotation-editor.md), choose the required
source explicitly when importing its detections as proposals.

## Saved outputs and interruptions

Every completed model/frame result is saved separately. An empty detection list
is a completed result; **Not processed** means no prediction was saved. A failed
or cancelled comparison retains completed results, which may cover only one
model or part of the frame selection. Start a new comparison to retry.

Cancellation during tiled inference is checked between crops and during merging.
An unfinished image does not publish a partial merged result; earlier completed
images remain available. Saved outputs retain all categories returned by the
detector before display filtering, with the checkpoint's class names and mappings.
Do not assume that numeric class IDs mean the same thing for every custom model.

The built-in detector adapters retain a native score floor of 0.001, NMS IoU 0.5
and at most 100 detections per detector call. Those native filters apply before
the display slider and before tiled merging. Lowering the displayed confidence
cannot recover predictions discarded by the detector. Models also have different
preprocessing and internal proposal rules; inspect their saved settings.

## Replay a video comparison

A comparison containing extracted video frames also offers **Video replay**.
Choose **Source video**, then a timeline marker or saved sample. Play the video
with **Follow saved samples during playback** enabled to advance the comparison
to the last sampled image at or before the playback position.

The video itself has no detection overlay. Boxes stay on the extracted images
below it, with their recorded timestamps. Between samples the view holds a saved
image and explains its offset from playback; it does not interpolate boxes or
create predictions for unsampled frames. Timeline markers distinguish complete,
partial and missing saved results. Still images keep the ordinary frame navigation.

Positions are approximate: extracted timestamps use the source frame index and
nominal frame rate. Variable-frame-rate video may not align with the browser's
clock. Inspect the saved image to judge its boxes. Missing footage or an unsupported
browser codec does not remove saved images or predictions. Replay starts no
transcoding, model download or inference.

## Read timing and quality separately

Each run uses batch size one, float32 and one excluded warmup. CPU execution uses
at most four Torch threads; CUDA timing boundaries synchronize device work.
The per-frame total includes image decoding and hash verification, but excludes
weight loading, warmup and database writes. Tiled totals also include cropping
and merging, while excluding progress callbacks. Tile forward times are summed.

Stage names must be read with each model's saved `timing_protocol`:

- **Torchvision detectors:** `inference_ms` includes the full model forward,
  internal resize and normalization, proposal filtering, NMS and coordinate
  restoration. `postprocess_ms` covers output transfer and JSON serialization.
- **YOLOX-Nano:** `inference_ms` includes forward execution and grid decoding,
  while `postprocess_ms` includes class scoring, clipping, NMS and serialization.

Forward time alone therefore has different scopes across architectures. These
measurements describe saved-image processing on this machine, not live-camera
latency or an exported model's frame rate. Counts, confidence and speed do not
establish detection quality. Use [Quality evaluation](evaluation.md) with a
frozen, human-reviewed validation set to measure precision, recall and AP.
