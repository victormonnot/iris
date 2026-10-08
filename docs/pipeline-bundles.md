# Portable detector and tracking bundles

IRIS 0.55 packages a saved detector recipe, its unchanged checkpoint, a tracker
profile and an optional selected-object recovery policy into a versioned local
archive. A consumer can inspect the package without IRIS, detector dependencies,
a workspace or network access.

This step defines and validates the exchange format. The package does not yet
provide the standalone video/frame tracking runtime planned for the next step.
An existing detector-only export remains a separate workflow with its own runner.

## Prepare a bundle

In Studio's Tracking workspace, choose an exact profile from a completed visual
comparison or profile study. Choose the detector target family, CPU or NVIDIA
CUDA, and optionally a completed selected-object scenario from the same source.
The optional scenario supplies the guarded geometry policy's settings. It does
not supply an initial person, live track number, reference identity or automatic
selection command.

Preview checks the frozen source and local checkpoint, then reports the contents
and compatibility constraints. Changes to the request invalidate that preview.
Packaging requires an explicit action and retains a saved attempt in Tasks.
Opening a source or reading a saved bundle never starts inference or packaging.

The target device is independent of the training and recorded source devices.
Packaging copies the original checkpoint bytes; it does not convert a model,
install dependencies, download weights or test the target hardware.

## Supported detector and tracking contracts

The first format supports full-image native PyTorch checkpoints for:

- SSDLite320 MobileNetV3-Large.
- Faster R-CNN MobileNetV3-Large 320 FPN.
- YOLOX-Nano.

Official checkpoints and IRIS-trained class heads retain their distinct loading
and class contracts. Torchvision and trained YOLOX checkpoints contain a state
dictionary; official YOLOX checkpoints contain that dictionary under `model`.
Packaging records this expectation without deserializing the checkpoint.

Tiled recipes are rejected. ONNX and TensorRT are not silently substituted for a
native checkpoint. The existing [YOLOX ONNX export](yolox-onnx.md) remains available
as a separate model-only format; it is not a temporal pipeline parity result.

The bundle freezes preprocessing, native filtering, the saved score floor,
class definitions and model bytes. Its class mapping distinguishes neural output
indices, native labels and the output category IDs consumed by the tracker.
These differ for official YOLOX and some trained taxonomies. Consumers must use
the mapping rather than assuming that a person always has class index zero or one.

The tracker is either ByteTrack or BoT-SORT without learned ReID. Its profile must
refer to classes produced by that detector. The saved detector score floor must
not exceed the tracker's low threshold. Increasing sensitivity cannot recover
predictions already removed by native filtering, NMS or the detector's output cap.

## Coordinates, time and state

Measured boxes use `xyxy` pixels in the original oriented image, with exclusive
right and bottom edges. Tracker predictions remain separate from measured
observations. Optical-flow camera compensation requires the corresponding
original image in BGR order; resized detector tensors are not a substitute.

The native tracker advances once per available processed frame. Its buffer is
measured in updates, not seconds. Missing source frames do not create synthetic
empty updates. Source indices and timestamps must retain their meaning, and a
new sequence resets the tracker. Track numbers are local association outputs,
not persistent physical identities across resets or recordings.

The optional selection policy preserves its update and source-time limits,
geometry gates and consecutive-source-frame recovery confirmation. A consumer
must initiate selection explicitly. Ambiguous candidates, release and expiry
retain their separate states. Geometry can confuse similar objects and is not an
appearance-based identity guarantee. See [selected-object semantics](selected-object.md).

## Validation and provenance

The archive records its schema, hashes, sizes, source identities, producer
version, detector recipe, tracker implementation provenance and source runtime
versions. It includes upstream license notices and distinguishes code licensing
from the conditions applicable to weights and training data. Packaging does not
grant additional rights or resolve those conditions for a consumer.

Inspection validates the supported format and compatibility rules, exact file
inventory and file contents. It rejects unsupported members, duplicate or unsafe
paths, symbolic links, malformed JSON and mismatched hashes. Checkpoint bytes are
hashed as data; inspection never imports model code or unpickles weights.
Hashes detect changes relative to the manifest, not the identity of its author.

Every bundle is **experimental**. Structural compatibility and intact bytes do
not establish target execution, output parity, speed, tracking accuracy or
independent qualification. A CPU/CUDA target declaration is not a measurement on
that device, and CUDA support does not establish compatibility with an arbitrary
embedded board. Those checks remain separate evidence.

The package includes model bytes and configuration/provenance metadata. It does
not include source videos, extracted images, human annotation payloads or API
credentials. Its source identifiers and hashes link it back to the saved IRIS
work; they do not make the package an independent evaluation report.

## Inspect a completed package

Download the ZIP from the saved bundle. With IRIS installed, inspection does not
open or modify a workspace:

```sh
iris pipeline inspect /path/to/pipeline.zip
```

To inspect it on a machine without IRIS, copy `inspect.py` and the `iris_bundle`
directory from a trusted generated package into the same directory, then run:

```sh
python inspect.py /path/to/pipeline.zip
```

Both commands inspect the ZIP as data and report its manifest and checksums.
The copied inspector needs only the Python standard library. It neither imports
nor executes code from the ZIP it is inspecting. Inspecting an externally supplied
package with the installed IRIS command avoids running that package's scripts.

## API and saved jobs

All endpoints use the owning `project_id` query parameter:

| Request | Purpose |
| --- | --- |
| `GET /api/temporal/pipeline-bundle-status` | Read supported format, target families and limits |
| `GET /api/temporal/pipeline-bundle-sources` | List exact saved profiles and matching optional policies |
| `POST /api/temporal/pipeline-bundles/preview` | Verify the source, weights and compatibility without creating a job |
| `POST /api/temporal/pipeline-bundles` | Queue the request with its `expected_fingerprint` |
| `GET /api/temporal/pipeline-bundles` | Read project history |
| `GET /api/temporal/pipeline-bundles/{id}` | Read a saved attempt and completed bundle manifest |
| `GET /api/temporal/pipeline-bundles/{id}/download` | Download the completed ZIP |
| `GET /api/temporal/pipeline-bundles/{id}/manifest` | Download its manifest JSON |

Requests contain `name`, an exact `source` descriptor, `target_device` (`cpu` or
`cuda`), and `selection_id` (a matching completed scenario or `null`). The source
contains `kind` (`comparison` or `study`), `job_id`, `sequence_id` and
`profile_sha256`. Saved bundles open with
`?project=PROJECT_ID&pipeline_bundle=JOB_ID`.

The `iris-pipeline-bundle-v1` archive contains `manifest.json`,
`detector/model.pth`, `README.md`, the inspector modules and license notices.
The manifest hashes every other file. Checkpoints are capped at 1 GiB, individual
text resources at 2 MiB, and the complete archive at 1 GiB plus 32 MiB. The format
uses uncompressed ZIP entries with an exact inventory; it rejects ZIP64,
encryption, unsupported member types and extraneous or hidden data.

A worker verifies the copied bytes before publishing the file and its successful
job receipt together. Cancellation or failure leaves no successful partial bundle.
Reading historical packages validates their frozen inventory; a later IRIS update
does not require old inspector files to match the newly installed source code.
Workspace backup and restoration preserve completed packages and their evidence.
