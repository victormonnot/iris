# YOLOX-Nano ONNX export

[Documentation](README.md)

IRIS exports a trained YOLOX-Nano checkpoint through **Dataset & Training →
Model exports**. Select a completed full-image evaluation, one to eight reference
images and the CPU destination. Training and reference evaluation may use CPU
or CUDA. Install export dependencies explicitly in the IRIS environment:

```sh
uv sync --extra ml --extra onnx
```

For an existing CUDA environment, preserve its CUDA Torch/Torchvision builds and
install the pinned `onnx==1.20.1` package separately; the default `ml` extra uses
the CPU index. The destination needs only the packages in the bundle's
`requirements.txt`, not PyTorch or an IRIS workspace.

Creating this export is a background conversion job. It freezes the source
checkpoint, class contract, evaluated image bytes, prediction records and runner
files. It exports ONNX opset 11, checks the graph and compares raw outputs from
PyTorch CPU and OpenCV CPU on every bundled image with fixed `rtol=0.001` and
`atol=0.001`. An unsuccessful conversion is not published. Cancellation preserves
the source artifacts and does not publish a partial bundle.

The conversion check is **numerical agreement of raw tensors**, not exact saved
prediction parity or evidence of detection quality. The evaluation may have run
on CUDA, and NMS/backend differences can change output coordinates, scores or
ordering. The standalone measurement records exact agreement separately, keeps
failed results and does not widen its comparison tolerance.

## Consumer contract

`manifest.json` declares `format: iris-yolox-onnx-v1` and
`architecture: yolox_nano`:

- Input `images`: float32 `[1,3,416,416]`, BGR values 0..255, aspect-preserving
  OpenCV linear resize, top-left placement on a 114-filled canvas.
- Output `output`: `[1,3549,5+C]`; raw YOLOX center/size grid encoding, strides
  8/16/32, followed by sigmoid objectness and class probabilities. Decode the
  grid exactly once. There is no background class.
- `classes`: ordered zero-based indices with stable class IDs, names and output
  category IDs. A consumer must explicitly identify its required categories.
- `model`: local `model.onnx` path, `size_bytes` and SHA-256.
- `source`: checkpoint hash, model/training/evaluation/dataset identities and
  dataset manifest hash. No application-specific dependency is included.
- `files`: bounded inventory with byte sizes/hashes, including runner,
  requirements, reference predictions and selected original image files.

Consumers choose their own explicitly documented output filtering. The bundled
IRIS runner retains objectness times best-class probability, score floor 0.001,
class-aware NMS 0.5, maximum 100 detections and oriented pixel coordinates.
An application's person-only filtering can produce different results; compare
its previous and new model under the **same application pipeline**.

## Run and retain evidence

Extract the ZIP into a new directory, provision its dependencies, then run:

```sh
python run.py inspect
python run.py predict image.png --output ../prediction.json
python run.py measure --repeats 3 --output ../measurement.json
```

`inspect` validates the files without importing a model runtime. `predict` and
`measure` explicitly run OpenCV CPU. Measurement uses one unmeasured warmup,
records decoded-image processing time separately from image decode and compares
the saved native predictions exactly. Exit 3 means completed measurement with an
exact parity mismatch; the report is still written and can be imported in IRIS.
Imported measurements are checked declarations, not independently authenticated
execution. Source data and previously published bundles remain immutable.

This profile does not provide ONNX Runtime CUDA, TensorRT, quantization or Jetson
qualification. Changing runtime/hardware requires its own measurements. A
conversion pass does not select or deploy a model in another application.
