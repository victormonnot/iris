# Tracking cost on saved frames

IRIS 0.52 measures a fresh detector and one frozen tracker profile in the same
frame loop. Open a completed **Tracking comparison**, choose its ByteTrack or
BoT-SORT lane, a CPU or CUDA device, and a processing policy in **Pipeline cost**.
Each local job starts a new worker. Its complete report survives restarts and
workspace backups. Loading a report does not run inference.

This is a cost measurement on saved images. It does not measure camera capture,
video demux, transport, display latency, target-lock behavior or tracking accuracy.
The separate [quality report](tracking-quality.md) evaluates the original saved
comparison; its scores cannot be attached to a new run that skips frames.

## What is measured

The frame pipeline verifies and decodes a saved source image, runs the detector,
validates and filters its outputs for the frozen profile, prepares camera-compensation pixels
when needed, and updates the tracker. The report records the outer detector and
tracker calls as well as their internal stages. Nested stages are breakdowns of
their enclosing call, not extra time to add to it. Association and camera motion
compensation remain distinct. Source verification includes the saved-frame
database lookup. The filtering stage includes output validation and its cancellation
checkpoint. Repetition wall time also includes bookkeeping, memory sampling and
progress callbacks; these are outside the reported per-frame pipeline service time.

Setup and warmup are outside the measured frames. A fresh tracker starts each
repetition; warmup observations do not seed the measured tracks. Repetitions
reuse the loaded detector and the same available images. The report retains
sample counts, frame indices and timing distributions. The median averages the
two middle samples when needed; p95 uses the nearest-rank method. A short repeated passage
does not establish long-run or worst-case hardware performance.

The model, weights, native class contract, score floor, preprocessing and detector
recipe come from the comparison's frozen cache. Inference runs again on the
requested device with its current verified runtime. Old cached detector timings
are never added to current tracker timings to claim complete pipeline latency.
CPU and CUDA results retain separate execution signatures. CUDA execution fails
explicitly if unavailable; it does not silently switch to CPU.

## All available frames or simulated cadence

**All available frames** processes every saved frame in order, as fast as the
worker can. It does not claim that unavailable source frames were processed.

**Simulated latest frame** assumes the explicit input FPS you enter. Arrival
spacing uses original source frame indices, preserving source gaps. A virtual
single worker keeps the latest available frame when processing falls behind;
older pending arrivals are counted as dropped. The detector and tracker actually
run on each selected image, and their measured service time advances the virtual
clock. The simulation does not sleep or capture a live camera.

Reports distinguish source gaps, dropped available frames, processed frames,
queue delay, service time and simulated completion latency. A missing source
frame is not a simulated drop. The tracker receives only the processed frames,
with their original indices and timestamps. The current native trackers advance
their internal state once per update; the simulation does not invent intermediate
updates or turn their retention counters into seconds.

Pipeline throughput is processed frames divided by measured pipeline service
time. Simulated output cadence uses intervals between completed outputs: the sum
of `processed frames - 1` across repetitions divided by the sum of each
repetition's last-minus-first completion time. A single output cannot establish
a cadence. Pipeline throughput, output cadence and input FPS describe different quantities.
Read their scopes before comparing them. In particular, a high throughput on a
small cached passage is not a measured live-camera frame rate.

## Memory and hardware

Host memory records boundary samples of the worker's resident memory and a
separately labeled process-lifetime high-water mark when the platform provides
one. Boundary samples can miss short peaks. The lifetime peak includes imports,
model loading and warmup; it is not a resettable peak for the measured loop.
See the [Linux process-memory definitions](https://docs.kernel.org/filesystems/proc.html).

CUDA measurements use the actual selected device's PyTorch allocated and
reserved memory, with peak statistics reset before each measured repetition,
after warmup. These numbers include
the resident model and allocator cache; they are not total board VRAM or memory
used by other processes. CPU runs leave CUDA memory unavailable. The existing
detector synchronizes CUDA timing boundaries; GPU work must finish before its
elapsed time is recorded. See [PyTorch CUDA semantics](https://docs.pytorch.org/docs/2.10/notes/cuda.html).

Reports identify their hardware, runtime, device, frozen profile, source and
measurement protocol. Results on a desktop do not certify a laptop, Jetson or
other target. Profile search, automatic parameter tuning and the standalone
tracking deployment runtime remain later steps.

## Measure elsewhere and import

Use a compatible IRIS workspace copy with the same comparison, source images
and verified model weights. Install IRIS and its optional detector/tracker
dependencies on the target. The read-only command writes a complete report to
a new file and refuses to overwrite an existing output:

```sh
iris --data-dir /path/to/workspace tracking measure \
  --comparison-id COMPARISON_ID --lane-index 0 --device cpu \
  --repeats 2 --policy offline_all --output measurement.json
```

For a simulated 30 FPS input, use `--policy simulated_latest --cadence-fps 30`.
Choose `--device cuda` explicitly for CUDA inference. Import the JSON into the
matching comparison's cost panel. Import validates the source/profile bindings,
raw schedule and recomputed aggregates; it does not run the model or fetch files
from paths inside the report.

Imported measurements remain **declared execution**. Checksums establish internal
consistency, not that the named machine actually ran the experiment. IRIS sets
this origin itself; a report cannot grant itself local-execution provenance.
The target need not be online while the report is imported.

## Persistence

Cost runs use the existing durable jobs table with kind `tracking_cost` and
schema 22. Successful results are complete and immutable. Cancellation,
interruption and failures publish no successful partial measurement. A new run
starts a fresh worker rather than resuming partially accumulated timing samples.
Workspace transfer checks cost results without requiring a detector, tracker or
GPU on the machine performing the check.

All API calls use the owning `project_id` query parameter:

| Request | Purpose |
| --- | --- |
| `GET /api/temporal/tracking-cost-status` | Read protocol and limits |
| `GET /api/temporal/tracking-comparisons/{id}/cost-runs` | Read run history |
| `POST /api/temporal/tracking-comparisons/{id}/cost-runs` | Queue a fresh local run |
| `POST /api/temporal/tracking-comparisons/{id}/cost-runs/import` | Import `{ "report": ... }` as declared evidence |
| `GET /api/temporal/tracking-cost-runs/{id}` | Read a run and its complete report |
| `GET /api/temporal/tracking-cost-runs/{id}/report` | Download the complete JSON report |

Creation accepts `name`, `lane_index` (0 or 1), `device` (`cpu` or `cuda`),
`repeats` (1–5), `policy` (`offline_all` or `simulated_latest`) and `cadence_fps`.
The cadence is null for all-frame processing and 0.1–240 FPS for simulation.
Each repetition uses at most 500 available source frames. A run can be reopened
with `?project=PROJECT_ID&tracking_cost=RUN_ID`.
