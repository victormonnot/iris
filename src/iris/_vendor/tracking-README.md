# Pinned tracking sources

These isolated packages contain the official association implementations used
by IRIS's optional temporal tracker adapters. They do not import ARGOS, a
detector, model weights, Torch, or FastReID when learned ReID is disabled.

| Algorithm | Official repository | Pinned revision | License |
| --- | --- | --- | --- |
| ByteTrack | https://github.com/FoundationVision/ByteTrack | `d1bf0191adff59bc8fcfeaa0b33d3d1642552a99` | MIT, Yifu Zhang |
| BoT-SORT | https://github.com/NirAharon/BoT-SORT | `251985436d6712aaf682aaaf5f71edb4987224bd` | MIT, Nir Aharon |

`tracking-provenance.json` records the official paths and SHA-256 digests, the
included source digests, and license and patch digests. Each package preserves
its upstream license and an exact `adaptations.patch`. Official pinned files
were fetched and their digests verified before reproducing the small adaptations
previously exercised in ARGOS. IRIS has no runtime dependency on that project.

Adaptations are limited to:

- Relative imports for independent packages.
- NumPy 2 compatibility: `np.float` becomes `float`; `np.float_` becomes
  `np.float64` without narrowing precision.
- Removal of unused Torch imports; optional FastReID and debug Matplotlib
  imports stay inside their respective branches. IRIS disables learned ReID.
- A `detection_index` passed through native score filtering, creation,
  update and reactivation. It preserves exact input-row provenance without
  influencing association costs or Kalman state. Adapter output can therefore
  distinguish original measured boxes from internally predicted geometry.
- Ruff suppression for upstream formatting and unused imports, plus empty
  package initializers. These do not alter execution.
- GMC status fields report initialization, estimation, the native identity
  transform on insufficient matches, or explicitly disabled compensation.
  Instrumentation changes neither the native transform nor failure handling.

No association, assignment, Kalman filter, score boundary, expiry order, or
GMC algorithm is replaced by a local approximation. Install the optional
`tracking` extra for `scipy==1.17.1`, `lap==0.5.12`, and
`cython_bbox==0.1.5`. Existing IRIS NumPy and headless OpenCV dependencies
supply the remaining runtime. `cython_bbox` may require a C compiler when a
compatible wheel is unavailable. Tracker imports are deferred until requested.

## Native semantics the adapter must preserve

The source is class-agnostic: separate native state is required for each
detector class. Native track counters are global to each package. An adapter
must isolate counters and reset state when starting a new replay.

High and low score comparisons are strict. A score exactly equal to the high
threshold is admitted by neither association pass. ByteTrack's low threshold
is fixed at `0.1`, and its birth threshold is `high + 0.1`; BoT-SORT accepts
explicit low and birth thresholds. Their second and unconfirmed assignment
limits are fixed at `0.5` and `0.7`. ByteTrack returns confirmed tracks only;
BoT-SORT also returns unconfirmed tracks.

Native memory is counted in update calls, using
`int(frame_rate / 30 * track_buffer)`. A missing detection on an analyzed frame
requires an empty update; an unanalyzed source frame is not evidence of an
empty scene. Native lost-track expiry follows association, so the configured
buffer is not a guarantee that a later match is impossible. This is not a
wall-clock memory duration.

BoT-SORT sparse optical flow GMC consumes original BGR pixels and uses no
detection mask in this revision. Its first update and insufficient matching
points can yield the native identity transform. Featureless input, invalid
optical flow, or a failed affine estimate can instead raise. These conditions
must not be silently represented as successful camera compensation. Explicit
`none` GMC is a separate configuration, not an error fallback.
