# Selected-object continuity

IRIS 0.54 replays an explicitly selected object through saved tracking outputs.
The selected object has its own logical identity, separate from the track number
produced by ByteTrack or BoT-SORT. This workflow measures selection, loss,
recovery and abstention; it does not rerun the detector or tracker.

## Prepare a scenario

In Studio's Tracking workspace, choose a completed comparison lane or a profile
from a completed [tracking study](tracking-studies.md). The scenario pins the
exact source job, sequence, profile and first replay pass.

Choose a frame and click a confirmed measured observation as the initial
selection. Predicted positions, estimated tracker boxes and unassigned detections
are not selectable observations. An optional later release ends the scenario.
Selecting another object creates a separate scenario rather than silently
changing the meaning of the existing result.

The policy settings are visible before execution. An explicit preview validates
the anchor, source, settings and optional human reference; running uses that
preview's fingerprint. Opening a source, moving the playhead and reading a saved
report never launch another job.

## Two diagnostic policies

Each scenario replays the same observations through two fixed policies:

| Policy | Behavior |
| --- | --- |
| Track ID only | Follows the initially selected class and track number, with confidence, expiry and release rules. |
| Guarded geometry | Checks class, confidence, box overlap, relative center distance and area ratio, including when the track number stays the same. |

The first policy is a diagnostic comparison implemented by IRIS. It is not a
replay of another application's complete selection or control logic.

The guarded policy accepts continuous observations only when there is one
compatible candidate. No compatible observation produces a loss; several
plausible observations produce ambiguity. After an interruption, a source-frame
gap or a track-number change, recovery requires repeated compatible observations
of the same candidate on consecutive source frames. Returning with the old track
number does not bypass those checks.

Pending candidates do not renew the last accepted observation's lifetime.
Recovery has a source-time limit and an available-update limit. If timestamps
are unknown, only the update limit can apply; the report preserves that clock
limitation. Expiry ends automatic recovery for the scenario. Release is explicit
and terminal. Prediction-only frames remain unobserved.

The viewer distinguishes `idle`, `observed`, `lost`, `recovering`, `ambiguous`,
`recovered`, `expired` and `released`. `Recovered` marks a transition; subsequent
accepted observations return to `observed`. The logical selection remains the
same through a successful change of track number.

These are geometric hypotheses. A different object can occupy a sufficiently
similar position and pass the checks, even over several frames. This version has
no learned appearance or re-identification model. Conservative checks can also
reject the correct object after rapid motion or a large camera movement. Neither
policy establishes physical identity from a track number or geometry alone.

## Assess the selected object

Behavior can be inspected without a human reference. To measure correctness,
choose an immutable reference revision and its intended identity explicitly,
then map native detector classes to the frozen reference taxonomy. The selected
initial observation must uniquely match that reference identity. The selection
policy never receives or consults the reference identity.

Only human-reviewed, complete and sufficiently certain reference frames are
eligible. Missing review, unknown identity or an unlocalized occlusion does not
become a negative example. A reference seeded from tracker proposals retains its
provenance after human review.

The report distinguishes:

- A selected observation matching the intended object.
- A selection matching another known identity.
- A selected box matching no eligible reference object.
- Overlap with multiple reference identities, which remains ambiguous.
- Abstention while the target is visible, and abstention while it is absent.

Incorrect identity substitution and an unmatched box are separate findings. An
unmatched box can indicate localization error as well as an incorrect selection.
Abstention must be read alongside coverage and incorrect recovery; recovering
more often is not automatically better.

Recovery events are assessed against the target reference. Unknown intervals
and sparse source gaps prevent unsupported claims about recovery across them.
Durations use left-hand samples only over consecutive, evaluable source-frame
intervals. The final frame is not extrapolated into a duration. Nominal-FPS time
remains an estimate of media time, and caller-provided timestamps remain declared
evidence rather than certified capture measurements.

Unavailable metrics stay unavailable. Repeated tracker outputs that differed
withhold quality claims; the first-pass behavior can still be inspected. This
workflow does not add new reference images, qualify an application or turn
development footage into an independent test. Reusing recordings to adjust
settings makes them tuning evidence.

## Saved evidence and limits

The `tracking_selection` job freezes its source, policy, initial selection,
optional release, reference revision and explicit identity. Completed reports
retain both state timelines, reasons, candidates and reference-based metrics.
Historical reads and workspace transfer recompute the pure result from its
frozen inputs without requiring optional inference or tracker dependencies.

Scenarios support at most 500 available source frames. The explicit cooperative
deadline is bounded to 120 seconds, and reports are capped at 48 MiB. Cancellation,
failure or deadline expiry publishes no successful partial report. A fresh run
starts from the initial selection with no inherited state.

The results concern saved observations, with their recorded gaps and clock.
They do not replay live scheduling, capture freshness or an application's
actuation rules. Deployment bundles and an external runtime remain separate
steps; downloading evidence does not apply the policy to another project.

All API requests use the owning `project_id` query parameter:

| Request | Purpose |
| --- | --- |
| `GET /api/temporal/tracking-selection-status` | Read defaults, policy bounds and protocol limits |
| `GET /api/temporal/tracking-selection-sources` | List available frozen sources |
| `POST /api/temporal/tracking-selection-source` | Read source observations and reference choices |
| `POST /api/temporal/tracking-selections/preview` | Validate the explicit scenario without creating a job |
| `POST /api/temporal/tracking-selections` | Queue the previewed request with `expected_fingerprint` |
| `GET /api/temporal/tracking-selections` | Read project history |
| `GET /api/temporal/tracking-selections/{id}` | Read a saved attempt and its complete report |
| `GET /api/temporal/tracking-selections/{id}/report` | Download the completed review evidence |

The source descriptor contains `kind` (`comparison` or `study`), `job_id`,
`sequence_id` and `profile_sha256`. A saved report opens directly with
`?project=PROJECT_ID&tracking_selection=JOB_ID`.

A saved profile and optional selected-object settings can be packaged in a
[versioned portable bundle](pipeline-bundles.md). The bundle records compatibility
and provenance; standalone tracking execution remains a separate step.
