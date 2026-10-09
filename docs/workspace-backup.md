# Workspace backup and recovery

[Documentation](README.md)

A workspace archive preserves the stored state of the local workshop. It is
different from a COCO export, which contains a frozen dataset, and an experiment
report, which presents selected evidence. Transfers perform no model inference,
training, annotation or external request.

## Create a backup

Use **Workspace backup** under **Storage**, including with no flight sessions.
The preview shows record counts, file categories, bytes to copy and disk space.
It includes saved changes only; finish editing annotations or notes first.

The source must have no queued or running jobs. Creation rejects a request while
another HTTP mutation is in progress. Once admitted, it blocks new workspace
mutations until it finishes, fails or is cancelled; reads remain available.
SQLite's backup API produces a consistent database snapshot. File identities,
sizes and checksums are checked while copying, and changes to the database or
file inventory cause failure instead of publishing a misleading archive. Avoid
changing workspace files with other tools during backup.

Transfers run in the background. Closing the dialog does not cancel an operation.
Reopen it to follow progress, cancel or download a completed archive. The browser
manages the download without loading the ZIP into a JavaScript Blob. Cancelling
a browser download does not remove the saved archive. The dialog can remove
completed backup or inspection files when nothing is using them; it never deletes
a restored workspace.

## Included state

- SQLite records: sessions, annotation revisions and suggestions, model outputs,
  jobs, training and evaluation histories, references and experiment reports.
- Original media, extracted images and imported COCO source material.
- Frozen dataset manifests and their images.
- Temporal sequence manifests, reference revision histories and temporal dataset
  versions, with the original videos and checksummed extracted frames they use.
- Temporal detector configurations, complete saved frame outputs, execution
  receipts and continuation histories, including partially calculated caches.
- Benchmark reference manifests and copied images, frozen configurations, trial
  outputs, correction revisions and recorded timer segments.
- Local detector weights, trained checkpoints and receipts.
- Portable pipeline bundles, including their saved jobs and exact ZIP bytes.
- Annotation and video-review previews, saved report images and job logs.
- Ollama blobs, manifests and metadata stored inside the workspace.

Only recognized IRIS artifacts are included. Locks, SQLite journals, temporary
uploads, unfinished files and previous workspace-transfer archives are excluded. Unknown
files, source code, software environments and configuration files are not a
substitute for installing the application on a new machine. Models outside the
workspace are not copied. API credentials supplied through environment variables
are not included.

The archive retains notes, prompts, model responses, recorded provenance and logs.
It is private project data, not a redacted publication format. No encryption or
remote backup service is implied by creating this local ZIP.

## Inspect and restore

Upload an IRIS archive in **Restore**. The file goes only to the local server.
Inspection checks every member, its size and SHA-256, database integrity and
references between records and files. Model bytes are read as data; checkpoints
are not loaded or executed.

After inspection succeeds, review the summary and enter a new folder name using
letters, digits, hyphens or underscores. The interface displays the destination
beside the current workspace and requires confirmation. The archive identity is
checked again, files are copied into a private staging directory, and the directory
is published only after validation succeeds. An existing file, directory or
symbolic link at the destination is never replaced, including one created while
restoration is running.

The active workspace stays open. The completion view gives the restored path and
a command to start a separate server. On another machine, install IRIS and any
optional model runtime first. Configure the annotation provider separately; if
using restored Ollama files, point that service at the restored model directory.
Original job statuses are retained, and restoration never resumes work itself.

Temporal detector caches remain readable without their original weights or model
runtime. Archives validate their frozen configuration, frame hashes, execution
receipts and attempt lineage as data. Continuing an unfinished cache separately
requires its source media, verified weights and compatible execution environment;
restoring the saved outputs does not establish that compatibility.

Tracking cost jobs retain complete timing samples, hardware declarations and
local/imported origin. Archives check these records against their frozen source
and profile without executing the optional model or tracker runtime. Importing
or restoring a remote measurement does not authenticate its declared execution.
See [tracking cost measurements](tracking-cost.md).

Bounded [profile studies](tracking-studies.md) also retain their frozen dataset,
reference pins, profiles, complete cached-input replays and computed results.
Archive validation checks that reserved test entries were excluded and recomputes
the saved quality/cost summaries without loading a tracker or detector.

[Selected-object scenarios](selected-object.md) retain their source, initial
observation, optional release, policy and explicit reference identity. Archive
validation reproduces their state transitions and quality metrics from saved
observations without executing optional inference or tracking runtimes.

[Portable pipeline bundles](pipeline-bundles.md) retain their saved jobs and exact
ZIP bytes. Archive validation checks their inventories and source bindings
without loading checkpoints.

The CLI provides `iris workspace backup`, `inspect` and `restore`. CLI backup
requires the source server to be stopped and uses its directory lock. Inspection
and restoration need no running server. The destination's parent must exist.

## Format and limits

The format is `iris-workspace-archive-v1`: an uncompressed ZIP64 archive containing
`manifest.json` and files at their original relative workspace paths. The manifest
records format and application versions, database schema, time, table counts and
each payload file's size and SHA-256. It does not hash itself. A separate SHA-256
identifies the whole completed ZIP.

This version supports SQLite schemas 12 through 22 and Linux publication semantics.
Each schema is checked against its own expected tables, columns, indexes and
references. Restoration preserves the archived database and files without
migrating them. Opening an older restored workspace in the current application
then migrates it to schema 22, assigning pre-project work to **Default project**,
pinning historical frames to the original class definitions, and adding empty
tables introduced by subsequent versions, including temporal records, detector
caches and pinned tracking quality reports, without rewriting saved annotations
or results.
Archives with unsupported schema versions are rejected. Backups include all projects;
the current project selection never limits their contents. Symlinks, special files, encrypted or
compressed members, duplicate or unsafe paths, unexpected files, bad hashes and
inconsistent references are rejected. Hashes check integrity, not authorship.

The archive and total payload are each limited to 64 GiB, with at most 100,000
payload files and a 16 MiB manifest. Copying is bounded and streamed. Creation
needs space for the archive and a database snapshot. Receiving an upload can
temporarily require two archive copies; restoration also needs space for the
unpacked workspace. The UI and services check free space.

If IRIS stops during a transfer, its receipt becomes **Interrupted** on restart;
the operation is not retried automatically. Interrupted staging files may need
manual removal after checking that no transfer is using them. Only completed,
verified archives should be used for recovery. Keep a separate copy of important
archives: a ZIP on the same disk does not protect against losing that disk.

## Verification scope

Tests use generated media, real SQLite snapshots and archive files. They verify
round-trip records and bytes, database references, concurrent writes, destination
races, cancellation, corruption, path validation, limits and operation recovery.
Browser checks cover desktop and mobile flows. Saved CPU fixture results can be
transferred without executing their models again. These checks establish data
recovery, not detector quality on real flights or runtime readiness on another
machine.
