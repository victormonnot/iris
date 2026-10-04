# Projects

A project keeps related sessions, dataset versions and experiments together in one
local workspace. Use the **Project** selector in the sidebar, or **New project** to
create a project with a name and optional description. Session names and scene
groups are meaningful within their project. Projects can use ordinary images and
videos; no ARGOS connection, drone metadata or special source format is required.

The supported task remains bounding-box detection. Projects start with the
`iris-objects-v1` **person** and **car** definitions. **Manage classes** publishes
immutable custom definitions for manual annotation and COCO import; see
[class versions](classes.md). Custom datasets freeze one class version with images,
reviewed labels and mappings, and can be trained, evaluated and exported as COCO.
Multimodal candidate review and disagreement ranking retain the original definitions.
Moving records between projects is not supported.

## What belongs to a project

Sessions own their source files, extracted frames, selections, annotations,
comparisons, assistance requests and batches. Dataset releases contain frames from
one project. Their training runs, trained checkpoints, evaluations, reference
history and experiment reports belong to that project too. A dataset cannot use
another project's frames or release as its parent, and a trained checkpoint from
another project is not offered for inference or training.

Official pretrained weights and provider availability are shared. One local worker
still processes jobs across the workspace; switching projects neither cancels nor
relaunches jobs. The Jobs panel shows the selected project's jobs. Workspace backup
and restoration remain whole-workspace operations and include every project.
Projects organize a single-user application; they are not separate user accounts.

Switching projects reloads the local page, clearing pending previews, dialogs and
selections from the previous view. Unsaved annotation or report edits trigger the
existing browser warning. Each project remembers its last session. An open tab
keeps its project if another tab chooses a different one. Returning to the same
project restores saved work, not unsaved edits.

## Dataset independence

Scene-group names and their split reservations are scoped to a project. Two
independent projects may each use a group called `session-one` without sharing a
reservation. Within a project, related sessions must still use the same group.

Exact image pixels retain their train/validation/test reservation throughout the
workspace, including reservations declared by COCO imports. Importing an identical
image into another project cannot assign it to a conflicting split. Parent-model
training lineage checks also remain active. Different hashes, different project
names and separate sessions do not establish scene independence; similar frames
and shared pretraining data remain limitations of these checks.

## Upgrade and recovery

Schema 13 adds a project table and ownership columns on sessions, dataset releases
and COCO imports. Existing records join **Default project**. Migration preserves
record IDs, source files, hashes, annotation revisions, dataset manifests, model
checkpoints and report snapshots. It validates foreign keys before committing and
is safe to repeat. The historical class definition remains unchanged.

Schema 14 adds immutable project class versions and pins each existing frame to
the original definitions. It preserves all previous saved fields and artifacts.

Schema-12, schema-13 and schema-14 workspace archives can be inspected and restored. Restore
writes an independent workspace and preserves the archived payload. Opening a
restored older workspace performs the same migration to the current schema.
Keep the original archive for use with its original application version; the old
application does not understand a database already migrated to schema 14.

New assistance requests use generic image/video wording and updated prompt-version
identifiers. Existing saved responses and results remain unchanged. A video-review
preview using an earlier prompt version must be prepared again before execution.

## Local API

`GET /api/projects` lists projects. `POST /api/projects` accepts `name` and optional
`description`. `GET /api/projects/{id}` retrieves a project.

Existing resource endpoints use the `project_id` query parameter, for example
`GET /api/sessions?project_id=<id>` and
`POST /api/sessions?project_id=<id>`. The same scope applies to dataset creation,
imports, histories, individual resources, images, video playback and downloads.
Clients resolving resource URLs returned inside JSON must preserve that scope.
Requests without `project_id` address **Default project**, preserving the original
API workflow. Unknown projects and records outside the selected project return 404.

System information, the project list, provider availability and `/api/workspace/`
operations do not depend on a selected project. No credentials or external service
are needed to create projects or use the manual workflow.
