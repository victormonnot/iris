# Class versions

Use **Manage classes** in the project sidebar to define the objects you want to
annotate. Every class needs a stable ID, a display name and a written definition.
Describe what belongs to the class, what to exclude and which visible extent to
box. Class IDs use lowercase letters, digits, underscores or hyphens, start with a
letter and have at most 64 characters. `exclude` is reserved for COCO imports.
Names have at most 120 characters, definitions 2,000, and a version 1–100 classes.

Projects begin with the original Person / Car definition. The first custom version
can replace these starter classes entirely, for example with `helmet` and
`damaged_panel`. Once a custom version is published, its IDs cannot be removed or
renamed. Names and definitions can change in a new version, and classes can be added.
Every published version remains readable in the history. Concurrent publication
from another tab produces a reload conflict.

## Images and human review

Newly imported images and newly extracted video frames keep the project's class
version at intake. COCO images keep the version saved by their import preview.
Changing project classes does not rewrite images, annotations, proposals, datasets
or model results. Opening an older image shows its own definitions and indicates
when newer project definitions exist.

To use the new definitions on an older image, save any local edits, then choose
**Use current project classes** in Annotation. This creates a draft revision with
the saved boxes and their provenance, clears the reviewer and requires validation
again. Old revisions keep their original definitions. If saved boxes contain a
class absent from the target version, adoption is refused: keep the old version or
explicitly remove those boxes before adopting. No label is silently remapped.

Review the whole image, including missing objects. Accept, correct or reject each
proposal, then enter a reviewer and validate. Empty images also require explicit
human validation. Proposals from an older class version are marked; a label absent
from the current image's classes cannot be accepted. Saved decisions and source
metadata remain part of the historical record.

## Mapping external categories

COCO import mapping is explicit for every source category: choose a target class
or exclude it. The preview displays the saved target definitions; matching names
alone do not establish equivalence. Updating the project later leaves this preview
unchanged. Original source categories, boxes, mapping and target definitions remain
in the imported image's provenance. Imports create proposals, never validated labels.

An optional COCO category ID on a project class enables importing that category
from saved standard detector outputs. This mapping is a user assertion of semantic
compatibility. Leave it blank for a class that the detector does not recognize.
One COCO category can map to only one target class. This does not train or modify
the detector, and imported boxes still require human review.

## Current scope

Custom classes work for manual annotation, negatives, saved detector proposals
with explicit mappings, COCO intake, frozen dataset releases and COCO export.
The dataset builder selects one saved class version and includes only fully
validated images using that version. Other versions remain selectable; there is
no implicit relabeling or merging of different definitions. The release records
the complete class snapshot and its internal and export numeric mappings.

Training, evaluation, disagreement ranking and multimodal candidate review still
support the original Person / Car definitions. Custom releases stay available for
inspection and export, with these limits shown in the interface. Existing releases
and results remain usable. No provider or network call is required for the manual
workflow or dataset export. See [frozen datasets](dataset-export.md).

## Local API

- `GET /api/projects/{id}/taxonomies` returns the current ID and version history.
- `POST /api/projects/{id}/taxonomies` publishes `classes` with `expected_taxonomy_id`.
- Annotation GET responses contain `taxonomy`, `current_taxonomy` and `taxonomy_outdated`.
- Annotation PUT accepts `taxonomy_id` alongside `expected_revision` to reject stale editors.
- `POST /api/frames/{id}/annotation/taxonomy` accepts `expected_revision`,
  `expected_taxonomy_id` and `target_taxonomy_id`; the target must be the project's
  current version. The response is the new draft annotation.
- Historical revision responses include the definitions used by that revision.

Frame and import endpoints retain their `project_id` query parameter. Import details
include a frozen `taxonomy`; commit mappings must use its class IDs or `exclude`.
