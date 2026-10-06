# Annotation editor

Open **Annotation**, choose a frame selected in **Data intake**, then draw or
review boxes. The class selector and definitions use the immutable class version
saved for that image, including custom classes. Coordinates always refer to pixels
in the original image, including when the view is zoomed or moved. Drawing,
changing classes, editing coordinates, saving drafts and validating images work
without any detector, provider, download or network connection.

## Generate and review proposals

Open **Generate proposals** and choose the open image or check a batch of up to
25 selected images. Choose a ready local detector, proposal score threshold and
full-image or tiled inference. **Preview proposals run** checks saved revisions,
image hashes, model identity and supported classes; it performs no inference.
Inspect excluded images, unsupported classes and the planned detector passes,
then explicitly choose **Create proposals**. No weights are downloaded here.

A trained checkpoint must match each image's saved class definitions. An official
detector uses explicit COCO mappings and can cover only part of the image's
taxonomy; the preview names uncovered class IDs. Inspect those classes manually.
Changing images or inference settings requires a fresh preview. Unsaved edits on
an included frame must be saved or discarded first.

The detector's raw output and separate proposals are retained in **Saved proposal
runs**. Per-image outcomes distinguish proposals ready for review, no proposals,
changed saved inputs, invalid output and interrupted processing. Partial results
remain available. **Job details and cancellation** opens the durable job record;
it does not start another run. If a creation response is lost, IRIS searches saved
history for the same preview fingerprint and never automatically repeats the POST.

You can also import proposals from an existing compatible **Saved detector output**.
For each proposal, accept it, reject it, or accept and correct its box/class in the
editor. Corrections are saved as human decisions. No inference result, score,
accept action or empty output validates an image automatically. Inspect the whole
image for missing targets and draw any missing boxes yourself.

**Multimodal review** and **Local batch review** examine existing candidate boxes;
they do not generate new boxes. Those reviewers currently support the original
person/car definitions. See [local review batches](annotation-batches.md).

## Review queue and provenance

Filter the queue by saved review status, low scores or uncertain recommendations,
or possible omissions. These are hints to prioritize inspection, not proof of an
error or accuracy measurements. A score below 0.5 is specific to its detector;
scores from different providers are not interchangeable. An empty compatible
detector output does not establish absence. Missing or partially mapped outputs
are not evidence that all target classes are absent.

**Previous**, **Next** and the frame selector follow the current queue filter.
**Validate & next** saves an explicit human validation and advances to the next
frame needing review in that filter. Changing a filter keeps the open frame and
unsaved edits visible, even if that image no longer matches. Resolve all pending
proposals and enter a reviewer name before validation. The proposal list can show
only pending decisions, low scores or uncertain recommendations; hidden pending
proposals still prevent validation.

The **Review pending proposals** button beside the save controls shows the total
number of unresolved proposals. It switches the list to pending decisions and
moves keyboard focus to the first available review action, including proposals
hidden by another filter. Opening this list does not accept or validate anything.

Each proposal has expandable provenance. **Inspect saved inputs and raw outputs**
in the proposal run shows the original detector output, class mapping and input
snapshots. The editor's revision history preserves saved boxes, decisions,
reviewer and definitions; **View model review records** exposes saved candidate
reviews and their raw responses. Inspecting a record never replaces current edits.
When the project's classes change, adopting them for an older frame is an explicit
action that creates a draft; earlier revisions keep their original definitions.

## Inspect small objects

- **Fit** shows the whole image. Blank margins may appear to preserve its aspect
  ratio; drawing starts only inside the image.
- **1:1** displays one image pixel per CSS pixel. It preserves the view center
  where possible and centers images smaller than the canvas.
- **Focus selected** centers the selected label and zooms around it with some padding.
  Select a label from the list first if it is too small to click accurately.
- **+ / −** zoom around the view center. With the canvas focused, the mouse wheel
  zooms around the pointer. Outside the focused canvas, the wheel scrolls normally.
- **Pan**, **Space + drag**, or the middle mouse button moves the view. On touch
  screens, choose **Pan** and drag with one finger. Pinch zoom is not supported.

Resize handles and label text retain their screen size. Zooming, panning and
selecting a label do not modify annotations or invalidate an API review preview.

## Correct and undo

Drawing, moving, resizing, changing a class, applying coordinates, deleting a
label, accepting or rejecting a proposal, and editing review text can be undone.
A complete drag counts as one action. **Escape** cancels the current drag.

The editor keeps up to **100 local edits** for the current frame. Undo restores
both the labels and their proposal decisions; redo restores the undone action.
Making a new edit after undo discards the redo branch. Returning to the loaded
revision clears the unsaved-change indicator.

A successful save, an explicit reload, a new server revision loaded while the
editor is clean, or a frame change starts a fresh local history. A failed save or
revision conflict preserves unsaved edits and their history. An unchanged
background refresh preserves the current view and redo history. Local undo does
not change the immutable revisions already saved on the server.

Use **Save draft** to persist work in progress. **Validate frame** and
**Validate & next** remain explicit human decisions; undo and redo never validate
an image. A saved negative image must still be explicitly validated as empty.

## Keyboard shortcuts

Shortcuts work inside the editor. Text fields keep their native editing shortcuts.

| Key | Action |
| --- | --- |
| `V` / `B` / `H` | Select / draw / pan tool |
| Hold `Space` and drag | Temporarily pan |
| `+` / `−` | Zoom in / out |
| `0` / `1` | Fit / 1:1 |
| `F` | Focus selected box |
| `Ctrl` or `Cmd` + `Z` | Undo local edit |
| `Ctrl` or `Cmd` + `Shift` + `Z` | Redo local edit |
| `Ctrl` or `Cmd` + `Y` | Redo local edit |
| `Delete` / `Backspace` | Delete selected label |
| `Escape` | Cancel active drag |

The shortcut reference is also available directly below the canvas.
