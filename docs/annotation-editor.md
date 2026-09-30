# Annotation editor

Open **Annotation**, choose a selected frame, then draw or review boxes. The
current taxonomy contains `person` and `car`. Coordinates always refer to pixels
in the original image, including when the view is zoomed or moved.

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
