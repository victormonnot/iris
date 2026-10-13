# Iris browser demo: licences and attribution

This file accompanies the static demo's versioned assets. The relative links
below resolve from that asset directory.

## Iris interface and demo code

Iris is copyright © 2026 Victor Monnot and distributed under the
[MIT licence](LICENSE). Keep that copyright and licence notice with copies of
the software.

## Bundled fonts

The interface serves these fonts locally under the SIL Open Font License 1.1:

| Font | Copyright | Included licence |
| --- | --- | --- |
| IBM Plex Sans | © 2017 IBM Corp. Reserved Font Name: Plex. | [IBM-Plex-Sans-OFL.txt](static/fonts/IBM-Plex-Sans-OFL.txt) |
| IBM Plex Mono | © 2017 IBM Corp. Reserved Font Name: Plex. | [IBM-Plex-Mono-OFL.txt](static/fonts/IBM-Plex-Mono-OFL.txt) |
| Marcellus | © 2012 Brian J. Bonislawsky DBA Astigmatic (AOETI). Reserved Font Name: Marcellus. | [Marcellus-OFL.txt](static/fonts/Marcellus-OFL.txt) |

Font binaries are unmodified. The accompanying
[font provenance](static/fonts/README.md) records their upstream sources,
versions and hashes. Preserve the three font licence files with the WOFF2 files.

## FPV images and human reference annotations

The two camera images and manual person boxes belong to **Victor Monnot**.
They are published by their author for this Iris demonstration, with all rights
reserved. They are separate from the Iris MIT software licence; no general
redistribution or training-data licence is granted.

JPEGs retain the original camera bytes. PNGs are RGB-normalized display copies
at the original dimensions, with no crop, drawn boxes or retouching baked in.
Human source boxes enter the visitor workspace as proposals awaiting review.

[CREDITS.txt](seed/CREDITS.txt) records the author, source and use terms.
[provenance.json](seed/provenance.json) records original image hashes and the
frozen human reference. Keep both with the sample media.

## Link preview image

[iris-demo.png](social/iris-demo.png) captures the real demo entry screen,
including blurred views of the FPV images. Those images retain Victor Monnot's
copyright and the terms in [CREDITS.txt](seed/CREDITS.txt). The interface retains
its MIT licence.

## Recorded model outputs

The example preserves selected saved predictions from original YOLOX-Nano and a
person-only version trained in Iris for 800 steps. The display comparison joins
two existing evaluations; it does not represent a new joint inference run.
Model identities, checkpoint hashes, source evaluation IDs, runtime metadata and
unchanged per-image CUDA timings appear in [fixture.json](seed/fixture.json).

The saved model source identifies the
[YOLOX Apache 2.0 licence](https://github.com/Megvii-BaseDetection/YOLOX/tree/419778480ab6ec0590e5d3831b3afb3b46ab2aa3/LICENSE).
The static artifact includes no model weights, Python ML runtimes or vendored
inference libraries, and it runs no new inference. Selected development images
are not an independent model-quality benchmark.
