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

## Street-scene photographs and annotations

The two photographs are **“Clongriffin” by infomatique** and **“Damaged Traffic
Light - Junction of High Road & White Hart Lane” by Alan Stanton**, under
Creative Commons Attribution-ShareAlike 2.0 Generic (CC BY-SA 2.0). The JPEGs
retain their original bytes. The PNGs are Iris's RGB-normalized display copies,
at the original dimensions, with no boxes or retouching baked into them; they
retain the same photo credits and licence.

Original COCO annotations are attributed to the **COCO Consortium** under
Creative Commons Attribution 4.0 International (CC BY 4.0). They enter the demo
as proposals for human review.

[CREDITS.txt](seed/CREDITS.txt) contains each photograph's original URL, author
link, licence link, annotation attribution and packaging changes.
[provenance.json](seed/provenance.json) records source provenance and file
hashes. Keep both with the sample media. These licences apply separately from
the Iris code licence.

## Link preview image

[iris-demo.png](social/iris-demo.png) is a screenshot of the real demo entry
screen, including blurred views of the sample photographs. Those photographic
adaptations retain the CC BY-SA 2.0 licence and the authors and source credits
listed in [CREDITS.txt](seed/CREDITS.txt). The interface retains its MIT licence.

## Recorded model outputs

The example includes saved CPU predictions from SSDLite320 MobileNetV3-Large
and Faster R-CNN MobileNetV3-Large 320 FPN. Their identities, checkpoint hashes,
runtime metadata and measured timings are preserved in
[fixture.json](seed/fixture.json). These are recorded examples; the static
artifact includes no model weights, Python ML runtimes or vendored inference
libraries, and it runs no new inference.
