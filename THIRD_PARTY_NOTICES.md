# Third-party code and media

Iris's original code and documentation are covered by the [MIT License](LICENSE).
Bundled third-party code, fonts, photographs and annotations keep their own
licences and credits below. The MIT grant does not relicense those materials.

## Code included in Iris

| Component | Licence | Included notices and provenance |
| --- | --- | --- |
| YOLOX model code | Apache-2.0 | [Licence](src/iris/_vendor/yolox/LICENSE), [NOTICE](src/iris/_vendor/yolox/NOTICE), including the upstream revision and Iris adaptations. Modified source files carry an Iris modification notice. |
| ByteTrack tracker code | MIT | [Licence](src/iris/_vendor/bytetrack/LICENSE), [adaptations](src/iris/_vendor/bytetrack/adaptations.patch) and [pinned provenance](src/iris/_vendor/tracking-provenance.json). |
| BoT-SORT tracker code | MIT | [Licence](src/iris/_vendor/botsort/LICENSE), [adaptations](src/iris/_vendor/botsort/adaptations.patch) and [pinned provenance](src/iris/_vendor/tracking-provenance.json). |
| Torchvision-derived training code and export attribution | BSD-3-Clause | [Licence](src/iris/_vendor/torchvision-LICENSE) and [source record](src/iris/_vendor/torchvision-license.json). |

These notices cover code, not pretrained model weights or training datasets.
Python dependencies installed separately retain their own licences; their
versions and sources are recorded in [pyproject.toml](pyproject.toml) and
[uv.lock](uv.lock).

## Fonts

Iris serves the following fonts locally, under SIL Open Font License 1.1:

- IBM Plex Sans — [OFL and copyright notice](src/iris/static/fonts/IBM-Plex-Sans-OFL.txt).
- IBM Plex Mono — [OFL and copyright notice](src/iris/static/fonts/IBM-Plex-Mono-OFL.txt).
- Marcellus — [OFL and copyright notice](src/iris/static/fonts/Marcellus-OFL.txt).

The [font source record](src/iris/static/fonts/README.md) identifies the upstream
files, checksums and reserved font names. The bundled font files are unmodified.

## Example photos, annotations and README media

The [street-scene example](examples/street-scenes/README.md) contains two photos:
*Clongriffin* by infomatique and *Damaged Traffic Light - Junction of High Road &
White Hart Lane* by Alan Stanton. Both are under CC BY-SA 2.0. Their original
COCO annotations are under CC BY 4.0. Author links, source links, licence links
and packaging changes are retained in [CREDITS.txt](examples/street-scenes/CREDITS.txt),
also included in the example ZIP.

The annotation GIF and video show the attributed *Clongriffin* photo inside
Iris. The other README figures use Victor Monnot's Argos recordings and saved
predictions. See [README media credits](docs/media/README.md) for each asset's
source, edits and applicable terms. The software licence does not grant rights
to the underlying Argos footage.

## Exported models and pipelines

New exports include the MIT notice for Iris's standalone runner code, alongside
the applicable upstream notices. Model weights and user-supplied data have
separate terms: an export does not establish permission to redistribute either.
Check the model's source references and the recorded dataset attribution.

The source distribution includes documentation media and the photo example;
the installed application wheel does not include those image files. Both retain
their applicable code and font notices. No single SPDX expression is declared
for both archives because their bundled materials differ.
