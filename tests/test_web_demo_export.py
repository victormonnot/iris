"""The hosted demo is a reproducible, self-contained public artifact."""

import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urljoin, urlsplit

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "web-demo"
FONT_NOTICES = {
    "IBM-Plex-Sans-OFL.txt",
    "IBM-Plex-Mono-OFL.txt",
    "Marcellus-OFL.txt",
    "README.md",
}
CSS_URLS = re.compile(r"url\(\s*['\"]?([^)'\"\s]+)['\"]?\s*\)")


def load_exporter(example):
    """Keep the preview module from another test/source checkout out of this import."""
    spec = importlib.util.spec_from_file_location("iris_test_site_export", example / "export.py")
    module = importlib.util.module_from_spec(spec)
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(example))
        patch.delitem(sys.modules, "serve", raising=False)
        patch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        patch.delitem(sys.modules, "serve")
    return module


def read_files(root):
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def assert_same_files(actual, expected):
    assert actual.keys() == expected.keys()
    assert not [name for name in actual if actual[name] != expected[name]]


def snapshot(root):
    """Compare refusals without following symlinks or overlooking empty directories."""
    result = {}
    for path in (root, *sorted(root.rglob("*"))):
        name = path.relative_to(root).as_posix()
        result[name] = (
            ("symlink", str(path.readlink()))
            if path.is_symlink()
            else ("directory",)
            if path.is_dir()
            else ("file", hashlib.sha256(path.read_bytes()).hexdigest())
        )
    return result


class BrowserReferences(HTMLParser):
    def __init__(self):
        super().__init__()
        self.dependencies = []
        self.seed = None
        self.home = None

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag in {"script", "img", "source"} and values.get("src"):
            self.dependencies.append(values["src"])
        if tag == "link" and values.get("href"):
            self.dependencies.append(values["href"])
        if "data-seed" in values:
            assert self.seed is None
            self.seed = values["data-seed"]
        if tag == "a" and values.get("aria-label") == "IRIS home":
            self.home = values["href"]


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    exporter = load_exporter(EXAMPLE)
    directory = tmp_path_factory.mktemp("demo-artifact")
    first, second = directory / "first", directory / "different-path"
    manifest = exporter.export_demo(first)
    assert exporter.export_demo(second) == manifest
    files = read_files(first)
    assert_same_files(files, read_files(second))
    return manifest, files


@pytest.fixture
def source_tree(tmp_path):
    """Use the actual shared sources, with all test pollution outside the repository."""
    root = tmp_path / "source"
    example = root / "examples" / "web-demo"
    example.mkdir(parents=True)
    for name in ("export.py", "serve.py", "THIRD_PARTY_NOTICES.md"):
        shutil.copy2(EXAMPLE / name, example / name)
    shutil.copy2(ROOT / "LICENSE", root / "LICENSE")
    shutil.copytree(EXAMPLE / "seed", example / "seed")
    shutil.copytree(EXAMPLE / "social", example / "social")
    shutil.copytree(ROOT / "src/iris/static", root / "src/iris/static")
    return root, example, load_exporter(example)


def test_export_is_deterministic_and_manifest_covers_every_published_byte(exported):
    manifest, files = exported
    assert manifest["format"] == "iris-web-demo-site-v1"
    assert re.fullmatch(r"[0-9a-f]{64}", manifest["asset_version"])
    assert json.loads(files["manifest.json"]) == manifest
    assert set(manifest["files"]) == set(files) - {"manifest.json"}
    prefix = f"assets/{manifest['asset_version']}/"
    for name, expected in manifest["files"].items():
        assert name == "index.html" or name.startswith(prefix)
        assert ".." not in PurePosixPath(name).parts
        assert expected == {
            "sha256": hashlib.sha256(files[name]).hexdigest(),
            "size": len(files[name]),
        }


@pytest.mark.parametrize("mount", ["/", "/projects/iris/"])
@pytest.mark.parametrize("entry", ["", "index.html"])
def test_browser_dependencies_resolve_at_root_and_under_subpath(exported, mount, entry):
    manifest, files = exported
    base = "https://demo.example" + mount
    page_url = base + entry
    parser = BrowserReferences()
    parser.feed(files["index.html"].decode())
    assert parser.dependencies and parser.seed and parser.home
    assert urljoin(page_url, parser.home) == base

    def resolve(reference, referring_url=page_url):
        assert not reference.startswith("/")
        absolute = urljoin(referring_url, reference)
        assert absolute.startswith(base)
        parsed = urlsplit(absolute)
        assert parsed.netloc == "demo.example" and not parsed.query and not parsed.fragment
        relative = unquote(parsed.path).removeprefix(mount)
        assert relative in files, (reference, referring_url, relative)
        return relative, absolute

    needed = set()
    for reference in parser.dependencies:
        relative, absolute = resolve(reference)
        needed.add(relative)
        if relative.endswith(".css"):
            for dependency in CSS_URLS.findall(files[relative].decode()):
                path, _ = resolve(dependency, absolute)
                needed.add(path)
    prefix = f"assets/{manifest['asset_version']}/"
    assert prefix + "static/demo/entry.js" in needed
    assert prefix + "static/demo/bootstrap.js" in needed
    assert prefix + "static/annotation.js" in needed
    assert prefix + "static/comparison.js" in needed
    notices = {prefix + "static/fonts/" + name for name in FONT_NOTICES}
    assert {name for name in files if name.startswith(prefix + "static/")} == needed | notices
    seed_url = urljoin(page_url, parser.seed)
    fixture_path, _ = resolve("fixture.json", seed_url)
    fixture = json.loads(files[fixture_path])
    for reference in [fixture["credits"], fixture["provenance"], *fixture["resources"].values()]:
        resolve(reference, seed_url)
    assert len(fixture["resources"]) == 4


def test_export_preserves_seed_provenance_and_redistribution_notices(exported):
    manifest, files = exported
    prefix = f"assets/{manifest['asset_version']}/"
    seed = {
        name.removeprefix(prefix + "seed/"): data
        for name, data in files.items()
        if name.startswith(prefix + "seed/")
    }
    assert_same_files(seed, read_files(EXAMPLE / "seed"))
    provenance = json.loads(seed["provenance.json"])
    assert set(provenance["files"]) == set(seed) - {"provenance.json"}
    for name, metadata in provenance["files"].items():
        assert hashlib.sha256(seed[name]).hexdigest() == metadata["sha256"]
        assert len(seed[name]) == metadata["size_bytes"]
    assert files[prefix + "LICENSE"] == (ROOT / "LICENSE").read_bytes()
    assert (
        files[prefix + "THIRD_PARTY_NOTICES.md"]
        == (EXAMPLE / "THIRD_PARTY_NOTICES.md").read_bytes()
    )
    for name in FONT_NOTICES:
        assert (
            files[prefix + "static/fonts/" + name]
            == (ROOT / "src/iris/static/fonts" / name).read_bytes()
        )


def test_cli_needs_only_stdlib_and_defaults_to_dist_demo(source_tree, tmp_path):
    root, example, _ = source_tree
    command = [sys.executable, "-S", str(example / "export.py")]
    default = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=15)
    assert default.returncode == 0, default.stderr
    output = tmp_path / "custom-site"
    explicit = subprocess.run(
        [*command, "--output", str(output)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert explicit.returncode == 0, explicit.stderr
    assert_same_files(read_files(root / "dist/demo"), read_files(output))


def test_unreferenced_files_never_enter_the_public_artifact(source_tree, tmp_path):
    root, example, exporter = source_tree
    clean = tmp_path / "clean"
    exporter.export_demo(clean)
    for path in (
        root / ".env",
        root / "private-plan.md",
        root / "src/iris/static/internal-notes.json",
        root / "src/iris/static/unused-client.js",
        root / "src/iris/static/fonts/private-font.woff2",
        example / "private-recording.json",
        example / "seed/unlisted.json",
        example / "seed/media/unlisted.jpg",
    ):
        path.write_text("PRIVATE-CONTENT-MUST-NOT-BE-PUBLISHED")
    poisoned = tmp_path / "with-extra-inputs"
    exporter.export_demo(poisoned)
    assert_same_files(read_files(poisoned), read_files(clean))


@pytest.mark.parametrize("changed", ["app.js", "studio-tokens.css", "index.html"])
def test_shared_frontend_changes_invalidate_assets_and_replace_prior_export(
    source_tree, tmp_path, changed
):
    root, _, exporter = source_tree
    output = tmp_path / "site"
    first = exporter.export_demo(output)
    target = root / "src/iris/static" / changed
    comment = "\n<!-- Updated shared UI -->\n" if changed.endswith(".html") else "\n/* update */\n"
    target.write_text(target.read_text() + comment)
    second = exporter.export_demo(output)
    assert second["asset_version"] != first["asset_version"]
    assert not (output / "assets" / first["asset_version"]).exists()
    assert (output / "assets" / second["asset_version"]).is_dir()
    assert exporter.export_demo(output) == second


@pytest.mark.parametrize("changed", ["fixture.json", "CREDITS.txt", "media/1601.jpg"])
def test_changed_seed_is_rejected_without_touching_a_previous_export(
    source_tree, tmp_path, changed
):
    _, example, exporter = source_tree
    output = tmp_path / "site"
    exporter.export_demo(output)
    before = snapshot(output)
    (example / "seed" / changed).write_bytes(b"Unverified replacement content")
    with pytest.raises(ValueError):
        exporter.export_demo(output)
    assert snapshot(output) == before


@pytest.mark.parametrize("name", ["../private.txt", "private.txt", "media/private.txt"])
def test_seed_inventory_cannot_authorize_private_paths(source_tree, tmp_path, name):
    _, example, exporter = source_tree
    content = b"not public media"
    (example / "seed" / name).write_bytes(content)
    provenance_file = example / "seed/provenance.json"
    provenance = json.loads(provenance_file.read_bytes())
    provenance["files"][name] = {
        "sha256": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
    }
    provenance_file.write_text(json.dumps(provenance))
    output = tmp_path / "site"
    with pytest.raises(ValueError):
        exporter.export_demo(output)
    assert not output.exists()


@pytest.mark.parametrize("target", [".", "src", "examples/web-demo/seed", "dist", ".."])
def test_output_cannot_replace_repository_or_parent_directories(source_tree, target):
    root, _, exporter = source_tree
    before = snapshot(root.parent)
    with pytest.raises(ValueError):
        exporter.export_demo(root / target)
    assert snapshot(root.parent) == before


def test_empty_output_is_accepted_but_unrelated_nonempty_directory_is_preserved(
    source_tree, tmp_path
):
    _, _, exporter = source_tree
    empty = tmp_path / "empty"
    empty.mkdir()
    exporter.export_demo(empty)
    assert (empty / "index.html").is_file()
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    (unrelated / "important.txt").write_text("Keep this existing work")
    before = snapshot(unrelated)
    with pytest.raises(ValueError):
        exporter.export_demo(unrelated)
    assert snapshot(unrelated) == before


@pytest.mark.parametrize("damage", ["changed", "missing", "extra", "empty-directory", "manifest"])
def test_only_intact_generated_output_can_be_replaced(source_tree, tmp_path, damage):
    _, _, exporter = source_tree
    output = tmp_path / "site"
    exporter.export_demo(output)
    if damage == "changed":
        (output / "index.html").write_text("User-maintained content")
    elif damage == "missing":
        (output / "index.html").unlink()
    elif damage == "extra":
        (output / "private.txt").write_text("Do not delete")
    elif damage == "empty-directory":
        (output / "another-project").mkdir()
    else:
        (output / "manifest.json").write_text("{}")
    before = snapshot(output)
    with pytest.raises(ValueError):
        exporter.export_demo(output)
    assert snapshot(output) == before


@pytest.mark.parametrize(
    "location",
    ["app.js", "index.html", "demo/entry.html", "output", "output-parent", "output-file"],
)
def test_symlinks_cannot_redirect_build_inputs_or_replacement(source_tree, tmp_path, location):
    root, _, exporter = source_tree
    output = tmp_path / "site"
    external = tmp_path / "external"
    external.mkdir()
    if location in {"app.js", "index.html", "demo/entry.html"}:
        original = root / "src/iris/static" / location
        shutil.move(original, external / original.name)
        original.symlink_to(external / original.name)
    elif location == "output":
        output.symlink_to(external, target_is_directory=True)
    elif location == "output-parent":
        output.symlink_to(external, target_is_directory=True)
        output = output / "child"
    else:
        exporter.export_demo(output)
        shutil.move(output / "index.html", external / "index.html")
        (output / "index.html").symlink_to(external / "index.html")
    before = snapshot(tmp_path)
    with pytest.raises(ValueError):
        exporter.export_demo(output)
    assert snapshot(tmp_path) == before
