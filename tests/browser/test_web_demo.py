"""Exercise the exported site in Chromium without an Iris API or model runtime.

Install the browser dependency group and Chromium before running this module.
IRIS_DEMO_ARTIFACT selects an existing artifact, as used by CI; otherwise the
session exports into a temporary directory. The tested artifact is never edited.
"""

import functools
import hashlib
import json
import os
import subprocess
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import pytest

playwright = pytest.importorskip(
    "playwright.sync_api", reason="Install the browser dependency group"
)
expect = playwright.expect
ROOT = Path(__file__).resolve().parents[2]
MOUNT = "/projects/iris/"


def verify_artifact(artifact):
    manifest = json.loads((artifact / "manifest.json").read_bytes())
    assert manifest["format"] == "iris-web-demo-site-v1"
    paths = {
        path.relative_to(artifact).as_posix() for path in artifact.rglob("*") if path.is_file()
    }
    assert paths == set(manifest["files"]) | {"manifest.json"}
    for relative, expected in manifest["files"].items():
        content = (artifact / relative).read_bytes()
        assert expected == {"sha256": hashlib.sha256(content).hexdigest(), "size": len(content)}
    return manifest


@pytest.fixture(scope="session")
def artifact(tmp_path_factory):
    supplied = os.environ.get("IRIS_DEMO_ARTIFACT")
    directory = Path(supplied).resolve() if supplied else tmp_path_factory.mktemp("site") / "demo"
    if not supplied:
        subprocess.run(
            [
                sys.executable,
                "-S",
                str(ROOT / "examples/web-demo/export.py"),
                "--output",
                str(directory),
            ],
            check=True,
            timeout=30,
        )
    verify_artifact(directory)
    before = (directory / "manifest.json").read_bytes()
    yield directory
    verify_artifact(directory)
    assert (directory / "manifest.json").read_bytes() == before, "The artifact must stay unchanged"


@pytest.fixture(scope="session")
def server(artifact):
    class Handler(SimpleHTTPRequestHandler):
        def translate_path(self, path):
            if urlsplit(path).path.startswith(MOUNT):
                path = path[len(MOUNT) - 1 :]
            return super().translate_path(path)

        def list_directory(self, path):
            self.send_error(404)
            return None

        def log_message(self, format, *args):
            pass

        def end_headers(self):
            self.send_header("Cache-Control", "no-store")
            super().end_headers()

    handler = functools.partial(Handler, directory=str(artifact))
    with ThreadingHTTPServer(("127.0.0.1", 0), handler) as http:
        thread = threading.Thread(target=http.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{http.server_port}"
        finally:
            http.shutdown()
            thread.join(timeout=5)


@pytest.fixture(scope="session")
def browser():
    with playwright.sync_playwright() as runtime:
        with runtime.chromium.launch(headless=True) as instance:
            yield instance


@pytest.fixture
def visit(browser, server, request):
    contexts = []
    errors = []
    previous_failures = request.session.testsfailed

    def create(path="/", **options):
        context = browser.new_context(viewport={"width": 1440, "height": 1000}, **options)
        contexts.append(context)
        context.tracing.start(screenshots=True, snapshots=True)

        def guard(route):
            url = urlsplit(route.request.url)
            if url.scheme in {"http", "https"} and (
                url.netloc != urlsplit(server).netloc or "/api/" in url.path
            ):
                errors.append(
                    f"Unexpected network request: {route.request.method} {route.request.url}"
                )
                route.abort()
            else:
                route.continue_()

        context.route("**/*", guard)
        context.on(
            "page", lambda page: page.on("pageerror", lambda error: errors.append(str(error)))
        )
        page = context.new_page()
        page.set_default_timeout(15_000)
        page.goto(server + path)
        return context, page

    yield create
    failed = request.session.testsfailed > previous_failures or errors
    cleanup_errors = []
    for index, context in enumerate(contexts):
        try:
            if failed:
                output = ROOT / "test-results/browser" / request.node.name
                output.mkdir(parents=True, exist_ok=True)
                (output / "network-errors.json").write_text(json.dumps(errors, indent=2))
                for number, page in enumerate(context.pages):
                    if not page.is_closed():
                        try:
                            page.screenshot(path=str(output / f"context-{index}-page-{number}.png"))
                        except (playwright.Error, OSError) as error:
                            cleanup_errors.append(str(error))
                context.tracing.stop(path=str(output / f"context-{index}.zip"))
            else:
                context.tracing.stop()
        except (playwright.Error, OSError) as error:
            cleanup_errors.append(str(error))
        finally:
            try:
                context.close()
            except playwright.Error as error:
                cleanup_errors.append(str(error))
    if cleanup_errors:
        print("Browser cleanup diagnostics:", cleanup_errors)
    if not failed:
        assert not cleanup_errors, cleanup_errors
    assert not errors, errors


def entry(page, *, ready=True):
    dialog = page.locator("#iris-demo-entry")
    expect(dialog).to_be_visible()
    assert dialog.evaluate('(el) => el.open && el.matches(":modal")')
    assert page.locator("#iris-demo-workbench").evaluate("(el) => el.inert")
    if ready:
        expect(page.locator("#iris-demo-try")).to_be_enabled()
        expect(page.locator("#frame-grid img")).to_have_count(2)
        page.wait_for_function("""() => [...document.querySelectorAll('#frame-grid img')]
            .every(image => image.complete && image.naturalWidth > 0)""")
    else:
        expect(page.locator("#iris-demo-try")).to_be_disabled()


def enter(page, *, keyboard=False):
    entry(page)
    button = page.locator("#iris-demo-try")
    if keyboard:
        button.press("Enter")
    else:
        button.click()
    expect(page.locator("#iris-demo-entry")).to_be_hidden()
    assert not page.locator("#iris-demo-workbench").evaluate("(el) => el.inert")


def annotation(page):
    page.locator("#workspace-annotation").click()
    expect(page.locator("#annotation-editor")).to_be_visible()
    page.locator("#annotation-frame").select_option("frame-6723")


@pytest.mark.parametrize("path", ["/", MOUNT, MOUNT + "index.html"])
def test_entry_assets_navigation_and_recorded_comparison(visit, path):
    _, page = visit(path)
    entry(page)
    assert page.locator("#iris-demo-entry").get_attribute("aria-describedby")
    page.keyboard.press("Escape")
    entry(page)
    for key in ["Tab"] * 6 + ["Shift+Tab"] * 6:
        page.keyboard.press(key)
        assert not page.evaluate("document.activeElement.closest('#iris-demo-workbench') !== null")
    page.locator("#iris-demo-entry .iris-demo-install").click()
    expect(page.locator("#iris-demo-install-dialog")).to_be_visible()
    page.keyboard.press("Escape")
    expect(page.locator("#iris-demo-entry .iris-demo-install")).to_be_focused()
    entry(page)
    page.evaluate("document.fonts.ready")
    fonts = page.evaluate(
        "[...document.fonts].filter(f => f.status === 'loaded').map(f => f.family)"
    )
    assert {"IBM Plex Sans", "IBM Plex Mono", "Marcellus"} <= set(fonts)
    page.evaluate("window.entryContinuity = true")
    enter(page)
    assert page.evaluate("window.entryContinuity")
    assert not page.locator("#main").evaluate('(el) => el.matches(":focus-visible")')
    page.locator("#workspace-comparison").click()
    expect(page.locator("#comparison-detail")).to_be_visible()
    confidence = page.locator("#comparison-confidence")
    confidence.evaluate("(el) => { el.value = '0.1'; el.dispatchEvent(new Event('input')); }")
    before = page.locator("#comparison-canvases .prediction-count").all_text_contents()
    confidence.evaluate("(el) => { el.value = '0.9'; el.dispatchEvent(new Event('input')); }")
    assert before != page.locator("#comparison-canvases .prediction-count").all_text_contents()
    page.locator("a.brand").click()
    assert urlsplit(page.url).path == path.removesuffix("index.html")
    enter(page, keyboard=True)
    expect(page.locator("#main")).to_be_focused()


@pytest.mark.parametrize("theme,reduced", [("light", "no-preference"), ("dark", "reduce")])
def test_mobile_entry_and_installation_dialog_fit_the_screen(visit, theme, reduced):
    _, page = visit(MOUNT, color_scheme=theme, reduced_motion=reduced)
    page.set_viewport_size({"width": 390, "height": 844})
    entry(page)
    assert page.evaluate("document.documentElement.dataset.theme") == "light"
    page.evaluate("theme => IRISAppearance.setPreference(theme)", theme)
    page.reload()
    entry(page)
    assert page.evaluate("document.documentElement.dataset.theme") == theme
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    for selector in ["#iris-demo-entry", "#iris-demo-entry .iris-demo-install", "#iris-demo-try"]:
        assert page.locator(selector).evaluate("""(el) => {
            const r = el.getBoundingClientRect();
            return r.left >= 0 && r.right <= innerWidth && r.top >= 0 && r.bottom <= innerHeight;
        }""")
    if reduced == "reduce":
        assert page.evaluate("matchMedia('(prefers-reduced-motion: reduce)').matches")
        assert page.locator("#iris-demo-entry").evaluate(
            "(el) => el.getAnimations({subtree: true}).every(a => a.playState !== 'running')"
        )
    page.locator("#iris-demo-entry .iris-demo-install").click()
    dialog = page.locator("#iris-demo-install-dialog")
    expect(dialog).to_be_visible()
    assert dialog.evaluate("""(el) => {
        const r = el.getBoundingClientRect();
        return r.left >= 0 && r.right <= innerWidth && r.top >= 0 && r.bottom <= innerHeight;
    }""")
    page.locator("#iris-demo-install-close").click()
    entry(page)


@pytest.mark.parametrize("path", ["/", MOUNT + "index.html"])
def test_saved_annotations_reload_export_and_reset(visit, path, tmp_path):
    _, page = visit(path)
    enter(page)
    annotation(page)
    page.locator("#annotation-proposals .annotation-proposal.pending").first.get_by_role(
        "button", name="Accept proposal", exact=True
    ).click()
    page.locator("#annotation-reviewer").fill("Browser regression draft")
    page.locator("#annotation-save").click()
    expect(page.locator("#annotation-status")).to_have_text("Draft · revision 1")
    page.reload()
    enter(page)
    annotation(page)
    expect(page.locator("#annotation-box-count")).to_have_text("1")
    expect(page.locator("#annotation-reviewer")).to_have_value("Browser regression draft")
    expect(page.locator("#annotation-status")).to_have_text("Draft · revision 1")
    with page.expect_download() as download:
        page.locator("#iris-demo-export").click()
    output = tmp_path / "annotations.json"
    download.value.save_as(output)
    exported = json.loads(output.read_bytes())
    assert len(exported["annotations"]) == 1 and len(exported["images"]) == 2
    accepted = exported["annotations"][0]
    assert accepted["image_id"] == 6723
    assert accepted["iris"]["source"]["kind"] == "imported"
    assert accepted["iris"]["review_state"] == "accepted"
    page.once("dialog", lambda dialog: dialog.accept())
    with page.expect_event("domcontentloaded"):
        page.locator("#iris-demo-reset").click()
    enter(page)
    annotation(page)
    expect(page.locator("#annotation-status")).to_have_text("Unannotated · revision 0")
    expect(page.locator("#annotation-box-count")).to_have_text("0")
    expect(page.locator("#annotation-proposals .annotation-proposal.pending")).to_have_count(3)
    expect(page.locator("#selected-total")).to_have_text("2")


def test_profiles_and_mounts_have_separate_browser_data(visit, server):
    _, original = visit(MOUNT)
    enter(original)
    annotation(original)
    original.locator("#annotation-reviewer").fill("Only this browser and mount")
    original.locator("#annotation-save").click()
    expect(original.locator("#annotation-status")).to_have_text("Draft · revision 1")
    _, separate = visit(MOUNT)
    enter(separate)
    annotation(separate)
    expect(separate.locator("#annotation-status")).to_have_text("Unannotated · revision 0")
    original.goto(server + "/")
    enter(original)
    annotation(original)
    expect(original.locator("#annotation-status")).to_have_text("Unannotated · revision 0")
    original.goto(server + MOUNT + "index.html")
    enter(original)
    annotation(original)
    expect(original.locator("#annotation-reviewer")).to_have_value("Only this browser and mount")


def test_changed_seed_preserves_old_edits_until_explicit_reset(visit, artifact):
    context, page = visit(MOUNT)
    enter(page)
    annotation(page)
    page.locator("#annotation-reviewer").fill("Keep this saved draft")
    page.locator("#annotation-save").click()
    expect(page.locator("#annotation-status")).to_have_text("Draft · revision 1")
    manifest = json.loads((artifact / "manifest.json").read_bytes())
    seed = artifact / "assets" / manifest["asset_version"] / "seed"
    changed = (seed / "fixture.json").read_bytes() + b"\n"
    provenance = json.loads((seed / "provenance.json").read_bytes())
    provenance["files"]["fixture.json"].update(
        sha256=hashlib.sha256(changed).hexdigest(), size_bytes=len(changed)
    )

    def intercept():
        context.route(
            "**/seed/fixture.json",
            lambda route: route.fulfill(status=200, content_type="application/json", body=changed),
        )
        context.route(
            "**/seed/provenance.json",
            lambda route: route.fulfill(status=200, json=provenance),
        )

    intercept()
    page.reload()
    entry(page, ready=False)
    reset = page.locator("#iris-demo-entry-reset")
    expect(reset).to_be_visible()
    page.once("dialog", lambda dialog: dialog.dismiss())
    reset.click()
    entry(page, ready=False)
    context.unroute("**/seed/fixture.json")
    context.unroute("**/seed/provenance.json")
    page.reload()
    enter(page)
    annotation(page)
    expect(page.locator("#annotation-reviewer")).to_have_value("Keep this saved draft")
    intercept()
    page.reload()
    expect(reset).to_be_visible()
    page.once("dialog", lambda dialog: dialog.accept())
    with page.expect_event("domcontentloaded"):
        reset.click()
    enter(page)
    annotation(page)
    expect(page.locator("#annotation-status")).to_have_text("Unannotated · revision 0")


@pytest.mark.parametrize(
    "resource", ["seed/fixture.json", "seed/media/6723.png", "static/demo/bootstrap.js"]
)
def test_missing_resources_keep_entry_closed_and_retry_recovers(visit, resource):
    context, page = visit()
    entry(page)
    pattern = "**/" + resource
    context.route(pattern, lambda route: route.fulfill(status=503, body="Unavailable"))
    page.reload()
    entry(page, ready=False)
    expect(page.locator("#iris-demo-entry-retry")).to_be_visible()
    page.keyboard.press("Escape")
    entry(page, ready=False)
    context.unroute(pattern)
    page.locator("#iris-demo-entry-retry").click()
    enter(page)
