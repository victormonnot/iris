"""The browser demo serves the complete shared UI without an Iris API."""

import http.client
import importlib.util
import json
import threading
from html.parser import HTMLParser
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "web-demo"


@pytest.fixture(scope="module")
def preview():
    spec = importlib.util.spec_from_file_location("iris_demo_preview", EXAMPLE / "serve.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with module.ThreadingHTTPServer(("127.0.0.1", 0), module.DemoHandler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield server
        finally:
            server.shutdown()
            thread.join(timeout=5)


def request(server, path, method="GET"):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    try:
        connection.request(method, path)
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


class Assets(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        url = (
            values.get("src") if tag == "script" else values.get("href") if tag == "link" else None
        )
        if url:
            self.urls.append(url)


def test_preview_composes_shared_html_and_all_browser_assets(preview):
    status, headers, body = request(preview, "/?project=web-demo")
    assert status == 200
    assert headers["Content-Type"] == "text/html; charset=utf-8"
    assert headers["Cache-Control"] == "no-store"
    html = body.decode()
    parser = Assets()
    parser.feed(html)
    assert html.index("demo/bootstrap.js") < html.index("project-scope.js") < html.index("app.js")
    assert html.index("demo/entry.js") < html.index("demo/bootstrap.js") < html.index("app.js")
    assert html.index("annotation.js") < html.index("demo/ui.js")
    assert '<div id="iris-demo-workbench" inert>' in html
    assert 'data-iris-runtime="demo"' in html
    assert 'aria-label="Try Iris"' in html
    assert "demo/install.js" in html
    native = Assets()
    native.feed((ROOT / "src/iris/static/index.html").read_text())
    assert set(native.urls) <= set(parser.urls), "Every native workspace module must be loaded"
    assert html.index("demo/catalog.js") < html.index("demo/adapter.js")
    assert 'data-seed="/seed/"' in html
    for path in parser.urls:
        status, headers, content = request(preview, path)
        assert status == 200, path
        assert content, path
        assert headers["X-Content-Type-Options"] == "nosniff"
    seed = json.loads(request(preview, "/seed/fixture.json")[2])
    for relative in seed["resources"].values():
        assert request(preview, "/seed/" + relative)[0] == 200
    assert b"demo/bootstrap.js" not in (ROOT / "src/iris/static/index.html").read_bytes()


@pytest.mark.parametrize(
    "path",
    [
        "/api/projects",
        "/api/jobs",
        "/seed/",
        "/static/",
        "/static/index.html",
        "/README.md",
        "/static/../../../pyproject.toml",
        "/seed/%2e%2e/recording.py",
        "/static/%00.css",
        "/static/%5c..%5cindex.html",
    ],
)
def test_preview_exposes_no_api_directories_or_repository_files(preview, path):
    assert request(preview, path)[0] == 404


def test_preview_rejects_mutations_and_supports_head(preview):
    assert request(preview, "/api/jobs", "POST")[0] == 501
    status, headers, content = request(preview, "/seed/fixture.json", "HEAD")
    assert status == 200 and content == b""
    assert int(headers["Content-Length"]) == (EXAMPLE / "seed/fixture.json").stat().st_size
