"""The browser demo serves the shared UI and exports labels usable by local Iris."""

import http.client
import importlib.util
import io
import json
import shutil
import subprocess
import threading
import zipfile
from html.parser import HTMLParser
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from iris.app import create_app

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
    assert "/static/tracking-quality.js" not in parser.urls
    assert "/static/benchmark-editor.js" not in parser.urls
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


def test_browser_annotation_export_reimports_through_real_api(tmp_path):
    node = shutil.which("node")
    if not node:
        pytest.skip("Cross-runtime COCO round trip requires Node.js")
    output = subprocess.run(
        [
            node,
            "-e",
            """
const domain = require('./src/iris/static/demo/annotations.js');
const seed = require('./examples/web-demo/seed/fixture.json');
const provenance = require('./examples/web-demo/seed/provenance.json');
const state = {...domain.createState(seed), epoch: 'export-contract'};
const doc = domain.document(seed, state, 'frame-6723');
const first = doc.suggestions[0];
domain.save(seed, state, doc.frame.id, {
  expected_revision: 0, demo_epoch: state.epoch, status: 'validated',
  reviewer: 'Automated contract check, not human ground truth',
  boxes: [
    {id:'accepted', label:first.label, box:first.box, suggestion_id:first.id},
    {id:'manual', label:'person', box:[10,20,30,60]},
  ],
  decisions: Object.fromEntries(doc.suggestions.map((s, i) => [s.id, i ? 'rejected' : 'accepted'])),
});
process.stdout.write(JSON.stringify(domain.exportCOCO(seed, state, {provenance})));
""",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    exported = json.loads(output.stdout)
    assert len(exported["annotations"]) == 2
    assert exported["iris"]["dataset_frozen"] is False
    assert [row["status"] for row in exported["iris"]["reviews"]] == ["validated", "unannotated"]
    package = io.BytesIO()
    with (
        zipfile.ZipFile(ROOT / "examples/street-scenes/street-scenes.zip") as original,
        zipfile.ZipFile(package, "w") as archive,
    ):
        archive.writestr("annotations.json", output.stdout)
        for image in exported["images"]:
            archive.writestr(image["file_name"], original.read(image["file_name"]))
    with TestClient(
        create_app(tmp_path / "workspace", run_jobs=False), base_url="http://127.0.0.1"
    ) as api:
        preview = api.post(
            "/api/dataset-imports",
            files={
                "file": ("browser-export.zip", package.getvalue(), "application/zip"),
            },
        )
        assert preview.status_code == 201, preview.text
        detail = preview.json()
        assert detail["image_count"] == 2 and detail["annotation_count"] == 2
        committed = api.post(
            f"/api/dataset-imports/{detail['id']}/commit",
            json={
                "name": "Browser demo export",
                "scene_group": "street-scenes",
                "source_split": "val",
                "source_url": "https://github.com/victormonnot/iris/tree/main/examples/street-scenes",
                "license_name": "Photos CC BY-SA 2.0; source annotations CC BY 4.0",
                "attribution": "See source provenance in the browser export",
                "category_mapping": {"1": "person", "3": "car"},
            },
        )
        assert committed.status_code == 201, committed.text
        assert committed.json()["proposal_count"] == 2
        documents = [
            api.get(f"/api/frames/{fid}/annotation").json() for fid in committed.json()["frame_ids"]
        ]
        assert all(doc["revision"] == 0 and doc["status"] == "unannotated" for doc in documents)
        proposals = [item for doc in documents for item in doc["suggestions"]]
        assert len(proposals) == 2
        assert any(
            item["label"] == "person" and item["box"] == [10, 20, 30, 60] for item in proposals
        )
