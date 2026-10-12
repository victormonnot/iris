"""Preview the shared Iris interface with static data, without starting its API."""

from __future__ import annotations

import argparse
import mimetypes
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

EXAMPLE = Path(__file__).resolve().parent
STATIC = EXAMPLE.parents[1] / "src" / "iris" / "static"
SEED = EXAMPLE / "seed"
# Retain the shared modules used by the supported workspaces. Processing modules
# have no browser implementation; their controls are explained by demo/ui.js.
DEMO_SCRIPTS = {
    "appearance.js",
    "project-scope.js",
    "runtime.js",
    "taxonomy-tools.js",
    "intake-tools.js",
    "job-tools.js",
    "app.js",
    "navigation.js",
    "taxonomies.js",
    "comparison-replay.js",
    "comparison.js",
    "annotation-tools.js",
    "preannotation-tools.js",
    "annotation.js",
}


def demo_index() -> bytes:
    """Compose the demo entry point from the maintained local application HTML."""
    source = (STATIC / "index.html").read_text(encoding="utf-8")
    source = re.sub(
        r'    <script src="/static/([^"/]+)"[^>]*></script>\n',
        lambda match: match[0] if match[1] in DEMO_SCRIPTS else "",
        source,
    )
    anchor = '    <script src="/static/project-scope.js" defer></script>'
    if any(
        source.count(token) != 1
        for token in (anchor, '<html lang="en">', "  </head>", "  <body>", "  </body>")
    ):
        raise ValueError("The shared Iris entry point changed; update the demo script order.")
    scripts = '    <script src="/static/demo/entry.js" defer></script>\n' + "\n".join(
        f'    <script src="/static/demo/{name}.js" defer></script>'
        for name in ("storage", "annotations", "adapter")
    )
    scripts += '\n    <script src="/static/demo/bootstrap.js" data-seed="/seed/" defer></script>'
    source = source.replace(anchor, scripts + "\n" + anchor)
    source = source.replace(
        "  </head>",
        '    <link rel="stylesheet" href="/static/demo/ui.css" />\n'
        '    <link rel="stylesheet" href="/static/demo/entry.css" />\n'
        '    <link rel="stylesheet" href="/static/demo/install.css" />\n'
        '    <script src="/static/demo/ui.js" defer></script>\n'
        '    <script src="/static/demo/install.js" defer></script>\n  </head>',
    )
    source = source.replace(
        '<html lang="en">',
        '<html lang="en" class="iris-demo iris-demo-entering" data-iris-runtime="demo">',
    )
    source = source.replace("  <body>", '  <body>\n    <div id="iris-demo-workbench" inert>')
    entry = (STATIC / "demo" / "entry.html").read_text(encoding="utf-8")
    source = source.replace("  </body>", "    </div>\n" + entry + "  </body>")
    return source.encode("utf-8")


class DemoHandler(BaseHTTPRequestHandler):
    """Serve only the public frontend and seed, with no API or directory listing."""

    def do_GET(self) -> None:
        self._serve(head=False)

    def do_HEAD(self) -> None:
        self._serve(head=True)

    def _serve(self, *, head: bool) -> None:
        path = unquote(urlsplit(self.path).path)
        if path in ("/", "/index.html"):
            content, content_type = demo_index(), "text/html; charset=utf-8"
        else:
            roots = {"/static/": STATIC, "/seed/": SEED}
            root = next((base for prefix, base in roots.items() if path.startswith(prefix)), None)
            relative = path.split("/", 2)[-1]
            if root is None or "\x00" in path or "\\" in path:
                self.send_error(404)
                return
            target = (root / relative).resolve()
            if (
                not target.is_relative_to(root)
                or not target.is_file()
                or target.suffix
                not in {".js", ".css", ".svg", ".woff2", ".json", ".txt", ".png", ".jpg"}
            ):
                self.send_error(404)
                return
            content = target.read_bytes()
            content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if not head:
            self.wfile.write(content)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    demo_index()  # Check the shared entry point before opening the socket.
    with ThreadingHTTPServer(("127.0.0.1", args.port), DemoHandler) as server:
        print(f"Iris browser demo: http://127.0.0.1:{server.server_port}/", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
