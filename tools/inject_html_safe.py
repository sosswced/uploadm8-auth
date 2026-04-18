"""Inject frontend/js/html-safe.js immediately after <head> in all HTML pages (idempotent)."""
from __future__ import annotations

import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent
FRONTEND = REPO / "frontend"


def main() -> None:
    for path in sorted(FRONTEND.rglob("*.html")):
        text = path.read_text(encoding="utf-8")
        if "html-safe.js" in text:
            continue
        lower = text.lower()
        idx = lower.find("<head")
        if idx == -1:
            continue
        end = text.find(">", idx)
        if end == -1:
            continue
        insert_at = end + 1
        if insert_at < len(text) and text[insert_at] == "\n":
            insert_at += 1
        rel = path.relative_to(FRONTEND)
        depth = len(rel.parts) - 1
        prefix = "../" * depth if depth else ""
        snippet = f'    <script src="{prefix}js/html-safe.js"></script>\n'
        path.write_text(text[:insert_at] + "\n" + snippet + text[insert_at:], encoding="utf-8")
        print("patched", rel.as_posix())


if __name__ == "__main__":
    main()
