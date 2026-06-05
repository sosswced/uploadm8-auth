#!/usr/bin/env python3
"""Insert early shell boot scripts into app-sidebar HTML pages."""
import re
from pathlib import Path

frontend = Path(__file__).resolve().parents[1] / "frontend"
EARLY = [
    "js/api-base.js",
    "js/auth-stack.js",
    "js/session-user-hydrate.js",
    "shared-sidebar.js",
]
HEAD_SNIP = '    <script src="js/session-chrome-pre.js"></script>'
BOOT_SNIP = '    <script src="js/shell-sidebar-boot.js"></script>'
ASIDE_RE = re.compile(r'(<aside\s+class="sidebar"\s+id="sidebar"></aside>)', re.I)

for path in sorted(frontend.glob("*.html")):
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8", errors="replace")
    if 'id="sidebar"' not in text:
        continue
    orig = text

    if "session-chrome-pre.js" not in text and "html-safe.js" in text:
        text = text.replace(
            '<script src="js/html-safe.js"></script>',
            '<script src="js/html-safe.js"></script>\n' + HEAD_SNIP,
            1,
        )

    if "shell-sidebar-boot.js" not in text:
        text, n = ASIDE_RE.subn(r"\1\n" + BOOT_SNIP, text, count=1)
        if n == 0:
            print("no aside match", path.name)

    for src in EARLY:
        pat = re.compile(r"\n<script src=\"" + re.escape(src) + r"\"></script>")
        text = pat.sub("", text)

    if text != orig:
        path.write_text(text, encoding="utf-8", newline="\n")
        print("patched", path.name)
