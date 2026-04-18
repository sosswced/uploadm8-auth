#!/usr/bin/env python3
"""
Lightweight static checks: local href/src in frontend HTML resolve to existing files.
Exit 0 on success, 1 on missing targets.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"

ATTR_RE = re.compile(
    r'(?:href|src)\s*=\s*["\']([^"\']+)["\']',
    re.IGNORECASE,
)


def main() -> int:
    if not FRONTEND.is_dir():
        print("frontend/ not found", file=sys.stderr)
        return 1

    missing: list[tuple[str, str]] = []
    for html in sorted(FRONTEND.rglob("*.html")):
        try:
            text = html.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            print(f"read {html}: {e}", file=sys.stderr)
            return 1
        for m in ATTR_RE.finditer(text):
            raw = m.group(1).strip()
            if not raw or raw.startswith(("#", "mailto:", "tel:", "javascript:", "data:")):
                continue
            # Skip JS template / framework placeholders inside HTML files
            if "${" in raw or "{{" in raw or raw.startswith("$"):
                continue
            if raw.startswith(("http://", "https://", "//")):
                continue
            path_part = raw.split("?", 1)[0].split("#", 1)[0]
            if not path_part:
                continue
            target = (html.parent / path_part).resolve()
            try:
                target.relative_to(FRONTEND.resolve())
            except ValueError:
                continue
            if not target.is_file():
                missing.append((str(html.relative_to(ROOT)), raw))

    if missing:
        print("Missing local link targets:")
        for src, href in missing[:200]:
            print(f"  {src} -> {href}")
        if len(missing) > 200:
            print(f"  ... and {len(missing) - 200} more")
        return 1

    print(f"OK: checked local href/src in {FRONTEND}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
