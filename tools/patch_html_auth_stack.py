"""Replace auth-api + auth-flow script tags with single auth-stack.js across frontend HTML."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "frontend"


def main() -> None:
    for p in sorted(ROOT.rglob("*.html")):
        t = p.read_text(encoding="utf-8")
        orig = t
        t = re.sub(
            r'([ \t]*)<script\s+src="((?:\.\./)?)js/auth-api\.js"></script>\s*\n'
            r'[ \t]*<script\s+src="\2js/auth-flow\.js"></script>',
            r'\1<script src="\2js/auth-stack.js"></script>',
            t,
        )
        t = re.sub(
            r'([ \t]*)<script\s+src="((?:\.\./)?)js/auth-api\.js"></script>',
            r'\1<script src="\2js/auth-stack.js"></script>',
            t,
        )
        if t != orig:
            p.write_text(t, encoding="utf-8")
            print("patched", p.relative_to(ROOT.parent))


if __name__ == "__main__":
    main()
