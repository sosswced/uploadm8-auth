"""Remove standalone js/auth-flow.js script tags (superseded by auth-stack.js)."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "frontend"
PAT = re.compile(r"\n[ \t]*<script\s+src=\"((?:\.\./)?)js/auth-flow\.js\"></script>[ \t]*\r?\n")


def main() -> None:
    for p in ROOT.rglob("*.html"):
        t = p.read_text(encoding="utf-8")
        t2 = PAT.sub("\n", t)
        if t2 != t:
            p.write_text(t2, encoding="utf-8")
            print("stripped", p.relative_to(ROOT.parent))


if __name__ == "__main__":
    main()
