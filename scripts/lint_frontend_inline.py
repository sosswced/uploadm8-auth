#!/usr/bin/env python3
"""Run scripts/lint-frontend-inline.js via Node."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JS = ROOT / "scripts" / "lint-frontend-inline.js"


def main() -> int:
    if not JS.is_file():
        print("lint-frontend-inline.js not found")
        return 1
    node = shutil.which("node")
    if not node:
        print("node not found — install Node.js to run frontend inline lint")
        return 1
    return subprocess.run([node, str(JS)], cwd=str(ROOT)).returncode


if __name__ == "__main__":
    sys.exit(main())
