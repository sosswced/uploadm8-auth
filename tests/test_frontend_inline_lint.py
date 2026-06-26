"""CI hook: frontend inline script syntax lint."""

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_frontend_inline_scripts_parse():
    node = shutil.which("node")
    if not node:
        return
    script = ROOT / "scripts" / "lint-frontend-inline.js"
    if not script.is_file():
        return
    proc = subprocess.run([node, str(script)], cwd=str(ROOT), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
