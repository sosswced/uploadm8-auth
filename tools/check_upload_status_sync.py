#!/usr/bin/env python3
"""Verify frontend scheduled-status.js matches backend status tuples."""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    sys.path.insert(0, str(ROOT))
    from services.upload.status import (
        SCHEDULED_PIPELINE_STATUSES,
        PROCESSING_STATUSES,
        COMPLETED_STATUSES,
        PARTIAL_STATUSES,
        FAILED_STATUSES,
    )

    js_path = ROOT / "frontend" / "js" / "scheduled-status.js"
    if not js_path.is_file():
        print("scheduled-status.js not found", file=sys.stderr)
        return 1

    text = js_path.read_text(encoding="utf-8", errors="replace")

    def extract_array(name: str) -> list[str]:
        m = re.search(rf"var {name}\s*=\s*\[([^\]]+)\]", text)
        if not m:
            raise ValueError(f"Could not find {name} in scheduled-status.js")
        inner = m.group(1)
        return [s.strip().strip("'\"") for s in inner.split(",") if s.strip()]

    pairs = [
        ("CANONICAL_SCHEDULED", SCHEDULED_PIPELINE_STATUSES),
        ("CANONICAL_PROCESSING", PROCESSING_STATUSES),
        ("CANONICAL_COMPLETED", COMPLETED_STATUSES),
        ("CANONICAL_PARTIAL", PARTIAL_STATUSES),
        ("CANONICAL_FAILED", FAILED_STATUSES),
    ]

    errors: list[str] = []
    for js_name, py_tuple in pairs:
        js_vals = extract_array(js_name)
        py_vals = list(py_tuple)
        if js_vals != py_vals:
            errors.append(f"{js_name}: JS={js_vals} PY={py_vals}")

    if errors:
        print("Upload status tuple mismatch:")
        for e in errors:
            print(f"  - {e}")
        return 1

    print("Upload status tuples in sync (scheduled-status.js <-> services.upload.status)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
