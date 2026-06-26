#!/usr/bin/env python3
"""afterFileEdit hook — suggest targeted test command."""
from __future__ import annotations

import json
import sys
from pathlib import Path

TEST_HINTS: list[tuple[str, str]] = [
    ("routers/", "python run_tests.py router-lint && python run_tests.py unit"),
    ("services/upload/", "python run_tests.py unit -k upload"),
    ("stages/", "python run_tests.py unit -k stage"),
    ("frontend/", "python run_tests.py frontend-lint"),
    ("tests/e2e/", "python run_tests.py e2e"),
    ("tests/", "python run_tests.py unit"),
]


def suggest(path: str) -> str | None:
    normalized = path.replace("\\", "/")
    for prefix, cmd in TEST_HINTS:
        if prefix in normalized:
            return cmd
    if normalized.endswith(".py"):
        return "python run_tests.py unit"
    return None


def main() -> int:
    payload = json.load(sys.stdin)
    file_path = payload.get("file_path") or payload.get("path") or ""
    if not file_path:
        return 0
    hint = suggest(str(file_path))
    if not hint:
        return 0
    name = Path(file_path).name
    print(json.dumps({
        "additional_context": f"Edited `{name}`. Suggested verification: `{hint}`"
    }))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
