#!/usr/bin/env python3
"""beforeShellExecution hook — block dangerous git operations."""
from __future__ import annotations

import json
import re
import sys

DENY_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"git\s+push\b[^;\n]*--force\b[^;\n]*(main|master)\b", re.I),
     "Force-push to main/master is blocked. Use a feature branch and PR."),
    (re.compile(r"git\s+push\b[^;\n]*-(f|force)\b[^;\n]*(main|master)\b", re.I),
     "Force-push to main/master is blocked."),
    (re.compile(r"git\s+add\b[^;\n]*\.env\b", re.I),
     "Staging .env is blocked — secrets must not be committed."),
    (re.compile(r"git\s+commit\b[^;\n]*\.env\b", re.I),
     "Committing .env is blocked."),
]

WARN_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"git\s+add\s+-A|git\s+add\s+\.", re.I),
     "Prefer scoped staging (see uploadm8-backend-push skill) over git add ."),
]


def main() -> int:
    payload = json.load(sys.stdin)
    command = payload.get("command") or ""
    for pattern, msg in DENY_PATTERNS:
        if pattern.search(command):
            print(json.dumps({
                "permission": "deny",
                "user_message": msg,
                "agent_message": f"Hook blocked: {msg}",
            }))
            return 0
    for pattern, msg in WARN_PATTERNS:
        if pattern.search(command):
            print(json.dumps({
                "permission": "ask",
                "user_message": msg,
                "agent_message": msg,
            }))
            return 0
    print(json.dumps({"permission": "allow"}))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
