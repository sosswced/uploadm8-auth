#!/usr/bin/env python3
"""subagentStop hook — chain eval loop after review subagents."""
from __future__ import annotations

import json
import sys

REVIEW_TYPES = {"bugbot", "security-review", "ci-investigator"}


def main() -> int:
    payload = json.load(sys.stdin)
    subagent_type = (payload.get("subagent_type") or payload.get("type") or "").lower()
    status = (payload.get("status") or "completed").lower()
    if subagent_type in REVIEW_TYPES and status in ("completed", "success", "done"):
        print(json.dumps({
            "followup_message": (
                "Review subagent finished. Run eval harness: "
                "`python scripts/agent/eval_loop.py --mode unit --json`. "
                "If failures exist, enter fix-tests loop."
            )
        }))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
