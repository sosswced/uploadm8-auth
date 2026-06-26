#!/usr/bin/env python3
"""sessionStart hook — inject AGENTS.md pointer."""
from __future__ import annotations

import json
import sys

def main() -> int:
    _ = json.load(sys.stdin)
    print(json.dumps({
        "additional_context": (
            "UploadM8 agent stack active (Levels 4–8+). "
            "Read AGENTS.md and .cursor/prompts/context-is-everything.md first. "
            "Eval: python scripts/agent/eval_loop.py --json. "
            "Workflow: python scripts/agent/dynamic_workflow.py --json. "
            "Slash: /trill, /fix-tests, /ascended-loop, /ultrathink, /ultracode, /orchestrate, "
            "/five-windows, /multi-repo-ship, /headless-eval, /parallel-audit."
        )
    }))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
