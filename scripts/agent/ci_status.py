#!/usr/bin/env python3
"""
CI status JSON for ascended CI agent workflows (Level 8).

Wraps gh pr checks and recent workflow runs into structured output.

Examples:
  python scripts/agent/ci_status.py --json
  python scripts/agent/ci_status.py --pr 42 --json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def gh(*args: str) -> tuple[int, str]:
    proc = subprocess.run(
        ["gh", *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def parse_checks(output: str) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []
    for line in output.strip().splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            checks.append({"name": parts[0].strip(), "status": parts[1].strip(), "url": parts[2].strip() if len(parts) > 2 else ""})
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description="UploadM8 CI status for agents")
    parser.add_argument("--pr", type=int, default=0, help="PR number (default: current branch)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    pr_args = ["pr", "checks"]
    if args.pr:
        pr_args.append(str(args.pr))
    code, checks_out = gh(*pr_args)
    checks = parse_checks(checks_out) if code == 0 else []

    run_code, runs_out = gh("run", "list", "--limit", "5", "--json", "databaseId,status,conclusion,name,headBranch,url")
    runs: list[dict] = []
    if run_code == 0 and runs_out.strip():
        try:
            runs = json.loads(runs_out)
        except json.JSONDecodeError:
            runs = []

    failing = [c for c in checks if c.get("status", "").lower() not in ("pass", "success", "skipping", "neutral")]
    report = {
        "ok": len(failing) == 0 and code == 0,
        "gh_available": code != 127,
        "checks": checks,
        "failing_checks": failing,
        "recent_runs": runs,
        "suggested_action": (
            "all checks green"
            if not failing
            else f"triage {len(failing)} failing check(s) with ci-investigator subagent"
        ),
    }

    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
