#!/usr/bin/env python3
"""
Headless self-heal driver for agent loops (Level 8).

Runs eval_loop repeatedly and reports structured JSON. Agents use this output
to patch code between iterations; this script only runs tests and aggregates.

Examples:
  python scripts/agent/self_heal.py --mode unit --budget 5 --json
  python scripts/agent/self_heal.py --mode full --budget 3
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "scripts" / "agent" / "eval_loop.py"


def run_eval(mode: str) -> dict:
    proc = subprocess.run(
        [sys.executable, str(EVAL), "--mode", mode, "--json"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    raw = (proc.stdout or "").strip()
    try:
        report = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        # Fallback: last line only (legacy one-line emitters).
        try:
            report = json.loads(raw.splitlines()[-1] if raw else "{}")
        except json.JSONDecodeError:
            report = {"ok": False, "parse_error": True, "raw": raw[-2000:]}
    report["exit_code"] = proc.returncode
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="UploadM8 self-heal eval driver")
    parser.add_argument("--mode", default="unit", choices=["unit", "frontend", "router", "full"])
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    iterations: list[dict] = []
    final: dict | None = None

    for i in range(1, args.budget + 1):
        report = run_eval(args.mode)
        report["iteration"] = i
        iterations.append(report)
        final = report
        if report.get("ok"):
            break

    summary = {
        "ok": bool(final and final.get("ok")),
        "mode": args.mode,
        "budget": args.budget,
        "iterations_run": len(iterations),
        "final": final,
        "history": iterations,
        "agent_action": (
            "none — tests green"
            if final and final.get("ok")
            else "fix failures listed in final.failures, then re-run self_heal or eval_loop"
        ),
    }

    print(json.dumps(summary, indent=2))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
