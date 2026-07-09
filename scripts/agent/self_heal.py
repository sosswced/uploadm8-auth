#!/usr/bin/env python3
"""
Headless self-heal driver for agent loops (Level 8 / /TUP).

Runs eval_loop (and optional live_result_workflow) repeatedly and reports
structured JSON. Agents use this output to patch code between iterations;
this script only runs tests and aggregates.

Examples:
  python scripts/agent/self_heal.py --mode unit --budget 5 --json
  python scripts/agent/self_heal.py --source auto --budget 5 --json
  python scripts/agent/self_heal.py --source live --json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "scripts" / "agent" / "eval_loop.py"
LIVE_PLAN = ROOT / "scripts" / "agent" / "live_result_workflow.py"


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
        try:
            report = json.loads(raw.splitlines()[-1] if raw else "{}")
        except json.JSONDecodeError:
            report = {"ok": False, "parse_error": True, "raw": raw[-2000:]}
    report["exit_code"] = proc.returncode
    return report


def run_live_plan() -> dict:
    if not LIVE_PLAN.is_file():
        return {"ok": True, "skipped": True, "signals": []}
    proc = subprocess.run(
        [sys.executable, str(LIVE_PLAN), "--json"],
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
        report = {"ok": False, "parse_error": True, "raw": raw[-2000:]}
    report["exit_code"] = proc.returncode
    return report


def merge_worklist(unit: dict, live: dict | None) -> dict:
    """Combine unit eval + live signals into one agent-facing report."""
    failures = list(unit.get("failures") or [])
    signals = list((live or {}).get("signals") or [])
    for sig in signals:
        if sig.startswith("api_5xx:") or sig.startswith("upload_status:") or sig == "live_demo_failed":
            failures.append({"file": "tests/e2e/helpers/live_demo.py", "nodeid": sig})
    modes = list(dict.fromkeys(
        [unit.get("mode") or "unit"] + list((live or {}).get("eval_modes") or [])
    ))
    ok = bool(unit.get("ok")) and (live is None or live.get("skipped") or bool(live.get("ok")))
    return {
        "ok": ok,
        "unit": unit,
        "live": live,
        "failures": failures[:30],
        "failure_count": len(failures),
        "eval_modes": modes,
        "pytest_focus": list((live or {}).get("pytest_focus") or []),
        "suggested_command": (live or {}).get("suggested_command")
        or unit.get("suggested_command")
        or "python run_tests.py unit",
        "agent_action": (
            "none — tests green"
            if ok
            else "fix failures listed, then re-run self_heal --source auto"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="UploadM8 self-heal eval driver")
    parser.add_argument("--mode", default="unit", choices=["unit", "frontend", "router", "full"])
    parser.add_argument(
        "--source",
        default="unit",
        choices=["unit", "live", "auto"],
        help="unit=eval only; live=live_result_workflow only; auto=both (prefer for /TUP)",
    )
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    iterations: list[dict] = []
    final: dict | None = None

    for i in range(1, args.budget + 1):
        live_report = None
        if args.source in ("live", "auto"):
            live_report = run_live_plan()
        if args.source == "live":
            merged = {
                "ok": bool(live_report.get("ok")),
                "live": live_report,
                "unit": None,
                "failures": live_report.get("failures") or [],
                "failure_count": live_report.get("failure_count") or 0,
                "eval_modes": live_report.get("eval_modes") or ["unit"],
                "pytest_focus": live_report.get("pytest_focus") or [],
                "suggested_command": live_report.get("suggested_command"),
                "agent_action": live_report.get("agent_action"),
            }
        else:
            unit_report = run_eval(args.mode)
            if args.source == "auto":
                # Also run extra eval modes suggested by live plan (once per iteration)
                extra_modes = [
                    m
                    for m in (live_report or {}).get("eval_modes") or []
                    if m != args.mode and m in ("unit", "frontend", "router", "full")
                ]
                for m in extra_modes[:2]:
                    extra = run_eval(m)
                    if not extra.get("ok"):
                        unit_report.setdefault("extra_evals", {})[m] = extra
                        unit_report["ok"] = False
                        for f in extra.get("failures") or []:
                            unit_report.setdefault("failures", []).append(f)
                merged = merge_worklist(unit_report, live_report)
            else:
                merged = merge_worklist(unit_report, None)

        merged["iteration"] = i
        iterations.append(merged)
        final = merged
        if merged.get("ok"):
            break

    summary = {
        "ok": bool(final and final.get("ok")),
        "mode": args.mode,
        "source": args.source,
        "budget": args.budget,
        "iterations_run": len(iterations),
        "final": final,
        "history": iterations,
        "agent_action": (
            "none — tests green"
            if final and final.get("ok")
            else (
                "fix failures in final.failures, then re-run self_heal --source auto "
                "(no re-upload; use /TUP --heal-only)"
            )
        ),
    }

    print(json.dumps(summary, indent=2))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
