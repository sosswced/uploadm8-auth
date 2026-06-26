#!/usr/bin/env python3
"""
TRILL — master agent orchestrator report (UploadM8).

Runs all scriptable stack checks and returns one JSON action plan for agents.
Agent phases (explore, bugbot, fix-tests, ship) consume this report.

Modes:
  audit      — route + eval + repos + CI (no overnight)
  pre-ship   — audit + frontend eval + full unit
  heal       — audit + self_heal iterations
  overnight  — audit + recommend/run marker (report only unless --run-overnight)
  full       — everything in audit + all eval modes from workflow

Examples:
  python scripts/agent/trill.py --json
  python scripts/agent/trill.py --mode pre-ship --json
  python scripts/agent/trill.py --mode heal --budget 3 --json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
AGENT = Path(__file__).resolve().parent


def run_json(script: str, *args: str) -> dict[str, Any]:
    proc = subprocess.run(
        [sys.executable, str(AGENT / script), *args, "--json"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    raw = (proc.stdout or "").strip()
    if not raw:
        return {"ok": False, "error": "empty output", "stderr": (proc.stderr or "")[-500:]}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"ok": False, "parse_error": True, "raw": raw[-2000:]}


def verify_env() -> dict[str, Any]:
    proc = subprocess.run(
        [sys.executable, str(ROOT / "run_tests.py"), "verify"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    try:
        gh = subprocess.run(["gh", "--version"], capture_output=True, text=True)
        gh_available = gh.returncode == 0
    except FileNotFoundError:
        gh_available = False
    mcp_script = AGENT / "mcp_server.py"
    return {
        "python_ok": proc.returncode == 0,
        "gh_available": gh_available,
        "mcp_server": mcp_script.is_file(),
        "hooks": (ROOT / ".cursor" / "hooks.json").is_file(),
        "agents_md": (ROOT / "AGENTS.md").is_file(),
    }


def stack_files_present() -> dict[str, bool]:
    paths = [
        ".cursor/mcp.json",
        ".cursor/skills/trill/SKILL.md",
        "scripts/agent/eval_loop.py",
        "scripts/agent/dynamic_workflow.py",
        "scripts/agent/self_heal.py",
        "scripts/agent/ci_status.py",
        "scripts/agent/multi_repo_status.py",
    ]
    return {p: (ROOT / p).is_file() for p in paths}


def build_actions(report: dict[str, Any]) -> list[dict[str, str]]:
    actions: list[dict[str, str]] = []
    workflow = report.get("workflow") or {}
    evals = report.get("evals") or {}

    if not report.get("env", {}).get("python_ok"):
        actions.append({"phase": "bootstrap", "action": "Fix Python/pytest: python run_tests.py verify"})

    missing = [k for k, v in (report.get("stack") or {}).items() if not v]
    if missing:
        actions.append({"phase": "bootstrap", "action": f"Missing stack files: {', '.join(missing)}"})

    wf_name = workflow.get("workflow", "")
    if wf_name:
        actions.append({"phase": "route", "action": f"Workflow: {wf_name} — slash: {workflow.get('slash_commands', ['/orchestrate'])}"})

    for mode, result in evals.items():
        if not result.get("ok"):
            actions.append({
                "phase": "heal",
                "action": f"/fix-tests — {mode} eval red ({result.get('failure_count', '?')} failures)",
            })

    if report.get("ci") and not report.get("ci", {}).get("ok"):
        actions.append({"phase": "ci", "action": "uploadm8-ascended-ci — triage failing_checks with ci-investigator"})

    repos = report.get("repos") or {}
    if repos.get("both_dirty"):
        actions.append({"phase": "ship", "action": "/multi-repo-ship — backend then frontend (user must approve push)"})

    if not any(a["phase"] == "heal" for a in actions):
        actions.append({"phase": "audit", "action": "/parallel-audit — explore + bugbot before ship"})

    if report.get("mode") in ("overnight", "full") and not report.get("overnight_started"):
        actions.append({"phase": "overnight", "action": "/overnight or /headless-eval — background E2E"})

    if not actions:
        actions.append({"phase": "done", "action": "All scriptable gates green — optional /multi-repo-ship if user asked"})

    return actions


def main() -> int:
    parser = argparse.ArgumentParser(description="TRILL master orchestrator report")
    parser.add_argument("--mode", default="audit", choices=["audit", "pre-ship", "heal", "overnight", "full"])
    parser.add_argument("--budget", type=int, default=5, help="self_heal budget for heal mode")
    parser.add_argument("--run-overnight", action="store_true", help="Actually start overnight pytest (long)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report: dict[str, Any] = {
        "trill": True,
        "version": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "env": verify_env(),
        "stack": stack_files_present(),
        "workflow": run_json("dynamic_workflow.py"),
        "repos": run_json("multi_repo_status.py"),
        "ci": run_json("ci_status.py") if verify_env()["gh_available"] else {"ok": None, "skipped": "gh not available"},
        "evals": {},
        "self_heal": None,
        "overnight_started": False,
    }

    wf = report["workflow"]
    eval_modes: list[str] = list(wf.get("eval_modes") or ["unit"])

    if args.mode == "pre-ship":
        for m in ("frontend", "router", "unit", "full"):
            if m not in eval_modes:
                eval_modes.append(m)
    elif args.mode == "full":
        eval_modes = list(dict.fromkeys(eval_modes + ["frontend", "router", "unit"]))

    for mode in eval_modes:
        key = mode if mode in ("unit", "frontend", "router", "full") else "unit"
        if key not in report["evals"]:
            report["evals"][key] = run_json("eval_loop.py", "--mode", key)

    if args.mode == "heal":
        primary = eval_modes[0] if eval_modes else "unit"
        if primary not in ("unit", "frontend", "router", "full"):
            primary = "unit"
        report["self_heal"] = run_json("self_heal.py", "--mode", primary, "--budget", str(args.budget))

    if args.mode in ("overnight", "full") and args.run_overnight:
        proc = subprocess.Popen(
            [sys.executable, str(ROOT / "run_tests.py"), "overnight"],
            cwd=str(ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        report["overnight_started"] = True
        report["overnight_pid"] = proc.pid

    all_eval_ok = all(v.get("ok") for v in report["evals"].values()) if report["evals"] else True
    ci_ok = report["ci"].get("ok") is not False  # None/skipped is ok
    heal_ok = report["self_heal"].get("ok") if report.get("self_heal") else True

    report["ok"] = all_eval_ok and ci_ok and heal_ok
    report["actions"] = build_actions(report)
    report["agent_phases"] = [
        {"order": 1, "name": "route", "tool": "dynamic_workflow.py", "slash": None},
        {"order": 2, "name": "parallel-audit", "tool": None, "slash": "/parallel-audit"},
        {"order": 3, "name": "eval", "tool": "eval_loop.py", "slash": "/fix-tests"},
        {"order": 4, "name": "review-chain", "tool": None, "slash": "subagentStop → eval"},
        {"order": 5, "name": "overnight", "tool": "run_tests.py overnight", "slash": "/overnight"},
        {"order": 6, "name": "multi-repo", "tool": "multi_repo_status.py", "slash": "/multi-repo-ship"},
        {"order": 7, "name": "ci", "tool": "ci_status.py", "slash": "uploadm8-ascended-ci"},
        {"order": 8, "name": "ship", "tool": None, "slash": "/ship-backend /ship-frontend"},
    ]
    report["mcp_tools"] = ["eval_run", "self_heal", "workflow_plan", "ci_status", "multi_repo_status", "trill_run"]
    report["slash_entry"] = "/trill"

    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
