#!/usr/bin/env python3
"""
UploadM8 live-ready gate — must pass before /333 deploy.

Runs scriptable checks (eval harness, router lint, checklist Excel, TRILL pre-ship).
Agent must separately confirm Sentry (MCP) and bugbot (no critical) before /333.

Examples:
  python scripts/agent/ship_gate.py --json
  python scripts/agent/ship_gate.py --checklist tests/e2e/artifacts/UploadM8_Full_App_Checklist_*.xlsx --json
  python scripts/agent/ship_gate.py --require-overnight-junit tests/e2e/artifacts/overnight_report.xml --json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
AGENT = Path(__file__).resolve().parent
DEFAULT_SENTRY_ARTIFACT = ROOT / "tests" / "e2e" / "artifacts" / "sentry_triage.json"


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


def run_cmd(cmd: list[str]) -> dict[str, Any]:
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return {
        "ok": proc.returncode == 0,
        "exit_code": proc.returncode,
        "stdout_tail": (proc.stdout or "")[-800:],
        "stderr_tail": (proc.stderr or "")[-800:],
    }


def check_router_lint() -> dict[str, Any]:
    script = ROOT / "tools" / "lint_routers.py"
    if not script.is_file():
        return {"ok": True, "skipped": "tools/lint_routers.py missing"}
    result = run_cmd([sys.executable, str(script)])
    result["name"] = "router_lint"
    return result


def check_checklist(xlsx: Path | None) -> dict[str, Any]:
    sys.path.insert(0, str(ROOT))
    try:
        from tests.e2e.helpers.checklist_recorder import (
            find_latest_checklist_xlsx,
            summarize_checklist_excel,
        )
    except ImportError as e:
        return {"ok": False, "error": f"checklist_recorder import failed: {e}"}

    path = find_latest_checklist_xlsx(prefer=str(xlsx) if xlsx else None)
    if not path:
        return {"ok": False, "skipped": False, "error": "No checklist Excel in tests/e2e/artifacts/"}
    summary = summarize_checklist_excel(path)
    failed = int(summary.get("failed") or 0)
    return {
        "ok": failed == 0,
        "path": summary.get("path"),
        "total": summary.get("total"),
        "passed": summary.get("passed"),
        "failed": failed,
        "skipped": summary.get("skipped"),
        "pass_rate": summary.get("pass_rate"),
        "failures_by_category": summary.get("failures_by_category"),
    }


def check_overnight_junit(junit_path: Path) -> dict[str, Any]:
    if not junit_path.is_file():
        return {"ok": False, "error": f"JUnit file not found: {junit_path}"}
    tree = ET.parse(junit_path)
    root = tree.getroot()
    if root.tag == "testsuites":
        suites = root.findall("testsuite")
    else:
        suites = [root]
    failures = errors = skipped = tests = 0
    for suite in suites:
        failures += int(suite.attrib.get("failures", 0))
        errors += int(suite.attrib.get("errors", 0))
        skipped += int(suite.attrib.get("skipped", 0))
        tests += int(suite.attrib.get("tests", 0))
    return {
        "ok": failures == 0 and errors == 0,
        "path": str(junit_path.resolve()),
        "tests": tests,
        "failures": failures,
        "errors": errors,
        "skipped": skipped,
    }


def check_sentry_triage(artifact_path: Path | None) -> dict[str, Any]:
    path = artifact_path or DEFAULT_SENTRY_ARTIFACT
    try:
        from scripts.agent.sentry_triage import load_artifact, validate_triage
    except ImportError:
        return {"ok": False, "error": "sentry_triage module missing"}
    data = load_artifact(path)
    if not data:
        return {"ok": False, "path": str(path), "error": "sentry_triage.json missing — run /overnight Phase 7"}
    ok, blockers = validate_triage(data)
    return {
        "ok": ok,
        "path": str(path.resolve()),
        "unresolved_count": data.get("unresolved_count"),
        "actionable_count": data.get("actionable_count"),
        "cleared": data.get("cleared"),
        "fixed_issue_ids": data.get("fixed_issue_ids") or [],
        "blockers": blockers,
    }


def build_blockers(report: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    for key in ("verify", "router_lint", "checklist", "overnight_junit", "trill_pre_ship", "sentry_triage"):
        block = report.get(key) or {}
        if block.get("skipped"):
            continue
        if not block.get("ok", True):
            if key == "checklist":
                blockers.append(
                    f"Checklist Excel has {block.get('failed', '?')} FAIL rows — "
                    f"run /full-app-checklist and fix before /333"
                )
            elif key == "overnight_junit":
                blockers.append(
                    f"Overnight E2E: {block.get('failures', 0)} failed, "
                    f"{block.get('errors', 0)} errors — run /overnight phase 2"
                )
            elif key == "sentry_triage":
                for b in block.get("blockers") or [block.get("error", "Sentry not cleared")]:
                    blockers.append(f"Sentry: {b}")
            elif key == "trill_pre_ship":
                for mode, ev in (block.get("evals") or {}).items():
                    if not ev.get("ok"):
                        blockers.append(f"TRILL pre-ship {mode} eval red")
            elif key == "router_lint":
                blockers.append("Router lint failed — python tools/lint_routers.py")
            elif key == "verify":
                blockers.append("python run_tests.py verify failed")
            else:
                blockers.append(f"{key} not ok")
    for mode, ev in (report.get("evals") or {}).items():
        if not ev.get("ok"):
            blockers.append(f"eval_loop {mode} red ({ev.get('failure_count', '?')} failures)")
    if not report.get("agent_gates", {}).get("sentry_cleared"):
        blockers.append("Sentry not cleared — run /overnight Phase 7 (sentry-workflow)")
    if not report.get("agent_gates", {}).get("bugbot_no_critical"):
        blockers.append("bugbot critical findings open — run /parallel-audit")
    if not report.get("agent_gates", {}).get("self_healed"):
        blockers.append("Self-heal incomplete — trill.py --mode heal or /fix-tests")
    return blockers


def main() -> int:
    parser = argparse.ArgumentParser(description="UploadM8 /333 ship gate")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--checklist", help="Checklist xlsx path (default: newest artifact)")
    parser.add_argument("--skip-checklist", action="store_true")
    parser.add_argument("--require-overnight-junit", help="Path to overnight junit xml")
    parser.add_argument("--sentry-cleared", action="store_true", help="Agent confirmed Sentry MCP clear")
    parser.add_argument("--sentry-artifact", default=str(DEFAULT_SENTRY_ARTIFACT), help="sentry_triage.json path")
    parser.add_argument("--skip-sentry-artifact", action="store_true")
    parser.add_argument("--bugbot-clear", action="store_true", help="Agent confirmed no critical bugbot")
    parser.add_argument("--self-healed", action="store_true", help="Agent confirmed self-heal green")
    parser.add_argument(
        "--eval-modes",
        default="unit,frontend,router,full",
        help="Comma-separated eval_loop modes (default: unit,frontend,router,full)",
    )
    args = parser.parse_args()

    eval_modes = [m.strip() for m in args.eval_modes.split(",") if m.strip()]

    report: dict[str, Any] = {
        "ship_gate": True,
        "version": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ready_for_333": False,
        "verify": run_cmd([sys.executable, str(ROOT / "run_tests.py"), "verify"]),
        "router_lint": check_router_lint(),
        "evals": {},
        "trill_pre_ship": run_json("trill.py", "--mode", "pre-ship"),
        "agent_gates": {
            "sentry_cleared": args.sentry_cleared,
            "bugbot_no_critical": args.bugbot_clear,
            "self_healed": args.self_healed,
        },
        "next_command_if_green": "/333",
        "next_skill_if_green": "333",
    }

    if not args.skip_checklist:
        checklist_path = Path(args.checklist) if args.checklist else None
        report["checklist"] = check_checklist(checklist_path)
    else:
        report["checklist"] = {"ok": True, "skipped": True}

    if args.require_overnight_junit:
        report["overnight_junit"] = check_overnight_junit(Path(args.require_overnight_junit))

    if not args.skip_sentry_artifact:
        sentry = check_sentry_triage(Path(args.sentry_artifact))
        report["sentry_triage"] = sentry
        if sentry.get("ok"):
            report["agent_gates"]["sentry_cleared"] = True
    elif args.sentry_cleared:
        report["agent_gates"]["sentry_cleared"] = True

    for mode in eval_modes:
        if mode in ("unit", "frontend", "router", "full"):
            report["evals"][mode] = run_json("eval_loop.py", "--mode", mode)

    script_ok = (
        report["verify"].get("ok")
        and report["router_lint"].get("ok", True)
        and report.get("checklist", {}).get("ok", True)
        and all(v.get("ok") for v in report["evals"].values())
        and report["trill_pre_ship"].get("ok", False)
    )
    if args.require_overnight_junit:
        script_ok = script_ok and report.get("overnight_junit", {}).get("ok", False)

    agent_ok = all(report["agent_gates"].values())
    report["ok"] = script_ok and agent_ok
    report["ready_for_333"] = report["ok"]
    report["blockers"] = build_blockers(report)
    if report["ok"]:
        report["blockers"] = []
        report["deploy"] = {
            "command": "/333",
            "skill": ".cursor/skills/333/SKILL.md",
            "note": "User must explicitly request deploy; /333 commits and pushes both repos",
        }

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        status = "READY for /333" if report["ok"] else "BLOCKED"
        print(f"Ship gate: {status}")
        for b in report["blockers"]:
            print(f"  - {b}")
        if report["ok"]:
            print("Run /333 when user asks to deploy.")

    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
