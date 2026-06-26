#!/usr/bin/env python3
"""
Sentry triage artifact for /overnight Phase 7 (sentry-workflow).

Agent runs Sentry MCP (search_issues, fix via sentry-fix-issues), then writes this JSON
so ship_gate.py can verify clearance without MCP in Python.

Workflow (agent):
  1. find_organizations → find_projects (uploadm8)
  2. search_issues query=is:unresolved level:error project:uploadm8
  3. For each actionable issue: sentry-fix-issues (analyze, patch, unit test)
  4. Re-search until actionable count is 0 (or document infra-only waivers)
  5. Write artifact: python scripts/agent/sentry_triage.py --record ...

Examples:
  python scripts/agent/sentry_triage.py --template
  python scripts/agent/sentry_triage.py --validate
  python scripts/agent/sentry_triage.py --record --unresolved 0 --actionable 0 --json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT = ROOT / "tests" / "e2e" / "artifacts" / "sentry_triage.json"

SENTRY_ORG = "gillespie-and-gillespie-invest"
SENTRY_PROJECT = "uploadm8"
SENTRY_QUERY = "is:unresolved level:error"
SENTRY_DASHBOARD = (
    f"https://{SENTRY_ORG}.sentry.io/issues/"
    f"?query={SENTRY_QUERY.replace(' ', '+')}"
)


def template() -> dict[str, Any]:
    return {
        "version": 1,
        "organization_slug": SENTRY_ORG,
        "project": SENTRY_PROJECT,
        "query": SENTRY_QUERY,
        "dashboard_url": SENTRY_DASHBOARD,
        "unresolved_count": None,
        "actionable_count": None,
        "waived_issue_ids": [],
        "fixed_issue_ids": [],
        "top_issues": [],
        "cleared": False,
        "recorded_at": None,
        "notes": "Agent: populate after Sentry MCP search_issues + sentry-fix-issues",
    }


def validate_triage(data: dict[str, Any]) -> tuple[bool, list[str]]:
    blockers: list[str] = []
    if not data.get("cleared"):
        blockers.append("sentry_triage.json cleared=false")
    actionable = data.get("actionable_count")
    unresolved = data.get("unresolved_count")
    if actionable is None or unresolved is None:
        blockers.append("sentry_triage.json missing unresolved_count or actionable_count")
    elif int(actionable) > 0:
        blockers.append(
            f"Sentry actionable errors remain: {actionable} — fix in repo, then update_issue resolved"
        )
    elif int(unresolved) > 0:
        resolved = data.get("resolved_issue_ids") or data.get("fixed_issue_ids") or []
        if len(resolved) < int(unresolved):
            blockers.append(
                f"Sentry unresolved={unresolved} but only {len(resolved)} marked resolved — "
                "resolve every addressed issue in Sentry MCP (no ignore/waive)"
            )
    waived = data.get("waived_issue_ids") or []
    if waived:
        blockers.append(
            f"waived_issue_ids not allowed ({len(waived)}); fix or resolve each issue in Sentry"
        )
    return len(blockers) == 0, blockers


def load_artifact(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Sentry triage artifact for overnight ship gate")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--template", action="store_true", help="Write template artifact")
    parser.add_argument("--validate", action="store_true", help="Validate existing artifact")
    parser.add_argument("--artifact", default=str(DEFAULT_ARTIFACT))
    parser.add_argument("--record", action="store_true", help="Write clearance record")
    parser.add_argument("--unresolved", type=int, default=0)
    parser.add_argument("--actionable", type=int, default=0)
    parser.add_argument("--fixed", default="", help="Comma-separated issue IDs fixed this run")
    parser.add_argument("--waived", default="", help="Comma-separated infra-only issue IDs")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    artifact = Path(args.artifact)

    if args.template:
        artifact.parent.mkdir(parents=True, exist_ok=True)
        t = template()
        artifact.write_text(json.dumps(t, indent=2), encoding="utf-8")
        if args.json:
            print(json.dumps({"ok": True, "artifact": str(artifact), "template": t}, indent=2))
        else:
            print(f"Template: {artifact}")
        return 0

    if args.record:
        fixed = [x.strip() for x in args.fixed.split(",") if x.strip()]
        waived = [x.strip() for x in args.waived.split(",") if x.strip()]
        cleared = args.actionable == 0
        record = {
            **template(),
            "unresolved_count": args.unresolved,
            "actionable_count": args.actionable,
            "fixed_issue_ids": fixed,
            "waived_issue_ids": waived,
            "cleared": cleared,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "notes": args.notes,
        }
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(json.dumps(record, indent=2), encoding="utf-8")
        ok, blockers = validate_triage(record)
        out = {"ok": ok, "cleared": cleared, "artifact": str(artifact), "blockers": blockers}
        print(json.dumps(out, indent=2) if args.json else (
            "Sentry CLEARED" if ok else "Sentry BLOCKED: " + "; ".join(blockers)
        ))
        return 0 if ok else 1

    if args.validate:
        data = load_artifact(artifact)
        if not data:
            out = {"ok": False, "error": f"Missing or invalid {artifact}"}
            print(json.dumps(out, indent=2) if args.json else out["error"])
            return 1
        ok, blockers = validate_triage(data)
        out = {"ok": ok, "artifact": str(artifact), "data": data, "blockers": blockers}
        print(json.dumps(out, indent=2) if args.json else (
            "Sentry triage OK" if ok else "BLOCKED: " + "; ".join(blockers)
        ))
        return 0 if ok else 1

    # Default: print workflow for agents
    workflow = {
        "skill": "sentry-workflow → sentry-fix-issues",
        "organization_slug": SENTRY_ORG,
        "project": SENTRY_PROJECT,
        "mcp_tools": ["find_organizations", "find_projects", "search_issues", "analyze_issue_with_seer", "update_issue"],
        "query": SENTRY_QUERY,
        "dashboard_url": SENTRY_DASHBOARD,
        "artifact_path": str(artifact),
        "record_command": (
            "python scripts/agent/sentry_triage.py --record --unresolved 0 --actionable 0 --json"
        ),
        "validate_command": "python scripts/agent/sentry_triage.py --validate --json",
    }
    print(json.dumps(workflow, indent=2) if args.json else json.dumps(workflow, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
