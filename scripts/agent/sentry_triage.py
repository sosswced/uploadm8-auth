#!/usr/bin/env python3
"""
Sentry triage artifact for /overnight Phase 6 (sentry-workflow).

Agent runs Sentry MCP (search_issues, fix via sentry-fix-issues), then writes this JSON
so ship_gate.py can verify clearance. Counts alone are not enough: --record requires
--mcp-verified (agent attesting a live MCP search_issues pass in the same session).

Workflow (agent):
  1. find_organizations → find_projects (uploadm8)
  2. search_issues query=is:unresolved level:error project:uploadm8
  3. For each actionable issue: sentry-fix-issues (analyze, patch, unit test)
  4. Re-search until actionable count is 0
  5. Write artifact with MCP attestation:
       python scripts/agent/sentry_triage.py --record --mcp-verified \\
         --unresolved 0 --actionable 0 --fixed "ID1,ID2" --json

Examples:
  python scripts/agent/sentry_triage.py --template
  python scripts/agent/sentry_triage.py --validate
  python scripts/agent/sentry_triage.py --record --mcp-verified --unresolved 0 --actionable 0 --json
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

# Max age of a cleared artifact before ship_gate treats it as stale (hours).
DEFAULT_MAX_AGE_HOURS = 12


def template() -> dict[str, Any]:
    return {
        "version": 2,
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
        "mcp_verified": False,
        "mcp_verified_at": None,
        "recorded_at": None,
        "notes": (
            "Agent: after live Sentry MCP search_issues shows actionable=0, "
            "run --record --mcp-verified (required for ship_gate)"
        ),
    }


def _parse_recorded_at(data: dict[str, Any]) -> datetime | None:
    raw = data.get("mcp_verified_at") or data.get("recorded_at")
    if not raw or not isinstance(raw, str):
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def validate_triage(
    data: dict[str, Any],
    *,
    max_age_hours: float | None = DEFAULT_MAX_AGE_HOURS,
    require_mcp_verified: bool = True,
) -> tuple[bool, list[str]]:
    blockers: list[str] = []
    if not data.get("cleared"):
        blockers.append("sentry_triage.json cleared=false")
    if require_mcp_verified and not data.get("mcp_verified"):
        blockers.append(
            "sentry_triage.json mcp_verified=false — re-run Sentry MCP search_issues, "
            "then: python scripts/agent/sentry_triage.py --record --mcp-verified ..."
        )
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
    if max_age_hours is not None and data.get("cleared"):
        ts = _parse_recorded_at(data)
        if ts is None:
            blockers.append("sentry_triage.json missing mcp_verified_at/recorded_at timestamp")
        else:
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            age_h = (datetime.now(timezone.utc) - ts).total_seconds() / 3600.0
            if age_h > float(max_age_hours):
                blockers.append(
                    f"sentry_triage.json stale ({age_h:.1f}h > {max_age_hours}h) — "
                    "re-run Phase 6 MCP search and --record --mcp-verified"
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
    parser.add_argument(
        "--mcp-verified",
        action="store_true",
        help=(
            "Required with --record for ship_gate clearance: attest that Sentry MCP "
            "search_issues was run in this session and counts match --unresolved/--actionable"
        ),
    )
    parser.add_argument("--unresolved", type=int, default=0)
    parser.add_argument("--actionable", type=int, default=0)
    parser.add_argument("--fixed", default="", help="Comma-separated issue IDs fixed this run")
    parser.add_argument("--waived", default="", help="Comma-separated infra-only issue IDs")
    parser.add_argument("--notes", default="")
    parser.add_argument(
        "--max-age-hours",
        type=float,
        default=DEFAULT_MAX_AGE_HOURS,
        help="Reject cleared artifacts older than this many hours (validate)",
    )
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
        if not args.mcp_verified:
            out = {
                "ok": False,
                "error": (
                    "--mcp-verified is required with --record. "
                    "Run Sentry MCP search_issues first, then re-record with --mcp-verified."
                ),
            }
            print(json.dumps(out, indent=2) if args.json else out["error"])
            return 1
        fixed = [x.strip() for x in args.fixed.split(",") if x.strip()]
        waived = [x.strip() for x in args.waived.split(",") if x.strip()]
        now = datetime.now(timezone.utc).isoformat()
        cleared = args.actionable == 0
        record = {
            **template(),
            "unresolved_count": args.unresolved,
            "actionable_count": args.actionable,
            "fixed_issue_ids": fixed,
            "waived_issue_ids": waived,
            "cleared": cleared,
            "mcp_verified": True,
            "mcp_verified_at": now,
            "recorded_at": now,
            "notes": args.notes
            or "MCP search_issues verified in agent session; counts match this record",
        }
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(json.dumps(record, indent=2), encoding="utf-8")
        ok, blockers = validate_triage(record, max_age_hours=args.max_age_hours)
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
        ok, blockers = validate_triage(data, max_age_hours=args.max_age_hours)
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
        "mcp_tools": [
            "find_organizations",
            "find_projects",
            "search_issues",
            "analyze_issue_with_seer",
            "update_issue",
        ],
        "query": SENTRY_QUERY,
        "dashboard_url": SENTRY_DASHBOARD,
        "artifact_path": str(artifact),
        "record_command": (
            "python scripts/agent/sentry_triage.py --record --mcp-verified "
            "--unresolved 0 --actionable 0 --json"
        ),
        "validate_command": "python scripts/agent/sentry_triage.py --validate --json",
        "ship_gate_note": (
            "ship_gate rejects artifacts without mcp_verified=true or older than "
            f"{DEFAULT_MAX_AGE_HOURS}h; --sentry-cleared alone is not enough"
        ),
    }
    print(json.dumps(workflow, indent=2) if args.json else json.dumps(workflow, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
