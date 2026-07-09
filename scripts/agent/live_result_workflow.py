#!/usr/bin/env python3
"""
Plan heal / eval actions from live demo + checklist + Sentry artifacts (/TUP).

Unlike dynamic_workflow.py (git diff), this reads journey results.

Examples:
  python scripts/agent/live_result_workflow.py --json
  python scripts/agent/live_result_workflow.py --live-demo path/to/live_demo.json --json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "tests" / "e2e" / "artifacts"
DEFAULT_SENTRY = ARTIFACTS / "sentry_triage.json"


def _parse_iso(raw: Any) -> datetime | None:
    if not raw or not isinstance(raw, str):
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def _mtime(path: Path) -> datetime | None:
    if not path.is_file():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def find_latest_live_demo(prefer: Path | None = None) -> Path | None:
    if prefer and prefer.is_file():
        return prefer
    if not ARTIFACTS.is_dir():
        return None
    files = sorted(ARTIFACTS.glob("live_demo_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def load_json(path: Path | None) -> dict[str, Any] | None:
    if not path or not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _collect_api_5xx(report: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for page in report.get("pages_visited") or []:
        if not isinstance(page, dict):
            continue
        for item in page.get("api_5xx") or []:
            if isinstance(item, str):
                paths.append(item)
            elif isinstance(item, dict) and item.get("path"):
                paths.append(str(item["path"]))
    bg = report.get("background_checks") or {}
    for sample in bg.get("failed_samples") or []:
        if not isinstance(sample, dict):
            continue
        detail = str(sample.get("detail") or sample.get("id") or "")
        if "5" in detail or sample.get("category") == "api":
            paths.append(detail[:120])
    # de-dupe preserve order
    seen: set[str] = set()
    out: list[str] = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out[:30]


def plan_from_artifacts(
    *,
    live_demo: dict[str, Any] | None,
    live_path: Path | None,
    sentry: dict[str, Any] | None,
    checklist_failed: int | None,
    max_age_hours: float = 12.0,
) -> dict[str, Any]:
    signals: list[str] = []
    eval_modes: list[str] = ["unit"]
    pytest_focus: list[str] = []
    skills: list[str] = ["uploadm8-eval-loop"]
    failures: list[dict[str, str]] = []

    age_ok = True
    if live_path:
        mt = _mtime(live_path)
        if mt:
            age_h = (datetime.now(timezone.utc) - mt).total_seconds() / 3600.0
            if age_h > max_age_hours:
                age_ok = False
                signals.append(f"live_demo_stale:{age_h:.1f}h")

    if live_demo is None:
        signals.append("live_demo_missing")
        skills.append("uploadm8-routines")
    else:
        if live_demo.get("ok") is False or live_demo.get("error"):
            signals.append("live_demo_failed")
            err = str(live_demo.get("error") or "journey failed")[:200]
            failures.append({"file": "tests/e2e/helpers/live_demo.py", "nodeid": err})
            pytest_focus.extend(["upload", "queue", "live_demo"])

        upload = live_demo.get("upload") or {}
        status = str(upload.get("status") or "").lower()
        if status in ("failed", "error", "cancelled", "partial"):
            signals.append(f"upload_status:{status}")
            pytest_focus.extend(["upload", "worker"])
            eval_modes.append("router")
        elif not status and live_demo.get("ok") is not False:
            # Missing terminal status while claiming ok is suspicious
            if not (live_demo.get("upload_ids") or upload.get("id")):
                signals.append("upload_ids_missing")

        api_5xx = _collect_api_5xx(live_demo)
        for p in api_5xx:
            signals.append(f"api_5xx:{p}")
        if api_5xx:
            eval_modes.append("router")
            pytest_focus.append("api")

        bg = live_demo.get("background_checks") or {}
        if bg.get("failures"):
            signals.append(f"background_failures:{bg.get('failures')}")

    if checklist_failed is not None and checklist_failed > 0:
        signals.append(f"checklist_fail:{checklist_failed}")
        skills.append("uploadm8-routines")

    if sentry:
        actionable = sentry.get("actionable_count")
        if actionable is None:
            signals.append("sentry_counts_missing")
        elif int(actionable) > 0:
            signals.append(f"sentry_actionable:{actionable}")
            skills.append("sentry-workflow")
        if sentry.get("cleared") is False:
            signals.append("sentry_not_cleared")
            skills.append("sentry-workflow")
        if not sentry.get("mcp_verified"):
            signals.append("sentry_mcp_unverified")
    else:
        signals.append("sentry_artifact_missing")

    # de-dupe eval modes / skills
    eval_modes = list(dict.fromkeys(eval_modes))
    skills = list(dict.fromkeys(skills))
    pytest_focus = list(dict.fromkeys(pytest_focus))

    api_bad = any(s.startswith("api_5xx:") for s in signals)
    upload_bad = any(s.startswith("upload_status:") for s in signals)
    sentry_bad = any(
        s.startswith("sentry_actionable:") or s == "sentry_not_cleared" for s in signals
    )
    checklist_bad = any(s.startswith("checklist_fail:") for s in signals)
    journey_ok = bool(live_demo and live_demo.get("ok") is True)

    ok = (
        journey_ok
        and age_ok
        and not api_bad
        and not upload_bad
        and not sentry_bad
        and not checklist_bad
        and "live_demo_missing" not in signals
        and "live_demo_failed" not in signals
    )

    agent_action = (
        "none — live + unit signals clear"
        if ok
        else (
            "eval + attack signals via /TUP --heal-only or self_heal --source auto "
            "(do NOT re-run upload/sign-in; mimic artifact is enough)"
        )
    )

    return {
        "workflow": "tup-heal" if not ok else "tup-green",
        "ok": ok,
        "signals": signals,
        "eval_modes": eval_modes,
        "pytest_focus": pytest_focus,
        "skills": skills,
        "failures": failures,
        "failure_count": len(failures) + sum(1 for s in signals if s.startswith("api_5xx:")),
        "live_demo_path": str(live_path.resolve()) if live_path else None,
        "agent_action": agent_action,
        "budget_skips": ["pikzels_mutate_after_setup", "thumbnail_studio_generate"],
        "suggested_command": (
            "python scripts/agent/eval_loop.py --mode unit --json"
            if ok
            else (
                "python run_tests.py unit -k "
                + (" or ".join(pytest_focus[:4]) if pytest_focus else "upload")
            )
        ),
    }


def _checklist_failed_count() -> int | None:
    try:
        sys.path.insert(0, str(ROOT))
        from tests.e2e.helpers.checklist_recorder import (
            find_latest_checklist_xlsx,
            summarize_checklist_excel,
        )
    except ImportError:
        return None
    path = find_latest_checklist_xlsx()
    if not path:
        return None
    summary = summarize_checklist_excel(path)
    return int(summary.get("failed") or 0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan heal actions from /TUP live artifacts")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--live-demo", help="Path to live_demo_*.json")
    parser.add_argument("--sentry-artifact", default=str(DEFAULT_SENTRY))
    parser.add_argument("--max-age-hours", type=float, default=12.0)
    parser.add_argument("--skip-checklist", action="store_true")
    args = parser.parse_args()

    live_path = find_latest_live_demo(Path(args.live_demo) if args.live_demo else None)
    live_demo = load_json(live_path)
    sentry = load_json(Path(args.sentry_artifact))
    checklist_failed = None if args.skip_checklist else _checklist_failed_count()

    report = plan_from_artifacts(
        live_demo=live_demo,
        live_path=live_path,
        sentry=sentry,
        checklist_failed=checklist_failed,
        max_age_hours=args.max_age_hours,
    )

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"workflow={report['workflow']} ok={report['ok']}")
        for s in report["signals"]:
            print(f"  signal: {s}")
        print(f"action: {report['agent_action']}")
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
