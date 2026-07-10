#!/usr/bin/env python3
"""
/TUP — Test Upload Pipeline: combined overnight proof + fix-tests heal.

  1. Live journey: sign-in → upload (all platforms + persona) → browse → confirm
  2. Pikzels only on first setup run of the cycle (then skip until /333 reset)
  3. Optional pytest overnight marker suite
  4. live_result_workflow → self_heal --source auto
  5. Emit ship-gate inputs (live_demo + heal report)

Examples:
  python scripts/agent/tup.py --json
  python scripts/agent/tup.py --skip-overnight-pytest --pipeline-timeout-min 120
  python scripts/agent/tup.py --reset-pikzels
  python scripts/agent/tup.py --heal-only
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
AGENT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "tests" / "e2e" / "artifacts"
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except Exception:
    pass


def _run_json(script: str, *args: str) -> dict[str, Any]:
    proc = subprocess.run(
        [sys.executable, str(AGENT / script), *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    raw = (proc.stdout or "").strip()
    try:
        data = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        try:
            data = json.loads(raw.splitlines()[-1] if raw else "{}")
        except json.JSONDecodeError:
            data = {"ok": False, "parse_error": True, "raw": raw[-2000:]}
    if not isinstance(data, dict):
        data = {"ok": False, "error": "non-object json"}
    data.setdefault("exit_code", proc.returncode)
    return data


def _configure_tup_env(*, force_pikzels: bool, skip_pikzels: bool) -> None:
    os.environ["E2E_TUP"] = "1"
    os.environ.setdefault("E2E_UPLOAD_PLATFORMS", "all")
    os.environ.setdefault("E2E_USE_PERSONA", "1")
    os.environ.setdefault("E2E_SKIP_MUTATIONS", "1")
    os.environ.setdefault("E2E_WORKER_SAFE", "1")
    os.environ.setdefault("E2E_INCLUDE_SLOW_API", "0")
    os.environ.setdefault("RATE_LIMIT_LOOPBACK_BYPASS", "1")
    # Single-origin local API only — never point E2E at production BASE_URL or :8080 static.
    base = (os.environ.get("E2E_BASE_URL") or "").strip().rstrip("/")
    if not (
        base.startswith("http://127.")
        or base.startswith("http://localhost")
        or base.startswith("http://[::1]")
    ):
        os.environ["E2E_BASE_URL"] = "http://127.0.0.1:8000"
    os.environ.setdefault("E2E_WORKER_SAFE_API_PER_PAGE", "1")
    os.environ.setdefault("E2E_REQUEST_DELAY_MS", "1500")
    os.environ.setdefault("E2E_SMOKE_READ_TIMEOUT_S", "45")
    os.environ.setdefault("E2E_API_TIMEOUT_S", "60")
    if force_pikzels:
        os.environ["E2E_FORCE_PIKZELS"] = "1"
    if skip_pikzels:
        os.environ["E2E_SKIP_PIKZELS"] = "1"


def _run_live_journey(
    *,
    video: str | None,
    telemetry: str | None,
    pipeline_timeout_min: int,
    headless: bool,
    skip_api_smoke: bool,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_live_demo_journey.py"),
        "--pipeline-timeout-min",
        str(pipeline_timeout_min),
    ]
    if video:
        cmd.extend(["--video", video])
    if telemetry:
        cmd.extend(["--telemetry", telemetry])
    if headless:
        cmd.append("--headless")
    if skip_api_smoke:
        cmd.append("--skip-api-smoke")

    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    # Journey writes live_demo_*.json; pick newest
    from scripts.agent.live_result_workflow import find_latest_live_demo, load_json

    path = find_latest_live_demo()
    report = load_json(path) or {}
    return {
        "ok": proc.returncode == 0 and bool(report.get("ok", proc.returncode == 0)),
        "exit_code": proc.returncode,
        "live_demo_path": str(path) if path else None,
        "upload_status": (report.get("upload") or {}).get("status"),
        "upload_ids": report.get("upload_ids"),
        "stdout_tail": (proc.stdout or "")[-1200:],
        "stderr_tail": (proc.stderr or "")[-800:],
    }


def _run_overnight_pytest() -> dict[str, Any]:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    junit = ARTIFACTS / "overnight_report.xml"
    html = ARTIFACTS / "overnight_report.html"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "run_tests.py"),
            "overnight",
            "-v",
            "--tb=short",
            "--timeout=600",
            f"--html={html}",
            "--self-contained-html",
            f"--junitxml={junit}",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return {
        "ok": proc.returncode == 0,
        "exit_code": proc.returncode,
        "junit": str(junit),
        "html": str(html),
        "stdout_tail": (proc.stdout or "")[-800:],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="/TUP — Test Upload Pipeline")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--video", default=os.environ.get("E2E_TEST_VIDEO"))
    parser.add_argument("--telemetry", default=os.environ.get("E2E_TEST_TELEMETRY_MAP"))
    parser.add_argument("--pipeline-timeout-min", type=int, default=120)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--skip-api-smoke", action="store_true")
    parser.add_argument("--skip-overnight-pytest", action="store_true")
    parser.add_argument("--skip-journey", action="store_true", help="Heal from existing live_demo only")
    parser.add_argument("--heal-only", action="store_true", help="Alias: skip journey + overnight pytest")
    parser.add_argument("--skip-heal", action="store_true")
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument("--force-pikzels", action="store_true", help="Use Pikzels even if gate consumed")
    parser.add_argument("--skip-pikzels", action="store_true", help="Never use Pikzels this run")
    parser.add_argument("--reset-pikzels", action="store_true", help="Reset once-per-setup gate (post-/333)")
    parser.add_argument("--status", action="store_true", help="Print pikzels gate + latest plan only")
    args = parser.parse_args(argv)

    from tests.e2e.helpers.pikzels_gate import gate_status, reset_after_ship, should_use_pikzels

    if args.reset_pikzels:
        path = reset_after_ship(note="manual --reset-pikzels")
        out = {"ok": True, "reset_pikzels": True, "path": str(path), "gate": gate_status()}
        print(json.dumps(out, indent=2) if args.json else f"Pikzels gate reset → {path}")
        return 0

    if args.status:
        plan = _run_json("live_result_workflow.py", "--json")
        out = {"gate": gate_status(), "plan": plan, "will_use_pikzels": should_use_pikzels()}
        print(json.dumps(out, indent=2))
        return 0 if plan.get("ok") else 1

    if args.heal_only:
        args.skip_journey = True
        args.skip_overnight_pytest = True

    _configure_tup_env(force_pikzels=args.force_pikzels, skip_pikzels=args.skip_pikzels)

    report: dict[str, Any] = {
        "tup": True,
        "version": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platforms": os.environ.get("E2E_UPLOAD_PLATFORMS", "all"),
        "use_persona": os.environ.get("E2E_USE_PERSONA", "1"),
        "will_use_pikzels": should_use_pikzels(),
        "gate_before": gate_status(),
        "journey": None,
        "overnight_pytest": None,
        "live_plan": None,
        "self_heal": None,
        "ok": False,
    }

    if not args.skip_journey:
        from tests.e2e.helpers.api_ready import wait_for_api_ready

        ready = wait_for_api_ready(os.environ.get("E2E_BASE_URL"), timeout_s=120.0)
        report["api_ready"] = ready
        if not ready.get("ok"):
            report["ok"] = False
            report["agent_action"] = (
                "Start a single uvicorn on 127.0.0.1:8000 (SERVE_FRONTEND_STATIC=1); "
                "do not run a second http.server port. Re-run /TUP when /api/health returns db:true."
            )
            ARTIFACTS.mkdir(parents=True, exist_ok=True)
            out_path = ARTIFACTS / f"tup_{time.strftime('%Y%m%d_%H%M%S')}.json"
            out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
            report["artifact"] = str(out_path)
            if args.json:
                print(json.dumps(report, indent=2, default=str))
            else:
                print(f"TUP: RED — API not ready: {ready}")
            return 2

        print(
            f"[TUP] Live journey — platforms={report['platforms']} "
            f"persona={report['use_persona']} pikzels={report['will_use_pikzels']}",
            flush=True,
        )
        report["journey"] = _run_live_journey(
            video=args.video,
            telemetry=args.telemetry,
            pipeline_timeout_min=args.pipeline_timeout_min,
            headless=args.headless,
            skip_api_smoke=args.skip_api_smoke,
        )
    else:
        report["journey"] = {"ok": True, "skipped": True}

    if not args.skip_overnight_pytest:
        print("[TUP] Overnight pytest marker suite…", flush=True)
        report["overnight_pytest"] = _run_overnight_pytest()
    else:
        report["overnight_pytest"] = {"ok": True, "skipped": True}

    report["live_plan"] = _run_json("live_result_workflow.py", "--json")

    if not args.skip_heal:
        print("[TUP] Self-heal (source=auto)…", flush=True)
        report["self_heal"] = _run_json(
            "self_heal.py",
            "--mode",
            "unit",
            "--source",
            "auto",
            "--budget",
            str(args.budget),
            "--json",
        )
    else:
        report["self_heal"] = {"ok": True, "skipped": True}

    report["gate_after"] = gate_status()
    journey_ok = bool((report.get("journey") or {}).get("ok", False))
    overnight_ok = bool((report.get("overnight_pytest") or {}).get("ok", False))
    heal_ok = bool((report.get("self_heal") or {}).get("ok", False))
    report["ok"] = journey_ok and overnight_ok and heal_ok
    report["agent_action"] = (
        "continue /TUP phases (checklist → Sentry → ship_gate) — do not /333 until ready_for_333"
        if report["ok"]
        else (
            "eval + attack failures from live_plan/self_heal "
            "(python scripts/agent/tup.py --heal-only) — do NOT re-run the upload mimic"
        )
    )
    report["duration_hint_s"] = None

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out_path = ARTIFACTS / f"tup_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    report["artifact"] = str(out_path)

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        status = "OK" if report["ok"] else "RED"
        print(f"TUP: {status}")
        print(f"  artifact: {out_path}")
        print(f"  pikzels next: {report['gate_after'].get('will_use_pikzels_next')}")
        print(f"  action: {report['agent_action']}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
