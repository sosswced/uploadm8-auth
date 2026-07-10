#!/usr/bin/env python3
"""
UploadM8 unified session — ONE Chrome window, ONE run.

  Master-admin login (.env credentials)
  → upload video + .map to TikTok profile (E2E_UPLOAD_PLATFORMS=tiktok)
  → worker-safe quiet period (90s) so FFmpeg can start without API flood
  → slow page tour while upload processes (light API, skip heavy ML/KPI pages)
  → deferred API sweep + dashboard/queue verify when pipeline finishes

  python scripts/run_live_demo_journey.py --pipeline-timeout-min 120
  .\\tools\\run_live_demo.ps1

Requires .env:
  E2E_MASTER_ADMIN_EMAIL / E2E_MASTER_ADMIN_PASSWORD
  E2E_TARGET_USER_ID / E2E_TARGET_USER_NAME (optional; defaults to Johnny Omeadows)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except Exception:
    pass

from tests.e2e.helpers.browser_session import single_browser_session
from tests.e2e.helpers.config import e2e_base_url, e2e_master_email, e2e_target_user_id, e2e_target_user_name
from tests.e2e.helpers.live_demo import LiveDemoLog, resolve_demo_paths, run_live_demo_journey


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Upload + admin marathon (single Chrome window, admin while upload pending)"
    )
    parser.add_argument(
        "--video",
        default=os.environ.get("E2E_TEST_VIDEO", r"C:\Users\Earl\Videos\20250301_0058_CAM_EVNT.MP4"),
    )
    parser.add_argument(
        "--telemetry",
        default=os.environ.get("E2E_TEST_TELEMETRY_MAP", r"C:\Users\Earl\Videos\20250301_0058_CAM_EVNT.map"),
    )
    parser.add_argument("--base-url", default=e2e_base_url())
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--pipeline-timeout-min", type=int, default=120)
    parser.add_argument("--api-per-page", type=int, default=2, help="OpenAPI GET cases per page (worker-safe caps lower)")
    parser.add_argument("--no-worker-safe", action="store_true", help="Disable quiet period + slow tour (not recommended locally)")
    parser.add_argument("--include-slow-api", action="store_true", help="Include coach/KPI/ML-heavy GETs")
    parser.add_argument("--skip-api-smoke", action="store_true", help="Skip OpenAPI GET sweep")
    args = parser.parse_args(argv)

    base = args.base_url.rstrip("/")
    # Single origin only — never fall back to a second static http.server port.
    os.environ["E2E_BASE_URL"] = base
    os.environ.setdefault("E2E_HEADED", "0" if args.headless else "1")
    os.environ.setdefault("RATE_LIMIT_LOOPBACK_BYPASS", "1")
    os.environ.setdefault("E2E_TARGET_USER_ID", "ae995094-abb6-4a41-8d51-460ca8f0fd8c")
    os.environ.setdefault("E2E_TARGET_USER_NAME", "Johnny Omeadows")
    # When invoked under /TUP, keep all-platform + persona defaults from tup.py.
    if os.environ.get("E2E_TUP", "").lower() in ("1", "true", "yes"):
        os.environ.setdefault("E2E_UPLOAD_PLATFORMS", "all")
        os.environ.setdefault("E2E_USE_PERSONA", "1")
    os.environ.setdefault("E2E_WORKER_SAFE", "0" if args.no_worker_safe else "1")
    os.environ.setdefault("E2E_UPLOAD_QUIET_SEC", "90")
    os.environ.setdefault("E2E_UPLOAD_PAGE_DELAY_MS", "10000")
    os.environ.setdefault("E2E_WORKER_SAFE_API_PER_PAGE", "1")
    os.environ.setdefault("E2E_WORKER_SAFE_MAX_CLICKS", "4")
    os.environ.setdefault("E2E_REQUEST_DELAY_MS", "1500")
    os.environ.setdefault("E2E_SMOKE_READ_TIMEOUT_S", "45")
    os.environ.setdefault("E2E_API_TIMEOUT_S", "60")

    from tests.e2e.helpers.api_ready import wait_for_api_ready

    ready = wait_for_api_ready(base, timeout_s=120.0, require_db=True)
    if not ready.get("ok"):
        print(json.dumps({"ok": False, "error": "api_not_ready", "ready": ready}, indent=2))
        return 2
    print(f"API/DB ready @ {base} (attempt {ready.get('attempt')})", flush=True)
    video, telemetry = resolve_demo_paths(Path(args.video), Path(args.telemetry) if args.telemetry else None)
    log = LiveDemoLog()
    report: dict = {
        "base_url": base,
        "admin_email": e2e_master_email(),
        "target_user_id": e2e_target_user_id(),
        "target_user_name": e2e_target_user_name(),
        "video": str(video),
        "telemetry": str(telemetry) if telemetry else None,
        "started_at": time.time(),
    }

    print(f"Unified session -> {base}")
    print(f"Admin: {e2e_master_email()}")
    print(f"Target user: {e2e_target_user_name()} ({e2e_target_user_id()})")
    print(f"Video: {video}")
    if telemetry:
        print(f"Telemetry: {telemetry}")
    ws = os.environ.get("E2E_WORKER_SAFE", "1")
    print(
        f"ONE Chrome window — worker_safe={ws} (quiet 90s, ~10s/page, light API while FFmpeg runs).",
        flush=True,
    )

    try:
        with single_browser_session(headed=not args.headless) as page:
            result = run_live_demo_journey(
                page,
                base,
                video=video,
                telemetry=telemetry,
                pipeline_timeout_s=max(600, args.pipeline_timeout_min * 60),
                api_per_page=args.api_per_page,
                include_slow_api=args.include_slow_api,
                skip_api_smoke=args.skip_api_smoke,
                log=log,
            )
        report["ok"] = result.get("ok", True)
        report["upload"] = result.get("upload")
        report["upload_ids"] = result.get("upload_ids")
        report["background_checks"] = result.get("background_checks")
        report["admin_ui"] = result.get("admin_ui")
        report["pages_visited"] = result.get("pages_visited", [])
        report["steps"] = result.get("steps", log.steps)
        if not args.headless:
            print("Session complete — browser stays open 45s for review", flush=True)
            time.sleep(45)
    except Exception as e:
        report["ok"] = False
        report["error"] = str(e)
        report["steps"] = log.steps
        print(f"FAILED: {e}", flush=True)

    report["finished_at"] = time.time()
    report["duration_s"] = int(report["finished_at"] - report.get("started_at", report["finished_at"]))

    bg = report.get("background_checks") or {}
    if bg.get("failures"):
        print(f"Background check failures: {bg['failures']}", flush=True)
        for sample in (bg.get("failed_samples") or [])[:5]:
            print(f"  {sample.get('category')} {sample.get('id')}: {sample.get('detail', '')[:80]}", flush=True)
    api_stats = bg.get("api") or {}
    if api_stats:
        print(
            f"API sweep: {api_stats.get('run', 0)}/{api_stats.get('total', 0)} "
            f"({api_stats.get('fail', 0)} fail)",
            flush=True,
        )
    print(f"Pages visited: {len(report.get('pages_visited', []))}", flush=True)
    print(f"Upload status: {(report.get('upload') or {}).get('status')}", flush=True)

    out = ROOT / "tests" / "e2e" / "artifacts" / f"live_demo_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"Report: {out}")
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
