#!/usr/bin/env python3
"""
UploadM8 Human App Tour — ONE Chrome window, continuous session.

Real human pace: scroll every page, click safe controls, sweep all API GET
routers while you watch. No pytest browsers, no popup dialogs, loopback rate
limit bypass. Exports Excel checklist at the end.

Usage:
  python scripts/run_human_app_tour.py
  python scripts/run_human_app_tour.py --with-upload
  .\\tools\\run_human_app_tour.ps1

Requires API on :8000 and E2E_MASTER_ADMIN_* in .env.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except Exception:
    pass

from tests.e2e.helpers.browser_session import single_browser_session
from tests.e2e.helpers.checklist_recorder import ChecklistRecorder, export_checklist_excel
from tests.e2e.helpers.config import e2e_base_url
from tests.e2e.helpers.human_app_tour import run_human_app_tour
from tests.e2e.helpers.live_demo import LiveDemoLog

ARTIFACTS = ROOT / "tests" / "e2e" / "artifacts"


def _configure_human_env(*, headed: bool) -> None:
    os.environ.setdefault("E2E_BASE_URL", e2e_base_url())
    os.environ.setdefault("E2E_HEADED", "1" if headed else "0")
    os.environ.setdefault("RATE_LIMIT_LOOPBACK_BYPASS", "1")
    # Full tour — not worker-safe throttling (that skips pages and caps clicks).
    os.environ.setdefault("E2E_WORKER_SAFE", "0")
    os.environ.setdefault("E2E_REQUEST_DELAY_MS", "450")
    os.environ.setdefault("E2E_REQUEST_JITTER_MS", "200")
    os.environ.setdefault("E2E_CLICK_DELAY_MS", "350")
    os.environ.setdefault("E2E_SLOW_MO_MS", "120")
    os.environ.setdefault("E2E_TOUR_PAGE_DELAY_MS", "800")
    os.environ.setdefault("E2E_TOUR_MAX_CLICKS", "28")
    os.environ.setdefault("E2E_TOUR_ADMIN_MAX_CLICKS", "35")
    os.environ.setdefault("E2E_API_PER_PAGE", "12")
    os.environ.setdefault("E2E_SMOKE_READ_TIMEOUT_S", "12")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Human app tour — one browser, full API + UI")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--with-upload", action="store_true", help="Start upload before page tour")
    parser.add_argument("--include-slow-api", action="store_true")
    parser.add_argument("--no-public", action="store_true", help="Skip marketing/public pages")
    parser.add_argument("--keep-open-sec", type=int, default=60, help="Seconds to leave browser open at end")
    args = parser.parse_args(argv)

    _configure_human_env(headed=not args.headless)
    base = (args.base_url or e2e_base_url()).rstrip("/")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    log_path = ARTIFACTS / f"human_tour_{stamp}.jsonl"
    xlsx_path = ARTIFACTS / f"UploadM8_Human_App_Tour_{stamp}.xlsx"

    recorder = ChecklistRecorder(log_path)
    recorder.reset()
    log = LiveDemoLog()

    print("UploadM8 Human App Tour")
    print(f"  Base URL:    {base}")
    print(f"  One window:  headed={not args.headless}")
    print(f"  Rate limit:  loopback bypass ON")
    print(f"  Excel out:   {xlsx_path}")
    print("  Stop other Playwright/checklist runs first.", flush=True)

    exit_code = 1
    result: dict = {}
    try:
        with single_browser_session(headed=not args.headless, timeout_ms=180_000) as page:
            result = run_human_app_tour(
                page,
                base,
                recorder=recorder,
                include_upload=args.with_upload,
                include_slow_api=args.include_slow_api,
                include_public_pages=not args.no_public,
                log=log,
            )
        exit_code = 0 if result.get("ok") else 1
        if not args.headless and args.keep_open_sec > 0:
            print(f"Tour complete — browser stays open {args.keep_open_sec}s for review", flush=True)
            time.sleep(args.keep_open_sec)
    except Exception as e:
        log.note(f"Tour failed: {e}")
        recorder.record(
            check_id="tour.fatal",
            category="Infrastructure",
            name="Human app tour",
            status="FAIL",
            detail=str(e)[:800],
        )
        print(f"FAILED: {e}", flush=True)
        exit_code = 1
    finally:
        items = recorder.load_all()
        if items:
            export_checklist_excel(items, xlsx_path)
            passed = sum(1 for r in items if r.get("status") == "PASS")
            failed = sum(1 for r in items if r.get("status") == "FAIL")
            print(f"\nChecks: {len(items)} | PASS {passed} | FAIL {failed}")
            print(f"Excel: {xlsx_path}")

    bg = result.get("background_checks") or {}
    if bg:
        api = bg.get("api") or {}
        print(
            f"API sweep: {api.get('run', 0)}/{api.get('total', 0)} "
            f"({api.get('fail', 0)} fail)",
            flush=True,
        )
    print(f"Pages toured: {result.get('pages_visited', 0)}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
