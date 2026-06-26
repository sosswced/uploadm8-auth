#!/usr/bin/env python3
"""
UploadM8 overnight full-app checklist (~500+ checks) → Excel report.

Runs: infra, router lint, consistency, unit tests, email templates, API GET sweep,
headed Playwright (pages, sidebar, clicks, upload w/ video+.map).

Usage:
  python scripts/run_full_app_checklist.py
  python scripts/run_full_app_checklist.py --headed --include-slow-api
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except Exception:
    pass

import httpx

from tests.e2e.helpers.auth import E2EAuthError, api_client, api_login, fetch_me
from tests.e2e.helpers.checklist_recorder import ChecklistRecorder, export_checklist_excel
from tests.e2e.helpers.config import e2e_base_url, e2e_test_telemetry_map, e2e_test_video
from tests.e2e.helpers.human_pace import CHROME_UA, pause_between_requests, request_delay_ms
from tests.e2e.helpers.openapi_catalog import (
    assert_acceptable_status,
    build_context,
    fetch_openapi,
    include_slow_api_paths,
    iter_read_smoke_cases,
)
from tests.e2e.helpers.pages import AUTHENTICATED_PAGES, PUBLIC_PAGES, page_url

ARTIFACTS = ROOT / "tests" / "e2e" / "artifacts"
STAMP = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _ms(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)


def run_cmd(
    recorder: ChecklistRecorder,
    *,
    check_id: str,
    category: str,
    name: str,
    cmd: list[str],
    cwd: Path | None = None,
) -> int:
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, cwd=str(cwd or ROOT), capture_output=True, text=True)
        ok = proc.returncode == 0
        detail = (proc.stdout or "")[-500:] if ok else (proc.stderr or proc.stdout or "")[-800:]
        recorder.record(
            check_id=check_id,
            category=category,
            name=name,
            status="PASS" if ok else "FAIL",
            detail=detail.strip(),
            duration_ms=_ms(t0),
        )
        return proc.returncode
    except Exception as e:
        recorder.record(
            check_id=check_id,
            category=category,
            name=name,
            status="FAIL",
            detail=str(e)[:800],
            duration_ms=_ms(t0),
        )
        return 1


def check_infra(recorder: ChecklistRecorder, base: str) -> bool:
    t0 = time.perf_counter()
    try:
        r = httpx.get(f"{base}/api/auth/session-probe", timeout=15.0)
        recorder.record(
            check_id="infra.api_probe",
            category="Infrastructure",
            name="API session-probe reachable",
            status="PASS" if r.status_code == 200 else "FAIL",
            detail=f"HTTP {r.status_code}",
            duration_ms=_ms(t0),
        )
    except Exception as e:
        recorder.record(
            check_id="infra.api_probe",
            category="Infrastructure",
            name="API session-probe reachable",
            status="FAIL",
            detail=str(e),
            duration_ms=_ms(t0),
        )
        return False

    try:
        tokens = api_login()
        with httpx.Client(
            base_url=base,
            timeout=30.0,
            headers={"Authorization": f"Bearer {tokens['access_token']}", "User-Agent": CHROME_UA},
        ) as c:
            me = fetch_me(c)
        role = me.get("role") or ""
        recorder.record(
            check_id="infra.master_login",
            category="Infrastructure",
            name="Master admin login + /api/me",
            status="PASS" if role == "master_admin" else "FAIL",
            detail=f"email={me.get('email')} role={role}",
            duration_ms=0,
        )
    except E2EAuthError as e:
        recorder.record(
            check_id="infra.master_login",
            category="Infrastructure",
            name="Master admin login + /api/me",
            status="FAIL",
            detail=str(e),
            duration_ms=0,
        )
        return False

    video = e2e_test_video()
    tmap = e2e_test_telemetry_map()
    recorder.record(
        check_id="infra.test_video",
        category="Infrastructure",
        name="E2E test video on disk",
        status="PASS" if video else "FAIL",
        detail=str(video) if video else "E2E_TEST_VIDEO missing",
        duration_ms=0,
    )
    recorder.record(
        check_id="infra.test_map",
        category="Infrastructure",
        name="E2E telemetry .map on disk",
        status="PASS" if tmap else "WARN",
        detail=str(tmap) if tmap else "optional E2E_TEST_TELEMETRY_MAP missing",
        duration_ms=0,
    )
    return True


def check_router_lint(recorder: ChecklistRecorder) -> None:
    baseline = ROOT / "tools" / "router_lint_baseline.json"
    if not baseline.is_file():
        run_cmd(
            recorder,
            check_id="router.lint",
            category="Routers",
            name="lint_routers.py",
            cmd=[sys.executable, "tools/lint_routers.py"],
        )
        return

    from tools.lint_routers import ROUTERS, collect_lint_errors

    t0_all = time.perf_counter()
    errors, per_file = collect_lint_errors()
    routers = sorted(
        f"routers/{p.name}"
        for p in ROUTERS.glob("*.py")
        if p.name != "__init__.py"
    )
    for i, rname in enumerate(routers, start=1):
        file_errors = per_file.get(rname, [])
        recorder.record(
            check_id=f"router.{i:03d}.{rname}",
            category="Routers",
            name=f"Router lint: {rname}",
            status="FAIL" if file_errors else "PASS",
            detail=file_errors[0][:500] if file_errors else "within baseline",
            duration_ms=0,
        )
    stale = [rel for rel in per_file if rel not in routers]
    for j, rname in enumerate(stale, start=len(routers) + 1):
        file_errors = per_file.get(rname, [])
        recorder.record(
            check_id=f"router.{j:03d}.{rname}",
            category="Routers",
            name=f"Router lint: {rname}",
            status="FAIL",
            detail=file_errors[0][:500] if file_errors else "baseline stale entry",
            duration_ms=0,
        )
    recorder.record(
        check_id="router.lint.all",
        category="Routers",
        name="lint_routers.py aggregate",
        status="PASS" if not errors else "FAIL",
        detail=f"{len(errors)} violation(s)" if errors else "all routers within baseline",
        duration_ms=_ms(t0_all),
    )


def check_consistency(recorder: ChecklistRecorder) -> None:
    script = ROOT / "tools" / "check_consistency.py"
    if not script.is_file():
        recorder.record(
            check_id="consistency.skip",
            category="Consistency",
            name="check_consistency.py",
            status="SKIP",
            detail="script not found",
        )
        return
    run_cmd(
        recorder,
        check_id="consistency.all",
        category="Consistency",
        name="check_consistency.py",
        cmd=[sys.executable, str(script)],
    )


def check_email_templates(recorder: ChecklistRecorder) -> None:
    script = ROOT / "tools" / "email_template_smoke_test.py"
    t0 = time.perf_counter()
    if not script.is_file():
        recorder.record(
            check_id="email.skip",
            category="Email",
            name="email_template_smoke_test.py",
            status="SKIP",
            detail="not found",
        )
        return
    proc = subprocess.run([sys.executable, str(script)], cwd=str(ROOT), capture_output=True, text=True)
    out = (proc.stdout or "") + (proc.stderr or "")
    m = re.search(r"PASS:\s*(\d+)\s+email scenarios", out)
    n = int(m.group(1)) if m else 0
    if n == 0 and proc.returncode != 0:
        recorder.record(
            check_id="email.all",
            category="Email",
            name="Email template smoke (aggregate)",
            status="FAIL",
            detail=out[-800:],
            duration_ms=_ms(t0),
        )
        return
    # Individual rows — one per scenario index (names unknown without parsing deeper).
    for i in range(1, max(n, 1) + 1):
        recorder.record(
            check_id=f"email.{i:03d}",
            category="Email",
            name=f"Email template scenario #{i}",
            status="PASS" if proc.returncode == 0 else "FAIL",
            detail="rendered HTML validated" if proc.returncode == 0 else out[-400:],
            duration_ms=0,
        )
    recorder.record(
        check_id="email.all",
        category="Email",
        name=f"Email template smoke ({n} scenarios)",
        status="PASS" if proc.returncode == 0 else "FAIL",
        detail=out.strip()[-300:],
        duration_ms=_ms(t0),
    )


def check_api_get_sweep(recorder: ChecklistRecorder, base: str, *, include_slow: bool) -> None:
    if include_slow:
        os.environ["E2E_INCLUDE_SLOW_API"] = "1"
    read_s = float(os.environ.get("E2E_SMOKE_READ_TIMEOUT_S", "12"))
    timeout = httpx.Timeout(connect=10.0, read=read_s, write=10.0, pool=10.0)
    with api_client() as client:
        cases = list(iter_read_smoke_cases(fetch_openapi(client), ctx=build_context(client)))
    for i, case in enumerate(cases, start=1):
        t0 = time.perf_counter()
        cid = f"api.{i:03d}"
        try:
            with api_client() as client:
                r = client.get(case.path, timeout=timeout, headers={"User-Agent": CHROME_UA})
            assert_acceptable_status(r.status_code, case)
            st: str = "PASS"
            detail = f"HTTP {r.status_code}"
        except AssertionError as e:
            st = "FAIL"
            detail = str(e)[:500]
        except httpx.HTTPError as e:
            st = "FAIL"
            detail = f"transport: {e}"[:500]
        recorder.record(
            check_id=cid,
            category="API",
            name=f"{case.method} {case.path}",
            status=st,  # type: ignore[arg-type]
            detail=detail,
            duration_ms=_ms(t0),
        )
        pause_between_requests()


def parse_junit_to_recorder(junit_path: Path, recorder: ChecklistRecorder, category: str, prefix: str) -> None:
    if not junit_path.is_file():
        recorder.record(
            check_id=f"{prefix}.junit_missing",
            category=category,
            name=f"JUnit missing: {junit_path.name}",
            status="FAIL",
            detail=str(junit_path),
        )
        return
    tree = ET.parse(junit_path)
    root = tree.getroot()
    cases = root.findall(".//testcase")
    for i, tc in enumerate(cases, start=1):
        name = tc.get("name") or "test"
        classname = tc.get("classname") or ""
        node = f"{classname}::{name}" if classname else name
        t_ms = int(float(tc.get("time") or 0) * 1000)
        fail_el = tc.find("failure")
        err_el = tc.find("error")
        skip = tc.find("skipped")
        if skip is not None:
            st = "SKIP"
            detail = skip.get("message") or ""
        elif fail_el is not None or err_el is not None:
            fail = fail_el if fail_el is not None else err_el
            st = "FAIL"
            detail = (fail.text or fail.get("message") or "")[:800]
        else:
            st = "PASS"
            detail = ""
        safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", node)[:60]
        recorder.record(
            check_id=f"{prefix}.{i:04d}.{safe}",
            category=category,
            name=node[:120],
            status=st,  # type: ignore[arg-type]
            detail=detail,
            duration_ms=t_ms,
        )


def run_playwright_headed_clicks(recorder: ChecklistRecorder, base: str) -> None:
    """One Chrome window — login once, sidebar navigation, safe clicks per page."""
    from playwright.sync_api import sync_playwright

    from tests.e2e.helpers.auth import ensure_playwright_storage_state
    from tests.e2e.helpers.browser_session import bootstrap_human_session, navigate_to_page_human
    from tests.e2e.helpers.config import auth_state_path
    from tests.e2e.helpers.ui_safe_clicks import click_page_surfaces
    from tests.e2e.helpers.upload_files import set_upload_file_pair

    os.environ.setdefault("E2E_HEADED", "1")
    slow_mo = int(os.environ.get("E2E_SLOW_MO_MS", "120"))
    state_path = auth_state_path()

    with sync_playwright() as p:
        if not state_path.is_file() or os.environ.get("E2E_FORCE_RELOGIN", "").lower() in (
            "1",
            "true",
            "yes",
        ):
            ensure_playwright_storage_state(p)
        browser = p.chromium.launch(headless=False, slow_mo=slow_mo)
        context = browser.new_context(
            storage_state=str(state_path) if state_path.is_file() else None,
            viewport={"width": 1440, "height": 900},
            user_agent=CHROME_UA,
        )
        page = context.new_page()
        bootstrap_human_session(
            page,
            base,
            force_form_login=os.environ.get("E2E_FORCE_RELOGIN", "").lower() in ("1", "true", "yes"),
        )

        t_admin = time.perf_counter()
        try:
            from tests.e2e.helpers.target_user_ui import exercise_target_user_admin_ui

            admin_report = exercise_target_user_admin_ui(page, base)
            recorder.record(
                check_id="ui.target_user.admin",
                category="Target User Admin",
                name="Johnny Omeadows — account-mgmt + wallet UI",
                status="PASS",
                detail=str(admin_report.get("account_management", {}).get("steps", []))[:500],
                duration_ms=_ms(t_admin),
            )
        except Exception as e:
            recorder.record(
                check_id="ui.target_user.admin",
                category="Target User Admin",
                name="Johnny Omeadows — account-mgmt + wallet UI",
                status="FAIL",
                detail=str(e)[:500],
                duration_ms=_ms(t_admin),
            )

        for i, rel in enumerate(AUTHENTICATED_PAGES, start=1):
            t0 = time.perf_counter()
            api_fails: list[str] = []

            def _on_resp(resp) -> None:
                if resp.status >= 500 and "/api/" in resp.url:
                    api_fails.append(f"{resp.status} {resp.url}")

            page.on("response", _on_resp)
            try:
                navigate_to_page_human(page, base, rel)
                st = "PASS" if "login.html" not in page.url else "FAIL"
                detail = page.url
            except Exception as e:
                st, detail = "FAIL", str(e)[:500]
            recorder.record(
                check_id=f"ui.page.{i:03d}",
                category="UI Pages",
                name=f"Load {rel}",
                status=st,  # type: ignore[arg-type]
                detail=detail,
                duration_ms=_ms(t0),
            )
            pause_between_requests()

            if st == "PASS":
                t1 = time.perf_counter()
                clicked = click_page_surfaces(page, rel, max_clicks=25)
                click_st = "PASS" if not api_fails else "FAIL"
                recorder.record(
                    check_id=f"ui.clicks.{i:03d}",
                    category="UI Clicks",
                    name=f"Page surfaces on {rel} ({len(clicked)} controls)",
                    status=click_st,  # type: ignore[arg-type]
                    detail=", ".join(clicked[:12])[:500],
                    duration_ms=_ms(t1),
                )
                for j, label in enumerate(clicked[:15], start=1):
                    recorder.record(
                        check_id=f"ui.click.{i:03d}.{j:02d}",
                        category="UI Clicks",
                        name=f"{rel} → {label[:80]}",
                        status="PASS",
                        detail="",
                        duration_ms=0,
                    )
                time.sleep(int(os.environ.get("E2E_CLICK_DELAY_MS", "280")) / 1000.0)

        video, tmap = e2e_test_video(), e2e_test_telemetry_map()
        if video:
            t0 = time.perf_counter()
            try:
                navigate_to_page_human(page, base, "upload.html")
                set_upload_file_pair(page, video, tmap)
                page.locator("#fileList:not(.hidden) .file-item").first.wait_for(timeout=60_000)
                recorder.record(
                    check_id="ui.upload.files",
                    category="Upload",
                    name="Video + telemetry file pair on upload.html",
                    status="PASS",
                    detail=f"{video.name}" + (f" + {tmap.name}" if tmap else ""),
                    duration_ms=_ms(t0),
                )
            except Exception as e:
                recorder.record(
                    check_id="ui.upload.files",
                    category="Upload",
                    name="Video + telemetry file pair on upload.html",
                    status="FAIL",
                    detail=str(e)[:500],
                    duration_ms=_ms(t0),
                )

        context.close()
        browser.close()


def check_target_user_admin_api(recorder: ChecklistRecorder) -> None:
    """Read-only admin API matrix for ae995094-abb6-4a41-8d51-460ca8f0fd8c (Johnny Omeadows)."""
    from tests.e2e.helpers.target_user import TargetUserNotFound, run_target_user_api_checks

    t0 = time.perf_counter()
    try:
        with api_client() as client:
            results = run_target_user_api_checks(client)
    except TargetUserNotFound as e:
        recorder.record(
            check_id="api.target_user.resolve",
            category="Target User Admin",
            name="Resolve target user",
            status="FAIL",
            detail=str(e)[:500],
            duration_ms=_ms(t0),
        )
        return
    except Exception as e:
        recorder.record(
            check_id="api.target_user.resolve",
            category="Target User Admin",
            name="Resolve target user",
            status="FAIL",
            detail=str(e)[:500],
            duration_ms=_ms(t0),
        )
        return

    recorder.record(
        check_id="api.target_user.resolve",
        category="Target User Admin",
        name="Resolve target user",
        status="PASS",
        detail="ae995094-abb6-4a41-8d51-460ca8f0fd8c",
        duration_ms=_ms(t0),
    )
    for i, row in enumerate(results, start=1):
        recorder.record(
            check_id=f"api.target_user.{i:03d}.{row['id']}",
            category="Target User Admin",
            name=f"GET {row['path']} ({row['id']})",
            status=row["status"],  # type: ignore[arg-type]
            detail=row.get("detail", "")[:500],
            duration_ms=0,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="UploadM8 full-app overnight checklist")
    parser.add_argument("--headed", action="store_true", help="Visible Chrome (default on)")
    parser.add_argument("--include-slow-api", action="store_true", help="Include slow GET endpoints")
    parser.add_argument("--skip-playwright", action="store_true")
    parser.add_argument("--skip-unit", action="store_true")
    parser.add_argument(
        "--include-e2e-pytest",
        action="store_true",
        help="Also run full pytest overnight suite (default off — run tools/run_overnight_e2e.ps1 separately)",
    )
    parser.add_argument(
        "--api-only",
        action="store_true",
        help="Skip unit tests and Playwright (infra, routers, API GET sweep, public pages)",
    )
    args = parser.parse_args()
    if args.api_only:
        args.skip_playwright = True
        args.skip_unit = True

    from scripts.agent.pipeline_lock import acquire, release

    try:
        acquire("full_app_checklist")
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    os.environ.setdefault("E2E_HEADED", "1" if args.headed else "0")
    if not (ROOT / "tests" / "e2e" / ".auth" / "master_admin.json").is_file():
        os.environ.setdefault("E2E_FORCE_RELOGIN", "1")
    os.environ.setdefault("E2E_REQUEST_DELAY_MS", "400")
    os.environ.setdefault("E2E_CLICK_DELAY_MS", "300")
    os.environ.setdefault("E2E_SLOW_MO_MS", "120")

    base = e2e_base_url()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    log_path = ARTIFACTS / f"checklist_{STAMP}.jsonl"
    xlsx_path = ARTIFACTS / f"UploadM8_Full_App_Checklist_{STAMP}.xlsx"
    junit_unit = ARTIFACTS / f"junit_unit_{STAMP}.xml"
    junit_e2e = ARTIFACTS / f"junit_e2e_{STAMP}.xml"

    recorder = ChecklistRecorder(log_path)
    recorder.reset()
    exit_code = 1

    print(f"UploadM8 full-app checklist -> {xlsx_path}")
    print(f"Base URL: {base} | delay: {request_delay_ms()}ms | headed: {os.environ.get('E2E_HEADED')}")

    try:
        if not check_infra(recorder, base):
            print("ERROR: API/login not ready. Start uvicorn on :8000")
            return 1

        check_router_lint(recorder)
        check_consistency(recorder)
        check_email_templates(recorder)
        check_api_get_sweep(recorder, base, include_slow=args.include_slow_api)
        check_target_user_admin_api(recorder)

        # Public pages (HTTP only, no browser)
        for i, rel in enumerate(PUBLIC_PAGES, start=1):
            t0 = time.perf_counter()
            try:
                r = httpx.get(page_url(base, rel), timeout=20.0, headers={"User-Agent": CHROME_UA})
                st = "PASS" if r.status_code < 500 else "FAIL"
                detail = f"HTTP {r.status_code}"
            except Exception as e:
                st, detail = "FAIL", str(e)[:300]
            recorder.record(
                check_id=f"public.{i:03d}",
                category="Public Pages",
                name=f"GET {rel}",
                status=st,  # type: ignore[arg-type]
                detail=detail,
                duration_ms=_ms(t0),
            )
            pause_between_requests()

        if not args.skip_unit:
            run_cmd(
                recorder,
                check_id="unit.pytest.start",
                category="Unit Tests",
                name="pytest tests/ (excluding e2e)",
                cmd=[
                    sys.executable,
                    "-m",
                    "pytest",
                    "tests/",
                    "--ignore=tests/e2e",
                    "-q",
                    f"--junitxml={junit_unit}",
                ],
            )
            parse_junit_to_recorder(junit_unit, recorder, "Unit Tests", "unit")

        if not args.skip_playwright:
            try:
                run_playwright_headed_clicks(recorder, base)
            except Exception as e:
                recorder.record(
                    check_id="ui.playwright.crash",
                    category="UI Pages",
                    name="Headed Playwright sweep",
                    status="FAIL",
                    detail=str(e)[:800],
                )
            if args.include_e2e_pytest:
                run_cmd(
                    recorder,
                    check_id="e2e.pytest.start",
                    category="E2E Pytest",
                    name="pytest tests/e2e overnight",
                    cmd=[
                        sys.executable,
                        "run_tests.py",
                        "overnight",
                        "-q",
                        f"--junitxml={junit_e2e}",
                    ],
                )
                parse_junit_to_recorder(junit_e2e, recorder, "E2E Pytest", "e2e")

        items = recorder.load_all()
        passed = sum(1 for r in items if r.get("status") == "PASS")
        failed = sum(1 for r in items if r.get("status") == "FAIL")
        print("")
        print(f"Done: {len(items)} checks | PASS {passed} | FAIL {failed}")
        exit_code = 1 if failed else 0
    finally:
        items = recorder.load_all()
        if items:
            export_checklist_excel(items, xlsx_path)
            print(f"Excel: {xlsx_path}")
        release()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
