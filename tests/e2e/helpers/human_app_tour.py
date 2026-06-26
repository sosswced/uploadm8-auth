"""
Human app tour — one Chrome window, real clicks, full API sweep, Excel report.

No pytest subprocesses, no second browser, no popup dialogs.
Interleaves OpenAPI GET checks + router lint while visiting every app page.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import httpx

from tests.e2e.helpers.auth import api_client, close_api_client
from tests.e2e.helpers.background_checks import BackgroundCheckRunner
from tests.e2e.helpers.browser_session import (
    bootstrap_human_session,
    ensure_authed,
    human_scroll,
    navigate_to_page_human,
    wait_for_authenticated_shell,
)
from tests.e2e.helpers.checklist_recorder import ChecklistRecorder
from tests.e2e.helpers.config import e2e_base_url, e2e_master_email
from tests.e2e.helpers.human_pace import CHROME_UA, click_delay_ms, pause_between_requests
from tests.e2e.helpers.live_demo import LiveDemoLog, resolve_demo_paths, start_upload_on_page
from tests.e2e.helpers.pages import (
    ADMIN_SETTINGS_PAGES,
    AUTHENTICATED_PAGES,
    PUBLIC_PAGES,
    page_url,
)
from tests.e2e.helpers.target_user_ui import exercise_target_user_admin_ui
from tests.e2e.helpers.ui_safe_clicks import click_page_surfaces


def _tour_max_clicks(*, is_admin: bool) -> int:
    try:
        base = int(os.environ.get("E2E_TOUR_MAX_CLICKS", "28"))
    except ValueError:
        base = 28
    if is_admin:
        try:
            admin_floor = int(os.environ.get("E2E_TOUR_ADMIN_MAX_CLICKS", "35"))
        except ValueError:
            admin_floor = 35
        return max(base, admin_floor)
    return base


def _api_per_page() -> int:
    try:
        return max(1, int(os.environ.get("E2E_API_PER_PAGE", "12")))
    except ValueError:
        return 12


def _page_pause_sec() -> float:
    try:
        return float(os.environ.get("E2E_TOUR_PAGE_DELAY_MS", "800")) / 1000.0
    except ValueError:
        return 0.8


def _tour_page_order() -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for rel in ADMIN_SETTINGS_PAGES:
        if rel not in seen:
            seen.add(rel)
            ordered.append(rel)
    for rel in AUTHENTICATED_PAGES:
        if rel not in seen:
            seen.add(rel)
            ordered.append(rel)
    return tuple(ordered)


def _record_bg_rows(recorder: ChecklistRecorder, rows: list[dict[str, Any]], *, prefix: str) -> None:
    for i, row in enumerate(rows, start=1):
        cat_map = {
            "api": "API",
            "router": "Routers",
            "target_user": "Target User Admin",
            "consistency": "Consistency",
        }
        recorder.record(
            check_id=f"{prefix}.{row.get('id', i)}",
            category=cat_map.get(row.get("category", ""), row.get("category", "Check")),
            name=row.get("path") or row.get("id") or str(i),
            status=row.get("status", "FAIL"),  # type: ignore[arg-type]
            detail=(row.get("detail") or "")[:800],
        )


def run_human_app_tour(
    page,
    base_url: str,
    *,
    recorder: ChecklistRecorder,
    include_upload: bool = False,
    video: Path | None = None,
    telemetry: Path | None = None,
    include_slow_api: bool = False,
    include_public_pages: bool = True,
    log: LiveDemoLog | None = None,
) -> dict[str, Any]:
    """
    Single continuous browser session: login → every page → scroll → safe clicks
    → API/router checks interleaved → optional upload → Excel via recorder.
    """
    log = log or LiveDemoLog()
    base = base_url.rstrip("/")
    t0 = time.perf_counter()

    bootstrap_human_session(page, base)
    log.note(f"Session started as {e2e_master_email() or 'master admin'} @ {base}")
    recorder.record(
        check_id="tour.session.login",
        category="Infrastructure",
        name="Master admin session in one browser tab",
        status="PASS",
        detail=page.url,
        duration_ms=int((time.perf_counter() - t0) * 1000),
    )

    upload_meta: dict[str, Any] | None = None
    if include_upload:
        try:
            v, t = resolve_demo_paths(video, telemetry)
            stem, upload_ids = start_upload_on_page(page, base, v, t, log)
            upload_meta = {"stem": stem, "upload_ids": upload_ids, "video": str(v)}
            recorder.record(
                check_id="tour.upload.started",
                category="Upload",
                name="Video + telemetry on upload.html",
                status="PASS",
                detail=str(v),
            )
        except Exception as e:
            log.note(f"Upload skipped: {e}")
            recorder.record(
                check_id="tour.upload.started",
                category="Upload",
                name="Video + telemetry on upload.html",
                status="WARN",
                detail=str(e)[:500],
            )

    client = api_client()
    bg = BackgroundCheckRunner(
        log,
        api_per_page=_api_per_page(),
        include_slow_api=include_slow_api,
        skip_api_smoke=False,
    )
    bg.prepare(client)
    static = bg.run_startup_static()
    if static.get("consistency"):
        _record_bg_rows(recorder, [static["consistency"]], prefix="consistency")

    pages_visited: list[dict[str, Any]] = []
    tour = _tour_page_order()

    try:
        for i, rel in enumerate(tour, start=1):
            ensure_authed(page, base)
            leaf = rel.split("/")[-1]
            is_admin = leaf in ADMIN_SETTINGS_PAGES
            page_t0 = time.perf_counter()
            page_report: dict[str, Any] = {"page": rel, "clicks": [], "api_5xx": []}

            batch, client = bg.run_page_batch(
                client,
                rel,
                include_target_user=(i % 4 == 1),
                include_router_lint=True,
            )
            if batch.get("router"):
                _record_bg_rows(recorder, [batch["router"]], prefix=f"router.{i:03d}")
            if batch.get("api"):
                _record_bg_rows(recorder, batch["api"], prefix=f"api.{i:03d}")
            if batch.get("target_user"):
                _record_bg_rows(recorder, [batch["target_user"]], prefix=f"target.{i:03d}")

            api_fails: list[str] = []

            def _on_resp(resp) -> None:
                if resp.status >= 500 and "/api/" in resp.url:
                    api_fails.append(f"{resp.status} {resp.url}")

            page.on("response", _on_resp)
            ui_status = "PASS"
            detail = ""
            clicked: list[str] = []
            try:
                navigate_to_page_human(page, base, rel)
                wait_for_authenticated_shell(page)
                human_scroll(page, passes=3)
                clicked = click_page_surfaces(
                    page,
                    rel,
                    max_clicks=_tour_max_clicks(is_admin=is_admin),
                )
                page_report["clicks"] = clicked
                detail = f"{len(clicked)} controls clicked"
                if "login.html" in page.url:
                    ui_status = "FAIL"
                    detail = "redirected to login"
            except Exception as e:
                ui_status = "FAIL"
                detail = str(e)[:500]
                log.note(f"{rel}: {e}")
                if "login.html" in page.url:
                    from tests.e2e.helpers.browser_session import human_login_via_form

                    human_login_via_form(page, base)
            finally:
                try:
                    page.remove_listener("response", _on_resp)
                except Exception:
                    pass

            if api_fails:
                page_report["api_5xx"] = api_fails[:10]
                ui_status = "FAIL"
                detail = (detail + " | " if detail else "") + api_fails[0][:200]

            recorder.record(
                check_id=f"ui.page.{i:03d}.{leaf}",
                category="UI Pages",
                name=f"Tour {rel}",
                status=ui_status,  # type: ignore[arg-type]
                detail=detail,
                duration_ms=int((time.perf_counter() - page_t0) * 1000),
            )
            for j, label in enumerate(clicked[:20], start=1):
                recorder.record(
                    check_id=f"ui.click.{i:03d}.{j:02d}",
                    category="UI Clicks",
                    name=f"{rel} → {label[:80]}",
                    status="PASS",
                    detail="",
                )

            pages_visited.append(page_report)
            log.note(f"[{i}/{len(tour)}] {rel} — {len(clicked)} clicks")
            time.sleep(_page_pause_sec())
            pause_between_requests()

        # Target user admin UI (account-mgmt + wallet) in same window
        t_admin = time.perf_counter()
        try:
            admin_report = exercise_target_user_admin_ui(page, base)
            recorder.record(
                check_id="ui.target_user.admin",
                category="Target User Admin",
                name="Account-mgmt + wallet UI tour",
                status="PASS",
                detail=str(admin_report.get("account_management", {}).get("steps", []))[:500],
                duration_ms=int((time.perf_counter() - t_admin) * 1000),
            )
        except Exception as e:
            recorder.record(
                check_id="ui.target_user.admin",
                category="Target User Admin",
                name="Account-mgmt + wallet UI tour",
                status="FAIL",
                detail=str(e)[:500],
                duration_ms=int((time.perf_counter() - t_admin) * 1000),
            )

        drained, client = bg.drain_remaining_api(client)
        log.note(f"Drained {drained} remaining API GET checks")

        if include_public_pages:
            for k, rel in enumerate(PUBLIC_PAGES, start=1):
                if rel in tour:
                    continue
                t_pub = time.perf_counter()
                try:
                    page.goto(page_url(base, rel), wait_until="domcontentloaded")
                    pause_between_requests()
                    human_scroll(page, passes=2)
                    clicked = click_page_surfaces(page, rel, max_clicks=min(12, _tour_max_clicks(is_admin=False)))
                    st = "PASS" if page.url and "login.html" not in page.url else "WARN"
                    recorder.record(
                        check_id=f"public.{k:03d}",
                        category="Public Pages",
                        name=f"GET {rel}",
                        status=st,  # type: ignore[arg-type]
                        detail=f"HTTP render, {len(clicked)} clicks",
                        duration_ms=int((time.perf_counter() - t_pub) * 1000),
                    )
                except Exception as e:
                    recorder.record(
                        check_id=f"public.{k:03d}",
                        category="Public Pages",
                        name=f"GET {rel}",
                        status="FAIL",
                        detail=str(e)[:400],
                        duration_ms=int((time.perf_counter() - t_pub) * 1000),
                    )
                pause_between_requests()

    finally:
        close_api_client(client)

    summary = bg.summary()
    page_5xx = [f for p in pages_visited for f in (p.get("api_5xx") or [])]
    ok = bool(summary.get("ok") and not page_5xx)
    return {
        "ok": ok,
        "pages_visited": len(pages_visited),
        "background_checks": summary,
        "upload": upload_meta,
        "steps": log.steps,
    }
