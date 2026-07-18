"""Pacing while the upload worker runs FFmpeg — avoid starving worker + API."""

from __future__ import annotations

import os
import time

# Pages that hammer ML/KPI/analytics APIs — skip during active pipeline processing.
WORKER_BUSY_PAGES: frozenset[str] = frozenset(
    {
        "admin-kpi.html",
        "admin-ml-observability.html",
        "admin-upload-ai-trace.html",
        "admin-upload-smart-trace.html",
        "admin-marketing.html",
        "smart-insights.html",
        "analytics.html",
        "thumbnail-studio.html",
        "kpi.html",
        "trill-leaderboard.html",
    }
)


def worker_safe_mode() -> bool:
    """Default on for live demo — slow tour so worker.py keeps FFmpeg slots stable."""
    return os.environ.get("E2E_WORKER_SAFE", "1").lower() not in ("0", "false", "no")


def upload_quiet_sec() -> float:
    """No page navigation right after publish — worker claims the job first."""
    default = "90" if worker_safe_mode() else "20"
    try:
        return float(os.environ.get("E2E_UPLOAD_QUIET_SEC", default))
    except ValueError:
        return 90.0 if worker_safe_mode() else 20.0


def upload_page_delay_sec() -> float:
    """Pause between page visits while upload is still processing."""
    if worker_safe_mode():
        try:
            return float(os.environ.get("E2E_UPLOAD_PAGE_DELAY_MS", "10000")) / 1000.0
        except ValueError:
            return 10.0
    from tests.e2e.helpers.human_pace import request_delay_ms

    return request_delay_ms() / 1000.0


def effective_api_per_page(requested: int) -> int:
    if worker_safe_mode():
        try:
            # Default 1 — one GET per page visit while FFmpeg + Neon recover.
            cap = int(os.environ.get("E2E_WORKER_SAFE_API_PER_PAGE", "1"))
        except ValueError:
            cap = 1
        return max(0, min(requested, cap))
    return requested


def effective_max_clicks(requested: int, *, is_admin_page: bool) -> int:
    if worker_safe_mode():
        try:
            return int(os.environ.get("E2E_WORKER_SAFE_MAX_CLICKS", "4"))
        except ValueError:
            return 4
    if is_admin_page:
        return max(requested, 20)
    return requested


def upload_poll_interval_sec() -> float:
    if worker_safe_mode():
        try:
            return float(os.environ.get("E2E_UPLOAD_POLL_SEC", "60"))
        except ValueError:
            return 60.0
    return 15.0


def upload_poll_every_n_pages() -> int:
    if worker_safe_mode():
        try:
            return max(1, int(os.environ.get("E2E_UPLOAD_POLL_EVERY_PAGES", "3")))
        except ValueError:
            return 3
    return 1


def skip_page_while_processing(rel: str) -> bool:
    if not worker_safe_mode():
        return False
    return rel.split("/")[-1] in WORKER_BUSY_PAGES


def pause_between_upload_pages() -> None:
    delay = upload_page_delay_sec()
    if delay > 0:
        time.sleep(delay)
