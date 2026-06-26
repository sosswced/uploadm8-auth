"""Inventory of static pages for master-admin overnight UI sweeps."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FRONTEND = ROOT / "frontend"

# Mirrors app.py _SMART_LEGACY_REDIRECTS (301 to canonical smart-* pages).
LEGACY_STATIC_REDIRECTS: dict[str, str] = {
    "ai-insights.html": "smart-insights.html",
    "admin-upload-ai-trace.html": "admin-upload-smart-trace.html",
    "ai-social-media-scheduler.html": "smart-social-media-scheduler.html",
    "ai-thumbnail-generator-for-youtube.html": "smart-thumbnail-generator-for-youtube.html",
}


def page_url_matches(current_url: str, rel_path: str) -> bool:
    """True when ``current_url`` is the page or its static legacy redirect target."""
    leaf = rel_path.split("/")[-1]
    if leaf in current_url:
        return True
    canonical = LEGACY_STATIC_REDIRECTS.get(leaf)
    return bool(canonical and canonical in current_url)


def canonical_page_path(rel_path: str) -> str:
    leaf = rel_path.split("/")[-1]
    return LEGACY_STATIC_REDIRECTS.get(leaf, leaf)


# Authenticated static pages without app-shell bootstrap (no um8-shell-ready).
NO_APP_SHELL_PAGES: tuple[str, ...] = (
    "walkthrough.html",
)

# App-shell pages (require auth; sidebar injected).
AUTHENTICATED_PAGES: tuple[str, ...] = (
    "dashboard.html",
    "queue.html",
    "scheduled.html",
    "thumbnail-studio.html",
    "upload.html",
    "groups.html",
    "platforms.html",
    "analytics.html",
    "trill-leaderboard.html",
    "kpi.html",
    "smart-insights.html",
    "settings.html",
    "billing.html",
    "account-management.html",
    "admin-kpi.html",
    "admin.html",
    "admin-billing-catalog.html",
    "admin-stripe-catalog.html",
    "admin-calculator.html",
    "admin-ml-observability.html",
    "admin-marketing.html",
    "admin-incidents.html",
    "admin-billing-weights.html",
    "admin-upload-ai-trace.html",
    "admin-upload-smart-trace.html",
    "admin-wallet.html",
    "admin-users.html",
    "admin-data-integrity.html",
    "color-preferences.html",
    "guide.html",
    "report-bug.html",
    "walkthrough.html",
)

# Public marketing / legal — load without auth (no 500, basic render).
PUBLIC_PAGES: tuple[str, ...] = (
    "index.html",
    "login.html",
    "signup.html",
    "forgot-password.html",
    "how-it-works.html",
    "about.html",
    "contact.html",
    "blog.html",
    "privacy.html",
    "terms.html",
    "cookies.html",
    "security.html",
    "refunds.html",
    "dmca.html",
    "subprocessors.html",
    "support.html",
    "guide.html",
    "report-bug.html",
    "data-deletion.html",
    "platforms.html",
    "multi-platform-video-uploader.html",
    "youtube-tiktok-instagram-scheduler.html",
    "ai-social-media-scheduler.html",
    "smart-social-media-scheduler.html",
    "ai-thumbnail-generator-for-youtube.html",
    "smart-thumbnail-generator-for-youtube.html",
    "social-media-agency-video-workflow.html",
    "compare/buffer.html",
    "compare/hootsuite.html",
    "compare/later.html",
    "compare/metricool.html",
)

# Admin panel, settings, and user management — tabs, billing nav, action cards.
ADMIN_SETTINGS_PAGES: tuple[str, ...] = (
    "settings.html",
    "account-management.html",
    "admin.html",
    "admin-kpi.html",
    "admin-billing-catalog.html",
    "admin-stripe-catalog.html",
    "admin-billing-weights.html",
    "admin-wallet.html",
    "admin-calculator.html",
    "admin-ml-observability.html",
    "admin-marketing.html",
    "admin-incidents.html",
    "admin-upload-ai-trace.html",
    "admin-upload-smart-trace.html",
    "admin-data-integrity.html",
    "billing.html",
    "color-preferences.html",
)

# Sidebar nav links from shared-sidebar.js (authenticated).
SIDEBAR_HREFS: tuple[str, ...] = (
    "dashboard.html",
    "queue.html",
    "scheduled.html",
    "thumbnail-studio.html",
    "upload.html",
    "groups.html",
    "platforms.html",
    "analytics.html",
    "trill-leaderboard.html",
    "kpi.html",
    "smart-insights.html",
    "account-management.html",
    "admin-kpi.html",
    "admin.html",
    "admin-billing-catalog.html",
    "admin-stripe-catalog.html",
    "admin-calculator.html",
    "admin-ml-observability.html",
    "admin-marketing.html",
    "admin-incidents.html",
    "admin-billing-weights.html",
    "admin-upload-ai-trace.html",
    "admin-data-integrity.html",
    "admin-wallet.html",
    "billing.html",
    "guide.html",
    "report-bug.html",
    "settings.html",
)


def page_url(base: str, rel: str) -> str:
    rel = rel.lstrip("/")
    return f"{base.rstrip('/')}/{rel}"
