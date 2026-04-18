"""
Inject js/delegated-ui.js after auth-stack and strip common onclick= patterns.
Run: python tools/migrate_onclick_html.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


def inject_delegated(text: str) -> str:
    if "delegated-ui.js" in text:
        return text
    for prefix in ("js/", "../js/"):
        needle = f'<script src="{prefix}auth-stack.js"></script>'
        if needle in text:
            rep = needle + "\n" + f'<script src="{prefix}delegated-ui.js"></script>'
            return text.replace(needle, rep, 1)
    return text


def migrate_html(text: str) -> str:
    t = inject_delegated(text)

    subs: list[tuple[str, str]] = [
        (r'\s+onclick="toggleTheme\(\)"', ' data-um8-fn="toggleTheme"'),
        (r'\s+onclick="history\.back\(\)"', ' data-um8-fn="um8HistoryBack"'),
        (r'\s+onclick="toggleAuthMobileNav\(\)"', ' data-um8-fn="toggleAuthMobileNav"'),
        (r'\s+onclick="togglePasswordVisibility\(\)"', ' data-um8-fn="togglePasswordVisibility"'),
        (r'\s+onclick="toggleHideFigures\(\)"', ' data-um8-fn="toggleHideFigures"'),
        (r'\s+onclick="refreshQueue\(\)"', ' data-um8-fn="refreshQueue"'),
        (r'\s+onclick="retrySelected\(\)"', ' data-um8-fn="retrySelected"'),
        (r'\s+onclick="deleteSelected\(\)"', ' data-um8-fn="deleteSelected"'),
        (r'\s+onclick="clearSelection\(\)"', ' data-um8-fn="clearSelection"'),
        (r'\s+onclick="loadQueue\(\)"', ' data-um8-fn="loadQueue"'),
        (r'\s+onclick="changePage\(-1\)"', ' data-um8-fn="changePage" data-um8-args="[-1]"'),
        (r'\s+onclick="changePage\(1\)"', ' data-um8-fn="changePage" data-um8-args="[1]"'),
        (r'\s+onclick="openQueueEditModal\(\)"', ' data-um8-fn="openQueueEditModal"'),
        (r'\s+onclick="generateQueueThumbnail\(\)"', ' data-um8-fn="generateQueueThumbnail"'),
        (r'\s+onclick="saveQueueEdit\(\)"', ' data-um8-fn="saveQueueEdit"'),
        (r'\s+onclick="dismissUploadBanner\(\)"', ' data-um8-fn="dismissUploadBanner"'),
        (r'\s+onclick="closeQueueEditModal\(\)"', ' data-um8-fn="closeQueueEditModal"'),
        (r'\s+onclick="closeDeleteModal\(\)"', ' data-um8-fn="closeDeleteModal"'),
        (r'\s+onclick="confirmDelete\(\)"', ' data-um8-fn="confirmDelete"'),
        (r'\s+onclick="closeModal\(\'uploadDetailModal\'\)"', ' data-um8-fn="closeModal" data-um8-arg="uploadDetailModal"'),
        (r'\s+onclick="retryUpload\(\)"', ' data-um8-fn="um8RetryUploadCurrent"'),
        (r'\s+onclick="refresh\(\)"', ' data-um8-fn="refresh"'),
        (r'\s+onclick="exportExcel\(\)"', ' data-um8-fn="exportExcel"'),
        (r'\s+onclick="openAnnouncementModal\(\)"', ' data-um8-fn="openAnnouncementModal"'),
        (r'\s+onclick="closeAnnouncementModal\(\)"', ' data-um8-fn="closeAnnouncementModal"'),
        (r'\s+onclick="sendAnnouncement\(\)"', ' data-um8-fn="sendAnnouncement"'),
        (r'\s+onclick="rebuildPlatformKpiRollups\(\)"', ' data-um8-fn="rebuildPlatformKpiRollups"'),
        (r'\s+onclick="testWebhook\(\)"', ' data-um8-fn="testWebhook"'),
        (r'\s+onclick="saveNotificationSettings\(\)"', ' data-um8-fn="saveNotificationSettings"'),
        (r'\s+onclick="startUpload\(\)"', ' data-um8-fn="startUpload"'),
        (r'\s+onclick="retryFailed\(\)"', ' data-um8-fn="retryFailed"'),
        (r'\s+onclick="goToQueueFromUpload\(\)"', ' data-um8-fn="goToQueueFromUpload"'),
        (r'\s+onclick="goToToday\(\)"', ' data-um8-fn="goToToday"'),
        (r'\s+onclick="closeDayPopup\(\)"', ' data-um8-fn="closeDayPopup"'),
        (r'\s+onclick="setView\(\'calendar\'\)"', ' data-um8-fn="setView" data-um8-arg="calendar"'),
        (r'\s+onclick="setView\(\'list\'\)"', ' data-um8-fn="setView" data-um8-arg="list"'),
        (r'\s+onclick="changeMonth\(-1\)"', ' data-um8-fn="changeMonth" data-um8-args="[-1]"'),
        (r'\s+onclick="changeMonth\(1\)"', ' data-um8-fn="changeMonth" data-um8-args="[1]"'),
        (r'\s+onclick="refreshData\(\)"', ' data-um8-fn="refreshData"'),
        (r'\s+onclick="event\.stopPropagation\(\)"', ' data-um8-stop-propagation="1"'),
        (
            r'\s+onclick="window\.open\(\'https://dashboard\.stripe\.com\',\'_blank\'\)"',
            ' data-um8-open-blank="1" data-um8-href="https://dashboard.stripe.com" role="link" tabindex="0"',
        ),
        (r'\s+onclick="toggleCard\(this\)"', ' type="button" data-um8-fn="toggleCard"'),
        (r'\s+onclick="manualRefreshAllAggregates\(\)"', ' data-um8-fn="manualRefreshAllAggregates"'),
        (r'\s+onclick="closeAccountModal\(\)"', ' data-um8-fn="closeAccountModal"'),
        (r'\s+onclick="refreshAccount\(\)"', ' data-um8-fn="refreshAccount"'),
        (r'\s+onclick="disconnectAccount\(\)"', ' data-um8-fn="disconnectAccount"'),
        (r'\s+onclick="loadAccounts\(\)"', ' data-um8-fn="loadAccounts"'),
        (r'\s+onclick="exportToExcel\(\)"', ' data-um8-fn="exportToExcel"'),
        (r'\s+onclick="closeBillingModal\(\)"', ' data-um8-fn="closeBillingModal"'),
        (r'\s+onclick="window\.loadCrmDeepDive\(\)"', ' data-um8-fn="loadCrmDeepDive"'),
        (r'\s+onclick="window\._crmSyncTimelineFromPageHeader\(\)"', ' data-um8-fn="_crmSyncTimelineFromPageHeader"'),
    ]

    for pat, rep in subs:
        t = re.sub(pat, rep, t)

    t = re.sub(
        r'onclick="runEmailJob\(\'([^\']+)\',\s*this\)"',
        lambda m: f'data-um8-fn="runEmailJob" data-um8-args=\'["{m.group(1)}"]\'',
        t,
    )

    t = re.sub(
        r'onclick="scrollToSection\(\'([a-z0-9_]+)\'\)"',
        r'data-um8-fn="scrollToSection" data-um8-arg="\1"',
        t,
    )
    t = re.sub(
        r'onclick="connectAccount\(\'([a-z]+)\'\)"',
        r'data-um8-fn="connectAccount" data-um8-arg="\1"',
        t,
    )
    t = re.sub(
        r'onclick="hideModal\(\'([a-zA-Z0-9_]+)\'\)"',
        r'data-um8-fn="hideModal" data-um8-arg="\1"',
        t,
    )
    t = re.sub(
        r'onclick="filterUploads\(\'([a-z]+)\',this\)"',
        r'data-um8-fn="filterUploads" data-um8-arg="\1"',
        t,
    )
    t = re.sub(
        r'onclick="switchTab\(\'([a-z]+)\'\)"',
        r'data-um8-fn="switchTab" data-um8-arg="\1"',
        t,
    )
    t = re.sub(
        r'onclick="refreshPlatformMetrics\(true\)"',
        r'data-um8-fn="refreshPlatformMetrics" data-um8-args="[true]"',
        t,
    )

    return t


def main() -> int:
    root = Path(__file__).resolve().parents[1] / "frontend"
    for path in sorted(root.rglob("*.html")):
        raw = path.read_text(encoding="utf-8")
        new = migrate_html(raw)
        if new != raw:
            path.write_text(new, encoding="utf-8")
            print("updated", path.relative_to(root))
    return 0


if __name__ == "__main__":
    sys.exit(main())
