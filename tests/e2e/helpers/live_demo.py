"""
Live demo journey: login, upload, browse while processing, verify dashboard/queue.

Single module for:
  - scripts/run_live_demo_journey.py  (CLI)
  - tests/e2e/test_live_demo_journey.py (pytest)

Browser lifecycle lives in browser_session.SingleBrowserSession — this file is journey logic only.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from playwright.sync_api import Page

from tests.e2e.helpers.auth import _TRANSIENT_API_EXC, api_client, api_get_with_retry, close_api_client
from tests.e2e.helpers.browser_session import (
    bootstrap_human_session,
    ensure_authed,
    human_login_via_form,
    human_scroll,
    navigate_to_page_human,
    wait_for_authenticated_shell,
)
from tests.e2e.helpers.human_pace import CHROME_UA, click_delay_ms, pause_between_requests
from tests.e2e.helpers.config import (
    e2e_master_email,
    e2e_persona_id,
    e2e_target_user_id,
    e2e_target_user_name,
    e2e_tiktok_profile,
    e2e_upload_platforms,
    e2e_use_persona,
)
from tests.e2e.helpers.pikzels_gate import consume_pikzels_slot, should_use_pikzels
from tests.e2e.helpers.pages import ADMIN_SETTINGS_PAGES, AUTHENTICATED_PAGES
from tests.e2e.helpers.background_checks import BackgroundCheckRunner
from tests.e2e.helpers.target_user import TargetUserNotFound, resolve_target_user
from tests.e2e.helpers.target_user_ui import exercise_target_user_admin_ui
from tests.e2e.helpers.ui_safe_clicks import click_page_surfaces
from tests.e2e.helpers.upload_files import set_upload_file_pair
from tests.e2e.helpers.upload_pace import (
    effective_api_per_page,
    effective_max_clicks,
    pause_between_upload_pages,
    skip_page_while_processing,
    upload_page_delay_sec,
    upload_poll_every_n_pages,
    upload_poll_interval_sec,
    upload_quiet_sec,
    worker_safe_mode,
)

TERMINAL_UPLOAD_STATUSES = frozenset(
    {
        "completed",
        "succeeded",
        "failed",
        "partial",
        "cancelled",
        "error",
        "staged",
        "ready_to_publish",
    }
)
PROCESSING_STATUSES = frozenset(
    {"pending", "processing", "uploading", "queued", "transcoding", "publishing", "staging"}
)

DEFAULT_VIDEO = Path(r"C:\Users\Earl\Videos\20250301_0058_CAM_EVNT.MP4")
DEFAULT_MAP = Path(r"C:\Users\Earl\Videos\20250301_0058_CAM_EVNT.map")


@dataclass
class LiveDemoLog:
    steps: list[str] = field(default_factory=list)

    def note(self, msg: str) -> None:
        self.steps.append(msg)
        print(f"[live-demo] {msg}", flush=True)


def resolve_demo_paths(
    video: Path | None = None,
    telemetry: Path | None = None,
) -> tuple[Path, Path | None]:
    from tests.e2e.helpers.config import e2e_test_telemetry_map, e2e_test_video

    v = video or e2e_test_video() or (DEFAULT_VIDEO if DEFAULT_VIDEO.is_file() else None)
    if v is None or not v.is_file():
        raise FileNotFoundError(
            "Set E2E_TEST_VIDEO or pass --video (default: C:\\Users\\Earl\\Videos\\20250301_0058_CAM_EVNT.MP4)"
        )
    t = telemetry
    if t is None:
        t = e2e_test_telemetry_map()
    if t is None and DEFAULT_MAP.is_file():
        t = DEFAULT_MAP
    if t is not None and not t.is_file():
        t = None
    return v, t


def _tiktok_settings_valid(page: Page) -> bool:
    return page.evaluate(
        """() => {
            if (!window.TikTokExport || !TikTokExport.isVisible || !TikTokExport.isVisible()) return true;
            return TikTokExport.isValid && TikTokExport.isValid();
        }"""
    )


def _uncheck_other_platforms(page: Page, keep: set[str]) -> None:
    for cb in page.locator('input[name="platforms"]:checked').all():
        val = (cb.get_attribute("value") or "").lower()
        if val and val not in keep:
            cb.uncheck(force=True)
            page.wait_for_timeout(click_delay_ms() // 2)


def _select_tiktok_profile(page: Page, log: LiveDemoLog, profile_hint: str | None) -> None:
    page.wait_for_selector(
        '#accountPickerContainer .account-chip[data-platform="tiktok"]',
        state="attached",
        timeout=90_000,
    )
    chips = page.locator('.account-chip[data-platform="tiktok"]')
    count = chips.count()
    if count == 0:
        log.note("No TikTok profiles in account picker")
        return

    target_idx = 0
    if profile_hint:
        hint = profile_hint.lower().lstrip("@")
        for i in range(count):
            text = chips.nth(i).inner_text(timeout=5_000).lower()
            if hint in text:
                target_idx = i
                break

    for i in range(count):
        cb = chips.nth(i).locator(".account-picker-cb")
        want_checked = i == target_idx
        if cb.is_checked() != want_checked:
            chips.nth(i).click()
            page.wait_for_timeout(click_delay_ms())

    label = chips.nth(target_idx).inner_text(timeout=5_000).strip().split("\n")[0]
    log.note(f"TikTok profile: {label[:80]}")


def _prepare_tiktok_for_upload(page: Page, log: LiveDemoLog, profile_hint: str | None) -> None:
    """Select TikTok profile + fill privacy/consent so Upload & Publish enables."""
    page.wait_for_function(
        """() => {
            const card = document.getElementById('tiktokExportCard');
            return card && !card.classList.contains('hidden');
        }""",
        timeout=120_000,
    )
    _select_tiktok_profile(page, log, profile_hint)

    page.wait_for_function(
        """() => {
            const host = document.getElementById('tiktokExportAccounts');
            if (!host || host.querySelector('.fa-spinner')) return false;
            return !!host.querySelector('.tiktok-export-account .tt-privacy');
        }""",
        timeout=120_000,
    )

    filled = page.evaluate(
        """() => {
            let n = 0;
            document.querySelectorAll('.tiktok-export-account[data-account-id]').forEach((root) => {
                const accId = root.getAttribute('data-account-id');
                const priv = root.querySelector('select.tt-privacy');
                if (priv) {
                    const opt = Array.from(priv.options).find((o) => o.value && !o.disabled);
                    if (opt && priv.value !== opt.value) {
                        priv.value = opt.value;
                        priv.dispatchEvent(new Event('change', { bubbles: true }));
                    }
                }
                const consent = root.querySelector('input.tt-consent');
                if (consent && !consent.checked) {
                    consent.checked = true;
                    consent.dispatchEvent(new Event('change', { bubbles: true }));
                }
                n += 1;
            });
            if (window.TikTokExport && TikTokExport.isValid) TikTokExport.isValid();
            if (typeof validateForm === 'function') validateForm();
            return n;
        }"""
    )
    page.wait_for_timeout(click_delay_ms())
    if not filled:
        raise RuntimeError("TikTok export panels did not load — connect a TikTok account on Platforms")
    if not _tiktok_settings_valid(page):
        raise RuntimeError("TikTok post settings still invalid after auto-fill (privacy/consent)")
    log.note("TikTok post settings configured")


def select_upload_platforms(
    page: Page,
    log: LiveDemoLog,
    *,
    preferred: tuple[str, ...] | None = None,
    tiktok_profile: str | None = None,
) -> list[str]:
    """
    Enable connected platforms from E2E_UPLOAD_PLATFORMS (or all under /TUP).

    Leaves each platform's default privacy / caption prefs from Settings intact;
    only TikTok needs explicit privacy+consent fill for the publish button.
    """
    wait_for_authenticated_shell(page)
    page.wait_for_function(
        """() => document.querySelectorAll('input[name="platforms"]:not([disabled])').length > 0""",
        timeout=120_000,
    )
    page.wait_for_timeout(800)

    want = {p.lower() for p in (preferred or e2e_upload_platforms())}
    fallback = ["tiktok", "youtube", "instagram", "facebook"]
    order = [p for p in fallback if p in want]
    order.extend(p for p in want if p not in order)

    selected: list[str] = []
    profile_hint = tiktok_profile if tiktok_profile is not None else e2e_tiktok_profile()

    for platform in order:
        option = page.locator(f'label.platform-option[data-platform="{platform}"]:not(.disabled)')
        if option.count() == 0 or not option.first.is_visible():
            log.note(f"Platform not connected/visible — skip: {platform}")
            continue
        cb = page.locator(f'input[name="platforms"][value="{platform}"]:not([disabled])')
        if cb.count() and not cb.first.is_checked():
            option.first.click()
            page.wait_for_timeout(click_delay_ms())
        if platform == "tiktok":
            try:
                _prepare_tiktok_for_upload(page, log, profile_hint or None)
            except Exception as exc:
                if cb.count():
                    cb.first.uncheck(force=True)
                log.note(f"Skipped TikTok — {exc}")
                continue
        selected.append(platform)

    if not selected:
        for platform in fallback:
            if platform in want:
                continue
            option = page.locator(f'label.platform-option[data-platform="{platform}"]:not(.disabled)')
            if option.count() == 0 or not option.first.is_visible():
                continue
            option.first.click()
            page.wait_for_timeout(click_delay_ms())
            if platform == "tiktok":
                _prepare_tiktok_for_upload(page, log, profile_hint or None)
            selected.append(platform)
            break

    if not selected:
        option = page.locator("label.platform-option:not(.disabled)").first
        if option.count() == 0:
            raise RuntimeError("No connected platforms on upload.html — connect accounts on Platforms first")
        val = option.get_attribute("data-platform") or "unknown"
        option.click()
        selected.append(val)
        log.note(f"Selected platform (fallback): {val}")
    else:
        _uncheck_other_platforms(page, {p.lower() for p in selected})
        for platform in selected:
            log.note(f"Selected platform: {platform}")

    # Re-validate so platform default privacy / schedule prefs from Settings apply.
    page.evaluate(
        """() => {
            if (typeof validateForm === 'function') validateForm();
            return true;
        }"""
    )
    return selected


def apply_upload_thumbnail_controls(
    page: Page,
    log: LiveDemoLog,
    *,
    use_pikzels: bool,
    use_persona: bool | None = None,
    persona_id: str | None = None,
) -> dict[str, Any]:
    """
    Set Thumbnail Studio controls on upload.html.

    Persona is selected/called when available (default persona or first Pikzels-linked).
    Pikzels engine checkbox follows the once-per-setup gate (use_pikzels).
    """
    want_persona = e2e_use_persona() if use_persona is None else use_persona
    want_id = (persona_id if persona_id is not None else e2e_persona_id()).strip()

    page.wait_for_selector("#uploadUseStudioEngine", state="attached", timeout=60_000)
    # Wait for persona select to finish async init (options beyond placeholder).
    page.wait_for_timeout(1200)
    page.wait_for_function(
        """() => {
            const sel = document.getElementById('uploadPersonaSelect');
            return !!sel && sel.options && sel.options.length >= 1;
        }""",
        timeout=60_000,
    )

    result = page.evaluate(
        """({ usePikzels, wantPersona, wantId }) => {
            const engine = document.getElementById('uploadUseStudioEngine');
            const personaCb = document.getElementById('uploadUsePersona');
            const personaSel = document.getElementById('uploadPersonaSelect');
            const strength = document.getElementById('uploadPersonaStrength');
            const out = {
                studio_blocked: window._uploadStudioBlocked === true,
                pikzels: false,
                persona: false,
                persona_id: '',
                persona_name: '',
                linked_count: 0,
                note: '',
            };
            if (!engine || !personaCb || !personaSel) {
                out.note = 'thumbnail controls missing';
                return out;
            }
            if (out.studio_blocked) {
                engine.checked = false;
                personaCb.checked = false;
                out.note = 'studio blocked (auto thumbs / studio flow off in Settings)';
                if (typeof validateForm === 'function') validateForm();
                return out;
            }

            const options = Array.from(personaSel.options || []).filter((o) => o.value);
            out.linked_count = options.length;

            let pick = '';
            if (wantId && options.some((o) => o.value === wantId)) {
                pick = wantId;
            } else if (options.length) {
                pick = options[0].value;
            }

            engine.checked = !!usePikzels;
            engine.dispatchEvent(new Event('change', { bubbles: true }));

            const canPersona = wantPersona && !!pick;
            personaCb.checked = canPersona;
            personaCb.dispatchEvent(new Event('change', { bubbles: true }));
            if (canPersona) {
                personaSel.value = pick;
                personaSel.dispatchEvent(new Event('change', { bubbles: true }));
                // Only force engine when this run is allowed to spend Pikzels.
                if (usePikzels && !engine.checked) {
                    engine.checked = true;
                    engine.dispatchEvent(new Event('change', { bubbles: true }));
                }
                out.persona = true;
                out.persona_id = pick;
                const opt = options.find((o) => o.value === pick);
                out.persona_name = (opt && opt.textContent || '').trim().slice(0, 120);
                if (!usePikzels) {
                    out.note = 'persona selected; Pikzels engine skipped (once-per-setup gate)';
                }
            } else if (wantPersona && !pick) {
                out.note = 'persona requested but no Pikzels-linked persona available';
            }

            out.pikzels = !!engine.checked;
            if (typeof syncUploadPersonaFieldDisabled === 'function') syncUploadPersonaFieldDisabled();
            if (typeof validateForm === 'function') validateForm();
            return out;
        }""",
        {"usePikzels": use_pikzels, "wantPersona": want_persona, "wantId": want_id},
    )
    if result.get("pikzels"):
        log.note("Pikzels / AuroraRender: ON for this upload")
    else:
        log.note("Pikzels / AuroraRender: OFF (gate skip or studio blocked)")
    if result.get("persona"):
        log.note(
            f"Persona applied: {result.get('persona_name') or result.get('persona_id')}"
        )
    elif result.get("note"):
        log.note(str(result["note"]))
    else:
        log.note("Persona: not applied")
    return result if isinstance(result, dict) else {}


def _read_upload_session_ids(page: Page) -> list[str]:
    return page.evaluate(
        """() => {
            try {
                const s = JSON.parse(sessionStorage.getItem('uploadm8.uploadSession') || '{}');
                return (s.ids || []).map(String);
            } catch (_) { return []; }
        }"""
    )


def start_upload_on_page(
    page: Page,
    base_url: str,
    video: Path,
    telemetry: Path | None,
    log: LiveDemoLog,
    *,
    use_pikzels: bool | None = None,
) -> tuple[str, list[str]]:
    """upload.html → attach files → platforms + persona → Upload & Publish."""
    navigate_to_page_human(page, base_url, "upload.html")
    wait_for_authenticated_shell(page)

    set_upload_file_pair(page, video, telemetry)
    page.wait_for_function(
        """() => {
            const fl = document.getElementById('fileList');
            return fl && !fl.classList.contains('hidden') && fl.querySelector('.file-item');
        }""",
        timeout=180_000,
    )
    log.note(f"Attached {video.name}" + (f" + {telemetry.name}" if telemetry else ""))

    select_upload_platforms(page, log)

    if use_pikzels is None:
        use_pikzels = consume_pikzels_slot(note="live_demo start_upload")
    elif use_pikzels and should_use_pikzels():
        consume_pikzels_slot(note="live_demo start_upload forced")
    thumb = apply_upload_thumbnail_controls(page, log, use_pikzels=bool(use_pikzels))
    log.note(
        f"Thumbnail controls: pikzels={thumb.get('pikzels')} persona={thumb.get('persona')}"
    )

    page.wait_for_function(
        """() => {
            const btn = document.getElementById('uploadBtn');
            return btn && !btn.disabled;
        }""",
        timeout=180_000,
    )
    page.locator("#uploadBtn").click()
    log.note("Clicked Upload & Publish")

    page.locator("#uploadProgress:not(.hidden)").wait_for(state="visible", timeout=120_000)
    page.wait_for_function(
        """() => {
            try {
                const s = JSON.parse(sessionStorage.getItem('uploadm8.uploadSession') || '{}');
                return Array.isArray(s.ids) && s.ids.length > 0;
            } catch (_) { return false; }
        }""",
        timeout=300_000,
    )
    upload_ids = _read_upload_session_ids(page)
    if upload_ids:
        log.note(f"Upload IDs: {', '.join(upload_ids)}")
    log.note("Upload progress panel visible")
    return video.stem, upload_ids


def find_upload_via_api(
    client: httpx.Client,
    *,
    filename_hint: str,
    upload_ids: list[str] | None = None,
    timeout_s: float = 7200.0,
    poll_s: float = 15.0,
    log: LiveDemoLog | None = None,
    raise_on_timeout: bool = True,
) -> tuple[dict[str, Any] | None, httpx.Client]:
    """Poll GET /api/uploads until the demo upload reaches a terminal pipeline status."""
    deadline = time.time() + timeout_s
    hint = filename_hint.lower()
    id_set = set(upload_ids or [])
    last_status = ""
    active = client
    while time.time() < deadline:
        if id_set:
            for uid in id_set:
                r, active = api_get_with_retry(active, f"/api/uploads/{uid}", headers={"User-Agent": CHROME_UA})
                if r.status_code != 200:
                    continue
                row = r.json()
                status = str(row.get("status") or "").lower()
                if status != last_status:
                    last_status = status
                    if log:
                        log.note(f"Upload {uid[:8]}… status={status}")
                if status in TERMINAL_UPLOAD_STATUSES:
                    return row, active
                if status and status not in PROCESSING_STATUSES and status not in TERMINAL_UPLOAD_STATUSES:
                    if log:
                        log.note(f"Treat status '{status}' as terminal")
                    return row, active

        r, active = api_get_with_retry(active, "/api/uploads", params={"limit": 40}, headers={"User-Agent": CHROME_UA})
        r.raise_for_status()
        rows = r.json()
        if isinstance(rows, dict):
            rows = rows.get("uploads") or rows.get("items") or []
        match = None
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, dict):
                continue
            rid = str(row.get("id") or "")
            if id_set and rid in id_set:
                match = row
                break
            name = " ".join(
                str(row.get(k) or "")
                for k in ("filename", "original_filename", "title", "video_filename", "name")
            ).lower()
            if hint in name or hint.replace("_", "") in name.replace("_", ""):
                match = row
                break
        if match:
            status = str(match.get("status") or "").lower()
            if status != last_status:
                last_status = status
                if log:
                    log.note(f"Upload {str(match.get('id', '?'))[:8]}… status={status}")
            if status in TERMINAL_UPLOAD_STATUSES:
                return match, active
            if status and status not in PROCESSING_STATUSES and status not in TERMINAL_UPLOAD_STATUSES:
                if log:
                    log.note(f"Treat status '{status}' as terminal")
                return match, active
        pause_between_requests()
        time.sleep(max(poll_s, 1.0))
    if raise_on_timeout:
        raise TimeoutError(
            f"Upload matching {filename_hint!r} did not finish within {timeout_s:.0f}s (last={last_status})"
        )
    return None, active


def verify_upload_on_dashboard_and_queue(
    page: Page,
    base_url: str,
    *,
    filename_hint: str,
    upload_ids: list[str] | None = None,
    log: LiveDemoLog | None = None,
) -> None:
    """Open dashboard + queue and confirm the demo upload is visible."""
    hint = filename_hint.lower()
    id_set = set(upload_ids or [])

    navigate_to_page_human(page, base_url, "dashboard.html")
    wait_for_authenticated_shell(page)
    human_scroll(page)
    page.locator("#recentUploads .upload-row").first.wait_for(state="attached", timeout=60_000)
    dash_text = page.locator("#recentUploads").inner_text(timeout=10_000).lower()
    dash_ok = hint in dash_text or hint.replace("_", "") in dash_text.replace("_", "")
    if not dash_ok and id_set:
        dash_ok = any(uid in dash_text for uid in id_set)
    if not dash_ok:
        raise AssertionError(f"Demo upload not visible on dashboard (expected {filename_hint})")
    if log:
        log.note("Dashboard shows recent upload")

    navigate_to_page_human(page, base_url, "queue.html")
    wait_for_authenticated_shell(page)
    human_scroll(page)
    page.locator("#queueList .upload-row").first.wait_for(state="visible", timeout=60_000)
    queue_text = page.locator("#queueList").inner_text(timeout=10_000).lower()
    queue_ok = hint in queue_text or hint.replace("_", "") in queue_text.replace("_", "")
    if not queue_ok and id_set:
        queue_ok = any(uid in queue_text for uid in id_set)
    if not queue_ok:
        raise AssertionError(f"Demo upload not visible on queue (expected {filename_hint})")
    if log:
        log.note("Queue lists the upload")

    page.locator("#queueList .upload-row").first.hover()
    page.wait_for_timeout(click_delay_ms())


def _upload_tour_order(browse_pages: tuple[str, ...] | None = None) -> tuple[str, ...]:
    """Admin/settings first (while upload processes), then the rest of the app."""
    if browse_pages:
        return browse_pages
    seen: set[str] = set()
    ordered: list[str] = []
    for rel in ADMIN_SETTINGS_PAGES:
        if rel not in seen and rel != "upload.html":
            seen.add(rel)
            ordered.append(rel)
    for rel in AUTHENTICATED_PAGES:
        if rel not in seen and rel != "upload.html":
            seen.add(rel)
            ordered.append(rel)
    return tuple(ordered)


def _poll_upload_processing(
    client: httpx.Client,
    *,
    stem: str,
    upload_ids: list[str],
    log: LiveDemoLog,
) -> tuple[dict[str, Any] | None, httpx.Client]:
    """Quick poll — returns upload row when terminal, else None while still processing."""
    return find_upload_via_api(
        client,
        filename_hint=stem,
        upload_ids=upload_ids,
        timeout_s=6.0,
        poll_s=1.5,
        log=log,
        raise_on_timeout=False,
    )


def _resolve_target_user_for_checks(client: httpx.Client, log: LiveDemoLog) -> dict[str, Any] | None:
    try:
        user = resolve_target_user(client)
        log.note(f"Target user: {user.get('name')} ({user.get('id')})")
        return user
    except TargetUserNotFound as e:
        log.note(f"Target user resolve: {e}")
        return None


def _wait_upload_quiet_period(
    client: httpx.Client,
    *,
    stem: str,
    upload_ids: list[str],
    log: LiveDemoLog,
) -> tuple[dict[str, Any] | None, httpx.Client]:
    """Let worker.py claim FFmpeg slots before the browser/API marathon."""
    quiet = upload_quiet_sec()
    if quiet <= 0:
        return None, client
    log.note(f"Worker quiet period {quiet:.0f}s — status poll only (no page tour)")
    deadline = time.time() + quiet
    last_poll = 0.0
    terminal: dict[str, Any] | None = None
    while time.time() < deadline and terminal is None:
        now = time.time()
        remaining = deadline - now
        if remaining <= 0:
            break
        if now - last_poll >= min(20.0, upload_poll_interval_sec()):
            row, client = find_upload_via_api(
                client,
                filename_hint=stem,
                upload_ids=upload_ids,
                timeout_s=min(8.0, max(1.0, remaining)),
                poll_s=2.0,
                log=log,
                raise_on_timeout=False,
            )
            last_poll = now
            if row:
                terminal = row
                log.note(f"Upload terminal during quiet period: {row.get('status')}")
                break
        remaining = deadline - time.time()
        if remaining <= 0:
            break
        time.sleep(min(5.0, remaining))
    return terminal, client


def run_admin_ui_while_upload_pending(page: Page, base_url: str, log: LiveDemoLog) -> dict[str, Any]:
    """Account-mgmt + wallet UI on target user while worker still processes upload."""
    log.note("Admin UI on target user (upload still pending)…")
    try:
        report = exercise_target_user_admin_ui(page, base_url)
        report["ok"] = True
        am = report.get("account_management", {}).get("steps", [])
        wal = report.get("wallet", {}).get("steps", [])
        log.note(f"Account-mgmt steps: {am}")
        log.note(f"Wallet steps: {wal}")
        return report
    except Exception as e:
        log.note(f"Admin UI failed: {e}")
        return {"ok": False, "error": str(e)}


def run_live_demo_journey(
    page: Page,
    base_url: str,
    *,
    video: Path,
    telemetry: Path | None = None,
    pipeline_timeout_s: float = 7200.0,
    browse_pages: tuple[str, ...] | None = None,
    max_clicks_per_page: int = 8,
    api_per_page: int = 8,
    include_slow_api: bool = False,
    skip_api_smoke: bool = False,
    log: LiveDemoLog | None = None,
) -> dict[str, Any]:
    """
    One session: master-admin login → upload → router lint + API GET sweep + Johnny UI
    + full app tour **while upload is still processing** → verify dashboard/queue when done.

    Caller must provide ``page`` from SingleBrowserSession (one window for the whole run).
    """
    log = log or LiveDemoLog()
    admin_email = e2e_master_email()

    from tests.e2e.helpers.api_ready import wait_for_api_ready

    ready = wait_for_api_ready(base_url, timeout_s=120.0, require_db=True)
    if not ready.get("ok"):
        raise RuntimeError(
            f"API/DB not ready at {base_url} before live journey "
            f"(last={ready}). Start uvicorn on :8000 only — do not use a second static port."
        )
    log.note(f"API ready (db ok) after {ready.get('attempt')} probe(s)")

    bootstrap_human_session(page, base_url)
    log.note(f"Logged in as {admin_email or 'master admin'} @ {base_url}")

    stem, upload_ids = start_upload_on_page(page, base_url, video, telemetry, log)
    if worker_safe_mode():
        log.note(
            "Upload started — worker-safe mode: quiet period, then slow tour "
            f"({upload_page_delay_sec():.0f}s/page, light API)"
        )
    else:
        log.note("Upload started — lint/API/UI tour while pipeline processes…")

    client: httpx.Client | None = api_client()
    safe_api = effective_api_per_page(api_per_page)
    bg = BackgroundCheckRunner(
        log,
        api_per_page=safe_api,
        include_slow_api=include_slow_api,
        skip_api_smoke=skip_api_smoke,
    )
    bg.prepare(client)
    bg.run_startup_static()

    result: dict[str, Any] = {
        "admin_email": admin_email,
        "target_user_id": e2e_target_user_id(),
        "target_user_name": e2e_target_user_name(),
        "target_user_resolved": None,
        "upload_ids": upload_ids,
        "admin_ui": None,
        "background_checks": None,
        "pages_visited": [],
        "worker_safe": worker_safe_mode(),
    }

    terminal_row, client = _wait_upload_quiet_period(
        client, stem=stem, upload_ids=upload_ids, log=log
    )

    if terminal_row is None:
        target_user = _resolve_target_user_for_checks(client, log)
        result["target_user_resolved"] = target_user
        result["admin_ui"] = run_admin_ui_while_upload_pending(page, base_url, log)

    browse_deadline = time.time() + max(600.0, pipeline_timeout_s - 180.0)
    tour = _upload_tour_order(browse_pages)
    idx = 0
    pages_since_poll = 0
    last_status_poll = time.time()

    try:
        while time.time() < browse_deadline and terminal_row is None:
            rel = tour[idx % len(tour)]
            idx += 1
            if skip_page_while_processing(rel):
                log.note(f"[pending] skip {rel} — heavy page deferred while worker processes")
                pause_between_upload_pages()
                continue

            ensure_authed(page, base_url)
            page_report: dict[str, Any] = {
                "page": rel,
                "clicks": 0,
                "upload_status": None,
                "api_5xx": [],
                "background": None,
                "skipped": False,
            }

            poll_due = (
                pages_since_poll >= upload_poll_every_n_pages()
                or time.time() - last_status_poll >= upload_poll_interval_sec()
            )
            include_target = not worker_safe_mode() or idx % 5 == 1
            include_lint = not worker_safe_mode() or idx % 2 == 1

            try:
                page_report["background"], client = bg.run_page_batch(
                    client,
                    rel,
                    include_target_user=include_target,
                    include_router_lint=include_lint,
                )
                api_fails: list[str] = []

                def _on_resp(resp) -> None:
                    if resp.status >= 500 and "/api/" in resp.url:
                        api_fails.append(f"{resp.status} {resp.url}")

                page.on("response", _on_resp)
                try:
                    navigate_to_page_human(page, base_url, rel)
                    human_scroll(page, passes=1 if worker_safe_mode() else 2)
                    leaf = rel.split("/")[-1]
                    click_budget = effective_max_clicks(
                        max_clicks_per_page,
                        is_admin_page=leaf in ADMIN_SETTINGS_PAGES,
                    )
                    clicked = click_page_surfaces(page, rel, max_clicks=click_budget)
                    page_report["clicks"] = len(clicked)
                    bg_note = page_report["background"]
                    api_n = len((bg_note or {}).get("api") or [])
                    router_n = 1 if (bg_note or {}).get("router") else 0
                    log.note(f"[pending] {rel} — {len(clicked)} clicks, {api_n} API, {router_n} lint")
                except Exception as exc:
                    page_report["error"] = str(exc)[:300]
                    log.note(f"Browse {rel}: {exc}")
                    if "login.html" in page.url:
                        human_login_via_form(page, base_url)
                finally:
                    try:
                        page.remove_listener("response", _on_resp)
                    except Exception:
                        pass

                if api_fails:
                    page_report["api_5xx"] = api_fails[:8]
                    log.note(f"  page API 5xx: {api_fails[0][:100]}")

                pages_since_poll += 1
                if poll_due:
                    row, client = _poll_upload_processing(client, stem=stem, upload_ids=upload_ids, log=log)
                    pages_since_poll = 0
                    last_status_poll = time.time()
                    if row:
                        page_report["upload_status"] = row.get("status")
                        terminal_row = row
                        log.note(f"Upload terminal during tour: {row.get('status')}")
            except _TRANSIENT_API_EXC as exc:
                page_report["error"] = str(exc)[:300]
                log.note(f"Transient API error on {rel} (continuing): {exc}")
            result["pages_visited"].append(page_report)

            if terminal_row:
                break
            pause_between_upload_pages()

        if terminal_row is None:
            terminal_row, client = find_upload_via_api(
                client,
                filename_hint=stem,
                upload_ids=upload_ids,
                timeout_s=max(120.0, pipeline_timeout_s),
                log=log,
            )
        assert terminal_row is not None

        if bg.api_remaining():
            drained, client = bg.drain_remaining_api(client)
            log.note(f"Drained {drained} remaining API checks after tour")
    finally:
        close_api_client(client)

    verify_upload_on_dashboard_and_queue(
        page, base_url, filename_hint=stem, upload_ids=upload_ids, log=log
    )

    check_summary = bg.summary()
    result["background_checks"] = check_summary
    admin_ui = result.get("admin_ui")
    ui_ok = True if admin_ui is None else bool(admin_ui.get("ok"))
    if worker_safe_mode() and admin_ui and not admin_ui.get("ok"):
        ui_ok = True
    page_5xx = [f for p in result["pages_visited"] for f in (p.get("api_5xx") or [])]
    upload_status = str(terminal_row.get("status") or "").lower()
    upload_ok = upload_status in TERMINAL_UPLOAD_STATUSES and upload_status not in {"failed", "error", "cancelled"}
    result["upload"] = terminal_row
    result["steps"] = log.steps
    result["ok"] = bool(check_summary.get("ok") and ui_ok and upload_ok and not page_5xx)
    if not result["ok"]:
        if not check_summary.get("ok"):
            log.note(f"Background checks: {check_summary.get('failures')} failure(s)")
        if page_5xx:
            log.note(f"Page API 5xx during UI: {len(page_5xx)}")
        if not upload_ok:
            log.note(f"Upload ended with status={upload_status}")
    return result
