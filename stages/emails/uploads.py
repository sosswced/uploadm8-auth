"""
UploadM8 — Phase 4a: Upload Notification Emails  (v2 — Enhanced Design)
=========================================================================
  send_upload_completed_email  → upload worker: job reaches succeeded/completed status
  send_upload_failed_email     → upload worker: job reaches failed status

Both respect user_preferences.email_notifications.

v2 upgrades:
  - Both emails have preheader_text
  - Completed: section_tag "Upload Live", metric_hero for platform count, improved badges
  - Failed: section_tag "Upload Failed", enhanced error display
"""

import html
import logging
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

from stages.context import JobContext
from stages.notify_stage import _fallback_post_url, _normalize_post_url
from services.upload_notification_preview import thumbnail_quality_summary_text

from .base import (
    send_email, mailgun_ready,
    email_shell, intro_row, body_row, cta_button, tinted_box,
    check_list, stat_grid, secondary_links,
    section_tag, metric_hero, BrandContext,
    GRAD_GREEN, GRAD_RED, GRAD_ORANGE, GRAD_BLUE,
    URL_DASHBOARD, URL_SETTINGS, SUPPORT_EMAIL, FRONTEND_URL,
)

logger = logging.getLogger("uploadm8-worker")

# Platform brand colours (used in mini-badges)
PLATFORM_COLORS = {
    "tiktok":    "#1a1a1a",
    "youtube":   "#ff0000",
    "instagram": "#c13584",
    "facebook":  "#1877f2",
}

PLATFORM_NAMES = {
    "tiktok":    "TikTok",
    "youtube":   "YouTube Shorts",
    "instagram": "Instagram Reels",
    "facebook":  "Facebook Reels",
}


def _platform_badges(platforms: list[str]) -> str:
    """Renders small coloured platform pills. v2: slightly larger, bolder."""
    badges = "".join(
        f'<span style="display:inline-block;background:{PLATFORM_COLORS.get(p,"#374151")};'
        f'color:#ffffff;font-size:12px;font-weight:700;padding:5px 14px;border-radius:99px;'
        f'margin:4px 5px;border:1px solid rgba(255,255,255,0.12);">'
        f'{PLATFORM_NAMES.get(p, p.title())}</span>'
        for p in platforms
    )
    return (
        f'<tr><td style="padding:0 40px 28px;text-align:center;">{badges}</td></tr>'
    )


def _video_duration_seconds(ctx: JobContext) -> int:
    vi = getattr(ctx, "video_info", None) or {}
    try:
        return int(float(vi.get("duration") or 0))
    except (TypeError, ValueError):
        return 0


def _platform_post_links_html(ctx: JobContext) -> str:
    rows: List[str] = []
    for r in getattr(ctx, "platform_results", []) or []:
        if not getattr(r, "success", False):
            continue
        plat = (getattr(r, "platform", None) or "").lower()
        name = PLATFORM_NAMES.get(plat, plat.title() if plat else "Platform")
        url = _normalize_post_url(r) or _fallback_post_url(r)
        if url:
            rows.append(
                f'<p style="margin:10px 0;"><strong>{html.escape(name)}</strong> — '
                f'<a href="{html.escape(url, quote=True)}" style="color:#38bdf8;">View post</a></p>'
            )
        else:
            rows.append(
                f'<p style="margin:10px 0;"><strong>{html.escape(name)}</strong> — '
                f'<span style="color:#9ca3af;">Live URL not synced yet — open the app for the link.</span></p>'
            )
    if not rows:
        return ""
    return "".join(rows)


def _prefs_summary_html(us: Optional[Dict[str, Any]]) -> str:
    if not us:
        return "<p style=\"color:#9ca3af;\">—</p>"

    def _onoff(camel: str, snake: str, *, default_true: bool = True) -> str:
        v = us.get(camel)
        if v is None:
            v = us.get(snake)
        if v is None:
            return "on" if default_true else "off"
        return "on" if v else "off"

    priv = str(us.get("defaultPrivacy") or us.get("default_privacy") or "public").strip()
    lines = [
        f"Auto captions: {_onoff('autoCaptions', 'auto_captions')}",
        f"Auto thumbnails: {_onoff('autoThumbnails', 'auto_thumbnails', default_true=False)}",
        f"Styled thumbnails: {_onoff('styledThumbnails', 'styled_thumbnails')}",
        f"Thumbnail Studio: {_onoff('thumbnailStudioEnabled', 'thumbnail_studio_enabled')}",
        f"Aurora / Pikzels engine: {_onoff('thumbnailStudioEngineEnabled', 'thumbnail_studio_engine_enabled')}",
        f"Persona on uploads: {_onoff('thumbnailPersonaEnabled', 'thumbnail_persona_enabled', default_true=False)}",
        f"AI hashtags: {_onoff('aiHashtagsEnabled', 'ai_hashtags_enabled', default_true=False)}",
        f"Default privacy: {priv or 'public'}",
    ]
    return (
        '<p style="margin:4px 0;font-size:14px;line-height:1.65;color:#d1d5db;">'
        + "<br>".join(html.escape(x, quote=False) for x in lines)
        + "</p>"
    )


def _partial_failures_block(ctx: JobContext) -> str:
    if not ctx.is_partial_success():
        return ""
    fails = ctx.get_failed_platforms()
    if not fails:
        return ""
    lbl = ", ".join(html.escape(str(p).title(), quote=False) for p in fails)
    return (
        f'<p style="margin:0;color:#fcd34d;font-size:14px;line-height:1.55;">'
        f"Partial success — these platforms did not publish: <strong>{lbl}</strong>.</p>"
    )


def build_upload_completed_email_extensions(
    ctx: JobContext,
    *,
    put_balance: Optional[int] = None,
    aic_balance: Optional[int] = None,
) -> Dict[str, Any]:
    """Keyword args for ``send_upload_completed_email`` (plus ``duration_seconds``)."""
    title = ctx.get_effective_title()
    cap = (ctx.get_effective_caption() or "").strip()[:2000]
    tags = ctx.get_effective_hashtags()
    tags_line = " ".join(tags[:60]).strip() if tags else ""
    posted = ""
    if ctx.finished_at:
        posted = ctx.finished_at.strftime("%Y-%m-%d %H:%M UTC")
    tq = thumbnail_quality_summary_text(ctx)
    scene_story_value = ""
    try:
        if isinstance(ctx.output_artifacts, dict):
            scene_story_value = str(ctx.output_artifacts.get("scene_story") or "").strip()
    except Exception:
        scene_story_value = ""
    return {
        "duration_seconds": _video_duration_seconds(ctx),
        "detail_title": title,
        "detail_caption": cap,
        "hashtags_line": tags_line,
        "posted_at_display": posted,
        "put_balance": put_balance,
        "aic_balance": aic_balance,
        "prefs_summary_html": _prefs_summary_html(ctx.user_settings),
        "platform_links_html": _platform_post_links_html(ctx),
        "partial_warning_html": _partial_failures_block(ctx),
        "thumbnail_score_summary": tq,
        "scene_story": scene_story_value[:1600],
    }


# ─────────────────────────────────────────────────────────────────────────────
# 1. Upload completed
# ─────────────────────────────────────────────────────────────────────────────
async def send_upload_completed_email(
    email: str,
    name: str,
    filename: str,
    platforms: list[str],
    put_spent: int = 0,
    aic_spent: int = 0,
    upload_id: str = "",
    duration_seconds: int = 0,
    *,
    detail_title: str = "",
    detail_caption: str = "",
    hashtags_line: str = "",
    posted_at_display: str = "",
    put_balance: Optional[int] = None,
    aic_balance: Optional[int] = None,
    prefs_summary_html: str = "",
    platform_links_html: str = "",
    partial_warning_html: str = "",
    thumbnail_score_summary: str = "",
    preview_image_url: Optional[str] = None,
    scene_story: str = "",
    brand: Optional[BrandContext] = None,
) -> None:
    """
    Sent when an upload job reaches ``succeeded`` or ``partial`` status.
    Only fires if ``user_preferences.email_notifications`` is not explicitly false (caller).

    The worker loads wallet balances **after** capture/refund, then passes keyword extras from
    ``build_upload_completed_email_extensions`` (title, caption, hashtags, post links, prefs snapshot).

    When ``preview_image_url`` is an ``https://`` presigned R2 URL, the email shows the analyzed thumbnail image.
    """
    if not mailgun_ready():
        return

    platform_count = len(platforms)
    platform_word  = "platform" if platform_count == 1 else "platforms"
    dur_label      = f"{duration_seconds}s" if duration_seconds else "—"

    token_stats: List[Tuple[str, str]] = []
    if put_spent:
        token_stats.append(("PUT spent", str(put_spent)))
    if aic_spent:
        token_stats.append(("AIC spent", str(aic_spent)))
    if put_balance is not None:
        token_stats.append(("PUT remaining", str(put_balance)))
    if aic_balance is not None:
        token_stats.append(("AIC remaining", str(aic_balance)))
    token_stats.append(("Platforms live", str(platform_count)))
    token_stats.append(("Duration", dur_label))

    queue_href = (
        f"{FRONTEND_URL.rstrip('/')}/queue.html?focus={quote(str(upload_id), safe='')}&open=1"
        if upload_id
        else URL_DASHBOARD
    )

    preview_box = ""
    tq_plain = (thumbnail_score_summary or "").strip()
    prev_url = (preview_image_url or "").strip()
    if tq_plain or prev_url:
        inner_bits: List[str] = []
        if tq_plain:
            inner_bits.append(
                f'<p style="margin:0 0 12px;color:#e5e7eb;font-size:14px;line-height:1.65;">'
                f'{html.escape(tq_plain, quote=False)}</p>'
            )
        if prev_url.startswith("https://"):
            u_esc = html.escape(prev_url, quote=True)
            inner_bits.append(
                f'<p style="margin:0;text-align:center;">'
                f'<img src="{u_esc}" alt="Thumbnail preview" width="560" '
                f'style="max-width:100%;height:auto;border-radius:12px;border:1px solid rgba(255,255,255,0.08);">'
                f"</p>"
            )
        if inner_bits:
            preview_box = tinted_box(
                f'<p style="margin:0 0 10px;color:#6b7280;font-size:10px;text-transform:uppercase;'
                f'letter-spacing:1.2px;font-weight:600;">Thumbnail preview</p>'
                + "".join(inner_bits),
                hex_color="#f59e0b",
            )

    copy_box = ""
    if (detail_title or "").strip() or (detail_caption or "").strip() or (hashtags_line or "").strip():
        t_esc = html.escape((detail_title or "").strip() or "—", quote=False)
        c_esc = html.escape((detail_caption or "").strip() or "—", quote=False)
        h_esc = html.escape((hashtags_line or "").strip() or "—", quote=False)
        copy_box = tinted_box(
            f'<p style="margin:0 0 6px;color:#6b7280;font-size:10px;text-transform:uppercase;'
            f'letter-spacing:1.2px;font-weight:600;">Title &amp; caption</p>'
            f'<p style="margin:0 0 10px;color:#ffffff;font-size:17px;font-weight:700;">{t_esc}</p>'
            f'<p style="margin:0 0 10px;color:#e5e7eb;font-size:14px;line-height:1.65;">{c_esc}</p>'
            f'<p style="margin:0;color:#9ca3af;font-size:12px;line-height:1.55;">'
            f'<span style="color:#6b7280;font-weight:600;">Hashtags</span><br>{h_esc}</p>',
            hex_color="#22c55e",
        )

    links_box = ""
    if (platform_links_html or "").strip():
        links_box = tinted_box(
            f'<p style="margin:0 0 10px;color:#6b7280;font-size:10px;text-transform:uppercase;'
            f'letter-spacing:1.2px;font-weight:600;">Direct links</p>'
            f"{platform_links_html}",
            hex_color="#0ea5e9",
        )

    scene_box = ""
    sc = (scene_story or "").strip()
    if sc:
        sc_esc = html.escape(sc, quote=False)
        scene_box = tinted_box(
            f'<p style="margin:0 0 6px;color:#6b7280;font-size:10px;text-transform:uppercase;'
            f'letter-spacing:1.2px;font-weight:600;">Scene Story</p>'
            f'<p style="margin:0;color:#e5e7eb;font-size:13px;line-height:1.65;white-space:pre-wrap;">{sc_esc}</p>',
            hex_color="#10b981",
        )

    prefs_box = ""
    if (prefs_summary_html or "").strip():
        prefs_box = tinted_box(
            f'<p style="margin:0 0 10px;color:#6b7280;font-size:10px;text-transform:uppercase;'
            f'letter-spacing:1.2px;font-weight:600;">What was enabled</p>'
            f"{prefs_summary_html}",
            hex_color="#a855f7",
        )

    partial_row = body_row(partial_warning_html, padding="0 40px 16px") if partial_warning_html else ""

    posted_row = ""
    if (posted_at_display or "").strip():
        pl = html.escape(posted_at_display.strip(), quote=False)
        posted_row = body_row(
            f'<p style="margin:0;color:#9ca3af;font-size:13px;">Completed (UTC): <strong style="color:#e5e7eb;">{pl}</strong></p>',
            padding="0 40px 20px",
        )

    body_html = email_shell(
        gradient=GRAD_GREEN,
        tagline="Upload once. Publish everywhere.",
        preheader_text=f"{filename} is live! Published to {platform_count} {platform_word} — open links below.",
        brand=brand,
        body_rows=(
            section_tag("Upload Live &#127775;", "#16a34a")
            + intro_row(
                "Your upload is live! &#127775;",
                f"<strong style='color:#ffffff;'>{html.escape(filename, quote=False)}</strong> has been published to "
                f"<strong style='color:#22c55e;'>{platform_count} {platform_word}</strong>. "
                "Use the queue link to review the job, or jump straight to each post.",
            )
            + metric_hero(
                str(platform_count),
                f"Platform{'s' if platform_count != 1 else ''} live",
                html.escape(filename, quote=False),
                "#16a34a",
            )
            + _platform_badges(platforms)
            + preview_box
            + copy_box
            + scene_box
            + links_box
            + prefs_box
            + partial_row
            + stat_grid(*token_stats)
            + posted_row
            + cta_button("Open in queue", queue_href, pt="20px", pb="12px")
            + cta_button("Dashboard", URL_DASHBOARD, pt="4px", pb="20px")
            + secondary_links(
                ("Queue", queue_href),
                ("Dashboard", URL_DASHBOARD),
                ("Settings", URL_SETTINGS),
            )
        ),
        footer_note="You received this because upload email notifications are enabled on your account.",
    )

    subject = f"🌟 Your upload is live on {platform_count} {platform_word}!"
    if brand:
        subject = f"{brand.company_name} — upload live on {platform_count} {platform_word}"
    await send_email(email, subject, body_html)


# ─────────────────────────────────────────────────────────────────────────────
# 1b. Processing complete — scheduled publish (deferred / staged uploads)
# ─────────────────────────────────────────────────────────────────────────────
async def send_upload_staged_processing_email(
    email: str,
    name: str,
    filename: str,
    platforms: list[str],
    upload_id: str = "",
    *,
    scheduled_at_label: str = "",
) -> None:
    """Sent when processing finishes but publish is deferred until ``scheduled_time``."""
    if not mailgun_ready():
        return

    plat_n = len(platforms or [])
    plat_word = "platform" if plat_n == 1 else "platforms"
    queue_href = (
        f"{FRONTEND_URL.rstrip('/')}/queue.html?focus={quote(str(upload_id), safe='')}&open=1"
        if upload_id
        else URL_DASHBOARD
    )
    sched = (scheduled_at_label or "").strip() or "your scheduled time (UTC)"

    body_html = email_shell(
        gradient=GRAD_BLUE,
        tagline="Upload once. Publish everywhere.",
        preheader_text=f"{filename} is processed and queued — publishes {sched}.",
        body_rows=(
            section_tag("Ready to publish", "#2563eb")
            + intro_row(
                "Video processed — waiting for your schedule &#9200;",
                f"Hi {html.escape(name, quote=False)}, "
                f"<strong style='color:#ffffff;'>{html.escape(filename, quote=False)}</strong> "
                f"finished AI processing and is queued for "
                f"<strong style='color:#60a5fa;'>{html.escape(sched, quote=False)}</strong>. "
                f"It will go live to <strong style='color:#93c5fd;'>{plat_n} {plat_word}</strong> automatically. "
                "Open the queue if you want to edit metadata or cancel before publish.",
            )
            + tinted_box(
                f'<p style="margin:0;color:#9ca3af;font-size:13px;line-height:1.65;">'
                f'You received this because upload email notifications are enabled.</p>'
                + (
                    f'<p style="margin:10px 0 0;color:#9ca3af;font-size:12px;">'
                    f'Upload ID: <code style="color:#f97316;">{html.escape(str(upload_id), quote=False)}</code></p>'
                    if upload_id
                    else ""
                ),
                hex_color="#374151",
                pb="28px",
            )
            + cta_button("Open queue", queue_href, pt="14px", pb="20px")
            + secondary_links(
                ("Queue", queue_href),
                ("Dashboard", URL_DASHBOARD),
                ("Settings", URL_SETTINGS),
            )
        ),
        footer_note="You received this because upload email notifications are enabled on your account.",
    )

    await send_email(
        email,
        f"⏰ Processed — scheduled publish: {filename}",
        body_html,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Upload failed
# ─────────────────────────────────────────────────────────────────────────────
async def send_upload_failed_email(
    email: str,
    name: str,
    filename: str,
    platforms: list[str],
    error_reason: str = "",
    upload_id: str = "",
    stage: str = "",
    scene_story: str = "",
    brand: Optional[BrandContext] = None,
) -> None:
    """
    Sent when an upload job reaches 'failed' status.
    Only fires if user_preferences.email_notifications is True (checked by caller).

    stage examples: "ingest", "transcode", "publish", "platform_api"
    """
    if not mailgun_ready():
        return

    reason_display = error_reason or "An unexpected error occurred during processing."
    stage_label    = stage.replace("_", " ").title() if stage else "Processing"

    scene_box = ""
    sc = (scene_story or "").strip()
    if sc:
        sc_esc = html.escape(sc, quote=False)
        scene_box = tinted_box(
            f'<p style="margin:0 0 6px;color:#6b7280;font-size:10px;text-transform:uppercase;'
            f'letter-spacing:1.2px;font-weight:600;">Scene Story (what we saw before it failed)</p>'
            f'<p style="margin:0;color:#e5e7eb;font-size:13px;line-height:1.65;white-space:pre-wrap;">{sc_esc}</p>',
            hex_color="#10b981",
        )

    body_html = email_shell(
        gradient=GRAD_RED,
        tagline="Upload once. Publish everywhere.",
        preheader_text=f"Upload failed: {filename} could not be published. Your tokens have been refunded.",
        brand=brand,
        body_rows=(
            section_tag("Upload Failed", "#ef4444")
            + intro_row(
                "Upload failed &#10060;",
                f"Unfortunately, <strong style='color:#ffffff;'>{filename}</strong> "
                f"could not be published. The error occurred during the "
                f"<strong style='color:#f87171;'>{stage_label}</strong> stage.",
            )
            + tinted_box(
                f'<p style="margin:0 0 6px;color:#6b7280;font-size:10px;text-transform:uppercase;'
                f'letter-spacing:1.2px;font-weight:600;">Error Details</p>'
                f'<p style="margin:0;color:#f87171;font-size:14px;line-height:1.65;'
                f'font-family:\'Courier New\',Courier,monospace;">{reason_display}</p>'
                + (
                    f'<p style="margin:10px 0 0;color:#6b7280;font-size:12px;">'
                    f'Upload ID: <code style="color:#f97316;">{upload_id}</code></p>'
                    if upload_id else ""
                ),
                hex_color="#ef4444",
            )
            + scene_box
            + check_list(
                "Tokens used for this upload have been refunded",
                "Your other uploads are not affected",
                "You can retry the upload from your dashboard",
                hex_color="#22c55e",
            )
            + cta_button("Retry Upload", URL_DASHBOARD, pt="4px", pb="20px")
            + tinted_box(
                f'<p style="margin:0;color:#9ca3af;font-size:13px;line-height:1.65;">'
                f'If this keeps happening, contact us at '
                f'<a href="mailto:{SUPPORT_EMAIL}" style="color:#f97316;text-decoration:none;">'
                f'{SUPPORT_EMAIL}</a> with your upload ID: '
                f'<code style="color:#f97316;font-size:12px;">{upload_id or "—"}</code></p>',
                hex_color="#374151",
                pb="36px",
            )
        ),
        footer_note="You received this because upload email notifications are enabled on your account.",
    )

    subject = f"❌ Upload failed — {filename}"
    if brand:
        subject = f"{brand.company_name} — upload failed — {filename}"
    await send_email(email, subject, body_html)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Scheduled publish alert
#
# Fires from the daily ``run_scheduled_publish_alerts`` admin job for two cases:
#   - status="upcoming"  → user has a publish landing in the next 24 hours;
#                          gives them a chance to edit/cancel before it fires.
#   - status="failed" / "stuck" → a scheduled publish that did not fire on time;
#                          user should know so they can retry from the queue.
# ─────────────────────────────────────────────────────────────────────────────
async def send_scheduled_publish_alert_email(
    email: str,
    name: str,
    filename: str,
    scheduled_at_label: str,
    status: str = "upcoming",
    reason: str = "",
    upload_id: str = "",
) -> None:
    if not mailgun_ready():
        return

    s = (status or "upcoming").strip().lower()
    if s in ("failed", "stuck", "error"):
        gradient   = GRAD_RED
        tag_color  = "#ef4444"
        section    = "Scheduled Publish Failed"
        headline   = f"Scheduled publish did not fire &#10060;"
        body_html  = (
            f"Your scheduled publish of <strong style='color:#ffffff;'>{filename}</strong> "
            f"set for <strong style='color:#f87171;'>{scheduled_at_label}</strong> did not "
            f"fire on time. The job is sitting in your queue — open it to retry or reschedule."
        )
        subject    = f"⚠️ Scheduled publish failed — {filename}"
    else:
        gradient   = GRAD_BLUE
        tag_color  = "#2563eb"
        section    = "Scheduled Publish Soon"
        headline   = "Heads up — a scheduled publish is coming"
        body_html  = (
            f"<strong style='color:#ffffff;'>{filename}</strong> is scheduled to publish at "
            f"<strong style='color:#60a5fa;'>{scheduled_at_label}</strong>. If you need to "
            f"edit the caption, swap platforms, or cancel, open the queue before then."
        )
        subject    = f"⏰ Scheduled publish coming up — {filename}"

    reason_box = (
        tinted_box(
            f'<p style="margin:0 0 6px;color:#6b7280;font-size:10px;text-transform:uppercase;'
            f'letter-spacing:1.2px;font-weight:600;">Detail</p>'
            f'<p style="margin:0;color:#f87171;font-size:14px;line-height:1.65;'
            f'font-family:\'Courier New\',Courier,monospace;">{reason}</p>',
            hex_color=tag_color,
        )
        if reason else ""
    )

    upload_id_html = (
        f'<p style="margin:10px 0 0;color:#9ca3af;font-size:12px;">'
        f'Upload ID: <code style="color:#f97316;">{upload_id}</code></p>'
        if upload_id else ""
    )

    shell_html = email_shell(
        gradient=gradient,
        tagline="Scheduled publish notice",
        preheader_text=f"{filename} — scheduled for {scheduled_at_label}",
        body_rows=(
            section_tag(section, tag_color)
            + intro_row(headline, body_html)
            + reason_box
            + cta_button("Open Queue", URL_DASHBOARD, pt="14px", pb="20px")
            + tinted_box(
                f'<p style="margin:0;color:#9ca3af;font-size:13px;line-height:1.65;">'
                f'Need help? Reply to this email or contact us at '
                f'<a href="mailto:{SUPPORT_EMAIL}" style="color:#f97316;text-decoration:none;">'
                f'{SUPPORT_EMAIL}</a>.</p>'
                + upload_id_html,
                hex_color="#374151",
                pb="36px",
            )
        ),
        footer_note="You received this because email notifications are enabled and you have a scheduled upload.",
    )

    await send_email(email, subject, shell_html)
