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

import logging
from .base import (
    send_email, mailgun_ready,
    email_shell, intro_row, body_row, cta_button, tinted_box,
    check_list, stat_grid, secondary_links, alert_banner, spacer,
    section_tag, metric_hero, divider_accent,
    GRAD_GREEN, GRAD_RED, GRAD_ORANGE,
    URL_DASHBOARD, SUPPORT_EMAIL,
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
) -> None:
    """
    Sent when an upload job reaches 'succeeded' or 'completed' status.
    Only fires if user_preferences.email_notifications is True (checked by caller).

    Usage in worker finish handler:
        prefs = await conn.fetchrow(
            "SELECT email_notifications FROM user_preferences WHERE user_id=$1", user_id
        )
        if prefs and prefs["email_notifications"]:
            await send_upload_completed_email(email, name, filename, platforms, ...)
    """
    if not mailgun_ready():
        return

    platform_count = len(platforms)
    platform_word  = "platform" if platform_count == 1 else "platforms"
    dur_label      = f"{duration_seconds}s" if duration_seconds else "—"

    token_stats = []
    if put_spent:
        token_stats.append(("PUT Spent", str(put_spent)))
    if aic_spent:
        token_stats.append(("AIC Spent", str(aic_spent)))
    token_stats.append(("Platforms", str(platform_count)))

    html = email_shell(
        gradient=GRAD_GREEN,
        tagline="Upload once. Publish everywhere.",
        preheader_text=f"{filename} is live! Published to {platform_count} {platform_word} and reaching your audience now.",
        body_rows=(
            section_tag("Upload Live &#127775;", "#16a34a")
            + intro_row(
                "Your upload is live! &#127775;",
                f"<strong style='color:#ffffff;'>{filename}</strong> has been successfully "
                f"published to <strong style='color:#22c55e;'>{platform_count} {platform_word}</strong>. "
                "Your content is now reaching your audience.",
            )
            + metric_hero(
                str(platform_count),
                f"Platform{'s' if platform_count != 1 else ''} Live",
                f"{filename}",
                "#16a34a",
            )
            + _platform_badges(platforms)
            + stat_grid(*token_stats)
            + cta_button("View Upload Details", URL_DASHBOARD, pt="20px", pb="20px")
            + secondary_links(("Dashboard", URL_DASHBOARD),)
        ),
        footer_note="You received this because upload email notifications are enabled on your account.",
    )

    await send_email(email, f"🌟 Your upload is live on {platform_count} {platform_word}!", html)


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

    html = email_shell(
        gradient=GRAD_RED,
        tagline="Upload once. Publish everywhere.",
        preheader_text=f"Upload failed: {filename} could not be published. Your tokens have been refunded.",
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

    await send_email(email, f"❌ Upload failed — {filename}", html)
