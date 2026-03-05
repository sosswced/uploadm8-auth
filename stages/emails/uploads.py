"""
UploadM8 — Phase 4a: Upload Notification Emails
================================================
  send_upload_completed_email  → upload worker: job reaches succeeded/completed status
  send_upload_failed_email     → upload worker: job reaches failed status

Both respect user_preferences.email_notifications.
Call the guard helper check_email_prefs() before firing.
"""

import logging
from .base import (
    send_email, mailgun_ready,
    email_shell, intro_row, body_row, cta_button, tinted_box,
    check_list, stat_grid, secondary_links, alert_banner, spacer,
    GRAD_GREEN, GRAD_RED, GRAD_ORANGE,
    URL_DASHBOARD, SUPPORT_EMAIL,
)

logger = logging.getLogger("uploadm8-worker")

# Platform brand colours (used in mini-badges)
PLATFORM_COLORS = {
    "tiktok":    "#000000",
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
    """Renders small coloured platform pills."""
    badges = "".join(
        f'<span style="display:inline-block;background:{PLATFORM_COLORS.get(p,"#374151")};'
        f'color:#ffffff;font-size:12px;font-weight:600;padding:4px 12px;border-radius:99px;'
        f'margin:3px 4px;">{PLATFORM_NAMES.get(p, p.title())}</span>'
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
    token_stats.append(("Platforms",    str(platform_count)))

    html = email_shell(
        gradient=GRAD_GREEN,
        tagline="Upload once. Publish everywhere.",
        body_rows=(
            intro_row(
                "Your upload is live! &#127775;",
                f"<strong style='color:#ffffff;'>{filename}</strong> has been successfully "
                f"published to <strong style='color:#22c55e;'>{platform_count} {platform_word}</strong>. "
                "Your content is now reaching your audience.",
            )
            + _platform_badges(platforms)
            + stat_grid(*token_stats)
            + cta_button("View Upload Details", URL_DASHBOARD, pt="20px", pb="20px")
            + secondary_links(("Dashboard", URL_DASHBOARD),)
        ),
        footer_note="You received this because upload email notifications are enabled on your account.",
    )

    await send_email(email, f"&#127775; Your upload is live on {platform_count} {platform_word}!", html)


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
        body_rows=(
            intro_row(
                "Upload failed &#10060;",
                f"Unfortunately, <strong style='color:#ffffff;'>{filename}</strong> "
                f"could not be published. The error occurred during the "
                f"<strong style='color:#f87171;'>{stage_label}</strong> stage.",
            )
            + tinted_box(
                f'<p style="margin:0 0 6px;color:#6b7280;font-size:11px;text-transform:uppercase;'
                f'letter-spacing:1px;">Error Details</p>'
                f'<p style="margin:0;color:#f87171;font-size:14px;line-height:1.6;'
                f'font-family:monospace;">{reason_display}</p>',
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
                f'<p style="margin:0;color:#9ca3af;font-size:13px;line-height:1.6;">'
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

    await send_email(email, f"&#10060; Upload failed — {filename}", html)
