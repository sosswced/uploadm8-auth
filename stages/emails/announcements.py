"""
UploadM8 — Phase 5b: Announcement Email
=========================================
  send_announcement_email  → replaces the bare <h1>title</h1><p>body</p>
                             in _execute_announcement_deliveries() at line ~7566

Drop-in replacement. The caller signature stays identical — just swap the
send_email call in that function with send_announcement_email().
"""

import logging
from .base import (
    send_email, mailgun_ready,
    email_shell, intro_row, body_row, cta_button, tinted_box,
    secondary_links, spacer,
    GRAD_ORANGE,
    URL_DASHBOARD, SUPPORT_EMAIL,
)

logger = logging.getLogger("uploadm8-worker")


# ─────────────────────────────────────────────────────────────────────────────
# Announcement email
# ─────────────────────────────────────────────────────────────────────────────
async def send_announcement_email(
    to: str,
    title: str,
    body: str,
    cta_label: str = "",
    cta_url: str = "",
    tier_label: str = "",
    gradient: str = GRAD_ORANGE,
) -> None:
    """
    Branded version of the announcement email that currently sends raw HTML.

    Drop-in replacement for the send_email() call inside
    _execute_announcement_deliveries() in app.py:

    BEFORE (line ~7566):
        await send_email(dest, title, f"<h1>{title}</h1><p>{body}</p>")

    AFTER:
        from stages.emails.announcements import send_announcement_email
        await send_announcement_email(dest, title, body)

    Optional arguments:
      cta_label   — button label  (e.g. "Read More", "See What's New")
      cta_url     — button URL    (defaults to dashboard if cta_label given)
      tier_label  — shown as a small badge if targeting a specific tier
      gradient    — override header gradient (default: GRAD_ORANGE)
    """
    if not mailgun_ready():
        return

    # Tier badge (optional — shown when announcement is tier-targeted)
    tier_badge = (
        f'<tr><td style="padding:0 40px 0;text-align:center;">'
        f'<span style="display:inline-block;background:rgba(249,115,22,0.12);'
        f'border:1px solid rgba(249,115,22,0.3);color:#f97316;font-size:11px;'
        f'font-weight:700;text-transform:uppercase;letter-spacing:1px;'
        f'padding:4px 14px;border-radius:99px;">'
        f'For {tier_label} members</span>'
        f'</td></tr>'
        f'<tr><td style="height:20px;"></td></tr>'
        if tier_label else ""
    )

    # Format body: preserve simple line breaks as <br> if no HTML tags present
    formatted_body = body
    if "<" not in body:
        formatted_body = body.replace("\n\n", "</p><p style='margin:12px 0;color:#9ca3af;font-size:15px;line-height:1.7;'>")
        formatted_body = formatted_body.replace("\n", "<br>")

    # Optional CTA button
    btn_row = ""
    if cta_label:
        target_url = cta_url or URL_DASHBOARD
        btn_row = cta_button(cta_label, target_url, pt="20px", pb="20px")

    html = email_shell(
        gradient=gradient,
        tagline="A message from the UploadM8 team",
        body_rows=(
            tier_badge
            + f'<tr><td style="padding:36px 40px 20px;">'
            f'<h2 style="margin:0 0 16px;color:#ffffff;font-size:23px;font-weight:700;line-height:1.35;">'
            f'&#128226;&nbsp; {title}</h2>'
            f'<p style="margin:0;color:#9ca3af;font-size:15px;line-height:1.7;">'
            f'{formatted_body}</p>'
            f'</td></tr>'
            + btn_row
            + tinted_box(
                f'<p style="margin:0;color:#9ca3af;font-size:13px;line-height:1.6;">'
                f'Questions or feedback? Reply to this email or reach out at '
                f'<a href="mailto:{SUPPORT_EMAIL}" style="color:#f97316;text-decoration:none;">'
                f'{SUPPORT_EMAIL}</a>.</p>',
                hex_color="#374151",
                pb="36px",
            )
        ),
        footer_note="You received this announcement because you are an UploadM8 user.",
    )

    await send_email(to, f"&#128226; {title}", html)
