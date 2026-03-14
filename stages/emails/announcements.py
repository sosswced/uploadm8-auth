"""
UploadM8 — Phase 5b: Announcement Email  (v2 — Enhanced Design)
================================================================
  send_announcement_email  → replaces the bare <h1>title</h1><p>body</p>
                             in _execute_announcement_deliveries()

Drop-in replacement. The caller signature stays identical — just swap the
send_email call in that function with send_announcement_email().

v2 upgrades:
  - preheader_text auto-generated from title + body snippet
  - section_tag "Announcement" badge above the title
  - Tier badge upgraded to use section_tag style
  - Body formatting preserved
  - CTA button optional as before
"""

import logging
from .base import (
    send_email, mailgun_ready,
    email_shell, intro_row, body_row, cta_button, tinted_box,
    secondary_links, spacer,
    section_tag, divider_accent,
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
    Branded version of the announcement email.

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
      tier_label  — shown as a tier-targeted badge if targeting a specific tier
      gradient    — override header gradient (default: GRAD_ORANGE)
    """
    if not mailgun_ready():
        return

    # Auto-generate preheader from body (first 120 chars, stripped of HTML)
    body_plain = body.replace("<br>", " ").replace("\n", " ")
    # Strip any basic HTML tags for preheader
    import re
    body_stripped = re.sub(r'<[^>]+>', '', body_plain)
    preheader = f"{title}: {body_stripped[:100]}..." if len(body_stripped) > 100 else f"{title}: {body_stripped}"

    # Tier targeting badge (shown when announcement is tier-targeted)
    tier_badge_row = ""
    if tier_label:
        tier_badge_row = (
            '<tr><td style="padding:20px 40px 0;text-align:center;">'
            f'<span style="display:inline-block;background:rgba(249,115,22,0.12);'
            f'border:1px solid rgba(249,115,22,0.35);color:#f97316;'
            f'font-size:10px;font-weight:800;text-transform:uppercase;'
            f'letter-spacing:2px;padding:5px 18px;border-radius:99px;">'
            f'For {tier_label} members</span>'
            f'</td></tr>'
            f'<tr><td style="height:8px;"></td></tr>'
        )

    # Format body: preserve simple line breaks as <br> if no HTML tags present
    formatted_body = body
    if "<" not in body:
        formatted_body = body.replace("\n\n", "</p><p style='margin:12px 0;color:#9ca3af;font-size:15px;line-height:1.75;'>")
        formatted_body = formatted_body.replace("\n", "<br>")

    # Optional CTA button
    btn_row = ""
    if cta_label:
        target_url = cta_url or URL_DASHBOARD
        btn_row = cta_button(cta_label, target_url, pt="20px", pb="20px")

    html = email_shell(
        gradient=gradient,
        tagline="A message from the UploadM8 team",
        preheader_text=preheader,
        body_rows=(
            section_tag("Announcement &#128226;", "#f97316")
            + tier_badge_row
            + f'<tr><td style="padding:28px 40px 20px;">'
            f'<h2 style="margin:0 0 16px;color:#ffffff;font-size:26px;font-weight:800;'
            f'line-height:1.3;letter-spacing:-0.3px;">'
            f'{title}</h2>'
            f'<p style="margin:0;color:#9ca3af;font-size:15px;line-height:1.75;">'
            f'{formatted_body}</p>'
            f'</td></tr>'
            + btn_row
            + divider_accent()
            + tinted_box(
                f'<p style="margin:0;color:#9ca3af;font-size:13px;line-height:1.65;">'
                f'Questions or feedback? Reply to this email or reach out at '
                f'<a href="mailto:{SUPPORT_EMAIL}" style="color:#f97316;text-decoration:none;">'
                f'{SUPPORT_EMAIL}</a>.</p>',
                hex_color="#374151",
                pb="36px",
            )
        ),
        footer_note="You received this announcement because you are an UploadM8 user.",
    )

    await send_email(to, f"📢 {title}", html)
