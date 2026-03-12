"""
UploadM8 Email Base
===================
Single source of truth for:
  - Mailgun config + low-level sender (send_email)
  - Branded HTML shell   (email_shell)
  - All reusable row components

Every email module imports from here. Never call Mailgun directly elsewhere.
"""

import os
import logging
import httpx

logger = logging.getLogger("uploadm8-worker")

# ── Env config ────────────────────────────────────────────────────────────────
MAILGUN_API_KEY = os.environ.get("MAILGUN_API_KEY", "")
MAILGUN_DOMAIN  = os.environ.get("MAILGUN_DOMAIN", "")
MAIL_FROM       = os.environ.get("MAIL_FROM", "UploadM8 <no-reply@uploadm8.com>")
FRONTEND_URL    = os.environ.get("FRONTEND_URL", "https://app.uploadm8.com")

# ── Brand constants ───────────────────────────────────────────────────────────
SUPPORT_EMAIL = "support@uploadm8.com"
LOGO_URL      = "https://app.uploadm8.com/images/logo.png"

URL_DASHBOARD    = f"{FRONTEND_URL}/dashboard.html"
URL_SETTINGS     = f"{FRONTEND_URL}/settings.html"
URL_BILLING      = f"{FRONTEND_URL}/billing.html"
URL_PRICING      = f"{FRONTEND_URL}/index.html#pricing"
URL_UNSUBSCRIBE  = f"{FRONTEND_URL}/unsubscribe.html"

# ── Tier display names ────────────────────────────────────────────────────────
TIER_NAMES: dict = {
    "free":           "Free",
    "launch":         "Launch",
    "creator_pro":    "Creator Pro",
    "studio":         "Studio",
    "agency":         "Agency",
    "friends_family": "Friends & Family",
    "lifetime":       "Lifetime",
    "master_admin":   "Master Admin",
}

def tier_label(tier: str) -> str:
    return TIER_NAMES.get(tier, tier.replace("_", " ").title())

# ── Gradient presets ──────────────────────────────────────────────────────────
GRAD_ORANGE = "linear-gradient(135deg,#f97316 0%,#ea580c 50%,#c2410c 100%)"
GRAD_GREEN  = "linear-gradient(135deg,#16a34a 0%,#15803d 100%)"
GRAD_RED    = "linear-gradient(135deg,#dc2626 0%,#b91c1c 100%)"
GRAD_PURPLE = "linear-gradient(135deg,#7c3aed 0%,#6d28d9 100%)"
GRAD_BLUE   = "linear-gradient(135deg,#2563eb 0%,#1d4ed8 100%)"
GRAD_GOLD   = "linear-gradient(135deg,#d97706 0%,#92400e 100%)"
GRAD_DARK   = "linear-gradient(135deg,#374151 0%,#111827 100%)"
GRAD_TEAL   = "linear-gradient(135deg,#0d9488 0%,#0f766e 100%)"

# ── Mailgun sender ────────────────────────────────────────────────────────────
def mailgun_ready() -> bool:
    return bool(MAILGUN_API_KEY and MAILGUN_DOMAIN)


async def send_email(to: str, subject: str, html: str) -> None:
    """Low-level Mailgun sender. All email modules call this."""
    if not mailgun_ready():
        logger.info(f"Email skipped (Mailgun not configured): {to} | {subject}")
        return
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"https://api.mailgun.net/v3/{MAILGUN_DOMAIN}/messages",
                auth=("api", MAILGUN_API_KEY),
                data={"from": MAIL_FROM, "to": to, "subject": subject, "html": html},
            )
        if resp.status_code != 200:
            logger.warning(f"Mailgun failed ({resp.status_code}): {to} | {subject}")
        else:
            logger.info(f"Email sent → {to}: {subject}")
    except Exception as e:
        logger.warning(f"Mailgun error: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# HTML shell
# ═════════════════════════════════════════════════════════════════════════════

def email_shell(
    body_rows: str,
    gradient: str = GRAD_ORANGE,
    tagline: str = "Upload once. Publish everywhere.",
    footer_note: str = "",
) -> str:
    """
    Wraps body_rows (concatenated <tr> strings) in the full branded card.
    Logo is a clickable hyperlink to the app.

    Usage:
        html = email_shell(
            body_rows=intro_row(...) + cta_button(...),
            gradient=GRAD_GREEN,
        )
    """
    extra_footer = (
        f'<p style="margin:6px 0 0;color:#4b5563;font-size:12px;">{footer_note}</p>'
        if footer_note else ""
    )

    # Logo is wrapped in an <a> tag so it hyperlinks back to the app
    header = (
        f'<tr><td style="background:{gradient};padding:30px 40px 26px;text-align:center;">'
        f'<a href="{FRONTEND_URL}" style="display:inline-block;text-decoration:none;">'
        f'<img src="{LOGO_URL}" alt="UploadM8" height="46" '
        f'style="display:block;margin:0 auto 10px;max-width:200px;border:0;">'
        f'</a>'
        f'<p style="margin:0;color:rgba(255,255,255,0.82);font-size:13px;'
        f'letter-spacing:0.4px;">{tagline}</p>'
        f'</td></tr>'
    )

    hr_row = (
        '<tr><td style="padding:0 40px;">'
        '<hr style="border:none;border-top:1px solid rgba(255,255,255,0.07);margin:0;">'
        '</td></tr>'
    )

    footer = (
        '<tr><td style="padding:22px 40px;text-align:center;">'
        '<p style="margin:0 0 5px;color:#4b5563;font-size:13px;">'
        '&#169; 2025 UploadM8 &middot; All rights reserved</p>'
        f'<p style="margin:0 0 6px;color:#4b5563;font-size:12px;">Need help? '
        f'<a href="mailto:{SUPPORT_EMAIL}" style="color:#f97316;text-decoration:none;">'
        f'{SUPPORT_EMAIL}</a></p>'
        f'<p style="margin:0;color:#374151;font-size:11px;">'
        f'<a href="{URL_UNSUBSCRIBE}" style="color:#4b5563;text-decoration:underline;">Manage email preferences</a>'
        f' &middot; '
        f'<a href="{FRONTEND_URL}" style="color:#4b5563;text-decoration:underline;">app.uploadm8.com</a>'
        f'</p>'
        + extra_footer
        + '</td></tr>'
    )

    return (
        '<!DOCTYPE html><html lang="en"><head>'
        '<meta charset="UTF-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1.0">'
        '</head>'
        '<body style="margin:0;padding:0;background-color:#0f0f0f;'
        "font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;\">"
        '<table width="100%" cellpadding="0" cellspacing="0" '
        'style="background-color:#0f0f0f;padding:40px 20px;">'
        '<tr><td align="center">'
        '<table width="600" cellpadding="0" cellspacing="0" '
        'style="max-width:600px;width:100%;background-color:#1a1a1a;'
        'border-radius:16px;overflow:hidden;border:1px solid rgba(255,255,255,0.08);">'
        + header
        + body_rows
        + hr_row
        + footer
        + '</table></td></tr></table></body></html>'
    )


# ═════════════════════════════════════════════════════════════════════════════
# Reusable row components  (each returns one or more <tr>…</tr> strings)
# ═════════════════════════════════════════════════════════════════════════════

def intro_row(headline: str, body: str, pb: str = "4px") -> str:
    """Large heading + grey paragraph body copy."""
    return (
        f'<tr><td style="padding:36px 40px {pb};">'
        f'<h2 style="margin:0 0 12px;color:#ffffff;font-size:23px;'
        f'font-weight:700;line-height:1.35;">{headline}</h2>'
        f'<p style="margin:0;color:#9ca3af;font-size:15px;line-height:1.7;">{body}</p>'
        f'</td></tr>'
    )


def body_row(content: str, padding: str = "0 40px 28px") -> str:
    """Generic padded row for arbitrary HTML."""
    return f'<tr><td style="padding:{padding};">{content}</td></tr>'


def cta_button(label: str, url: str, pt: str = "8px", pb: str = "40px") -> str:
    """Primary orange CTA button row."""
    return (
        f'<tr><td style="padding:{pt} 40px {pb};text-align:center;">'
        f'<a href="{url}" style="display:inline-block;'
        'background:linear-gradient(135deg,#f97316,#ea580c);'
        'color:#ffffff;text-decoration:none;padding:15px 44px;'
        'border-radius:10px;font-size:16px;font-weight:700;letter-spacing:0.3px;">'
        f'{label} &#8594;</a>'
        '</td></tr>'
    )


def secondary_links(*links) -> str:
    """Small grey link row.  links = [("Label", "url"), ...]"""
    parts = " &nbsp;&middot;&nbsp; ".join(
        f'<a href="{u}" style="color:#6b7280;text-decoration:none;font-size:13px;">{l}</a>'
        for l, u in links
    )
    return (
        '<tr><td style="padding:0 40px 36px;text-align:center;">'
        f'{parts}</td></tr>'
    )


def tinted_box(inner_html: str, hex_color: str = "#f97316", pb: str = "28px") -> str:
    """Tinted rounded box with border."""
    rgb = _hex_rgb(hex_color)
    return (
        f'<tr><td style="padding:0 40px {pb};">'
        f'<table width="100%" cellpadding="0" cellspacing="0" '
        f'style="background:rgba({rgb},0.09);border:1px solid rgba({rgb},0.28);'
        f'border-radius:12px;">'
        f'<tr><td style="padding:20px 24px;">{inner_html}</td></tr>'
        f'</table></td></tr>'
    )


def stat_grid(*stats, pb: str = "28px") -> str:
    """
    Horizontal stat pills.  stats = [("Label", "Value"), ...]
    Rendered inside a tinted orange box.
    """
    cells = "".join(
        f'<td style="text-align:center;padding:0 16px;">'
        f'<p style="margin:0;color:#6b7280;font-size:11px;text-transform:uppercase;'
        f'letter-spacing:1px;">{lbl}</p>'
        f'<p style="margin:5px 0 0;color:#f97316;font-size:20px;font-weight:700;">{val}</p>'
        f'</td>'
        for lbl, val in stats
    )
    inner = f'<table cellpadding="0" cellspacing="0" align="center"><tr>{cells}</tr></table>'
    return tinted_box(inner, pb=pb)


def check_list(*items, hex_color: str = "#f97316", pb: str = "28px") -> str:
    """Checked feature list in a tinted box."""
    rows = "".join(
        f'<tr>'
        f'<td width="26" valign="top" style="padding-bottom:10px;">'
        f'<div style="width:19px;height:19px;background:{hex_color};border-radius:50%;'
        f'text-align:center;line-height:19px;color:#fff;font-size:11px;font-weight:700;">'
        f'&#10003;</div></td>'
        f'<td style="padding-left:10px;padding-bottom:10px;vertical-align:top;">'
        f'<p style="margin:0;color:#d1d5db;font-size:14px;line-height:1.55;">{item}</p>'
        f'</td></tr>'
        for item in items
    )
    return tinted_box(
        f'<table cellpadding="0" cellspacing="0" width="100%">{rows}</table>',
        hex_color=hex_color,
        pb=pb,
    )


def numbered_steps(*steps, pb: str = "32px") -> str:
    """Numbered step guide.  steps = [("Title", "Description"), ...]"""
    rows = "".join(
        f'<table width="100%" cellpadding="0" cellspacing="0" style="margin-bottom:14px;">'
        f'<tr>'
        f'<td width="36" valign="top">'
        f'<div style="width:30px;height:30px;background:#f97316;border-radius:50%;'
        f'text-align:center;line-height:30px;color:#fff;font-weight:700;font-size:14px;">{idx}</div>'
        f'</td>'
        f'<td style="padding-left:12px;vertical-align:top;">'
        f'<p style="margin:0;color:#ffffff;font-size:15px;font-weight:600;">{title}</p>'
        f'<p style="margin:4px 0 0;color:#6b7280;font-size:14px;line-height:1.5;">{desc}</p>'
        f'</td></tr></table>'
        for idx, (title, desc) in enumerate(steps, 1)
    )
    return body_row(rows, padding=f"0 40px {pb}")


def plan_change_visual(old_tier: str, new_tier: str, hex_new: str = "#f97316") -> str:
    """Old plan ➜ New plan visual inside a tinted box."""
    inner = (
        f'<table cellpadding="0" cellspacing="0" align="center"><tr>'
        f'<td style="text-align:center;padding:0 20px;">'
        f'<p style="margin:0;color:#6b7280;font-size:11px;text-transform:uppercase;letter-spacing:1px;">Previous Plan</p>'
        f'<p style="margin:5px 0 0;color:#9ca3af;font-size:18px;font-weight:600;">{tier_label(old_tier)}</p>'
        f'</td>'
        f'<td style="padding:0 16px;"><p style="margin:0;color:#f97316;font-size:26px;">&#8594;</p></td>'
        f'<td style="text-align:center;padding:0 20px;">'
        f'<p style="margin:0;color:{hex_new};font-size:11px;text-transform:uppercase;letter-spacing:1px;">New Plan</p>'
        f'<p style="margin:5px 0 0;color:{hex_new};font-size:18px;font-weight:700;">{tier_label(new_tier)}</p>'
        f'</td></tr></table>'
    )
    return tinted_box(inner, hex_color=hex_new)


def platform_logos_row() -> str:
    """TikTok / YouTube / Instagram / Facebook logo strip."""
    logos = [
        ("#000000", "https://logo.clearbit.com/tiktok.com",    "TikTok"),
        ("#ff0000", "https://logo.clearbit.com/youtube.com",   "YouTube"),
        ("linear-gradient(45deg,#f09433,#e6683c,#dc2743,#cc2366,#bc1888)",
                     "https://logo.clearbit.com/instagram.com", "Instagram"),
        ("#1877f2", "https://logo.clearbit.com/facebook.com",  "Facebook"),
    ]
    cells = "".join(
        f'<td style="padding:0 10px;text-align:center;">'
        f'<div style="width:50px;height:50px;background:{bg};border-radius:12px;'
        f'display:inline-block;">'
        f'<img src="{src}" alt="{name}" width="30" height="30" '
        f'style="display:block;margin:10px auto;border-radius:4px;"></div>'
        f'<p style="margin:6px 0 0;color:#9ca3af;font-size:11px;font-weight:500;">{name}</p>'
        f'</td>'
        for bg, src, name in logos
    )
    inner = (
        '<p style="margin:0 0 14px;color:#f97316;font-size:11px;font-weight:700;'
        'text-transform:uppercase;letter-spacing:1px;text-align:center;">Your Platforms</p>'
        f'<table cellpadding="0" cellspacing="0" align="center"><tr>{cells}</tr></table>'
    )
    return tinted_box(inner)


def receipt_row(label: str, value: str, is_total: bool = False) -> str:
    """Single line of a receipt table."""
    color   = "#ffffff" if is_total else "#9ca3af"
    weight  = "700"     if is_total else "400"
    border  = "border-top:1px solid rgba(255,255,255,0.07);padding-top:10px;margin-top:6px;" if is_total else ""
    return (
        f'<tr>'
        f'<td style="padding:5px 0;color:{color};font-size:14px;font-weight:{weight};{border}">{label}</td>'
        f'<td style="padding:5px 0;color:{color};font-size:14px;font-weight:{weight};{border}'
        f'text-align:right;">{value}</td>'
        f'</tr>'
    )


def alert_banner(msg: str, hex_color: str = "#ef4444", pb: str = "28px") -> str:
    return tinted_box(
        f'<p style="margin:0;color:#ffffff;font-size:15px;font-weight:600;">{msg}</p>',
        hex_color=hex_color,
        pb=pb,
    )


def spacer(h: str = "8px") -> str:
    return f'<tr><td style="height:{h};"></td></tr>'


# ── Internal ──────────────────────────────────────────────────────────────────
def _hex_rgb(h: str) -> str:
    h = h.lstrip("#")
    return f"{int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)}"
