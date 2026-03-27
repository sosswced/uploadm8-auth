"""
UploadM8 Email Base  —  v2 (Enhanced Design)
=============================================
Single source of truth for:
  - Mailgun config + low-level sender (send_email)
  - Branded HTML shell (email_shell) with preheader support
  - All reusable row components (enhanced + new)

Every email module imports from here. Never call Mailgun directly elsewhere.

DESIGN v2 UPGRADES:
  - preheader_text parameter on email_shell (inbox preview text — huge for open rates)
  - Gradient orange accent stripe below every header
  - Enhanced CTA button with glow box-shadow and deeper gradient
  - Left-border accent on all tinted_box components
  - Bordered, padded stat pill grid
  - Larger gradient check-circles in check_list
  - Improved receipt_row with separated total row
  - Improved footer with subtle border-top separator
  - NEW: divider_accent()  — gradient section separator
  - NEW: section_tag()     — small pill badge label (e.g. "Action Required")
  - NEW: metric_hero()     — large centered number for key stats
  - NEW: progress_bar()    — visual % bar for tokens / days remaining
"""

import os
import logging
import httpx

logger = logging.getLogger("uploadm8-worker")

# ── Env config ────────────────────────────────────────────────────────────────
MAILGUN_API_KEY = os.environ.get("MAILGUN_API_KEY", "")
MAILGUN_DOMAIN  = os.environ.get("MAILGUN_DOMAIN", "")
MAIL_FROM       = os.environ.get("MAIL_FROM", "UploadM8 <no-reply@uploadm8.com>")
MAIL_FROM_SUPPORT = os.environ.get("MAIL_FROM_SUPPORT", "UploadM8 Support <support@uploadm8.com>")
MAIL_FROM_HELLO   = os.environ.get("MAIL_FROM_HELLO", "UploadM8 <hello@uploadm8.com>")
FRONTEND_URL    = os.environ.get("FRONTEND_URL", "https://app.uploadm8.com").rstrip("/")

# Public Discord (community) — override via env for staging if needed
DISCORD_INVITE_URL = os.environ.get("DISCORD_INVITE_URL", "https://discord.gg/TVDAc8fnwu")

# ── Brand constants ───────────────────────────────────────────────────────────
SUPPORT_EMAIL = "support@uploadm8.com"
LOGO_URL      = f"{FRONTEND_URL}/images/logo.png"

URL_DASHBOARD    = f"{FRONTEND_URL}/dashboard.html"
URL_SETTINGS     = f"{FRONTEND_URL}/settings.html"
URL_SETTINGS_TOKENS = f"{FRONTEND_URL}/settings.html#billing-panel"
URL_BILLING      = f"{FRONTEND_URL}/billing.html"
URL_PRICING      = f"{FRONTEND_URL}/index.html#pricing"
URL_UNSUBSCRIBE  = f"{FRONTEND_URL}/unsubscribe.html"
URL_CONTACT      = f"{FRONTEND_URL}/contact.html"
URL_SUPPORT      = f"{FRONTEND_URL}/support.html"
URL_LOGIN        = f"{FRONTEND_URL}/login.html"
URL_TERMS        = f"{FRONTEND_URL}/terms.html"
URL_PRIVACY      = f"{FRONTEND_URL}/privacy.html"


def asset_url(path: str) -> str:
    """Absolute URL for static images in transactional email (logo, platform icons)."""
    return f"{FRONTEND_URL}/{path.lstrip('/')}"

# ── Tier display names ────────────────────────────────────────────────────────
TIER_NAMES: dict = {
    "free":           "Free",
    "launch":         "Launch",
    "creator_lite":   "Creator Lite",
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


async def send_email(to: str, subject: str, html: str, from_addr: str = None, reply_to: str = None) -> None:
    """Low-level Mailgun sender. All email modules call this."""
    sender = from_addr or MAIL_FROM
    data = {"from": sender, "to": to, "subject": subject, "html": html}
    if reply_to:
        data["h:Reply-To"] = reply_to
    if not mailgun_ready():
        logger.info(f"Email skipped (Mailgun not configured): {to} | {subject}")
        return
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"https://api.mailgun.net/v3/{MAILGUN_DOMAIN}/messages",
                auth=("api", MAILGUN_API_KEY),
                data=data,
            )
        if resp.status_code != 200:
            logger.warning(f"Mailgun failed ({resp.status_code}): {to} | {subject}")
        else:
            logger.info(f"Email sent → {to}: {subject}")
    except Exception as e:
        logger.warning(f"Mailgun error: {e}")


# ── Internal helpers ──────────────────────────────────────────────────────────
def _hex_rgb(h: str) -> str:
    h = h.lstrip("#")
    return f"{int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)}"


def _preheader_block(text: str) -> str:
    """
    Invisible preview text shown in inbox list before the email is opened.
    Padded with zero-width non-joiner characters so no body text leaks into preview.
    This single addition materially improves open rates.
    """
    filler = "&zwnj;&nbsp;" * 80
    return (
        f'<div style="display:none;font-size:1px;color:#0f0f0f;'
        f'line-height:1px;max-height:0px;max-width:0px;opacity:0;overflow:hidden;">'
        f'{text} {filler}'
        f'</div>'
    )


# ═════════════════════════════════════════════════════════════════════════════
# HTML shell  —  v2
# ═════════════════════════════════════════════════════════════════════════════

def email_shell(
    body_rows: str,
    gradient: str = GRAD_ORANGE,
    tagline: str = "Upload once. Publish everywhere.",
    footer_note: str = "",
    preheader_text: str = "",
) -> str:
    """
    Wraps body_rows (concatenated <tr> strings) in the full branded card.

    v2 changes:
      - preheader_text: visible in inbox preview pane before opening
      - Gradient orange accent stripe immediately below the header image
      - Improved footer with border-top separator
      - Logo is a clickable hyperlink to the app

    Usage:
        html = email_shell(
            body_rows=intro_row(...) + cta_button(...),
            gradient=GRAD_GREEN,
            preheader_text="Your upload just went live on 4 platforms!",
        )
    """
    preheader_div = _preheader_block(preheader_text) if preheader_text else ""

    extra_footer = (
        f'<p style="margin:8px 0 0;color:#4b5563;font-size:12px;">{footer_note}</p>'
        if footer_note else ""
    )

    header = (
        f'<tr><td style="background:{gradient};padding:34px 40px 30px;text-align:center;">'
        f'<a href="{FRONTEND_URL}" style="display:inline-block;text-decoration:none;">'
        f'<img src="{LOGO_URL}" alt="UploadM8" height="46" '
        f'style="display:block;margin:0 auto 12px;max-width:200px;border:0;">'
        f'</a>'
        f'<p style="margin:0;color:rgba(255,255,255,0.85);font-size:13px;'
        f'letter-spacing:0.5px;font-weight:500;">{tagline}</p>'
        f'</td></tr>'
    )

    # Orange-to-transparent glow stripe — brand signature below every header
    accent_stripe = (
        '<tr><td style="padding:0;line-height:0;font-size:0;">'
        '<div style="height:3px;'
        'background:linear-gradient(90deg,#f97316 0%,#fb923c 35%,#fdba74 65%,rgba(249,115,22,0) 100%);">'
        '</div></td></tr>'
    )

    sign_in_row = (
        '<tr><td style="padding:18px 40px 4px;text-align:center;">'
        f'<a href="{URL_LOGIN}" '
        'style="display:inline-block;'
        'background:linear-gradient(135deg,#2563eb 0%,#1d4ed8 100%);'
        'color:#ffffff;text-decoration:none;padding:10px 26px;'
        'border-radius:8px;font-size:13px;font-weight:700;letter-spacing:0.3px;'
        'border:1px solid rgba(255,255,255,0.15);'
        'box-shadow:0 2px 10px rgba(37,99,235,0.35);">'
        'Sign In to UploadM8 &#8594;</a>'
        '</td></tr>'
    )

    quick_links = (
        '<tr><td style="padding:22px 40px 0;text-align:center;">'
        '<p style="margin:0 0 8px;color:#9ca3af;font-size:11px;font-weight:700;'
        'text-transform:uppercase;letter-spacing:0.12em;">Quick links</p>'
        '<p style="margin:0 0 10px;color:#6b7280;font-size:13px;line-height:1.85;">'
        f'<a href="{URL_DASHBOARD}" style="color:#f97316;text-decoration:none;">Dashboard</a>'
        f' &middot; '
        f'<a href="{URL_SETTINGS_TOKENS}" style="color:#f97316;text-decoration:none;">Token balance</a>'
        f' &middot; '
        f'<a href="{URL_CONTACT}" style="color:#f97316;text-decoration:none;">Contact</a>'
        f' &middot; '
        f'<a href="{DISCORD_INVITE_URL}" style="color:#5865F2;text-decoration:none;">Discord community</a>'
        '</p>'
        '<p style="margin:0 0 14px;color:#6b7280;font-size:12px;line-height:1.75;">'
        f'<a href="{URL_TERMS}" style="color:#6b7280;text-decoration:underline;">Terms of Service</a>'
        f' &middot; '
        f'<a href="{URL_PRIVACY}" style="color:#6b7280;text-decoration:underline;">Privacy Policy</a>'
        f' &middot; '
        f'<a href="{URL_SUPPORT}" style="color:#6b7280;text-decoration:underline;">Support</a>'
        '</p>'
        '</td></tr>'
    )

    footer = (
        sign_in_row
        + quick_links
        + '<tr><td style="padding:18px 40px 24px;text-align:center;'
        'border-top:1px solid rgba(255,255,255,0.06);">'
        '<p style="margin:0 0 6px;color:#4b5563;font-size:13px;">'
        '&#169; 2026 UploadM8 &middot; All rights reserved</p>'
        f'<p style="margin:0 0 8px;color:#4b5563;font-size:12px;">Need help? '
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
        '<title>UploadM8</title>'
        '</head>'
        '<body style="margin:0;padding:0;background-color:#0f0f0f;'
        "font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;\">"
        + preheader_div
        + '<table width="100%" cellpadding="0" cellspacing="0" '
        'style="background-color:#0f0f0f;padding:40px 20px;">'
        '<tr><td align="center">'
        '<table width="600" cellpadding="0" cellspacing="0" '
        'style="max-width:600px;width:100%;background-color:#1a1a1a;'
        'border-radius:16px;overflow:hidden;border:1px solid rgba(255,255,255,0.09);">'
        + header
        + accent_stripe
        + body_rows
        + footer
        + '</table></td></tr></table></body></html>'
    )


# ═════════════════════════════════════════════════════════════════════════════
# Core row components
# ═════════════════════════════════════════════════════════════════════════════

def intro_row(headline: str, body: str, pb: str = "4px") -> str:
    """Large heading + grey paragraph body copy. v2: 26px headline, 800 weight."""
    return (
        f'<tr><td style="padding:40px 40px {pb};">'
        f'<h2 style="margin:0 0 14px;color:#ffffff;font-size:26px;'
        f'font-weight:800;line-height:1.3;letter-spacing:-0.3px;">{headline}</h2>'
        f'<p style="margin:0;color:#9ca3af;font-size:15px;line-height:1.75;">{body}</p>'
        f'</td></tr>'
    )


def body_row(content: str, padding: str = "0 40px 28px") -> str:
    """Generic padded row for arbitrary HTML."""
    return f'<tr><td style="padding:{padding};">{content}</td></tr>'


def cta_button(label: str, url: str, pt: str = "8px", pb: str = "40px") -> str:
    """
    Primary orange CTA button row.
    v2: deeper gradient, 800 weight, subtle border + orange glow box-shadow.
    """
    return (
        f'<tr><td style="padding:{pt} 40px {pb};text-align:center;">'
        f'<a href="{url}" style="display:inline-block;'
        'background:linear-gradient(135deg,#f97316 0%,#ea580c 55%,#c2410c 100%);'
        'color:#ffffff;text-decoration:none;padding:16px 50px;'
        'border-radius:10px;font-size:16px;font-weight:800;letter-spacing:0.4px;'
        'border:1px solid rgba(255,255,255,0.18);'
        "box-shadow:0 4px 20px rgba(249,115,22,0.4),0 2px 6px rgba(0,0,0,0.5);\">"
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
    """
    Tinted rounded box with border.
    v2: left accent bar (3px solid colored border-left) for visual depth.
    """
    rgb = _hex_rgb(hex_color)
    return (
        f'<tr><td style="padding:0 40px {pb};">'
        f'<table width="100%" cellpadding="0" cellspacing="0" '
        f'style="background:rgba({rgb},0.08);border:1px solid rgba({rgb},0.25);'
        f'border-left:3px solid {hex_color};border-radius:10px;">'
        f'<tr><td style="padding:20px 22px;">{inner_html}</td></tr>'
        f'</table></td></tr>'
    )


def stat_grid(*stats, pb: str = "28px") -> str:
    """
    Horizontal stat pills. v2: each stat is a bordered pill with padding.
    stats = [("Label", "Value"), ...]
    """
    cells = "".join(
        f'<td style="text-align:center;padding:0 6px;">'
        f'<div style="background:rgba(249,115,22,0.08);'
        f'border:1px solid rgba(249,115,22,0.22);border-radius:10px;padding:14px 18px;">'
        f'<p style="margin:0;color:#6b7280;font-size:10px;text-transform:uppercase;'
        f'letter-spacing:1.2px;font-weight:600;">{lbl}</p>'
        f'<p style="margin:6px 0 0;color:#f97316;font-size:21px;font-weight:800;'
        f'letter-spacing:-0.5px;">{val}</p>'
        f'</div>'
        f'</td>'
        for lbl, val in stats
    )
    return (
        f'<tr><td style="padding:0 40px {pb};">'
        f'<table cellpadding="0" cellspacing="0" align="center" width="100%">'
        f'<tr>{cells}</tr></table>'
        f'</td></tr>'
    )


def check_list(*items, hex_color: str = "#f97316", pb: str = "28px") -> str:
    """
    Checked feature list in a tinted box.
    v2: 22px gradient circles, 800-weight checkmark.
    """
    rows = "".join(
        f'<tr>'
        f'<td width="34" valign="top" style="padding-bottom:12px;">'
        f'<div style="width:22px;height:22px;'
        f'background:linear-gradient(135deg,{hex_color},{hex_color}bb);'
        f'border-radius:50%;text-align:center;line-height:22px;'
        f'color:#fff;font-size:12px;font-weight:800;">'
        f'&#10003;</div></td>'
        f'<td style="padding-left:10px;padding-bottom:12px;vertical-align:top;">'
        f'<p style="margin:0;color:#d1d5db;font-size:14px;line-height:1.6;">{item}</p>'
        f'</td></tr>'
        for item in items
    )
    return tinted_box(
        f'<table cellpadding="0" cellspacing="0" width="100%">{rows}</table>',
        hex_color=hex_color,
        pb=pb,
    )


def numbered_steps(*steps, pb: str = "32px") -> str:
    """Numbered step guide. steps = [("Title", "Description"), ...]"""
    rows = "".join(
        f'<table width="100%" cellpadding="0" cellspacing="0" style="margin-bottom:16px;">'
        f'<tr>'
        f'<td width="42" valign="top">'
        f'<div style="width:32px;height:32px;'
        f'background:linear-gradient(135deg,#f97316,#ea580c);border-radius:50%;'
        f'text-align:center;line-height:32px;color:#fff;font-weight:800;font-size:14px;">{idx}</div>'
        f'</td>'
        f'<td style="padding-left:12px;vertical-align:top;">'
        f'<p style="margin:0;color:#ffffff;font-size:15px;font-weight:700;">{title}</p>'
        f'<p style="margin:5px 0 0;color:#6b7280;font-size:14px;line-height:1.55;">{desc}</p>'
        f'</td></tr></table>'
        for idx, (title, desc) in enumerate(steps, 1)
    )
    return body_row(rows, padding=f"0 40px {pb}")


def plan_change_visual(old_tier: str, new_tier: str, hex_new: str = "#f97316") -> str:
    """Old plan  New plan visual inside a tinted box. v2: bolder type, strikethrough old plan."""
    inner = (
        f'<table cellpadding="0" cellspacing="0" align="center"><tr>'
        f'<td style="text-align:center;padding:0 22px;">'
        f'<p style="margin:0;color:#6b7280;font-size:10px;text-transform:uppercase;'
        f'letter-spacing:1.2px;font-weight:600;">Previous Plan</p>'
        f'<p style="margin:8px 0 0;color:#6b7280;font-size:19px;font-weight:700;'
        f'text-decoration:line-through;opacity:0.7;">{tier_label(old_tier)}</p>'
        f'</td>'
        f'<td style="padding:0 18px;">'
        f'<p style="margin:0;color:#f97316;font-size:30px;font-weight:900;line-height:1;">&#8594;</p>'
        f'</td>'
        f'<td style="text-align:center;padding:0 22px;">'
        f'<p style="margin:0;color:{hex_new};font-size:10px;text-transform:uppercase;'
        f'letter-spacing:1.2px;font-weight:600;">New Plan</p>'
        f'<p style="margin:8px 0 0;color:{hex_new};font-size:21px;font-weight:800;">{tier_label(new_tier)}</p>'
        f'</td></tr></table>'
    )
    return tinted_box(inner, hex_color=hex_new)


def platform_logos_row() -> str:
    """TikTok / YouTube / Instagram / Facebook — hosted on FRONTEND_URL for reliable email loading."""
    logos = [
        ("#1a1a1a", asset_url("images/platforms/tiktok.png"), "TikTok"),
        ("#ff0000", asset_url("images/platforms/youtube.png"), "YouTube"),
        ("#c13584", asset_url("images/platforms/instagram.png"), "Instagram"),
        ("#1877f2", asset_url("images/platforms/facebook.png"), "Facebook"),
    ]
    cells = "".join(
        f'<td style="padding:0 8px;text-align:center;">'
        f'<div style="width:52px;height:52px;background:{bg};border-radius:14px;'
        f'display:inline-block;border:1px solid rgba(255,255,255,0.14);">'
        f'<img src="{src}" alt="{name}" width="30" height="30" '
        f'style="display:block;margin:11px auto;border-radius:4px;"></div>'
        f'<p style="margin:7px 0 0;color:#9ca3af;font-size:11px;font-weight:600;">{name}</p>'
        f'</td>'
        for bg, src, name in logos
    )
    inner = (
        '<p style="margin:0 0 16px;color:#f97316;font-size:10px;font-weight:700;'
        'text-transform:uppercase;letter-spacing:1.5px;text-align:center;">Your Platforms</p>'
        f'<table cellpadding="0" cellspacing="0" align="center"><tr>{cells}</tr></table>'
    )
    return tinted_box(inner)


def receipt_row(label: str, value: str, is_total: bool = False) -> str:
    """
    Single line of a receipt table.
    v2: total row has a separator line above it and orange value color.
    """
    if is_total:
        return (
            f'<tr><td colspan="2" style="padding:8px 0 0;">'
            f'<div style="height:1px;background:rgba(255,255,255,0.08);margin-bottom:8px;"></div>'
            f'</td></tr>'
            f'<tr>'
            f'<td style="padding:4px 0 4px;color:#ffffff;font-size:15px;font-weight:800;">{label}</td>'
            f'<td style="padding:4px 0 4px;color:#f97316;font-size:15px;font-weight:800;'
            f'text-align:right;">{value}</td>'
            f'</tr>'
        )
    return (
        f'<tr>'
        f'<td style="padding:6px 0;color:#9ca3af;font-size:14px;">{label}</td>'
        f'<td style="padding:6px 0;color:#d1d5db;font-size:14px;text-align:right;">{value}</td>'
        f'</tr>'
    )


def alert_banner(msg: str, hex_color: str = "#ef4444", pb: str = "28px") -> str:
    """Alert / warning banner using tinted_box with left accent border."""
    return tinted_box(
        f'<p style="margin:0;color:#ffffff;font-size:15px;font-weight:600;line-height:1.55;">{msg}</p>',
        hex_color=hex_color,
        pb=pb,
    )


def spacer(h: str = "8px") -> str:
    return f'<tr><td style="height:{h};"></td></tr>'


# ═════════════════════════════════════════════════════════════════════════════
# NEW v2 components
# ═════════════════════════════════════════════════════════════════════════════

def divider_accent(
    gradient: str = "linear-gradient(90deg,rgba(249,115,22,0) 0%,#f97316 50%,rgba(249,115,22,0) 100%)"
) -> str:
    """
    Centered gradient divider line — visual separator between email sections.
    Default is a centered orange fade. Pass a custom gradient to match the email theme.
    """
    return (
        '<tr><td style="padding:4px 40px 28px;">'
        f'<div style="height:1px;background:{gradient};border-radius:999px;"></div>'
        '</td></tr>'
    )


def section_tag(text: str, hex_color: str = "#f97316") -> str:
    """
    Small pill badge — used to label a section before the intro_row.
    Great for 'ACTION REQUIRED', 'CONFIRMED', 'SECURITY ALERT', 'NEW FEATURE', etc.
    """
    rgb = _hex_rgb(hex_color)
    return (
        '<tr><td style="padding:28px 40px 0;text-align:center;">'
        f'<span style="display:inline-block;background:rgba({rgb},0.12);'
        f'border:1px solid rgba({rgb},0.35);color:{hex_color};'
        f'font-size:10px;font-weight:800;text-transform:uppercase;'
        f'letter-spacing:2px;padding:5px 18px;border-radius:99px;">'
        f'{text}</span>'
        '</td></tr>'
    )


def metric_hero(
    value: str,
    label: str,
    sublabel: str = "",
    hex_color: str = "#f97316",
    pb: str = "28px",
) -> str:
    """
    Large centered metric — the focal point for key numbers.
    value    = big bold number or text  (e.g. "30", "+500 PUT", "4 Platforms")
    label    = descriptor below value   (e.g. "Free Tokens Waiting", "Now Live")
    sublabel = optional smaller note    (e.g. "ready to use right now")

    Perfect for: token grants, platform counts, amounts, days remaining.
    """
    sublabel_html = (
        f'<p style="margin:6px 0 0;color:#6b7280;font-size:13px;">{sublabel}</p>'
        if sublabel else ""
    )
    inner = (
        f'<div style="text-align:center;padding:10px 0 4px;">'
        f'<p style="margin:0;color:{hex_color};font-size:54px;font-weight:900;'
        f'letter-spacing:-2px;line-height:1;">{value}</p>'
        f'<p style="margin:10px 0 0;color:#ffffff;font-size:13px;font-weight:700;'
        f'text-transform:uppercase;letter-spacing:1.2px;">{label}</p>'
        + sublabel_html
        + '</div>'
    )
    return tinted_box(inner, hex_color=hex_color, pb=pb)


def progress_bar(
    pct: int,
    label: str = "",
    hex_color: str = "#f97316",
    pb: str = "28px",
) -> str:
    """
    Visual progress / usage bar.
    pct   = 0–100 integer representing remaining percentage
    label = optional text above the bar

    Use for: token balance remaining, trial days left, quota used.
    """
    rgb = _hex_rgb(hex_color)
    pct_safe = max(0, min(100, pct))
    label_html = (
        f'<p style="margin:0 0 10px;color:#9ca3af;font-size:13px;font-weight:500;">{label}</p>'
        if label else ""
    )
    inner = (
        label_html
        + f'<div style="background:rgba({rgb},0.15);border-radius:999px;height:10px;overflow:hidden;">'
        + f'<div style="width:{pct_safe}%;height:10px;'
        + f'background:linear-gradient(90deg,{hex_color},{hex_color}99);'
        + f'border-radius:999px;"></div>'
        + f'</div>'
        + f'<p style="margin:8px 0 0;color:{hex_color};font-size:12px;font-weight:700;'
        + f'text-align:right;">{pct_safe}% remaining</p>'
    )
    return tinted_box(inner, hex_color=hex_color, pb=pb)
