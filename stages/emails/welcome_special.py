"""
UploadM8 — Phase 5a: Special / Heartfelt Welcome Emails  (v2 — Enhanced Design)
=================================================================================
  send_friends_family_welcome_email  → friends_family tier granted
  send_agency_welcome_email          → agency tier starts (Stripe or admin grant)
  send_master_admin_welcome_email    → master_admin tier granted

v2 upgrades:
  - All emails have preheader_text
  - Friends & Family: metric_hero for "unlimited access" feel, warm section_tag
  - Agency: section_tag "Agency Plan Active", stat_grid for key details
  - Master Admin: section_tag "Full Access Granted", enhanced responsibility section
"""

import logging
from .base import (
    send_email, mailgun_ready,
    email_shell, intro_row, body_row, cta_button, tinted_box,
    check_list, platform_logos_row, secondary_links, spacer,
    section_tag, metric_hero, divider_accent, stat_grid,
    GRAD_ORANGE, GRAD_GOLD, GRAD_DARK, GRAD_PURPLE,
    URL_DASHBOARD, URL_SETTINGS, URL_BILLING, SUPPORT_EMAIL, FRONTEND_URL,
)

logger = logging.getLogger("uploadm8-worker")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Friends & Family welcome
# ─────────────────────────────────────────────────────────────────────────────
async def send_friends_family_welcome_email(email: str, name: str) -> None:
    """
    Fires when a user is granted the friends_family tier — either by an admin
    or via a special invite flow.

    Tone: warm, personal, genuine. These are people Earl trusts personally.
    """
    if not mailgun_ready():
        return

    html = email_shell(
        gradient=GRAD_ORANGE,
        tagline="Upload once. Publish everywhere.",
        preheader_text=f"You're one of us, {name}. Full access to UploadM8 — compliments of the team.",
        body_rows=(
            section_tag("Friends &amp; Family Access", "#f97316")
            + intro_row(
                f"Welcome to UploadM8, {name}. &#128149;",
                "You've been given access to UploadM8 as one of our Friends &amp; Family members — "
                "and that means more to us than any subscription ever could. "
                "This is the same platform we work on every single day, shared with you "
                "because we believe in what you're creating.",
            )
            + metric_hero(
                "&#128149;",
                "Friends &amp; Family Access",
                "full platform — no subscriptions, no credit card, just UploadM8",
                "#f97316",
            )
            + divider_accent()
            + tinted_box(
                '<p style="margin:0 0 10px;color:#f97316;font-size:13px;font-weight:700;'
                'text-transform:uppercase;letter-spacing:1px;">What you have access to</p>'
                '<p style="margin:0;color:#d1d5db;font-size:14px;line-height:1.75;">'
                'Your account has been set up with our Friends &amp; Family plan — full platform '
                'access, complimentary upload tokens, and everything you need to get your content '
                'out to the world. No subscriptions. No credit card. Just UploadM8, '
                'because you\'re one of us.</p>',
                hex_color="#f97316",
            )
            + platform_logos_row()
            + cta_button("Let's Get Started", URL_DASHBOARD, pt="24px", pb="20px")
            + tinted_box(
                '<p style="margin:0;color:#9ca3af;font-size:14px;line-height:1.75;">'
                'You have a direct line to us — always. If something doesn\'t work, '
                'if you have an idea, or if you just want to talk about the product, '
                f'reply to this email or reach out at '
                f'<a href="mailto:{SUPPORT_EMAIL}" style="color:#f97316;text-decoration:none;">'
                f'{SUPPORT_EMAIL}</a>. We mean it.</p>',
                hex_color="#374151",
                pb="36px",
            )
        ),
        footer_note="You received this because you were granted Friends & Family access to UploadM8.",
    )

    await send_email(email, f"Welcome to UploadM8, {name} — you're one of us ❤️", html)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Agency welcome
# ─────────────────────────────────────────────────────────────────────────────
async def send_agency_welcome_email(
    email: str,
    name: str,
    amount: float = 0.0,
    next_billing_date: str = "",
) -> None:
    """
    Fires when a user activates (or is granted) the agency tier.
    Tone: professional respect — they're running a real operation.
    """
    if not mailgun_ready():
        return

    billing_note = (
        f'<p style="margin:12px 0 0;color:#9ca3af;font-size:13px;">'
        f'Your plan renews on <strong style="color:#ffffff;">{next_billing_date}</strong> '
        f'at <strong style="color:#ffffff;">${amount:.2f}/mo</strong>.</p>'
        if amount and next_billing_date else ""
    )

    stats = [("Plan", "Agency")]
    if amount:
        stats.append(("Monthly Rate", f"${amount:.2f}"))
    if next_billing_date:
        stats.append(("Next Billing", next_billing_date))

    html = email_shell(
        gradient=GRAD_GOLD,
        tagline="Built for teams that create at scale",
        preheader_text=f"Your UploadM8 Agency account is live, {name}. Priority processing, max quotas, direct support.",
        body_rows=(
            section_tag("Agency Plan Active", "#d97706")
            + intro_row(
                f"Welcome to Agency, {name}. &#127775;",
                "We don't take lightly the trust it takes to run a creative agency and choose a "
                "new platform to be part of your workflow. Thank you — genuinely — for giving us "
                "that chance. We built the Agency tier for exactly what you do: "
                "high-volume publishing, multiple creators, and results that matter to clients.",
            )
            + stat_grid(*stats, pb="28px")
            + check_list(
                "Maximum upload quota across all four platforms",
                "Priority processing — your jobs move to the front of the queue",
                "Dedicated platform slots for your entire team",
                "AI caption generation at full capacity",
                "Direct support line for your account",
                hex_color="#d97706",
            )
            + platform_logos_row()
            + tinted_box(
                '<p style="margin:0;color:#d1d5db;font-size:14px;line-height:1.75;">'
                'We\'re actively building features that agency workflows demand — '
                'team management, client reporting, and bulk scheduling are on the roadmap. '
                'As an Agency member, your feedback directly shapes what we build next. '
                'Please don\'t hesitate to tell us what would make your operation run better.'
                + billing_note
                + '</p>',
                hex_color="#d97706",
            )
            + cta_button("Set Up Your Agency Account", URL_DASHBOARD, pt="20px", pb="20px")
            + secondary_links(
                ("Billing &amp; Invoices", URL_BILLING),
                ("Account Settings",       URL_SETTINGS),
            )
            + tinted_box(
                f'<p style="margin:0;color:#9ca3af;font-size:14px;line-height:1.75;">'
                f'Your direct line: '
                f'<a href="mailto:{SUPPORT_EMAIL}" style="color:#f97316;text-decoration:none;">'
                f'{SUPPORT_EMAIL}</a>. '
                f'For Agency accounts we aim to respond within a few hours. '
                f'We\'re here, and we\'re in your corner.</p>',
                hex_color="#374151",
                pb="36px",
            )
        ),
        footer_note="You received this because you activated an UploadM8 Agency plan.",
    )

    await send_email(email, f"Welcome to UploadM8 Agency — let's build something great, {name} 🌟", html)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Master Admin welcome
# ─────────────────────────────────────────────────────────────────────────────
async def send_master_admin_welcome_email(email: str, name: str) -> None:
    """
    Fires when a user is granted master_admin privileges — either via the
    BOOTSTRAP_ADMIN_EMAIL mechanism on startup, or via an admin panel grant.

    Tone: weight of responsibility. This is not a marketing email.
    """
    if not mailgun_ready():
        return

    html = email_shell(
        gradient=GRAD_DARK,
        tagline="UploadM8 Platform Administration",
        preheader_text=f"Master admin access has been granted to your account, {name}. Full platform controls are now active.",
        body_rows=(
            section_tag("Full Access Granted", "#6b7280")
            + intro_row(
                f"Full admin access granted, {name}.",
                "You now have master administrator access to UploadM8. "
                "This comes with the highest level of trust we extend — "
                "full visibility into the platform, user management, billing controls, "
                "and the ability to shape how the system operates for everyone on it.",
            )
            + metric_hero(
                "&#128737;&#65039;",
                "Master Admin",
                "full platform visibility and control — every action is logged",
                "#6b7280",
            )
            + divider_accent(
                "linear-gradient(90deg,rgba(107,114,128,0) 0%,#6b7280 50%,rgba(107,114,128,0) 100%)"
            )
            + check_list(
                "Full user management — view, edit, ban, restore",
                "Billing and subscription controls for all users",
                "Wallet credits — grant PUT and AIC tokens",
                "Admin audit log — every action is tracked",
                "Announcement system — email all users",
                "KPI dashboards and revenue analytics",
                "Platform-wide settings and configuration",
                hex_color="#6b7280",
            )
            + tinted_box(
                '<p style="margin:0 0 10px;color:#ffffff;font-size:15px;font-weight:700;">'
                'A note on responsibility</p>'
                '<p style="margin:0;color:#9ca3af;font-size:14px;line-height:1.75;">'
                'With admin access comes responsibility for the people using this platform. '
                'Every action you take as an admin is logged and timestamped. '
                'User data, payment information, and personal details must be handled '
                'with care and respect. We trust you completely — and we know you\'ll '
                'handle this the right way.</p>',
                hex_color="#374151",
            )
            + cta_button("Open Admin Dashboard", f"{FRONTEND_URL}/admin.html", pt="20px", pb="20px")
            + tinted_box(
                f'<p style="margin:0;color:#9ca3af;font-size:14px;line-height:1.75;">'
                f'If you believe you received this in error, or if you have questions about '
                f'your access level, contact the platform owner immediately at '
                f'<a href="mailto:{SUPPORT_EMAIL}" style="color:#f97316;text-decoration:none;">'
                f'{SUPPORT_EMAIL}</a>.</p>',
                hex_color="#374151",
                pb="36px",
            )
        ),
        footer_note="You received this because master admin privileges were granted to this account.",
    )

    await send_email(email, "Master Admin access granted — UploadM8 Platform", html)
