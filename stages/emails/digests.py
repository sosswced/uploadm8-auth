"""
UploadM8 — Digest Emails
========================
  send_monthly_user_kpi_digest_email -> monthly per-user KPI recap
  send_admin_weekly_kpi_digest_email -> weekly admin KPI summary
"""

import logging
from .base import (
    send_email, mailgun_ready, tier_label, MAIL_FROM_SUPPORT, SUPPORT_EMAIL,
    email_shell, intro_row, cta_button, tinted_box, stat_grid, secondary_links,
    section_tag, metric_hero, divider_accent,
    GRAD_BLUE, GRAD_PURPLE,
    URL_DASHBOARD, URL_BILLING, URL_PRICING, URL_SUPPORT, URL_LOGIN,
)

logger = logging.getLogger("uploadm8-worker")


async def send_monthly_user_kpi_digest_email(
    email: str,
    name: str,
    tier: str,
    period_label: str,
    uploads: int,
    success_rate_pct: int,
    views: int,
    likes: int,
    put_used: int,
    aic_used: int,
    put_balance: int,
    aic_balance: int,
) -> None:
    if not mailgun_ready():
        return

    plan = tier_label(tier)
    html = email_shell(
        gradient=GRAD_BLUE,
        tagline="Your monthly performance snapshot",
        preheader_text=f"Your {period_label} UploadM8 recap is ready: {uploads} uploads, {success_rate_pct}% success rate.",
        body_rows=(
            section_tag("Monthly KPI Digest", "#2563eb")
            + intro_row(
                f"Your {period_label} recap is ready, {name}",
                f"Here is your monthly summary for your <strong style='color:#60a5fa;'>{plan}</strong> account, "
                "including upload performance and token usage.",
            )
            + metric_hero(
                str(uploads),
                "Uploads This Month",
                f"{success_rate_pct}% successful completions",
                "#2563eb",
            )
            + stat_grid(
                ("Views", f"{views:,}"),
                ("Likes", f"{likes:,}"),
                ("PUT Used", f"{put_used:,}"),
                ("AIC Used", f"{aic_used:,}"),
            )
            + divider_accent("linear-gradient(90deg,rgba(37,99,235,0) 0%,#2563eb 50%,rgba(37,99,235,0) 100%)")
            + tinted_box(
                f'<p style="margin:0;color:#d1d5db;font-size:14px;line-height:1.7;">'
                f'<strong style="color:#ffffff;">Current wallet:</strong> {put_balance:,} PUT and {aic_balance:,} AIC available. '
                f'Need more capacity? Upgrade your tier or add a one-time top-up from Billing.</p>',
                hex_color="#2563eb",
            )
            + cta_button("Open Dashboard", URL_DASHBOARD, pt="16px", pb="20px")
            + secondary_links(
                ("Billing & Upgrades", URL_BILLING),
                ("Plans", URL_PRICING),
                ("Support", URL_SUPPORT),
            )
        ),
        footer_note="You received this monthly digest because email notifications are enabled.",
    )
    await send_email(
        email,
        f"UploadM8 monthly recap — {period_label}",
        html,
        from_addr=MAIL_FROM_SUPPORT,
        reply_to=SUPPORT_EMAIL,
    )


async def send_admin_weekly_kpi_digest_email(
    email: str,
    name: str,
    week_label: str,
    total_users: int,
    new_users: int,
    paid_users: int,
    uploads: int,
    revenue: float,
    cost: float,
    margin_pct: float,
) -> None:
    if not mailgun_ready():
        return

    html = email_shell(
        gradient=GRAD_PURPLE,
        tagline="Weekly platform leadership summary",
        preheader_text=f"{week_label}: {new_users} new users, {uploads} uploads, ${revenue:.2f} revenue.",
        body_rows=(
            section_tag("Admin Weekly KPI Digest", "#7c3aed")
            + intro_row(
                f"Weekly KPI summary, {name}",
                f"This report covers <strong style='color:#c4b5fd;'>{week_label}</strong> across user growth, "
                "usage, and margin indicators.",
            )
            + metric_hero(
                f"${revenue:,.0f}",
                "Weekly Revenue",
                f"estimated margin {margin_pct:.1f}%",
                "#7c3aed",
            )
            + stat_grid(
                ("Total Users", f"{total_users:,}"),
                ("New Users", f"{new_users:,}"),
                ("Paid Users", f"{paid_users:,}"),
                ("Uploads", f"{uploads:,}"),
            )
            + tinted_box(
                f'<p style="margin:0;color:#d1d5db;font-size:14px;line-height:1.7;">'
                f'<strong style="color:#ffffff;">Cost estimate:</strong> ${cost:,.2f}. '
                f'<strong style="color:#ffffff;">Revenue estimate:</strong> ${revenue:,.2f}. '
                f'This digest is for operational awareness and should be validated against Stripe exports for accounting.</p>',
                hex_color="#7c3aed",
                pb="36px",
            )
        ),
        footer_note="You received this digest because weekly admin reporting is enabled.",
    )
    await send_email(
        email,
        f"UploadM8 admin weekly KPI digest — {week_label}",
        html,
        from_addr=MAIL_FROM_SUPPORT,
        reply_to=SUPPORT_EMAIL,
    )


async def send_report_ready_email(
    email: str,
    name: str,
    report_title: str,
    download_url: str,
    expires_at_label: str,
) -> None:
    """Notify user that an async export is ready to download."""
    if not mailgun_ready():
        return

    html = email_shell(
        gradient=GRAD_BLUE,
        tagline="Your requested report is ready",
        preheader_text=f"{report_title} is ready. Download before {expires_at_label}.",
        body_rows=(
            section_tag("Report Ready", "#2563eb")
            + intro_row(
                f"{report_title} is ready, {name}",
                "Your async export completed successfully. Use the secure download link below.",
            )
            + cta_button("Download Report", download_url, pt="18px", pb="20px")
            + tinted_box(
                f'<p style="margin:0;color:#d1d5db;font-size:14px;line-height:1.7;">'
                f'This secure link expires on <strong style="color:#ffffff;">{expires_at_label}</strong>. '
                f'If it expires, request a new export from Analytics.</p>',
                hex_color="#2563eb",
            )
            + secondary_links(
                ("Open Analytics", URL_DASHBOARD),
                ("Sign In", URL_LOGIN),
                ("Support", URL_SUPPORT),
            )
        ),
        footer_note="You received this because you requested an analytics export.",
    )
    await send_email(
        email,
        f"UploadM8 report ready — {report_title}",
        html,
        from_addr=MAIL_FROM_SUPPORT,
        reply_to=SUPPORT_EMAIL,
    )
