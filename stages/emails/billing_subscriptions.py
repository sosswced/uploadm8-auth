"""
UploadM8 — Phase 2: Billing Subscription Emails  (v2 — Enhanced Design)
=========================================================================
  send_subscription_started_email  → checkout.session.completed (subscription, non-trial)
  send_trial_started_email         → checkout.session.completed (trial)
  send_trial_cancelled_email       → customer.subscription.deleted where cancel_at_period_end
                                     was set during a trial
  send_subscription_cancelled_email→ customer.subscription.deleted (paid, post-trial)
  send_renewal_receipt_email       → invoice.paid (recurring)

v2 upgrades:
  - All emails have preheader_text
  - Subscription started: section_tag "Subscription Active", metric_hero
  - Trial started: section_tag "Trial Active", metric_hero for trial days
  - Trial cancelled: section_tag "Trial Cancelled", cleaner layout
  - Subscription cancelled: section_tag "Subscription Cancelled", churn-prevention focus
  - Renewal receipt: section_tag "Payment Confirmed", improved receipt table
"""

import logging
from datetime import datetime, timezone
from .base import (
    send_email, mailgun_ready, tier_label, MAIL_FROM_SUPPORT,
    email_shell, intro_row, body_row, cta_button, tinted_box,
    check_list, stat_grid, secondary_links, alert_banner,
    receipt_row, platform_logos_row, spacer,
    section_tag, metric_hero, divider_accent,
    GRAD_ORANGE, GRAD_GREEN, GRAD_RED, GRAD_BLUE,
    URL_DASHBOARD, URL_BILLING, URL_SETTINGS, URL_PRICING,
    SUPPORT_EMAIL,
)

logger = logging.getLogger("uploadm8-worker")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Subscription started  (paid, no trial)
# ─────────────────────────────────────────────────────────────────────────────
async def send_subscription_started_email(
    email: str,
    name: str,
    tier: str,
    amount: float,
    next_billing_date: str = "",
) -> None:
    """
    Fires from checkout.session.completed when mode=subscription and
    there is no trial_period_days.
    """
    if not mailgun_ready():
        return

    plan = tier_label(tier)
    next_date = next_billing_date or "your next billing cycle"

    html = email_shell(
        gradient=GRAD_GREEN,
        tagline="You're officially live on UploadM8",
        preheader_text=f"Your {plan} subscription is active, {name}. Full access unlocked — let's get publishing.",
        body_rows=(
            section_tag("Subscription Active", "#16a34a")
            + intro_row(
                f"Welcome to {plan}, {name}! &#127881;",
                f"Your <strong style='color:#22c55e;'>{plan}</strong> subscription is active. "
                "You now have full access to your plan's upload quota, AI captions, and "
                "multi-platform publishing — all in one place.",
            )
            + stat_grid(
                ("Plan", plan),
                ("Amount", f"${amount:.2f}/mo"),
                ("Next Billing", next_date),
            )
            + platform_logos_row()
            + cta_button("Go to Dashboard", URL_DASHBOARD, pt="24px", pb="20px")
            + secondary_links(
                ("Manage Billing", URL_BILLING),
                ("Settings", URL_SETTINGS),
                ("View Plans", URL_PRICING),
            )
        ),
        footer_note="You received this because you started an UploadM8 subscription.",
    )

    await send_email(email, f"Welcome to UploadM8 {plan} — your subscription is active", html, from_addr=MAIL_FROM_SUPPORT, reply_to=SUPPORT_EMAIL)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Trial started
# ─────────────────────────────────────────────────────────────────────────────
async def send_trial_started_email(
    email: str,
    name: str,
    tier: str,
    trial_end_date: str,
    trial_days: int = 14,
) -> None:
    """
    Fires from checkout.session.completed when trial_period_days is set.
    """
    if not mailgun_ready():
        return

    plan = tier_label(tier)

    html = email_shell(
        gradient=GRAD_BLUE,
        tagline="Your free trial starts now",
        preheader_text=f"Your {plan} free trial is live! {trial_days} days of full access — no charge until {trial_end_date}.",
        body_rows=(
            section_tag("Trial Active", "#2563eb")
            + intro_row(
                f"Your {plan} trial is live, {name}! &#127385;",
                f"You have <strong style='color:#60a5fa;'>{trial_days} days</strong> of full "
                f"<strong style='color:#60a5fa;'>{plan}</strong> access — no charge until "
                f"<strong style='color:#ffffff;'>{trial_end_date}</strong>. "
                "Cancel any time before then and you'll never be billed.",
            )
            + metric_hero(
                str(trial_days),
                "Free Trial Days",
                f"full {plan} access — cancel any time",
                "#2563eb",
            )
            + stat_grid(
                ("Trial Length", f"{trial_days} days"),
                ("Plan", plan),
                ("Charge Date", trial_end_date),
            )
            + check_list(
                f"Full {plan} quota unlocked immediately",
                "AI captions &amp; smart scheduling enabled",
                "All four platforms ready to connect",
                "Cancel before trial ends — zero charge",
                hex_color="#3b82f6",
            )
            + cta_button("Start Uploading Now", URL_DASHBOARD, pt="24px", pb="20px")
            + secondary_links(
                ("Manage Subscription", URL_BILLING),
                ("View Your Plan", URL_PRICING),
            )
        ),
        footer_note=f"Your trial ends on {trial_end_date}. We'll email you a reminder before any charge.",
    )

    await send_email(email, f"Your UploadM8 {plan} trial starts now", html, from_addr=MAIL_FROM_SUPPORT, reply_to=SUPPORT_EMAIL)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Trial cancelled (before conversion)
# ─────────────────────────────────────────────────────────────────────────────
async def send_trial_cancelled_email(
    email: str,
    name: str,
    tier: str,
    trial_end_date: str,
) -> None:
    """
    Fires when a user cancels during their free trial period.
    Access continues until trial_end_date; then drops to free.
    """
    if not mailgun_ready():
        return

    plan = tier_label(tier)

    html = email_shell(
        gradient=GRAD_RED,
        tagline="We're sorry to see you go",
        preheader_text=f"Your {plan} trial has been cancelled. You won't be charged — access continues until {trial_end_date}.",
        body_rows=(
            section_tag("Trial Cancelled", "#dc2626")
            + intro_row(
                f"Trial cancelled, {name}",
                f"You've cancelled your <strong style='color:#f87171;'>{plan}</strong> trial. "
                f"You'll keep full access until <strong style='color:#ffffff;'>{trial_end_date}</strong> — "
                "after that your account moves to Free automatically. "
                "<strong style='color:#ffffff;'>You will not be charged.</strong>",
            )
            + stat_grid(
                ("Status",   "Cancelled"),
                ("Access Until", trial_end_date),
                ("Charge",   "$0.00"),
            )
            + tinted_box(
                '<p style="margin:0 0 8px;color:#ffffff;font-size:15px;font-weight:700;">'
                'Want to share what went wrong?</p>'
                '<p style="margin:0;color:#9ca3af;font-size:14px;line-height:1.65;">'
                'Even a single sentence helps us build a better product. Reply to this email or '
                f'message us at <a href="mailto:{SUPPORT_EMAIL}" style="color:#f97316;text-decoration:none;">'
                f'{SUPPORT_EMAIL}</a> — we read every response personally.</p>',
                hex_color="#374151",
            )
            + cta_button("Reactivate My Subscription", URL_PRICING, pt="20px", pb="20px")
            + secondary_links(
                ("View Plans", URL_PRICING),
                ("Dashboard", URL_DASHBOARD),
            )
        ),
        footer_note="You received this because you cancelled your free trial.",
    )

    await send_email(email, f"Your UploadM8 {plan} trial has been cancelled", html, from_addr=MAIL_FROM_SUPPORT, reply_to=SUPPORT_EMAIL)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Subscription cancelled (paid plan)
# ─────────────────────────────────────────────────────────────────────────────
async def send_subscription_cancelled_email(
    email: str,
    name: str,
    tier: str,
    access_until: str,
) -> None:
    """
    Fires from customer.subscription.deleted for a paid (post-trial) subscriber.
    Access continues until end of the current billing period.
    """
    if not mailgun_ready():
        return

    plan = tier_label(tier)

    html = email_shell(
        gradient=GRAD_RED,
        tagline="We're sorry to see you go",
        preheader_text=f"Your {plan} subscription has been cancelled. Access continues until {access_until}.",
        body_rows=(
            section_tag("Subscription Cancelled", "#dc2626")
            + intro_row(
                f"Subscription cancelled, {name}",
                f"Your <strong style='color:#f87171;'>{plan}</strong> subscription has been cancelled. "
                f"You'll keep all your {plan} features until "
                f"<strong style='color:#ffffff;'>{access_until}</strong>, "
                "then your account moves to Free.",
            )
            + stat_grid(
                ("Plan",         plan),
                ("Status",       "Cancelled"),
                ("Access Until", access_until),
            )
            + tinted_box(
                '<p style="margin:0 0 8px;color:#ffffff;font-size:15px;font-weight:700;">'
                'What happened?</p>'
                '<p style="margin:0;color:#9ca3af;font-size:14px;line-height:1.65;">'
                'Your feedback matters to us — a lot. If something didn\'t work the way you '
                'expected, or if there\'s anything we could do better, please reply to this '
                f'email or reach out at '
                f'<a href="mailto:{SUPPORT_EMAIL}" style="color:#f97316;text-decoration:none;">'
                f'{SUPPORT_EMAIL}</a>. We read every reply.</p>',
                hex_color="#374151",
            )
            + cta_button("Reactivate Subscription", URL_BILLING, pt="20px", pb="20px")
            + secondary_links(
                ("View Plans",  URL_PRICING),
                ("Dashboard",   URL_DASHBOARD),
                ("Settings",    URL_SETTINGS),
            )
        ),
        footer_note="You received this because your UploadM8 subscription was cancelled.",
    )

    await send_email(email, f"Your UploadM8 {plan} subscription has been cancelled", html, from_addr=MAIL_FROM_SUPPORT)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Monthly renewal receipt
# ─────────────────────────────────────────────────────────────────────────────
async def send_renewal_receipt_email(
    email: str,
    name: str,
    tier: str,
    amount: float,
    invoice_id: str = "",
    billing_period: str = "",
    next_billing_date: str = "",
    payment_method: str = "",
) -> None:
    """
    Fires from invoice.paid for a recurring (non-first) subscription invoice.
    """
    if not mailgun_ready():
        return

    plan      = tier_label(tier)
    inv_label = f"INV-{invoice_id[:8].upper()}" if invoice_id else "—"
    period    = billing_period or "Current period"
    next_date = next_billing_date or "Next cycle"
    pm_label  = payment_method or "Card on file"

    receipt_lines = (
        '<table cellpadding="0" cellspacing="0" width="100%">'
        + receipt_row("Plan", plan)
        + receipt_row("Billing Period", period)
        + receipt_row("Payment Method", pm_label)
        + receipt_row("Invoice #", inv_label)
        + receipt_row("Amount Charged", f"${amount:.2f}", is_total=True)
        + "</table>"
    )

    html = email_shell(
        gradient=GRAD_GREEN,
        tagline="Thanks for sticking with us",
        preheader_text=f"Payment of ${amount:.2f} received for {plan}. Your next billing date is {next_date}.",
        body_rows=(
            section_tag("Payment Confirmed", "#16a34a")
            + intro_row(
                f"Payment received — thank you, {name}! &#128176;",
                f"Your <strong style='color:#22c55e;'>{plan}</strong> subscription has been renewed. "
                f"Your next billing date is <strong style='color:#ffffff;'>{next_date}</strong>.",
            )
            + tinted_box(receipt_lines, hex_color="#22c55e")
            + cta_button("Go to Dashboard", URL_DASHBOARD, pt="20px", pb="20px")
            + secondary_links(
                ("Billing &amp; Invoices", URL_BILLING),
                ("Settings",              URL_SETTINGS),
            )
        ),
        footer_note="You received this because a payment was processed for your UploadM8 subscription.",
    )

    await send_email(email, f"UploadM8 payment received — ${amount:.2f}", html, from_addr=MAIL_FROM_SUPPORT, reply_to=SUPPORT_EMAIL)
