"""
UploadM8 — Lifecycle Emails  (v2 — Enhanced Design)
=====================================================
  send_payment_failed_email         → Stripe invoice.payment_failed webhook
  send_trial_ending_reminder_email  → Scheduled job: fires 3 days before trial_end
  send_low_token_warning_email      → Worker: fires when PUT or AIC balance drops below threshold

v2 upgrades:
  - All emails have preheader_text
  - Payment failed: section_tag "Payment Failed", urgency-focused layout
  - Trial ending: section_tag + progress_bar for days remaining
  - Low token: section_tag + progress_bar for token balance + metric_hero for count
"""

import logging
from .base import (
    send_email, mailgun_ready, tier_label,
    email_shell, intro_row, body_row, cta_button, tinted_box,
    check_list, stat_grid, alert_banner, secondary_links, spacer,
    section_tag, metric_hero, progress_bar, divider_accent,
    GRAD_RED, GRAD_ORANGE, GRAD_PURPLE, GRAD_BLUE,
    URL_DASHBOARD, URL_BILLING, URL_PRICING, URL_SETTINGS,
    SUPPORT_EMAIL, FRONTEND_URL,
)

logger = logging.getLogger("uploadm8-worker")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Payment failed  — Stripe invoice.payment_failed
# ─────────────────────────────────────────────────────────────────────────────
async def send_payment_failed_email(
    email: str,
    name: str,
    tier: str,
    amount: float,
    next_retry_date: str = "",
    invoice_id: str = "",
    failure_reason: str = "",
) -> None:
    """
    Fires when Stripe sends invoice.payment_failed for a subscription renewal.

    Hook into the Stripe webhook handler in app.py. Add a new elif block
    inside the stripe_webhook() function:

        elif event_type == "invoice.payment_failed":
            inv = event["data"]["object"]
            sub_id = inv.get("subscription")
            if sub_id:
                user_row = await conn.fetchrow(
                    "SELECT email, name, subscription_tier FROM users WHERE stripe_subscription_id=$1", sub_id
                )
                if user_row:
                    retry_ts = inv.get("next_payment_attempt")
                    retry_date = datetime.fromtimestamp(retry_ts, tz=timezone.utc).strftime("%B %d, %Y") if retry_ts else ""
                    failure_reason = inv.get("last_finalization_error", {}).get("message", "")
                    background_tasks.add_task(
                        send_payment_failed_email,
                        user_row["email"],
                        user_row["name"] or "there",
                        user_row["subscription_tier"],
                        inv.get("amount_due", 0) / 100,
                        retry_date,
                        inv.get("id", ""),
                        failure_reason,
                    )
    """
    if not mailgun_ready():
        return

    plan = tier_label(tier)
    retry_line = (
        f"Stripe will automatically retry on <strong style='color:#ffffff;'>{next_retry_date}</strong>. "
        if next_retry_date else
        "Stripe will retry automatically. "
    )
    reason_html = (
        f'<p style="margin:10px 0 0;color:#f87171;font-size:13px;'
        f'font-family:\'Courier New\',Courier,monospace;line-height:1.5;">'
        f'Reason: {failure_reason}</p>'
        if failure_reason else ""
    )
    inv_label = f"INV-{invoice_id[:8].upper()}" if invoice_id else "—"

    html = email_shell(
        gradient=GRAD_RED,
        tagline="Action required — payment failed",
        preheader_text=f"Payment of ${amount:.2f} for {plan} failed. Update your card to keep your account active.",
        body_rows=(
            section_tag("Payment Failed", "#ef4444")
            + intro_row(
                f"Payment failed for {plan} &#9888;&#65039;",
                f"We were unable to charge the card on file for your "
                f"<strong style='color:#f87171;'>{plan}</strong> subscription. "
                f"{retry_line}"
                "Update your payment method to keep your account active.",
            )
            + tinted_box(
                f'<p style="margin:0 0 6px;color:#6b7280;font-size:10px;text-transform:uppercase;'
                f'letter-spacing:1.2px;font-weight:600;">Payment Details</p>'
                f'<p style="margin:0;color:#ffffff;font-size:18px;font-weight:700;">'
                f'${amount:.2f} — {plan}</p>'
                f'<p style="margin:6px 0 0;color:#6b7280;font-size:13px;">Invoice: {inv_label}</p>'
                + reason_html,
                hex_color="#ef4444",
            )
            + check_list(
                "Update your card to resume uninterrupted service",
                "Your content and connections are still safe",
                f"Access continues on {plan} during the retry window",
                hex_color="#f97316",
            )
            + cta_button("Update Payment Method", URL_BILLING, pt="8px", pb="20px")
            + tinted_box(
                f'<p style="margin:0;color:#9ca3af;font-size:13px;line-height:1.65;">'
                f'Need help? Reply to this email or contact us at '
                f'<a href="mailto:{SUPPORT_EMAIL}" style="color:#f97316;text-decoration:none;">'
                f'{SUPPORT_EMAIL}</a>. We\'ll sort it out together.</p>',
                hex_color="#374151",
                pb="36px",
            )
        ),
        footer_note="You received this because a payment failed on your UploadM8 subscription.",
    )

    await send_email(email, f"⚠️ Payment failed — update your card to keep {plan} active", html)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Trial ending reminder  — fires 3 days before trial_end
# ─────────────────────────────────────────────────────────────────────────────
async def send_trial_ending_reminder_email(
    email: str,
    name: str,
    tier: str,
    trial_end_date: str,
    days_left: int = 3,
    amount: float = 0.0,
) -> None:
    """
    Proactive reminder email sent N days before the trial expires.
    Implement a scheduled job (e.g., APScheduler or a Render cron job) that
    runs daily and queries:

        SELECT u.id, u.email, u.name, u.subscription_tier, u.trial_end
        FROM users u
        WHERE u.subscription_status = 'trialing'
          AND u.trial_end BETWEEN NOW() AND NOW() + INTERVAL '4 days'
          AND u.trial_reminder_sent IS NULL

    Then fire this email and set trial_reminder_sent = NOW() to prevent duplicates.

    You'll need to add trial_reminder_sent TIMESTAMPTZ column to users table:
        ALTER TABLE users ADD COLUMN IF NOT EXISTS trial_reminder_sent TIMESTAMPTZ;
    """
    if not mailgun_ready():
        return

    plan = tier_label(tier)
    days_word = f"{days_left} day" if days_left == 1 else f"{days_left} days"
    charge_line = (
        f"<strong style='color:#ffffff;'>${amount:.2f}/month</strong> "
        if amount else ""
    )

    # Progress bar: if trial_days total = 14, and days_left = 3, that's 21% remaining
    # We use days_left out of 14 as the pct
    trial_total_days = 14
    pct_remaining = max(0, min(100, int((days_left / trial_total_days) * 100)))

    html = email_shell(
        gradient=GRAD_ORANGE,
        tagline="Your free trial is ending soon",
        preheader_text=f"Your {plan} trial ends in {days_word} on {trial_end_date}. Here's what happens next.",
        body_rows=(
            section_tag("Trial Ending Soon", "#f97316")
            + intro_row(
                f"Your {plan} trial ends in {days_word} &#9203;",
                f"Hey {name} — heads up. Your free {plan} trial expires on "
                f"<strong style='color:#f97316;'>{trial_end_date}</strong>. "
                f"After that, you'll be charged {charge_line}automatically to keep your "
                f"{plan} access. Cancel before then if you don't want to continue — no charge.",
            )
            + stat_grid(
                ("Plan",          plan),
                ("Trial Ends",    trial_end_date),
                ("Days Left",     str(days_left)),
            )
            + progress_bar(
                pct_remaining,
                label=f"{days_left} of {trial_total_days} trial days remaining",
                hex_color="#f97316",
            )
            + check_list(
                f"Keep all {plan} features after trial",
                "AI caption generation stays active",
                "Multi-platform publishing continues uninterrupted",
                "Cancel any time — no lock-in",
                hex_color="#f97316",
            )
            + cta_button("Continue My Subscription", URL_BILLING, pt="8px", pb="20px")
            + tinted_box(
                f'<p style="margin:0 0 8px;color:#ffffff;font-size:15px;font-weight:700;">'
                f'Want to cancel?</p>'
                f'<p style="margin:0;color:#9ca3af;font-size:14px;line-height:1.65;">'
                f'No hard feelings. Cancel from <a href="{URL_BILLING}" '
                f'style="color:#f97316;text-decoration:none;">Billing Settings</a> '
                f'before <strong style="color:#ffffff;">{trial_end_date}</strong> '
                f'and you won\'t be charged a cent.</p>',
                hex_color="#374151",
                pb="36px",
            )
        ),
        footer_note=f"You received this reminder because your {plan} trial ends on {trial_end_date}.",
    )

    await send_email(email, f"⏳ Your {plan} trial ends in {days_word} — here's what happens next", html)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Low token balance warning
# ─────────────────────────────────────────────────────────────────────────────
async def send_low_token_warning_email(
    email: str,
    name: str,
    wallet_type: str,          # "put" | "aic"
    current_balance: int,
    threshold: int = 5,
) -> None:
    """
    Fires when a user's PUT or AIC balance drops to or below the threshold
    during token spend. Check inside spend_tokens() in app.py after deducting:

        wallets = await conn.fetchrow(
            "SELECT put_balance, aic_balance FROM wallets WHERE user_id=$1", user_id
        )
        prefs = await conn.fetchrow(
            "SELECT email_notifications FROM user_preferences WHERE user_id=$1", user_id
        )
        LOW_TOKEN_THRESHOLD = 5
        if prefs and prefs.get("email_notifications", True):
            if wallets["put_balance"] <= LOW_TOKEN_THRESHOLD:
                background_tasks.add_task(
                    send_low_token_warning_email,
                    user["email"], user["name"] or "there",
                    "put", wallets["put_balance"], LOW_TOKEN_THRESHOLD
                )
    """
    if not mailgun_ready():
        return

    wt_label = "Platform Upload Tokens (PUT)" if wallet_type == "put" else "AI Credit Tokens (AIC)"
    wt_short  = "PUT" if wallet_type == "put" else "AIC"
    is_critical = current_balance <= 2
    balance_color = "#ef4444" if is_critical else "#f97316"
    urgency = "Critically Low" if is_critical else "Running Low"
    tag_color = "#ef4444" if is_critical else "#7c3aed"

    # Progress bar: show how close to zero. threshold=5, so pct = current_balance / threshold * 100
    pct_remaining = max(0, min(100, int((current_balance / max(threshold, 1)) * 100)))

    html = email_shell(
        gradient=GRAD_PURPLE,
        tagline="Your token balance needs attention",
        preheader_text=f"You only have {current_balance} {wt_short} tokens left. Top up now to keep your uploads running.",
        body_rows=(
            section_tag(f"Token Balance {urgency}", tag_color)
            + intro_row(
                f"Your {wt_short} balance is {urgency.lower()} &#129689;",
                f"Hey {name} — you only have "
                f"<strong style='color:{balance_color};'>{current_balance} {wt_short}</strong> "
                f"tokens remaining. Top up now to keep your uploads running without interruption.",
            )
            + metric_hero(
                str(current_balance),
                f"{wt_short} Tokens Remaining",
                "top up now to avoid interruptions",
                balance_color,
            )
            + progress_bar(
                pct_remaining,
                label=f"{current_balance} of {threshold} token threshold",
                hex_color=balance_color,
            )
            + cta_button("Top Up Tokens Now", URL_BILLING, pt="8px", pb="20px")
            + tinted_box(
                f'<p style="margin:0 0 8px;color:#ffffff;font-size:15px;font-weight:700;">'
                f'On a subscription plan?</p>'
                f'<p style="margin:0;color:#9ca3af;font-size:14px;line-height:1.65;">'
                f'Upgrade your plan for a larger monthly token allocation — or add a one-time '
                f'top-up to your existing plan. '
                f'<a href="{URL_PRICING}" style="color:#a78bfa;text-decoration:none;">'
                f'See all options &#8594;</a></p>',
                hex_color="#374151",
                pb="36px",
            )
        ),
        footer_note=f"You received this because your {wt_label} balance is low.",
    )

    await send_email(
        email,
        f"🪙 Low token balance — {current_balance} {wt_short} remaining",
        html,
    )
