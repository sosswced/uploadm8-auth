"""
UploadM8 — Phase 3: Billing Change Emails  (v2 — Enhanced Design)
==================================================================
  send_plan_upgraded_email    → customer.subscription.updated (tier went up)
  send_plan_downgraded_email  → customer.subscription.updated (tier went down)
  send_topup_receipt_email    → checkout.session.completed (mode=payment / token top-up)

v2 upgrades:
  - All emails have preheader_text
  - Plan upgraded: section_tag "Upgrade Complete", metric_hero for new plan name
  - Plan downgraded: section_tag "Plan Updated", cleaner plan_change_visual
  - Top-up receipt: section_tag "Tokens Added", metric_hero for token count
"""

import logging
from .base import (
    send_email, mailgun_ready, tier_label, MAIL_FROM_SUPPORT,
    email_shell, intro_row, body_row, cta_button, tinted_box,
    check_list, stat_grid, secondary_links, plan_change_visual,
    receipt_row, spacer,
    section_tag, metric_hero, divider_accent,
    GRAD_GREEN, GRAD_ORANGE, GRAD_PURPLE,
    URL_DASHBOARD, URL_BILLING, URL_PRICING, URL_SETTINGS,
    SUPPORT_EMAIL,
)

logger = logging.getLogger("uploadm8-worker")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Plan upgraded
# ─────────────────────────────────────────────────────────────────────────────
async def send_plan_upgraded_email(
    email: str,
    name: str,
    old_tier: str,
    new_tier: str,
    new_amount: float,
    next_billing_date: str = "",
) -> None:
    """
    Fires when customer.subscription.updated shows the new tier is higher
    than the previous one.

    Tier order: free < launch < creator_pro < studio < agency
    """
    if not mailgun_ready():
        return

    new_plan  = tier_label(new_tier)
    old_plan  = tier_label(old_tier)
    next_date = next_billing_date or "your next billing date"

    html = email_shell(
        gradient=GRAD_GREEN,
        tagline="More power, more platforms, more reach",
        preheader_text=f"You're now on {new_plan}! Upgraded from {old_plan} — effective immediately.",
        body_rows=(
            section_tag("Upgrade Complete &#128640;", "#16a34a")
            + intro_row(
                f"You've upgraded to {new_plan}! &#128640;",
                f"Excellent move, {name}. Your account has been upgraded from "
                f"<strong style='color:#9ca3af;'>{old_plan}</strong> to "
                f"<strong style='color:#22c55e;'>{new_plan}</strong> — effective immediately.",
            )
            + plan_change_visual(old_tier, new_tier, hex_new="#22c55e")
            + stat_grid(
                ("New Plan",      new_plan),
                ("New Rate",      f"${new_amount:.2f}/mo"),
                ("Next Billing",  next_date),
            )
            + check_list(
                f"All {new_plan} features unlocked right now",
                "Increased upload quota applied immediately",
                "Additional platform slots available",
                "Priority processing queue active",
                hex_color="#22c55e",
            )
            + cta_button("Explore Your New Plan", URL_DASHBOARD, pt="20px", pb="20px")
            + secondary_links(
                ("Manage Billing", URL_BILLING),
                ("Settings",       URL_SETTINGS),
            )
        ),
        footer_note="You received this because your UploadM8 subscription was upgraded.",
    )

    await send_email(email, f"You're now on {new_plan} — welcome to the upgrade! 🚀", html, from_addr=MAIL_FROM_SUPPORT, reply_to=SUPPORT_EMAIL)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Plan downgraded
# ─────────────────────────────────────────────────────────────────────────────
async def send_plan_downgraded_email(
    email: str,
    name: str,
    old_tier: str,
    new_tier: str,
    new_amount: float,
    effective_date: str = "",
) -> None:
    """
    Fires when customer.subscription.updated shows the new tier is lower.
    Downgrade typically takes effect at end of current billing period in Stripe.
    """
    if not mailgun_ready():
        return

    new_plan  = tier_label(new_tier)
    old_plan  = tier_label(old_tier)
    eff_date  = effective_date or "the end of your current billing period"

    html = email_shell(
        gradient=GRAD_ORANGE,
        tagline="Your plan has been updated",
        preheader_text=f"Your plan has been updated from {old_plan} to {new_plan}, effective {eff_date}.",
        body_rows=(
            section_tag("Plan Updated", "#f97316")
            + intro_row(
                f"Plan changed to {new_plan}, {name}",
                f"Your subscription has been moved from "
                f"<strong style='color:#f97316;'>{old_plan}</strong> to "
                f"<strong style='color:#f97316;'>{new_plan}</strong>. "
                f"Your {old_plan} features remain active until "
                f"<strong style='color:#ffffff;'>{eff_date}</strong>.",
            )
            + plan_change_visual(old_tier, new_tier, hex_new="#f97316")
            + stat_grid(
                ("New Plan",      new_plan),
                ("New Rate",      f"${new_amount:.2f}/mo"),
                ("Effective",     eff_date),
            )
            + tinted_box(
                '<p style="margin:0 0 8px;color:#ffffff;font-size:15px;font-weight:700;">'
                'Thinking of upgrading again?</p>'
                '<p style="margin:0;color:#9ca3af;font-size:14px;line-height:1.65;">'
                'You can switch back any time — upgrades take effect immediately. '
                f'<a href="{URL_PRICING}" style="color:#f97316;text-decoration:none;">See all plans &#8594;</a></p>',
                hex_color="#374151",
            )
            + cta_button("Go to Dashboard", URL_DASHBOARD, pt="20px", pb="20px")
            + secondary_links(
                ("Billing",      URL_BILLING),
                ("View Plans",   URL_PRICING),
                ("Settings",     URL_SETTINGS),
            )
        ),
        footer_note="You received this because your UploadM8 plan was changed.",
    )

    await send_email(email, f"Your UploadM8 plan has been updated to {new_plan}", html, from_addr=MAIL_FROM_SUPPORT, reply_to=SUPPORT_EMAIL)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Token top-up receipt  (PUT or AIC)
# ─────────────────────────────────────────────────────────────────────────────
async def send_topup_receipt_email(
    email: str,
    name: str,
    wallet_type: str,         # "put" | "aic"
    tokens_added: int,
    amount: float,
    new_balance: int = 0,
    stripe_payment_id: str = "",
) -> None:
    """
    Fires from checkout.session.completed when mode=payment (token top-up).
    wallet_type is "put" (Platform Upload Tokens) or "aic" (AI Credit tokens).
    Hooks: notify_topup() already fires Discord; this sends the user a receipt.
    """
    if not mailgun_ready():
        return

    wt_label  = "Platform Upload Tokens (PUT)" if wallet_type == "put" else "AI Credit Tokens (AIC)"
    wt_short  = "PUT"                           if wallet_type == "put" else "AIC"
    inv_label = f"PAY-{stripe_payment_id[:8].upper()}" if stripe_payment_id else "—"

    receipt_lines = (
        '<table cellpadding="0" cellspacing="0" width="100%">'
        + receipt_row("Token Type",      wt_label)
        + receipt_row("Tokens Added",    f"+{tokens_added:,} {wt_short}")
        + receipt_row("New Balance",     f"{new_balance:,} {wt_short}" if new_balance else "See dashboard")
        + receipt_row("Payment Ref",     inv_label)
        + receipt_row("Amount Charged",  f"${amount:.2f}", is_total=True)
        + "</table>"
    )

    html = email_shell(
        gradient=GRAD_PURPLE,
        tagline="Your token wallet has been topped up",
        preheader_text=f"+{tokens_added:,} {wt_short} tokens added to your wallet. Ready to use immediately.",
        body_rows=(
            section_tag(f"+ {tokens_added:,} {wt_short} Tokens Added", "#7c3aed")
            + intro_row(
                f"Top-up confirmed, {name}! &#128176;",
                f"<strong style='color:#a78bfa;'>+{tokens_added:,} {wt_short}</strong> tokens have been "
                "added to your wallet and are ready to use immediately.",
            )
            + metric_hero(
                f"+{tokens_added:,}",
                f"{wt_short} Tokens Added",
                "available in your wallet right now",
                "#7c3aed",
            )
            + tinted_box(receipt_lines, hex_color="#7c3aed")
            + cta_button("Start Uploading", URL_DASHBOARD, pt="20px", pb="20px")
            + secondary_links(
                ("Token Wallet &amp; Balance", URL_BILLING),
                ("Dashboard",                  URL_DASHBOARD),
            )
        ),
        footer_note="You received this because a token top-up payment was processed on your account.",
    )

    await send_email(email, f"UploadM8 top-up confirmed — +{tokens_added:,} {wt_short} tokens 💰", html, from_addr=MAIL_FROM_SUPPORT, reply_to=SUPPORT_EMAIL)
