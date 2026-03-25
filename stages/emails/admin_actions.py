"""
UploadM8 — Phase 4b: Admin Action Emails  (v2 — Enhanced Design)
=================================================================
  send_admin_wallet_topup_email   → Master admin credits PUT/AIC tokens to a user
  send_admin_tier_switch_email    → Admin manually changes a user's subscription tier

Both fire TO THE USER being acted upon.

v2 upgrades:
  - Both emails have preheader_text
  - Wallet top-up: section_tag "Complimentary Tokens", metric_hero for token count
  - Tier switch: section_tag "Plan Updated" or "Plan Upgraded", plan_change_visual
"""

import logging
from .base import (
    send_email, mailgun_ready, tier_label,
    email_shell, intro_row, body_row, cta_button, tinted_box,
    check_list, stat_grid, plan_change_visual, secondary_links, spacer,
    section_tag, metric_hero, divider_accent,
    GRAD_PURPLE, GRAD_GOLD, GRAD_GREEN,
    URL_DASHBOARD, URL_BILLING, SUPPORT_EMAIL,
)

logger = logging.getLogger("uploadm8-worker")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Admin manually credits wallet tokens (PUT or AIC)
# ─────────────────────────────────────────────────────────────────────────────
async def send_admin_wallet_topup_email(
    email: str,
    name: str,
    wallet_type: str,       # "put" | "aic"
    tokens_added: int,
    new_balance: int = 0,
    reason: str = "",
    admin_note: str = "",
) -> None:
    """
    Sent to the user when a master admin calls credit_wallet() on their behalf.
    This covers the PUT and AIC grant flows from /api/admin/users/{id}/wallet.
    No charge occurs — this is a complimentary grant.
    """
    if not mailgun_ready():
        return

    wt_label = "Platform Upload Tokens (PUT)" if wallet_type == "put" else "AI Credit Tokens (AIC)"
    wt_short = "PUT" if wallet_type == "put" else "AIC"
    note_html = (
        f'<p style="margin:12px 0 0;color:#9ca3af;font-size:13px;line-height:1.6;">'
        f'<em>Note from UploadM8: {admin_note}</em></p>'
        if admin_note else ""
    )

    html = email_shell(
        gradient=GRAD_PURPLE,
        tagline="A little something extra from the UploadM8 team",
        preheader_text=f"You received +{tokens_added:,} complimentary {wt_short} tokens from the UploadM8 team.",
        body_rows=(
            section_tag("Complimentary Tokens &#127873;", "#7c3aed")
            + intro_row(
                f"You've received free tokens, {name}! &#127873;",
                f"The UploadM8 team has credited your account with "
                f"<strong style='color:#a78bfa;'>+{tokens_added:,} {wt_short}</strong> tokens — "
                "completely on us. They're in your wallet and ready to use right now.",
            )
            + metric_hero(
                f"+{tokens_added:,}",
                f"Complimentary {wt_short} Tokens",
                f"new balance: {new_balance:,} {wt_short}" if new_balance else "available in your wallet now",
                "#7c3aed",
            )
            + tinted_box(
                f'<p style="margin:0;color:#d1d5db;font-size:14px;line-height:1.7;">'
                f'{reason or "These tokens were granted as a complimentary addition to your account."}'
                f'{note_html}</p>',
                hex_color="#7c3aed",
            )
            + cta_button("Use My Tokens Now", URL_DASHBOARD, pt="20px", pb="20px")
            + secondary_links(
                ("Dashboard",      URL_DASHBOARD),
                ("Token Balance",  URL_BILLING),
            )
        ),
        footer_note="You received this because an admin credited tokens to your account.",
    )

    await send_email(
        email,
        f"🎁 You received +{tokens_added:,} {wt_short} tokens — compliments of UploadM8",
        html,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Admin manually switches a user's tier / entitlements
# ─────────────────────────────────────────────────────────────────────────────
async def send_admin_tier_switch_email(
    email: str,
    name: str,
    old_tier: str,
    new_tier: str,
    admin_note: str = "",
    is_upgrade: bool = True,
) -> None:
    """
    Sent to the user when an admin directly sets their subscription_tier in the DB
    (e.g. granting friends_family, agency, master_admin, or correcting a billing issue).

    This is separate from the Stripe webhook flow — this fires from the admin panel action.
    """
    if not mailgun_ready():
        return

    new_plan  = tier_label(new_tier)
    old_plan  = tier_label(old_tier)
    gradient  = GRAD_GREEN if is_upgrade else GRAD_GOLD
    hex_new   = "#22c55e"  if is_upgrade else "#d97706"

    action_word = "upgraded" if is_upgrade else "updated"
    emoji       = "&#128640;" if is_upgrade else "&#128295;"
    tag_text    = f"Plan {action_word.title()} {emoji}"
    tag_color   = "#22c55e" if is_upgrade else "#d97706"

    note_html = (
        f'<p style="margin:12px 0 0;color:#9ca3af;font-size:13px;line-height:1.6;">'
        f'<em>Note from UploadM8: {admin_note}</em></p>'
        if admin_note else ""
    )

    html = email_shell(
        gradient=gradient,
        tagline="A change has been made to your account",
        preheader_text=f"Your UploadM8 plan has been {action_word} from {old_plan} to {new_plan} — effective immediately.",
        body_rows=(
            section_tag(tag_text, tag_color)
            + intro_row(
                f"Your plan has been {action_word}, {name}! {emoji}",
                f"The UploadM8 team has manually moved your account from "
                f"<strong style='color:#9ca3af;'>{old_plan}</strong> to "
                f"<strong style='color:{hex_new};'>{new_plan}</strong>. "
                "This change is effective immediately.",
            )
            + plan_change_visual(old_tier, new_tier, hex_new=hex_new)
            + tinted_box(
                f'<p style="margin:0;color:#d1d5db;font-size:14px;line-height:1.75;">'
                f'Your new entitlements, upload quotas, and features are now active. '
                f'Head to your dashboard to see what\'s unlocked.'
                f'{note_html}</p>',
                hex_color=hex_new,
            )
            + cta_button("See What's Unlocked", URL_DASHBOARD, pt="20px", pb="20px")
            + secondary_links(
                ("Dashboard",  URL_DASHBOARD),
                ("Billing",    URL_BILLING),
            )
            + tinted_box(
                f'<p style="margin:0;color:#9ca3af;font-size:13px;line-height:1.65;">'
                f'Have questions about this change? Reply to this email or reach out at '
                f'<a href="mailto:{SUPPORT_EMAIL}" style="color:#f97316;text-decoration:none;">'
                f'{SUPPORT_EMAIL}</a>.</p>',
                hex_color="#374151",
                pb="36px",
            )
        ),
        footer_note="You received this because an admin updated your subscription tier.",
    )

    await send_email(
        email,
        f"{emoji} Your UploadM8 plan has been {action_word} to {new_plan}",
        html,
    )
