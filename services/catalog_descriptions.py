"""
services/catalog_descriptions.py

Canonical description templates for UploadM8 Stripe catalog.

ONE source of truth for product descriptions. Templates use Python str.format()
placeholders. The sync script calls render() to produce the final text that
gets pushed to Stripe.

Update wording here — never edit descriptions in Stripe dashboard or in
the PowerShell script. The sync run will overwrite any manual edits.
"""
from __future__ import annotations
from typing import Any, Dict


# =============================================================
# TOP-UP TEMPLATES
# =============================================================
# Pattern from Earl's canonical PUT 50 description.
# Placeholders: {amount}
PUT_TOPUP_TEMPLATE = (
    "One-time purchase of {amount:,} PUT (upload) tokens credited instantly to "
    "the purchaser's UploadM8 wallet. Each PUT token authorises one multi-platform "
    "video publish event across any combination of connected accounts on TikTok, "
    "YouTube Shorts, Instagram Reels, and Facebook Reels. Tokens are available "
    "for use immediately upon payment confirmation. Tokens never expire and "
    "carry over indefinitely. This is a non-refundable digital goods purchase "
    "— once tokens are credited to the wallet they cannot be reversed or "
    "refunded. A first-time purchaser bonus may be applied automatically. "
    "Service delivered digitally via uploadm8.com."
)

# Pattern from Earl's canonical AIC 100 description.
# Placeholders: {amount}
AIC_TOPUP_TEMPLATE = (
    "One-time purchase of {amount:,} AIC (AI credit) tokens credited instantly "
    "to the purchaser's UploadM8 wallet. AIC tokens power the UploadM8 AI "
    "feature suite: automated caption generation, hashtag strategy optimisation, "
    "thumbnail image analysis, smart scheduling suggestions, and content "
    "metadata enrichment for publications on TikTok, YouTube Shorts, "
    "Instagram Reels, and Facebook Reels. Tokens are available for use "
    "immediately upon payment confirmation. Tokens never expire and carry "
    "over indefinitely. This is a non-refundable digital goods purchase "
    "— once tokens are credited to the wallet they cannot be reversed or "
    "refunded. A first-time purchaser bonus may be applied automatically. "
    "Service delivered digitally via uploadm8.com."
)

# =============================================================
# SUBSCRIPTION TEMPLATE
# =============================================================
# Pattern from Earl's canonical Studio description. Boolean entitlements
# render as natural-language phrases via the helpers below.
#
# Placeholders:
#   {display_name}                  -> "Studio"
#   {max_accounts}                  -> 75
#   {put_monthly}                   -> 3,500
#   {aic_monthly}                   -> 1,000
#   {lookahead_hours}               -> 72
#   {queue_depth_phrase}            -> "job queue depth of 2,500" / "unlimited job queue"
#   {priority_phrase}               -> "turbo throughput processing" / "priority processing lane" / ""
#   {watermark_phrase}              -> "no watermark applied to published content" / "watermark on published content"
#   {analytics_phrase}              -> "analytics data export" / "standard analytics"
#   {white_label_phrase}            -> "white-label publishing options, " / ""
#   {team_phrase}                   -> "expanded team seats, " (3+) / ""
#   {webhooks_phrase}               -> "webhook delivery, " / ""
#   {ai_depth_phrase}               -> "maximum-depth AI content optimization"
SUBSCRIPTION_TEMPLATE = (
    "UploadM8 {display_name} is a recurring monthly SaaS subscription granting "
    "access to the UploadM8 multi-platform video publishing platform. Subscriber "
    "receives: {max_accounts} connected social media accounts per platform "
    "(TikTok, YouTube Shorts, Instagram Reels, Facebook Reels), "
    "{put_monthly:,} PUT upload tokens credited to wallet each billing cycle, "
    "{aic_monthly:,} AIC AI-credit tokens credited to wallet each billing cycle, "
    "{lookahead_hours}-hour scheduling lookahead, "
    "{queue_depth_phrase}, "
    "{priority_phrase}"
    "{watermark_phrase}, "
    "{analytics_phrase}, "
    "{white_label_phrase}"
    "{team_phrase}"
    "{webhooks_phrase}"
    "template library access, and {ai_depth_phrase}. "
    "Service is delivered digitally via uploadm8.com and access is granted "
    "immediately upon payment confirmation. Subscription renews automatically "
    "each month. A 7-day free trial is included for new subscribers — no "
    "charge during trial period. Cancellation takes effect at the end of "
    "the current paid billing period; no partial refunds are issued for "
    "unused days. Monthly token allocations do not roll over to the "
    "following billing cycle."
)


# =============================================================
# Phrase helpers — turn DB booleans/enums into natural language
# =============================================================

_AI_DEPTH_PHRASES = {
    "basic":    "AI content scanning",
    "enhanced": "enhanced AI content scanning",
    "advanced": "advanced AI content optimization",
    "max":      "maximum-depth AI content optimization",
}

_PRIORITY_PHRASES = {
    "p0": "dedicated processing lane, ",
    "p1": "turbo throughput processing, ",
    "p2": "priority processing lane, ",
    "p3": "",
    "p4": "",
}

_ANALYTICS_PHRASES = {
    "basic":       "basic analytics",
    "standard":    "standard analytics",
    "full_export": "analytics data export",
}


def _queue_depth_phrase(queue_depth: int) -> str:
    if queue_depth >= 99999:
        return "unlimited job queue"
    return f"job queue depth of {queue_depth:,}"


def _bool_phrase(flag: bool, true_text: str, false_text: str = "") -> str:
    return true_text if flag else false_text


# =============================================================
# Public API
# =============================================================

def render_topup_description(wallet: str, amount: int) -> str:
    """Render description for a PUT or AIC top-up product."""
    wallet = (wallet or "").lower()
    if wallet == "put":
        return PUT_TOPUP_TEMPLATE.format(amount=amount)
    if wallet == "aic":
        return AIC_TOPUP_TEMPLATE.format(amount=amount)
    raise ValueError(f"Unknown wallet kind: {wallet!r}")


def render_subscription_description(row: Dict[str, Any]) -> str:
    """Render subscription description from a catalog_products row dict.

    Required keys: display_name, max_accounts, put_monthly, aic_monthly,
    lookahead_hours, queue_depth, priority_class, watermark, white_label,
    team_seats, webhooks, analytics, ai_depth.
    """
    qd = int(row.get("queue_depth") or 0)
    pri = (row.get("priority_class") or "p3").lower()
    ad  = (row.get("ai_depth") or "basic").lower()
    an  = (row.get("analytics") or "basic").lower()
    seats = int(row.get("team_seats") or 1)

    ctx = {
        "display_name":      row["display_name"],
        "max_accounts":      int(row["max_accounts"]),
        "put_monthly":       int(row["put_monthly"]),
        "aic_monthly":       int(row["aic_monthly"]),
        "lookahead_hours":   int(row["lookahead_hours"]),
        "queue_depth_phrase": _queue_depth_phrase(qd),
        "priority_phrase":    _PRIORITY_PHRASES.get(pri, ""),
        "watermark_phrase":   _bool_phrase(
            not row.get("watermark"),
            "no watermark applied to published content",
            "watermark applied to published content",
        ),
        "analytics_phrase":   _ANALYTICS_PHRASES.get(an, "standard analytics"),
        "white_label_phrase": _bool_phrase(row.get("white_label"), "white-label publishing options, "),
        "team_phrase":        _bool_phrase(seats >= 3, "expanded team seats, "),
        "webhooks_phrase":    _bool_phrase(row.get("webhooks"), "webhook delivery, "),
        "ai_depth_phrase":    _AI_DEPTH_PHRASES.get(ad, "AI content optimization"),
    }
    return SUBSCRIPTION_TEMPLATE.format(**ctx)


def render_description(row: Dict[str, Any]) -> str:
    """Dispatch on product_kind. Returns final Stripe-ready description."""
    kind = row["product_kind"]
    if kind == "subscription":
        return render_subscription_description(row)
    if kind in ("topup_put", "topup_aic"):
        return render_topup_description(row["wallet"], row["token_amount"])
    raise ValueError(f"Unknown product_kind: {kind!r}")


if __name__ == "__main__":
    # Sanity preview when run directly: python -m services.catalog_descriptions
    sample_sub = {
        "product_kind": "subscription", "display_name": "Studio",
        "max_accounts": 75, "put_monthly": 3500, "aic_monthly": 1000,
        "lookahead_hours": 72, "queue_depth": 2500, "priority_class": "p1",
        "watermark": False, "white_label": True, "team_seats": 10,
        "hud": False, "webhooks": True, "analytics": "full_export",
        "ai_depth": "max",
    }
    print("=== SUB: Studio ===")
    print(render_description(sample_sub))
    print()
    print("=== TOPUP: PUT 250 ===")
    print(render_description({"product_kind": "topup_put", "wallet": "put", "token_amount": 250}))
    print()
    print("=== TOPUP: AIC 1000 ===")
    print(render_description({"product_kind": "topup_aic", "wallet": "aic", "token_amount": 1000}))
