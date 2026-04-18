"""Assemble GET /api/wallet JSON (marketing + daily_topup + ledger) — logic extracted from app.py."""
from __future__ import annotations

import calendar
import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

from services.wallet import get_wallet
from services.wallet_marketing import build_wallet_marketing_payload

logger = logging.getLogger(__name__)


async def load_wallet_page_data(
    conn,
    user: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    promo_defaults: Mapping[str, Any],
    admin_settings_cache: Mapping[str, Any],
) -> tuple[dict[str, Any], list[Any], dict[str, Any]]:
    """
    Returns (wallet dict, ledger rows, marketing dict). On partial failure uses safe fallbacks
    for ledger/marketing but still returns wallet row when possible.
    """
    uid = user["id"]
    fallback_wallet = {"put_balance": 0, "aic_balance": 0, "put_reserved": 0, "aic_reserved": 0}
    fallback_marketing = {
        "burn_put_pct": 0.0,
        "burn_aic_pct": 0.0,
        "put_capacity": int(plan.get("put_monthly", 30) or 30),
        "aic_capacity": int(plan.get("aic_monthly", 0) or 0),
        "ai_enabled": True,
        "banners": [],
        "links": {
            "topup": "/settings.html#billing",
            "topup_put": "/settings.html?topup=uploadm8_put_500#billing",
            "topup_aic": "/settings.html?topup=uploadm8_aic_1000#billing",
            "upgrade": "/settings.html#billing",
        },
        "period_start": None,
        "put_spent_period": 0,
        "aic_spent_period": 0,
        "put_available": 0,
        "aic_available": 0,
        "sales_opportunities": [],
        "experiments": {},
        "suppression": {},
    }

    try:
        wallet = await get_wallet(conn, uid)
        try:
            ledger = await conn.fetch(
                "SELECT * FROM token_ledger WHERE user_id = $1 ORDER BY created_at DESC LIMIT 50",
                uid,
            )
        except Exception as e:
            logger.warning("/api/wallet: token_ledger history unavailable: %s", e)
            ledger = []

        settings_row = None
        try:
            settings_row = await conn.fetchrow(
                """
                SELECT auto_generate_captions, auto_generate_hashtags, auto_generate_thumbnails
                FROM user_settings WHERE user_id = $1
                """,
                uid,
            )
        except Exception as e:
            logger.warning("/api/wallet: user_settings row unavailable: %s", e)

        try:
            marketing = await build_wallet_marketing_payload(
                conn,
                str(uid),
                user.get("subscription_tier"),
                wallet,
                plan,
                dict(settings_row) if settings_row else None,
                {**promo_defaults, **admin_settings_cache},
                bool(user.get("flex_enabled")),
            )
        except Exception as e:
            logger.error("/api/wallet marketing payload failed user=%s: %s", uid, e)
            marketing = fallback_marketing

        return wallet, list(ledger), marketing
    except Exception as e:
        logger.error("/api/wallet failed user=%s: %s", uid, e)
        return fallback_wallet, [], fallback_marketing


def build_wallet_api_json(
    wallet: Mapping[str, Any],
    ledger: list[Any],
    marketing: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    now_utc = datetime.now(timezone.utc)
    today_utc = now_utc.date()
    next_utc_midnight = datetime.combine(today_utc + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
    seconds_until_next = max(0, int((next_utc_midnight - now_utc).total_seconds()))
    _days_in_month = calendar.monthrange(today_utc.year, today_utc.month)[1]
    put_daily = max(0, int(math.ceil((int(plan.get("put_monthly", 0) or 0)) / max(1, _days_in_month))))
    aic_daily = max(0, int(math.ceil((int(plan.get("aic_monthly", 0) or 0)) / max(1, _days_in_month))))
    put_monthly = int(plan.get("put_monthly", 0) or 0)
    aic_monthly = int(plan.get("aic_monthly", 0) or 0)
    last_refill = (wallet or {}).get("last_refill_date")
    if isinstance(last_refill, str):
        try:
            last_refill = datetime.fromisoformat(last_refill).date()
        except Exception:
            last_refill = None
    refilled_today = bool(last_refill and last_refill >= today_utc)
    month_key = f"{today_utc.year}-{today_utc.month:02d}"
    wk = wallet or {}
    if (wk.get("subscription_drip_month") or "") != month_key:
        put_drip_m = 0
        aic_drip_m = 0
    else:
        put_drip_m = int(wk.get("put_drip_granted") or 0)
        aic_drip_m = int(wk.get("aic_drip_granted") or 0)
    put_sub_remaining = max(0, put_monthly - put_drip_m)
    aic_sub_remaining = max(0, aic_monthly - aic_drip_m)
    tier_slug = str(plan.get("tier", "free") or "free")
    is_free_tier = tier_slug == "free"
    can_refill_now = bool(
        is_free_tier
        and ((put_daily > 0 and put_sub_remaining > 0) or (aic_daily > 0 and aic_sub_remaining > 0))
        and not refilled_today
    )
    daily_topup = {
        "enabled": bool(is_free_tier and (put_daily > 0 or aic_daily > 0)),
        "token_type": "both",
        "amount": put_daily,
        "amount_put": put_daily,
        "amount_aic": aic_daily,
        "cap_monthly_put": put_monthly,
        "cap_monthly_aic": aic_monthly,
        "put_subscription_granted_month": put_drip_m,
        "aic_subscription_granted_month": aic_drip_m,
        "put_subscription_remaining_month": put_sub_remaining,
        "aic_subscription_remaining_month": aic_sub_remaining,
        "rollover_unlimited": True,
        "refill_policy": "free_daily" if is_free_tier else "paid_on_invoice",
        "last_refill_date": last_refill.isoformat() if hasattr(last_refill, "isoformat") else None,
        "refilled_today": refilled_today,
        "can_refill_now": can_refill_now,
        "next_refill_at": (now_utc if can_refill_now else next_utc_midnight).isoformat(),
        "seconds_until_refill": 0 if can_refill_now else seconds_until_next,
    }

    _links = dict(marketing.get("links", {}) or {})
    _links.setdefault("topup", "/settings.html#billing")
    _links.setdefault("topup_put", "/settings.html?topup=uploadm8_put_500#billing")
    _links.setdefault("topup_aic", "/settings.html?topup=uploadm8_aic_1000#billing")
    _links.setdefault("upgrade", "/settings.html#billing")

    return {
        "wallet": wallet,
        "plan_limits": {
            "put_daily": plan.get("put_daily", 1),
            "put_monthly": plan.get("put_monthly", 30),
            "aic_monthly": plan.get("aic_monthly", 0),
        },
        "ledger": [dict(l) for l in ledger],
        "burn_put_pct": marketing.get("burn_put_pct", 0.0),
        "burn_aic_pct": marketing.get("burn_aic_pct", 0.0),
        "put_capacity": marketing.get("put_capacity", int(plan.get("put_monthly", 30) or 30)),
        "aic_capacity": marketing.get("aic_capacity", int(plan.get("aic_monthly", 0) or 0)),
        "ai_enabled": bool(marketing.get("ai_enabled", True)),
        "banners": marketing.get("banners", []),
        "links": _links,
        "daily_topup": daily_topup,
        "period_start": marketing.get("period_start"),
        "put_spent_period": marketing.get("put_spent_period", 0),
        "aic_spent_period": marketing.get("aic_spent_period", 0),
        "put_available": marketing.get("put_available"),
        "aic_available": marketing.get("aic_available"),
        "sales_opportunities": marketing.get("sales_opportunities", []),
        "experiments": marketing.get("experiments", {}),
        "suppression": marketing.get("suppression", {}),
    }


async def wallet_endpoint_payload(
    pool,
    user: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    promo_defaults: Mapping[str, Any],
    admin_settings_cache: Mapping[str, Any],
) -> dict[str, Any]:
    async with pool.acquire() as conn:
        wallet, ledger, marketing = await load_wallet_page_data(
            conn, user, plan, promo_defaults=promo_defaults, admin_settings_cache=admin_settings_cache
        )
    return build_wallet_api_json(wallet, ledger, marketing, plan)
