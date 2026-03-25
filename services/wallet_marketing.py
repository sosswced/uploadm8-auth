"""
Server-side wallet marketing: burn %, capacities, and banner list for GET /api/wallet.
Rules align with product spec (PUT/AIC spend vs period capacity, AI upsell, flex, promos).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from stages.entitlements import normalize_tier


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _f(v: Any) -> float:
    try:
        return float(v or 0)
    except (TypeError, ValueError):
        return 0.0


def _i(v: Any) -> int:
    try:
        return int(v or 0)
    except (TypeError, ValueError):
        return 0


async def _period_anchor(conn, user_id: str) -> datetime:
    """Start of current usage period: last monthly refill or calendar month (UTC)."""
    try:
        row = await conn.fetchrow(
            """
            SELECT MAX(created_at) AS m
            FROM token_ledger
            WHERE user_id = $1::uuid AND reason = 'monthly_refill'
            """,
            user_id,
        )
        last_refill = row["m"] if row else None
    except Exception:
        last_refill = None
    month_start = await conn.fetchval("SELECT date_trunc('month', timezone('utc', now()))")
    if last_refill:
        return max(last_refill, month_start)
    return month_start


async def _ledger_sums(
    conn,
    user_id: str,
    period_start: datetime,
) -> Dict[str, int]:
    """Aggregate token_ledger rows since period_start."""
    q = """
    SELECT
      COALESCE(SUM(CASE WHEN token_type = 'put' AND reason = 'spend' AND delta < 0
                     THEN ABS(delta::bigint) ELSE 0 END), 0)::bigint AS put_spent,
      COALESCE(SUM(CASE WHEN token_type = 'aic' AND reason = 'spend' AND delta < 0
                     THEN ABS(delta::bigint) ELSE 0 END), 0)::bigint AS aic_spent,
      COALESCE(SUM(CASE WHEN token_type = 'put' AND reason IN ('topup', 'topup_purchase') AND delta > 0
                     THEN delta::bigint ELSE 0 END), 0)::bigint AS put_topup,
      COALESCE(SUM(CASE WHEN token_type = 'aic' AND reason IN ('topup', 'topup_purchase') AND delta > 0
                     THEN delta::bigint ELSE 0 END), 0)::bigint AS aic_topup,
      COALESCE(SUM(CASE WHEN token_type = 'put' AND reason = 'signup_bonus' AND delta > 0
                     THEN delta::bigint ELSE 0 END), 0)::bigint AS put_promo,
      COALESCE(SUM(CASE WHEN token_type = 'aic' AND reason = 'signup_bonus' AND delta > 0
                     THEN delta::bigint ELSE 0 END), 0)::bigint AS aic_promo,
      COALESCE(SUM(CASE WHEN token_type = 'put' AND reason LIKE 'admin_%' AND delta > 0
                     THEN delta::bigint ELSE 0 END), 0)::bigint AS put_admin,
      COALESCE(SUM(CASE WHEN token_type = 'aic' AND reason LIKE 'admin_%' AND delta > 0
                     THEN delta::bigint ELSE 0 END), 0)::bigint AS aic_admin
    FROM token_ledger
    WHERE user_id = $1::uuid AND created_at >= $2
    """
    row = await conn.fetchrow(q, user_id, period_start)
    if not row:
        return {k: 0 for k in (
            "put_spent", "aic_spent", "put_topup", "aic_topup",
            "put_promo", "aic_promo", "put_admin", "aic_admin",
        )}
    return {k: _i(row[k]) for k in row.keys()}


async def _platform_spare_slots(conn, user_id: str, per_platform_cap: int) -> int:
    """Rough count of unused per-platform connection slots (sum over platforms)."""
    if per_platform_cap <= 0:
        return 0
    try:
        rows = await conn.fetch(
            "SELECT platform, COUNT(*)::int AS c FROM platform_tokens WHERE user_id = $1::uuid GROUP BY platform",
            user_id,
        )
    except Exception:
        return 0
    spare = 0
    for r in rows:
        spare += max(0, per_platform_cap - _i(r["c"]))
    return spare


def _internal_tier(tier: str) -> bool:
    return tier in ("master_admin", "friends_family", "lifetime")


async def build_wallet_marketing_payload(
    conn,
    user_id: str,
    subscription_tier: Optional[str],
    wallet: Dict[str, Any],
    plan: Dict[str, Any],
    user_settings: Optional[Dict[str, Any]],
    admin_settings: Dict[str, Any],
    flex_enabled: bool = False,
) -> Dict[str, Any]:
    """
    Returns burn_put_pct, burn_aic_pct, put_capacity, aic_capacity, ai_enabled, banners[], links.
    """
    tier = normalize_tier(subscription_tier or "free")
    put_bal = _i(wallet.get("put_balance"))
    aic_bal = _i(wallet.get("aic_balance"))
    put_res = _i(wallet.get("put_reserved"))
    aic_res = _i(wallet.get("aic_reserved"))
    put_available = max(0, put_bal - put_res)
    aic_available = max(0, aic_bal - aic_res)

    put_month = _i(plan.get("put_monthly"))
    aic_month = _i(plan.get("aic_monthly"))
    per_pf = _i(plan.get("max_accounts_per_platform"))

    internal = _internal_tier(tier)
    period_start = await _period_anchor(conn, user_id)
    sums = await _ledger_sums(conn, user_id, period_start) if not internal else {
        "put_spent": 0,
        "aic_spent": 0,
        "put_topup": 0,
        "aic_topup": 0,
        "put_promo": 0,
        "aic_promo": 0,
        "put_admin": 0,
        "aic_admin": 0,
    }

    put_promo_total = sums["put_promo"] + sums["put_admin"]
    aic_promo_total = sums["aic_promo"] + sums["aic_admin"]

    put_capacity = max(1, put_month + sums["put_topup"] + put_promo_total)
    aic_capacity = max(1, aic_month + sums["aic_topup"] + aic_promo_total)

    if internal:
        burn_put = 0.0
        burn_aic = 0.0
        return {
            "burn_put_pct": 0.0,
            "burn_aic_pct": 0.0,
            "put_capacity": put_capacity,
            "aic_capacity": aic_capacity,
            "put_spent_period": 0,
            "aic_spent_period": 0,
            "put_available": put_available,
            "aic_available": aic_available,
            "ai_enabled": True,
            "period_start": period_start.isoformat() if hasattr(period_start, "isoformat") else None,
            "banners": [],
            "links": {
                "topup": "/settings.html#billing",
                "upgrade": "/settings.html#billing",
                "pricing": "/index.html#pricing",
            },
        }
    burn_put = float(sums["put_spent"]) / float(put_capacity)
    burn_aic = float(sums["aic_spent"]) / float(aic_capacity)

    # AI enabled: any of the three prefs true (defaults True)
    us = user_settings or {}
    cap_on = us.get("auto_generate_captions")
    hash_on = us.get("auto_generate_hashtags")
    thumb_on = us.get("auto_generate_thumbnails")
    if cap_on is None and hash_on is None and thumb_on is None:
        ai_enabled = True
    else:
        ai_enabled = bool(cap_on) or bool(hash_on) or bool(thumb_on)

    aic_low_threshold = max(5, int(aic_month * 0.10)) if aic_month > 0 else 5
    aic_low = aic_available < aic_low_threshold

    links = {
        "topup": "/settings.html#billing",
        "upgrade": "/settings.html#billing",
        "pricing": "/index.html#pricing",
    }

    banners: List[Dict[str, Any]] = []

    if not internal and put_available <= 0:
        banners.append({
            "type": "put_blocking",
            "severity": "blocking",
            "title": "No PUT credits available",
            "body": "Publishing is paused until you add PUT tokens or your plan renews.",
            "cta_label": "Top up PUT",
            "cta_link": links["topup"],
        })
    elif not internal and burn_put >= 0.90:
        banners.append({
            "type": "put_urgent",
            "severity": "urgent",
            "title": "PUT usage is critically high",
            "body": f"You've used about {int(round(burn_put * 100))}% of this period's PUT capacity. Top up or upgrade to avoid interruptions.",
            "cta_label": "Add credits",
            "cta_link": links["topup"],
        })
    elif not internal and burn_put >= 0.70:
        banners.append({
            "type": "put_warning",
            "severity": "warning",
            "title": "Running low on PUT allowance",
            "body": f"About {int(round(burn_put * 100))}% of this period's PUT credits are used. Consider topping up before you hit the limit.",
            "cta_label": "Top up",
            "cta_link": links["topup"],
        })

    # Burst week (paid + high burn + admin toggle)
    burst_on = bool(admin_settings.get("promo_burst_week_enabled"))
    if (
        burst_on
        and tier not in ("free",)
        and not internal
        and burn_put >= 0.70
    ):
        banners.append({
            "type": "promo_burst",
            "severity": "promo",
            "title": "Burst Week bonus",
            "body": "High usage this period — check billing for limited-time bonus credit offers.",
            "cta_label": "View billing",
            "cta_link": links["upgrade"],
        })

    # AI upsell: captions off OR AIC low
    if not internal and (not ai_enabled or aic_low):
        banners.append({
            "type": "ai_upsell",
            "severity": "info",
            "title": "Enable AI captions & thumbnails to save time",
            "body": "Turn on AI generation in Settings, or add AIC packs if you're running low on AI credits.",
            "cta_label": "Buy AIC / upgrade",
            "cta_link": links["topup"],
        })

    # Flex opportunity — spare per-platform connection slots
    spare = await _platform_spare_slots(conn, user_id, per_pf)
    if spare > 0 and put_available > 0:
        flex_hint = (
            " Use Flex to move PUT between connected accounts."
            if (flex_enabled or plan.get("can_flex"))
            else ""
        )
        banners.append({
            "type": "flex_opportunity",
            "severity": "info",
            "title": "Unused platform allowance",
            "body": "You have spare connection capacity on one or more platforms — connect more accounts to use your full plan."
            + flex_hint,
            "cta_label": "Manage platforms",
            "cta_link": "/platforms.html",
        })

    # Referral
    ref_on = bool(admin_settings.get("promo_referral_enabled"))
    if ref_on and tier in ("free", "launch", "creator_lite", "creator_pro") and not internal:
        banners.append({
            "type": "referral",
            "severity": "promo",
            "title": "Refer creators, earn rewards",
            "body": "Share UploadM8 with other creators when referrals are active.",
            "cta_label": "Open settings",
            "cta_link": links["upgrade"],
        })

    # Severity order for clients that show one slot
    order = {"blocking": 0, "urgent": 1, "warning": 2, "info": 3, "promo": 4}

    def _sort_key(b: Dict[str, Any]) -> Tuple[int, str]:
        return (order.get(b.get("severity"), 9), b.get("type", ""))

    banners.sort(key=_sort_key)

    return {
        "burn_put_pct": round(burn_put, 4),
        "burn_aic_pct": round(burn_aic, 4),
        "put_capacity": put_capacity,
        "aic_capacity": aic_capacity,
        "put_spent_period": sums["put_spent"],
        "aic_spent_period": sums["aic_spent"],
        "put_available": put_available,
        "aic_available": aic_available,
        "ai_enabled": ai_enabled,
        "period_start": period_start.isoformat() if hasattr(period_start, "isoformat") else None,
        "banners": banners,
        "links": links,
    }
