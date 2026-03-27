"""
Server-side wallet marketing: burn %, capacities, and banner list for GET /api/wallet.
Rules align with product spec (PUT/AIC spend vs period capacity, AI upsell, flex, promos).
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from typing import Any, Dict, List, Optional, Tuple

from stages.entitlements import TIER_CONFIG, get_next_public_upgrade_tier, normalize_tier


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


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


async def _platform_connection_count(conn, user_id: str) -> int:
    try:
        return _i(await conn.fetchval(
            "SELECT COUNT(*)::int FROM platform_tokens WHERE user_id = $1::uuid",
            user_id,
        ))
    except Exception:
        return 0


async def _recent_revenue_conversion(conn, user_id: str, within_days: int = 7) -> bool:
    try:
        c = await conn.fetchval(
            """
            SELECT COUNT(*)::int
            FROM revenue_tracking
            WHERE user_id = $1::uuid
              AND created_at >= NOW() - ($2::text || ' days')::interval
              AND source IN ('topup', 'subscription')
            """,
            user_id,
            str(int(max(1, within_days))),
        )
        return _i(c) > 0
    except Exception:
        return False


def _stable_ab_variant(user_id: str, key: str) -> str:
    h = hashlib.sha256(f"{user_id}|{key}".encode("utf-8")).hexdigest()
    return "B" if (int(h[:8], 16) % 2) else "A"


def _internal_tier(tier: str) -> bool:
    return tier in ("master_admin", "friends_family", "lifetime")


def _append_opportunity(
    opps: List[Dict[str, Any]],
    *,
    type_: str,
    severity: str,
    title: str,
    body: str,
    cta_label: str,
    cta_link: str,
) -> None:
    opps.append({
        "type": type_,
        "severity": severity,
        "title": title,
        "body": body,
        "cta_label": cta_label,
        "cta_link": cta_link,
    })


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
    Returns burn %, capacities, ai_enabled, banners[], sales_opportunities[], links.
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
    lookahead_h = _i(plan.get("lookahead_hours"))

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

    links = {
        "topup": "/settings.html#billing",
        "upgrade": "/settings.html#billing",
        "pricing": "/index.html#pricing",
        "platforms": "/platforms.html",
        "analytics": "/analytics.html",
    }
    experiments = {
        "cta_variant": _stable_ab_variant(user_id, "cta_text"),
        "urgency_variant": _stable_ab_variant(user_id, "urgency_language"),
        "ordering_variant": _stable_ab_variant(user_id, "ordering_topup_upgrade"),
    }

    if internal:
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
            "sales_opportunities": [],
            "links": links,
            "experiments": experiments,
        }

    burn_put = float(sums["put_spent"]) / float(put_capacity)
    burn_aic = float(sums["aic_spent"]) / float(aic_capacity)

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

    opps: List[Dict[str, Any]] = []

    # ── PUT capacity pressure ─────────────────────────────────────────────
    if put_available <= 0:
        _append_opportunity(
            opps,
            type_="put_blocking",
            severity="blocking",
            title="No PUT credits available",
            body="Publishing is paused until you add PUT tokens or your plan renews.",
            cta_label="Top up PUT",
            cta_link=links["topup"],
        )
    elif burn_put >= 0.90:
        _append_opportunity(
            opps,
            type_="put_urgent",
            severity="urgent",
            title="PUT usage is critically high",
            body=f"You've used about {int(round(burn_put * 100))}% of this period's PUT capacity. Top up or upgrade to avoid interruptions.",
            cta_label="Add credits",
            cta_link=links["topup"],
        )
    elif burn_put >= 0.70:
        _append_opportunity(
            opps,
            type_="put_warning",
            severity="warning",
            title="Running low on PUT allowance",
            body=f"About {int(round(burn_put * 100))}% of this period's PUT credits are used. Consider topping up before you hit the limit.",
            cta_label="Top up",
            cta_link=links["topup"],
        )

    # ── AIC pressure (when plan includes AIC) ─────────────────────────────
    if aic_month > 0:
        if aic_available <= 0:
            _append_opportunity(
                opps,
                type_="aic_blocking",
                severity="urgent",
                title="No AIC credits left",
                body="AI captions, hashtags, and thumbnails need AIC. Add a pack or upgrade for a higher monthly AI allowance.",
                cta_label="Buy AIC",
                cta_link=links["topup"],
            )
        elif burn_aic >= 0.90:
            _append_opportunity(
                opps,
                type_="aic_urgent",
                severity="urgent",
                title="AIC usage is critically high",
                body=f"About {int(round(burn_aic * 100))}% of this period's AI credits are used. Top up AIC to avoid degraded AI output.",
                cta_label="Add AIC",
                cta_link=links["topup"],
            )
        elif burn_aic >= 0.70:
            _append_opportunity(
                opps,
                type_="aic_warning",
                severity="warning",
                title="Running low on AI credits",
                body=f"Roughly {int(round(burn_aic * 100))}% of this period's AIC is used. Stock up before long posting sessions.",
                cta_label="Top up AIC",
                cta_link=links["topup"],
            )

    burst_on = bool(admin_settings.get("promo_burst_week_enabled"))
    if burst_on and tier not in ("free",) and burn_put >= 0.70:
        _append_opportunity(
            opps,
            type_="promo_burst",
            severity="promo",
            title="Burst Week bonus",
            body="High usage this period — check billing for limited-time bonus credit offers.",
            cta_label="View billing",
            cta_link=links["upgrade"],
        )

    if not ai_enabled or aic_low:
        _append_opportunity(
            opps,
            type_="ai_upsell",
            severity="info",
            title="Enable AI captions & thumbnails to save time",
            body="Turn on AI generation in Settings, or add AIC packs if you're running low on AI credits.",
            cta_label="Buy AIC / upgrade",
            cta_link=links["topup"],
        )

    spare = await _platform_spare_slots(conn, user_id, per_pf)
    if spare > 0 and put_available > 0:
        flex_hint = (
            " Use Flex to move PUT between connected accounts."
            if (flex_enabled or plan.get("can_flex"))
            else ""
        )
        _append_opportunity(
            opps,
            type_="flex_opportunity",
            severity="info",
            title="Unused platform allowance",
            body="You have spare connection capacity on one or more platforms — connect more accounts to use your full plan."
            + flex_hint,
            cta_label="Manage platforms",
            cta_link=links["platforms"],
        )

    ref_on = bool(admin_settings.get("promo_referral_enabled"))
    if ref_on and tier in ("free", "launch", "creator_lite", "creator_pro"):
        _append_opportunity(
            opps,
            type_="referral",
            severity="promo",
            title="Refer creators, earn rewards",
            body="Share UploadM8 with other creators when referrals are active.",
            cta_label="Open settings",
            cta_link=links["upgrade"],
        )

    next_slug = get_next_public_upgrade_tier(tier)
    if next_slug:
        ncfg = TIER_CONFIG.get(next_slug, {})
        ccfg = TIER_CONFIG.get(tier, TIER_CONFIG["free"])
        put_delta = _i(ncfg.get("put_monthly")) - _i(ccfg.get("put_monthly"))
        aic_delta = _i(ncfg.get("aic_monthly")) - _i(ccfg.get("aic_monthly"))
        q_delta = _i(ncfg.get("queue_depth")) - _i(ccfg.get("queue_depth"))
        lh_delta = _i(ncfg.get("lookahead_hours")) - _i(ccfg.get("lookahead_hours"))
        active_usage = burn_put >= 0.35 or burn_aic >= 0.35 or sums["put_spent"] >= 15
        if active_usage:
            bits = []
            if put_delta > 0:
                bits.append(f"+{put_delta:,} PUT/mo")
            if aic_delta > 0:
                bits.append(f"+{aic_delta:,} AIC/mo")
            if q_delta > 0:
                bits.append(f"queue {_i(ccfg.get('queue_depth')):,} → {_i(ncfg.get('queue_depth')):,}")
            if lh_delta > 0:
                bits.append(f"schedule up to {_i(ncfg.get('lookahead_hours'))}h ahead")
            detail = "; ".join(bits) if bits else "More capacity and features for growing channels."
            _append_opportunity(
                opps,
                type_="tier_upgrade",
                severity="promo",
                title=f"Upgrade to {ncfg.get('name', next_slug)}",
                body=f"You're using UploadM8 heavily this period. {detail}",
                cta_label=f"View {ncfg.get('name', 'plan')}",
                cta_link=links["upgrade"],
            )

    if plan.get("analytics") == "basic":
        _append_opportunity(
            opps,
            type_="analytics_upgrade",
            severity="info",
            title="Unlock deeper analytics",
            body="Paid plans include richer performance insights — see what content drives views and engagement.",
            cta_label="Compare plans",
            cta_link=links["pricing"],
        )

    if tier in ("free", "creator_lite") and lookahead_h <= 24 and burn_put >= 0.12:
        _append_opportunity(
            opps,
            type_="scheduler_upgrade",
            severity="info",
            title="Schedule further ahead",
            body=f"Your plan schedules up to {lookahead_h}h out. Higher tiers add longer lookahead and larger queues for batch workflows.",
            cta_label="See scheduling limits",
            cta_link=links["pricing"],
        )

    n_conn = await _platform_connection_count(conn, user_id)
    plan_flex = bool(plan.get("can_flex"))
    if n_conn >= 3 and not flex_enabled and not plan_flex and tier in ("creator_lite", "creator_pro", "studio"):
        _append_opportunity(
            opps,
            type_="flex_addon",
            severity="info",
            title="Multiple accounts connected",
            body="Flex lets you move PUT between connected accounts so no platform sits idle.",
            cta_label="Billing & add-ons",
            cta_link=links["upgrade"],
        )

    if tier == "creator_pro" and burn_put >= 0.50:
        _append_opportunity(
            opps,
            type_="team_upgrade",
            severity="promo",
            title="Growing a team?",
            body="Studio and Agency add more seats, exports, and collaboration headroom for multi-creator workflows.",
            cta_label="Explore Studio / Agency",
            cta_link=links["pricing"],
        )

    if tier == "studio" and burn_put >= 0.5:
        _append_opportunity(
            opps,
            type_="agency_upgrade",
            severity="promo",
            title="Agency-ready workflows",
            body="White-label, maximum queue depth, and Flex are built for agencies managing many client accounts.",
            cta_label="View Agency",
            cta_link=links["pricing"],
        )

    order = {"blocking": 0, "urgent": 1, "warning": 2, "info": 3, "promo": 4}

    def _sort_key(b: Dict[str, Any]) -> Tuple[int, str]:
        return (order.get(b.get("severity"), 9), b.get("type", ""))

    opps.sort(key=_sort_key)
    recently_converted = await _recent_revenue_conversion(conn, user_id, within_days=7)
    if recently_converted:
        # Suppress low-intent/promotional nudges right after a purchase/upgrade.
        opps = [o for o in opps if o.get("severity") in ("blocking", "urgent", "warning")]
    sales_opportunities = [
        {
            **b,
            "signals": {
                "burn_put_pct": round(burn_put, 4),
                "burn_aic_pct": round(burn_aic, 4),
                "tier": tier,
                "next_tier": next_slug,
            },
        }
        for b in opps
    ]

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
        "banners": list(opps),
        "sales_opportunities": sales_opportunities,
        "links": links,
        "experiments": experiments,
        "suppression": {
            "recently_converted_7d": bool(recently_converted),
            "suppressed_low_intent_nudges": bool(recently_converted),
        },
    }
