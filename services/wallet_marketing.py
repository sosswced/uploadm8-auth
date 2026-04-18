"""
Server-side wallet marketing: burn %, capacities, and banner list for GET /api/wallet.
Rules align with product spec (PUT/AIC spend vs period capacity, AI upsell, flex, promos).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
from typing import Any, Dict, List, Optional, Tuple

from stages.entitlements import TIER_CONFIG, get_next_public_upgrade_tier, normalize_tier
from services.growth_intelligence import fetch_user_engagement_snapshot


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
    month_start = await conn.fetchval(
        "SELECT date_trunc('month', now() AT TIME ZONE 'utc') AT TIME ZONE 'utc'"
    )
    if month_start and month_start.tzinfo is None:
        month_start = month_start.replace(tzinfo=timezone.utc)
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
            """
            SELECT platform, COUNT(*)::int AS c FROM platform_tokens
            WHERE user_id = $1::uuid AND revoked_at IS NULL
            GROUP BY platform
            """,
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
            "SELECT COUNT(*)::int FROM platform_tokens WHERE user_id = $1::uuid AND revoked_at IS NULL",
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
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    opps.append({
        "type": type_,
        "severity": severity,
        "title": title,
        "body": body,
        "cta_label": cta_label,
        "cta_link": cta_link,
        "metadata": metadata or {},
    })


def _range_to_minutes(range_key: Optional[str]) -> int:
    r = str(range_key or "30d").strip().lower()
    table = {
        "24h": 24 * 60,
        "7d": 7 * 24 * 60,
        "30d": 30 * 24 * 60,
        "90d": 90 * 24 * 60,
        "6m": 180 * 24 * 60,
        "1y": 365 * 24 * 60,
    }
    return int(table.get(r, 30 * 24 * 60))


async def _user_campaign_features(conn, user_id: str, range_key: str) -> Dict[str, Any]:
    minutes = max(60, _range_to_minutes(range_key))
    since = _now_utc().replace(microsecond=0) - timedelta(minutes=minutes)
    row = await conn.fetchrow(
        """
        WITH up AS (
          SELECT user_id, COUNT(*)::int AS uploads_window
          FROM uploads
          WHERE user_id = $1::uuid AND created_at >= $2
          GROUP BY user_id
        ),
        rv AS (
          SELECT user_id, COALESCE(SUM(amount), 0)::decimal AS revenue_7d
          FROM revenue_tracking
          WHERE user_id = $1::uuid AND created_at >= NOW() - INTERVAL '7 days'
          GROUP BY user_id
        ),
        me AS (
          SELECT user_id,
                 COALESCE(SUM(CASE WHEN event_type='shown' THEN 1 ELSE 0 END),0)::int AS shown,
                 COALESCE(SUM(CASE WHEN event_type='clicked' THEN 1 ELSE 0 END),0)::int AS clicked
          FROM marketing_events
          WHERE user_id = $1::uuid AND created_at >= $2
          GROUP BY user_id
        ),
        pf AS (
          SELECT user_id, COUNT(*)::int AS connected_accounts
          FROM platform_tokens
          WHERE user_id = $1::uuid AND revoked_at IS NULL
          GROUP BY user_id
        )
        SELECT
          COALESCE(up.uploads_window, 0)::int AS uploads_window,
          COALESCE(rv.revenue_7d, 0)::decimal AS revenue_7d,
          COALESCE(me.shown, 0)::int AS shown,
          COALESCE(me.clicked, 0)::int AS clicked,
          COALESCE(pf.connected_accounts, 0)::int AS connected_accounts
        FROM (SELECT 1) seed
        LEFT JOIN up ON TRUE
        LEFT JOIN rv ON TRUE
        LEFT JOIN me ON TRUE
        LEFT JOIN pf ON TRUE
        """,
        user_id,
        since,
    )
    uploads = _i((row or {}).get("uploads_window"))
    shown = _i((row or {}).get("shown"))
    clicked = _i((row or {}).get("clicked"))
    connected = _i((row or {}).get("connected_accounts"))
    ctr = (float(clicked) / float(max(shown, 1))) * 100.0
    score = 0.0
    score += min(40.0, uploads * 2.2)
    score += min(25.0, connected * 4.5)
    score += min(20.0, ctr * 0.6)
    return {
        "uploads_window": uploads,
        "revenue_7d": float((row or {}).get("revenue_7d") or 0),
        "shown": shown,
        "clicked": clicked,
        "nudge_ctr_pct": ctr,
        "connected_accounts": connected,
        "enterprise_fit_score": max(0.0, min(score, 100.0)),
    }


async def _live_campaign_for_user(conn, user_id: str, tier: str) -> Optional[Dict[str, Any]]:
    rows = await conn.fetch(
        """
        SELECT id, name, objective, channel, status, range_key, targeting, schedule_at, created_at
        FROM marketing_campaigns
        WHERE status IN ('active', 'scheduled')
          AND channel IN ('in_app', 'mixed', 'discount')
          AND (schedule_at IS NULL OR schedule_at <= NOW())
        ORDER BY
          CASE WHEN status = 'active' THEN 0 ELSE 1 END,
          COALESCE(schedule_at, created_at) DESC
        LIMIT 60
        """
    )
    if not rows:
        return None
    tier_norm = normalize_tier(tier or "free")
    features_cache: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        c = dict(r)
        targeting = c.get("targeting") or {}
        if not isinstance(targeting, dict):
            continue
        tiers = [normalize_tier(t) for t in (targeting.get("tiers") or []) if normalize_tier(t)]
        if tiers and tier_norm not in tiers:
            continue
        range_key = str(c.get("range_key") or "30d")
        if range_key not in features_cache:
            features_cache[range_key] = await _user_campaign_features(conn, user_id, range_key)
        feats = features_cache[range_key]
        if feats["uploads_window"] < _i(targeting.get("min_uploads_30d")):
            continue
        if float(feats["nudge_ctr_pct"]) < float(targeting.get("min_nudge_ctr_pct") or 0):
            continue
        if float(feats["enterprise_fit_score"]) < float(targeting.get("min_enterprise_fit_score") or 0):
            continue
        if bool(targeting.get("require_no_revenue_7d")) and float(feats["revenue_7d"]) > 0:
            continue
        c["targeting"] = targeting
        c["audience_eval"] = feats
        return c
    return None


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
        "dashboard": "/dashboard.html",
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
    if ref_on and tier in ("free", "creator_lite", "creator_pro"):
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

    max_ac_plan = _i(plan.get("max_accounts"))
    per_pf_cap = _i(plan.get("max_accounts_per_platform"))
    spare_pf_slots = await _platform_spare_slots(conn, user_id, per_pf_cap) if per_pf_cap > 0 else 0
    if (
        tier in ("creator_pro", "studio", "agency", "friends_family", "lifetime")
        and max_ac_plan > 0
        and n_conn < max_ac_plan
    ):
        headroom = max_ac_plan - n_conn
        high_tier = tier in ("studio", "agency", "friends_family", "lifetime")
        if headroom >= 2 or (high_tier and headroom >= 1 and n_conn >= 2):
            sev = "promo" if tier in ("studio", "agency") and headroom >= 3 else "info"
            spare_hint = (
                f" You also have about {spare_pf_slots} spare per-platform connection slot(s) — link another account where you still have room."
                if spare_pf_slots > 0
                else ""
            )
            _append_opportunity(
                opps,
                type_="high_tier_platform_headroom",
                severity=sev,
                title="Use your plan's extra channel slots",
                body=(
                    f"Your plan supports up to {max_ac_plan} connected channels ({n_conn} linked; only active logins count). "
                    "Add another YouTube, TikTok, Instagram, or Facebook login to widen reach without upgrading."
                )
                + spare_hint,
                cta_label="Connect platforms",
                cta_link=links["platforms"],
                metadata={
                    "max_accounts": max_ac_plan,
                    "max_accounts_per_platform": per_pf_cap,
                    "connected": n_conn,
                    "headroom": headroom,
                    "per_platform_spare_slots": spare_pf_slots,
                },
            )

    try:
        from services.marketing_touchpoint_runner import pending_in_app_as_opportunities

        opps.extend(await pending_in_app_as_opportunities(conn, user_id, links))
    except Exception:
        pass

    # Runtime activation: when a campaign is active/scheduled and this user matches
    # targeting filters, surface it as an in-app nudge opportunity.
    live_campaign = await _live_campaign_for_user(conn, user_id, tier)
    if live_campaign:
        ch = str(live_campaign.get("channel") or "in_app")
        cta_link = links["dashboard"]
        cta_label = "Open campaign"
        if ch == "discount":
            cta_link = links["upgrade"]
            cta_label = "Claim offer"
        elif ch == "mixed":
            cta_link = links["upgrade"]
            cta_label = "View campaign"
        _append_opportunity(
            opps,
            type_=f"campaign_{live_campaign.get('id')}",
            severity="promo",
            title=str(live_campaign.get("name") or "Recommended offer"),
            body=str(live_campaign.get("objective") or "A campaign matched your recent activity and plan usage."),
            cta_label=cta_label,
            cta_link=cta_link,
            metadata={
                "campaign_id": str(live_campaign.get("id") or ""),
                "campaign_channel": ch,
                "campaign_status": str(live_campaign.get("status") or ""),
                "campaign_range": str(live_campaign.get("range_key") or "30d"),
                "audience_eval": live_campaign.get("audience_eval") or {},
            },
        )

    order = {"blocking": 0, "urgent": 1, "warning": 2, "info": 3, "promo": 4}

    def _sort_key(b: Dict[str, Any]) -> Tuple[int, str]:
        return (order.get(b.get("severity"), 9), b.get("type", ""))

    opps.sort(key=_sort_key)
    recently_converted = await _recent_revenue_conversion(conn, user_id, within_days=7)
    if recently_converted:
        # Suppress low-intent/promotional nudges right after a purchase/upgrade.
        opps = [o for o in opps if o.get("severity") in ("blocking", "urgent", "warning")]
    engagement_snapshot = await fetch_user_engagement_snapshot(conn, user_id)
    sales_opportunities = [
        {
            **b,
            "signals": {
                "burn_put_pct": round(burn_put, 4),
                "burn_aic_pct": round(burn_aic, 4),
                "tier": tier,
                "next_tier": next_slug,
                "engagement_rate_pct_30d": engagement_snapshot.get("engagement_rate_pct"),
                "avg_views_30d": engagement_snapshot.get("avg_views"),
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
        "engagement_snapshot": engagement_snapshot,
    }
