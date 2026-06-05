"""ML/AI intelligence bundle for Account Groups and Connected Accounts pages."""

from __future__ import annotations

import json
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, FrozenSet, List, Optional, Set

from services.ai_insights_hub import fetch_user_platform_engagement
from services.content_insights import fetch_meta_setups_by_platform
from services.growth_intelligence import m8_engine_identity_payload
from services.platform_accounts import (
    _PLATFORM_TOKEN_SELECT,
    fetch_auth_errors_by_token,
    serialize_platform_account_flat,
)

_PLATFORM_LABELS = {
    "tiktok": "TikTok",
    "youtube": "YouTube",
    "instagram": "Instagram",
    "facebook": "Facebook",
}


async def _load_platform_metrics_cache(conn, user_id: str) -> Optional[dict]:
    try:
        row = await conn.fetchrow(
            "SELECT fetched_at, data FROM platform_metrics_cache WHERE user_id = $1",
            user_id,
        )
        if not row:
            return None
        data = row["data"]
        if isinstance(data, str):
            data = json.loads(data)
        if not isinstance(data, dict):
            return None
        out = dict(data)
        out["fetched_at"] = (
            row["fetched_at"].isoformat() if row["fetched_at"] else out.get("fetched_at")
        )
        return out
    except Exception:
        return None


async def _per_account_upload_counts(conn, user_id: str) -> Dict[str, int]:
    rows = await conn.fetch(
        """
        SELECT elem->>'token_row_id' AS token_id, COUNT(*)::int AS cnt
        FROM uploads u
        CROSS JOIN LATERAL jsonb_array_elements(
            CASE
                WHEN u.platform_results IS NULL THEN '[]'::jsonb
                WHEN jsonb_typeof(u.platform_results) = 'array' THEN u.platform_results
                ELSE '[]'::jsonb
            END
        ) AS elem
        WHERE u.user_id = $1
          AND elem->>'token_row_id' IS NOT NULL
          AND elem->>'token_row_id' <> ''
          AND COALESCE((elem->>'success')::boolean, false) = true
        GROUP BY 1
        """,
        user_id,
    )
    return {str(r["token_id"]): int(r["cnt"] or 0) for r in rows}


def _simplify_platform_metrics(cached: Optional[dict]) -> Dict[str, Any]:
    if not cached:
        return {}
    platforms = cached.get("platforms") or {}
    out: Dict[str, Any] = {}
    for plat, data in platforms.items():
        if not isinstance(data, dict):
            continue
        out[plat] = {
            "status": data.get("status", "not_connected"),
            "uploads": int(data.get("uploads") or 0),
            "views": int(data.get("views") or 0),
            "likes": int(data.get("likes") or 0),
            "followers": data.get("followers") or data.get("subscriber_count"),
            "accounts_live": int(data.get("accounts_live") or data.get("accounts_polled") or 0),
        }
    return out


def _per_account_metrics(cached: Optional[dict]) -> Dict[str, Dict[str, Any]]:
    if not cached:
        return {}
    platforms = cached.get("platforms") or {}
    out: Dict[str, Dict[str, Any]] = {}
    for plat, pdata in platforms.items():
        if not isinstance(pdata, dict):
            continue
        for acc in pdata.get("accounts") or []:
            if not isinstance(acc, dict):
                continue
            tid = str(acc.get("token_row_id") or "")
            if not tid:
                continue
            metrics = acc.get("metrics") or {}
            out[tid] = {
                "platform": plat,
                "status": metrics.get("status") or acc.get("status") or "unknown",
                "views": int(metrics.get("views") or 0),
                "likes": int(metrics.get("likes") or 0),
                "followers": metrics.get("followers") or metrics.get("subscriber_count"),
            }
    return out


def _meta_hints(meta_setup: Optional[dict], limit: int = 4) -> List[Dict[str, str]]:
    platforms = (meta_setup or {}).get("platforms") or {}
    hints: List[Dict[str, str]] = []
    for plat, meta in platforms.items():
        if not isinstance(meta, dict):
            continue
        tip = (
            meta.get("caption_style_hint")
            or meta.get("hook_pattern_hint")
            or meta.get("posting_cadence_hint")
            or meta.get("summary")
        )
        if tip:
            hints.append({"platform": str(plat), "hint": str(tip)[:240]})
    return hints[:limit]


def _parse_iso_datetime(value: Any) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        s = str(value).replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _account_health(accounts: List[dict]) -> Dict[str, Any]:
    total = len(accounts)
    needs_reconnect = [a for a in accounts if a.get("status") != "active"]
    now = datetime.now(timezone.utc)
    idle_30d = 0
    never_used = 0
    for acc in accounts:
        last_used = _parse_iso_datetime(acc.get("last_used_at"))
        if not last_used:
            never_used += 1
            continue
        if (now - last_used).days >= 30:
            idle_30d += 1

    score = 100
    if total:
        score -= len(needs_reconnect) * 25
        score -= idle_30d * 8
        score -= min(never_used, total) * 3
    else:
        score = 0

    return {
        "total": total,
        "active": max(0, total - len(needs_reconnect)),
        "needs_reconnect": len(needs_reconnect),
        "idle_30d": idle_30d,
        "never_used": never_used,
        "score": max(0, min(100, score)),
    }


def _group_coverage(accounts: List[dict], group_rows: List[Any]) -> Dict[str, Any]:
    account_ids = {str(a.get("id")) for a in accounts if a.get("id")}
    grouped_ids: Set[str] = set()
    group_count = 0
    for row in group_rows or []:
        ids = row.get("account_ids") if isinstance(row, dict) else row["account_ids"]
        if ids:
            group_count += 1
            grouped_ids.update(str(x) for x in ids)
    grouped_owned = grouped_ids.intersection(account_ids)
    ungrouped = sorted(account_ids.difference(grouped_owned))
    total = len(account_ids)
    return {
        "groups": group_count,
        "grouped_accounts": len(grouped_owned),
        "ungrouped_accounts": len(ungrouped),
        "ungrouped_account_ids": ungrouped,
        "coverage_pct": round(100.0 * len(grouped_owned) / max(total, 1), 1) if total else 0.0,
    }


def _next_actions(
    accounts: List[dict],
    health: Dict[str, Any],
    coverage: Dict[str, Any],
    cached_metrics: Optional[dict],
    suggestions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not accounts:
        return [
            {
                "id": "connect_first_account",
                "label": "Connect your first account",
                "detail": "Connect TikTok, YouTube, Instagram, or Facebook so M8 can learn where your content performs.",
                "href": "platforms.html",
                "priority": "high",
            }
        ]
    actions: List[Dict[str, Any]] = []
    if health.get("needs_reconnect", 0) > 0:
        actions.append(
            {
                "id": "reconnect_accounts",
                "label": "Reconnect accounts",
                "detail": f"{health['needs_reconnect']} account needs OAuth attention before scheduled posts can run cleanly.",
                "href": "platforms.html",
                "priority": "high",
            }
        )
    if coverage.get("groups", 0) == 0 and len(accounts) > 1:
        actions.append(
            {
                "id": "create_first_group",
                "label": "Create your first account group",
                "detail": "Group client, brand, or platform accounts to make bulk uploads one click.",
                "href": "groups.html",
                "priority": "high" if suggestions else "medium",
            }
        )
    elif coverage.get("ungrouped_accounts", 0) > 0:
        actions.append(
            {
                "id": "group_ungrouped_accounts",
                "label": "Group remaining accounts",
                "detail": f"{coverage['ungrouped_accounts']} connected account is not in a group yet.",
                "href": "groups.html",
                "priority": "medium",
            }
        )
    if not cached_metrics:
        actions.append(
            {
                "id": "refresh_metrics",
                "label": "Refresh analytics aggregates",
                "detail": "Pull fresh platform metrics so the ML panel can compare live account performance.",
                "href": "platforms.html",
                "priority": "medium",
            }
        )
    if not actions:
        actions.append(
            {
                "id": "review_smart_insights",
                "label": "Review Smart Insights",
                "detail": "Your connected accounts look healthy. Use Smart Insights to tune captions, timing, and channel mix.",
                "href": "smart-insights.html",
                "priority": "low",
            }
        )
    return actions[:3]


def _existing_member_sets(groups: List[Any]) -> Set[FrozenSet[str]]:
    out: Set[FrozenSet[str]] = set()
    for g in groups:
        ids = g.get("account_ids") if isinstance(g, dict) else g["account_ids"]
        if ids:
            out.add(frozenset(str(x) for x in ids))
    return out


def build_group_suggestions(
    accounts: List[dict],
    platform_engagement: List[dict],
    existing_member_sets: Set[FrozenSet[str]],
) -> List[Dict[str, Any]]:
    by_platform: Dict[str, List[str]] = defaultdict(list)
    for acc in accounts:
        plat = str(acc.get("platform") or "").lower()
        aid = str(acc.get("id") or "")
        if plat and aid:
            by_platform[plat].append(aid)

    top_plat = str((platform_engagement[0] or {}).get("platform") or "").lower() if platform_engagement else ""
    suggestions: List[Dict[str, Any]] = []

    for plat, ids in sorted(by_platform.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        if len(ids) < 2:
            continue
        member_set = frozenset(ids)
        if member_set in existing_member_sets:
            continue
        label = _PLATFORM_LABELS.get(plat, plat.title())
        reason = f"Bundle {len(ids)} {label} accounts for one-tap bulk uploads."
        if plat == top_plat and platform_engagement:
            pe = platform_engagement[0]
            er = pe.get("avg_engagement_rate_pct")
            if er:
                reason = (
                    f"{label} is your top-performing platform ({er}% avg engagement). "
                    f"Group these {len(ids)} accounts to post winners everywhere at once."
                )
        suggestions.append(
            {
                "name": f"All {label}",
                "account_ids": ids,
                "platform": plat,
                "reason": reason,
                "confidence": "high" if plat == top_plat else "medium",
            }
        )

    if top_plat and len(by_platform.get(top_plat, [])) >= 1:
        solo_ids = by_platform.get(top_plat, [])
        if len(solo_ids) == 1 and solo_ids[0] not in {x for s in existing_member_sets for x in s}:
            label = _PLATFORM_LABELS.get(top_plat, top_plat.title())
            suggestions.append(
                {
                    "name": f"{label} priority",
                    "account_ids": solo_ids,
                    "platform": top_plat,
                    "reason": f"Your ML engine ranks {label} highest for engagement — keep this account easy to target on upload.",
                    "confidence": "medium",
                }
            )

    seen_sets: Set[FrozenSet[str]] = set()
    deduped: List[Dict[str, Any]] = []
    for s in suggestions:
        key = frozenset(str(x) for x in s.get("account_ids") or [])
        if not key or key in seen_sets or key in existing_member_sets:
            continue
        seen_sets.add(key)
        deduped.append(s)
    return deduped[:4]


async def build_accounts_hub_insights(pool: Any, user_id: str) -> Dict[str, Any]:
    uid = uuid.UUID(str(user_id))
    async with pool.acquire() as conn:
        auth_errors = await fetch_auth_errors_by_token(conn, str(user_id))
        rows = await conn.fetch(_PLATFORM_TOKEN_SELECT, user_id)
        accounts = [
            serialize_platform_account_flat(r, r["platform"], auth_error_by_token=auth_errors)
            for r in rows
        ]
        platform_engagement = await fetch_user_platform_engagement(conn, uid, days=90, limit=6)
        meta_setup = await fetch_meta_setups_by_platform(conn, uid, lookback_days=120)
        group_rows = await conn.fetch(
            "SELECT account_ids FROM account_groups WHERE user_id = $1",
            user_id,
        )
        cached_metrics = await _load_platform_metrics_cache(conn, str(user_id))
        upload_counts = await _per_account_upload_counts(conn, str(user_id))

    existing_sets = _existing_member_sets([{"account_ids": g["account_ids"]} for g in group_rows])
    suggestions = build_group_suggestions(accounts, platform_engagement, existing_sets)
    coverage = _group_coverage(accounts, group_rows)
    ungrouped_ids = coverage.get("ungrouped_account_ids") or []
    if len(ungrouped_ids) >= 2:
        key = frozenset(str(x) for x in ungrouped_ids)
        if key not in existing_sets:
            suggestions.insert(
                0,
                {
                    "name": "Ungrouped accounts",
                    "account_ids": ungrouped_ids,
                    "platform": "mixed",
                    "reason": f"{len(ungrouped_ids)} connected accounts are not in a group yet. Bundle them so uploads are faster to target.",
                    "confidence": "medium",
                },
            )
            suggestions = suggestions[:4]
    health = _account_health(accounts)

    top_platform = None
    if platform_engagement:
        top_platform = {
            "platform": platform_engagement[0].get("platform"),
            "avg_engagement_rate_pct": platform_engagement[0].get("avg_engagement_rate_pct"),
            "uploads": platform_engagement[0].get("uploads"),
            "avg_views": platform_engagement[0].get("avg_views"),
        }

    return {
        "ok": True,
        "m8_engine": m8_engine_identity_payload(),
        "platform_engagement": platform_engagement,
        "top_platform": top_platform,
        "platform_metrics": _simplify_platform_metrics(cached_metrics),
        "per_account_metrics": _per_account_metrics(cached_metrics),
        "per_account_uploads": upload_counts,
        "meta_hints": _meta_hints(meta_setup),
        "group_suggestions": suggestions,
        "group_coverage": coverage,
        "account_health": health,
        "next_actions": _next_actions(accounts, health, coverage, cached_metrics, suggestions),
        "metrics_fetched_at": (cached_metrics or {}).get("fetched_at"),
        "connected_accounts": len(accounts),
    }
