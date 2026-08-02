"""
Compile and expose best posting times for users and admins.

Reuses smart-schedule / M8 hour learning — does not retrain. Surfaces:
  • Per-platform 24h local weights (user blend or fleet M8)
  • Pinpoint top hours
  • Best combinations: platform × local hour × weekday from upload outcomes
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence

from core.scheduling import (
    PLATFORM_HOT_WINDOWS,
    STATIC_PRIOR_RESEARCH_VERSION,
    _resolve_tz,
    hour_in_hot_windows,
    platform_hot_windows,
    static_hour_prior_24,
    utc_weights_as_local,
)
from services.smart_schedule_insights import (
    _MIN_USER_SAMPLES,
    build_hour_weights_for_platforms_batch,
    fetch_m8_hour_priors_batch,
)

logger = logging.getLogger("uploadm8.best_posting_times")

_PLATFORMS = ("tiktok", "youtube", "instagram", "facebook")
_DOW_LABELS = ("Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday")


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _hour_label(hour: int) -> str:
    h = int(hour) % 24
    suffix = "AM" if h < 12 else "PM"
    h12 = h % 12
    if h12 == 0:
        h12 = 12
    return f"{h12}:00 {suffix}"


def _serialize_hours(weights: Sequence[float], platform: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for h, w in enumerate(weights):
        out.append(
            {
                "hour": h,
                "weight": round(float(w), 6),
                "label": _hour_label(h),
                "in_hot_window": hour_in_hot_windows(h, platform),
            }
        )
    return out


def _top_hours(weights: Sequence[float], *, n: int = 5) -> List[Dict[str, Any]]:
    ranked = sorted(range(24), key=lambda h: float(weights[h]), reverse=True)
    return [
        {
            "hour": h,
            "weight": round(float(weights[h]), 6),
            "label": _hour_label(h),
        }
        for h in ranked[: max(1, n)]
        if float(weights[h]) > 0
    ]


async def _user_timezone(conn: Any, user_id: str) -> str:
    row = await conn.fetchrow("SELECT timezone FROM users WHERE id = $1", user_id)
    tz = (row.get("timezone") if row else None) or "America/Chicago"
    return str(tz).strip() or "America/Chicago"


async def _user_sample_counts(
    conn: Any,
    user_id: str,
    platforms: Sequence[str],
    *,
    lookback_days: int = 180,
) -> Dict[str, int]:
    plats = [str(p).strip().lower() for p in platforms if str(p).strip()]
    if not plats:
        return {}
    since = _now_utc() - timedelta(days=lookback_days)
    rows = await conn.fetch(
        """
        SELECT lower(trim(plat.raw::text)) AS platform, COUNT(*)::bigint AS n
          FROM uploads u
          CROSS JOIN LATERAL unnest(COALESCE(u.platforms, ARRAY[]::text[])) AS plat(raw)
         WHERE u.user_id = $1::uuid
           AND u.status IN ('completed', 'succeeded', 'partial')
           AND COALESCE(u.scheduled_time, u.completed_at, u.created_at) >= $2
           AND lower(trim(plat.raw::text)) = ANY($3::text[])
         GROUP BY 1
        """,
        user_id,
        since,
        plats,
    )
    return {str(r["platform"]): int(r["n"] or 0) for r in rows}


async def _combination_rows(
    conn: Any,
    *,
    user_id: Optional[str],
    user_timezone: str,
    lookback_days: int = 180,
    limit: int = 24,
) -> List[Dict[str, Any]]:
    """
    Rank platform × local-hour × weekday by log1p(views) engagement.

    Postgres DOW: 0=Sunday … 6=Saturday (matches display labels).
    """
    since = _now_utc() - timedelta(days=lookback_days)
    tz = (user_timezone or "UTC").strip() or "UTC"
    params: list[Any] = [tz, since, list(_PLATFORMS), limit]
    user_clause = ""
    if user_id:
        user_clause = "AND u.user_id = $5::uuid"
        params.append(user_id)

    sql = f"""
        SELECT
            platform,
            hr,
            dow,
            n,
            avg_views,
            score
        FROM (
            SELECT
                lower(trim(plat.raw::text)) AS platform,
                EXTRACT(
                    HOUR FROM timezone($1::text, COALESCE(u.scheduled_time, u.completed_at, u.created_at))
                )::int AS hr,
                EXTRACT(
                    DOW FROM timezone($1::text, COALESCE(u.scheduled_time, u.completed_at, u.created_at))
                )::int AS dow,
                COUNT(*)::bigint AS n,
                AVG(GREATEST(COALESCE(u.views, 0), 0))::double precision AS avg_views,
                SUM(LN(GREATEST(COALESCE(u.views, 0), 0) + 1.0))::double precision AS score
            FROM uploads u
            CROSS JOIN LATERAL unnest(COALESCE(u.platforms, ARRAY[]::text[])) AS plat(raw)
            WHERE u.status IN ('completed', 'succeeded', 'partial')
              AND COALESCE(u.scheduled_time, u.completed_at, u.created_at) >= $2
              AND lower(trim(plat.raw::text)) = ANY($3::text[])
              {user_clause}
            GROUP BY 1, 2, 3
        ) t
        WHERE hr BETWEEN 0 AND 23 AND dow BETWEEN 0 AND 6 AND n >= 2
        ORDER BY score DESC, avg_views DESC, n DESC
        LIMIT $4
    """
    try:
        rows = await conn.fetch(sql, *params)
    except Exception as e:
        logger.warning("best_posting_times combinations query failed: %s", e)
        return []

    out: List[Dict[str, Any]] = []
    for r in rows:
        dow = int(r["dow"] or 0)
        hr = int(r["hr"] or 0)
        plat = str(r["platform"] or "")
        out.append(
            {
                "platform": plat,
                "hour_local": hr,
                "hour_label": _hour_label(hr),
                "dow": dow,
                "dow_label": _DOW_LABELS[dow] if 0 <= dow <= 6 else str(dow),
                "posts": int(r["n"] or 0),
                "avg_views": round(float(r["avg_views"] or 0.0), 2),
                "score": round(float(r["score"] or 0.0), 4),
                "in_hot_window": hour_in_hot_windows(hr, plat),
                "combo_key": f"{plat}|{_DOW_LABELS[dow] if 0 <= dow <= 6 else dow}|{_hour_label(hr)}",
            }
        )
    return out


def _pinpoint_from_platforms(platforms_payload: Dict[str, Any]) -> Dict[str, Any]:
    best: Optional[Dict[str, Any]] = None
    per_platform: Dict[str, Any] = {}
    for plat, block in platforms_payload.items():
        tops = block.get("top_hours") or []
        if not tops:
            continue
        top = tops[0]
        entry = {
            "platform": plat,
            "hour_local": top["hour"],
            "label": f"{plat.title()} · {top['label']}",
            "weight": top["weight"],
        }
        per_platform[plat] = entry
        if best is None or float(top["weight"]) > float(best["weight"]):
            best = entry
    return {"best_overall": best, "by_platform": per_platform}


async def build_user_best_posting_times(
    pool: Any,
    user_id: str,
    *,
    platforms: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    plats = [str(p).strip().lower() for p in (platforms or _PLATFORMS) if str(p).strip()]
    plats = [p for p in plats if p in _PLATFORMS] or list(_PLATFORMS)

    async with pool.acquire() as conn:
        tz_name = await _user_timezone(conn, user_id)
        weights = await build_hour_weights_for_platforms_batch(
            conn, user_id, plats, user_timezone=tz_name
        )
        samples = await _user_sample_counts(conn, user_id, plats)
        m8 = await fetch_m8_hour_priors_batch(conn, plats)
        combos = await _combination_rows(
            conn, user_id=user_id, user_timezone=tz_name, limit=20
        )

    platforms_out: Dict[str, Any] = {}
    for plat in plats:
        w = weights.get(plat) or static_hour_prior_24(plat)
        n = int(samples.get(plat, 0))
        platforms_out[plat] = {
            "hours": _serialize_hours(w, plat),
            "top_hours": _top_hours(w, n=5),
            "sample_count": n,
            "sources": {
                "static_windows": True,
                "user_history": n >= _MIN_USER_SAMPLES,
                "m8_fleet_priors": plat in m8,
                "research_version": STATIC_PRIOR_RESEARCH_VERSION,
            },
            "hot_windows": platform_hot_windows(plat),
        }

    pinpoint = _pinpoint_from_platforms(platforms_out)
    if combos:
        pinpoint["best_combination"] = combos[0]

    return {
        "ok": True,
        "scope": "user",
        "user_id": str(user_id),
        "timezone": tz_name,
        "as_of": _now_utc().isoformat(),
        "platforms": platforms_out,
        "best_combinations": combos,
        "pinpoint": pinpoint,
        "notes": [
            "Hours are local to your account timezone.",
            "Weights blend research hot windows, fleet/M8 priors, and your upload outcomes.",
            "Combinations rank weekday × hour × platform by log-views on your completed posts.",
        ],
    }


async def build_admin_publish_hour_insights(
    pool: Any,
    *,
    reference_timezone: str = "America/Chicago",
) -> Dict[str, Any]:
    """Fleet M8 priors + app-wide best combinations for admin ML observability."""
    tz_name = (reference_timezone or "America/Chicago").strip() or "America/Chicago"
    tz = _resolve_tz(tz_name)
    now = _now_utc()

    async with pool.acquire() as conn:
        m8 = await fetch_m8_hour_priors_batch(conn, list(_PLATFORMS))
        meta = await conn.fetchrow(
            """
            SELECT trained_at, model_version, train_row_count, val_mae_log1p_views, training_run_id
              FROM m8_publish_hour_priors
             ORDER BY trained_at DESC
             LIMIT 1
            """
        )
        raw_rows = await conn.fetch(
            """
            SELECT lower(platform) AS platform, hour_utc, prior_weight, trained_at, model_version, train_row_count
              FROM m8_publish_hour_priors
             ORDER BY platform, hour_utc
            """
        )
        combos = await _combination_rows(
            conn, user_id=None, user_timezone=tz_name, lookback_days=180, limit=40
        )
        active_users = await conn.fetchval(
            """
            SELECT COUNT(DISTINCT user_id)::bigint
              FROM uploads
             WHERE status IN ('completed', 'succeeded', 'partial')
               AND COALESCE(scheduled_time, completed_at, created_at) >= $1
            """,
            now - timedelta(days=180),
        )

    by_plat_utc: Dict[str, List[float]] = {p: [0.0] * 24 for p in _PLATFORMS}
    for r in raw_rows:
        plat = str(r["platform"] or "")
        hr = int(r["hour_utc"] or 0)
        if plat in by_plat_utc and 0 <= hr <= 23:
            by_plat_utc[plat][hr] = float(r["prior_weight"] or 0.0)

    platforms_out: Dict[str, Any] = {}
    for plat in _PLATFORMS:
        utc_w = m8.get(plat) or by_plat_utc.get(plat) or static_hour_prior_24(plat)
        # Present as local for the admin reference timezone (pinpointable clocks).
        if plat in m8 or any(by_plat_utc.get(plat) or []):
            local_w = utc_weights_as_local(utc_w, tz, now)
            space = "m8_utc_remapped_local"
        else:
            local_w = static_hour_prior_24(plat)
            space = "static_hot_windows"
        platforms_out[plat] = {
            "hours": _serialize_hours(local_w, plat),
            "top_hours": _top_hours(local_w, n=5),
            "utc_hours": _serialize_hours(utc_w, plat),
            "sources": {
                "m8_table": plat in m8 or any(float(x) > 0 for x in (by_plat_utc.get(plat) or [])),
                "weight_space": space,
                "research_version": STATIC_PRIOR_RESEARCH_VERSION,
            },
            "hot_windows": platform_hot_windows(plat),
        }

    pinpoint = _pinpoint_from_platforms(platforms_out)
    if combos:
        pinpoint["best_combination"] = combos[0]

    return {
        "ok": True,
        "scope": "fleet",
        "timezone": tz_name,
        "as_of": now.isoformat(),
        "active_publishers_180d": int(active_users or 0),
        "m8_meta": (
            {
                "trained_at": meta["trained_at"].isoformat() if meta and meta["trained_at"] else None,
                "model_version": meta["model_version"] if meta else None,
                "train_row_count": int(meta["train_row_count"] or 0) if meta else 0,
                "val_mae_log1p_views": (
                    float(meta["val_mae_log1p_views"])
                    if meta and meta["val_mae_log1p_views"] is not None
                    else None
                ),
                "training_run_id": str(meta["training_run_id"]) if meta and meta["training_run_id"] else None,
            }
            if meta
            else None
        ),
        "hot_windows_catalog": PLATFORM_HOT_WINDOWS,
        "platforms": platforms_out,
        "best_combinations": combos,
        "pinpoint": pinpoint,
        "notes": [
            "Fleet priors come from m8_publish_hour_priors (PCI published_at model) when trained.",
            "Local hours use the admin reference timezone for readable pinpoints.",
            "Combinations are app-wide upload outcomes (platform × weekday × hour).",
        ],
    }


def best_posting_times_fallback(*, scope: str = "user", error: str = "unavailable") -> Dict[str, Any]:
    return {
        "ok": False,
        "scope": scope,
        "error": error,
        "platforms": {},
        "best_combinations": [],
        "pinpoint": {},
        "as_of": _now_utc().isoformat(),
    }
