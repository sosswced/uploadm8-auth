"""
Data-driven smart schedule priors (M8).

Batch SQL paths reduce presign from ~4 queries × N platforms to a handful of queries total.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence

from core.scheduling import calculate_smart_schedule, static_hour_prior_24

logger = logging.getLogger("uploadm8.smart_schedule")

_LOOKBACK_GLOBAL_DAYS = 120
_LOOKBACK_USER_DAYS = 180
_RECENT_DAYS = 14
_BASELINE_START_DAYS = 60
_BASELINE_END_DAYS = 14
_MIN_USER_SAMPLES = 5

_M8_PRIORS_MAX_AGE_HOURS = float(os.environ.get("M8_PRIORS_MAX_AGE_HOURS", str(7 * 24)))


def _normalize(vec: List[float]) -> List[float]:
    s = sum(max(0.0, x) for x in vec)
    if s <= 0:
        return [1.0 / 24.0] * 24
    return [max(0.0, x) / s for x in vec]


def _vec24_from_rows(rows: Sequence[Any]) -> List[float]:
    v = [0.0] * 24
    for r in rows:
        h = int(r["hr"])
        if 0 <= h <= 23:
            v[h] = float(r["score"] or 0.0)
    return v


def _momentum_multipliers(recent: List[float], baseline: List[float]) -> List[float]:
    out: List[float] = []
    for r, b in zip(recent, baseline):
        ratio = (r + 1e-4) / (b + 1e-4)
        m = ratio**0.38
        out.append(max(0.8, min(1.35, m)))
    return out


def _blend_vectors(
    static_w: List[float],
    global_w: List[float],
    user_w: List[float],
    user_sample_count: int,
    momentum: List[float],
) -> List[float]:
    if user_sample_count >= _MIN_USER_SAMPLES:
        blended = [0.22 * s + 0.48 * g + 0.30 * u for s, g, u in zip(static_w, global_w, user_w)]
    else:
        blended = [0.30 * s + 0.70 * g for s, g in zip(static_w, global_w)]
    tuned = [b * m for b, m in zip(blended, momentum)]
    return _normalize(tuned)


async def fetch_m8_hour_priors(conn: Any, platform: str) -> Optional[List[float]]:
    rows = await conn.fetch(
        """
        SELECT hour_utc, prior_weight, trained_at
        FROM m8_publish_hour_priors
        WHERE lower(platform) = lower($1)
        ORDER BY hour_utc
        """,
        platform.strip(),
    )
    if len(rows) != 24:
        return None
    trained = rows[0]["trained_at"]
    if trained is None:
        return None
    age_h = (datetime.now(timezone.utc) - trained).total_seconds() / 3600.0
    if age_h > _M8_PRIORS_MAX_AGE_HOURS:
        return None
    w = [float(r["prior_weight"] or 0.0) for r in rows]
    return _normalize(w)


async def fetch_m8_hour_priors_batch(
    conn: Any,
    platforms: Sequence[str],
) -> Dict[str, List[float]]:
    plats = sorted({p.strip().lower() for p in platforms if p and str(p).strip()})
    if not plats:
        return {}
    rows = await conn.fetch(
        """
        SELECT lower(platform) AS platform, hour_utc, prior_weight, trained_at
        FROM m8_publish_hour_priors
        WHERE lower(platform) = ANY($1::text[])
        ORDER BY platform, hour_utc
        """,
        plats,
    )
    by_plat: Dict[str, list] = {}
    for r in rows:
        by_plat.setdefault(r["platform"], []).append(r)
    out: Dict[str, List[float]] = {}
    now = datetime.now(timezone.utc)
    for plat, plat_rows in by_plat.items():
        if len(plat_rows) != 24:
            continue
        trained = plat_rows[0]["trained_at"]
        if trained is None:
            continue
        age_h = (now - trained).total_seconds() / 3600.0
        if age_h > _M8_PRIORS_MAX_AGE_HOURS:
            continue
        out[plat] = _normalize([float(r["prior_weight"] or 0.0) for r in plat_rows])
    return out


async def _fetch_hour_scores(
    conn: Any,
    platform: str,
    *,
    user_id: Optional[str],
    window_start: datetime,
    window_end: datetime,
) -> tuple[List[float], int]:
    batch = await _fetch_hour_scores_batch(
        conn,
        [platform],
        user_id=user_id,
        window_start=window_start,
        window_end=window_end,
    )
    vec, total_n = batch.get(platform.strip().lower(), ([0.0] * 24, 0))
    return vec, total_n


async def _fetch_hour_scores_batch(
    conn: Any,
    platforms: Sequence[str],
    *,
    user_id: Optional[str],
    window_start: datetime,
    window_end: datetime,
) -> Dict[str, tuple[List[float], int]]:
    plats = sorted({p.strip().lower() for p in platforms if p and str(p).strip()})
    if not plats:
        return {}

    user_clause = ""
    params: list[Any] = [plats, window_start, window_end]
    if user_id is not None:
        user_clause = "AND u.user_id = $4::uuid"
        params.append(user_id)

    sql = f"""
        SELECT platform, hr, score, n FROM (
            SELECT
                lower(trim(plat.raw::text)) AS platform,
                EXTRACT(HOUR FROM timezone('UTC', COALESCE(u.completed_at, u.created_at)))::int AS hr,
                SUM(LN(GREATEST(COALESCE(u.views, 0), 0) + 1.0))::double precision AS score,
                COUNT(*)::bigint AS n
            FROM uploads u
            CROSS JOIN LATERAL unnest(COALESCE(u.platforms, ARRAY[]::text[])) AS plat(raw)
            WHERE lower(trim(plat.raw::text)) = ANY($1::text[])
              AND u.status IN ('completed', 'succeeded', 'partial')
              AND COALESCE(u.completed_at, u.created_at) >= $2
              AND COALESCE(u.completed_at, u.created_at) < $3
              {user_clause}
            GROUP BY 1, 2
        ) t
        WHERE hr BETWEEN 0 AND 23
    """
    rows = await conn.fetch(sql, *params)
    grouped: Dict[str, list] = {p: [] for p in plats}
    for r in rows:
        grouped.setdefault(r["platform"], []).append(r)
    out: Dict[str, tuple[List[float], int]] = {}
    for plat in plats:
        plat_rows = grouped.get(plat) or []
        vec = _vec24_from_rows(plat_rows)
        total_n = int(sum(int(r["n"] or 0) for r in plat_rows))
        out[plat] = (vec, total_n)
    return out


async def build_hour_weights_for_platform(
    conn: Any,
    user_id: str,
    platform: str,
) -> List[float]:
    batch = await build_hour_weights_for_platforms_batch(conn, user_id, [platform])
    return batch.get(platform.strip().lower(), static_hour_prior_24(platform))


async def build_hour_weights_for_platforms_batch(
    conn: Any,
    user_id: str,
    platforms: Sequence[str],
) -> Dict[str, List[float]]:
    plats = sorted({p.strip().lower() for p in platforms if p and str(p).strip()})
    if not plats:
        return {}

    now = datetime.now(timezone.utc)
    g_start = now - timedelta(days=_LOOKBACK_GLOBAL_DAYS)
    u_start = now - timedelta(days=_LOOKBACK_USER_DAYS)
    recent_start = now - timedelta(days=_RECENT_DAYS)
    baseline_start = now - timedelta(days=_BASELINE_START_DAYS)
    baseline_end = now - timedelta(days=_BASELINE_END_DAYS)

    global_batch, user_batch, recent_batch, baseline_batch, m8_batch = await _gather_batch(
        conn,
        user_id,
        plats,
        g_start,
        u_start,
        now,
        recent_start,
        baseline_start,
        baseline_end,
    )

    out: Dict[str, List[float]] = {}
    for plat in plats:
        static_w = static_hour_prior_24(plat)
        global_vec, _ = global_batch.get(plat, ([0.0] * 24, 0))
        global_w = _normalize(global_vec)
        fleet_w = m8_batch.get(plat) or global_w

        user_vec, user_n = user_batch.get(plat, ([0.0] * 24, 0))
        user_w = _normalize(user_vec)

        recent_vec, _ = recent_batch.get(plat, ([0.0] * 24, 0))
        baseline_vec, _ = baseline_batch.get(plat, ([0.0] * 24, 0))
        momentum = _momentum_multipliers(_normalize(recent_vec), _normalize(baseline_vec))

        if sum(global_vec) <= 0 and sum(user_vec) <= 0 and plat not in m8_batch:
            out[plat] = static_w
        else:
            out[plat] = _blend_vectors(static_w, fleet_w, user_w, user_n, momentum)
    return out


async def _gather_batch(
    conn: Any,
    user_id: str,
    plats: List[str],
    g_start: datetime,
    u_start: datetime,
    now: datetime,
    recent_start: datetime,
    baseline_start: datetime,
    baseline_end: datetime,
):
    # Sequential on a single asyncpg connection. asyncio.gather on the same
    # ``conn`` raises InterfaceError ("another operation is in progress"), which
    # the API maps to a misleading 503 ("Database temporarily unavailable") —
    # see Sentry UPLOADM8-88 on POST /api/scheduled/.../randomize-schedule.
    global_batch = await _fetch_hour_scores_batch(
        conn, plats, user_id=None, window_start=g_start, window_end=now
    )
    user_batch = await _fetch_hour_scores_batch(
        conn, plats, user_id=user_id, window_start=u_start, window_end=now
    )
    recent_batch = await _fetch_hour_scores_batch(
        conn, plats, user_id=None, window_start=recent_start, window_end=now
    )
    baseline_batch = await _fetch_hour_scores_batch(
        conn, plats, user_id=None, window_start=baseline_start, window_end=baseline_end
    )
    m8_batch = await fetch_m8_hour_priors_batch(conn, plats)
    return global_batch, user_batch, recent_batch, baseline_batch, m8_batch


async def calculate_smart_schedule_data_driven(
    conn: Any,
    user_id: str,
    platforms: List[str],
    *,
    num_days: int = 14,
    blocked_day_offsets: Optional[set] = None,
    user_timezone: str = "UTC",
    random_seed: Optional[str] = None,
) -> Dict[str, datetime]:
    """Smart schedule using batched fleet + user engagement-by-hour signals (UTC)."""
    if not platforms:
        return {}

    hour_weights_by_platform: Dict[str, List[float]] = {}
    try:
        batch = await build_hour_weights_for_platforms_batch(conn, user_id, platforms)
        for p in sorted({str(x).strip().lower() for x in platforms if str(x).strip()}):
            hour_weights_by_platform[p] = batch.get(p, static_hour_prior_24(p))
    except Exception as e:
        logger.warning("smart_schedule batch signals failed: %s", e)
        for p in sorted({str(x).strip().lower() for x in platforms if str(x).strip()}):
            hour_weights_by_platform[p] = static_hour_prior_24(p)

    return calculate_smart_schedule(
        list(hour_weights_by_platform.keys()),
        num_days=num_days,
        user_timezone=user_timezone,
        hour_weights_by_platform=hour_weights_by_platform,
        blocked_day_offsets=blocked_day_offsets,
        random_seed=random_seed,
    )
