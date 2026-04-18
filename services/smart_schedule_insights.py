"""
Data-driven smart schedule priors (M8).

1) **M8 publish-hour model** (optional): reads ``m8_publish_hour_priors`` — 24-bin UTC
   weights per platform from ``jobs.train_m8_publish_hour_priors``. Training defaults
   to **PCI ``published_at`` only** (see ``M8_TRAIN_PCI_ONLY``); run history, SHAP
   summaries, and binned calibration live in ``m8_model_runs``. Staleness:
   ``M8_PRIORS_MAX_AGE_HOURS`` (default 168).

2) **SQL fleet + user signals**: successful ``uploads`` by UTC hour, ``LN(views+1)``,
   blended with static priors from ``core.scheduling`` and a recency vs baseline
   momentum multiplier.

When M8 priors are present and fresh, they replace the upload-based **fleet** slice
in the blend (user slice + momentum unchanged).
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

# Stale priors fall back to SQL fleet aggregates only.
_M8_PRIORS_MAX_AGE_HOURS = float(os.environ.get("M8_PRIORS_MAX_AGE_HOURS", str(7 * 24)))


async def fetch_m8_hour_priors(conn: Any, platform: str) -> Optional[List[float]]:
    """
    Load latest M8 model 24-bin UTC weights for ``platform`` if fresh and complete.
    """
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


def _vec24_from_rows(rows: Sequence[Any]) -> List[float]:
    v = [0.0] * 24
    for r in rows:
        h = int(r["hr"])
        if 0 <= h <= 23:
            v[h] = float(r["score"] or 0.0)
    return v


def _normalize(vec: List[float]) -> List[float]:
    s = sum(max(0.0, x) for x in vec)
    if s <= 0:
        return [1.0 / 24.0] * 24
    return [max(0.0, x) / s for x in vec]


def _momentum_multipliers(recent: List[float], baseline: List[float]) -> List[float]:
    """Hourly multiplier in ~[0.8, 1.35] when recent hour scores beat older ones."""
    out: List[float] = []
    for r, b in zip(recent, baseline):
        ratio = (r + 1e-4) / (b + 1e-4)
        # Soft squeeze so one wild hour does not dominate
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


async def _fetch_hour_scores(
    conn: Any,
    platform: str,
    *,
    user_id: Optional[str],
    window_start: datetime,
    window_end: datetime,
) -> tuple[List[float], int]:
    """
    Returns (24-vector of LN(views+1) sums, total row count contributing).
    """
    user_clause = ""
    params: list[Any] = [platform.strip().lower(), window_start, window_end]
    if user_id is not None:
        user_clause = "AND u.user_id = $4::uuid"
        params.append(user_id)

    sql = f"""
        SELECT hr, score, n FROM (
            SELECT
                EXTRACT(HOUR FROM timezone('UTC', COALESCE(u.completed_at, u.created_at)))::int AS hr,
                SUM(LN(GREATEST(COALESCE(u.views, 0), 0) + 1.0))::double precision AS score,
                COUNT(*)::bigint AS n
            FROM uploads u
            CROSS JOIN LATERAL unnest(COALESCE(u.platforms, ARRAY[]::text[])) AS plat(raw)
            WHERE lower(trim(plat.raw::text)) = $1
              AND u.status IN ('completed', 'succeeded', 'partial')
              AND COALESCE(u.completed_at, u.created_at) >= $2
              AND COALESCE(u.completed_at, u.created_at) < $3
              {user_clause}
            GROUP BY 1
        ) t
        WHERE hr BETWEEN 0 AND 23
    """
    rows = await conn.fetch(sql, *params)
    vec = _vec24_from_rows(rows)
    total_n = int(sum(int(r["n"] or 0) for r in rows))
    return vec, total_n


async def build_hour_weights_for_platform(
    conn: Any,
    user_id: str,
    platform: str,
) -> List[float]:
    """Single platform: blended 24h UTC weights for ``calculate_smart_schedule``."""
    now = datetime.now(timezone.utc)
    static_w = static_hour_prior_24(platform)

    g_start = now - timedelta(days=_LOOKBACK_GLOBAL_DAYS)
    global_vec, _ = await _fetch_hour_scores(conn, platform, user_id=None, window_start=g_start, window_end=now)
    global_w = _normalize(global_vec)

    m8_w: Optional[List[float]] = None
    try:
        m8_w = await fetch_m8_hour_priors(conn, platform)
    except Exception as e:
        logger.debug("m8 priors unavailable for %s: %s", platform, e)
        m8_w = None
    fleet_w = _normalize(m8_w) if m8_w is not None else global_w

    u_start = now - timedelta(days=_LOOKBACK_USER_DAYS)
    user_vec, user_n = await _fetch_hour_scores(conn, platform, user_id=user_id, window_start=u_start, window_end=now)
    user_w = _normalize(user_vec)

    recent_vec, _ = await _fetch_hour_scores(
        conn, platform, user_id=None, window_start=now - timedelta(days=_RECENT_DAYS), window_end=now
    )
    baseline_vec, _ = await _fetch_hour_scores(
        conn,
        platform,
        user_id=None,
        window_start=now - timedelta(days=_BASELINE_START_DAYS),
        window_end=now - timedelta(days=_BASELINE_END_DAYS),
    )
    momentum = _momentum_multipliers(_normalize(recent_vec), _normalize(baseline_vec))

    if sum(global_vec) <= 0 and sum(user_vec) <= 0 and m8_w is None:
        return static_w

    blended = _blend_vectors(static_w, fleet_w, user_w, user_n, momentum)
    return blended


async def calculate_smart_schedule_data_driven(
    conn: Any,
    user_id: str,
    platforms: List[str],
    *,
    num_days: int = 7,
    blocked_day_offsets: Optional[set] = None,
) -> Dict[str, datetime]:
    """
    Smart schedule using fleet + user engagement-by-hour signals (UTC).

    Falls back to static priors per platform if queries return empty vectors.
    """
    if not platforms:
        return {}

    hour_weights_by_platform: Dict[str, List[float]] = {}
    for p in sorted(set(platforms)):
        try:
            hour_weights_by_platform[p] = await build_hour_weights_for_platform(conn, user_id, p)
        except Exception as e:
            logger.warning("smart_schedule signals failed for platform=%s: %s", p, e)
            hour_weights_by_platform[p] = static_hour_prior_24(p)

    return calculate_smart_schedule(
        platforms,
        num_days=num_days,
        hour_weights_by_platform=hour_weights_by_platform,
        blocked_day_offsets=blocked_day_offsets,
    )
