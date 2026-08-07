"""
Periodic ML score rollups for strategy performance.

Produces per-user/per-day quality rows with confidence intervals so the generation
engine can bias toward empirically stronger strategies over time.

Engagement labels prefer successful ``platform_results`` metrics for TikTok,
YouTube, Instagram, and Facebook (Meta reactions / impressions aliases included),
falling back to ``uploads.views/likes/comments/shares`` when PR is empty — so
coach “What’s working for you” tips match Analytics-synced engagement.

Also rolls up ``mean_grounding`` (caption–evidence overlap) from
``uploads.output_artifacts`` for coach / accuracy observability — does not
replace engagement priors.
"""
from __future__ import annotations

import logging
import math
import statistics
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

import asyncpg

from services.ml_observability import OptionalTrackioRun
from services.upload_engagement import (
    effective_upload_metrics,
    engagement_rate_pct,
    grounding_score_from_artifacts,
    per_platform_upload_metrics,
    strategy_key_from_artifacts,
)

logger = logging.getLogger("uploadm8.ml_scoring_job")

_AggKey = Tuple[Any, date, str, str]  # user_id, day, platform, strategy_key


def _day_of(created_at: Any) -> Optional[date]:
    if created_at is None:
        return None
    if isinstance(created_at, datetime):
        return created_at.date()
    if isinstance(created_at, date):
        return created_at
    try:
        return datetime.fromisoformat(str(created_at).replace("Z", "+00:00")).date()
    except (TypeError, ValueError):
        return None


def _accumulate(
    buckets: Dict[_AggKey, Dict[str, Any]],
    key: _AggKey,
    *,
    views: float,
    engagement: float,
    grounding: Optional[float],
) -> None:
    b = buckets.get(key)
    if b is None:
        b = {
            "samples": 0,
            "eng": [],
            "views": [],
            "grounding": [],
        }
        buckets[key] = b
    b["samples"] += 1
    b["eng"].append(float(engagement))
    b["views"].append(float(views))
    if grounding is not None and math.isfinite(float(grounding)):
        b["grounding"].append(float(grounding))


def _finalize_row(key: _AggKey, b: Dict[str, Any]) -> Dict[str, Any]:
    user_id, day, platform, strategy_key = key
    samples = int(b["samples"])
    eng = list(b["eng"]) or [0.0]
    views = list(b["views"]) or [0.0]
    mean_engagement = float(statistics.fmean(eng))
    mean_views = float(statistics.fmean(views))
    engagement_stddev = float(statistics.pstdev(eng)) if len(eng) > 1 else 0.0
    mean_grounding = float(statistics.fmean(b["grounding"])) if b["grounding"] else None
    half = 1.96 * engagement_stddev / max(math.sqrt(float(samples)), 1.0)
    return {
        "user_id": user_id,
        "day": day,
        "platform": platform,
        "strategy_key": strategy_key,
        "samples": samples,
        "mean_engagement": mean_engagement,
        "mean_views": mean_views,
        "engagement_stddev": engagement_stddev,
        "ci95_low": max(0.0, mean_engagement - half),
        "ci95_high": mean_engagement + half,
        "mean_grounding": mean_grounding,
    }


def aggregate_quality_score_rows(upload_rows: List[Any]) -> List[Dict[str, Any]]:
    """
    Pure aggregation used by ``recompute_quality_scores`` (and unit tests).

    Builds ``platform='all'`` rows from effective upload metrics (columns ⋃ PR)
    and per-platform rows from true TikTok / YouTube / Meta PR stats when present.
    """
    buckets: Dict[_AggKey, Dict[str, Any]] = {}
    for row in upload_rows or []:
        get = row.get if hasattr(row, "get") else lambda k, d=None: getattr(row, k, d)
        day = _day_of(get("created_at"))
        user_id = get("user_id")
        if day is None or user_id is None:
            continue
        strategy_key = strategy_key_from_artifacts(get("output_artifacts"))
        grounding = grounding_score_from_artifacts(get("output_artifacts"))

        eff = effective_upload_metrics(row, shortform_only=True)
        _accumulate(
            buckets,
            (user_id, day, "all", strategy_key),
            views=float(eff["views"]),
            engagement=engagement_rate_pct(
                eff["views"], eff["likes"], eff["comments"], eff["shares"]
            ),
            grounding=grounding,
        )

        for plat_row in per_platform_upload_metrics(row):
            plat = str(plat_row.get("platform") or "").strip().lower()
            if not plat or plat == "all":
                continue
            _accumulate(
                buckets,
                (user_id, day, plat, strategy_key),
                views=float(plat_row["views"]),
                engagement=float(plat_row.get("engagement_rate_pct") or 0.0),
                grounding=grounding,
            )

    return [_finalize_row(k, b) for k, b in buckets.items()]


async def recompute_quality_scores(pool: asyncpg.Pool, lookback_days: int = 180) -> int:
    """
    Recompute daily quality score rows from uploads + platform_results + attribution keys.
    Returns number of rows inserted/updated (best effort).
    """
    lookback_days = max(7, min(int(lookback_days or 180), 3650))
    async with pool.acquire() as conn:
        await conn.execute(
            """
            DELETE FROM upload_quality_scores_daily
             WHERE day >= (CURRENT_DATE - ($1::int || ' days')::interval)::date
            """,
            lookback_days,
        )

        upload_rows = await conn.fetch(
            """
            SELECT user_id, created_at, platforms, views, likes, comments, shares,
                   platform_results, output_artifacts
              FROM uploads
             WHERE created_at >= (NOW() - ($1::int || ' days')::interval)
               AND status IN ('completed', 'succeeded', 'partial')
            """,
            lookback_days,
        )

        finalized = aggregate_quality_score_rows(list(upload_rows or []))
        if finalized:
            await conn.executemany(
                """
                INSERT INTO upload_quality_scores_daily
                    (user_id, day, platform, strategy_key, samples,
                     mean_engagement, mean_views, engagement_stddev, ci95_low, ci95_high,
                     mean_grounding, updated_at)
                VALUES (
                    $1, $2, $3::varchar(50), $4, $5,
                    $6, $7, $8, $9, $10,
                    $11, NOW()
                )
                ON CONFLICT (user_id, day, platform, strategy_key) DO UPDATE
                SET samples = EXCLUDED.samples,
                    mean_engagement = EXCLUDED.mean_engagement,
                    mean_views = EXCLUDED.mean_views,
                    engagement_stddev = EXCLUDED.engagement_stddev,
                    ci95_low = EXCLUDED.ci95_low,
                    ci95_high = EXCLUDED.ci95_high,
                    mean_grounding = EXCLUDED.mean_grounding,
                    updated_at = NOW()
                """,
                [
                    (
                        r["user_id"],
                        r["day"],
                        r["platform"],
                        r["strategy_key"],
                        r["samples"],
                        r["mean_engagement"],
                        r["mean_views"],
                        r["engagement_stddev"],
                        r["ci95_low"],
                        r["ci95_high"],
                        r["mean_grounding"],
                    )
                    for r in finalized
                ],
            )

        n = await conn.fetchval(
            """
            SELECT COUNT(*)::int
              FROM upload_quality_scores_daily
             WHERE day >= (CURRENT_DATE - ($1::int || ' days')::interval)::date
            """,
            lookback_days,
        )
        return int(n or 0)


async def run_ml_scoring_cycle(
    pool: asyncpg.Pool,
    lookback_days: int = 180,
    *,
    emit_trackio: bool = True,
) -> Optional[int]:
    """
    Recompute daily quality scores.

    ``emit_trackio`` should be ``False`` when called from within another active
    Trackio run (e.g. ``run_ml_engine_cycle``). Starting/finishing a nested run
    would tear down the parent's global trackio session and trigger
    "Call trackio.init() before trackio.log()" warnings.
    """
    track = OptionalTrackioRun("ml_quality_scoring_cycle") if emit_trackio else None
    if track is not None:
        track.start(config={"lookback_days": int(lookback_days)})
    try:
        n = await recompute_quality_scores(pool, lookback_days=lookback_days)
        logger.info("[ml-scoring] recompute complete | rows=%s lookback_days=%s", n, lookback_days)
        if track is not None:
            track.log({"rows_recomputed": int(n or 0), "lookback_days": int(lookback_days), "status": 1})
        return n
    except Exception as e:
        logger.warning("[ml-scoring] cycle failed: %s", e)
        if track is not None:
            track.log({"status": 0, "error": str(e)[:300], "lookback_days": int(lookback_days)})
        return None
    finally:
        if track is not None:
            track.finish()
