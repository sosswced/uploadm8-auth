"""
Periodic ML score rollups for strategy performance.

Produces per-user/per-day quality rows with confidence intervals so the generation
engine can bias toward empirically stronger strategies over time.

Also rolls up ``mean_grounding`` (caption–evidence overlap) from
``uploads.output_artifacts`` for coach / accuracy observability — does not
replace engagement priors.
"""
from __future__ import annotations

import logging
from typing import Optional

import asyncpg

from services.ml_observability import OptionalTrackioRun

logger = logging.getLogger("uploadm8.ml_scoring_job")

# Shared expression: prefer nested hydration_report, fall back to grounding_score_v1.
_GROUNDING_SQL = """
COALESCE(
    NULLIF(u.output_artifacts->'hydration_report'->>'grounding_score', '')::double precision,
    NULLIF(u.output_artifacts->'grounding_score_v1'->>'grounding_score', '')::double precision
)
"""


async def recompute_quality_scores(pool: asyncpg.Pool, lookback_days: int = 180) -> int:
    """
    Recompute daily quality score rows from uploads + output_artifacts attribution keys.
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

        # all-platform rollup per user/day/strategy (attribution from uploads.output_artifacts)
        await conn.execute(
            f"""
            WITH base AS (
                SELECT
                    u.user_id,
                    DATE(u.created_at) AS day,
                    COALESCE(
                        NULLIF(u.output_artifacts->>'content_attribution_key', ''),
                        CONCAT(
                            'legacy|tsel=',
                            COALESCE(NULLIF(u.output_artifacts->>'thumbnail_selection_method', ''), 'na'),
                            '|trend=',
                            COALESCE(NULLIF(u.output_artifacts->>'thumbnail_render_method', ''), 'na')
                        )
                    ) AS strategy_key,
                    GREATEST(COALESCE(u.views, 0), 0)::double precision AS views,
                    CASE
                        WHEN COALESCE(u.views, 0) > 0
                        THEN ((COALESCE(u.likes, 0) + COALESCE(u.comments, 0) + COALESCE(u.shares, 0))::double precision / u.views::double precision) * 100.0
                        ELSE 0.0
                    END AS engagement,
                    {_GROUNDING_SQL} AS grounding_score
                FROM uploads u
                WHERE u.created_at >= (NOW() - ($1::int || ' days')::interval)
                  AND u.status IN ('completed', 'succeeded', 'partial')
            ),
            agg AS (
                SELECT
                    user_id, day, strategy_key,
                    COUNT(*)::int AS samples,
                    AVG(engagement)::double precision AS mean_engagement,
                    AVG(views)::double precision AS mean_views,
                    COALESCE(STDDEV_POP(engagement), 0)::double precision AS engagement_stddev,
                    AVG(grounding_score) FILTER (WHERE grounding_score IS NOT NULL)::double precision AS mean_grounding
                FROM base
                GROUP BY user_id, day, strategy_key
            )
            INSERT INTO upload_quality_scores_daily
                (user_id, day, platform, strategy_key, samples,
                 mean_engagement, mean_views, engagement_stddev, ci95_low, ci95_high,
                 mean_grounding, updated_at)
            SELECT
                user_id,
                day,
                'all'::varchar(50),
                strategy_key,
                samples,
                mean_engagement,
                mean_views,
                engagement_stddev,
                GREATEST(0.0, mean_engagement - (1.96 * engagement_stddev / GREATEST(sqrt(samples::double precision), 1))),
                mean_engagement + (1.96 * engagement_stddev / GREATEST(sqrt(samples::double precision), 1)),
                mean_grounding,
                NOW()
            FROM agg
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
            lookback_days,
        )

        # platform-specific rollup by exploding uploads.platforms
        await conn.execute(
            f"""
            WITH base AS (
                SELECT
                    u.user_id,
                    DATE(u.created_at) AS day,
                    LOWER(p.platform)::varchar(50) AS platform,
                    COALESCE(
                        NULLIF(u.output_artifacts->>'content_attribution_key', ''),
                        CONCAT(
                            'legacy|tsel=',
                            COALESCE(NULLIF(u.output_artifacts->>'thumbnail_selection_method', ''), 'na'),
                            '|trend=',
                            COALESCE(NULLIF(u.output_artifacts->>'thumbnail_render_method', ''), 'na')
                        )
                    ) AS strategy_key,
                    GREATEST(COALESCE(u.views, 0), 0)::double precision AS views,
                    CASE
                        WHEN COALESCE(u.views, 0) > 0
                        THEN ((COALESCE(u.likes, 0) + COALESCE(u.comments, 0) + COALESCE(u.shares, 0))::double precision / u.views::double precision) * 100.0
                        ELSE 0.0
                    END AS engagement,
                    {_GROUNDING_SQL} AS grounding_score
                FROM uploads u
                CROSS JOIN LATERAL unnest(COALESCE(u.platforms, ARRAY[]::text[])) AS p(platform)
                WHERE u.created_at >= (NOW() - ($1::int || ' days')::interval)
                  AND u.status IN ('completed', 'succeeded', 'partial')
            ),
            agg AS (
                SELECT
                    user_id, day, platform, strategy_key,
                    COUNT(*)::int AS samples,
                    AVG(engagement)::double precision AS mean_engagement,
                    AVG(views)::double precision AS mean_views,
                    COALESCE(STDDEV_POP(engagement), 0)::double precision AS engagement_stddev,
                    AVG(grounding_score) FILTER (WHERE grounding_score IS NOT NULL)::double precision AS mean_grounding
                FROM base
                GROUP BY user_id, day, platform, strategy_key
            )
            INSERT INTO upload_quality_scores_daily
                (user_id, day, platform, strategy_key, samples,
                 mean_engagement, mean_views, engagement_stddev, ci95_low, ci95_high,
                 mean_grounding, updated_at)
            SELECT
                user_id,
                day,
                platform,
                strategy_key,
                samples,
                mean_engagement,
                mean_views,
                engagement_stddev,
                GREATEST(0.0, mean_engagement - (1.96 * engagement_stddev / GREATEST(sqrt(samples::double precision), 1))),
                mean_engagement + (1.96 * engagement_stddev / GREATEST(sqrt(samples::double precision), 1)),
                mean_grounding,
                NOW()
            FROM agg
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
            lookback_days,
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
