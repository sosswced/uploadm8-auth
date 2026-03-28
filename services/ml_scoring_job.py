"""
Periodic ML score rollups for strategy performance.

Produces per-user/per-day quality rows with confidence intervals so the generation
engine can bias toward empirically stronger strategies over time.
"""
from __future__ import annotations

import logging
from typing import Optional

import asyncpg

logger = logging.getLogger("uploadm8.ml_scoring_job")


async def recompute_quality_scores(pool: asyncpg.Pool, lookback_days: int = 180) -> int:
    """
    Recompute daily quality score rows from uploads + persisted feature events.
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

        # all-platform rollup per user/day/strategy
        await conn.execute(
            """
            WITH base AS (
                SELECT
                    u.user_id,
                    DATE(u.created_at) AS day,
                    COALESCE(
                        NULLIF(fe.output_artifacts->>'thumbnail_selection_method', ''),
                        NULLIF(fe.output_artifacts->>'thumbnail_render_method', ''),
                        'default'
                    ) AS strategy_key,
                    GREATEST(COALESCE(u.views, 0), 0)::double precision AS views,
                    CASE
                        WHEN COALESCE(u.views, 0) > 0
                        THEN ((COALESCE(u.likes, 0) + COALESCE(u.comments, 0) + COALESCE(u.shares, 0))::double precision / u.views::double precision) * 100.0
                        ELSE 0.0
                    END AS engagement
                FROM uploads u
                LEFT JOIN upload_feature_events fe
                  ON fe.upload_id = u.id
                WHERE u.created_at >= (NOW() - ($1::int || ' days')::interval)
                  AND u.status IN ('completed', 'succeeded', 'partial')
            ),
            agg AS (
                SELECT
                    user_id, day, strategy_key,
                    COUNT(*)::int AS samples,
                    AVG(engagement)::double precision AS mean_engagement,
                    AVG(views)::double precision AS mean_views,
                    COALESCE(STDDEV_POP(engagement), 0)::double precision AS engagement_stddev
                FROM base
                GROUP BY user_id, day, strategy_key
            )
            INSERT INTO upload_quality_scores_daily
                (user_id, day, platform, strategy_key, samples,
                 mean_engagement, mean_views, engagement_stddev, ci95_low, ci95_high, updated_at)
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
                NOW()
            FROM agg
            ON CONFLICT (user_id, day, platform, strategy_key) DO UPDATE
            SET samples = EXCLUDED.samples,
                mean_engagement = EXCLUDED.mean_engagement,
                mean_views = EXCLUDED.mean_views,
                engagement_stddev = EXCLUDED.engagement_stddev,
                ci95_low = EXCLUDED.ci95_low,
                ci95_high = EXCLUDED.ci95_high,
                updated_at = NOW()
            """,
            lookback_days,
        )

        # platform-specific rollup by exploding uploads.platforms
        await conn.execute(
            """
            WITH base AS (
                SELECT
                    u.user_id,
                    DATE(u.created_at) AS day,
                    LOWER(p.platform)::varchar(50) AS platform,
                    COALESCE(
                        NULLIF(fe.output_artifacts->>'thumbnail_selection_method', ''),
                        NULLIF(fe.output_artifacts->>'thumbnail_render_method', ''),
                        'default'
                    ) AS strategy_key,
                    GREATEST(COALESCE(u.views, 0), 0)::double precision AS views,
                    CASE
                        WHEN COALESCE(u.views, 0) > 0
                        THEN ((COALESCE(u.likes, 0) + COALESCE(u.comments, 0) + COALESCE(u.shares, 0))::double precision / u.views::double precision) * 100.0
                        ELSE 0.0
                    END AS engagement
                FROM uploads u
                LEFT JOIN upload_feature_events fe
                  ON fe.upload_id = u.id
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
                    COALESCE(STDDEV_POP(engagement), 0)::double precision AS engagement_stddev
                FROM base
                GROUP BY user_id, day, platform, strategy_key
            )
            INSERT INTO upload_quality_scores_daily
                (user_id, day, platform, strategy_key, samples,
                 mean_engagement, mean_views, engagement_stddev, ci95_low, ci95_high, updated_at)
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
                NOW()
            FROM agg
            ON CONFLICT (user_id, day, platform, strategy_key) DO UPDATE
            SET samples = EXCLUDED.samples,
                mean_engagement = EXCLUDED.mean_engagement,
                mean_views = EXCLUDED.mean_views,
                engagement_stddev = EXCLUDED.engagement_stddev,
                ci95_low = EXCLUDED.ci95_low,
                ci95_high = EXCLUDED.ci95_high,
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


async def run_ml_scoring_cycle(pool: asyncpg.Pool, lookback_days: int = 180) -> Optional[int]:
    try:
        n = await recompute_quality_scores(pool, lookback_days=lookback_days)
        logger.info("[ml-scoring] recompute complete | rows=%s lookback_days=%s", n, lookback_days)
        return n
    except Exception as e:
        logger.warning("[ml-scoring] cycle failed: %s", e)
        return None

