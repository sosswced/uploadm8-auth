"""

Consolidated ML/statistical context for M8 caption prompts.

"""



from __future__ import annotations



import logging

from typing import Any, Dict, List, Optional, Tuple



logger = logging.getLogger("uploadm8.m8_strategy")





async def _thumbnail_selection_priors(

    conn, user_id: str

) -> Tuple[Optional[float], int, Optional[float], int]:

    """Aggregate engagement means for AI vs sharpness thumbnail selection arms."""

    rows = await conn.fetch(

        """

        SELECT strategy_key,

               SUM(samples)::bigint AS samples,

               CASE WHEN SUM(samples) > 0 THEN

                 SUM(mean_engagement * samples::double precision) / SUM(samples::double precision)

               ELSE 0.0 END AS weighted_mean_engagement

          FROM upload_quality_scores_daily

         WHERE user_id = $1::uuid

           AND day >= (CURRENT_DATE - 120)

           AND strategy_key LIKE 'v1|%'

           AND platform = 'all'

         GROUP BY strategy_key

        """,

        user_id,

    )

    sharp_mean: Optional[float] = None

    sharp_n = 0

    ai_mean: Optional[float] = None

    ai_n = 0

    for r in rows or []:

        sk = str(r["strategy_key"] or "")

        samples = int(r["samples"] or 0)

        eng = float(r["weighted_mean_engagement"] or 0)

        if "tsm=sharpness" in sk or "tsel=sharpness" in sk:

            sharp_n += samples

            sharp_mean = eng if sharp_mean is None else max(sharp_mean, eng)

        elif "tsm=ai" in sk or "tsel=ai" in sk:

            ai_n += samples

            ai_mean = eng if ai_mean is None else max(ai_mean, eng)

    return sharp_mean, sharp_n, ai_mean, ai_n





async def build_m8_strategy_context(pool, user_id: str, ctx) -> str:

    """Single prompt block: quality priors, thumbnail bias, hour priors, attribution."""

    blocks: List[str] = []



    try:

        async with pool.acquire() as conn:

            from services.smart_schedule_insights import fetch_m8_hour_priors



            plat = (getattr(ctx, "platforms", None) or ["youtube"])[0]

            priors = await fetch_m8_hour_priors(conn, str(plat))

            if priors:

                peak = sorted(range(24), key=lambda h: priors[h], reverse=True)[:6]

                blocks.append(

                    "SMART SCHEDULE PRIORS (M8 hour model): favor posting near UTC hours "

                    + ", ".join(str(h) for h in peak)

                )

    except Exception as e:

        logger.debug("m8 strategy schedule hints skipped: %s", e)



    try:

        async with pool.acquire() as conn:

            from services.ml_strategy_utils import prefer_ai_thumbnail_vs_sharpness



            sharp_mean, sharp_n, ai_mean, ai_n = await _thumbnail_selection_priors(conn, str(user_id))

            use_ai, detail = prefer_ai_thumbnail_vs_sharpness(

                sharp_mean=sharp_mean,

                sharp_samples=sharp_n,

                ai_mean=ai_mean,

                ai_samples=ai_n,

            )

            reason = str(detail.get("reason") or "")

            if use_ai:

                blocks.append(

                    "THUMBNAIL STRATEGY BIAS: prefer AI frame selection "

                    f"(eb_ai={detail.get('eb_ai'):.2f} vs eb_sharp={detail.get('eb_sharp'):.2f}, {reason})"

                )

            elif sharp_n or ai_n:

                blocks.append(

                    "THUMBNAIL STRATEGY BIAS: prefer sharpness frame pick "

                    f"(eb_sharp={detail.get('eb_sharp'):.2f} vs eb_ai={detail.get('eb_ai'):.2f}, {reason})"

                )

    except Exception as e:

        logger.debug("m8 strategy thumbnail bias skipped: %s", e)



    try:

        async with pool.acquire() as conn:

            from services.content_insights import fetch_ranked_strategies



            import uuid

            uid = user_id if isinstance(user_id, uuid.UUID) else uuid.UUID(str(user_id))
            ranked = await fetch_ranked_strategies(conn, uid, lookback_days=120)

            if ranked:

                top = ranked[0]

                summary = str(top.get("summary") or "top packaging bucket")

                eng = top.get("weighted_mean_engagement_pct")

                samples = int(top.get("samples") or 0)

                blocks.append(

                    f"CONTENT ATTRIBUTION ML: lean toward “{summary}” "

                    f"({samples} scored uploads, {float(eng or 0):.1f}% engagement)"

                )

    except Exception as e:

        logger.debug("m8 strategy attribution skipped: %s", e)



    try:

        arts = getattr(ctx, "output_artifacts", None) or {}

        hot = arts.get("content_hotness")

        if hot:

            blocks.append(f"CONTENT HOTNESS (ML): {hot}")

    except Exception:

        pass



    if not blocks:

        return ""

    return "\n".join(blocks) + "\n"


