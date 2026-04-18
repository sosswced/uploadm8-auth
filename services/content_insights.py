"""
User-facing content attribution insights: which caption/thumbnail/settings buckets
correlate with engagement, simple anomaly flags, and optional preference apply.
"""

from __future__ import annotations

import json
import logging
import math
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from core.content_attribution import (
    parse_content_attribution_key,
    preferences_patch_from_parsed_attribution,
)

logger = logging.getLogger("uploadm8.content_insights")


def _finite(x: Any) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _human_strategy_summary(strategy_key: str) -> str:
    if not strategy_key:
        return "unknown packaging"
    if strategy_key.startswith("legacy|"):
        return "legacy thumbnail pipeline (upgrade uploads for full caption attribution)"
    p = parse_content_attribution_key(strategy_key)
    if not p:
        return strategy_key[:120]
    parts = []
    if p.get("caption_style"):
        parts.append(f"caption style “{p['caption_style']}”")
    if p.get("caption_tone"):
        parts.append(f"tone “{p['caption_tone']}”")
    if p.get("caption_voice"):
        parts.append(f"voice “{p['caption_voice']}”")
    if p.get("m8_engine"):
        parts.append("M8 multimodal engine")
    if p.get("thumbnail_selection_mode"):
        sm = p["thumbnail_selection_mode"]
        parts.append("AI frame pick" if sm == "ai" else "sharpest frame")
    if p.get("thumbnail_render_pipeline"):
        rp = p["thumbnail_render_pipeline"]
        parts.append(f"thumb render “{rp}”")
    if p.get("styled_thumbnails") is not None:
        parts.append("styled thumbnails on" if p["styled_thumbnails"] else "styled thumbnails off")
    if p.get("ai_hashtags_enabled"):
        parts.append("AI hashtags on")
    return ", ".join(parts) if parts else strategy_key[:120]


async def fetch_ranked_strategies(conn, user_id: uuid.UUID, lookback_days: int = 120) -> List[Dict[str, Any]]:
    rows = await conn.fetch(
        """
        SELECT strategy_key,
               SUM(samples)::bigint AS samples,
               CASE WHEN SUM(samples) > 0 THEN
                 SUM(mean_engagement * samples::double precision) / SUM(samples::double precision)
               ELSE 0.0 END AS weighted_mean_engagement,
               MAX(ci95_high)::double precision AS max_ci95_high,
               SUM(mean_views * samples::double precision) / NULLIF(SUM(samples::double precision), 0) AS weighted_mean_views
          FROM upload_quality_scores_daily
         WHERE user_id = $1::uuid
           AND day >= (CURRENT_DATE - $2::int)
         GROUP BY strategy_key
        HAVING SUM(samples) >= 3
         ORDER BY weighted_mean_engagement DESC NULLS LAST
         LIMIT 20
        """,
        user_id,
        max(30, min(int(lookback_days or 120), 730)),
    )
    out: List[Dict[str, Any]] = []
    for r in rows or []:
        sk = str(r["strategy_key"] or "")
        out.append(
            {
                "strategy_key": sk,
                "samples": int(r["samples"] or 0),
                "weighted_mean_engagement_pct": _finite(r["weighted_mean_engagement"]),
                "ci95_high_pct": _finite(r["max_ci95_high"]),
                "weighted_mean_views": _finite(r["weighted_mean_views"]),
                "summary": _human_strategy_summary(sk),
                "parsed": parse_content_attribution_key(sk),
            }
        )
    return out


async def fetch_engagement_anomaly(conn, user_id: uuid.UUID) -> Optional[Dict[str, Any]]:
    """Compare recent uploads vs prior window on engagement rate (likes+comments+shares)/views."""
    row = await conn.fetchrow(
        """
        WITH per AS (
            SELECT
                CASE WHEN created_at >= NOW() - interval '14 days' THEN 'recent' ELSE 'prior' END AS bucket,
                CASE WHEN COALESCE(views, 0) > 0 THEN
                  (COALESCE(likes,0)+COALESCE(comments,0)+COALESCE(shares,0))::double precision / views::double precision * 100.0
                ELSE NULL END AS er,
                COALESCE(views,0)::bigint AS v
            FROM uploads
            WHERE user_id = $1::uuid
              AND status IN ('completed', 'succeeded', 'partial')
              AND created_at >= NOW() - interval '60 days'
        ),
        agg AS (
            SELECT bucket,
                   COUNT(*) FILTER (WHERE er IS NOT NULL)::int AS n,
                   AVG(er) FILTER (WHERE er IS NOT NULL) AS mean_er,
                   COALESCE(STDDEV_POP(er), 0) AS std_er
              FROM per
             GROUP BY bucket
        )
        SELECT
            MAX(mean_er) FILTER (WHERE bucket = 'recent') AS recent_mean,
            MAX(n) FILTER (WHERE bucket = 'recent') AS recent_n,
            MAX(mean_er) FILTER (WHERE bucket = 'prior') AS prior_mean,
            MAX(n) FILTER (WHERE bucket = 'prior') AS prior_n,
            MAX(std_er) FILTER (WHERE bucket = 'prior') AS prior_std
        FROM agg
        """,
        user_id,
    )
    if not row:
        return None
    recent_m = _finite(row["recent_mean"])
    prior_m = _finite(row["prior_mean"])
    recent_n = int(row["recent_n"] or 0)
    prior_n = int(row["prior_n"] or 0)
    prior_std = _finite(row["prior_std"])
    if recent_n < 4 or prior_n < 6:
        return None
    delta = recent_m - prior_m
    threshold = max(0.8, 1.5 * prior_std) if prior_std > 0 else 1.0
    if delta < -threshold:
        return {
            "type": "engagement_drop",
            "severity": "warning",
            "title": "Engagement rate dipped vs your earlier uploads",
            "body": (
                f"Last ~{recent_n} posts average {recent_m:.2f}% interaction rate (likes+comments+shares vs views) "
                f"vs ~{prior_m:.2f}% on the previous window. Audiences may be reacting differently to recent packaging."
            ),
            "recent_mean_engagement_pct": recent_m,
            "prior_mean_engagement_pct": prior_m,
        }
    if delta > threshold and recent_m > prior_m * 1.15:
        return {
            "type": "engagement_lift",
            "severity": "info",
            "title": "Recent packaging is outperforming your earlier baseline",
            "body": (
                f"Interaction rate is ~{recent_m:.2f}% recently vs ~{prior_m:.2f}% before — "
                "consider keeping this cadence and cross-posting winners."
            ),
            "recent_mean_engagement_pct": recent_m,
            "prior_mean_engagement_pct": prior_m,
        }
    return None


def _output_artifacts_as_dict(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        try:
            d = json.loads(raw)
            return dict(d) if isinstance(d, dict) else {}
        except (json.JSONDecodeError, TypeError, ValueError):
            return {}
    if hasattr(raw, "keys"):
        try:
            return dict(raw)
        except Exception:
            return {}
    return {}


def _hashtag_slugs_from_upload_row(row: Any) -> List[str]:
    oa = _output_artifacts_as_dict(row.get("output_artifacts"))
    cav = oa.get("content_attribution_v1")
    if isinstance(cav, str):
        try:
            cav = json.loads(cav)
        except (json.JSONDecodeError, TypeError, ValueError):
            return []
    if not isinstance(cav, dict):
        return []
    raw_list = cav.get("hashtag_slugs_used")
    if not isinstance(raw_list, list):
        return []
    out: List[str] = []
    for x in raw_list:
        t = str(x).strip().lower().lstrip("#")
        if 1 < len(t) < 60:
            out.append(t)
    return out


async def fetch_hashtag_traction(
    conn,
    user_id: uuid.UUID,
    lookback_days: int = 120,
    min_uploads_per_tag: int = 2,
    limit: int = 40,
) -> Dict[str, Any]:
    """
    Per-hashtag aggregates: how often each tag appears on completed uploads and
    mean engagement rate (likes+comments+shares)/views when views > 0.
    """
    lookback_days = max(14, min(int(lookback_days or 120), 730))
    min_uploads_per_tag = max(2, min(int(min_uploads_per_tag or 2), 20))
    rows = await conn.fetch(
        """
        SELECT id, views, likes, comments, shares, created_at, output_artifacts
          FROM uploads
         WHERE user_id = $1::uuid
           AND status IN ('completed', 'succeeded', 'partial')
           AND created_at >= (NOW() - ($2::int || ' days')::interval)
        """,
        user_id,
        lookback_days,
    )
    # tag -> { uploads, sum_views, er_values for mean }
    agg: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"uploads": 0, "sum_views": 0, "er_sum": 0.0, "er_n": 0}
    )
    uploads_with_tags = 0
    for r in rows or []:
        tags = _hashtag_slugs_from_upload_row(dict(r))
        if not tags:
            continue
        uploads_with_tags += 1
        v = int(r["views"] or 0)
        lk = int(r["likes"] or 0)
        cm = int(r["comments"] or 0)
        sh = int(r["shares"] or 0)
        er: Optional[float] = None
        if v > 0:
            er = ((lk + cm + sh) / float(v)) * 100.0
        for tag in tags:
            a = agg[tag]
            a["uploads"] += 1
            a["sum_views"] += max(v, 0)
            if er is not None:
                a["er_sum"] += er
                a["er_n"] += 1

    total_tagged_uploads = uploads_with_tags
    ranked: List[Dict[str, Any]] = []
    for tag, a in agg.items():
        n = int(a["uploads"])
        if n < min_uploads_per_tag:
            continue
        er_n = int(a["er_n"])
        mean_er = (float(a["er_sum"]) / er_n) if er_n > 0 else 0.0
        consistency = (float(n) / total_tagged_uploads) if total_tagged_uploads > 0 else 0.0
        ranked.append(
            {
                "hashtag": tag,
                "upload_count": n,
                "sum_views": int(a["sum_views"]),
                "mean_engagement_pct": _finite(mean_er),
                "engagement_samples": er_n,
                "consistency_pct": round(100.0 * consistency, 2),
            }
        )
    ranked.sort(
        key=lambda x: (-x["mean_engagement_pct"], -x["upload_count"], x["hashtag"]),
    )
    ranked = ranked[:limit]

    # Tags used often but weaker engagement (caution list)
    by_volume = sorted(
        [x for x in ranked if x["upload_count"] >= min_uploads_per_tag],
        key=lambda x: (-x["upload_count"], x["mean_engagement_pct"]),
    )
    underperformers: List[Dict[str, Any]] = []
    if len(by_volume) >= 3:
        median_er = sorted(x["mean_engagement_pct"] for x in by_volume)[len(by_volume) // 2]
        if median_er < 0.02:
            median_er = 0.0
        for x in by_volume[:15]:
            if median_er > 0.02 and x["mean_engagement_pct"] < median_er * 0.65 and x["upload_count"] >= 3:
                underperformers.append(
                    {
                        "hashtag": x["hashtag"],
                        "upload_count": x["upload_count"],
                        "mean_engagement_pct": x["mean_engagement_pct"],
                        "note": "Used often but engagement per view trails your median for tagged uploads",
                    }
                )
        underperformers = underperformers[:8]

    return {
        "lookback_days": lookback_days,
        "uploads_with_hashtag_attribution": total_tagged_uploads,
        "top_by_engagement": ranked,
        "volume_underperformers": underperformers,
    }


async def build_user_content_insights(conn, user_id) -> Dict[str, Any]:
    try:
        uid = user_id if isinstance(user_id, uuid.UUID) else uuid.UUID(str(user_id))
    except (ValueError, TypeError):
        return {
            "ok": False,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "ranked_strategies": [],
            "hashtag_traction": None,
            "anomaly": None,
            "recommended": None,
            "narrative": "Sign in to see personalized content insights.",
        }

    ranked: List[Dict[str, Any]] = []
    try:
        ranked = await fetch_ranked_strategies(conn, uid, 120)
    except Exception as e:
        logger.warning("fetch_ranked_strategies: %s", e)

    anomaly = None
    try:
        anomaly = await fetch_engagement_anomaly(conn, uid)
    except Exception as e:
        logger.debug("fetch_engagement_anomaly: %s", e)

    hashtag_traction: Optional[Dict[str, Any]] = None
    try:
        hashtag_traction = await fetch_hashtag_traction(conn, uid, lookback_days=120)
    except Exception as e:
        logger.warning("fetch_hashtag_traction: %s", e)

    recommended = None
    narrative_parts: List[str] = []
    if ranked:
        top = ranked[0]
        recommended = {
            "strategy_key": top["strategy_key"],
            "summary": top["summary"],
            "confidence_note": (
                f"Based on {top['samples']} scored day-buckets in the last ~120 days "
                f"(mean engagement ~{top['weighted_mean_engagement_pct']:.2f}% vs views)."
            ),
            "preferences_patch": preferences_patch_from_parsed_attribution(top.get("parsed") or {}),
        }
        narrative_parts.append(
            f"Your strongest-performing packaging lately: {top['summary']} "
            f"(~{top['weighted_mean_engagement_pct']:.2f}% engagement rate where views > 0)."
        )
        if len(ranked) > 1:
            second = ranked[1]
            narrative_parts.append(
                f"Runner-up: {second['summary']} at ~{second['weighted_mean_engagement_pct']:.2f}%."
            )
    else:
        narrative_parts.append(
            "Not enough attributed uploads yet — publish a few more videos with AI captions enabled "
            "so we can compare which styles resonate."
        )

    suggested_hashtags: List[str] = []
    if hashtag_traction and (hashtag_traction.get("top_by_engagement") or []):
        top_ht = hashtag_traction["top_by_engagement"][:5]
        narrative_parts.append(
            "Hashtags that showed stronger interaction per view (when you used them on multiple posts): "
            + ", ".join(
                f"#{x['hashtag']} (~{x['mean_engagement_pct']:.2f}% on {x['upload_count']} posts)"
                for x in top_ht
            )
            + "."
        )
        suggested_hashtags = [x["hashtag"] for x in top_ht if x.get("mean_engagement_pct", 0) > 0]
        if hashtag_traction.get("volume_underperformers"):
            weak = hashtag_traction["volume_underperformers"][:3]
            narrative_parts.append(
                "Worth testing alternatives to: "
                + ", ".join(f"#{w['hashtag']}" for w in weak)
                + " (high repeat use but lower engagement vs your other tags)."
            )
    elif hashtag_traction and hashtag_traction.get("uploads_with_hashtag_attribution", 0) == 0:
        narrative_parts.append(
            "Hashtag learning needs a few more posts with saved packaging data — keep AI or manual tags on uploads."
        )

    if anomaly:
        narrative_parts.append(anomaly.get("body") or "")

    if recommended is not None and suggested_hashtags:
        recommended = dict(recommended)
        recommended["suggested_hashtags_for_traction"] = suggested_hashtags
    elif recommended is None and suggested_hashtags:
        recommended = {
            "strategy_key": None,
            "summary": "Hashtag traction signals (no ranked packaging bucket yet)",
            "confidence_note": "From tags repeated across your attributed uploads vs engagement per view.",
            "preferences_patch": {},
            "suggested_hashtags_for_traction": suggested_hashtags,
        }

    return {
        "ok": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ranked_strategies": ranked,
        "hashtag_traction": hashtag_traction,
        "anomaly": anomaly,
        "recommended": recommended,
        "narrative": " ".join(n for n in narrative_parts if n).strip(),
    }


def merge_preferences_patch_for_apply(
    recommended: Dict[str, Any],
    strategy_key_override: Optional[str],
) -> Dict[str, Any]:
    """Resolve camelCase preference patch from recommended row or explicit key."""
    if strategy_key_override:
        parsed = parse_content_attribution_key(strategy_key_override.strip())
        if not parsed:
            raise ValueError("Could not parse strategy_key")
        return preferences_patch_from_parsed_attribution(parsed)
    patch = recommended.get("preferences_patch") if isinstance(recommended, dict) else None
    if not patch:
        raise ValueError("No recommendation available to apply")
    return dict(patch)
