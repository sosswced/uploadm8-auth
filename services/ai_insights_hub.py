"""
Customer-facing AI Insights hub — aggregates coach, attribution, platforms, and setup.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from services.content_insights import build_user_content_insights
from services.growth_intelligence import (
    build_user_coach_payload,
    fetch_user_pikzels_studio_usage,
    parse_range_since_until,
)
from services.thumbnail_niches import normalize_niche
from services.visual_entity_memory import fetch_channel_catalog_detail

logger = logging.getLogger("uploadm8.ai_insights_hub")


def _prefs_summary(prefs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not prefs:
        return {}
    nested = prefs.get("thumbnailDefaultStrategy") or prefs.get("thumbnail_default_strategy") or {}
    if not isinstance(nested, dict):
        nested = {}
    return {
        "caption_style": prefs.get("captionStyle") or prefs.get("caption_style"),
        "caption_tone": prefs.get("captionTone") or prefs.get("caption_tone"),
        "caption_voice": prefs.get("captionVoice") or prefs.get("caption_voice"),
        "ai_hashtags_enabled": prefs.get("aiHashtagsEnabled") if prefs.get("aiHashtagsEnabled") is not None else prefs.get("ai_hashtags_enabled"),
        "auto_captions": prefs.get("autoCaptions") if prefs.get("autoCaptions") is not None else prefs.get("auto_captions"),
        "thumbnail_persona_enabled": prefs.get("thumbnailPersonaEnabled") if prefs.get("thumbnailPersonaEnabled") is not None else prefs.get("thumbnail_persona_enabled"),
        "thumbnail_default_persona_id": prefs.get("thumbnailDefaultPersonaId") or prefs.get("thumbnail_default_persona_id"),
        "audience_niche": nested.get("audience_niche") or prefs.get("audienceNiche") or prefs.get("audience_niche"),
        "thumbnail_selection_mode": nested.get("thumbnailSelectionMode") or nested.get("thumbnail_selection_mode"),
        "thumbnail_render_pipeline": nested.get("thumbnailRenderPipeline") or nested.get("thumbnail_render_pipeline"),
    }


async def fetch_user_platform_engagement(
    conn: Any, user_id: uuid.UUID, *, days: int = 90, limit: int = 8
) -> List[Dict[str, Any]]:
    since = datetime.now(timezone.utc) - timedelta(days=max(14, min(days, 365)))
    rows = await conn.fetch(
        """
        SELECT
            TRIM(LOWER(unnest(u.platforms))) AS platform,
            COUNT(*)::bigint AS uploads,
            COALESCE(AVG(u.views), 0)::float AS avg_views,
            COALESCE(AVG(u.likes), 0)::float AS avg_likes,
            COALESCE(AVG(u.comments), 0)::float AS avg_comments,
            COALESCE(AVG(u.shares), 0)::float AS avg_shares,
            COALESCE(SUM(COALESCE(u.views, 0)), 0)::bigint AS sum_views,
            COALESCE(SUM(COALESCE(u.likes, 0) + COALESCE(u.comments, 0) + COALESCE(u.shares, 0)), 0)::bigint AS sum_interactions,
            COALESCE(AVG(
                CASE WHEN COALESCE(u.views, 0) > 0 THEN
                    (COALESCE(u.likes, 0) + COALESCE(u.comments, 0) + COALESCE(u.shares, 0))::float
                    / NULLIF(u.views::float, 0) * 100.0
                ELSE NULL END
            ), 0)::float AS avg_engagement_rate_pct
        FROM uploads u
        WHERE u.user_id = $1::uuid
          AND u.created_at >= $2
          AND u.status IN ('completed', 'succeeded', 'partial')
          AND u.platforms IS NOT NULL AND array_length(u.platforms, 1) > 0
        GROUP BY 1
        HAVING COUNT(*) >= 1
        ORDER BY avg_engagement_rate_pct DESC NULLS LAST, uploads DESC
        LIMIT $3
        """,
        user_id,
        since,
        limit,
    )
    icons = {
        "youtube": "fab fa-youtube",
        "tiktok": "fab fa-tiktok",
        "instagram": "fab fa-instagram",
        "facebook": "fab fa-facebook",
    }
    out: List[Dict[str, Any]] = []
    for r in rows or []:
        plat = str(r["platform"] or "")
        out.append(
            {
                "platform": plat,
                "icon": icons.get(plat, "fas fa-globe"),
                "uploads": int(r["uploads"] or 0),
                "avg_views": round(float(r["avg_views"] or 0), 1),
                "avg_likes": round(float(r["avg_likes"] or 0), 1),
                "avg_comments": round(float(r["avg_comments"] or 0), 1),
                "avg_shares": round(float(r["avg_shares"] or 0), 1),
                "sum_views": int(r["sum_views"] or 0),
                "sum_interactions": int(r["sum_interactions"] or 0),
                "avg_engagement_rate_pct": round(float(r["avg_engagement_rate_pct"] or 0), 3),
            }
        )
    return out


def _artifacts_dict(raw: Any) -> Dict[str, Any]:
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
    return {}


def _content_attribution_from_artifacts(oa: Dict[str, Any]) -> Dict[str, Any]:
    cav = oa.get("content_attribution_v1")
    if isinstance(cav, str):
        try:
            cav = json.loads(cav)
        except (json.JSONDecodeError, TypeError, ValueError):
            return {}
    return dict(cav) if isinstance(cav, dict) else {}


def _engagement_rate_pct(views: int, likes: int, comments: int, shares: int) -> Optional[float]:
    if views <= 0:
        return None
    return round(((likes + comments + shares) / float(views)) * 100.0, 3)


async def fetch_platform_engagement_trends(
    conn: Any, user_id: uuid.UUID, *, weeks: int = 12
) -> Dict[str, Any]:
    """Weekly engagement + views per platform for Chart.js line charts."""
    since = datetime.now(timezone.utc) - timedelta(days=max(7, min(weeks, 52) * 7))
    rows = await conn.fetch(
        """
        SELECT
            date_trunc('week', u.created_at AT TIME ZONE 'UTC')::date AS week_start,
            TRIM(LOWER(unnest(u.platforms))) AS platform,
            COUNT(*)::bigint AS uploads,
            COALESCE(SUM(COALESCE(u.views, 0)), 0)::bigint AS sum_views,
            COALESCE(SUM(COALESCE(u.likes, 0)), 0)::bigint AS sum_likes,
            COALESCE(SUM(COALESCE(u.comments, 0)), 0)::bigint AS sum_comments,
            COALESCE(SUM(COALESCE(u.shares, 0)), 0)::bigint AS sum_shares,
            COALESCE(AVG(
                CASE WHEN COALESCE(u.views, 0) > 0 THEN
                    (COALESCE(u.likes, 0) + COALESCE(u.comments, 0) + COALESCE(u.shares, 0))::float
                    / NULLIF(u.views::float, 0) * 100.0
                ELSE NULL END
            ), 0)::float AS avg_engagement_rate_pct
        FROM uploads u
        WHERE u.user_id = $1::uuid
          AND u.created_at >= $2
          AND u.status IN ('completed', 'succeeded', 'partial')
          AND u.platforms IS NOT NULL AND array_length(u.platforms, 1) > 0
        GROUP BY 1, 2
        ORDER BY week_start ASC, platform ASC
        """,
        user_id,
        since,
    )
    week_set: set = set()
    by_plat: Dict[str, Dict[date, Dict[str, Any]]] = defaultdict(dict)
    for r in rows or []:
        ws = r["week_start"]
        if isinstance(ws, datetime):
            ws = ws.date()
        plat = str(r["platform"] or "")
        if not plat or not ws:
            continue
        week_set.add(ws)
        by_plat[plat][ws] = {
            "uploads": int(r["uploads"] or 0),
            "sum_views": int(r["sum_views"] or 0),
            "sum_likes": int(r["sum_likes"] or 0),
            "sum_comments": int(r["sum_comments"] or 0),
            "sum_shares": int(r["sum_shares"] or 0),
            "engagement_rate_pct": round(float(r["avg_engagement_rate_pct"] or 0), 3),
        }
    weeks_sorted = sorted(week_set)
    labels = [w.isoformat() for w in weeks_sorted]
    palette = {
        "youtube": "#ef4444",
        "tiktok": "#22d3ee",
        "instagram": "#e879f9",
        "facebook": "#3b82f6",
    }
    series: List[Dict[str, Any]] = []
    for plat in sorted(by_plat.keys()):
        pts = by_plat[plat]
        series.append(
            {
                "platform": plat,
                "color": palette.get(plat, "#f97316"),
                "engagement_rate_pct": [pts.get(w, {}).get("engagement_rate_pct", 0) for w in weeks_sorted],
                "views": [pts.get(w, {}).get("sum_views", 0) for w in weeks_sorted],
                "likes": [pts.get(w, {}).get("sum_likes", 0) for w in weeks_sorted],
                "comments": [pts.get(w, {}).get("sum_comments", 0) for w in weeks_sorted],
                "uploads": [pts.get(w, {}).get("uploads", 0) for w in weeks_sorted],
            }
        )
    return {"weeks": labels, "series": series}


def _packaging_label(parts: Dict[str, str]) -> str:
    bits: List[str] = []
    if parts.get("persona"):
        bits.append(f"persona: {parts['persona']}")
    if parts.get("pipeline"):
        bits.append(f"pipeline: {parts['pipeline']}")
    if parts.get("selection"):
        bits.append(f"frame pick: {parts['selection']}")
    if parts.get("render"):
        bits.append(f"render: {parts['render']}")
    if parts.get("category"):
        bits.append(f"category: {parts['category']}")
    if parts.get("variant"):
        bits.append(f"studio variant: {parts['variant'][:12]}")
    return " · ".join(bits) if bits else "default packaging"


async def fetch_packaging_variant_rollups(
    conn: Any, user_id: uuid.UUID, *, days: int = 120, min_uploads: int = 2, limit: int = 12
) -> Dict[str, Any]:
    """
    Correlate thumbnail templates, render pipelines, personas, and studio variant ids
    with per-upload engagement (likes, comments, shares vs views).
    """
    lookback = max(30, min(int(days or 120), 365))
    rows = await conn.fetch(
        """
        SELECT views, likes, comments, shares, output_artifacts, studio_content_variant_id
          FROM uploads
         WHERE user_id = $1::uuid
           AND status IN ('completed', 'succeeded', 'partial')
           AND created_at >= (NOW() - ($2::int || ' days')::interval)
        """,
        user_id,
        lookback,
    )
    studio_variants = 0
    try:
        studio_variants = int(
            await conn.fetchval(
                "SELECT COUNT(*)::int FROM thumbnail_recreate_variants WHERE user_id = $1::uuid",
                user_id,
            )
            or 0
        )
    except Exception:
        studio_variants = 0

    agg: Dict[str, Dict[str, Any]] = {}
    attributed = 0
    for r in rows or []:
        oa = _artifacts_dict(r.get("output_artifacts"))
        cav = _content_attribution_from_artifacts(oa)
        if oa or cav:
            attributed += 1
        parts = {
            "pipeline": str(
                cav.get("thumbnail_render_pipeline")
                or oa.get("thumbnail_render_pipeline")
                or ""
            ).strip()
            or "auto",
            "selection": str(
                cav.get("thumbnail_selection_mode")
                or oa.get("thumbnail_selection_method")
                or ""
            ).strip()
            or "ai",
            "render": str(oa.get("thumbnail_render_method") or cav.get("thumbnail_render_method") or "").strip()
            or "auto",
            "category": str(cav.get("thumbnail_category") or oa.get("thumbnail_category") or "").strip()
            or "general",
            "persona": str(cav.get("effective_persona") or cav.get("caption_voice") or "").strip(),
            "variant": str(r.get("studio_content_variant_id") or "").strip(),
        }
        key = "|".join(f"{k}={parts[k]}" for k in sorted(parts.keys()))
        bucket = agg.setdefault(
            key,
            {
                "parts": parts,
                "label": _packaging_label(parts),
                "uploads": 0,
                "er_sum": 0.0,
                "er_n": 0,
                "views_sum": 0,
                "likes_sum": 0,
                "comments_sum": 0,
            },
        )
        bucket["uploads"] += 1
        v = int(r["views"] or 0)
        lk = int(r["likes"] or 0)
        cm = int(r["comments"] or 0)
        sh = int(r["shares"] or 0)
        bucket["views_sum"] += max(v, 0)
        bucket["likes_sum"] += lk
        bucket["comments_sum"] += cm
        er = _engagement_rate_pct(v, lk, cm, sh)
        if er is not None:
            bucket["er_sum"] += er
            bucket["er_n"] += 1

    ranked: List[Dict[str, Any]] = []
    for key, b in agg.items():
        n = int(b["uploads"])
        if n < min_uploads:
            continue
        er_n = int(b["er_n"])
        mean_er = round(b["er_sum"] / er_n, 3) if er_n else 0.0
        ranked.append(
            {
                "key": key,
                "label": b["label"],
                "parts": b["parts"],
                "uploads": n,
                "mean_engagement_pct": mean_er,
                "avg_views": round(b["views_sum"] / n, 1) if n else 0,
                "avg_likes": round(b["likes_sum"] / n, 1) if n else 0,
                "avg_comments": round(b["comments_sum"] / n, 1) if n else 0,
            }
        )
    ranked.sort(key=lambda x: (-x["mean_engagement_pct"], -x["uploads"]))

    by_dimension: Dict[str, List[Dict[str, Any]]] = {}
    for dim in ("pipeline", "selection", "render", "persona", "category"):
        sub: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"uploads": 0, "er_sum": 0.0, "er_n": 0})
        for item in ranked:
            val = (item.get("parts") or {}).get(dim) or "—"
            s = sub[val]
            s["uploads"] += item["uploads"]
            if item["mean_engagement_pct"] > 0:
                s["er_sum"] += item["mean_engagement_pct"] * item["uploads"]
                s["er_n"] += item["uploads"]
        dim_rows = []
        for val, s in sub.items():
            n = int(s["uploads"])
            if n < min_uploads:
                continue
            dim_rows.append(
                {
                    "value": val,
                    "uploads": n,
                    "mean_engagement_pct": round(s["er_sum"] / max(s["er_n"], 1), 3),
                }
            )
        dim_rows.sort(key=lambda x: (-x["mean_engagement_pct"], -x["uploads"]))
        by_dimension[dim] = dim_rows[:6]

    return {
        "lookback_days": lookback,
        "uploads_analyzed": len(rows or []),
        "uploads_with_attribution": attributed,
        "studio_variant_rows": studio_variants,
        "combos": ranked[:limit],
        "by_dimension": by_dimension,
    }


async def _fetch_prefs_and_personas(conn: Any, user_id: uuid.UUID) -> Dict[str, Any]:
    row = await conn.fetchrow(
        """
        SELECT u.preferences AS user_prefs,
               (SELECT COUNT(*)::int FROM creator_personas cp
                 WHERE cp.user_id = u.id) AS persona_count
          FROM users u
         WHERE u.id = $1::uuid
        """,
        user_id,
    )
    if not row:
        return {"persona_count": 0, "setup": {}}
    raw = row.get("user_prefs") or {}
    if isinstance(raw, str):
        try:
            import json

            raw = json.loads(raw)
        except Exception:
            raw = {}
    if not isinstance(raw, dict):
        raw = {}
    return {"persona_count": int(row["persona_count"] or 0), "setup": _prefs_summary(raw)}


def ai_insights_hub_fallback(*, error: Optional[str] = None) -> Dict[str, Any]:
    """Degraded hub when DB is unavailable or a gather task fails."""
    return {
        "ok": False,
        "error": error or "insights_unavailable",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "readiness": "building",
        "winning_formula": None,
        "m8_engine": None,
        "tier": "free",
        "engagement_snapshot": {"samples_30d": 0},
        "baselines": {},
        "platforms": [],
        "platform_trends": {"weeks": [], "series": []},
        "packaging_rollups": {},
        "content_insights": None,
        "channel_catalog": None,
        "studio_usage": None,
        "current_setup": {},
        "persona_count": 0,
        "coach_suggestions": [],
        "playbook": [],
    }


async def build_ai_insights_hub(pool: Any, user_id) -> Dict[str, Any]:
    try:
        uid = user_id if isinstance(user_id, uuid.UUID) else uuid.UUID(str(user_id))
    except (ValueError, TypeError):
        return ai_insights_hub_fallback(error="invalid_user")

    since, until = parse_range_since_until("90d")

    async def _acquire_run(coro):
        async with pool.acquire() as c:
            return await coro(c)

    async def _platforms(c):
        return await fetch_user_platform_engagement(c, uid, days=90)

    async def _catalog():
        category = "general"
        async with pool.acquire() as conn:
            row = await conn.fetchrow("SELECT preferences FROM users WHERE id = $1::uuid", uid)
        if row:
            prefs = row.get("preferences") or {}
            if isinstance(prefs, dict):
                nested = prefs.get("thumbnailDefaultStrategy") or prefs.get("thumbnail_default_strategy")
                if isinstance(nested, dict) and nested.get("audience_niche"):
                    category = normalize_niche(str(nested["audience_niche"]))
        return await fetch_channel_catalog_detail(pool, user_id=str(uid), category=category, limit_per_bucket=12)

    async def _studio(c):
        return await fetch_user_pikzels_studio_usage(c, str(uid), since, until)

    async def _prefs(c):
        return await _fetch_prefs_and_personas(c, uid)

    async def _insights(c):
        return await build_user_content_insights(c, uid)

    async def _trends(c):
        return await fetch_platform_engagement_trends(c, uid, weeks=12)

    async def _packaging(c):
        return await fetch_packaging_variant_rollups(c, uid, days=120)

    try:
        (
            platforms,
            catalog,
            studio,
            prefs_block,
            content_insights,
            platform_trends,
            packaging_rollups,
            coach,
        ) = await asyncio.gather(
            _acquire_run(_platforms),
            _catalog(),
            _acquire_run(_studio),
            _acquire_run(_prefs),
            _acquire_run(_insights),
            _acquire_run(_trends),
            _acquire_run(_packaging),
            build_user_coach_payload(pool, uid),
        )
    except Exception:
        logger.exception("ai_insights_hub gather failed user_id=%s", user_id)
        return ai_insights_hub_fallback(error="insights_unavailable")
    eng = (coach or {}).get("engagement_snapshot") or {}
    baselines = (coach or {}).get("baselines") or {}

    top_platform = platforms[0] if platforms else None
    ranked = (content_insights or {}).get("ranked_strategies") or []
    top_strategy = ranked[0] if ranked else None
    hashtag_top = ((content_insights or {}).get("hashtag_traction") or {}).get("top_by_engagement") or []

    formula_parts: List[str] = []
    if top_platform:
        formula_parts.append(
            f"Strongest platform lately: {top_platform['platform'].title()} "
            f"(~{top_platform['avg_engagement_rate_pct']:.2f}% engagement on {top_platform['uploads']} posts)."
        )
    if top_strategy:
        formula_parts.append(f"Best packaging combo: {top_strategy.get('summary', '')}.")
    if hashtag_top:
        tags = ", ".join(f"#{h['hashtag']}" for h in hashtag_top[:3])
        formula_parts.append(f"Hashtags that earned traction: {tags}.")
    top_pack = (packaging_rollups or {}).get("combos") or []
    if top_pack:
        formula_parts.append(f"Top thumbnail/template combo: {top_pack[0].get('label', '')}.")

    readiness = "building"
    samples = int(eng.get("samples_30d") or 0)
    if samples >= 8 and ranked:
        readiness = "ready"
    elif samples >= 3:
        readiness = "emerging"

    playbook = [
        {"id": "analytics", "label": "Analytics", "href": "analytics.html", "icon": "fas fa-chart-line", "hint": "Views, likes, comments, shares by upload"},
        {"id": "kpi", "label": "Upload KPIs", "href": "kpi.html", "icon": "fas fa-chart-bar", "hint": "Throughput and success rates"},
        {"id": "studio", "label": "Thumbnail Studio", "href": "thumbnail-studio.html", "icon": "fas fa-images", "hint": "Personas, variants, and AI thumbnails"},
        {"id": "upload", "label": "Upload", "href": "upload.html", "icon": "fas fa-cloud-upload-alt", "hint": "Ship with your optimized defaults"},
        {"id": "settings", "label": "AI settings", "href": "settings.html#preferences", "icon": "fas fa-sliders-h", "hint": "Caption tone, hashtags, personas"},
        {"id": "platforms", "label": "Connected accounts", "href": "platforms.html", "icon": "fas fa-plug", "hint": "OAuth health per platform"},
        {"id": "scheduled", "label": "Scheduled", "href": "scheduled.html", "icon": "fas fa-calendar-alt", "hint": "Rhythm and peak windows"},
    ]

    return {
        "ok": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "readiness": readiness,
        "winning_formula": " ".join(formula_parts) if formula_parts else None,
        "m8_engine": (coach or {}).get("m8_engine"),
        "tier": (coach or {}).get("tier"),
        "engagement_snapshot": eng,
        "baselines": baselines,
        "platforms": platforms,
        "platform_trends": platform_trends,
        "packaging_rollups": packaging_rollups,
        "content_insights": content_insights,
        "channel_catalog": catalog,
        "studio_usage": studio,
        "current_setup": prefs_block.get("setup") or {},
        "persona_count": prefs_block.get("persona_count") or 0,
        "coach_suggestions": (coach or {}).get("suggestions") or [],
        "playbook": playbook,
    }
