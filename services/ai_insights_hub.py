"""
Customer-facing AI Insights hub — aggregates coach, attribution, platforms, and setup.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from services.content_insights import build_user_content_insights
from services.growth_intelligence import (
    build_user_coach_payload,
    fetch_user_pikzels_studio_usage,
    parse_range_since_until,
    sanitize_coach_payload_for_json,
)
from core.helpers import coerce_jsonb_dict
from services.ml_hub_config import get_ml_hub_urls, ml_hub_huggingface_dict
from services.thumbnail_niches import normalize_niche
from services.visual_entity_memory import fetch_channel_catalog_detail

logger = logging.getLogger("uploadm8.ai_insights_hub")


def fetch_content_success_rankings() -> Dict[str, Any]:
    """Latest content-success rankings from local ML report (admin/train cycle)."""
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for name in ("content_success_report.json", "content_success_baseline_report.json"):
        p = root / "data" / "ml" / name
        if not p.is_file():
            continue
        try:
            rep = json.loads(p.read_text(encoding="utf-8"))
            rankings = rep.get("rankings") or {}
            if rankings:
                return sanitize_coach_payload_for_json(rankings)
        except Exception:
            continue
    return {}


def _prefs_summary(prefs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not prefs:
        return {}
    from services.thumbnail_studio_strategy import read_thumbnail_studio_default_strategy, strategy_audience_niche

    nested = read_thumbnail_studio_default_strategy(prefs)
    return {
        "caption_style": prefs.get("captionStyle") or prefs.get("caption_style"),
        "caption_tone": prefs.get("captionTone") or prefs.get("caption_tone"),
        "caption_voice": prefs.get("captionVoice") or prefs.get("caption_voice"),
        "ai_hashtags_enabled": prefs.get("aiHashtagsEnabled") if prefs.get("aiHashtagsEnabled") is not None else prefs.get("ai_hashtags_enabled"),
        "auto_captions": prefs.get("autoCaptions") if prefs.get("autoCaptions") is not None else prefs.get("auto_captions"),
        "thumbnail_persona_enabled": prefs.get("thumbnailPersonaEnabled") if prefs.get("thumbnailPersonaEnabled") is not None else prefs.get("thumbnail_persona_enabled"),
        "thumbnail_default_persona_id": prefs.get("thumbnailDefaultPersonaId") or prefs.get("thumbnail_default_persona_id"),
        "audience_niche": strategy_audience_niche(prefs) if nested or prefs.get("audienceNiche") or prefs.get("audience_niche") else (prefs.get("audienceNiche") or prefs.get("audience_niche")),
        "thumbnail_selection_mode": nested.get("thumbnailSelectionMode") or nested.get("thumbnail_selection_mode"),
        "thumbnail_render_pipeline": nested.get("thumbnailRenderPipeline") or nested.get("thumbnail_render_pipeline"),
    }


async def fetch_user_platform_engagement(
    conn: Any, user_id: uuid.UUID, *, days: int = 90, limit: int = 8
) -> List[Dict[str, Any]]:
    """Per-platform engagement from true TikTok / YouTube / Meta platform_results when present."""
    from services.upload_engagement import COACH_ENGAGEMENT_PLATFORMS, per_platform_upload_metrics

    since = datetime.now(timezone.utc) - timedelta(days=max(14, min(days, 365)))
    rows = await conn.fetch(
        """
        SELECT platforms, views, likes, comments, shares, platform_results
          FROM uploads
         WHERE user_id = $1::uuid
           AND created_at >= $2
           AND status IN ('completed', 'succeeded', 'partial')
        """,
        user_id,
        since,
    )
    icons = {
        "youtube": "fab fa-youtube",
        "tiktok": "fab fa-tiktok",
        "instagram": "fab fa-instagram",
        "facebook": "fab fa-facebook",
    }
    agg: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "uploads": 0,
            "sum_views": 0,
            "sum_likes": 0,
            "sum_comments": 0,
            "sum_shares": 0,
            "sum_interactions": 0,
            "er_sum": 0.0,
            "er_n": 0,
        }
    )
    for r in rows or []:
        for m in per_platform_upload_metrics(r):
            plat = str(m.get("platform") or "").strip().lower()
            if plat not in COACH_ENGAGEMENT_PLATFORMS:
                continue
            a = agg[plat]
            a["uploads"] += 1
            a["sum_views"] += int(m["views"])
            a["sum_likes"] += int(m["likes"])
            a["sum_comments"] += int(m["comments"])
            a["sum_shares"] += int(m["shares"])
            a["sum_interactions"] += int(m["likes"]) + int(m["comments"]) + int(m["shares"])
            er = m.get("engagement_rate_pct")
            if er is not None and int(m["views"] or 0) > 0:
                a["er_sum"] += float(er)
                a["er_n"] += 1

    out: List[Dict[str, Any]] = []
    for plat, a in agg.items():
        n = max(int(a["uploads"]), 1)
        er_n = int(a["er_n"])
        out.append(
            {
                "platform": plat,
                "icon": icons.get(plat, "fas fa-globe"),
                "uploads": int(a["uploads"]),
                "avg_views": round(float(a["sum_views"]) / n, 1),
                "avg_likes": round(float(a["sum_likes"]) / n, 1),
                "avg_comments": round(float(a["sum_comments"]) / n, 1),
                "avg_shares": round(float(a["sum_shares"]) / n, 1),
                "sum_views": int(a["sum_views"]),
                "sum_interactions": int(a["sum_interactions"]),
                "avg_engagement_rate_pct": round(
                    (float(a["er_sum"]) / er_n) if er_n > 0 else 0.0, 3
                ),
            }
        )
    out.sort(key=lambda x: (-x["avg_engagement_rate_pct"], -x["uploads"]))
    return out[:limit]


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
    from services.upload_engagement import COACH_ENGAGEMENT_PLATFORMS, per_platform_upload_metrics

    since = datetime.now(timezone.utc) - timedelta(days=max(7, min(weeks, 52) * 7))
    rows = await conn.fetch(
        """
        SELECT created_at, platforms, views, likes, comments, shares, platform_results
          FROM uploads
         WHERE user_id = $1::uuid
           AND created_at >= $2
           AND status IN ('completed', 'succeeded', 'partial')
        """,
        user_id,
        since,
    )
    week_set: set = set()
    by_plat: Dict[str, Dict[date, Dict[str, Any]]] = defaultdict(dict)
    for r in rows or []:
        created = r["created_at"]
        if created is None:
            continue
        if isinstance(created, datetime):
            ws = created.astimezone(timezone.utc).date()
            # Align to week start (Monday) like date_trunc('week') in UTC-ish PG
            ws = ws - timedelta(days=ws.weekday())
        elif isinstance(created, date):
            ws = created - timedelta(days=created.weekday())
        else:
            continue
        for m in per_platform_upload_metrics(r):
            plat = str(m.get("platform") or "").strip().lower()
            if plat not in COACH_ENGAGEMENT_PLATFORMS:
                continue
            week_set.add(ws)
            cell = by_plat[plat].setdefault(
                ws,
                {
                    "uploads": 0,
                    "sum_views": 0,
                    "sum_likes": 0,
                    "sum_comments": 0,
                    "sum_shares": 0,
                    "er_sum": 0.0,
                    "er_n": 0,
                },
            )
            cell["uploads"] += 1
            cell["sum_views"] += int(m["views"])
            cell["sum_likes"] += int(m["likes"])
            cell["sum_comments"] += int(m["comments"])
            cell["sum_shares"] += int(m["shares"])
            if int(m["views"] or 0) > 0:
                cell["er_sum"] += float(m.get("engagement_rate_pct") or 0.0)
                cell["er_n"] += 1
            if cell["er_n"] > 0:
                cell["engagement_rate_pct"] = round(cell["er_sum"] / cell["er_n"], 3)
            else:
                cell["engagement_rate_pct"] = 0.0
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
    from services.upload_engagement import effective_upload_metrics, engagement_rate_pct

    rows = await conn.fetch(
        """
        SELECT views, likes, comments, shares, output_artifacts, studio_content_variant_id,
               platform_results
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
        m = effective_upload_metrics(r, shortform_only=True)
        v = int(m["views"] or 0)
        lk = int(m["likes"] or 0)
        cm = int(m["comments"] or 0)
        sh = int(m["shares"] or 0)
        bucket["views_sum"] += max(v, 0)
        bucket["likes_sum"] += lk
        bucket["comments_sum"] += cm
        er = _engagement_rate_pct(v, lk, cm, sh)
        if er is None and v > 0:
            er = engagement_rate_pct(v, lk, cm, sh)
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


def _unlock_progress(samples_30d: int, ranked_n: int, platforms_n: int, has_catalog: bool) -> Dict[str, Any]:
    """Guided checklist toward readiness:ready for the Smart Insights decoder."""
    steps = [
        {
            "id": "connect",
            "label": "Connect at least one platform",
            "done": platforms_n >= 1,
            "href": "platforms.html",
        },
        {
            "id": "publish",
            "label": "Publish 3+ videos with AI captions",
            "done": samples_30d >= 3,
            "href": "upload.html",
            "progress": min(samples_30d, 3),
            "target": 3,
        },
        {
            "id": "sync",
            "label": "Sync engagement on Analytics",
            "done": samples_30d >= 1,
            "href": "analytics.html",
        },
        {
            "id": "catalog",
            "label": "Build channel memory (Vision entities)",
            "done": has_catalog,
            "href": "upload.html",
        },
        {
            "id": "combo",
            "label": "Unlock exact packaging combo (8+ scored posts)",
            "done": samples_30d >= 8 and ranked_n >= 1,
            "href": "upload.html",
            "progress": min(samples_30d, 8),
            "target": 8,
        },
    ]
    done_n = sum(1 for s in steps if s.get("done"))
    return {
        "samples_30d": samples_30d,
        "samples_to_ready": max(0, 8 - samples_30d),
        "steps": steps,
        "completed": done_n,
        "total": len(steps),
        "pct": int(round(100.0 * done_n / max(len(steps), 1))),
    }


def ai_insights_hub_fallback(*, error: Optional[str] = None) -> Dict[str, Any]:
    """Degraded hub when DB is unavailable or a gather task fails."""
    return sanitize_coach_payload_for_json(
        {
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
            "smart_offer": None,
            "unlock_progress": _unlock_progress(0, 0, 0, False),
            "playbook": [],
        }
    )


async def build_ai_insights_hub(pool: Any, user_id, user: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
            prefs = coerce_jsonb_dict(row.get("preferences"))
            from services.thumbnail_studio_strategy import read_thumbnail_studio_default_strategy

            nested = read_thumbnail_studio_default_strategy(prefs)
            if nested.get("audience_niche"):
                category = normalize_niche(str(nested["audience_niche"]))
        return await fetch_channel_catalog_detail(pool, user_id=str(uid), category=category, limit_per_bucket=12)

    async def _studio(c):
        usage = await fetch_user_pikzels_studio_usage(c, str(uid), since, until)
        try:
            from services.pikzels_analyzer import fetch_analyzer_summary

            analyzer = await fetch_analyzer_summary(c, user_id=str(uid), days=90)
        except Exception:
            analyzer = {}
        return {**usage, "analyzer": analyzer or {}}

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

    catalog_entities = int((catalog or {}).get("entity_count") or 0) if isinstance(catalog, dict) else 0
    unlock = _unlock_progress(
        samples,
        len(ranked),
        len(platforms or []),
        catalog_entities > 0,
    )

    playbook = [
        {"id": "analytics", "label": "Analytics", "href": "analytics.html", "icon": "fas fa-chart-line", "hint": "Views, likes, comments, shares by upload"},
        {"id": "kpi", "label": "Upload KPIs", "href": "kpi.html", "icon": "fas fa-chart-bar", "hint": "Throughput and success rates"},
        {"id": "studio", "label": "Thumbnail Studio", "href": "thumbnail-studio.html", "icon": "fas fa-images", "hint": "Personas, variants, and AI thumbnails"},
        {"id": "upload", "label": "Upload", "href": "upload.html", "icon": "fas fa-cloud-upload-alt", "hint": "Ship with your optimized defaults"},
        {"id": "settings", "label": "AI settings", "href": "settings.html#preferences", "icon": "fas fa-sliders-h", "hint": "Caption tone, hashtags, personas"},
        {"id": "platforms", "label": "Connected accounts", "href": "platforms.html", "icon": "fas fa-plug", "hint": "OAuth health per platform"},
        {"id": "scheduled", "label": "Scheduled", "href": "scheduled.html", "icon": "fas fa-calendar-alt", "hint": "Rhythm and peak windows"},
        {"id": "billing", "label": "Wallet & plans", "href": "billing.html", "icon": "fas fa-wallet", "hint": "PUT/AIC balance for optimized volume"},
    ]

    out: Dict[str, Any] = {
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
        "smart_offer": (coach or {}).get("smart_offer"),
        "unlock_progress": unlock,
        "content_rankings": fetch_content_success_rankings(),
        "playbook": playbook,
    }
    role = str((user or {}).get("role") or "").strip().lower()
    if role in ("admin", "master_admin"):
        hub_urls = get_ml_hub_urls()
        hf = ml_hub_huggingface_dict()
        out["ml_hub"] = {
            "dataset_repo": hub_urls.get("dataset_repo"),
            "dataset_url": hub_urls.get("dataset_url"),
            "trackio_space_url": hub_urls.get("trackio_space_url"),
            "hf_sync_enabled": os.environ.get("UM8_HF_SYNC_VISUAL_ENTITIES", "").strip().lower()
            in ("1", "true", "yes"),
            "docs": {
                "datasets": hf.get("datasets_hub"),
                "trainer": hf.get("trl_docs"),
                "jobs": hf.get("hub_docs_jobs"),
                "evaluation": hf.get("evaluation_doc"),
            },
        }
    return sanitize_coach_payload_for_json(out)
