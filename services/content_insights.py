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


def _is_user_recommendable_strategy_key(strategy_key: str) -> bool:
    """Only fully attributed keys can produce useful user-facing recommendations."""
    return bool(parse_content_attribution_key(strategy_key or ""))


CAPTION_INSIGHT_PLATFORMS = frozenset({"tiktok", "youtube", "instagram", "facebook"})

_FACTOR_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("caption_style", "captionStyle", "caption style"),
    ("caption_tone", "captionTone", "tone"),
    ("caption_voice", "captionVoice", "voice"),
)


def _aggregate_factor_performance_from_rows(
    rows: List[Any],
    *,
    lookback_days: int,
    min_samples_per_value: int,
    platform: str,
) -> Dict[str, Any]:
    """Turn strategy_key rollup rows into per-factor ranked lists."""
    min_samples_per_value = max(2, min(int(min_samples_per_value or 4), 50))
    acc: Dict[str, Dict[str, Dict[str, float]]] = {
        f[0]: defaultdict(lambda: {"samples": 0.0, "eng_weight": 0.0, "view_weight": 0.0, "setups": 0.0, "ci": 0.0})
        for f in _FACTOR_FIELDS
    }
    total_samples = 0
    for r in rows or []:
        sk = str(r["strategy_key"] or "")
        parsed = parse_content_attribution_key(sk)
        if not parsed:
            continue
        s = float(int(r["samples"] or 0))
        ew = _finite(r["eng_weight"])
        vw = _finite(r["view_weight"])
        ci = _finite(r["max_ci95_high"])
        total_samples += int(s)
        for field, _camel, _label in _FACTOR_FIELDS:
            val = parsed.get(field)
            if not val:
                continue
            a = acc[field][str(val)]
            a["samples"] += s
            a["eng_weight"] += ew
            a["view_weight"] += vw
            a["setups"] += 1.0
            a["ci"] = max(a["ci"], ci)

    out: Dict[str, Any] = {
        "lookback_days": int(max(30, min(int(lookback_days or 120), 730))),
        "platform": platform,
        "total_samples": total_samples,
    }
    for field, _camel, _label in _FACTOR_FIELDS:
        ranked: List[Dict[str, Any]] = []
        bucket = acc[field]
        tot_s = sum(a["samples"] for a in bucket.values())
        tot_ew = sum(a["eng_weight"] for a in bucket.values())
        factor_avg = (tot_ew / tot_s) if tot_s > 0 else 0.0
        for val, a in bucket.items():
            if a["samples"] < min_samples_per_value:
                continue
            wm = (a["eng_weight"] / a["samples"]) if a["samples"] > 0 else 0.0
            wv = (a["view_weight"] / a["samples"]) if a["samples"] > 0 else 0.0
            lift = wm - factor_avg
            lift_pct = ((wm / factor_avg - 1.0) * 100.0) if factor_avg > 0 else 0.0
            ranked.append(
                {
                    "value": val,
                    "samples": int(a["samples"]),
                    "distinct_setups": int(a["setups"]),
                    "weighted_mean_engagement_pct": round(wm, 4),
                    "weighted_mean_views": round(wv, 1),
                    "ci95_high_pct": round(a["ci"], 4),
                    "lift_vs_factor_avg_pts": round(lift, 4),
                    "lift_vs_factor_avg_pct": round(lift_pct, 2),
                }
            )
        ranked.sort(key=lambda x: (-x["weighted_mean_engagement_pct"], -x["samples"], x["value"]))
        out[field] = {
            "factor_avg_engagement_pct": round(factor_avg, 4),
            "values_with_signal": len(ranked),
            "ranked": ranked,
        }
    return out


async def fetch_factor_performance(
    conn,
    user_id: uuid.UUID,
    lookback_days: int = 120,
    min_samples_per_value: int = 4,
    platform: str = "all",
) -> Dict[str, Any]:
    """Sample-weighted engagement per individual caption factor value."""
    lookback = max(30, min(int(lookback_days or 120), 730))
    plat = (platform or "all").lower().strip()
    rows = await conn.fetch(
        """
        SELECT strategy_key,
               SUM(samples)::bigint AS samples,
               SUM(mean_engagement * samples::double precision) AS eng_weight,
               SUM(mean_views * samples::double precision) AS view_weight,
               MAX(ci95_high)::double precision AS max_ci95_high
          FROM upload_quality_scores_daily
         WHERE user_id = $1::uuid
           AND day >= (CURRENT_DATE - $2::int)
           AND strategy_key LIKE 'v1|%'
           AND platform = $3::varchar
         GROUP BY strategy_key
        """,
        user_id,
        lookback,
        plat,
    )
    return _aggregate_factor_performance_from_rows(
        list(rows or []),
        lookback_days=lookback,
        min_samples_per_value=min_samples_per_value,
        platform=plat,
    )


def synthesize_meta_setup(
    factor_perf: Dict[str, Any],
    *,
    min_factors: int = 2,
    require_positive_lift: bool = True,
    platform_label: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Compose a recommended meta setup from the best individual factor values."""
    if not isinstance(factor_perf, dict):
        return None
    patch: Dict[str, Any] = {}
    chosen: List[Dict[str, Any]] = []
    summary_bits: List[str] = []
    confidence_samples = 0
    for field, camel, label in _FACTOR_FIELDS:
        node = factor_perf.get(field) or {}
        ranked = node.get("ranked") or []
        if not ranked:
            continue
        top = ranked[0]
        if require_positive_lift and len(ranked) > 1 and top.get("lift_vs_factor_avg_pts", 0.0) <= 0.0:
            continue
        patch[camel] = top["value"]
        confidence_samples += int(top.get("samples") or 0)
        chosen.append(
            {
                "factor": field,
                "label": label,
                "value": top["value"],
                "weighted_mean_engagement_pct": top.get("weighted_mean_engagement_pct"),
                "samples": top.get("samples"),
                "lift_vs_factor_avg_pct": top.get("lift_vs_factor_avg_pct"),
            }
        )
        lift = top.get("lift_vs_factor_avg_pct") or 0.0
        lift_txt = f" (+{lift:.0f}% vs your avg)" if lift > 0 else ""
        summary_bits.append(f"{label} “{top['value']}”{lift_txt}")

    if len(patch) < max(1, int(min_factors)):
        return None

    plat = (platform_label or factor_perf.get("platform") or "").strip()
    plat_prefix = f"On {plat.title()}: " if plat and plat != "all" else ""
    scope = f"for {plat} " if plat and plat != "all" else "across your uploads "

    return {
        "strategy_key": None,
        "source": "factor_synthesis",
        "platform": plat or "all",
        "summary": plat_prefix + "Synthesized from your best-performing factors: " + ", ".join(summary_bits),
        "confidence_note": (
            f"Composed from {confidence_samples} scored day-buckets {scope}"
            "by learning each caption factor independently (so we can recommend before any "
            "single exact combo has enough repeats)."
        ),
        "factors": chosen,
        "preferences_patch": patch,
    }


async def fetch_meta_setups_by_platform(
    conn,
    user_id: uuid.UUID,
    lookback_days: int = 120,
    min_platform_total_samples: int = 8,
    min_samples_per_value: int = 3,
) -> Dict[str, Any]:
    """Synthesized caption meta setup per platform (style + tone + voice winners)."""
    lookback = max(30, min(int(lookback_days or 120), 730))
    min_platform_total_samples = max(4, min(int(min_platform_total_samples or 8), 500))
    plat_rows = await conn.fetch(
        """
        SELECT platform, SUM(samples)::bigint AS total_samples
          FROM upload_quality_scores_daily
         WHERE user_id = $1::uuid
           AND day >= (CURRENT_DATE - $2::int)
           AND strategy_key LIKE 'v1|%'
           AND platform <> 'all'
         GROUP BY platform
        HAVING SUM(samples) >= $3::int
         ORDER BY total_samples DESC
        """,
        user_id,
        lookback,
        min_platform_total_samples,
    )
    platforms: Dict[str, Any] = {}
    for r in plat_rows or []:
        pl = str(r["platform"] or "").lower().strip()
        if pl not in CAPTION_INSIGHT_PLATFORMS:
            continue
        try:
            fp = await fetch_factor_performance(
                conn,
                user_id,
                lookback_days=lookback,
                min_samples_per_value=min_samples_per_value,
                platform=pl,
            )
            meta = synthesize_meta_setup(fp, min_factors=2)
            if meta:
                meta["platform"] = pl
                meta["platform_total_samples"] = int(r["total_samples"] or 0)
                platforms[pl] = meta
        except Exception as e:
            logger.debug("fetch_meta_setups_by_platform %s: %s", pl, e)
    return {"lookback_days": lookback, "platforms": platforms}


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
           AND strategy_key LIKE 'v1|%'
           AND platform = 'all'
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
        if not _is_user_recommendable_strategy_key(sk):
            continue
        parsed = parse_content_attribution_key(sk)
        out.append(
            {
                "strategy_key": sk,
                "samples": int(r["samples"] or 0),
                "weighted_mean_engagement_pct": _finite(r["weighted_mean_engagement"]),
                "ci95_high_pct": _finite(r["max_ci95_high"]),
                "weighted_mean_views": _finite(r["weighted_mean_views"]),
                "summary": _human_strategy_summary(sk),
                "parsed": parsed,
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
            "factor_performance": {},
            "meta_setup": None,
            "meta_setup_by_platform": {"lookback_days": 0, "platforms": {}},
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

    factor_performance: Dict[str, Any] = {}
    try:
        factor_performance = await fetch_factor_performance(conn, uid, lookback_days=120)
    except Exception as e:
        logger.warning("fetch_factor_performance: %s", e)

    meta_setup_synth: Optional[Dict[str, Any]] = None
    try:
        meta_setup_synth = synthesize_meta_setup(factor_performance)
    except Exception as e:
        logger.debug("synthesize_meta_setup: %s", e)

    meta_setup_by_platform: Dict[str, Any] = {"lookback_days": 0, "platforms": {}}
    try:
        meta_setup_by_platform = await fetch_meta_setups_by_platform(conn, uid, lookback_days=120)
    except Exception as e:
        logger.warning("fetch_meta_setups_by_platform: %s", e)

    recommended = None
    narrative_parts: List[str] = []
    if ranked:
        top = ranked[0]
        recommended = {
            "strategy_key": top["strategy_key"],
            "source": "exact_combo",
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
    elif meta_setup_synth:
        recommended = dict(meta_setup_synth)
        narrative_parts.append(
            "No single exact packaging combo has enough repeats yet, but factor-by-factor "
            f"your numbers favor: {', '.join(f['label'] + ' “' + str(f['value']) + '”' for f in meta_setup_synth.get('factors') or [])}. "
            "Lock these in to converge on your meta setup faster."
        )
    else:
        narrative_parts.append(
            "Not enough attributed uploads yet — publish a few more videos with AI captions enabled "
            "so we can compare which styles resonate."
        )

    for field, _camel, label in _FACTOR_FIELDS:
        node = (factor_performance or {}).get(field) or {}
        rk = node.get("ranked") or []
        if len(rk) >= 2 and rk[0].get("lift_vs_factor_avg_pct", 0.0) >= 10.0:
            narrative_parts.append(
                f"Best {label}: “{rk[0]['value']}” (~{rk[0]['weighted_mean_engagement_pct']:.2f}% engagement, "
                f"+{rk[0]['lift_vs_factor_avg_pct']:.0f}% vs your average across {rk[0]['samples']} samples)."
            )

    plat_metas = (meta_setup_by_platform or {}).get("platforms") or {}
    if len(plat_metas) >= 2:
        bits = []
        for pl, meta in sorted(plat_metas.items(), key=lambda x: -int(x[1].get("platform_total_samples") or 0)):
            patch = meta.get("preferences_patch") or {}
            style = patch.get("captionStyle") or "—"
            tone = patch.get("captionTone") or "—"
            voice = patch.get("captionVoice") or "—"
            bits.append(f"{pl}: {style} + {tone} + {voice}")
        narrative_parts.append(
            "Per-platform meta setups differ — your winning caption mix is not one-size-fits-all: "
            + "; ".join(bits[:4])
            + ". Apply a platform-specific setup from Smart Insights when you mostly post there."
        )
    elif len(plat_metas) == 1:
        pl, meta = next(iter(plat_metas.items()))
        narrative_parts.append(
            f"Strongest platform-specific signal on {pl}: {meta.get('summary') or 'see per-platform card below'}."
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

    # Accuracy ladder P3: surface low caption–evidence grounding when rollup exists.
    try:
        g_avg = await conn.fetchval(
            """
            SELECT AVG(mean_grounding)::double precision
              FROM upload_quality_scores_daily
             WHERE user_id = $1
               AND platform = 'all'
               AND mean_grounding IS NOT NULL
               AND day >= (CURRENT_DATE - INTERVAL '60 days')
            """,
            uid,
        )
        if g_avg is not None and float(g_avg) < 0.35:
            narrative_parts.append(
                f"Caption grounding vs video evidence is low lately (~{float(g_avg):.0%}) — "
                "enable Speech-to-Text (no extra AIC) and Frame Inspector so titles mention "
                "what is actually in the clip."
            )
        elif g_avg is not None and float(g_avg) >= 0.55:
            narrative_parts.append(
                f"Your captions are staying well grounded in clip evidence (~{float(g_avg):.0%} match)."
            )
    except Exception:
        pass

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
        "factor_performance": factor_performance,
        "meta_setup": meta_setup_synth,
        "meta_setup_by_platform": meta_setup_by_platform,
        "hashtag_traction": hashtag_traction,
        "anomaly": anomaly,
        "recommended": recommended,
        "narrative": " ".join(n for n in narrative_parts if n).strip(),
    }



def resolve_insights_recommendation(
    insights: Dict[str, Any],
    *,
    strategy_key_override: Optional[str] = None,
    platform: Optional[str] = None,
) -> Dict[str, Any]:
    """Pick the recommendation row to apply (global, per-platform synthesis, or explicit key)."""
    if strategy_key_override:
        parsed = parse_content_attribution_key(strategy_key_override.strip())
        if not parsed:
            raise ValueError("Could not parse strategy_key")
        return {
            "strategy_key": strategy_key_override.strip(),
            "source": "strategy_key_override",
            "preferences_patch": preferences_patch_from_parsed_attribution(parsed),
        }
    pl = (platform or "").lower().strip()
    if pl:
        if pl not in CAPTION_INSIGHT_PLATFORMS:
            raise ValueError(f"Unsupported platform '{pl}'")
        by_plat = (insights.get("meta_setup_by_platform") or {}).get("platforms") or {}
        rec = by_plat.get(pl)
        if not isinstance(rec, dict) or not rec.get("preferences_patch"):
            raise ValueError(
                f"Not enough {pl} uploads yet to synthesize a platform meta setup — "
                "publish a few more attributed posts to that platform."
            )
        return rec
    rec = insights.get("recommended") if isinstance(insights, dict) else None
    if not isinstance(rec, dict) or not rec.get("preferences_patch"):
        raise ValueError("No recommendation available to apply")
    return rec


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


PREF_FIELD_LABELS: Dict[str, str] = {
    "captionStyle": "Caption style",
    "captionTone": "Caption tone",
    "captionVoice": "Caption voice / persona",
    "captionFrameCount": "Caption frame count",
    "styledThumbnails": "Styled thumbnails",
    "autoCaptions": "Auto captions",
    "autoThumbnails": "Auto thumbnails",
    "aiHashtagsEnabled": "AI hashtags",
    "aiHashtagCount": "AI hashtag count",
    "aiHashtagStyle": "AI hashtag style",
    "thumbnailSelectionMode": "Thumbnail frame selection",
    "thumbnailRenderPipeline": "Thumbnail render pipeline",
    "thumbnailDefaultPersonaId": "Default thumbnail persona",
    "alwaysHashtags": "Always-include hashtags",
}

_VALID_PIPELINES = frozenset({"auto", "studio_renderer", "ai_edit", "template", "none"})
_VALID_SELECTION = frozenset({"ai", "sharpness"})


def _pref_display_value(key: str, val: Any) -> str:
    if val is None:
        return "—"
    if key == "alwaysHashtags" and isinstance(val, list):
        return ", ".join(str(x) for x in val[:8]) or "—"
    if isinstance(val, bool):
        return "On" if val else "Off"
    return str(val)


def _current_pref_value(prefs: Dict[str, Any], key: str) -> Any:
    from services.thumbnail_studio_strategy import read_thumbnail_studio_default_strategy

    nested = read_thumbnail_studio_default_strategy(prefs)
    snake_map = {
        "captionStyle": "caption_style",
        "captionTone": "caption_tone",
        "captionVoice": "caption_voice",
        "thumbnailSelectionMode": "thumbnail_selection_mode",
        "thumbnailRenderPipeline": "thumbnail_render_pipeline",
        "thumbnailDefaultPersonaId": "thumbnail_default_persona_id",
    }
    if key in prefs:
        return prefs[key]
    snake = snake_map.get(key)
    if snake and snake in prefs:
        return prefs[snake]
    if key in nested:
        return nested[key]
    if snake and snake in nested:
        return nested[snake]
    return None


def _normalize_pipeline_value(raw: str) -> Optional[str]:
    v = str(raw or "").strip().lower()
    if v in _VALID_PIPELINES:
        return v
    if v in ("studio", "renderer", "studio_renderer"):
        return "studio_renderer"
    return None


def _normalize_selection_value(raw: str) -> Optional[str]:
    v = str(raw or "").strip().lower()
    if v in _VALID_SELECTION:
        return v
    if v in ("sharp", "sharpness", "sharpest"):
        return "sharpness"
    if v in ("ai", "auto", "smart"):
        return "ai"
    return None


async def _fetch_user_preferences_raw(conn, user_id: uuid.UUID) -> Dict[str, Any]:
    row = await conn.fetchrow("SELECT preferences FROM users WHERE id = $1::uuid", user_id)
    if not row:
        return {}
    raw = row.get("preferences") or {}
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, TypeError, ValueError):
            return {}
    return dict(raw) if isinstance(raw, dict) else {}


async def _lookup_persona_id(conn, user_id: uuid.UUID, name_or_voice: str) -> Optional[str]:
    needle = str(name_or_voice or "").strip()
    if not needle or len(needle) > 120:
        return None
    row = await conn.fetchrow(
        """
        SELECT id::text FROM creator_personas
         WHERE user_id = $1::uuid
           AND (LOWER(name) = LOWER($2) OR LOWER(COALESCE(profile_json->>'voice', '')) = LOWER($2))
         ORDER BY created_at DESC
         LIMIT 1
        """,
        user_id,
        needle,
    )
    return str(row["id"]) if row else None


def _packaging_dimension_patches(
    packaging: Dict[str, Any],
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Map packaging rollup winners to preference keys + rationale rows."""
    patch: Dict[str, Any] = {}
    notes: List[Dict[str, Any]] = []
    by_dim = (packaging or {}).get("by_dimension") or {}

    dim_map = (
        ("pipeline", "thumbnailRenderPipeline", _normalize_pipeline_value, "packaging_ml"),
        ("selection", "thumbnailSelectionMode", _normalize_selection_value, "packaging_ml"),
    )
    for dim, pref_key, normalizer, source in dim_map:
        rows = by_dim.get(dim) or []
        if not rows:
            continue
        top = rows[0]
        norm = normalizer(str(top.get("value") or ""))
        if not norm:
            continue
        patch[pref_key] = norm
        notes.append(
            {
                "field": pref_key,
                "source": source,
                "rationale": (
                    f"Packaging ML ranked “{top.get('value')}” as your top {dim} "
                    f"(~{float(top.get('mean_engagement_pct') or 0):.2f}% engagement across "
                    f"{int(top.get('uploads') or 0)} attributed uploads)."
                ),
                "stats": {
                    "mean_engagement_pct": float(top.get("mean_engagement_pct") or 0),
                    "uploads": int(top.get("uploads") or 0),
                    "dimension": dim,
                },
            }
        )

    persona_rows = by_dim.get("persona") or []
    if persona_rows:
        top_p = persona_rows[0]
        pname = str(top_p.get("value") or "").strip()
        if pname and pname not in ("—", "general", ""):
            patch["_persona_name_hint"] = pname
            notes.append(
                {
                    "field": "captionVoice",
                    "source": "packaging_ml",
                    "rationale": (
                        f"Persona/voice “{pname}” led packaging engagement "
                        f"(~{float(top_p.get('mean_engagement_pct') or 0):.2f}% across "
                        f"{int(top_p.get('uploads') or 0)} uploads)."
                    ),
                    "stats": {
                        "mean_engagement_pct": float(top_p.get("mean_engagement_pct") or 0),
                        "uploads": int(top_p.get("uploads") or 0),
                        "dimension": "persona",
                    },
                }
            )

    combos = (packaging or {}).get("combos") or []
    if combos:
        top_combo = combos[0]
        parts = top_combo.get("parts") or {}
        notes.insert(
            0,
            {
                "field": "_combo",
                "source": "packaging_ml",
                "rationale": (
                    f"Best template/persona combo: {top_combo.get('label') or 'top packaging'} "
                    f"(~{float(top_combo.get('mean_engagement_pct') or 0):.2f}% engagement, "
                    f"{int(top_combo.get('uploads') or 0)} uploads)."
                ),
                "stats": {
                    "mean_engagement_pct": float(top_combo.get("mean_engagement_pct") or 0),
                    "uploads": int(top_combo.get("uploads") or 0),
                    "parts": parts,
                },
            },
        )
        for dim, pref_key, normalizer, _src in dim_map:
            if pref_key in patch:
                continue
            norm = normalizer(str(parts.get(dim) or ""))
            if norm:
                patch[pref_key] = norm

    return patch, notes


async def build_optimize_settings_plan(conn, user_id) -> Dict[str, Any]:
    """
    Stat-backed settings optimization: merges content attribution, factor synthesis,
    packaging rollups, and hashtag traction into a previewable patch + change log.
    """
    try:
        uid = user_id if isinstance(user_id, uuid.UUID) else uuid.UUID(str(user_id))
    except (ValueError, TypeError):
        return {"ok": False, "message": "Invalid user", "changes": [], "patch": {}}

    from services.ai_insights_hub import fetch_packaging_variant_rollups

    insights = await build_user_content_insights(conn, uid)
    packaging = await fetch_packaging_variant_rollups(conn, uid, days=120)
    current = await _fetch_user_preferences_raw(conn, uid)

    patch: Dict[str, Any] = {}
    rationale_notes: List[Dict[str, Any]] = []
    ml_sources: List[str] = []

    try:
        rec = resolve_insights_recommendation(insights if isinstance(insights, dict) else {})
        base_patch = dict(rec.get("preferences_patch") or {})
        patch.update(base_patch)
        src = str(rec.get("source") or "content_attribution")
        ml_sources.append(src)
        if rec.get("summary"):
            rationale_notes.append(
                {
                    "field": "_strategy",
                    "source": src,
                    "rationale": str(rec.get("summary")),
                    "stats": {"confidence_note": rec.get("confidence_note")},
                }
            )
        for field, camel, label in _FACTOR_FIELDS:
            if camel not in base_patch:
                continue
            node = (insights.get("factor_performance") or {}).get(field) or {}
            rk = node.get("ranked") or []
            top = rk[0] if rk else None
            if top:
                rationale_notes.append(
                    {
                        "field": camel,
                        "source": "factor_ml",
                        "rationale": (
                            f"Content attribution ML: best {label} “{top.get('value')}” "
                            f"(~{float(top.get('weighted_mean_engagement_pct') or 0):.2f}% engagement, "
                            f"{int(top.get('samples') or 0)} scored samples"
                            + (
                                f", +{float(top.get('lift_vs_factor_avg_pct') or 0):.0f}% vs your {label} average"
                                if top.get("lift_vs_factor_avg_pct")
                                else ""
                            )
                            + ")."
                        ),
                        "stats": dict(top),
                    }
                )
    except ValueError:
        pass

    pack_patch, pack_notes = _packaging_dimension_patches(packaging)
    for k, v in pack_patch.items():
        if k.startswith("_"):
            continue
        if k not in patch:
            patch[k] = v
    rationale_notes.extend(pack_notes)
    if pack_notes:
        ml_sources.append("packaging_ml")

    persona_hint = patch.pop("_persona_name_hint", None) or pack_patch.get("_persona_name_hint")
    if persona_hint:
        pid = await _lookup_persona_id(conn, uid, persona_hint)
        if pid:
            patch["thumbnailDefaultPersonaId"] = pid
        elif persona_hint and "captionVoice" not in patch:
            patch["captionVoice"] = persona_hint

    ht = (insights.get("hashtag_traction") or {}) if isinstance(insights, dict) else {}
    top_ht = ht.get("top_by_engagement") or []
    if top_ht and not patch.get("aiHashtagsEnabled"):
        patch["aiHashtagsEnabled"] = True
        ml_sources.append("hashtag_ml")
        rationale_notes.append(
            {
                "field": "aiHashtagsEnabled",
                "source": "hashtag_ml",
                "rationale": (
                    "Hashtag traction ML found tags that outperform your baseline — enabling AI hashtags "
                    "so new posts can discover similar winners. Top: "
                    + ", ".join(
                        f"#{x['hashtag']} (~{float(x.get('mean_engagement_pct') or 0):.2f}%)"
                        for x in top_ht[:3]
                    )
                    + "."
                ),
                "stats": {"top_tags": top_ht[:5]},
            }
        )

    if not patch:
        return {
            "ok": False,
            "message": (
                insights.get("narrative")
                if isinstance(insights, dict) and insights.get("narrative")
                else "Not enough attributed uploads yet — publish a few more with AI captions on."
            ),
            "changes": [],
            "patch": {},
            "narrative": insights.get("narrative") if isinstance(insights, dict) else "",
            "ml_sources": [],
            "readiness": "building",
        }

    changes: List[Dict[str, Any]] = []
    seen_fields: set[str] = set()
    for note in rationale_notes:
        field = note.get("field")
        if not field or field.startswith("_") or field in seen_fields:
            continue
        if field not in patch:
            continue
        before = _current_pref_value(current, field)
        after = patch[field]
        if before == after:
            continue
        seen_fields.add(field)
        changes.append(
            {
                "field": field,
                "label": PREF_FIELD_LABELS.get(field, field),
                "before": _pref_display_value(field, before),
                "after": _pref_display_value(field, after),
                "source": note.get("source") or "ml",
                "rationale": note.get("rationale") or "",
                "stats": note.get("stats") or {},
            }
        )

    for field, after in patch.items():
        if field in seen_fields or field.startswith("_"):
            continue
        before = _current_pref_value(current, field)
        if before == after:
            continue
        changes.append(
            {
                "field": field,
                "label": PREF_FIELD_LABELS.get(field, field),
                "before": _pref_display_value(field, before),
                "after": _pref_display_value(field, after),
                "source": "content_attribution",
                "rationale": "Recommended from your top-performing packaging attribution bucket.",
                "stats": {},
            }
        )

    readiness = "ready" if len(changes) >= 2 else ("emerging" if changes else "building")
    summary_bits = [c["label"] + ": " + c["after"] for c in changes[:6]]

    return {
        "ok": bool(changes),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "readiness": readiness,
        "message": (
            f"{len(changes)} setting(s) can be optimized from your upload ML signals."
            if changes
            else "Your settings already match the top ML recommendations."
        ),
        "summary": "; ".join(summary_bits) if summary_bits else "",
        "narrative": insights.get("narrative") if isinstance(insights, dict) else "",
        "ml_sources": list(dict.fromkeys(ml_sources)),
        "patch": patch,
        "changes": changes,
        "packaging_samples": int(packaging.get("uploads_with_attribution") or 0),
        "attribution_strategies": len((insights.get("ranked_strategies") or []) if isinstance(insights, dict) else []),
    }
