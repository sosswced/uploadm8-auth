"""
Pure (DB-free) feature engineering for the content-success ML loop.

Turns raw ``uploads`` rows into one record per (upload x platform) with per-platform
engagement (views / likes / comments / shares) on the *label* side and the
upload-flow / topic choices on the *feature* side, then derives a within-user
"hotness" label and ranks the best / hottest topics and content.

Kept dependency-light (stdlib + pandas) so it is unit-testable without the heavy
dataset/sklearn stack the training scripts pull in as UV deps.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from services.ml_feature_registry import active_cat, active_num, label

SHORTFORM_PLATFORMS = frozenset({"tiktok", "youtube", "instagram", "facebook"})
SUCCESS_STATUSES = frozenset({"published", "succeeded", "success", "completed", "partial"})

# Upload-flow CHOICES only — engagement metrics are the label and must never be features.
# Derived from the feature registry (single source of truth).
FEATURES_NUM = active_num("content")
FEATURES_CAT = active_cat("content")
TARGET = label("content")


def _as_dict(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return {}
        try:
            d = json.loads(s)
            return dict(d) if isinstance(d, dict) else {}
        except (json.JSONDecodeError, TypeError, ValueError):
            return {}
    if hasattr(raw, "keys"):
        try:
            return dict(raw)
        except Exception:
            return {}
    return {}


def platform_results_list(raw: Any) -> List[Dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return []
        try:
            raw = json.loads(s)
        except json.JSONDecodeError:
            return []
    if isinstance(raw, dict):
        return [
            {"platform": k, **v} if isinstance(v, dict) else {"platform": k}
            for k, v in raw.items()
        ]
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    return []


def _pick_int(d: Dict[str, Any], *keys: str) -> int:
    for k in keys:
        if k in d and d[k] is not None:
            v = d[k]
            if isinstance(v, bool):
                continue
            try:
                return max(0, int(round(float(v))))
            except (TypeError, ValueError):
                continue
    return 0


def entry_successful(e: Dict[str, Any]) -> bool:
    if e.get("success") is True:
        return True
    return str(e.get("status") or "").strip().lower() in SUCCESS_STATUSES


def entry_metrics(e: Dict[str, Any], platform: str) -> Dict[str, int]:
    likes_keys = ("likes", "like_count", "likeCount", "reactions", "reaction_count")
    if platform == "facebook":
        likes_keys = ("reactions", "reaction_count", "likes", "like_count", "likeCount")
    return {
        "views": _pick_int(e, "views", "view_count", "play_count", "playCount", "video_views", "impressions"),
        "likes": _pick_int(e, *likes_keys),
        "comments": _pick_int(e, "comments", "comment_count", "commentCount"),
        "shares": _pick_int(e, "shares", "share_count", "shareCount"),
    }


def attribution_snapshot(output_artifacts: Any) -> Dict[str, Any]:
    """The content_attribution_v1 snapshot carries topic + packaging choices."""
    oa = _as_dict(output_artifacts)
    snap = oa.get("content_attribution_v1")
    if isinstance(snap, str):
        try:
            snap = json.loads(snap)
        except (json.JSONDecodeError, TypeError, ValueError):
            snap = None
    return dict(snap) if isinstance(snap, dict) else {}


def _primary_hashtag(snap: Dict[str, Any], hashtags: Any) -> str:
    tags = snap.get("hashtag_slugs_used")
    if isinstance(tags, list) and tags:
        t = str(tags[0]).strip().lower().lstrip("#")
        if 1 < len(t) < 60:
            return t
    if isinstance(hashtags, (list, tuple)) and hashtags:
        t = str(hashtags[0]).strip().lower().lstrip("#")
        if 1 < len(t) < 60:
            return t
    return "none"


def _bool_int(v: Any) -> int:
    return 1 if bool(v) else 0


def _evidence_density_features(output_artifacts: Any, row: Dict[str, Any]) -> Dict[str, Any]:
    """Experimental P3 features — produced for curated views; not yet active model inputs."""
    oa = _as_dict(output_artifacts)
    hr = _as_dict(oa.get("hydration_report"))
    gs = hr.get("grounding_score")
    if gs is None:
        gsv = _as_dict(oa.get("grounding_score_v1"))
        gs = gsv.get("grounding_score")
    try:
        grounding_score = float(gs) if gs is not None else None
    except (TypeError, ValueError):
        grounding_score = None

    lanes = 0
    if oa.get("place_evidence_v1") or oa.get("place_evidence"):
        lanes += 1
    if oa.get("shot_list_v1"):
        lanes += 1
    if oa.get("m8_claims_v1") or oa.get("m8_evidence_catalog_v1"):
        lanes += 1
    if hr.get("evidence") or hr.get("evidence_present"):
        lanes += 1
    if oa.get("depth_route_v1") or oa.get("multimodal_depth_v1"):
        lanes += 1

    transcript = (
        str(row.get("ai_transcript") or "")
        or str(_as_dict(oa.get("audio_context")).get("transcript") or "")
        or str(hr.get("transcript_phrase") or "")
    )
    return {
        "grounding_score": grounding_score,
        "evidence_lane_count": int(lanes),
        "transcript_chars": len(transcript.strip()),
    }


def _base_features(row: Dict[str, Any]) -> Dict[str, Any]:
    snap = attribution_snapshot(row.get("output_artifacts"))
    created = row.get("created_at")
    # Postgres EXTRACT(DOW): Sun=0..Sat=6 (Python Mon=0..Sun=6)
    sent_dow_pg = ((int(created.weekday()) + 1) % 7) if created is not None else -1
    sent_hour = int(created.hour) if created is not None else -1

    def _cat(key: str) -> Optional[str]:
        # Explicit null when the choice was not captured, so the model can learn
        # "missing" via has_attribution rather than from a plausible fake default.
        v = snap.get(key)
        if v is None or str(v).strip() == "":
            return None
        return str(v).strip().lower()

    out = {
        "content_category": _cat("content_category"),
        # Kept for ranking surfaces only; deprecated as a model feature.
        "primary_hashtag": _primary_hashtag(snap, row.get("hashtags")),
        "caption_style": _cat("caption_style"),
        "caption_tone": _cat("caption_tone"),
        "caption_voice": _cat("caption_voice"),
        "hashtag_style": _cat("hashtag_style"),
        "thumbnail_selection_mode": _cat("thumbnail_selection_mode"),
        "thumbnail_render_pipeline": _cat("thumbnail_render_pipeline"),
        "thumbnail_audience_niche": _cat("thumbnail_audience_niche"),
        "thumbnail_engine_mode": _cat("thumbnail_engine_mode"),
        "thumbnail_layout_pattern": _cat("thumbnail_layout_pattern"),
        "m8_engine": _bool_int(snap.get("m8_engine")),
        "ai_hashtags_enabled": _bool_int(snap.get("ai_hashtags_enabled")),
        "ai_hashtag_count": int(snap.get("ai_hashtag_count") or 0),
        "caption_frame_count": int(snap.get("caption_frame_count") or 0),
        "thumbnail_studio_enabled": _bool_int(snap.get("thumbnail_studio_enabled", True)),
        "thumbnail_studio_engine_enabled": _bool_int(snap.get("thumbnail_studio_engine_enabled", True)),
        "thumbnail_persona_enabled": _bool_int(snap.get("thumbnail_persona_enabled")),
        "thumbnail_persona_strength": int(snap.get("thumbnail_persona_strength") or 0),
        "studio_variant_ctr_score": float(snap["studio_variant_ctr_score"])
        if snap.get("studio_variant_ctr_score") is not None
        else None,
        "studio_pikzels_main_score": float(snap["studio_pikzels_main_score"])
        if snap.get("studio_pikzels_main_score") is not None
        else None,
        "sent_dow_utc": sent_dow_pg,
        "sent_hour_utc": sent_hour,
        "has_attribution": 1 if snap else 0,
    }
    out.update(_evidence_density_features(row.get("output_artifacts"), row))
    return out


def _assemble_row(
    row: Dict[str, Any],
    platform: str,
    metrics: Dict[str, int],
    base_features: Dict[str, Any],
) -> Dict[str, Any]:
    views = max(0, int(metrics["views"]))
    likes = max(0, int(metrics["likes"]))
    comments = max(0, int(metrics["comments"]))
    shares = max(0, int(metrics["shares"]))
    interactions = likes + comments + shares
    engagement_rate = (interactions / float(views) * 100.0) if views > 0 else 0.0
    rec: Dict[str, Any] = {
        "upload_id": row.get("upload_id"),
        "user_id": row.get("user_id"),
        "platform": platform,
        "is_shortform": 1 if platform in SHORTFORM_PLATFORMS else 0,
        "views": views,
        "likes": likes,
        "comments": comments,
        "shares": shares,
        "interactions": interactions,
        "engagement_rate_pct": round(engagement_rate, 4),
    }
    rec.update(base_features)
    return rec


def expand_upload_to_rows(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One record per (upload x platform) with engagement + upload-flow features."""
    base_features = _base_features(row)
    pr = platform_results_list(row.get("platform_results"))
    out: List[Dict[str, Any]] = []
    for e in pr:
        if not entry_successful(e):
            continue
        plat = str(e.get("platform") or "").strip().lower()
        if not plat:
            continue
        m = entry_metrics(e, plat)
        if (m["views"] + m["likes"] + m["comments"] + m["shares"]) <= 0:
            continue
        out.append(_assemble_row(row, plat, m, base_features))

    if not out:
        platforms = [str(p).strip().lower() for p in (row.get("platforms") or []) if str(p).strip()]
        if len(platforms) == 1:
            m = {
                "views": int(row.get("views") or 0),
                "likes": int(row.get("likes") or 0),
                "comments": int(row.get("comments") or 0),
                "shares": int(row.get("shares") or 0),
            }
            if (m["views"] + m["likes"] + m["comments"] + m["shares"]) > 0:
                out.append(_assemble_row(row, platforms[0], m, base_features))
    return out


def build_records(upload_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for row in upload_rows:
        records.extend(expand_upload_to_rows(row))
    return records


def label_hotness(df):  # type: ignore[no-untyped-def]
    """
    Blended percentile of engagement_rate + views → top tercile is hot.

    Uses a within-user percentile when the user has 2+ scored uploads, and falls
    back to a cross-user, per-platform percentile for single-upload users so they
    are not all forced to ``is_hot = 0`` (which previously starved the label of
    positive variance during cold-start).
    """
    import pandas as pd

    if df is None or df.empty:
        for col in ("views_pct", "engagement_pct", "hotness_score", "is_hot"):
            df[col] = []
        return df

    df = df.copy()

    def _wu_rank(group):
        if group.notna().sum() <= 1:
            return pd.Series(float("nan"), index=group.index)
        return group.rank(pct=True, method="average")

    user_counts = df.groupby("user_id")["views"].transform("size")
    use_wu = user_counts >= 2

    wu_views = df.groupby("user_id")["views"].transform(_wu_rank)
    wu_eng = df.groupby("user_id")["engagement_rate_pct"].transform(_wu_rank)
    cu_views = df.groupby("platform")["views"].rank(pct=True, method="average")
    cu_eng = df.groupby("platform")["engagement_rate_pct"].rank(pct=True, method="average")

    df["views_pct"] = wu_views.where(use_wu, cu_views).fillna(0.5)
    df["engagement_pct"] = wu_eng.where(use_wu, cu_eng).fillna(0.5)
    df["hotness_score"] = (0.6 * df["engagement_pct"] + 0.4 * df["views_pct"]).round(5)
    df["is_hot"] = (df["hotness_score"] >= (2.0 / 3.0)).astype(int)
    return df


def _coerce(v: Any) -> Any:
    if isinstance(v, float) and v != v:  # NaN
        return None
    if hasattr(v, "item"):
        try:
            return v.item()
        except Exception:
            return str(v)
    return v


def rank_dimension(df, by: List[str], *, min_samples: int = 3, limit: int = 15) -> List[Dict[str, Any]]:
    """Rank a content dimension by a hotness-weighted engagement score."""
    if df is None or df.empty or not set(by).issubset(df.columns):
        return []
    grp = df.groupby(by, dropna=False)
    agg = grp.agg(
        samples=("engagement_rate_pct", "size"),
        mean_engagement_pct=("engagement_rate_pct", "mean"),
        mean_views=("views", "mean"),
        sum_views=("views", "sum"),
        sum_interactions=("interactions", "sum"),
        hot_rate=("is_hot", "mean"),
    ).reset_index()
    agg = agg[agg["samples"] >= min_samples]
    if agg.empty:
        return []
    eng_norm = agg["mean_engagement_pct"] / (agg["mean_engagement_pct"].max() or 1.0)
    view_norm = agg["mean_views"] / (agg["mean_views"].max() or 1.0)
    agg["hotness_index"] = (0.5 * eng_norm + 0.3 * agg["hot_rate"] + 0.2 * view_norm).round(5)
    agg = agg.sort_values(
        ["hotness_index", "mean_engagement_pct", "samples"], ascending=False
    ).head(limit)
    out: List[Dict[str, Any]] = []
    for _, r in agg.iterrows():
        rec: Dict[str, Any] = {col: _coerce(r[col]) for col in by}
        rec.update(
            {
                "samples": int(r["samples"]),
                "mean_engagement_pct": round(float(r["mean_engagement_pct"]), 4),
                "mean_views": round(float(r["mean_views"]), 1),
                "sum_views": int(r["sum_views"]),
                "sum_interactions": int(r["sum_interactions"]),
                "hot_rate": round(float(r["hot_rate"]), 4),
                "hotness_index": round(float(r["hotness_index"]), 5),
            }
        )
        out.append(rec)
    return out


def content_rankings(df) -> Dict[str, Any]:
    """The product answer: best / hottest topics and content."""
    return {
        "top_topics": rank_dimension(df, ["content_category"], min_samples=3, limit=15),
        "top_hashtags": rank_dimension(df, ["primary_hashtag"], min_samples=3, limit=20),
        "top_platform_topic": rank_dimension(df, ["platform", "content_category"], min_samples=3, limit=20),
        "top_packaging": rank_dimension(df, ["caption_style", "caption_tone", "caption_voice"], min_samples=4, limit=15),
        "top_by_platform": rank_dimension(df, ["platform"], min_samples=2, limit=10),
    }


def build_labeled_dataframe(upload_rows: List[Dict[str, Any]]):  # type: ignore[no-untyped-def]
    """Convenience: raw upload rows → labeled (upload x platform) DataFrame."""
    import pandas as pd

    df = pd.DataFrame.from_records(build_records(upload_rows))
    if df.empty:
        return df
    return label_hotness(df)
