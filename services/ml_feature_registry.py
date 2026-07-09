"""
Single source of truth for UploadM8 ML feature schemas.

Both learning loops (``promo`` targeting uplift and ``content`` success / hottest
topic) declare every column they consume here: its source, dtype, role, and
lifecycle status. The feature modules, training scripts, runtime scoring, and the
admin observability surface all derive their column lists from this registry so
they can never drift.

Roles:
- ``num``  : numeric model feature
- ``cat``  : categorical model feature
- ``label``: supervised target (primary)
- ``label_fallback``: alternate target used when the primary has no class variance
- ``id``   : row identifier (never a feature)
- ``meta`` : carried for bookkeeping / debugging (never a feature)

Status:
- ``active``       : included in training + scoring now
- ``experimental`` : produced by the curated views but not yet a model feature
- ``deprecated``   : kept for back-compat / inspection, excluded from models

Kept dependency-light (stdlib only) so the dependency-light feature modules can
import it without pulling in pandas/sklearn.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Feature:
    name: str
    source: str
    dtype: str
    role: str
    status: str = "active"
    notes: str = ""


# ---------------------------------------------------------------------------
# Promo targeting uplift loop
# ---------------------------------------------------------------------------
PROMO_FEATURES: List[Feature] = [
    Feature("touchpoint_id", "view:v_promo_targeting_features", "string", "id"),
    Feature("user_id", "view:v_promo_targeting_features", "string", "id"),
    Feature("row_source", "view (touchpoint|snapshot)", "string", "meta",
            notes="Which generative process produced the row; never a feature (leakage)."),
    Feature("is_snapshot", "view (derived from row_source)", "int", "num",
            notes="Explicit population indicator instead of leaking channel/delivery_status."),

    # Timing — meaningful only for real touchpoints; NULL on snapshots (no NOW() noise).
    Feature("sent_dow_utc", "mtd.sent_at", "int", "num"),
    Feature("sent_hour_utc", "mtd.sent_at", "int", "num"),

    # Wallet / tier
    Feature("put_balance", "wallets.put_balance", "int", "num"),
    Feature("aic_balance", "wallets.aic_balance", "int", "num"),
    Feature("subscription_tier", "users.subscription_tier", "categorical", "cat"),

    # Activity aggregates (30d)
    Feature("uploads_30d", "uploads agg 30d", "int", "num"),
    Feature("avg_views_30d", "uploads agg 30d", "float", "num"),
    Feature("avg_engagement_pct_30d", "uploads agg 30d", "float", "num"),
    Feature("content_items_30d", "platform_content_items agg 30d", "int", "num", status="deprecated",
            notes="Redundant with uploads_30d; dropped to reduce collinearity."),
    Feature("pci_avg_views_30d", "platform_content_items agg 30d", "float", "num", status="deprecated",
            notes="Redundant with avg_views_30d; dropped to reduce collinearity."),

    # Source-encoding categoricals — leak which query produced the row.
    Feature("channel", "mtd.channel / 'snapshot'", "categorical", "cat", status="deprecated",
            notes="Leakage: encodes row source. Replaced by is_snapshot."),
    Feature("delivery_status", "mtd.status / 'active_user'", "categorical", "cat", status="deprecated",
            notes="Leakage + post-treatment. Replaced by is_snapshot."),

    # Marketing history / tenure / recency / trend
    Feature("prior_touchpoints", "marketing_touchpoint_deliveries count", "int", "num"),
    Feature("opens_all", "marketing_events campaign_email_open", "int", "num"),
    Feature("clicks_all", "marketing_events clicked", "int", "num"),
    Feature("days_since_last_touchpoint", "marketing_touchpoint_deliveries", "float", "num"),
    Feature("account_age_days", "users.created_at", "float", "num"),
    Feature("days_since_last_upload", "uploads max(created_at)", "float", "num"),
    Feature("uploads_trend_30d", "uploads 30d vs prev 30d", "float", "num"),
    Feature("views_trend_30d", "uploads 30d vs prev 30d", "float", "num"),

    # Targets
    Feature("converted_7d", "revenue_tracking / marketing_events", "int", "label"),
    Feature("engaged_7d", "revenue_tracking / marketing_events", "int", "label_fallback"),
    Feature("revenue_7d", "revenue_tracking", "float", "meta"),
]


# ---------------------------------------------------------------------------
# Content success / hottest-topic loop
# ---------------------------------------------------------------------------
CONTENT_FEATURES: List[Feature] = [
    Feature("upload_id", "uploads.id", "string", "id"),
    Feature("user_id", "uploads.user_id", "string", "id"),
    Feature("platform", "platform_results / uploads.platforms", "categorical", "cat"),

    # Upload-flow choices (attribution snapshot) — these are the CHOICES we control.
    Feature("m8_engine", "output_artifacts.content_attribution_v1", "int", "num"),
    Feature("ai_hashtags_enabled", "output_artifacts.content_attribution_v1", "int", "num"),
    Feature("ai_hashtag_count", "output_artifacts.content_attribution_v1", "int", "num"),
    Feature("caption_frame_count", "output_artifacts.content_attribution_v1", "int", "num"),
    Feature("sent_dow_utc", "uploads.created_at", "int", "num"),
    Feature("sent_hour_utc", "uploads.created_at", "int", "num"),
    Feature("is_shortform", "platform", "int", "num"),

    Feature("content_category", "attribution snapshot", "categorical", "cat"),
    Feature("primary_hashtag", "attribution hashtag_slugs_used[0]", "categorical", "cat", status="deprecated",
            notes="High-cardinality + arbitrary; demoted in favor of hashtag_count."),
    Feature("caption_style", "attribution snapshot", "categorical", "cat"),
    Feature("caption_tone", "attribution snapshot", "categorical", "cat"),
    Feature("caption_voice", "attribution snapshot", "categorical", "cat"),
    Feature("hashtag_style", "attribution snapshot", "categorical", "cat"),
    Feature("thumbnail_selection_mode", "attribution snapshot", "categorical", "cat"),
    Feature("thumbnail_render_pipeline", "attribution snapshot", "categorical", "cat"),

    # Thumbnail Studio / Aurora packaging — consumed by content-success training.
    Feature("thumbnail_studio_enabled", "attribution snapshot", "int", "num"),
    Feature("thumbnail_studio_engine_enabled", "attribution snapshot", "int", "num"),
    Feature("thumbnail_persona_enabled", "attribution snapshot", "int", "num"),
    Feature("thumbnail_persona_strength", "attribution snapshot", "int", "num"),
    Feature("studio_variant_ctr_score", "studio_render_report via attribution", "float", "num"),
    Feature("studio_pikzels_main_score", "studio_render_report via attribution", "float", "num"),
    Feature("thumbnail_audience_niche", "studio default strategy", "categorical", "cat"),
    Feature("thumbnail_engine_mode", "thumbnail_render_method", "categorical", "cat"),
    Feature("thumbnail_layout_pattern", "studio default strategy", "categorical", "cat"),

    # Missing-attribution indicator (replaces fake-default categoricals).
    Feature("has_attribution", "output_artifacts presence", "int", "num"),

    # Age-normalized + intrinsic content signals (curated view).
    Feature("age_days", "platform_content_items.published_at/metrics_synced_at", "float", "meta"),
    Feature("views_per_day", "views / age_days", "float", "num",
            notes="Fixes the post-age confound on raw cumulative views."),
    Feature("duration_seconds", "platform_content_items.duration_seconds", "float", "num"),
    Feature("hashtag_count", "uploads.hashtags length", "int", "num"),
    Feature("title_len", "uploads.title length", "int", "num"),
    Feature("caption_len", "uploads.caption length", "int", "num"),
    Feature("has_people", "upload_recognition_summary.has_people", "int", "num"),
    Feature("object_track_count", "upload_recognition_summary", "int", "num"),
    Feature("hydration_score", "upload_recognition_summary.hydration_score", "float", "num"),

    # Engagement label inputs (kept as meta, used to build the label)
    Feature("views", "platform_results engagement", "int", "meta"),
    Feature("engagement_rate_pct", "derived", "float", "meta"),
    Feature("hotness_score", "within/cross-user percentile", "float", "meta"),
    Feature("is_hot", "top-tercile hotness", "int", "label"),
]


REGISTRY: Dict[str, List[Feature]] = {
    "promo": PROMO_FEATURES,
    "content": CONTENT_FEATURES,
}


def _loop(loop: str) -> List[Feature]:
    if loop not in REGISTRY:
        raise KeyError(f"Unknown ML loop: {loop!r} (expected one of {sorted(REGISTRY)})")
    return REGISTRY[loop]


def active_num(loop: str) -> List[str]:
    return [f.name for f in _loop(loop) if f.role == "num" and f.status == "active"]


def active_cat(loop: str) -> List[str]:
    return [f.name for f in _loop(loop) if f.role == "cat" and f.status == "active"]


def label(loop: str) -> str:
    for f in _loop(loop):
        if f.role == "label" and f.status == "active":
            return f.name
    raise KeyError(f"No active label feature for loop {loop!r}")


def label_fallback(loop: str) -> Optional[str]:
    for f in _loop(loop):
        if f.role == "label_fallback" and f.status == "active":
            return f.name
    return None


def catalog(loop: Optional[str] = None) -> List[Dict[str, str]]:
    """Serializable feature catalog for the admin observability surface."""
    loops = [loop] if loop else list(REGISTRY)
    out: List[Dict[str, str]] = []
    for lp in loops:
        for f in _loop(lp):
            out.append(
                {
                    "loop": lp,
                    "name": f.name,
                    "source": f.source,
                    "dtype": f.dtype,
                    "role": f.role,
                    "status": f.status,
                    "notes": f.notes,
                }
            )
    return out
