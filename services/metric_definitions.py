"""
Canonical copy and keys for API `metric_definitions` payloads.

All user-facing and admin metric explanation strings live here so endpoints
stay consistent. Upload *status* predicates and SQL fragments remain in
`services.upload_metrics`.
"""

from __future__ import annotations

from typing import Any, Dict

from services.upload_metrics import SUCCESSFUL_UPLOAD_STATUSES

# ── Shared building blocks (referenced by multiple endpoints) ───────────────

USER_ENGAGEMENT_ROLLUP = (
    "Canonical headline engagement: deduplicated per (platform, account_id, platform_video_id), "
    "aligned with platform_content_items uniqueness. Merges max(metrics) with successful platform_results; "
    "id-less pr rows and jobs with no successful pr use upload-level max(row, pr). "
    "UTC half-open windows on uploads.created_at and pci.published_at when bounded. "
    "Responses include rollup_version and breakdown.compute (latency, health, scope). "
    "platform_metrics_cache is live_aggregate only — not summed into headline "
    "(see kpi_sources; services.canonical_engagement.compute_canonical_engagement_rollup)."
)

NOT_MARKETING_NUDGE_CTR = (
    "In-app nudge CTR uses marketing_events (GET /api/admin/marketing/intel), not upload success_rate."
)

NOT_THUMBNAIL_CTR_SCORE = (
    "Thumbnail ctr_score and variant experiments are separate signals "
    "(thumbnail/studio APIs and GET /api/analytics/quality-scores)."
)

STRATEGY_SCORE_ROWS = (
    "Daily aggregates by strategy_key (samples, mean_engagement, CI). "
    "Internal learning / biasing only — not marketing nudge CTR, not video impression CTR, not thumbnail ctr_score."
)

PLATFORM_FILTER_OVERVIEW = (
    "When filters.platform is not all, upload counts use uploads.platforms[] containing the canonical slug "
    "(tiktok, youtube, instagram, facebook, …). Headline engagement uses the same upload filter; "
    "filters.platform_display names the product (e.g. Instagram Reels). Optional filters.instagram_reels / "
    "facebook_reels (and aliases) add catalog_surface=reels so platform_content_items is limited to reel rows "
    "for that network (services.platform_channels)."
)

REVENUE_SCOPE_NOTE_ANALYTICS_OVERVIEW = (
    "User billing total for the window from revenue_tracking; not filtered by upload platform."
)

GLOBAL_SUM_VIEWS_LIKES_ADMIN = (
    "total_views and total_likes are SUM(uploads.views) / SUM(uploads.likes) for the window across all users. "
    "Per-user GET /api/analytics (and dashboard/overview) use deduped canonical engagement "
    "(services.canonical_engagement.compute_canonical_engagement_rollup); those totals may differ from this raw SUM."
)

MARKETING_FUNNEL_CTR_PCT = (
    "Nudge / in-product CTR = marketing_events.clicked / marketing_events.shown in the selected window. "
    "Not upload pipeline success_rate, not video impression CTR, not thumbnail ctr_score."
)

PROMO_SCHEDULE_CONV_RATE_7D = (
    "Share of hour-bucket clicks that led to revenue_tracking within 7 days — exploratory timing only; not nudge CTR."
)

ADMIN_KPIS_MARKETING_NUDGE_POINTER = (
    "Defined on GET /api/admin/marketing/intel under metric_definitions.marketing_funnel.ctr_pct."
)

ADMIN_KPI_PROVIDER_COSTS_NOTE = (
    "Stripe/Mailgun/R2/Render/Redis/Postgres line items on the Admin KPI page are merged from "
    "GET /api/admin/kpi/provider-costs (live provider APIs + env) when the UI refresh runs; "
    "OpenAI/storage/compute in cost_tracking remain Postgres-sourced."
)

CATALOG_AGGREGATE_USER_ENGAGEMENT = (
    "Tenant-wide SUM over platform_content_items with optional filters (period, platform, source, account). "
    "Per-video metrics use GREATEST(catalog, linked uploads row) when the upload targets a single matching "
    "platform — same spirit as GET /api/catalog/content. "
    "Unlike GET /api/analytics headline, this is not compute_canonical_engagement_rollup "
    "(no merge with successful platform_results JSON dedupe); external-only catalog rows are included. "
    "See engagement_crosswalk on GET /api/analytics and this response."
)


# Optional: bump when changing keys or meaning of definitions (clients may log/compare).
CANONICAL_DEFINITIONS_VERSION = 1


def _base() -> Dict[str, Any]:
    return {"definitions_version": CANONICAL_DEFINITIONS_VERSION}


def engagement_crosswalk() -> Dict[str, Any]:
    """Single map of how headline rollup, catalog aggregate, live snapshot, and admin sums relate."""
    return {
        "canonical_headline": {
            "api": "GET /api/analytics",
            "fields": "views, likes, comments, shares",
            "computation": "services.canonical_engagement.compute_canonical_engagement_rollup",
            "scope": "single user_id",
            "dedup": "(platform, account_id, platform_video_id) via pci + successful platform_results",
            "time_basis": (
                "Half-open UTC window from engagement_time_window_for_analytics_range: "
                "pci rows filtered by published_at; successful uploads filtered by created_at "
                "(then merged/deduped — not the same per-row timestamp as catalog aggregate)."
            ),
        },
        "catalog_aggregate": {
            "api": "GET /api/catalog/aggregate",
            "fields": "views, likes, comments, shares",
            "computation": "services.catalog_sync.get_catalog_aggregate (SQL over pci + uploads join)",
            "scope": "single user_id; optional period/days/start-end, platform, source, account_id",
            "time_basis": (
                "Per-row COALESCE(pci.published_at, u.completed_at, u.created_at) inside the selected "
                "rolling or explicit UTC window — differs from headline filters (see canonical_headline.time_basis)."
            ),
            "vs_canonical": (
                "Different row membership and merge rules than headline; includes external catalog-only rows; "
                "does not apply platform_results JSON dedupe path."
            ),
        },
        "live_aggregate": {
            "api": "GET /api/analytics",
            "field": "live_aggregate",
            "source": "platform_metrics_cache",
            "vs_canonical": "Account poll snapshot only — not summed into headline (see response kpi_sources).",
        },
        "admin_global_upload_sum": {
            "api": "GET /api/admin/kpis",
            "fields": "total_views, total_likes (windowed)",
            "computation": "SUM(uploads.views), SUM(uploads.likes) across all users",
            "time_basis": "uploads.created_at in selected admin range (same raw columns as row totals).",
            "vs_canonical": "No per-user pci/pr dedupe; not catalog breadth.",
        },
    }


def for_get_analytics() -> Dict[str, Any]:
    out = _base()
    out.update(
        {
            "successful_upload_statuses": list(SUCCESSFUL_UPLOAD_STATUSES),
            "user_engagement": USER_ENGAGEMENT_ROLLUP,
            "kpi_sources": (
                "Response field kpi_sources explains headline vs live_aggregate: "
                "headline = DB catalog + uploads JSON; live_aggregate = platform_metrics_cache worker snapshot."
            ),
            "engagement_crosswalk_note": (
                "Top-level field engagement_crosswalk on GET /api/analytics maps headline vs catalog aggregate vs admin sums."
            ),
        }
    )
    return out


def for_quality_scores() -> Dict[str, Any]:
    out = _base()
    out.update({"strategy_rows": STRATEGY_SCORE_ROWS})
    return out


def for_analytics_overview() -> Dict[str, Any]:
    out = _base()
    out.update(
        {
            "successful_upload_statuses": list(SUCCESSFUL_UPLOAD_STATUSES),
            "user_engagement": USER_ENGAGEMENT_ROLLUP,
            "platform_filter": PLATFORM_FILTER_OVERVIEW,
            "not_marketing_nudge_ctr": NOT_MARKETING_NUDGE_CTR,
            "not_thumbnail_ctr_score": NOT_THUMBNAIL_CTR_SCORE,
        }
    )
    return out


def for_catalog_aggregate() -> Dict[str, Any]:
    out = _base()
    out.update(
        {
            "catalog_aggregate_engagement": CATALOG_AGGREGATE_USER_ENGAGEMENT,
            "engagement_crosswalk": engagement_crosswalk(),
        }
    )
    return out


def for_admin_kpis() -> Dict[str, Any]:
    out = _base()
    out.update(
        {
            "successful_upload_statuses": list(SUCCESSFUL_UPLOAD_STATUSES),
            "global_total_views_likes": GLOBAL_SUM_VIEWS_LIKES_ADMIN,
            "marketing_nudge_ctr": ADMIN_KPIS_MARKETING_NUDGE_POINTER,
            "provider_costs_merge": ADMIN_KPI_PROVIDER_COSTS_NOTE,
            "engagement_crosswalk_note": (
                "Top-level field engagement_crosswalk on GET /api/admin/kpis (same object as analytics; "
                "not nested here to avoid duplicating a large blob in metric_definitions and data_provenance)."
            ),
        }
    )
    return out


def admin_kpi_data_provenance(*, rollup_version: int) -> Dict[str, Any]:
    """
    Machine-readable map of where Admin KPI numbers come from vs product analytics rollup.
    """
    return {
        "engagement_crosswalk_note": (
            "Full map is at JSON root key `engagement_crosswalk` on GET /api/admin/kpis."
        ),
        "canonical_engagement_rollup_version": rollup_version,
        "total_views_total_likes": {
            "store": "postgres",
            "relation": "uploads",
            "aggregation": "SUM(views), SUM(likes) over rows with created_at in selected UTC window",
            "not": "Global admin totals are not compute_canonical_engagement_rollup (that is per user_id).",
            "compare_with": (
                "GET /api/analytics per tenant — deduped pci + platform_results; see kpi_sources and root engagement_crosswalk. "
                "GET /api/catalog/aggregate — catalog SQL totals (see that response kpi_sources and metric_definitions)."
            ),
        },
        "uploads_success_queue": {"store": "postgres", "relation": "uploads"},
        "mrr_tier_breakdown": {
            "store": "postgres",
            "relations": ["users"],
            "plan_prices": "stages/entitlements get_plan() — not Stripe live poll in this endpoint",
        },
        "revenue_window_topups": {"store": "postgres", "relation": "revenue_tracking"},
        "cost_categories": {"store": "postgres", "relation": "cost_tracking"},
        "sales_opportunity_levers": {
            "store": "postgres",
            "relations": ["users", "uploads", "wallets", "platform_tokens"],
        },
        "creative_freshness": {"store": "postgres", "relation": "upload_thumbnail_style_memory"},
        "whisper_estimate": {"store": "derived", "note": "Formula on successful_uploads count + env tunables — not OpenAI invoice API in this field"},
        "admin_page_client_merges": [
            "GET /api/admin/marketing/intel — marketing_events + revenue_tracking (funnel, promo schedule)",
            "GET /api/admin/kpi/provider-costs — external provider APIs + env for infra line items",
            "GET /api/admin/kpi/cost-tracker — tool estimates vs upload volume",
            "GET /api/admin/kpi/reliability — processing stats",
            "GET /api/admin/kpi/usage — same uploads SUM views/likes as overlap guard",
        ],
    }


def for_marketing_intel() -> Dict[str, Any]:
    out = _base()
    out.update(
        {
            "marketing_funnel.ctr_pct": MARKETING_FUNNEL_CTR_PCT,
            "promo_schedule_recommendations.conv_rate_7d": PROMO_SCHEDULE_CONV_RATE_7D,
        }
    )
    return out
