"""
Canonical upload KPI definitions (Analytics, user KPI page, dashboard, admin).

Successful upload (product / dashboard definition):
  status IN ('completed', 'succeeded', 'partial')

User-facing **headline** engagement is ``services.canonical_engagement.compute_canonical_engagement_rollup``
(deduped per platform + account + video id across ``platform_content_items`` + ``uploads.platform_results``).
``app._compute_upload_engagement_totals`` remains useful for legacy comparisons and digest code.
``merge_upload_and_catalog_engagement`` is legacy max(upload, catalog sum). Admin aggregates may use raw
``SUM(views)`` for speed. **success_rate** uses ``SUCCESSFUL_STATUS_SQL_IN``.

API prose for ``metric_definitions`` JSON lives in ``services.metric_definitions`` (single canonical copy).
"""

from __future__ import annotations

from services.platform_channels import CANONICAL_PUBLISH_PLATFORMS

SUCCESSFUL_UPLOAD_STATUSES: tuple[str, ...] = ("completed", "succeeded", "partial")

# For raw SQL fragments (asyncpg / PostgreSQL).
SUCCESSFUL_STATUS_SQL_IN = "('completed', 'succeeded', 'partial')"

# GET /api/analytics/overview ?platform= — canonical slugs in uploads.platforms[] plus aliases
# (e.g. instagram_reels) via services.platform_channels.resolve_analytics_platform_filter.
ANALYTICS_OVERVIEW_PLATFORMS = CANONICAL_PUBLISH_PLATFORMS
