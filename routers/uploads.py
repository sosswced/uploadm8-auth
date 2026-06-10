"""
Upload routes — backward-compat shim.

Routers are split into uploads_lifecycle, uploads_read, and uploads_analytics.
Register all three in app.py; this module re-exports helpers for legacy imports.
"""

from routers.uploads_analytics import router as analytics_router
from routers.uploads_lifecycle import router as lifecycle_router
from routers.uploads_read import _schedule_thumbnail_repair, router as read_router

__all__ = [
    "analytics_router",
    "lifecycle_router",
    "read_router",
    "_schedule_thumbnail_repair",
]
