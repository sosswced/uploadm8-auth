"""Backward-compatible registration for Thumbnail Studio routes."""

from __future__ import annotations

from fastapi import APIRouter

from routers.thumbnail_studio_api import router as thumbnail_studio_router


def register_thumbnail_studio_routes(router: APIRouter) -> None:
    """Attach ``/api/thumbnail-studio/*`` to an aggregate router (e.g. domain)."""
    router.include_router(thumbnail_studio_router)
