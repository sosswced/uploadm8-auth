"""Shim for legacy imports — use :mod:`routers.domain` directly."""
from __future__ import annotations

from routers.domain import populate_domain_router, register_domain_routes_on_app, router as domain_router

__all__ = ["domain_router", "populate_domain_router", "register_domain_routes_on_app"]
