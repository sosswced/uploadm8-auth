"""Legacy domain router aggregate.

Historically this re-registered ``app.api_*`` handlers. Endpoints now live on
focused routers included earlier in ``app.py``. :func:`populate_domain_router`
is safe: sub-registrars are no-ops, so this router adds no duplicate paths.

``app.py`` calls ``populate_domain_router()`` then ``include_router(domain_router)``
for backward compatibility with code that expected the aggregate to exist.
"""
from __future__ import annotations

from fastapi import APIRouter

from routers.domain_misc import register_domain_misc_routes
from routers.groups_platforms import register_groups_platforms_routes
from routers.me_settings import register_me_wallet_settings_routes

router = APIRouter()

_POPULATED = False


def populate_domain_router() -> None:
    global _POPULATED
    if _POPULATED:
        return
    _POPULATED = True

    register_me_wallet_settings_routes(router)
    register_groups_platforms_routes(router)
    register_domain_misc_routes(router)


def register_domain_routes_on_app(application) -> None:
    """Backward-compatible: populate routes and attach to *application*."""
    populate_domain_router()
    application.include_router(router)

