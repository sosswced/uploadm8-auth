"""Legacy domain slice: support, dashboard, Trill, and API keys.

Those paths are implemented on dedicated routers included from ``app.py``
(``support``, ``dashboard``, ``trill``, ``api_keys``). This module no longer
registers routes — it exists so imports of :func:`register_domain_misc_routes`
remain valid without referencing removed ``app.api_*`` handlers.
"""
from __future__ import annotations

from fastapi import APIRouter


def register_domain_misc_routes(router: APIRouter) -> None:
    """No-op; see module docstring."""
