"""Legacy domain slice: groups and platform account listing.

Implemented on ``routers.groups`` and ``routers.platforms`` (and ``routers.oauth``).
Registration here is a no-op.
"""
from __future__ import annotations

from fastapi import APIRouter


def register_groups_platforms_routes(router: APIRouter) -> None:
    """No-op; see module docstring."""
