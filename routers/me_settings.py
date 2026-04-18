"""Legacy domain slice: ``/api/me``, settings, wallet, preferences.

Implemented on ``routers.me``, ``routers.preferences``, and related modules
mounted from ``app.py``. Registration here is a no-op.
"""
from __future__ import annotations

from fastapi import APIRouter


def register_me_wallet_settings_routes(router: APIRouter) -> None:
    """No-op; see module docstring."""
