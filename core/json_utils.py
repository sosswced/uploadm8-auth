"""Shared JSON parsing for DB text/jsonb fields (defensive until schema is uniform)."""
from __future__ import annotations

import json
from typing import Any


def safe_json(v: Any, default: Any) -> Any:
    """Parse JSON stored as text or return already-parsed list/dict."""
    if v is None:
        return default
    if isinstance(v, (list, dict)):
        return v
    if isinstance(v, str):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return default
    return default
