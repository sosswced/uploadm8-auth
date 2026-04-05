"""
Canonical JSON parsing for pipeline stages.

Use these instead of ad-hoc ``try: json.loads`` blocks so behavior is consistent:
dict-shaped artifacts default to ``{}``; list-shaped artifacts default to ``[]``;
invalid JSON logs at DEBUG once and returns the default.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

_log = logging.getLogger("uploadm8-worker.safe_parse")


def json_dict(
    raw: Any,
    *,
    default: Optional[Dict[str, Any]] = None,
    context: str = "",
) -> Dict[str, Any]:
    """Parse ``raw`` into a dict. Non-dict JSON (e.g. list) yields default."""
    base = dict(default) if default is not None else {}
    if raw is None:
        return base
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return base
        try:
            val = json.loads(s)
        except json.JSONDecodeError as e:
            _log.debug(
                "json_dict: invalid JSON%s: %s",
                f" ({context})" if context else "",
                e,
            )
            return base
        if isinstance(val, dict):
            return val
        _log.debug(
            "json_dict: expected dict, got %s%s",
            type(val).__name__,
            f" ({context})" if context else "",
        )
        return base
    _log.debug(
        "json_dict: unsupported type %s%s",
        type(raw).__name__,
        f" ({context})" if context else "",
    )
    return base


def json_list(
    raw: Any,
    *,
    default: Optional[List[Any]] = None,
    context: str = "",
) -> List[Any]:
    """Parse ``raw`` into a list. Non-list JSON (e.g. dict) yields default."""
    base = list(default) if default is not None else []
    if raw is None:
        return base
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return base
        try:
            val = json.loads(s)
        except json.JSONDecodeError as e:
            _log.debug(
                "json_list: invalid JSON%s: %s",
                f" ({context})" if context else "",
                e,
            )
            return base
        if isinstance(val, list):
            return val
        _log.debug(
            "json_list: expected list, got %s%s",
            type(val).__name__,
            f" ({context})" if context else "",
        )
        return base
    _log.debug(
        "json_list: unsupported type %s%s",
        type(raw).__name__,
        f" ({context})" if context else "",
    )
    return base
