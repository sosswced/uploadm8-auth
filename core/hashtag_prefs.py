"""Shared hashtag preference helpers (ceiling, always-pin).

Contract:
  * ``resolve_hashtag_ceiling`` — publish/storage/evidence-pad cap = ``maxHashtags``
  * ``resolve_ai_hashtag_request`` — how many tags to ask the model for =
    ``min(aiHashtagCount, maxHashtags)``

Never invent seed/meta tags to hit either number. Always tags are user intent
and must survive seed-purge / junk scrub when re-stamped via ``pin_always_hashtags``.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Set

from core.helpers import coerce_hashtag_list, sanitize_hashtag_body


def _int_pref(us: Dict[str, Any], *keys: str, default: int) -> int:
    for k in keys:
        raw = us.get(k)
        if raw is None or (isinstance(raw, str) and not str(raw).strip()):
            continue
        try:
            return int(raw)
        except (TypeError, ValueError):
            continue
    return default


def resolve_hashtag_ceiling(
    user_settings: Optional[Dict[str, Any]] = None,
    *,
    default_max: int = 15,
) -> int:
    """Publish/storage/evidence-pad ceiling from maxHashtags (clamped 1..50)."""
    us = user_settings or {}
    max_n = _int_pref(
        us,
        "maxHashtags",
        "max_hashtags",
        default=default_max,
    )
    return max(1, min(50, max_n))


def resolve_ai_hashtag_request(
    user_settings: Optional[Dict[str, Any]] = None,
    *,
    default_ai: int = 15,
) -> int:
    """How many tags to ask the model for: min(ai count, publish ceiling)."""
    us = user_settings or {}
    ai_n = _int_pref(
        us,
        "aiHashtagCount",
        "ai_hashtag_count",
        "maxHashtags",
        "max_hashtags",
        default=default_ai,
    )
    ai_n = max(1, min(50, ai_n))
    return min(ai_n, resolve_hashtag_ceiling(us, default_max=ai_n))


def always_hashtag_bodies(user_settings: Optional[Dict[str, Any]] = None) -> List[str]:
    """Bare always-tag bodies from settings (camel or snake)."""
    us = user_settings or {}
    raw = us.get("alwaysHashtags")
    if raw is None:
        raw = us.get("always_hashtags")
    out: List[str] = []
    seen: Set[str] = set()
    for t in coerce_hashtag_list(raw or []):
        body = sanitize_hashtag_body(str(t))
        if not body or body in seen:
            continue
        seen.add(body)
        out.append(body)
    return out


def always_hashtag_protect_set(user_settings: Optional[Dict[str, Any]] = None) -> Set[str]:
    return set(always_hashtag_bodies(user_settings))


def pin_always_hashtags(
    tags: Iterable[Any],
    always: Iterable[Any],
    *,
    cap: Optional[int] = None,
) -> List[str]:
    """Always first, then remaining tags, deduped, optional cap."""
    out: List[str] = []
    seen: Set[str] = set()
    for raw in list(always or []) + list(tags or []):
        body = sanitize_hashtag_body(str(raw or "").lstrip("#"))
        if not body or body in seen:
            continue
        seen.add(body)
        out.append(body)
        if cap is not None and len(out) >= cap:
            break
    return out


__all__ = [
    "always_hashtag_bodies",
    "always_hashtag_protect_set",
    "pin_always_hashtags",
    "resolve_ai_hashtag_request",
    "resolve_hashtag_ceiling",
]
