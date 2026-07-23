"""
Dynamic hard-ban registry for weak Vision taxonomy / colors / vague categories.

Builtin seed (code) is the bootstrap baseline. Runtime overlay lives in
``admin_settings.settings_json["generic_hard_ban"]`` and can:

* add slugs (admin or auto-learn promote)
* remove / unban slugs (admin error correction — always wins)
* learn from pipeline scrubs (count hits; auto-promote after N)

Workers refresh from DB periodically so admin edits apply without redeploy.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Set

logger = logging.getLogger("uploadm8")

SETTINGS_KEY = "generic_hard_ban"
DEFAULT_AUTO_PROMOTE_AFTER = 2
_CACHE_TTL_SEC = 45.0

_effective_cache: Optional[frozenset[str]] = None
_overlay_cache: Optional[Dict[str, Any]] = None
_cache_loaded_at: float = 0.0


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_ban_slug(raw: Any) -> str:
    from core.vision_labels import vision_label_slug

    return vision_label_slug(raw)


def builtin_ban_slugs() -> frozenset[str]:
    """Code seed — baseline only; admin ``removed`` can exclude any of these."""
    from core.vision_labels import COLOR_HASHTAG_SLUGS, GENERIC_VISION_LABEL_SLUGS

    return frozenset(GENERIC_VISION_LABEL_SLUGS | COLOR_HASHTAG_SLUGS)


def empty_overlay() -> Dict[str, Any]:
    return {
        "added": [],
        "removed": [],
        "learned": {},
        "auto_promote_after": DEFAULT_AUTO_PROMOTE_AFTER,
        "updated_at": None,
        "updated_by": None,
    }


def normalize_overlay(raw: Any) -> Dict[str, Any]:
    base = empty_overlay()
    if not isinstance(raw, dict):
        return base
    added = sorted(
        {
            s
            for s in (normalize_ban_slug(x) for x in (raw.get("added") or []))
            if s and len(s) <= 48
        }
    )
    removed = sorted(
        {
            s
            for s in (normalize_ban_slug(x) for x in (raw.get("removed") or []))
            if s and len(s) <= 48
        }
    )
    learned_in = raw.get("learned") or {}
    learned: Dict[str, Any] = {}
    if isinstance(learned_in, dict):
        for k, v in learned_in.items():
            slug = normalize_ban_slug(k)
            if not slug or len(slug) > 48 or not isinstance(v, dict):
                continue
            try:
                count = max(0, int(v.get("count") or 0))
            except (TypeError, ValueError):
                count = 0
            status = str(v.get("status") or "pending")[:24]
            if status not in ("pending", "approved", "rejected", "auto"):
                status = "pending"
            learned[slug] = {
                "count": count,
                "last_seen": str(v.get("last_seen") or "")[:64] or None,
                "sources": [
                    str(s)[:64]
                    for s in (v.get("sources") or [])
                    if str(s).strip()
                ][:8],
                "status": status,
            }
        if len(learned) > 500:
            ranked = sorted(
                learned.items(),
                key=lambda kv: int((kv[1] or {}).get("count") or 0),
                reverse=True,
            )[:500]
            learned = dict(ranked)
    try:
        auto_n = int(raw.get("auto_promote_after", DEFAULT_AUTO_PROMOTE_AFTER))
    except (TypeError, ValueError):
        auto_n = DEFAULT_AUTO_PROMOTE_AFTER
    auto_n = max(0, min(50, auto_n))
    return {
        "added": added,
        "removed": removed,
        "learned": learned,
        "auto_promote_after": auto_n,
        "updated_at": raw.get("updated_at"),
        "updated_by": raw.get("updated_by"),
    }


def get_overlay_from_memory() -> Dict[str, Any]:
    """Read overlay from API/worker in-memory admin_settings_cache."""
    try:
        import core.state as state

        raw = (state.admin_settings_cache or {}).get(SETTINGS_KEY)
        return normalize_overlay(raw)
    except Exception:
        return empty_overlay()


def effective_ban_slugs(overlay: Optional[Dict[str, Any]] = None) -> frozenset[str]:
    """Builtin ∪ added ∪ auto/approved learned − removed."""
    o = normalize_overlay(overlay if overlay is not None else get_overlay_from_memory())
    removed = set(o["removed"])
    out: Set[str] = set(builtin_ban_slugs())
    out.update(o["added"])
    auto_after = int(o.get("auto_promote_after") or 0)
    for slug, meta in (o.get("learned") or {}).items():
        if not isinstance(meta, dict):
            continue
        status = str(meta.get("status") or "pending")
        if status == "rejected":
            continue
        try:
            count = int(meta.get("count") or 0)
        except (TypeError, ValueError):
            count = 0
        if status == "approved" or (auto_after > 0 and count >= auto_after):
            out.add(slug)
    out -= removed
    return frozenset(out)


def get_effective_ban_slugs_cached() -> frozenset[str]:
    """Process-local cache (refreshed by ``refresh_from_db`` / admin writes)."""
    global _effective_cache, _overlay_cache, _cache_loaded_at
    now = time.monotonic()
    if _effective_cache is not None and (now - _cache_loaded_at) < _CACHE_TTL_SEC:
        return _effective_cache
    o = get_overlay_from_memory()
    _overlay_cache = o
    _effective_cache = effective_ban_slugs(o)
    _cache_loaded_at = now
    return _effective_cache


def invalidate_ban_cache() -> None:
    global _effective_cache, _overlay_cache, _cache_loaded_at
    _effective_cache = None
    _overlay_cache = None
    _cache_loaded_at = 0.0


def is_hard_banned_slug(raw: Any) -> bool:
    slug = normalize_ban_slug(raw)
    if not slug:
        return True
    return slug in get_effective_ban_slugs_cached()


def m8_hard_ban_prompt_block(*, limit: int = 80) -> str:
    """Compact instruction block for the M8 caption prompt."""
    slugs = sorted(get_effective_ban_slugs_cached())
    sample = ", ".join(slugs[: max(1, limit)])
    more = len(slugs) - limit
    tail = f" (+{more} more in admin registry)" if more > 0 else ""
    return (
        "HARD-BAN REGISTRY (dynamic admin list — NEVER use these as title, caption, "
        "or hashtag tokens; they are weak taxonomy / colors / vague categories):\n"
        f"{sample}{tail}\n"
        "Prefer geo, route, brand/logo, driver, music, and specific landmarks instead.\n"
    )


def admin_report(overlay: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    o = normalize_overlay(overlay if overlay is not None else get_overlay_from_memory())
    effective = sorted(effective_ban_slugs(o))
    builtin = sorted(builtin_ban_slugs())
    return {
        "overlay": o,
        "builtin_count": len(builtin),
        "effective_count": len(effective),
        "added": o["added"],
        "removed": o["removed"],
        "learned": o["learned"],
        "auto_promote_after": o["auto_promote_after"],
        "effective": effective,
        "builtin_sample": builtin[:40],
        "updated_at": o.get("updated_at"),
        "updated_by": o.get("updated_by"),
    }


def apply_add(overlay: Dict[str, Any], slugs: Iterable[Any], *, by: str = "admin") -> Dict[str, Any]:
    o = normalize_overlay(overlay)
    added = set(o["added"])
    removed = set(o["removed"])
    for raw in slugs:
        slug = normalize_ban_slug(raw)
        if not slug:
            continue
        removed.discard(slug)
        added.add(slug)
        learned = o["learned"]
        if slug in learned and isinstance(learned[slug], dict):
            learned[slug]["status"] = "approved"
    o["added"] = sorted(added)
    o["removed"] = sorted(removed)
    o["updated_at"] = _now_iso()
    o["updated_by"] = (by or "admin")[:120]
    return o


def apply_remove(overlay: Dict[str, Any], slugs: Iterable[Any], *, by: str = "admin") -> Dict[str, Any]:
    """Exclude slugs from the effective ban list (even if builtin/learned)."""
    o = normalize_overlay(overlay)
    added = set(o["added"])
    removed = set(o["removed"])
    learned = dict(o["learned"])
    for raw in slugs:
        slug = normalize_ban_slug(raw)
        if not slug:
            continue
        added.discard(slug)
        removed.add(slug)
        if slug in learned and isinstance(learned[slug], dict):
            learned[slug] = {**learned[slug], "status": "rejected"}
    o["added"] = sorted(added)
    o["removed"] = sorted(removed)
    o["learned"] = learned
    o["updated_at"] = _now_iso()
    o["updated_by"] = (by or "admin")[:120]
    return o


def apply_restore(overlay: Dict[str, Any], slugs: Iterable[Any], *, by: str = "admin") -> Dict[str, Any]:
    """Undo a remove — slug can be banned again via builtin/added/learned."""
    o = normalize_overlay(overlay)
    removed = set(o["removed"])
    for raw in slugs:
        slug = normalize_ban_slug(raw)
        if slug:
            removed.discard(slug)
    o["removed"] = sorted(removed)
    o["updated_at"] = _now_iso()
    o["updated_by"] = (by or "admin")[:120]
    return o


def apply_learn_hits(
    overlay: Dict[str, Any],
    slugs: Iterable[Any],
    *,
    source: str = "hydration_scrub",
) -> Dict[str, Any]:
    """Increment learned counts for scrubbed weak tokens (pipeline learn path)."""
    o = normalize_overlay(overlay)
    removed = set(o["removed"])
    learned = dict(o["learned"])
    added = set(o["added"])
    auto_after = int(o.get("auto_promote_after") or 0)
    changed = False
    for raw in slugs:
        slug = normalize_ban_slug(raw)
        if not slug or slug in removed:
            continue
        # Skip if already effectively banned via builtin/added (no need to learn).
        if slug in builtin_ban_slugs() or slug in added:
            continue
        meta = dict(learned.get(slug) or {})
        if str(meta.get("status") or "") == "rejected":
            continue
        try:
            count = int(meta.get("count") or 0) + 1
        except (TypeError, ValueError):
            count = 1
        sources = list(meta.get("sources") or [])
        src = (source or "pipeline")[:64]
        if src and src not in sources:
            sources = (sources + [src])[:8]
        status = str(meta.get("status") or "pending")
        if auto_after > 0 and count >= auto_after:
            status = "approved"
            added.add(slug)
        meta.update(
            {
                "count": count,
                "last_seen": _now_iso(),
                "sources": sources,
                "status": status,
            }
        )
        learned[slug] = meta
        changed = True
    if not changed:
        return o
    o["learned"] = learned
    o["added"] = sorted(added)
    o["updated_at"] = _now_iso()
    o["updated_by"] = f"learn:{source}"[:120]
    return o


def set_auto_promote_after(overlay: Dict[str, Any], n: int, *, by: str = "admin") -> Dict[str, Any]:
    o = normalize_overlay(overlay)
    o["auto_promote_after"] = max(0, min(50, int(n)))
    o["updated_at"] = _now_iso()
    o["updated_by"] = (by or "admin")[:120]
    return o


async def load_overlay_from_db(pool: Any) -> Dict[str, Any]:
    if pool is None:
        return get_overlay_from_memory()
    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT settings_json FROM admin_settings WHERE id = 1"
            )
        if not row or not row["settings_json"]:
            return get_overlay_from_memory()
        data = row["settings_json"]
        if isinstance(data, str):
            data = json.loads(data)
        if not isinstance(data, dict):
            return empty_overlay()
        return normalize_overlay(data.get(SETTINGS_KEY))
    except Exception as e:
        logger.debug("generic_hard_ban load_overlay_from_db failed: %s", e)
        return get_overlay_from_memory()


async def refresh_from_db(pool: Any) -> frozenset[str]:
    """Worker/API: reload overlay from DB into cache + admin_settings_cache."""
    global _effective_cache, _overlay_cache, _cache_loaded_at
    o = await load_overlay_from_db(pool)
    try:
        import core.state as state

        if isinstance(state.admin_settings_cache, dict):
            state.admin_settings_cache[SETTINGS_KEY] = o
    except Exception:
        pass
    _overlay_cache = o
    _effective_cache = effective_ban_slugs(o)
    _cache_loaded_at = time.monotonic()
    return _effective_cache


async def persist_overlay(
    pool: Any,
    overlay: Dict[str, Any],
    *,
    merge_from_cache: bool = False,
) -> Dict[str, Any]:
    """Merge overlay into admin_settings.settings_json and refresh caches.

    Only the ``generic_hard_ban`` key is written. Other settings keys are left
    untouched so workers (which hold default caches) cannot clobber production
    watermark/billing fields. Uses ``SELECT … FOR UPDATE`` to serialize writers.
    """
    o = normalize_overlay(overlay)
    if pool is None:
        raise RuntimeError("Database unavailable")
    import core.state as state

    async with pool.acquire() as conn:
        async with conn.transaction():
            row = await conn.fetchrow(
                "SELECT settings_json FROM admin_settings WHERE id = 1 FOR UPDATE"
            )
            blob: Dict[str, Any] = {}
            if row and row["settings_json"]:
                raw = row["settings_json"]
                if isinstance(raw, str):
                    blob = json.loads(raw)
                elif isinstance(raw, dict):
                    blob = dict(raw)
            blob[SETTINGS_KEY] = o
            # Optional: API process may backfill missing keys — NEVER from worker.
            if merge_from_cache:
                for k, v in (state.admin_settings_cache or {}).items():
                    if k != SETTINGS_KEY and k not in blob:
                        blob[k] = v
            await conn.execute(
                "UPDATE admin_settings SET settings_json = $1::jsonb, updated_at = NOW() WHERE id = 1",
                json.dumps(blob),
            )
    if isinstance(state.admin_settings_cache, dict):
        state.admin_settings_cache[SETTINGS_KEY] = o
    invalidate_ban_cache()
    get_effective_ban_slugs_cached()
    return o


async def persist_learn_hits(
    pool: Any,
    slugs: Iterable[Any],
    *,
    source: str = "hydration_scrub",
) -> Optional[Dict[str, Any]]:
    """Pipeline learn path — merge hits under row lock (no lost increments)."""
    cleaned = [normalize_ban_slug(s) for s in slugs if normalize_ban_slug(s)]
    cleaned = cleaned[:48]
    if not cleaned or pool is None:
        return None
    import core.state as state

    async with pool.acquire() as conn:
        async with conn.transaction():
            row = await conn.fetchrow(
                "SELECT settings_json FROM admin_settings WHERE id = 1 FOR UPDATE"
            )
            blob: Dict[str, Any] = {}
            if row and row["settings_json"]:
                raw = row["settings_json"]
                if isinstance(raw, str):
                    blob = json.loads(raw)
                elif isinstance(raw, dict):
                    blob = dict(raw)
            current = normalize_overlay(blob.get(SETTINGS_KEY))
            new_o = apply_learn_hits(current, cleaned, source=source)
            if new_o == current:
                return None
            # Cap learned map growth (ops safety).
            learned = new_o.get("learned") or {}
            if isinstance(learned, dict) and len(learned) > 500:
                ranked = sorted(
                    learned.items(),
                    key=lambda kv: int((kv[1] or {}).get("count") or 0),
                    reverse=True,
                )[:500]
                new_o["learned"] = dict(ranked)
            blob[SETTINGS_KEY] = new_o
            await conn.execute(
                "UPDATE admin_settings SET settings_json = $1::jsonb, updated_at = NOW() WHERE id = 1",
                json.dumps(blob),
            )
    if isinstance(state.admin_settings_cache, dict):
        state.admin_settings_cache[SETTINGS_KEY] = new_o
    invalidate_ban_cache()
    get_effective_ban_slugs_cached()
    return new_o


def ban_prompt_slugs_for_tests(overlay: Dict[str, Any]) -> List[str]:
    return sorted(effective_ban_slugs(overlay))
