"""
AV-read operator flag bundle — one master-admin switch for production path flags.

Env vars still work as overrides (set any flag to 1). The admin button stores
``av_read_operator_bundle_v1`` in ``admin_settings.settings_json`` so Railway
does not need a redeploy.

End users never see these flags — they only get ``aiServiceRecognitionTraining``
consent. Soft-bias / skip-TL / Hub promote remain operator-only.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("uploadm8.av_read_flags")

SETTINGS_KEY = "av_read_operator_bundle_v1"

# One-button bundle — all default off until master-admin enables (or env set).
BUNDLE_FLAG_KEYS: List[str] = [
    "AV_READ_SOFT_BIAS",
    "AV_READ_DEPTH_SKIP_TL",
    "UM8_AV_READ_SKIP_TL_FLOORS_OK",
    "UM8_AV_READ_HUB_PROMOTE",
    "UM8_ML_ENGINE_RUN_AV_READ",
    "AV_READ_PROPOSE_ONLY",
]

_CACHE: Dict[str, Any] = {"at": 0.0, "bundle": None}
_CACHE_TTL_S = float(os.environ.get("AV_READ_FLAGS_CACHE_TTL_S", "30") or "30")


def _env_on(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in ("1", "true", "yes", "on")


def _default_bundle() -> Dict[str, Any]:
    return {
        "enabled": False,
        "all": True,
        "flags": {k: False for k in BUNDLE_FLAG_KEYS},
        "floors_ack": False,
        "enabled_by": None,
        "enabled_at": None,
        "notes": None,
        "updated_at": None,
    }


def _normalize_bundle(raw: Any) -> Dict[str, Any]:
    base = _default_bundle()
    if not isinstance(raw, dict):
        return base
    flags = dict(base["flags"])
    raw_flags = raw.get("flags") if isinstance(raw.get("flags"), dict) else {}
    for k in BUNDLE_FLAG_KEYS:
        if k in raw_flags:
            flags[k] = bool(raw_flags[k])
        elif raw.get("enabled") and raw.get("all", True):
            flags[k] = True
    base.update(
        {
            "enabled": bool(raw.get("enabled")),
            "all": bool(raw.get("all", True)),
            "flags": flags,
            "floors_ack": bool(raw.get("floors_ack")),
            "enabled_by": str(raw.get("enabled_by") or "") or None,
            "enabled_at": raw.get("enabled_at"),
            "notes": raw.get("notes"),
            "updated_at": raw.get("updated_at"),
        }
    )
    return base


def invalidate_cache() -> None:
    _CACHE["at"] = 0.0
    _CACHE["bundle"] = None


def get_bundle_from_memory() -> Dict[str, Any]:
    try:
        from core import state

        raw = (state.admin_settings_cache or {}).get(SETTINGS_KEY)
        return _normalize_bundle(raw)
    except Exception:
        return _default_bundle()


async def load_bundle_from_db(conn) -> Dict[str, Any]:
    try:
        row = await conn.fetchval(
            "SELECT settings_json FROM admin_settings WHERE id = 1"
        )
        data = row
        if isinstance(data, str):
            data = json.loads(data)
        if not isinstance(data, dict):
            return _default_bundle()
        bundle = _normalize_bundle(data.get(SETTINGS_KEY))
        try:
            from core import state

            if isinstance(state.admin_settings_cache, dict):
                state.admin_settings_cache[SETTINGS_KEY] = bundle
        except Exception:
            pass
        _CACHE["bundle"] = bundle
        _CACHE["at"] = time.time()
        return bundle
    except Exception as e:
        logger.debug("av_read bundle load failed: %s", e)
        return get_bundle_from_memory()


def _bundle_cached_sync() -> Dict[str, Any]:
    now = time.time()
    if _CACHE.get("bundle") is not None and (now - float(_CACHE.get("at") or 0)) < _CACHE_TTL_S:
        return _normalize_bundle(_CACHE["bundle"])
    mem = get_bundle_from_memory()
    _CACHE["bundle"] = mem
    _CACHE["at"] = now
    return mem


def flag_enabled(flag: str) -> bool:
    """True if env is on OR admin one-button bundle enables this flag."""
    name = str(flag or "").strip()
    if not name:
        return False
    if _env_on(name):
        return True
    bundle = _bundle_cached_sync()
    if not bundle.get("enabled"):
        return False
    flags = bundle.get("flags") or {}
    if name in flags:
        return bool(flags[name])
    return bool(bundle.get("all"))


def status_snapshot() -> Dict[str, Any]:
    bundle = _bundle_cached_sync()
    effective = {k: flag_enabled(k) for k in BUNDLE_FLAG_KEYS}
    env_only = {k: _env_on(k) for k in BUNDLE_FLAG_KEYS}
    return {
        "settings_key": SETTINGS_KEY,
        "bundle": bundle,
        "effective": effective,
        "env_overrides": env_only,
        "bundle_flag_keys": list(BUNDLE_FLAG_KEYS),
        "note": (
            "One master-admin button enables the full operator path. "
            "Env=1 still overrides. End users only control recognition-training consent."
        ),
    }


async def set_bundle(
    conn,
    *,
    enabled: bool,
    floors_ack: bool,
    master_user_id: Optional[str],
    notes: Optional[str] = None,
    confirm: bool = False,
) -> Dict[str, Any]:
    if not confirm:
        raise ValueError("Set confirm=true to change AV-read operator flags")
    if enabled and not floors_ack:
        raise ValueError(
            "floors_ack=true required: confirm hero-fact/grounding floors before enabling"
        )

    row = await conn.fetchval(
        "SELECT settings_json FROM admin_settings WHERE id = 1 FOR UPDATE"
    )
    data: Dict[str, Any] = {}
    if isinstance(row, str):
        try:
            data = json.loads(row) if row else {}
        except Exception:
            data = {}
    elif isinstance(row, dict):
        data = dict(row)

    now_iso = datetime.now(timezone.utc).isoformat()
    flags = {k: bool(enabled) for k in BUNDLE_FLAG_KEYS}
    bundle = {
        "enabled": bool(enabled),
        "all": True,
        "flags": flags,
        "floors_ack": bool(floors_ack) if enabled else False,
        "enabled_by": str(master_user_id or "") or None,
        "enabled_at": now_iso if enabled else None,
        "notes": (notes or None),
        "updated_at": now_iso,
    }
    data[SETTINGS_KEY] = bundle
    await conn.execute(
        "UPDATE admin_settings SET settings_json = $1::jsonb, updated_at = NOW() WHERE id = 1",
        json.dumps(data),
    )
    try:
        from core import state

        if isinstance(state.admin_settings_cache, dict):
            state.admin_settings_cache.clear()
            state.admin_settings_cache.update(data)
    except Exception:
        pass
    invalidate_cache()
    _CACHE["bundle"] = bundle
    _CACHE["at"] = time.time()
    return status_snapshot()


__all__ = [
    "SETTINGS_KEY",
    "BUNDLE_FLAG_KEYS",
    "flag_enabled",
    "status_snapshot",
    "load_bundle_from_db",
    "set_bundle",
    "invalidate_cache",
    "get_bundle_from_memory",
]
