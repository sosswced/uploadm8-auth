"""
Canonical code defaults for upload pipeline, Redis queues, and ML engine tuning.

Environment variables always win when set. Use ``effective_pipeline_env()`` to audit
local .env vs what the app will actually use.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

# (env_name, code_default, type: bool|str|int|float)
_PIPELINE_DEFAULTS: List[Tuple[str, Any, str]] = [
    ("WATERMARK_SINGLE_PASS", True, "bool"),
    ("TWELVE_LABS_PARALLEL", True, "bool"),
    ("REDIS_JOB_USE_STREAMS", True, "bool"),
    ("REDIS_JOB_LEGACY_DRAIN", True, "bool"),
    ("USER_PROCESS_MAX_PARALLEL", 3, "int"),
    ("USER_PROCESS_MAX_PARALLEL_PRIORITY", 6, "int"),
    ("WORKER_HEAVY_PIPELINE_SLOTS", 1, "int"),
    ("WORKER_LANE", "full", "str"),
    ("STREAM_RECLAIM_INTERVAL_SEC", 25, "int"),
    ("STREAM_RECLAIM_MIN_IDLE_MS", 120000, "int"),
    ("STREAM_RECLAIM_COUNT", 8, "int"),
    ("UM8_ML_ENGINE_ENABLED", True, "bool"),
    ("UM8_ML_ENGINE_RUN_CONTENT_SUCCESS", True, "bool"),
    ("UM8_ML_ENGINE_RUN_QUALITY_SCORING", True, "bool"),
    ("UM8_ML_ENGINE_SYNC_TRACKIO", True, "bool"),
    ("MARKETING_ML_TARGETING_ENABLED", True, "bool"),
    ("UM8_PROMO_MIN_SCORE", 0.35, "float"),
    ("AI_TRACE_TIERS", "pro,studio,agency,master_admin,friends_family", "str"),
]


def env_bool(name: str, default: bool) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw not in ("0", "false", "no", "off")


def env_int(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def env_str(name: str, default: str) -> str:
    v = (os.environ.get(name) or "").strip()
    return v if v else default


def _resolve(name: str, default: Any, typ: str) -> Any:
    if typ == "bool":
        return env_bool(name, bool(default))
    if typ == "int":
        return env_int(name, int(default))
    if typ == "float":
        return env_float(name, float(default))
    return env_str(name, str(default))


def effective_pipeline_env() -> Dict[str, Any]:
    """Map of env name → effective value (explicit env or code default)."""
    out: Dict[str, Any] = {}
    for name, default, typ in _PIPELINE_DEFAULTS:
        out[name] = _resolve(name, default, typ)
    return out


def pipeline_env_audit() -> List[Dict[str, Any]]:
    """Rows for comparing .env vs code defaults."""
    rows: List[Dict[str, Any]] = []
    for name, default, typ in _PIPELINE_DEFAULTS:
        raw = (os.environ.get(name) or "").strip()
        effective = _resolve(name, default, typ)
        rows.append(
            {
                "name": name,
                "env_set": bool(raw),
                "env_raw": raw or None,
                "code_default": default,
                "effective": effective,
                "matches_default": (not raw) or (str(effective) == str(default) if typ != "bool" else effective == default),
            }
        )
    return rows
