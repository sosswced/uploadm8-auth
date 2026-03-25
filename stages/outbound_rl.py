"""
Second-layer outbound throttling (worker process → OpenAI, Meta Graph, TikTok, Playwright).

Independent from the public API rate limits in app.py (RATE_LIMIT_*, RL_*).

Environment
-----------
OUTBOUND_RL_ENABLED
    Default true. Set false only for local debugging (not recommended in shared environments).

OUTBOUND_RL_PROFILE
    strict | standard | relaxed | enterprise — sets default max concurrent calls and optional
    minimum spacing per provider (see presets below).

Per-provider overrides (optional integers):
    OUTBOUND_OPENAI_MAX_CONCURRENT, OUTBOUND_OPENAI_MIN_INTERVAL_MS
    OUTBOUND_PLAYWRIGHT_MAX_CONCURRENT, OUTBOUND_PLAYWRIGHT_MIN_INTERVAL_MS
    OUTBOUND_META_MAX_CONCURRENT, OUTBOUND_META_MIN_INTERVAL_MS
    OUTBOUND_TIKTOK_MAX_CONCURRENT, OUTBOUND_TIKTOK_MIN_INTERVAL_MS

MAX_CONCURRENT ≤ 0 falls back to the profile default. MIN_INTERVAL_MS = 0 disables spacing.

Staging vs production
---------------------
Use a separate Render/hosting service (or env group) for staging and set e.g.:
    OUTBOUND_RL_PROFILE=relaxed
    RATE_LIMIT_PROFILE=relaxed
so staging workers and API are less likely to hammer external quotas during QA.
Production keeps standard or enterprise. Do not reuse production API keys on staging if
your provider forbids it — use separate OpenAI/Meta/TikTok apps when required.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict

logger = logging.getLogger("uploadm8-worker")

_OUTBOUND_ENABLED = os.environ.get("OUTBOUND_RL_ENABLED", "true").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
_PROFILE = os.environ.get("OUTBOUND_RL_PROFILE", "standard").strip().lower()

# (max_concurrent, min_interval_ms) — interval is enforced between call *starts* under the gate.
_PRESETS: Dict[str, Dict[str, tuple]] = {
    "strict": {
        "openai": (2, 400),
        "playwright": (1, 200),
        "meta": (2, 300),
        "tiktok": (2, 300),
    },
    "standard": {
        "openai": (4, 0),
        "playwright": (2, 0),
        "meta": (4, 0),
        "tiktok": (4, 0),
    },
    "relaxed": {
        "openai": (8, 0),
        "playwright": (4, 0),
        "meta": (8, 0),
        "tiktok": (8, 0),
    },
    "enterprise": {
        "openai": (16, 0),
        "playwright": (8, 0),
        "meta": (16, 0),
        "tiktok": (16, 0),
    },
}


def _parse_int(name: str, default: int) -> int:
    v = os.environ.get(name, "").strip()
    if not v:
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _base_for(name: str) -> tuple:
    prof = _PRESETS.get(_PROFILE) or _PRESETS["standard"]
    return prof.get(name, _PRESETS["standard"][name])


def _resolved_limits(name: str) -> tuple:
    max_c, min_ms = _base_for(name)
    mc = _parse_int(f"OUTBOUND_{name.upper()}_MAX_CONCURRENT", max_c)
    if mc <= 0:
        mc = max_c
    mi = _parse_int(f"OUTBOUND_{name.upper()}_MIN_INTERVAL_MS", int(min_ms))
    if mi < 0:
        mi = 0
    return mc, float(mi)


class _Gate:
    __slots__ = ("_sem", "_space_lock", "_last", "_min_s")

    def __init__(self, max_concurrent: int, min_interval_ms: float):
        self._sem = asyncio.Semaphore(max(1, int(max_concurrent)))
        self._space_lock = asyncio.Lock()
        self._last = 0.0
        self._min_s = max(0.0, float(min_interval_ms) / 1000.0)

    @asynccontextmanager
    async def acquire(self):
        await self._sem.acquire()
        try:
            if self._min_s > 0:
                async with self._space_lock:
                    now = time.monotonic()
                    wait = self._last + self._min_s - now
                    if wait > 0:
                        await asyncio.sleep(wait)
                    self._last = time.monotonic()
            yield
        finally:
            self._sem.release()


_gates: Dict[str, _Gate] = {}


def _get_gate(name: str) -> _Gate:
    if name not in _gates:
        mc, mi = _resolved_limits(name)
        _gates[name] = _Gate(mc, mi)
    return _gates[name]


@asynccontextmanager
async def outbound_slot(provider: str):
    """
    Gate outbound calls: openai (Whisper + GPT + image API), playwright, meta, tiktok.
    """
    if not _OUTBOUND_ENABLED:
        yield
        return
    p = (provider or "").strip().lower()
    if p not in ("openai", "playwright", "meta", "tiktok"):
        yield
        return
    async with _get_gate(p).acquire():
        yield


def startup_log_line() -> str:
    if not _OUTBOUND_ENABLED:
        return "outbound_rl: disabled (OUTBOUND_RL_ENABLED=0)"
    parts = [f"profile={_PROFILE}"]
    for k in ("openai", "playwright", "meta", "tiktok"):
        mc, mi = _resolved_limits(k)
        parts.append(f"{k}=conc:{mc},space_ms:{int(mi)}")
    return "outbound_rl: " + " ".join(parts)
