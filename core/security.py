"""
UploadM8 rate limiting and security — IP extraction, rate limit middleware.
Extracted from app.py; uses core.state for Redis and in-memory buckets.
"""

import time
import logging
from ipaddress import ip_address
from typing import Any, Dict

from fastapi import FastAPI, Request
from starlette.responses import JSONResponse

import core.state
from core.config import TRUST_PROXY_HEADERS

logger = logging.getLogger("uploadm8-api")


def _rl_now() -> float:
    return time.time()


async def rate_limit_allowed(key: str, limit: int, window_sec: int) -> bool:
    """
    Fixed-window rate limit. Prefers Redis (distributed) when configured,
    falls back to in-memory buckets for single-instance dev.
    """
    if core.state.redis_client is not None:
        try:
            count = await core.state.redis_client.incr(key)
            if count == 1:
                await core.state.redis_client.expire(key, window_sec)
            return int(count) <= int(limit)
        except Exception as e:
            logger.warning(f"Redis rate limit failed, falling back to memory: {e}")

    bucket = core.state._RATE_BUCKETS.get(key)
    t = _rl_now()
    if not bucket or t > bucket["reset_at"]:
        core.state._RATE_BUCKETS[key] = {"count": 1, "reset_at": t + window_sec}
        return True
    if bucket["count"] >= limit:
        return False
    bucket["count"] += 1
    return True


def client_ip(req: Request) -> str:
    if TRUST_PROXY_HEADERS:
        for hdr in ("cf-connecting-ip", "true-client-ip", "x-real-ip"):
            val = (req.headers.get(hdr) or "").strip()
            if val:
                try:
                    return str(ip_address(val))
                except Exception:
                    pass

        xff = (req.headers.get("x-forwarded-for") or "").strip()
        if xff:
            candidate = xff.split(",")[0].strip()
            try:
                return str(ip_address(candidate))
            except Exception:
                pass

    return (req.client.host if req.client else "unknown")


def _json_429(detail: str) -> JSONResponse:
    return JSONResponse(status_code=429, content={"detail": detail})


def install_rate_limit_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def rl_middleware(request: Request, call_next):
        ip = client_ip(request)
        path = request.url.path

        # Global guardrail
        if not await rate_limit_allowed(f"ip:{ip}:global", limit=300, window_sec=60):
            return _json_429("Rate limit exceeded (global)")

        # Sensitive surfaces
        if path.startswith("/api/auth/"):
            if not await rate_limit_allowed(f"ip:{ip}:auth", limit=30, window_sec=60):
                return _json_429("Rate limit exceeded (auth)")
        if path.startswith("/api/admin/"):
            if not await rate_limit_allowed(f"ip:{ip}:admin", limit=60, window_sec=60):
                return _json_429("Rate limit exceeded (admin)")

        return await call_next(request)
