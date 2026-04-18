"""Health, readiness, metrics, and API contract routes (state reads app module at runtime)."""
from __future__ import annotations

import os

import asyncpg
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from redis.exceptions import RedisError

import core.state
from core.helpers import _now_utc
from core.config import (
    PRIORITY_JOB_QUEUE,
    PROCESS_NORMAL_QUEUE,
    PROCESS_PRIORITY_QUEUE,
    UPLOAD_JOB_QUEUE,
)
from services.ops_readiness import run_readiness_checks

router = APIRouter(tags=["ops"])

# Primary liveness probe lives on ``app: /health`` to avoid duplicate routes.


@router.get("/ready")
async def readiness():
    healthy, checks = await run_readiness_checks(core.state.db_pool, core.state.redis_client)

    status = 200 if healthy else 503
    return JSONResponse(
        status_code=status,
        content={
            "status": "ok" if healthy else "degraded",
            "checks": checks,
            "timestamp": _now_utc().isoformat(),
        },
    )


@router.get("/api/v1", tags=["meta"])
async def api_v1_contract():
    """Stable entry for external integrators; OpenAPI for all routes remains at /openapi.json and /docs."""
    return {
        "api_version": "1",
        "openapi_json": "/openapi.json",
        "docs": "/docs",
        "alias": "Every /api/v1/... request is handled by the same route as /api/...",
        "examples": [
            "/api/v1/auth/login",
            "/api/v1/me",
            "/api/v1/uploads/presign",
            "/api/v1/billing/checkout",
        ],
        "auth_note": "Bearer header or HttpOnly access cookie (credentials: include).",
    }


@router.get("/metrics")
async def metrics(request: Request):
    """Lightweight metrics for monitoring dashboards. Admin-only or internal."""
    auth = request.headers.get("authorization", "")
    metrics_key = os.environ.get("METRICS_API_KEY", "")
    if metrics_key and auth != f"Bearer {metrics_key}":
        raise HTTPException(403, "Forbidden")
    try:
        async with core.state.db_pool.acquire() as conn:
            total_users = await conn.fetchval("SELECT COUNT(*) FROM users")
            active_24h = await conn.fetchval(
                "SELECT COUNT(*) FROM users WHERE last_active_at > NOW() - INTERVAL '24 hours'"
            )
            uploads_24h = await conn.fetchval(
                "SELECT COUNT(*) FROM uploads WHERE created_at > NOW() - INTERVAL '24 hours'"
            )
            processing_now = await conn.fetchval(
                "SELECT COUNT(*) FROM uploads WHERE status = 'processing'"
            )
            queued_now = await conn.fetchval(
                "SELECT COUNT(*) FROM uploads WHERE status IN ('queued', 'staged')"
            )
            failed_24h = await conn.fetchval(
                "SELECT COUNT(*) FROM uploads WHERE status = 'failed' AND updated_at > NOW() - INTERVAL '24 hours'"
            )
        pool = core.state.db_pool
        pool_size = pool.get_size() if hasattr(pool, "get_size") else -1
        pool_free = pool.get_idle_size() if hasattr(pool, "get_idle_size") else -1
        redis_queues: dict = {}
        if core.state.redis_client:
            try:
                from stages.redis_job_queue import (
                    list_lengths,
                    process_stream_keys_ordered,
                    stream_lengths,
                    use_redis_streams,
                )

                lp = [
                    PROCESS_PRIORITY_QUEUE,
                    PROCESS_NORMAL_QUEUE,
                    PRIORITY_JOB_QUEUE,
                    UPLOAD_JOB_QUEUE,
                ]
                redis_queues["use_streams"] = use_redis_streams()
                redis_queues["lists"] = await list_lengths(core.state.redis_client, lp)
                if use_redis_streams():
                    sk = process_stream_keys_ordered(
                        PROCESS_PRIORITY_QUEUE,
                        PROCESS_NORMAL_QUEUE,
                        PRIORITY_JOB_QUEUE,
                        UPLOAD_JOB_QUEUE,
                    )
                    redis_queues["streams"] = await stream_lengths(core.state.redis_client, sk)
            except (RedisError, ImportError, TypeError, ValueError) as _rqe:
                redis_queues["error"] = str(_rqe)
        return {
            "users": {"total": total_users, "active_24h": active_24h},
            "uploads": {
                "last_24h": uploads_24h,
                "processing": processing_now,
                "queued": queued_now,
                "failed_24h": failed_24h,
            },
            "db_pool": {"size": pool_size, "idle": pool_free},
            "redis_job_queues": redis_queues,
            "timestamp": _now_utc().isoformat(),
        }
    except (
        asyncpg.exceptions.PostgresError,
        asyncpg.exceptions.InterfaceError,
        TimeoutError,
        OSError,
        AttributeError,
    ) as e:
        raise HTTPException(500, f"Metrics error: {e}") from e


@router.post("/cdn-cgi/rum")
async def cf_rum_beacon_sink() -> Response:
    """Accept Cloudflare RUM beacons (no-op)."""
    return Response(status_code=204)
