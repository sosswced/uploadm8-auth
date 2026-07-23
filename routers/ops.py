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
    PUBLISH_NORMAL_QUEUE,
    PUBLISH_PRIORITY_QUEUE,
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


@router.get("/api/ops/worker-health")
async def worker_health():
    """Worker lane + queue depth + fleet heartbeat snapshot for Render split deployments."""
    from services.worker_fleet_snapshot import build_worker_fleet_snapshot

    worker_lane = (os.environ.get("WORKER_LANE") or "full").strip().lower()
    base = {
        "worker_lane": worker_lane,
        "worker_concurrency": int(os.environ.get("WORKER_CONCURRENCY", "1") or 1),
        "publish_concurrency": int(os.environ.get("PUBLISH_CONCURRENCY", "1") or 1),
        "heavy_pipeline_slots": int(os.environ.get("WORKER_HEAVY_PIPELINE_SLOTS", "1") or 1),
        "async_publish_queue": os.environ.get("ASYNC_PUBLISH_QUEUE", "false"),
        "timestamp": _now_utc().isoformat(),
    }
    if core.state.db_pool is not None:
        fleet = await build_worker_fleet_snapshot(core.state.db_pool, core.state.redis_client)
        base["fleet"] = fleet.get("fleet")
        base["workers"] = fleet.get("workers")
        base["redis_queues"] = fleet.get("redis_queues")
        base["uploads"] = fleet.get("uploads")
        try:
            from services.worker_fleet_watchdog import evaluate_fleet_alerts

            base["watchdog_alerts"] = [
                {"incident_type": a.incident_type, "severity": a.severity, "subject": a.subject}
                for a in evaluate_fleet_alerts(
                    fleet.get("fleet") or {},
                    fleet.get("uploads") or {},
                    fleet.get("redis_queues") or {},
                )
            ]
        except Exception:
            base["watchdog_alerts"] = []
    else:
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
                    PUBLISH_PRIORITY_QUEUE,
                    PUBLISH_NORMAL_QUEUE,
                ]
                redis_queues["use_streams"] = use_redis_streams()
                redis_queues["lists"] = await list_lengths(core.state.redis_client, lp)
                if use_redis_streams():
                    from stages.redis_job_queue import stream_key_for_list

                    sk = process_stream_keys_ordered(
                        PROCESS_PRIORITY_QUEUE,
                        PROCESS_NORMAL_QUEUE,
                        PRIORITY_JOB_QUEUE,
                        UPLOAD_JOB_QUEUE,
                    ) + [
                        stream_key_for_list(PUBLISH_PRIORITY_QUEUE),
                        stream_key_for_list(PUBLISH_NORMAL_QUEUE),
                    ]
                    redis_queues["streams"] = await stream_lengths(core.state.redis_client, sk)
            except (RedisError, ImportError, TypeError, ValueError) as e:
                redis_queues["error"] = str(e)
        base["redis_queues"] = redis_queues
    return base


def _ops_bearer_authorized(request: Request) -> bool:
    """Return True if Authorization matches METRICS/RENDER_OPS key."""
    key = (
        (os.environ.get("RENDER_OPS_API_KEY") or "").strip()
        or (os.environ.get("METRICS_API_KEY") or "").strip()
    )
    if not key:
        return False
    return request.headers.get("authorization", "") == f"Bearer {key}"


def _require_ops_metrics_auth(request: Request) -> None:
    """Fail-closed on Render / when monitor keys exist; open only for local bare runs."""
    key = (
        (os.environ.get("RENDER_OPS_API_KEY") or "").strip()
        or (os.environ.get("METRICS_API_KEY") or "").strip()
    )
    on_render = bool(os.environ.get("RENDER"))
    from services.render_platform import render_api_configured

    must_auth = bool(key) or on_render or render_api_configured()
    if not must_auth:
        return
    if not key or not _ops_bearer_authorized(request):
        raise HTTPException(403, "Forbidden — set METRICS_API_KEY (or RENDER_OPS_API_KEY)")


@router.get("/api/ops/render-live")
async def render_live_status(request: Request):
    """Live Render worker events (Bearer METRICS_API_KEY / RENDER_OPS_API_KEY)."""
    _require_ops_metrics_auth(request)
    from services.render_platform import build_render_live_snapshot

    return await build_render_live_snapshot(event_limit=20, include_logs=True, use_cache=True)


@router.get("/metrics")
async def metrics(request: Request):
    """Lightweight metrics for monitoring dashboards. Admin-only or internal."""
    _require_ops_metrics_auth(request)
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
                    stream_key_for_list,
                    stream_lengths,
                    use_redis_streams,
                )

                lp = [
                    PROCESS_PRIORITY_QUEUE,
                    PROCESS_NORMAL_QUEUE,
                    PRIORITY_JOB_QUEUE,
                    UPLOAD_JOB_QUEUE,
                    PUBLISH_PRIORITY_QUEUE,
                    PUBLISH_NORMAL_QUEUE,
                ]
                redis_queues["use_streams"] = use_redis_streams()
                redis_queues["lists"] = await list_lengths(core.state.redis_client, lp)
                redis_queues["total_pending"] = sum(
                    int(v or 0) for v in (redis_queues.get("lists") or {}).values()
                )
                if use_redis_streams():
                    sk = process_stream_keys_ordered(
                        PROCESS_PRIORITY_QUEUE,
                        PROCESS_NORMAL_QUEUE,
                        PRIORITY_JOB_QUEUE,
                        UPLOAD_JOB_QUEUE,
                    ) + [
                        stream_key_for_list(PUBLISH_PRIORITY_QUEUE),
                        stream_key_for_list(PUBLISH_NORMAL_QUEUE),
                    ]
                    redis_queues["streams"] = await stream_lengths(core.state.redis_client, sk)
            except (RedisError, ImportError, TypeError, ValueError) as _rqe:
                redis_queues["error"] = str(_rqe)
        worker_fleet = None
        watchdog_alerts = []
        render_summary = None
        try:
            from services.worker_fleet_snapshot import build_worker_fleet_snapshot
            from services.worker_fleet_watchdog import evaluate_fleet_alerts

            worker_fleet = await build_worker_fleet_snapshot(core.state.db_pool, core.state.redis_client)
            watchdog_alerts = [
                {"incident_type": a.incident_type, "severity": a.severity, "subject": a.subject}
                for a in evaluate_fleet_alerts(
                    (worker_fleet or {}).get("fleet") or {},
                    (worker_fleet or {}).get("uploads") or {},
                    (worker_fleet or {}).get("redis_queues") or {},
                )
            ]
        except Exception:
            worker_fleet = None
        try:
            from services.render_platform import build_render_live_snapshot, render_api_configured

            # Only poll Render when caller is authenticated (never from open local scrapes).
            if render_api_configured() and _ops_bearer_authorized(request):
                render_live = await build_render_live_snapshot(
                    event_limit=10,
                    include_logs=False,
                    include_metrics=True,
                    use_cache=True,
                )
                render_summary = {
                    "health": render_live.get("health"),
                    "summary": render_live.get("summary"),
                    "instances": render_live.get("instances"),
                    "latest_deploy": render_live.get("latest_deploy"),
                    "metrics": render_live.get("metrics"),
                    "cached": render_live.get("cached"),
                }
        except Exception:
            render_summary = None
        fleet_summary = (worker_fleet or {}).get("fleet") if worker_fleet else None
        return {
            "users": {"total": total_users, "active_24h": active_24h},
            "uploads": {
                "last_24h": uploads_24h,
                "processing": processing_now,
                "queued": queued_now,
                "failed_24h": failed_24h,
                "ready_to_publish": (worker_fleet or {}).get("uploads", {}).get("ready_to_publish")
                if worker_fleet
                else None,
            },
            "db_pool": {"size": pool_size, "idle": pool_free},
            "redis_job_queues": redis_queues,
            # Backward-compatible: worker_fleet = summary (alive_count, etc.)
            "worker_fleet": fleet_summary,
            "fleet": fleet_summary,
            "workers": (worker_fleet or {}).get("workers") if worker_fleet else None,
            "worker_fleet_detail": worker_fleet,
            "watchdog_alerts": watchdog_alerts,
            "render": render_summary,
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
