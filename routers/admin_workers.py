"""Admin endpoints for Render worker fleet live monitoring + watchdog."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

import core.state
from core.deps import require_master_admin
from core.helpers import _now_utc

router = APIRouter(prefix="/api/admin", tags=["admin", "workers"])


@router.get("/workers/render-live")
async def admin_workers_render_live(user: dict = Depends(require_master_admin)):
    """Live Render platform status + in-app fleet + watchdog evaluation.

    Combines heartbeat/queue snapshot with Render API events (server_failed,
    restarts, autoscaling) so ops can monitor crashes without waiting for email.
    Requires ``RENDER_MONITOR_API_KEY`` + ``RENDER_MONITOR_SERVICE_ID`` on the API
    (legacy ``RENDER_API_KEY`` / ``RENDER_WORKER_SERVICE_ID`` still accepted).
    """
    from services.render_platform import build_render_live_snapshot
    from services.worker_fleet_snapshot import build_worker_fleet_snapshot
    from services.worker_fleet_watchdog import (
        dangerous_concurrency_warnings,
        evaluate_fleet_alerts,
        evaluate_render_event_alerts,
    )

    if core.state.db_pool is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    fleet_snap = await build_worker_fleet_snapshot(core.state.db_pool, core.state.redis_client)
    render_live = await build_render_live_snapshot(event_limit=20)
    fleet_alerts = evaluate_fleet_alerts(
        fleet_snap.get("fleet") or {},
        fleet_snap.get("uploads") or {},
        fleet_snap.get("redis_queues") or {},
    )
    event_alerts = evaluate_render_event_alerts(render_live.get("events") or [])
    return {
        "fleet": fleet_snap,
        "render": render_live,
        "watchdog": {
            "alerts": [
                {
                    "incident_type": a.incident_type,
                    "severity": a.severity,
                    "subject": a.subject,
                    "body": a.body,
                }
                for a in (fleet_alerts + event_alerts)
            ],
            "dangerous_config": dangerous_concurrency_warnings(),
        },
        "timestamp": _now_utc().isoformat(),
    }


@router.post("/workers/watchdog/run")
async def admin_workers_watchdog_run(user: dict = Depends(require_master_admin)):
    """Force one fleet-watchdog tick (records incidents + Discord/email if needed)."""
    from services.worker_fleet_watchdog import run_fleet_watchdog_once

    if core.state.db_pool is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    return await run_fleet_watchdog_once(core.state.db_pool, core.state.redis_client)
