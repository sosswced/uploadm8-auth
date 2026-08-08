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
        "memory_debug_path": "/api/admin/workers/memory-debug",
        "timestamp": _now_utc().isoformat(),
    }


@router.post("/workers/watchdog/run")
async def admin_workers_watchdog_run(user: dict = Depends(require_master_admin)):
    """Force one fleet-watchdog tick (records incidents + Discord/email if needed)."""
    from services.worker_fleet_watchdog import run_fleet_watchdog_once

    if core.state.db_pool is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    return await run_fleet_watchdog_once(core.state.db_pool, core.state.redis_client)


@router.get("/workers/memory-debug")
async def admin_workers_memory_debug(
    worker_id: str | None = None,
    samples: int = 60,
    user: dict = Depends(require_master_admin),
):
    """Full in-app visibility into worker RSS history + last OOM precursor.

    Use this when Render says ``Ran out of memory (used over 512MB)`` — the
    Redis ring + sticky last-context survive long enough after a kill to show
    which upload/stage was active and parent vs FFmpeg child RSS.
    """
    from core.process_stats import is_small_memory_plan, memory_limit_mb
    from services.worker_fleet_snapshot import fetch_worker_heartbeat_rows
    from services.worker_fleet_watchdog import dangerous_concurrency_warnings
    from services.worker_memory_telemetry import (
        fetch_global_last_oom_context,
        fetch_last_context,
        fetch_memory_samples,
        starter_plan_recommendations,
    )

    if core.state.db_pool is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    redis = core.state.redis_client
    workers = await fetch_worker_heartbeat_rows(core.state.db_pool)
    limit_mb = memory_limit_mb()
    # Prefer an explicit worker_id; else first alive; else first row.
    chosen = (worker_id or "").strip()
    if not chosen:
        for w in workers or []:
            if (w.get("status") or "").lower() == "alive":
                chosen = str(w.get("worker_id") or "")
                break
        if not chosen and workers:
            chosen = str((workers[0] or {}).get("worker_id") or "")

    history = await fetch_memory_samples(redis, chosen, limit=max(1, min(int(samples or 60), 200)))
    last_ctx = await fetch_last_context(redis, chosen) if chosen else None
    global_oom = await fetch_global_last_oom_context(redis)

    # Peak from history for quick triage.
    peak = None
    peak_sample = None
    for s in history or []:
        eff = s.get("effective_rss_mb")
        if eff is None:
            parent = s.get("rss_mb")
            child = s.get("children_rss_mb") or 0
            if parent is not None:
                eff = float(parent) + float(child or 0)
        if eff is None:
            continue
        if peak is None or float(eff) > float(peak):
            peak = float(eff)
            peak_sample = s

    hb_row = None
    for w in workers or []:
        if str(w.get("worker_id") or "") == chosen:
            hb_row = w
            break

    worker_limit = None
    if hb_row and hb_row.get("memory_limit_mb") is not None:
        try:
            worker_limit = float(hb_row.get("memory_limit_mb"))
        except (TypeError, ValueError):
            worker_limit = None
    tips_limit = worker_limit if worker_limit is not None else limit_mb

    return {
        "worker_id": chosen or None,
        "heartbeat": hb_row,
        "last_context": last_ctx,
        "last_oom_precursor": global_oom,
        "history": history,
        "history_peak_mb": peak,
        "history_peak_sample": peak_sample,
        "fleet_workers": [
            {
                "worker_id": w.get("worker_id"),
                "status": w.get("status"),
                "memory_rss_mb": w.get("memory_rss_mb"),
                "memory_peak_mb": w.get("memory_peak_mb"),
                "memory_limit_mb": w.get("memory_limit_mb"),
                "memory_pct": w.get("memory_pct"),
                "memory_pressure": w.get("memory_pressure"),
                "admission_blocked": w.get("admission_blocked"),
                "active_process_jobs": w.get("active_process_jobs"),
                "active_publish_jobs": w.get("active_publish_jobs"),
                "seconds_since_last_beat": w.get("seconds_since_last_beat"),
            }
            for w in (workers or [])
        ],
        "api_process_limit_mb": limit_mb,
        "small_plan": is_small_memory_plan(tips_limit),
        "recommendations": starter_plan_recommendations(tips_limit),
        "dangerous_config": dangerous_concurrency_warnings(),
        "timestamp": _now_utc().isoformat(),
    }
