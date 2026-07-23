"""Proactive worker fleet watchdog — alert before Render crash emails pile up.

Polls in-app heartbeat/fleet snapshot (+ optional Render API events) and
records operational incidents when:

* no alive workers (fleet down / OOM kill)
* memory warn on live instances
* process slots saturated with deep queue (overload)
* Render API reports recent ``server_failed`` / restart events

Gated by ``WORKER_FLEET_WATCHDOG_ENABLED`` (default on when DB is available).
"""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger("uploadm8.worker_fleet_watchdog")


def _env_bool(name: str, default: bool = True) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)) or default)
    except ValueError:
        return default


@dataclass(frozen=True)
class FleetAlert:
    incident_type: str
    subject: str
    body: str
    details: Dict[str, Any]
    severity: str = "warning"  # warning | critical


def evaluate_fleet_alerts(
    fleet: Dict[str, Any],
    uploads: Optional[Dict[str, Any]] = None,
    queues: Optional[Dict[str, Any]] = None,
    *,
    queue_pending_threshold: Optional[int] = None,
) -> List[FleetAlert]:
    """Pure evaluation of heartbeat fleet snapshot → alerts (unit-testable)."""
    uploads = uploads or {}
    queues = queues or {}
    pending_thresh = (
        queue_pending_threshold
        if queue_pending_threshold is not None
        else _env_int("WATCHDOG_QUEUE_PENDING_ALERT", 8)
    )

    alerts: List[FleetAlert] = []
    worker_count = int(fleet.get("worker_count") or 0)
    alive = int(fleet.get("alive_count") or 0)
    stale = int(fleet.get("stale_count") or 0)
    dead = int(fleet.get("dead_count") or 0)
    mem_warn = int(fleet.get("workers_memory_warn") or 0)
    proc_cap = int(fleet.get("process_capacity") or 0)
    proc_free = int(fleet.get("process_slots_free") or 0)
    proc_in_use = int(fleet.get("process_slots_in_use") or 0)
    pending = int(queues.get("total_pending") or 0)
    processing = int(uploads.get("processing") or 0)

    if worker_count > 0 and alive == 0:
        alerts.append(
            FleetAlert(
                incident_type="worker_fleet_down",
                subject="Render worker fleet DOWN — no alive heartbeats",
                body=(
                    f"All {worker_count} worker heartbeat row(s) are stale/dead "
                    f"(stale={stale}, dead={dead}). Likely OOM kill or crash; "
                    f"uploads processing={processing}, redis pending={pending}."
                ),
                details={
                    "fleet": fleet,
                    "uploads": uploads,
                    "queues_pending": pending,
                },
                severity="critical",
            )
        )
    elif dead > 0 and alive > 0:
        alerts.append(
            FleetAlert(
                incident_type="worker_instance_dead",
                subject=f"Render worker instance dead ({dead}) while {alive} alive",
                body=(
                    f"{dead} worker heartbeat(s) marked dead; {alive} still alive. "
                    f"Autoscaling may be recovering; check Render events + RAM."
                ),
                details={"fleet": fleet, "uploads": uploads},
                severity="warning",
            )
        )

    if mem_warn > 0 and alive > 0:
        alerts.append(
            FleetAlert(
                incident_type="worker_memory_pressure",
                subject=f"Worker memory pressure — {mem_warn} instance(s) ≥85% RAM",
                body=(
                    f"{mem_warn} live worker(s) above 85% of memory limit. "
                    f"Admission should block new encodes; verify WORKER_CONCURRENCY=1 "
                    f"and RENDER_MEMORY_LIMIT_MB. slots={proc_in_use}/{proc_cap}."
                ),
                details={
                    "fleet": fleet,
                    "max_memory_rss_mb": fleet.get("max_memory_rss_mb"),
                },
                severity="critical" if mem_warn >= alive else "warning",
            )
        )

    if proc_cap > 0 and proc_free == 0 and pending >= pending_thresh:
        alerts.append(
            FleetAlert(
                incident_type="worker_overload",
                subject="Worker overload — process slots full with deep queue",
                body=(
                    f"Process slots saturated ({proc_in_use}/{proc_cap}) with "
                    f"redis pending={pending} (threshold={pending_thresh}) and "
                    f"db processing={processing}. Risk of backlog → memory climb."
                ),
                details={
                    "fleet": fleet,
                    "uploads": uploads,
                    "queues_pending": pending,
                    "threshold": pending_thresh,
                },
                severity="warning",
            )
        )

    return alerts


def evaluate_render_event_alerts(events: List[dict]) -> List[FleetAlert]:
    """Turn recent critical Render platform events into FleetAlerts."""
    alerts: List[FleetAlert] = []
    seen_types: set = set()
    for ev in events or []:
        if (ev.get("severity") or "") != "critical":
            continue
        et = str(ev.get("type") or "server_failed")
        if et in seen_types:
            continue
        seen_types.add(et)
        alerts.append(
            FleetAlert(
                incident_type=f"render_event_{et}"[:120],
                subject=f"Render platform event: {et}",
                body=(
                    f"Live Render API reported ``{et}`` at {ev.get('timestamp')}. "
                    f"This is an early signal — do not wait for the crash email."
                ),
                details={"event": ev},
                severity="critical",
            )
        )
    return alerts


def dangerous_concurrency_warnings() -> List[str]:
    """Flag env that historically OOMs a 2GB Render worker."""
    warnings: List[str] = []
    on_render = bool(os.environ.get("RENDER"))
    try:
        conc = int(os.environ.get("WORKER_CONCURRENCY", "1") or 1)
    except ValueError:
        conc = 1
    try:
        pub = int(os.environ.get("PUBLISH_CONCURRENCY", "1") or 1)
    except ValueError:
        pub = 1
    try:
        mem = float(os.environ.get("RENDER_MEMORY_LIMIT_MB") or 0)
    except ValueError:
        mem = 0.0
    # On API web service RENDER may be set too — only warn when this looks like
    # a full/process worker profile or when concurrency is explicitly high.
    lane = (os.environ.get("WORKER_LANE") or "").strip().lower()
    if on_render and conc >= 2 and (not mem or mem <= 2048):
        warnings.append(
            f"WORKER_CONCURRENCY={conc} on ≤2GB risks Instance-failed OOM "
            "(set WORKER_CONCURRENCY=1 or upgrade RAM)."
        )
    if on_render and pub >= 5 and (lane in ("", "full", "publish")) and (not mem or mem <= 2048):
        warnings.append(
            f"PUBLISH_CONCURRENCY={pub} on ≤2GB can OOM during concurrent downloads "
            "(prefer PUBLISH_CONCURRENCY=1)."
        )
    return warnings


async def run_fleet_watchdog_once(db_pool, redis_client=None) -> Dict[str, Any]:
    """Single watchdog tick: evaluate fleet (+ Render events) and record incidents."""
    from services.ops_incidents import record_operational_incident
    from services.worker_fleet_snapshot import build_worker_fleet_snapshot

    snap = await build_worker_fleet_snapshot(db_pool, redis_client)
    fleet = snap.get("fleet") or {}
    uploads = snap.get("uploads") or {}
    queues = snap.get("redis_queues") or {}

    alerts = evaluate_fleet_alerts(fleet, uploads, queues)

    render_live: Optional[Dict[str, Any]] = None
    if _env_bool("WATCHDOG_RENDER_API_ENABLED", True):
        try:
            from services.render_platform import build_render_live_snapshot, render_api_configured

            if render_api_configured():
                render_live = await build_render_live_snapshot(event_limit=15)
                alerts.extend(evaluate_render_event_alerts(render_live.get("events") or []))
        except Exception as e:
            logger.warning("watchdog render live fetch: %s", e)

    for w in dangerous_concurrency_warnings():
        alerts.append(
            FleetAlert(
                incident_type="worker_dangerous_config",
                subject="Dangerous worker concurrency config on Render",
                body=w,
                details={"warning": w},
                severity="warning",
            )
        )

    try:
        from services.worker_admission import scale_out_hint

        hint = scale_out_hint(fleet, int(queues.get("total_pending") or 0))
        if hint:
            alerts.append(
                FleetAlert(
                    incident_type="worker_scale_out_hint",
                    subject="Scale out Render workers (keep concurrency=1)",
                    body=hint,
                    details={"fleet": fleet, "queues_pending": queues.get("total_pending")},
                    severity="warning",
                )
            )
    except Exception:
        pass

    recorded: List[str] = []
    for alert in alerts:
        dedupe = _env_int(
            "WATCHDOG_ALERT_DEDUPE_SECONDS",
            600 if alert.severity == "critical" else 900,
        )
        inc_id = await record_operational_incident(
            db_pool,
            source="worker_fleet_watchdog",
            incident_type=alert.incident_type,
            subject=alert.subject,
            body=alert.body,
            details={
                **alert.details,
                "severity": alert.severity,
                "render_summary": (render_live or {}).get("summary") if render_live else None,
            },
            send_alerts=True,
            alert_email=True,
            alert_discord=True,
            dedupe_seconds=dedupe,
            dedupe_key=f"watchdog:{alert.incident_type}",
        )
        if inc_id:
            recorded.append(inc_id)

    return {
        "fleet": fleet,
        "alert_count": len(alerts),
        "alerts": [
            {"incident_type": a.incident_type, "severity": a.severity, "subject": a.subject}
            for a in alerts
        ],
        "incident_ids": recorded,
        "render_configured": bool((render_live or {}).get("configured")),
        "render_summary": (render_live or {}).get("summary"),
        "timestamp": snap.get("timestamp"),
    }


async def run_worker_fleet_watchdog_loop(db_pool, redis_client=None) -> None:
    """API lifespan background loop — defensive monitoring for Render workers."""
    if not _env_bool("WORKER_FLEET_WATCHDOG_ENABLED", True):
        logger.info("worker fleet watchdog disabled (WORKER_FLEET_WATCHDOG_ENABLED=0)")
        return

    interval = max(30, _env_int("WORKER_FLEET_WATCHDOG_INTERVAL_SEC", 60))
    await asyncio.sleep(20)  # let pool/heartbeat settle after boot
    logger.info("worker fleet watchdog started (interval=%ss)", interval)

    while True:
        try:
            result = await run_fleet_watchdog_once(db_pool, redis_client)
            if result.get("alert_count"):
                logger.warning(
                    "fleet watchdog alerts=%s types=%s",
                    result["alert_count"],
                    [a["incident_type"] for a in result.get("alerts") or []],
                )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("worker fleet watchdog tick failed: %s", e)
        await asyncio.sleep(interval)
