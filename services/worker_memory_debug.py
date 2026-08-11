"""Build the admin worker memory-debug payload (RSS history + OOM precursor)."""

from __future__ import annotations

from typing import Any, Dict, Optional


async def build_worker_memory_debug_payload(
    *,
    db_pool: Any,
    redis_client: Any,
    worker_id: Optional[str] = None,
    samples: int = 60,
) -> Dict[str, Any]:
    """Full in-app visibility into worker RSS history + last OOM precursor."""
    from core.helpers import _now_utc
    from core.process_stats import is_small_memory_plan, memory_limit_mb
    from services.worker_fleet_snapshot import fetch_worker_heartbeat_rows
    from services.worker_fleet_watchdog import dangerous_concurrency_warnings
    from services.worker_memory_telemetry import (
        fetch_global_last_oom_context,
        fetch_last_context,
        fetch_memory_samples,
        starter_plan_recommendations,
    )

    workers = await fetch_worker_heartbeat_rows(db_pool)
    limit_mb = memory_limit_mb()
    chosen = (worker_id or "").strip()
    if not chosen:
        for w in workers or []:
            if (w.get("status") or "").lower() == "alive":
                chosen = str(w.get("worker_id") or "")
                break
        if not chosen and workers:
            chosen = str((workers[0] or {}).get("worker_id") or "")

    history = await fetch_memory_samples(
        redis_client, chosen, limit=max(1, min(int(samples or 60), 200))
    )
    last_ctx = await fetch_last_context(redis_client, chosen) if chosen else None
    global_oom = await fetch_global_last_oom_context(redis_client)

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
