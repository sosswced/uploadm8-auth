"""Capacity admission: maximize upload throughput without Render OOM stress.

Rules of thumb on Standard (2GB) workers:
* Keep WORKER_CONCURRENCY=1 per instance (encode RAM).
* Scale *out* (more instances) for more parallel uploads — not concurrency up.
* Never dispatch new process work under hard memory pressure.
* Under soft pressure, wait (caller) / skip scheduler dispatch.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)) or default)
    except ValueError:
        return default


def active_pipeline_stale_minutes() -> int:
    """Minutes before a staged-in-flight job is treated as reclaimable.

    Long HD/4K FFmpeg can exceed 20m without a stage heartbeat; default 90.
    Override with ACTIVE_PIPELINE_STALE_MINUTES or fall back to STALE_PROCESSING_MINUTES.
    """
    raw = (os.environ.get("ACTIVE_PIPELINE_STALE_MINUTES") or "").strip()
    if raw:
        try:
            return max(15, int(raw))
        except ValueError:
            pass
    return max(15, _env_int("STALE_PROCESSING_MINUTES", 90))


def process_dispatch_limit(
    *,
    local_free_slots: int,
    memory_blocks: bool,
    fleet: Optional[Dict[str, Any]] = None,
) -> int:
    """How many staged→queued jobs this scheduler tick may claim.

    Maximizes throughput by filling free *local* process slots, but backs off
    when this instance is under memory pressure or the fleet reports hard RAM.
    """
    if memory_blocks or local_free_slots <= 0:
        return 0

    limit = max(0, int(local_free_slots))

    if fleet:
        # Any hard pressure in the fleet → don't pile more work onto Redis.
        if int(fleet.get("workers_hard_pressure") or 0) > 0:
            return 0
        # Memory warn on *this* capacity class: still allow at most 1 if we have a free slot
        # (local admit will wait/soft-block); prefer not flooding the queue.
        warn = int(fleet.get("workers_memory_warn") or 0)
        alive = int(fleet.get("alive_count") or 0)
        if warn > 0 and alive > 0 and warn >= alive:
            return min(limit, 1)

        # Optional hard cluster cap (0 = unlimited beyond local slots).
        cluster_cap = _env_int("CLUSTER_PROCESS_DISPATCH_CAP", 0)
        if cluster_cap > 0:
            in_use = int(fleet.get("process_slots_in_use") or 0)
            free_cluster = max(0, cluster_cap - in_use)
            limit = min(limit, free_cluster)

    return max(0, limit)


def scale_out_hint(fleet: Optional[Dict[str, Any]], pending: int) -> Optional[str]:
    """Suggest Render scale-out when backlog grows but RAM is healthy (don't raise concurrency)."""
    if not fleet:
        return None
    pending_thresh = _env_int("WATCHDOG_QUEUE_PENDING_ALERT", 8)
    alive = int(fleet.get("alive_count") or 0)
    free = int(fleet.get("process_slots_free") or 0)
    warn = int(fleet.get("workers_memory_warn") or 0)
    hard = int(fleet.get("workers_hard_pressure") or 0)
    if pending < pending_thresh or alive <= 0:
        return None
    if hard or warn:
        return None
    if free <= 0:
        return (
            f"Backlog pending={pending} with {alive} worker(s) at process capacity and healthy RAM. "
            "Scale out Render instances (keep WORKER_CONCURRENCY=1) — do not raise concurrency on 2GB."
        )
    return None
