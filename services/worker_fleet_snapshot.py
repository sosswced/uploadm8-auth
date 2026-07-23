"""Aggregate worker heartbeat rows, Redis queues, and DB processing counts."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import asyncpg

logger = logging.getLogger("uploadm8.worker_fleet")


def _serialize_row(row: asyncpg.Record) -> dict:
    d = dict(row)
    for k, v in list(d.items()):
        if hasattr(v, "isoformat"):
            d[k] = v.isoformat()
        elif isinstance(v, (dict, list)):
            continue
        elif hasattr(v, "__float__") and not isinstance(v, (int, float, bool)):
            try:
                d[k] = float(v)
            except (TypeError, ValueError):
                pass
    return d


def _worker_status(seconds_since: Optional[int]) -> str:
    if seconds_since is None:
        return "unknown"
    if seconds_since < 30:
        return "alive"
    if seconds_since < 120:
        return "stale"
    return "dead"


async def fetch_worker_heartbeat_rows(db_pool) -> List[dict]:
    if not db_pool:
        return []
    try:
        async with db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    worker_id,
                    last_seen_at,
                    started_at,
                    worker_concurrency,
                    publish_concurrency,
                    version,
                    worker_lane,
                    service_name,
                    region,
                    memory_rss_mb,
                    memory_peak_mb,
                    memory_limit_mb,
                    heavy_pipeline_slots,
                    process_slots_in_use,
                    publish_slots_in_use,
                    active_process_jobs,
                    active_publish_jobs,
                    hostname,
                    git_commit,
                    memory_pct,
                    memory_pressure,
                    heavy_slots_in_use,
                    load_1m,
                    admission_blocked,
                    EXTRACT(EPOCH FROM (NOW() - last_seen_at))::int AS seconds_since_last_beat,
                    EXTRACT(EPOCH FROM (NOW() - started_at))::int AS uptime_seconds
                FROM worker_heartbeat
                ORDER BY last_seen_at DESC
                """
            )
    except asyncpg.UndefinedTableError:
        return []
    except asyncpg.PostgresError as e:
        # Older DBs may lack observability columns — fall back to core columns.
        if "memory_pct" in str(e) or "does not exist" in str(e).lower():
            try:
                async with db_pool.acquire() as conn:
                    rows = await conn.fetch(
                        """
                        SELECT
                            worker_id,
                            last_seen_at,
                            started_at,
                            worker_concurrency,
                            publish_concurrency,
                            version,
                            worker_lane,
                            service_name,
                            region,
                            memory_rss_mb,
                            memory_peak_mb,
                            memory_limit_mb,
                            heavy_pipeline_slots,
                            process_slots_in_use,
                            publish_slots_in_use,
                            active_process_jobs,
                            active_publish_jobs,
                            hostname,
                            git_commit,
                            EXTRACT(EPOCH FROM (NOW() - last_seen_at))::int AS seconds_since_last_beat,
                            EXTRACT(EPOCH FROM (NOW() - started_at))::int AS uptime_seconds
                        FROM worker_heartbeat
                        ORDER BY last_seen_at DESC
                        """
                    )
            except asyncpg.PostgresError as e2:
                logger.warning("worker_heartbeat fetch failed: %s", e2)
                return []
        else:
            logger.warning("worker_heartbeat fetch failed: %s", e)
            return []

    out: List[dict] = []
    for r in rows:
        d = _serialize_row(r)
        sec = d.get("seconds_since_last_beat")
        d["status"] = _worker_status(int(sec) if sec is not None else None)
        for jkey in ("active_process_jobs", "active_publish_jobs"):
            raw = d.get(jkey)
            if isinstance(raw, str):
                try:
                    d[jkey] = json.loads(raw)
                except json.JSONDecodeError:
                    d[jkey] = []
            elif raw is None:
                d[jkey] = []
        out.append(d)
    return out


def summarize_fleet(workers: List[dict]) -> dict:
    alive = stale = dead = 0
    total_proc_cap = total_pub_cap = 0
    proc_in_use = pub_in_use = 0
    max_rss = 0.0
    memory_warn = 0
    admission_blocked = 0
    hard_pressure = 0
    max_load = 0.0
    for w in workers:
        st = w.get("status") or "unknown"
        if st == "alive":
            alive += 1
        elif st == "stale":
            stale += 1
        else:
            dead += 1
        if st in ("alive", "stale"):
            total_proc_cap += int(w.get("worker_concurrency") or 0)
            total_pub_cap += int(w.get("publish_concurrency") or 0)
            proc_in_use += int(w.get("process_slots_in_use") or 0)
            pub_in_use += int(w.get("publish_slots_in_use") or 0)
            rss = float(w.get("memory_rss_mb") or 0)
            if rss > max_rss:
                max_rss = rss
            limit = float(w.get("memory_limit_mb") or 0)
            pct = w.get("memory_pct")
            if pct is None and limit and rss:
                pct = 100.0 * rss / limit
            if (limit and rss >= limit * 0.85) or (pct is not None and float(pct) >= 85):
                memory_warn += 1
            if w.get("admission_blocked") or (w.get("memory_pressure") or "") in ("soft", "hard"):
                admission_blocked += 1
            if (w.get("memory_pressure") or "") == "hard":
                hard_pressure += 1
            load = float(w.get("load_1m") or 0)
            if load > max_load:
                max_load = load
    return {
        "worker_count": len(workers),
        "alive_count": alive,
        "stale_count": stale,
        "dead_count": dead,
        "process_capacity": total_proc_cap,
        "process_slots_in_use": proc_in_use,
        "process_slots_free": max(0, total_proc_cap - proc_in_use),
        "publish_capacity": total_pub_cap,
        "publish_slots_in_use": pub_in_use,
        "publish_slots_free": max(0, total_pub_cap - pub_in_use),
        "max_memory_rss_mb": round(max_rss, 1) if max_rss else None,
        "workers_memory_warn": memory_warn,
        "workers_admission_blocked": admission_blocked,
        "workers_hard_pressure": hard_pressure,
        "max_load_1m": round(max_load, 2) if max_load else None,
    }


async def fetch_redis_queue_snapshot(redis_client) -> dict:
    if not redis_client:
        return {"available": False}
    try:
        from core.config import (
            PRIORITY_JOB_QUEUE,
            PROCESS_NORMAL_QUEUE,
            PROCESS_PRIORITY_QUEUE,
            PUBLISH_NORMAL_QUEUE,
            PUBLISH_PRIORITY_QUEUE,
            UPLOAD_JOB_QUEUE,
        )
        from stages.redis_job_queue import (
            list_lengths,
            process_stream_keys_ordered,
            stream_lengths,
            stream_key_for_list,
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
        snap: dict = {"available": True, "use_streams": use_redis_streams(), "lists": {}}
        snap["lists"] = await list_lengths(redis_client, lp)
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
            snap["streams"] = await stream_lengths(redis_client, sk)
        snap["total_pending"] = sum(int(v or 0) for v in snap.get("lists", {}).values())
        return snap
    except Exception as e:
        return {"available": False, "error": str(e)}


async def fetch_db_processing_counts(db_pool) -> dict:
    if not db_pool:
        return {}
    try:
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT
                    COUNT(*) FILTER (WHERE status = 'processing')::int AS processing,
                    COUNT(*) FILTER (WHERE status IN ('queued', 'staged'))::int AS queued,
                    COUNT(*) FILTER (WHERE status = 'ready_to_publish')::int AS ready_to_publish
                FROM uploads
                """
            )
        return dict(row) if row else {}
    except asyncpg.PostgresError as e:
        logger.warning("processing counts fetch failed: %s", e)
        return {}


async def build_worker_fleet_snapshot(db_pool, redis_client=None) -> dict:
    workers = await fetch_worker_heartbeat_rows(db_pool)
    fleet = summarize_fleet(workers)
    queues = await fetch_redis_queue_snapshot(redis_client)
    db_counts = await fetch_db_processing_counts(db_pool)
    return {
        "workers": workers,
        "fleet": fleet,
        "redis_queues": queues,
        "uploads": db_counts,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
