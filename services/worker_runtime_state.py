"""In-process worker job tracking for heartbeat / fleet observability."""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List

_lock = asyncio.Lock()
_process_jobs: Dict[str, str] = {}
_publish_jobs: Dict[str, str] = {}


async def track_process_start(upload_id: str, *, stage: str = "init") -> None:
    async with _lock:
        _process_jobs[str(upload_id)] = stage


async def track_process_stage(upload_id: str, stage: str) -> None:
    async with _lock:
        uid = str(upload_id)
        if uid in _process_jobs:
            _process_jobs[uid] = stage


async def track_process_end(upload_id: str) -> None:
    async with _lock:
        _process_jobs.pop(str(upload_id), None)


async def track_publish_start(upload_id: str, *, stage: str = "publish") -> None:
    async with _lock:
        _publish_jobs[str(upload_id)] = stage


async def track_publish_end(upload_id: str) -> None:
    async with _lock:
        _publish_jobs.pop(str(upload_id), None)


def snapshot() -> Dict[str, Any]:
    """Sync snapshot safe to call from heartbeat loop (no await while holding jobs)."""
    proc = dict(_process_jobs)
    pub = dict(_publish_jobs)
    return {
        "active_process_jobs": [{"upload_id": k, "stage": v} for k, v in proc.items()],
        "active_publish_jobs": [{"upload_id": k, "stage": v} for k, v in pub.items()],
        "process_count": len(proc),
        "publish_count": len(pub),
    }


def active_upload_ids(max_items: int = 12) -> List[str]:
    ids = list(_process_jobs.keys()) + [u for u in _publish_jobs if u not in _process_jobs]
    return ids[:max_items]
