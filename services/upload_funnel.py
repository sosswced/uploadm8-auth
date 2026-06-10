"""
Upload funnel observability — durable DB events + in-memory ring buffer fallback.

Events: presign_ok, r2_complete, worker_started, stage_*, terminal_{status}.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional

logger = logging.getLogger("uploadm8-funnel")

_MAX_EVENTS_PER_UPLOAD = 64
_MAX_UPLOADS_TRACKED = 500
_lock = threading.Lock()
_events: Dict[str, Deque[dict]] = {}
_order: Deque[str] = deque()

_db_pool = None


def set_funnel_db_pool(pool) -> None:
    """Called from app/worker lifespan to enable durable funnel writes."""
    global _db_pool
    _db_pool = pool


def _memory_emit(uid: str, row: dict) -> None:
    with _lock:
        if uid not in _events:
            _events[uid] = deque(maxlen=_MAX_EVENTS_PER_UPLOAD)
            _order.append(uid)
            while len(_order) > _MAX_UPLOADS_TRACKED:
                old = _order.popleft()
                _events.pop(old, None)
        _events[uid].append(row)


async def _db_emit(uid: str, event: str, details: Dict[str, Any]) -> None:
    if _db_pool is None:
        return
    try:
        import json

        async with _db_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO upload_funnel_events (upload_id, event, details)
                VALUES ($1::uuid, $2, $3::jsonb)
                """,
                uid,
                event,
                json.dumps(details or {}),
            )
    except Exception as e:
        logger.debug("funnel db emit failed: %s", e)


_TERMINAL_FUNNEL_STATES = frozenset(
    {
        "cancelled",
        "failed",
        "completed",
        "partial",
        "success",
        "succeeded",
        "degraded",
    }
)


def emit_funnel_terminal_if_needed(upload_id: str, ctx: Any) -> None:
    """Emit ``terminal_{state}`` once per upload job (cancel/fail/success paths)."""
    if not upload_id or ctx is None:
        return
    if getattr(ctx, "_funnel_terminal_emitted", False):
        return
    state = str(getattr(ctx, "state", None) or "").strip().lower()
    if not state or state == "processing" or state == "staged":
        return
    if state not in _TERMINAL_FUNNEL_STATES:
        return
    emit_upload_funnel_event(
        str(upload_id),
        f"terminal_{state}",
        {"error_code": getattr(ctx, "error_code", None)},
    )
    try:
        setattr(ctx, "_funnel_terminal_emitted", True)
    except Exception:
        pass


def emit_upload_funnel_event(
    upload_id: str,
    event: str,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """Record a funnel event (non-blocking, best-effort)."""
    uid = str(upload_id or "").strip()
    if not uid:
        return
    row = {
        "upload_id": uid,
        "event": str(event or "").strip(),
        "ts": time.time(),
        "details": dict(details or {}),
    }
    try:
        _memory_emit(uid, row)
        pool = _db_pool
        if pool is not None:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(_db_emit(uid, row["event"], row["details"]))
            except RuntimeError:
                pass
    except Exception as e:
        logger.debug("funnel emit failed: %s", e)


async def get_upload_funnel_events_async(upload_id: str) -> List[dict]:
    """Return funnel events for an upload (DB first, memory fallback)."""
    uid = str(upload_id or "").strip()
    if not uid:
        return []
    if _db_pool is not None:
        try:
            async with _db_pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT event, ts, details
                    FROM upload_funnel_events
                    WHERE upload_id = $1::uuid
                    ORDER BY ts ASC
                    LIMIT 128
                    """,
                    uid,
                )
            if rows:
                out: List[dict] = []
                for r in rows:
                    ts = r["ts"]
                    out.append({
                        "upload_id": uid,
                        "event": r["event"],
                        "ts": ts.timestamp() if hasattr(ts, "timestamp") else time.time(),
                        "details": dict(r["details"] or {}),
                    })
                return out
        except Exception as e:
            logger.debug("funnel db read failed: %s", e)
    return get_upload_funnel_events(uid)


def get_upload_funnel_events(upload_id: str) -> List[dict]:
    """Return funnel events for an upload (newest last) from memory."""
    uid = str(upload_id or "").strip()
    with _lock:
        q = _events.get(uid)
        return list(q) if q else []


def list_recent_funnel_upload_ids(limit: int = 50) -> List[str]:
    """Most recently touched upload ids (for admin dashboards)."""
    with _lock:
        return list(_order)[-max(1, int(limit)) :]
