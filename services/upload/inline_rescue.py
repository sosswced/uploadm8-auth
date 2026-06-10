"""Dev-only inline rescue when worker loop is unavailable."""

import asyncio
import logging
import os
from typing import Dict

import core.state
from core.helpers import _safe_json

logger = logging.getLogger("uploadm8-api")

_INLINE_RESCUE_ENABLED = os.environ.get("UPLOAD_INLINE_RESCUE_ENABLED", "true").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
_INLINE_RESCUE_DELAY_SEC = max(30, int(os.environ.get("UPLOAD_INLINE_RESCUE_DELAY_SEC", "150") or 150))


async def inline_rescue_if_stuck(upload_id: str, user_id: str, user_prefs: Dict[str, object], ent) -> None:
    """
    DEV ONLY — safety net for local/single-process setups.

    If a just-completed immediate upload remains at stage ``upload`` for too long,
    run the processing pipeline inline from API so thumbnails/persona are still produced.
    Not intended for production multi-worker deployments.
    """
    if not _INLINE_RESCUE_ENABLED:
        return
    if core.state.db_pool is None:
        return
    await asyncio.sleep(float(_INLINE_RESCUE_DELAY_SEC))
    try:
        async with core.state.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, status, processing_stage, processing_progress, updated_at,
                       processing_started_at, thumbnail_r2_key, output_artifacts
                FROM uploads
                WHERE id = $1::uuid AND user_id = $2::uuid
                """,
                upload_id,
                user_id,
            )
        if not row:
            return
        status = str(row.get("status") or "").lower()
        stage = str(row.get("processing_stage") or "").lower()
        progress = int(row.get("processing_progress") or 0)
        thumb_key = str(row.get("thumbnail_r2_key") or "").strip()
        arts = _safe_json(row.get("output_artifacts"), {}) or {}
        if thumb_key:
            return
        if isinstance(arts, dict) and (
            arts.get("thumbnail_trace")
            or arts.get("pikzels_prompt_by_platform")
            or arts.get("hydration_payload")
        ):
            return
        if status not in ("queued", "processing", "staged", "pending"):
            return
        if stage not in ("", "upload"):
            return
        if progress not in (0, 87):
            return

        logger.warning(
            "[%s] inline-rescue: stuck at stage=%s progress=%s status=%s after %ss — running pipeline inline",
            upload_id,
            stage or "-",
            progress,
            status,
            _INLINE_RESCUE_DELAY_SEC,
        )
        import worker as worker_runtime

        worker_runtime.db_pool = core.state.db_pool
        worker_runtime.redis_client = core.state.redis_client

        job_data = {
            "upload_id": upload_id,
            "user_id": user_id,
            "preferences": user_prefs or {},
            "priority_class": getattr(ent, "priority_class", "normal"),
            "plan_features": {
                "ai": getattr(ent, "can_ai", True),
                "priority": getattr(ent, "can_priority", False),
                "watermark": getattr(ent, "can_watermark", True),
                "ai_depth": getattr(ent, "ai_depth", "standard"),
                "caption_frames": getattr(ent, "max_caption_frames", 6),
            },
        }
        ok = await worker_runtime.run_processing_pipeline(job_data)
        logger.warning("[%s] inline-rescue pipeline finished ok=%s", upload_id, bool(ok))
    except Exception as exc:
        logger.exception("[%s] inline-rescue failed: %s", upload_id, exc)
