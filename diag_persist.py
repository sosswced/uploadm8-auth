"""
Stage-level self-persistence for diagnostic artifacts.

Every diag artifact (``hydration_payload``, ``studio_render_report``,
``thumbnail_trace``, ``thumbnail_brief_json``, ``pikzels_prompt_by_platform``,
``hydration_report``) is generated into ``ctx.output_artifacts`` while a stage
runs. Historically the only path that flushed these to ``uploads.output_artifacts``
was ``stages.db.save_generated_metadata`` at the very end of caption_stage. If
caption_stage skipped, errored, or an older worker was deployed, the artifact
existed in memory and then died with the process — the admin upload-trace
showed ``0 ai / 0 thumb / missing artifacts`` even after Pikzels actually ran.

This module gives every producer stage a one-liner:

    from services.diag_persist import persist_artifact_now
    await persist_artifact_now(ctx, "studio_render_report")

It pulls ``db_pool`` off ``ctx._db_pool`` (set by ``worker.run_processing_pipeline``)
and merges the in-memory value into ``uploads.output_artifacts`` immediately. It
silently no-ops when no pool is attached (tests, dry-runs, sub-pipelines), so
stages remain safe to call from any orchestrator.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Iterable, Optional

logger = logging.getLogger("uploadm8-worker.diag-persist")


def _coerce_to_string_value(value: Any) -> Optional[str]:
    """Coerce an artifact value to the ``str`` shape expected by JSONB merge."""
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value if value.strip() else None
    try:
        return json.dumps(value, default=str, ensure_ascii=False)
    except Exception:
        try:
            return str(value)
        except Exception:
            return None


async def persist_artifact_now(ctx: Any, *keys: str) -> None:
    """Merge selected ``ctx.output_artifacts`` keys into ``uploads.output_artifacts``.

    Best-effort: any failure is logged at DEBUG and swallowed. Safe to call
    repeatedly — JSONB ``||`` is idempotent for identical values.
    """
    if not keys:
        return
    arts = getattr(ctx, "output_artifacts", None)
    if not isinstance(arts, dict):
        return
    upload_id = getattr(ctx, "upload_id", None)
    if not upload_id:
        return
    pool = getattr(ctx, "_db_pool", None)
    if pool is None:
        # Older worker.py won't have set ``_db_pool`` — try the well-known
        # globals as fallback so stages still self-persist on stale deploys.
        try:
            import worker as _worker  # type: ignore

            pool = getattr(_worker, "db_pool", None)
        except Exception:
            pool = None
        if pool is None:
            try:
                from core import state as _core_state  # type: ignore

                pool = getattr(_core_state, "db_pool", None)
            except Exception:
                pool = None
    if pool is None:
        return

    patch: dict = {}
    for k in keys:
        if not k:
            continue
        coerced = _coerce_to_string_value(arts.get(k))
        if coerced is not None:
            patch[k] = coerced
    if not patch:
        return

    try:
        from stages import db as _db_stage  # local import to avoid cycles

        await _db_stage.merge_job_output_artifacts_strings(pool, str(upload_id), patch)
    except Exception as exc:
        logger.debug("[%s] persist_artifact_now skipped (%s): %s", upload_id, list(patch.keys()), exc)


def schedule_persist_artifact_now(ctx: Any, *keys: str) -> None:
    """Sync-friendly variant: schedule ``persist_artifact_now`` on the running loop.

    For call sites inside synchronous helpers (``trace_append`` is sync, but
    every producer is invoked from async stages) we want a fire-and-forget
    flush without forcing every caller to ``await``. Falls back to a noop when
    no event loop is available (tests).
    """
    if not keys:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    try:
        loop.create_task(persist_artifact_now(ctx, *keys))
    except Exception as exc:
        logger.debug("schedule_persist_artifact_now skipped: %s", exc)


def list_known_diag_keys() -> Iterable[str]:
    """Single source of truth for the diag artifact key set."""
    return (
        "hydration_payload",
        "hydration_report",
        "studio_render_report",
        "thumbnail_brief_json",
        "thumbnail_trace",
        "thumbnail_render_method",
        "thumbnail_selection_method",
        "thumbnail_category",
        "platform_thumbnail_map",
        "platform_thumbnail_r2_keys",
        "thumbnail_r2_candidates",
        "pikzels_prompt_by_platform",
        "provider_error_trace",
        "ai_pipeline_trace_v1",
    )
