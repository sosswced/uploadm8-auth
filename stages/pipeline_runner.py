"""
Pipeline stage runner helpers: diagnostics + timeout wrappers for worker.py.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, Optional, TypeVar

from stages.context import JobContext
from stages.errors import SkipStage, StageError
from stages.pipeline_manifest import diag_step

logger = logging.getLogger("uploadm8-worker")

T = TypeVar("T")


async def run_stage_with_diag(
    ctx: JobContext,
    *,
    stage: str,
    provider: str = "",
    coro_factory: Callable[[], Awaitable[T]],
    timeout_sec: Optional[float] = None,
    on_timeout: Optional[str] = None,
) -> T:
    """
    Run one pipeline stage with pipeline_manifest diag_step bookkeeping.
    Raises SkipStage / StageError / asyncio.TimeoutError to caller.
    """
    diag_step(ctx, stage=stage, status="started", provider=provider)
    try:
        if timeout_sec and timeout_sec > 0:
            result = await asyncio.wait_for(coro_factory(), timeout=timeout_sec)
        else:
            result = await coro_factory()
        diag_step(
            ctx,
            stage=stage,
            status="ok",
            provider=provider,
            extra={"timeout_sec": timeout_sec} if timeout_sec else None,
        )
        return result
    except asyncio.TimeoutError:
        reason = on_timeout or f"timeout after {timeout_sec}s"
        diag_step(ctx, stage=stage, status="failed", provider=provider, reason=reason)
        raise
    except SkipStage as e:
        diag_step(ctx, stage=stage, status="skipped", provider=provider, reason=e.reason)
        raise
    except StageError as e:
        diag_step(ctx, stage=stage, status="failed", provider=provider, reason=e.message)
        raise
    except Exception as e:
        diag_step(ctx, stage=stage, status="failed", provider=provider, reason=str(e)[:400])
        raise


async def persist_pipeline_manifest(pool, upload_id: str, ctx: JobContext, terminal_status: str) -> None:
    from stages.pipeline_manifest import finalize_pipeline_diag
    from stages import db as db_stage

    manifest = finalize_pipeline_diag(ctx, terminal_status=terminal_status)
    try:
        await db_stage.save_pipeline_manifest(pool, upload_id, manifest)
    except Exception as e:
        logger.debug("[%s] pipeline_manifest persist skipped: %s", upload_id, e)
