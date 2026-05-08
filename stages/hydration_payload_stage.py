"""Build canonical ``ctx.hydration_payload`` once multimodal signals are ready (before thumbnails)."""

from __future__ import annotations

import logging

from .caption_stage import _detect_content_category
from .context import JobContext
from services.hydration_payload import (
    _minimal_hydration_payload,
    build_hydration_payload,
    persist_hydration_payload_artifact,
    validate_hydration_payload,
)

logger = logging.getLogger("uploadm8-worker.hydration_payload_stage")


async def run_hydration_payload_stage(ctx: JobContext) -> JobContext:
    from core.config import HYDRATION_PAYLOAD_ENABLED

    if not HYDRATION_PAYLOAD_ENABLED:
        return ctx
    ctx.mark_stage("hydration_payload")

    uid = str(getattr(ctx, "upload_id", "") or "")

    # If the payload was pre-seeded (e.g. from an API hint), use its category and source.
    hp0 = getattr(ctx, "hydration_payload", None)
    if isinstance(hp0, dict) and str(hp0.get("category") or "").strip():
        category = str(hp0["category"]).strip().lower()
        category_source = str(hp0.get("category_source") or "pre_seeded")
    else:
        category = _detect_content_category(ctx)
        category_source = "detector"

    logger.info("[%s] hydration_payload_stage: category=%r source=%s", uid, category, category_source)
    ctx.thumbnail_category = category

    hp = None
    try:
        hp = build_hydration_payload(ctx, category=category, category_source=category_source)
        ok, reason = validate_hydration_payload(hp)
        if not ok:
            logger.warning(
                "[hydration_payload_stage] payload failed validation (%s) — using minimal fallback",
                reason,
            )
            hp = None
    except Exception as exc:
        logger.warning(
            "[hydration_payload_stage] build_hydration_payload raised %r — using minimal fallback",
            exc,
        )

    if hp is None:
        hp = _minimal_hydration_payload(
            category=category,
            trace_id=str(getattr(ctx, "upload_id", "") or ""),
        )

    ctx.hydration_payload = hp
    persist_hydration_payload_artifact(ctx)
    logger.debug(
        "[hydration_payload_stage] done: category=%r source=%s anchor_len=%d",
        hp.get("category"),
        hp.get("category_source"),
        len(str(hp.get("anchor_phrase") or "")),
    )
    return ctx
