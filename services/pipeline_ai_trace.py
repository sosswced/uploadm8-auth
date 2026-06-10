"""
Opt-in accumulation of `[ai-trace]` events plus a single per-job `[ai-pipeline-summary]` line.

Controlled by AI_TRACE_ENABLED. Per-stage `[ai-trace]` log lines optional via AI_TRACE_PER_STAGE.
Final JSON blob optional via AI_TRACE_PIPELINE_SUMMARY.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _env_bool(name: str, default: str) -> bool:
    return os.environ.get(name, default).lower() in ("1", "true", "yes", "on")


def ai_trace_enabled(*, tier: Optional[str] = None) -> bool:
    if _env_bool("AI_TRACE_ENABLED", "false"):
        return True
    from core.pipeline_env_defaults import env_str

    tiers_raw = env_str(
        "AI_TRACE_TIERS",
        "pro,studio,agency,master_admin,friends_family",
    ).strip().lower()
    if tiers_raw and tier:
        allowed = {t.strip() for t in tiers_raw.split(",") if t.strip()}
        if (tier or "").strip().lower() in allowed:
            return True
    return False


def ai_trace_enabled_for_ctx(ctx) -> bool:
    tier = ""
    try:
        ent = getattr(ctx, "entitlements", None)
        if ent is not None:
            tier = str(getattr(ent, "tier", "") or "")
    except Exception:
        tier = ""
    arts = getattr(ctx, "output_artifacts", None) or {}
    if isinstance(arts, dict) and str(arts.get("ai_trace_requested") or "").lower() in ("1", "true", "yes"):
        return True
    return ai_trace_enabled(tier=tier)


def ai_trace_preview_chars() -> int:
    return int(os.environ.get("AI_TRACE_PREVIEW_CHARS", "280") or 280)


def ai_trace_per_stage_log() -> bool:
    return _env_bool("AI_TRACE_PER_STAGE", "true")


def ai_trace_pipeline_summary_enabled() -> bool:
    return _env_bool("AI_TRACE_PIPELINE_SUMMARY", "true")


def ai_trace_summary_max_log_chars() -> int:
    return int(os.environ.get("AI_TRACE_SUMMARY_MAX_LOG_CHARS", "65536") or 65536)


def ai_trace_artifacts_max_chars() -> int:
    return int(os.environ.get("AI_TRACE_ARTIFACTS_MAX_CHARS", "490000") or 490000)


def truncate_ai_trace_payload(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    preview = ai_trace_preview_chars()
    safe: Dict[str, Any] = {}
    for k, v in (payload or {}).items():
        if isinstance(v, str):
            safe[k] = v[:preview] if preview > 0 else ""
        else:
            safe[k] = v
    return safe


def record_ai_pipeline_trace(
    ctx: Optional[Any],
    upload_id: str,
    stage: str,
    payload: Optional[Dict[str, Any]],
    *,
    log: Optional[logging.Logger] = None,
) -> None:
    if ctx is not None and not ai_trace_enabled_for_ctx(ctx):
        return
    if ctx is None and not ai_trace_enabled():
        return
    safe = truncate_ai_trace_payload(payload)
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "payload": safe,
    }
    if ctx is not None:
        buf = getattr(ctx, "pipeline_ai_trace", None)
        if isinstance(buf, list):
            buf.append(entry)

    if ai_trace_per_stage_log() and log is not None:
        log.info("[%s] [ai-trace] %s %s", upload_id, stage, json.dumps(safe, default=str))


def _count_events_by_stage(events: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for e in events:
        key = str(e.get("stage") or "")
        counts[key] = counts.get(key, 0) + 1
    return counts


async def emit_ai_pipeline_summary(
    ctx: Optional[Any],
    upload_id: str,
    log: logging.Logger,
    db_pool: Optional[Any] = None,
) -> None:
    if ctx is None or not ai_trace_enabled_for_ctx(ctx) or not ai_trace_pipeline_summary_enabled():
        return
    events = list(getattr(ctx, "pipeline_ai_trace", None) or [])
    uid = str(upload_id or getattr(ctx, "upload_id", "") or "")
    summary: Dict[str, Any] = {
        "upload_id": uid,
        "user_id": str(getattr(ctx, "user_id", "") or ""),
        "job_id": str(getattr(ctx, "job_id", "") or ""),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "event_count": len(events),
        "by_stage": _count_events_by_stage(events),
        "events": events,
    }

    max_art = ai_trace_artifacts_max_chars()
    try:
        blob = json.dumps(summary, default=str)
        if len(blob) <= max_art:
            ctx.output_artifacts["ai_pipeline_trace_v1"] = blob
        else:
            ctx.output_artifacts["ai_pipeline_trace_v1"] = (
                blob[: max_art - len("...[truncated_ai_pipeline_trace]")]
                + "...[truncated_ai_pipeline_trace]"
            )
    except Exception:
        log.debug("[%s] ai_pipeline_trace artifact failed", uid, exc_info=True)

    max_log = ai_trace_summary_max_log_chars()
    art_key = "ai_pipeline_trace_v1"
    try:
        line = json.dumps(summary, default=str)
    except Exception:
        log.warning("[%s] [ai-pipeline-summary] serialization failed", uid, exc_info=True)
        return
    if len(line) <= max_log:
        log.info("[%s] [ai-pipeline-summary] %s", uid, line)
    else:
        head = line[: max(0, max_log - len("...(truncated)"))]
        log.info("[%s] [ai-pipeline-summary] %s...(truncated,len=%s)", uid, head, len(line))

    if db_pool is not None and art_key in getattr(ctx, "output_artifacts", {}):
        try:
            from stages import db as db_stage

            await db_stage.merge_job_output_artifacts_strings(
                db_pool, uid, {art_key: ctx.output_artifacts[art_key]}
            )
        except Exception:
            log.debug("[%s] ai_pipeline_trace DB merge skipped", uid, exc_info=True)
