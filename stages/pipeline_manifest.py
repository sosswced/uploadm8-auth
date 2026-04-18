"""
Structured pipeline diagnostics: which external providers / subsystems ran and how.

Persisted on uploads.pipeline_manifest for queue UI, support, and dev iteration.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .context import JobContext


def _bool_env(name: str) -> Optional[bool]:
    raw = (os.environ.get(name) or "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return None


def init_pipeline_diag(ctx: JobContext, upload_record: dict, *, is_deferred: bool) -> None:
    """Call once after create_context + user_settings merge."""
    us = ctx.user_settings or {}
    ai_keys = (
        "aiServiceTelemetry",
        "aiServiceAudioSignals",
        "aiServiceMusicDetection",
        "aiServiceAudioSummary",
        "aiServiceCaptionWriter",
        "aiServiceThumbnailDesigner",
        "aiServiceFrameInspector",
        "aiServiceSpeechToText",
        "aiServiceVideoAnalyzer",
        "aiServiceSceneUnderstanding",
    )
    toggles: Dict[str, bool] = {}

    def _camel_to_snake(camel_key: str) -> str:
        s = camel_key[0].lower()
        for ch in camel_key[1:]:
            s += ("_" + ch.lower()) if ch.isupper() else ch
        return s

    for k in ai_keys:
        sk = _camel_to_snake(k)
        raw = us.get(k, us.get(sk, True))
        toggles[k] = bool(raw)

    ctx.pipeline_diag = {
        "v": 1,
        "modes": {
            "pipeline_path": "deferred_to_ready_to_publish" if is_deferred else "immediate_through_publish",
            "schedule_mode": (upload_record.get("schedule_mode") or "immediate"),
            "job_deferred_flag": bool(is_deferred),
            "platforms": list(ctx.platforms or []),
            "target_accounts_count": len(ctx.target_accounts or []),
            "reframe_mode": getattr(ctx, "reframe_mode", None) or "auto",
            "privacy": getattr(ctx, "privacy", None) or "public",
            "can_watermark": bool(ctx.entitlements.can_watermark) if ctx.entitlements else None,
            "can_burn_hud": bool(ctx.entitlements.can_burn_hud) if ctx.entitlements else None,
            "can_ai": bool(ctx.entitlements.can_ai) if ctx.entitlements else None,
            "user_ai_service_toggles": toggles,
            "worker_env": {
                "WORKER_PIPELINE_PROFILE": os.environ.get("WORKER_PIPELINE_PROFILE") or None,
                "WORKER_CONCURRENCY": int(os.environ.get("WORKER_CONCURRENCY", "2") or 2),
                "WORKER_HEAVY_PIPELINE_SLOTS": int(os.environ.get("WORKER_HEAVY_PIPELINE_SLOTS", "1") or 1),
                "PUBLISH_CONCURRENCY": int(os.environ.get("PUBLISH_CONCURRENCY", "5") or 5),
                "PUBLISH_PARALLEL": _bool_env("PUBLISH_PARALLEL"),
                "TWELVE_LABS_API_KEY_SET": bool((os.environ.get("TWELVE_LABS_API_KEY") or "").strip()),
                "RENDER_SERVICE_ID": os.environ.get("RENDER_SERVICE_ID") or None,
                "RENDER_INSTANCE_ID": os.environ.get("RENDER_INSTANCE_ID") or None,
            },
        },
        "steps": [],
    }


def diag_step(
    ctx: JobContext,
    *,
    stage: str,
    status: str,
    provider: str = "",
    reason: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Append one pipeline step (idempotent safe)."""
    root = getattr(ctx, "pipeline_diag", None)
    if not isinstance(root, dict):
        return
    steps = root.get("steps")
    if not isinstance(steps, list):
        steps = []
        root["steps"] = steps
    row: Dict[str, Any] = {
        "t": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stage": stage,
        "status": status,
    }
    if provider:
        row["provider"] = provider
    if reason:
        row["reason"] = (reason or "")[:400]
    if extra:
        row["extra"] = {str(k): v for k, v in list(extra.items())[:24]}
    steps.append(row)
    if len(steps) > 150:
        del steps[:-150]


def finalize_pipeline_diag(ctx: JobContext, *, terminal_status: str) -> Dict[str, Any]:
    """Attach outcome summary from ctx; return manifest for DB JSON."""
    root = getattr(ctx, "pipeline_diag", None)
    if not isinstance(root, dict):
        root = {"v": 1, "modes": {}, "steps": []}
    ac = getattr(ctx, "audio_context", None) or {}
    vc = getattr(ctx, "vision_context", None) or {}
    vu = getattr(ctx, "video_understanding", None) or {}
    vi = getattr(ctx, "video_intelligence_context", None) or {}
    steps = root.get("steps") if isinstance(root.get("steps"), list) else []
    skipped_or_failed = sum(
        1
        for s in steps
        if isinstance(s, dict) and (s.get("status") in ("skipped", "failed", "partial"))
    )
    root["outcome"] = {
        "terminal_status": terminal_status,
        "upload_id": str(ctx.upload_id),
        "has_audio_context": bool(ac),
        "has_vision_context": bool(vc) and not (isinstance(vc, dict) and vc.get("skipped")),
        "has_twelvelabs": bool(vu) and not (isinstance(vu, dict) and vu.get("error")),
        "has_video_intelligence": bool(vi) and not (isinstance(vi, dict) and vi.get("error")),
        "platform_results_count": len(getattr(ctx, "platform_results", None) or []),
        "publish_success_any": bool(
            [r for r in (getattr(ctx, "platform_results", None) or []) if getattr(r, "success", False)]
        ),
        "pipeline_expectations": {
            "note": (
                "Stages are best-effort: timeouts, missing API keys, or user toggles may skip or "
                "partially run a feature while the upload still completes."
            ),
            "steps_skipped_or_partial": skipped_or_failed,
        },
    }
    return root
