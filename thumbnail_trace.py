"""Thumbnail → Pikzels diagnostics: ``thumbnail_trace`` + ``pikzels_prompt_by_platform`` artifacts."""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("uploadm8-worker.thumb-trace")

_THUMB_TRACE_JSON_CAP = 120_000
_PIKZELS_PROMPT_MAP_CAP = 48_000
_MAX_EVENTS = 48


def trace_append(ctx: Any, event: str, data: Optional[Dict[str, Any]] = None) -> None:
    """Append one trace row and persist under ``thumbnail_trace``."""
    arts = getattr(ctx, "output_artifacts", None)
    if not isinstance(arts, dict):
        return

    raw = arts.get("thumbnail_trace")
    events: List[Dict[str, Any]] = []
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                events = parsed
        except (json.JSONDecodeError, TypeError):
            events = []

    row: Dict[str, Any] = {"event": event}
    if data:
        row.update(data)
    events.append(row)
    if len(events) > _MAX_EVENTS:
        events = events[-_MAX_EVENTS :]

    try:
        arts["thumbnail_trace"] = json.dumps(events, default=str)[:_THUMB_TRACE_JSON_CAP]
    except (TypeError, ValueError):
        return

    uid = getattr(ctx, "upload_id", "") or ""
    preview = {k: row[k] for k in list(row.keys())[:12]}
    logger.info("[thumb-trace] upload=%s %s", uid, preview)

    # Self-persist immediately so the trace survives any stage failure and any
    # orchestrator drift (older worker.py deployments). Fire-and-forget; the
    # helper no-ops when no db_pool is attached.
    try:
        from services.diag_persist import schedule_persist_artifact_now

        schedule_persist_artifact_now(ctx, "thumbnail_trace")
    except Exception:
        pass


def trace_sink_factory(ctx: Any) -> Callable[[str, Optional[Dict[str, Any]]], None]:
    """Build a callback for subsystems that emit ``(event, payload)`` trace rows."""

    def _sink(event: str, data: Optional[Dict[str, Any]] = None) -> None:
        trace_append(ctx, event, data)

    return _sink


def persist_pikzels_prompt_for_platform(ctx: Any, platform: str, prompt: str) -> None:
    """Merge per-platform final Pikzels prompt strings into ``pikzels_prompt_by_platform``."""
    arts = getattr(ctx, "output_artifacts", None)
    if not isinstance(arts, dict):
        return
    plat = (platform or "").strip().lower()
    if not plat:
        return

    m: Dict[str, str] = {}
    raw = arts.get("pikzels_prompt_by_platform")
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                m = {str(k): str(v) for k, v in parsed.items()}
        except (json.JSONDecodeError, TypeError):
            m = {}

    m[plat] = str(prompt or "")
    try:
        arts["pikzels_prompt_by_platform"] = json.dumps(m, ensure_ascii=False)[:_PIKZELS_PROMPT_MAP_CAP]
    except (TypeError, ValueError):
        return

    try:
        from services.diag_persist import schedule_persist_artifact_now

        schedule_persist_artifact_now(ctx, "pikzels_prompt_by_platform")
    except Exception:
        pass
