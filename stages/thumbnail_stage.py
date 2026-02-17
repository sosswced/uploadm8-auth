"""stages/telemetry_stage.py

UploadM8 Telemetry Stage
-----------------------

This module MUST export `run_telemetry_stage(ctx)` because worker.py imports it.

Design goals (operational):
- Never crash the pipeline if telemetry is missing or malformed.
- Ensure downstream stages can safely reference `ctx.telemetry`.

Telemetry parsing for .map files can be expanded later. For now we do a
best-effort read + minimal summary.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("uploadm8-worker")


def _ensure_ctx_fields(ctx: Any) -> None:
    """Ensure common fields exist to prevent AttributeErrors downstream."""
    if not hasattr(ctx, "telemetry") or getattr(ctx, "telemetry") is None:
        setattr(ctx, "telemetry", {"data_points": [], "summary": {}, "skipped": True})

    if not hasattr(ctx, "output_artifacts") or getattr(ctx, "output_artifacts") is None:
        setattr(ctx, "output_artifacts", {})


def _safe_path(p: Any) -> Optional[Path]:
    if p is None:
        return None
    try:
        return p if isinstance(p, Path) else Path(str(p))
    except Exception:
        return None


def _summarize_map_file(path: Path) -> Dict[str, Any]:
    """Minimal .map summary without making assumptions about file format."""
    size = path.stat().st_size
    # Read a small chunk to avoid memory blowups.
    try:
        with path.open("rb") as f:
            head = f.read(4096)
    except Exception:
        head = b""

    summary: Dict[str, Any] = {
        "file": str(path),
        "bytes": int(size),
        "head_bytes": int(len(head)),
    }

    # Heuristic: count lines if it's text-like
    try:
        text = head.decode("utf-8", errors="ignore")
        summary["head_preview"] = text[:300]
        summary["head_lines"] = int(text.count("\n"))
    except Exception:
        pass

    return summary


async def run_telemetry_stage(ctx: Any) -> Any:
    """Worker entrypoint: compute telemetry artifacts and attach to ctx."""
    _ensure_ctx_fields(ctx)

    telem_path = _safe_path(getattr(ctx, "local_telemetry_path", None))
    if not telem_path or not telem_path.exists():
        # No telemetry is a normal condition.
        ctx.telemetry = {"data_points": [], "summary": {}, "skipped": True}
        return ctx

    try:
        summary = _summarize_map_file(telem_path)
        # Keep schema stable for downstream HUD stage.
        ctx.telemetry = {
            "data_points": [],
            "summary": summary,
            "skipped": False,
        }
        ctx.output_artifacts["telemetry"] = {"source": str(telem_path)}
        logger.info(f"Telemetry stage completed (bytes={summary.get('bytes')})")
        return ctx

    except Exception as e:
        # Never fail pipeline for telemetry.
        logger.warning(f"Telemetry stage failed (non-fatal): {e}")
        ctx.telemetry = {"data_points": [], "summary": {"error": str(e)}, "skipped": True}
        return ctx
