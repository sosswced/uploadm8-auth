"""
Hard pre-publish gate for metadata quality (#14–15).

Hydration may rewrite weak copy, but publish must not ship obvious boilerplate
when ``collect_evidence`` shows real signals (speed, geo, music, vision, etc.).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from core.helpers import strip_stray_hashtag_json_blob

from stages.context import JobContext
from stages.errors import ErrorCode, PublishError

from services.hydration_enforcer import (
    collect_evidence,
    _caption_uses_evidence,
    _is_generic_caption,
)


def publish_metadata_strict_enabled() -> bool:
    v = os.environ.get("UPLOADM8_PUBLISH_METADATA_STRICT", "1").strip().lower()
    return v not in ("0", "false", "no", "off", "")


def _gate_caption(ctx: JobContext, platform: str) -> str:
    raw = (
        ctx.get_effective_caption(platform=platform)
        if hasattr(ctx, "get_effective_caption")
        else (
            getattr(ctx, "ai_caption", None)
            or getattr(ctx, "caption", None)
            or getattr(ctx, "description", None)
            or ""
        )
    )
    return strip_stray_hashtag_json_blob(str(raw or "").strip())


def _gate_title(ctx: JobContext, platform: str) -> str:
    if hasattr(ctx, "get_effective_title"):
        return str(ctx.get_effective_title(platform=platform) or "").strip()
    return str(
        getattr(ctx, "ai_title", None)
        or getattr(ctx, "title", None)
        or getattr(ctx, "video_title", None)
        or getattr(ctx, "name", None)
        or ""
    ).strip()


def evaluate_publish_metadata_gate(
    ctx: JobContext, publish_targets: List[Tuple[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Return a structured block payload when strict gate fails, else None."""
    if not publish_metadata_strict_enabled():
        return None
    pool = collect_evidence(ctx)
    if not pool.has_any_evidence():
        return None

    failures: List[Dict[str, Any]] = []
    seen: set = set()
    for platform, _ in publish_targets or []:
        pl = str(platform or "").strip().lower()
        if pl in seen:
            continue
        seen.add(pl)
        cap = _gate_caption(ctx, pl)
        ttl = _gate_title(ctx, pl)
        reasons: List[str] = []

        if not cap.strip():
            reasons.append("empty_caption")
        elif _is_generic_caption(cap) and not _caption_uses_evidence(cap, pool):
            reasons.append("generic_caption_without_evidence")

        # Short titles are often intentional placeholders; only enforce the
        # generic detector when the string is long enough for it to be meaningful.
        if len(ttl) >= 12:
            if _is_generic_caption(ttl) and not _caption_uses_evidence(ttl, pool):
                reasons.append("generic_title_without_evidence")

        if reasons:
            failures.append({"platform": pl or "(default)", "reasons": reasons})

    if not failures:
        return None
    return {
        "strict": True,
        "failures": failures,
        "evidence_present": True,
    }


def assert_publish_metadata_gate(
    ctx: JobContext, publish_targets: List[Tuple[str, Any]]
) -> None:
    """Raise :class:`PublishError` when strict metadata gate fails."""
    payload = evaluate_publish_metadata_gate(ctx, publish_targets)
    if not payload:
        return
    arts = getattr(ctx, "output_artifacts", None)
    if isinstance(arts, dict):
        try:
            arts["publish_metadata_gate"] = json.dumps(payload, default=str)[:12000]
        except Exception:
            arts["publish_metadata_gate"] = '{"error":"publish_metadata_gate_json_failed"}'
    raise PublishError(
        "Publish blocked: caption/title is generic boilerplate while rich evidence "
        "exists. Re-run caption/hydration or disable UPLOADM8_PUBLISH_METADATA_STRICT=0 for emergency.",
        code=ErrorCode.PUBLISH_METADATA_REJECTED,
        meta={"publish_metadata_gate": payload},
        retryable=False,
    )


__all__ = [
    "assert_publish_metadata_gate",
    "evaluate_publish_metadata_gate",
    "publish_metadata_strict_enabled",
]
