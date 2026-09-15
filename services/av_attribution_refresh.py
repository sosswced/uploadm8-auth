"""
Merge AV training pack coarse flags into content_attribution after pack write.

Caption attribution runs before the pack stage; this patch keeps strategy_key
stable (pack fields are hash-excluded) while logging pack_* for content_success.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

logger = logging.getLogger("uploadm8-worker")


def refresh_attribution_pack_flags(ctx: Any) -> Dict[str, Any]:
    """
    Update content_attribution_v1 pack_* fields from av_training_pack_v1 meta.
    Does not change content_attribution_key (pack fields are hash-excluded).
    """
    arts = getattr(ctx, "output_artifacts", None)
    if not isinstance(arts, dict):
        return {"ok": False, "reason": "no_artifacts"}
    pack = arts.get("av_training_pack_v1")
    if isinstance(pack, str) and pack.strip():
        try:
            pack = json.loads(pack)
        except Exception:
            pack = {}
    if not isinstance(pack, dict) or not pack:
        return {"ok": False, "reason": "no_pack"}

    from services.av_read_soft_bias import coarse_pack_flags

    flags = coarse_pack_flags(arts)
    raw = arts.get("content_attribution_v1")
    snap: Dict[str, Any] = {}
    if isinstance(raw, dict):
        snap = dict(raw)
    elif isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                snap = parsed
        except Exception:
            snap = {}
    if not snap:
        return {"ok": False, "reason": "no_attribution_snap"}

    snap["pack_present"] = int(flags.get("pack_present") or 0)
    snap["pack_tl_status"] = str(flags.get("pack_tl_status") or "na")[:16]
    snap["pack_needs_deep_teacher"] = int(flags.get("pack_needs_deep_teacher") or 0)
    snap["pack_hero_class"] = str(flags.get("pack_hero_class") or "")[:32]
    arts["content_attribution_v1"] = json.dumps(snap, default=str)
    # Keep existing strategy key — pack fields excluded from hash by design.
    return {
        "ok": True,
        "pack_present": snap["pack_present"],
        "pack_tl_status": snap["pack_tl_status"],
        "pack_hero_class": snap["pack_hero_class"],
    }


__all__ = ["refresh_attribution_pack_flags"]
