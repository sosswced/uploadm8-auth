"""
P8 propose-only AV read proposals — never ship without identity merge + grounding.

Flag: AV_READ_PROPOSE_ONLY=1 to attach proposals on ctx; default off.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List


def _enabled() -> bool:
    try:
        from services.av_read_runtime_flags import flag_enabled

        return flag_enabled("AV_READ_PROPOSE_ONLY")
    except Exception:
        return (os.environ.get("AV_READ_PROPOSE_ONLY") or "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )


def build_proposals_from_pack_meta(ctx: Any) -> Dict[str, Any]:
    """Return propose-only structure from pack meta / identity — judge still required."""
    if not _enabled():
        return {"enabled": False, "proposals": []}
    arts = getattr(ctx, "output_artifacts", None) or {}
    if not isinstance(arts, dict):
        return {"enabled": True, "proposals": []}
    pack = arts.get("av_training_pack_v1") if isinstance(arts.get("av_training_pack_v1"), dict) else {}
    identity = arts.get("content_identity_v1") if isinstance(arts.get("content_identity_v1"), dict) else {}
    proposals: List[Dict[str, Any]] = []
    for h in (identity.get("hero_facts") or [])[:3]:
        if isinstance(h, dict) and h.get("text"):
            proposals.append(
                {
                    "kind": "hero_fact",
                    "text": str(h.get("text"))[:200],
                    "class": h.get("class"),
                    "status": "propose_only",
                }
            )
    if pack.get("needs_deep_teacher"):
        proposals.append({"kind": "needs_deep_teacher", "value": True, "status": "propose_only"})
    return {"enabled": True, "proposals": proposals, "judge": "identity+grounding_required"}


__all__ = ["build_proposals_from_pack_meta"]
