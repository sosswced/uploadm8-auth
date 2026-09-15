"""
P9 Hub promote / canary gates for AV-read distill.

Fail-closed: never promote unless floors + explicit env go.
Student never becomes judge — promote only ships a *published* model artifact
for soft-bias / skip-TL canary, not for identity/grounding bypass.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional


def promote_floors_met(
    *,
    hero_fact_f1: Optional[float] = None,
    grounding_ok: bool = False,
    cohort_n: int = 0,
    min_f1: float = 0.55,
    min_cohort: int = 200,
) -> Dict[str, Any]:
    reasons = []
    ok = True
    if not grounding_ok:
        ok = False
        reasons.append("grounding_floor_not_met")
    try:
        f1 = float(hero_fact_f1) if hero_fact_f1 is not None else None
    except (TypeError, ValueError):
        f1 = None
    if f1 is None or f1 < min_f1:
        ok = False
        reasons.append(f"hero_fact_f1_below_{min_f1}")
    if int(cohort_n or 0) < int(min_cohort):
        ok = False
        reasons.append(f"cohort_below_{min_cohort}")
    return {"ok": ok, "reasons": reasons, "f1": f1, "cohort_n": int(cohort_n or 0)}


def hub_promote_allowed(floors: Dict[str, Any]) -> Dict[str, Any]:
    """
    Explicit dual gate: floors + UM8_AV_READ_HUB_PROMOTE=1 (env or admin bundle).
    Default: refuse (trained_not_published stays local).
    """
    try:
        from services.av_read_runtime_flags import flag_enabled

        go = flag_enabled("UM8_AV_READ_HUB_PROMOTE")
    except Exception:
        go = (os.environ.get("UM8_AV_READ_HUB_PROMOTE") or "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
    if not go:
        return {
            "promote": False,
            "status": "trained_not_published",
            "reason": "hub_promote_env_off",
        }
    if not floors.get("ok"):
        return {
            "promote": False,
            "status": "trained_not_published",
            "reason": "floors_not_met",
            "floor_reasons": list(floors.get("reasons") or []),
        }
    return {
        "promote": True,
        "status": "promote_allowed",
        "reason": "floors_and_env_go",
        "note": "operator must still canary; never bypass identity/grounding",
    }


__all__ = ["promote_floors_met", "hub_promote_allowed"]
