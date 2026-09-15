"""
Teacher disagreement / poison filter for AV training packs (data-engine harden).

Drops rows from distill training when teacher lanes are thin or captions are
poorly grounded — so soft-bias/skip-TL later cannot learn from garbage packs.

Does NOT use engagement/views (dirty analytics risk). Labels stay teacher-only.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

# Default floor aligned with coach "low grounding" narrative (~0.35).
DEFAULT_MIN_GROUNDING_SCORE = 0.35
DEFAULT_MIN_EVIDENCE_LANES = 1


def _as_float(val: Any) -> Optional[float]:
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def pack_row_refuse_reason(
    row: Dict[str, Any],
    *,
    min_grounding_score: float = DEFAULT_MIN_GROUNDING_SCORE,
    min_evidence_lanes: int = DEFAULT_MIN_EVIDENCE_LANES,
) -> Optional[str]:
    """
    Return a short refuse reason, or None if the row may train.

    Fail-closed on missing grounding when band is explicitly low; allow mid/high
    and unknown band only when score meets floor or lanes look healthy.
    """
    if not isinstance(row, dict) or not row:
        return "empty_row"

    band = str(row.get("grounding_band") or "").strip().lower()
    score = _as_float(row.get("grounding_score"))
    try:
        lanes = int(row.get("evidence_lane_count") or 0)
    except (TypeError, ValueError):
        lanes = 0
    hero = str(row.get("identity_hero_fact_class") or "").strip().lower()
    conf = str(row.get("identity_confidence") or "").strip().lower()
    tl = str(row.get("tl_status") or "").strip().lower()

    if band == "low":
        return "grounding_band_low"
    if score is not None and score < float(min_grounding_score):
        return f"grounding_score_below_{min_grounding_score}"

    if lanes < int(min_evidence_lanes):
        return "evidence_lanes_thin"

    if hero in ("", "unknown", "na") and band != "high":
        return "missing_hero_class"

    if conf == "low" and lanes < 2:
        return "identity_low_conf_thin_lanes"

    # TL failed with no fusion backup signal is weak teacher agreement.
    fusion = str(row.get("fusion_source") or "").strip().lower()
    if tl == "failed" and fusion in ("", "none", "unknown"):
        return "tl_failed_without_fusion"

    return None


def filter_pack_training_rows(
    rows: list,
    *,
    min_grounding_score: float = DEFAULT_MIN_GROUNDING_SCORE,
    min_evidence_lanes: int = DEFAULT_MIN_EVIDENCE_LANES,
) -> Tuple[list, list]:
    """Split into (accepted, refused) where refused items are {row, reason}."""
    accepted = []
    refused = []
    for row in rows or []:
        if not isinstance(row, dict):
            refused.append({"row": row, "reason": "not_dict"})
            continue
        reason = pack_row_refuse_reason(
            row,
            min_grounding_score=min_grounding_score,
            min_evidence_lanes=min_evidence_lanes,
        )
        if reason:
            refused.append({"row": row, "reason": reason})
        else:
            accepted.append(row)
    return accepted, refused


__all__ = [
    "DEFAULT_MIN_GROUNDING_SCORE",
    "DEFAULT_MIN_EVIDENCE_LANES",
    "pack_row_refuse_reason",
    "filter_pack_training_rows",
]
