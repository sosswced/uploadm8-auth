"""Default garage vehicle (Ford Mustang, etc.) applies only for automotive niches."""

from __future__ import annotations

import json
from typing import Any, Optional, Tuple

AUTOMOTIVE_NICHES = frozenset({"automotive", "dashcam"})


def _as_dict(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def studio_niche_from_prefs(user_prefs: Optional[dict]) -> str:
    prefs = user_prefs if isinstance(user_prefs, dict) else {}
    strat = _as_dict(
        prefs.get("thumbnail_studio_default_strategy")
        or prefs.get("thumbnailStudioDefaultStrategy")
    )
    niche = (
        strat.get("audience_niche")
        or strat.get("audienceNiche")
        or prefs.get("audience_niche")
        or prefs.get("audienceNiche")
        or ""
    )
    return str(niche or "").strip().lower()


def studio_niche_is_automotive(user_prefs: Optional[dict]) -> bool:
    return studio_niche_from_prefs(user_prefs) in AUTOMOTIVE_NICHES


def thumbnail_category_is_automotive(ctx: Any) -> bool:
    cat = str(getattr(ctx, "thumbnail_category", "") or "").strip().lower()
    return cat in AUTOMOTIVE_NICHES


def resolve_presign_vehicle_ids(
    data: Any,
    user_prefs: Optional[dict],
) -> Tuple[Optional[int], Optional[int]]:
    """Stamp garage defaults only when Studio niche is automotive/dashcam.

    An explicit vehicle on the presign body still wins (user chose it this upload).
    """
    vm_id = getattr(data, "vehicle_make_id", None)
    vmd_id = getattr(data, "vehicle_model_id", None)
    explicit = vm_id is not None or vmd_id is not None
    if not explicit and not studio_niche_is_automotive(user_prefs):
        return None, None
    prefs = user_prefs if isinstance(user_prefs, dict) else {}
    if vm_id is None:
        vm_id = prefs.get("default_vehicle_make_id") or prefs.get("defaultVehicleMakeId")
    if vmd_id is None:
        vmd_id = prefs.get("default_vehicle_model_id") or prefs.get("defaultVehicleModelId")
    return vm_id, vmd_id
