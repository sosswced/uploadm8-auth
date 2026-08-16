"""Deterministic driving evidence — dashcam / HUD / .map, not GPS speed alone.

Peak MPH on a backpack or glasses walk is not a car. Callers that used to
stamp ``automotive`` from ``peak_mph >= 10`` must go through this helper.
"""

from __future__ import annotations

from typing import Any, Iterable

# Union of Vision ambient tokens + depth-router filename tokens.
DASHCAM_FILENAME_TOKENS = (
    "CAM_",
    "DASHCAM",
    "DASH",
    "M8_",
    "ESCORT",
    "DRIVECAM",
    "BLACKVU",
    "BLACKVUE",
    "THINKWARE",
    "GOPRO",
    "DRIFT",
    "_EVNT",
)

WINDSHIELD_LABEL_MARKERS = (
    "windshield",
    "windscreen",
    "rear-view mirror",
    "rearview mirror",
    "automotive exterior",
    "automotive mirror",
    "automotive side-view mirror",
    "hood",
)


def _filename_is_dashcam(filename: str) -> bool:
    fname = str(filename or "").upper()
    if not fname:
        return False
    return any(tok in fname for tok in DASHCAM_FILENAME_TOKENS)


def _osd_hud_present(ctx: Any) -> bool:
    osd = getattr(ctx, "dashcam_osd_context", None) or {}
    if not isinstance(osd, dict) or not osd or osd.get("skipped"):
        return False
    gps_path = osd.get("gps_path") or []
    if isinstance(gps_path, (list, tuple)) and len(gps_path) >= 1:
        return True
    try:
        if float(osd.get("max_speed_mph") or 0) >= 5:
            return True
    except (TypeError, ValueError):
        pass
    return False


def _map_points_present(ctx: Any) -> bool:
    tel = getattr(ctx, "telemetry", None) or getattr(ctx, "telemetry_data", None)
    if tel is None:
        return False
    pts = getattr(tel, "points", None) or []
    try:
        return len(pts) >= 2
    except TypeError:
        return False


def _label_blob(ctx: Any) -> str:
    parts: list[str] = []
    vc = getattr(ctx, "vision_context", None) or {}
    if isinstance(vc, dict):
        for key in ("label_names", "labels"):
            for item in vc.get(key) or []:
                if isinstance(item, dict):
                    parts.append(str(item.get("description") or ""))
                else:
                    parts.append(str(item or ""))
    vi = (
        getattr(ctx, "video_intelligence", None)
        or getattr(ctx, "video_intelligence_context", None)
        or {}
    )
    if isinstance(vi, dict):
        for track in vi.get("object_tracks") or []:
            if isinstance(track, dict):
                parts.append(str(track.get("description") or ""))
    return " ".join(parts).lower()


def windshield_marker_count(ctx: Any, labels: Iterable[Any] | None = None) -> int:
    if labels is not None:
        blob = " ".join(str(x).lower() for x in labels)
    else:
        blob = _label_blob(ctx)
    return sum(1 for m in WINDSHIELD_LABEL_MARKERS if m in blob)


def has_driving_evidence(ctx: Any) -> bool:
    """True only for dashcam filename tokens, OSD HUD, .map points, or windshield stack."""
    if ctx is None:
        return False
    try:
        if _filename_is_dashcam(getattr(ctx, "filename", "") or ""):
            return True
        if _osd_hud_present(ctx):
            return True
        if _map_points_present(ctx):
            return True
        if windshield_marker_count(ctx) >= 2:
            return True
    except Exception:
        return False
    return False


__all__ = [
    "DASHCAM_FILENAME_TOKENS",
    "WINDSHIELD_LABEL_MARKERS",
    "has_driving_evidence",
    "windshield_marker_count",
]
