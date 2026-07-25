"""Canonical speed consensus across every speed-bearing provider.

One artifact (``speed_consensus_v1``) answers "what speed can we publish?"
for titles, captions, hashtags, prompts, and overlays — instead of each
call site re-reading raw telemetry/OSD fields and disagreeing.

Sources fused (same priority order as ``trusted_peak_speed_mph``):
  * ``telemetry``   — .map file peak (highest trust)
  * ``osd``         — dashcam HUD aggregate (spike-capped by series)
  * ``osd_series``  — trusted per-frame HUD samples
  * ``vision_ocr``  — unit-labeled HUD lines from Vision OCR
  * ``gps_implied`` — speed implied by OSD GPS fixes (corroboration only)
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from core.caption_creative import osd_series_peak_mph, trusted_peak_speed_mph

SPEED_CONSENSUS_ARTIFACT = "speed_consensus_v1"

_MPH_CLAIM_RE = re.compile(
    r"(?:\b(?:at|around|about|over|hitting|hit|reaching|reached|up\s+to|near(?:ly)?|~)\s+)?"
    r"\b(\d{1,3})(?:\.\d+)?\s*(mph|kph|km/?h|kmh)\b\.?",
    re.IGNORECASE,
)
# Legit road-sign context — "45 mph speed limit sign" is not a vehicle-speed claim.
_SIGN_CONTEXT_RE = re.compile(r"speed\s*limit|limit\s*sign|posted|school\s*zone|work\s*zone|zone\s*sign", re.IGNORECASE)


def speed_tolerance_mph(peak_mph: float) -> float:
    """Allowed drift for a speed mention to still count as the trusted peak."""
    try:
        p = float(peak_mph or 0)
    except (TypeError, ValueError):
        p = 0.0
    return max(8.0, p * 0.12)


def _f(v: Any) -> float:
    try:
        out = float(v or 0)
    except (TypeError, ValueError):
        return 0.0
    return out if out > 0 else 0.0


def build_speed_consensus(ctx: Any) -> Dict[str, Any]:
    """Fuse all speed readings on ``ctx`` into one canonical, confident artifact."""
    tel = getattr(ctx, "telemetry", None) or getattr(ctx, "telemetry_data", None)
    telemetry_max = _f(getattr(tel, "max_speed_mph", 0) if tel is not None else 0)

    osd = getattr(ctx, "dashcam_osd_context", None) or {}
    osd_max = 0.0
    gps_implied = 0.0
    if isinstance(osd, dict) and osd and not osd.get("skipped"):
        osd_max = _f(osd.get("max_speed_mph"))
        sq = osd.get("speed_quality") or {}
        if isinstance(sq, dict):
            gps_implied = _f(sq.get("gps_implied_peak_mph"))
    series_peak = osd_series_peak_mph(osd if isinstance(osd, dict) else None)

    vision_peak = 0.0
    vc = getattr(ctx, "vision_context", None) or {}
    if isinstance(vc, dict):
        ocr = str(vc.get("ocr_text") or "")
        if ocr:
            try:
                # Lazy import: hydration_enforcer imports this module at top level.
                from services.hydration_enforcer import _vision_ocr_peak_mph

                vision_peak = _f(_vision_ocr_peak_mph(ocr))
            except Exception:
                vision_peak = 0.0

    peak, source = trusted_peak_speed_mph(
        telemetry_max=telemetry_max,
        osd_max=osd_max,
        series_peak=series_peak,
        vision_peak=vision_peak,
    )
    tol = speed_tolerance_mph(peak)

    sources: Dict[str, float] = {}
    for name, val in (
        ("telemetry", telemetry_max),
        ("osd", osd_max),
        ("osd_series", series_peak),
        ("vision_ocr", vision_peak),
        ("gps_implied", gps_implied),
    ):
        if val >= 5:
            sources[name] = round(val, 1)

    agreeing = [n for n, v in sources.items() if peak >= 5 and abs(v - peak) <= tol]
    outliers = [n for n, v in sources.items() if peak >= 5 and abs(v - peak) > tol]

    if peak < 5:
        confidence = "none"
    elif source == "telemetry" or len(agreeing) >= 2:
        confidence = "high"
    elif not outliers:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "version": 1,
        "peak_mph": round(peak, 1) if peak >= 5 else 0.0,
        "source": source or None,
        "confidence": confidence,
        "tolerance_mph": round(tol, 1),
        "sources": sources,
        "agreeing": agreeing,
        "outliers": outliers,
    }


def get_speed_consensus(ctx: Any) -> Dict[str, Any]:
    """Return the persisted ``speed_consensus_v1`` artifact, building + caching it once."""
    arts = getattr(ctx, "output_artifacts", None)
    if isinstance(arts, dict):
        existing = arts.get(SPEED_CONSENSUS_ARTIFACT)
        if isinstance(existing, dict) and existing.get("version"):
            return existing
    consensus = build_speed_consensus(ctx)
    if isinstance(arts, dict):
        arts[SPEED_CONSENSUS_ARTIFACT] = consensus
    return consensus


def consensus_peak_mph(ctx: Any) -> float:
    """Canonical publishable peak MPH (0.0 when no trusted source)."""
    return _f(get_speed_consensus(ctx).get("peak_mph"))


def scrub_untrusted_speed_claims(
    text: str,
    peak_mph: float,
    *,
    tolerance_mph: Optional[float] = None,
) -> str:
    """Remove MPH/KPH claims that contradict the trusted peak from narrative text.

    Twelve Labs scene prose, transcripts, and OCR routinely invent speeds
    ("cruising at 46 MPH" on a 154 MPH run). Rules:
      * a claim within tolerance of the trusted peak is kept;
      * claims in explicit road-sign context ("45 mph speed limit") are kept;
      * everything else is dropped — including all claims when no trusted
        peak exists (unverifiable numbers never reach prompts or copy).
    """
    if not text:
        return text
    try:
        peak = float(peak_mph or 0)
    except (TypeError, ValueError):
        peak = 0.0
    tol = float(tolerance_mph) if tolerance_mph is not None else speed_tolerance_mph(peak)

    def _replace(m: "re.Match[str]") -> str:
        try:
            val = float(m.group(1))
        except (TypeError, ValueError):
            return ""
        unit = m.group(2).lower().replace("/", "")
        val_mph = val * 0.621371 if unit.startswith("k") else val
        if peak >= 5 and abs(val_mph - peak) <= tol:
            return m.group(0)
        window = text[max(0, m.start() - 30): m.end() + 30]
        if _SIGN_CONTEXT_RE.search(window):
            return m.group(0)
        return ""

    out = _MPH_CLAIM_RE.sub(_replace, text)
    out = re.sub(r"\s{2,}", " ", out)
    out = re.sub(r"\s+([,.;!?])", r"\1", out)
    out = re.sub(r"([,.;])\s*\1+", r"\1", out)
    return out.strip()


__all__ = [
    "SPEED_CONSENSUS_ARTIFACT",
    "build_speed_consensus",
    "get_speed_consensus",
    "consensus_peak_mph",
    "scrub_untrusted_speed_claims",
    "speed_tolerance_mph",
]
