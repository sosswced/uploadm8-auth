"""Speed-unit guards: MPH/KMH only — never degrees, lat/lon, or bare numbers.

Consensus, OCR peaks, and claim scrubbing must refuse:
  * decimal-degree coordinates (36.136162° / -115.178398°)
  * heading/bearing degrees (270° HDG)
  * temperature (°C / °F)
  * unitless integers
"""

from __future__ import annotations

import re
from typing import Optional

# Explicit vehicle-speed units only (after optional OCR junk).
SPEED_UNIT_RE = re.compile(
    r"(?i)^\s*(mph|mi/?h|mi\.?\s*p\.?\s*h\.?|kph|kmh|km/?h|kilomet(?:er|re)s?\s+per\s+hour|"
    r"miles\s+per\s+hour)\s*\.?$"
)

# Lat/lon pair — never a speed source.
_LATLON_PAIR_RE = re.compile(
    r"(?i)[+\-]?\d{1,2}\.\d{2,7}\s*[°ºo*]?\s+[+\-]?\d{1,3}\.\d{2,7}\s*[°ºo*]?"
)

# Standalone degree / heading / temperature — not MPH/KMH.
_DEGREE_ONLY_RE = re.compile(
    r"(?i)\b\d{1,3}(?:\.\d+)?\s*[°º]\s*(?:hdg|brg|deg(?:rees?)?|c|f)?\b"
    r"|\b\d{1,3}(?:\.\d+)?\s*degrees?\b"
)

# Explicit speed unit present (allows glued HUD forms like ``88MPH``).
_HAS_SPEED_UNIT_RE = re.compile(
    r"(?i)(?:\d\s*)?(?:mph|kph|kmh|km\s*/\s*h|mi\s*/\s*h|m\.?\s*p\.?\s*h\.?|k\.?\s*m\.?\s*h\.?)"
)


def has_speed_unit_token(text: str) -> bool:
    """True when text contains an MPH/KMH unit (including ``88MPH`` glued form)."""
    return bool(_HAS_SPEED_UNIT_RE.search(str(text or "")))


def is_speed_unit(unit: Optional[str]) -> bool:
    """True when ``unit`` is an MPH/KMH family token."""
    if not unit:
        return False
    return bool(SPEED_UNIT_RE.match(str(unit).strip()))


def looks_like_coordinate_or_degree(text: str) -> bool:
    """True when text is GPS/heading/temperature with no MPH/KMH unit."""
    t = str(text or "").strip()
    if not t:
        return False
    # HUD lines often carry lat/lon *and* ``88MPH`` — those are not degree-only.
    if has_speed_unit_token(t):
        return False
    if _LATLON_PAIR_RE.search(t):
        return True
    if _DEGREE_ONLY_RE.search(t):
        return True
    return False

def normalize_speed_unit(unit: Optional[str]) -> str:
    """Return ``mph`` | ``kph`` | ````."""
    if not is_speed_unit(unit):
        return ""
    u = re.sub(r"[^a-z]", "", str(unit).lower())
    if u.startswith("k"):
        return "kph"
    return "mph"


__all__ = [
    "SPEED_UNIT_RE",
    "has_speed_unit_token",
    "is_speed_unit",
    "looks_like_coordinate_or_degree",
    "normalize_speed_unit",
]
