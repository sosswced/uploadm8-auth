"""Speed-unit guards: MPH/KMH only — never degrees, lat/lon, or bare numbers.

Consensus, OCR peaks, and claim scrubbing must refuse:
  * decimal-degree coordinates (36.136162° / -115.178398°)
  * heading/bearing degrees (270° HDG)
  * temperature (°C / °F)
  * unitless integers
  * lon/lat *integer bleed* into fake HUD speeds (``-115MPH``, ``122°MPH``)
"""

from __future__ import annotations

import re
from typing import Optional, Set, Tuple

# Explicit vehicle-speed units only (after optional OCR junk).
SPEED_UNIT_RE = re.compile(
    r"(?i)^\s*(mph|mi/?h|mi\.?\s*p\.?\s*h\.?|kph|kmh|km/?h|kilomet(?:er|re)s?\s+per\s+hour|"
    r"miles\s+per\s+hour)\s*\.?$"
)

# Lat/lon pair — never a speed source.
_LATLON_PAIR_RE = re.compile(
    r"(?i)[+\-]?\d{1,2}\.\d{2,7}\s*[°ºo*]?\s+[+\-]?\d{1,3}\.\d{2,7}\s*[°ºo*]?"
)

# Truncated / OCR-damaged lon after a lat (minus + 2–3 digits, optional °, no mph).
_TRUNCATED_LON_RE = re.compile(
    r"(?i)([+\-]?\d{1,2}\.\d{2,7})\s*[°ºo*]?\s+"
    r"([+\-\u2010-\u2015])(\d{2,3})(?!\.\d)\s*[°ºo*]?(?!\s*(?:mph|kph|kmh|km\s*/\s*h|mi\s*/\s*h))"
)

# Standalone signed integer+degree that looks like lon/lat without a speed unit.
_SIGNED_DEGREE_RE = re.compile(
    r"(?i)(^|[\s,;])([+\-\u2010-\u2015])(\d{2,3})(?:\.\d{2,7})?\s*[°º]\b"
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

# Decimal coordinate token (lat or lon fragment) — digits before the point.
_COORD_DECIMAL_RE = re.compile(
    r"(?i)([+\-\u2010-\u2015]?)(\d{1,3})\.(\d{2,7})\s*[°ºo*]?"
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
    if _TRUNCATED_LON_RE.search(t):
        return True
    if _DEGREE_ONLY_RE.search(t):
        return True
    return False


def coordinate_integer_parts(text: str) -> Set[int]:
    """Integer degrees present as lat/lon tokens on the line (for bleed checks)."""
    t = str(text or "")
    out: Set[int] = set()
    for m in _COORD_DECIMAL_RE.finditer(t):
        try:
            out.add(int(m.group(2)))
        except (TypeError, ValueError):
            continue
    for m in _TRUNCATED_LON_RE.finditer(t):
        try:
            out.add(int(m.group(3)))
        except (TypeError, ValueError):
            continue
    for m in _SIGNED_DEGREE_RE.finditer(t):
        try:
            out.add(int(m.group(3)))
        except (TypeError, ValueError):
            continue
    return out


def coordinate_spans(text: str) -> Tuple[Tuple[int, int], ...]:
    """Character spans that belong to lat/lon tokens (not vehicle speed)."""
    t = str(text or "")
    spans = []
    for rx in (_LATLON_PAIR_RE, _TRUNCATED_LON_RE, _COORD_DECIMAL_RE, _SIGNED_DEGREE_RE):
        for m in rx.finditer(t):
            spans.append((m.start(), m.end()))
    return tuple(spans)


def speed_match_is_coordinate_bleed(
    line: str,
    *,
    match_start: int,
    match_end: int,
    speed_digits: str,
) -> bool:
    """True when a unit-labeled ``NN MPH`` is really lat/lon OCR bleed.

    Catches:
      * ``-115MPH`` / ``-122MPH`` (signed lon glued to a unit)
      * ``115°MPH`` when 115 is the lon integer on the same HUD line
      * any speed match whose span overlaps a coordinate token
      * speed digits that are the integer prefix of a decimal coord on the line
        *and* sit inside/adjacent to that coord (not a separate HUD sample)
    """
    t = str(line or "")
    if not t or match_start < 0 or match_end > len(t):
        return False
    try:
        n = int(str(speed_digits).strip())
    except (TypeError, ValueError):
        return False

    # Signed lon glued onto unit: "-115MPH" — never a real Escort/M8 HUD form.
    if match_start > 0 and t[match_start - 1] in "-−–—‐-":
        return True

    for a, b in coordinate_spans(t):
        # Overlap with a GPS token.
        if match_start < b and match_end > a:
            return True
        # Speed immediately glued after a coordinate with no whitespace gap.
        if a < match_start <= b:
            return True
        if match_start == b:
            return True

    # Degree used as OCR "separator" but number is the lon/lat integer on-line.
    window = t[max(0, match_start - 1) : min(len(t), match_end + 1)]
    if "°" in window or "º" in window:
        if n in coordinate_integer_parts(t):
            return True

    # Digits are the integer part of a decimal coordinate that contains this span.
    for m in _COORD_DECIMAL_RE.finditer(t):
        try:
            integ = int(m.group(2))
        except (TypeError, ValueError):
            continue
        if integ != n:
            continue
        # Match starts at the integer part of this decimal → bleed, not HUD.
        int_start = m.start(2)
        if match_start == int_start:
            return True
        # Match fully inside the decimal token.
        if m.start() <= match_start < m.end():
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
    "coordinate_integer_parts",
    "coordinate_spans",
    "has_speed_unit_token",
    "is_speed_unit",
    "looks_like_coordinate_or_degree",
    "normalize_speed_unit",
    "speed_match_is_coordinate_bleed",
]
