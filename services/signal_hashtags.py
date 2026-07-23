"""
Deterministic signal-driven hashtag enrichment
==============================================

After the M8 / legacy caption stage runs the LLM and writes prompt-derived
hashtag arrays to the JobContext, we still want a *deterministic guarantee*
that the high-value structured signals every dashcam upload produces actually
appear in final hashtags — regardless of how the model felt that day.

This module turns the following ctx signals into a curated, slug-safe tag list:

* Vision landmarks       (Google Vision LANDMARK_DETECTION result)
* Vision logos / brands  (LOGO_DETECTION)
* Geo location           (city / state / road from Nominatim reverse geocode,
                          US Census gazetteer nearest place, PADUS protected unit,
                          fed from .map telemetry OR HUD-backfilled OSD GPS)
* ACR Cloud music ID     (artist + track from acrcloud_identify)
* Trill score bucket     (telemetry-driven driving energy bucket)
* Top-line speed         (max_speed_mph → ``highway`` / ``triple_digits`` etc.)
* Highway hints from OCR (Vision text-detect on the burned HUD or scene)

These tags are merged into:

* ``ctx.ai_hashtags``                — generic legacy AI list (used when M8
                                       output is empty or as a final
                                       fallback consumed by
                                       ``get_effective_hashtags``)
* ``ctx.m8_platform_hashtags[pl]``   — every populated platform variant from
                                       the M8 caption engine

Why both? ``get_effective_hashtags`` merges
``always → user platform → upload base → m8 platform → ai`` under
``maxHashtags``. Injecting into BOTH the per-platform M8 list AND the legacy
``ai_hashtags`` ensures these signals never get squeezed out by long M8
arrays, and they still land if M8 was disabled / empty for any platform.

The helper is deliberately conservative:

* All tags pass through ``sanitize_hashtag_body`` so they're publish-safe.
* Geo tags include both human-readable city/state ("losangeles", "california")
  and combined forms ("losangelesCA") — small and unambiguous.
* We never emit a leading '#'; ``get_effective_hashtags`` adds it.
* Each call is bounded (default 12 extras) so we don't blow the per-platform
  ``hashtag_count`` cap when the LLM already returned a full array.

Designed to be **idempotent**: running it twice on the same context produces
the exact same outputs.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Iterable, List, Optional

from core.helpers import sanitize_hashtag_body
from core.vision_labels import is_generic_vision_label, is_junk_hashtag_body, vision_label_slug
from stages.context import JobContext

logger = logging.getLogger("uploadm8-worker")

# Slugs we never want to surface even if a signal contains them — these are
# meta-spam tags and dilute discovery. Mirrors the M8 / caption blocklist.
_BLOCKED_META = {
    "viral",
    "trending",
    "follow",
    "like",
    "subscribe",
    "fyp",
    "foryou",
    "foryoupage",
    "video",
    "reels",
    "content",
    "youtube",
    "tiktok",
    "instagram",
    "facebook",
}

# Common state-name → 2-letter abbreviation so we can offer compact composite
# tags like ``losangelesCA``. Covers US 50 states + DC. Other countries fall
# back to country code (e.g. "AU") via ``location_country``.
_US_STATE_ABBR: Dict[str, str] = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "district of columbia": "DC", "florida": "FL", "georgia": "GA", "hawaii": "HI",
    "idaho": "ID", "illinois": "IL", "indiana": "IN", "iowa": "IA",
    "kansas": "KS", "kentucky": "KY", "louisiana": "LA", "maine": "ME",
    "maryland": "MD", "massachusetts": "MA", "michigan": "MI", "minnesota": "MN",
    "mississippi": "MS", "missouri": "MO", "montana": "MT", "nebraska": "NE",
    "nevada": "NV", "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM",
    "new york": "NY", "north carolina": "NC", "north dakota": "ND", "ohio": "OH",
    "oklahoma": "OK", "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI",
    "south carolina": "SC", "south dakota": "SD", "tennessee": "TN", "texas": "TX",
    "utah": "UT", "vermont": "VT", "virginia": "VA", "washington": "WA",
    "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
}

# Trill bucket → tag set (mirrors stages.telemetry_stage.get_trill_modifiers,
# but exposed here so we can apply them deterministically to every platform's
# hashtag list rather than relying on the legacy title modifier path).
_TRILL_TAGS: Dict[str, List[str]] = {
    "gloryBoy": ["GloryBoyTour", "TrillScore100", "SendIt", "DashCam", "CarLife"],
    "euphoric": ["Euphoric", "TrillScore", "SpeedDemon", "DashCam"],
    "sendIt":   ["SendIt", "TrillScore", "Spirited", "DashCam"],
    "spirited": ["SpiritedDrive", "TrillScore", "DashCam"],
    "chill":    ["TrillScore", "CruiseControl", "DashCam"],
}

# Highway / road keywords we recognize from Vision OCR (burned road signs,
# mile markers, exit boards, etc.). Hits get a short `highway` tag plus the
# matched route slug when it looks like one (e.g. "I15", "US101", "SR2").
_HIGHWAY_PATTERNS = (
    re.compile(r"\b(I[-\s]?\d{1,3})\b", re.IGNORECASE),                   # interstate
    re.compile(r"\b(US[-\s]?\d{1,3})\b", re.IGNORECASE),                  # US route
    re.compile(r"\b(SR[-\s]?\d{1,3})\b", re.IGNORECASE),                  # state route
    re.compile(r"\b(HWY[-\s]?\d{1,3})\b", re.IGNORECASE),                 # generic hwy
    re.compile(r"\b(ROUTE[-\s]?\d{1,3})\b", re.IGNORECASE),               # route N
)

# Speed bucketing thresholds (mph). Captures driving intensity even when no
# Trill score was computed (e.g. .map missing AND OSD backfill below ML thresh).
_SPEED_BUCKETS: List[tuple] = [
    (130.0, ["TripleDigits", "OverHundred", "TopSpeed"]),
    (100.0, ["TripleDigits", "OverHundred"]),
    (80.0,  ["HighwaySpeed", "FastLane"]),
    (60.0,  ["FreewayDrive"]),
]


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #


def _slug(raw: Any, *, max_len: int = 36) -> str:
    """Sanitize-then-clamp; ``sanitize_hashtag_body`` already strips noise."""
    return sanitize_hashtag_body(str(raw or ""), max_len=max_len)


def _push(tags: List[str], seen: set, candidate: Any, *, max_len: int = 36) -> None:
    """Append candidate slug if non-empty, not duplicate, not blocklisted."""
    body = _slug(candidate, max_len=max_len)
    if not body or body in seen or body in _BLOCKED_META:
        return
    if is_junk_hashtag_body(body):
        return
    seen.add(body)
    tags.append(body)


def _state_abbr(state: Optional[str], country: Optional[str]) -> Optional[str]:
    """Return 2-letter abbr for US states; else short country code if non-US."""
    if state:
        key = state.strip().lower()
        if key in _US_STATE_ABBR:
            return _US_STATE_ABBR[key]
    if country:
        c = country.strip().upper()
        # If location_country already looks like a 2-3 letter code, keep it.
        if 2 <= len(c) <= 3 and c.isalpha():
            return c
    return None


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def build_signal_hashtags(ctx: JobContext, *, max_extra: int = 12) -> List[str]:
    """Return up to ``max_extra`` deterministic, signal-derived hashtag bodies.

    The order encodes priority — earlier entries are more likely to survive
    ``maxHashtags`` truncation downstream. Priority:

      1. Vision landmark names           (very specific, high discovery value)
      2. ACR music artist/track          (when ACRCloud matched)
      3. Geo road/highway name           (also very specific)
      4. Gazetteer place (Census)        (when it differs from Nominatim city)
      5. Geo city + state combined       (e.g. ``losangelesCA``)
      6. Geo city                        (e.g. ``losangeles``)
      7. Geo state                       (e.g. ``california``)
      8. PADUS unit / public-lands hint  (protected area name or ``publiclands``)
      9. Highway hits parsed from OCR    (e.g. ``i15``)
     10. Vision logos / brands           (when prominent on screen)
     11. Trill bucket tags               (driving energy from telemetry)
     12. Speed-bucket tags               (max_speed_mph thresholds)
    """
    tags: List[str] = []
    seen: set = set()

    # Cap individual buckets so one signal can't crowd out the others.
    def _take(items: Iterable[Any], n: int, *, max_len: int = 36) -> None:
        added = 0
        for item in items:
            if added >= n:
                break
            before = len(tags)
            _push(tags, seen, item, max_len=max_len)
            if len(tags) > before:
                added += 1

    # ── Vision: landmarks ────────────────────────────────────────────────
    vc = (ctx.vision_context or {}) if isinstance(ctx.vision_context, dict) else {}
    landmark_names = list(vc.get("landmark_names") or [])
    _take(landmark_names, 4)

    # ── ACR music identification ─────────────────────────────────────────
    # Keep this high priority: artist/track tags are exact catalogue signals and
    # can otherwise be crowded out by geo/vision-heavy dashcam clips.
    ac = (ctx.audio_context or {}) if isinstance(ctx.audio_context, dict) else {}
    if ac.get("music_detected"):
        artist = ac.get("music_artist") or ""
        title = ac.get("music_title") or ""
        _push(tags, seen, artist)
        _push(tags, seen, title)

    # ── Geo (telemetry / OSD backfill, after reverse-geocode) ────────────
    tel = ctx.telemetry or ctx.telemetry_data
    if tel is not None:
        road = getattr(tel, "location_road", None)
        city = getattr(tel, "location_city", None)
        state = getattr(tel, "location_state", None)
        country = getattr(tel, "location_country", None)
        gaz_place = getattr(tel, "gazetteer_place_name", None)
        _push(tags, seen, road)
        if gaz_place:
            gz_body = _slug(gaz_place)
            city_body = _slug(city or "")
            if not city_body or gz_body != city_body:
                _push(tags, seen, gaz_place)
        abbr = _state_abbr(state, country)
        if city and abbr:
            # Compose without a separator so sanitize_hashtag_body keeps it.
            _push(tags, seen, f"{city}{abbr}")
        _push(tags, seen, city)
        _push(tags, seen, state)
        pun = getattr(tel, "padus_unit_name", None)
        if pun:
            _push(tags, seen, pun)
        elif getattr(tel, "near_padus", False):
            _push(tags, seen, "publiclands")

    # ── Highway hits parsed from Vision OCR ──────────────────────────────
    # Truncate OCR text so regex matching stays fast even on dense text-heavy frames.
    ocr_text = ((vc.get("ocr_text") or "") if isinstance(vc, dict) else "")[:4000]
    hwy_hits: List[str] = []
    for pat in _HIGHWAY_PATTERNS:
        for m in pat.findall(ocr_text):
            hwy_hits.append(str(m))
    _take(hwy_hits, 2)

    # ── Vision: logos (brands visible on screen) ─────────────────────────
    _take(list(vc.get("logo_names") or []), 3)

    # ── Video Intelligence logos + selective on-screen text ──────────────
    # Never slugify raw HUD lines (speed/GPS/timestamps) into hashtags.
    vi = getattr(ctx, "video_intelligence", None) or getattr(ctx, "video_intelligence_context", None) or {}
    if isinstance(vi, dict):
        for lg in list(vi.get("logos") or [])[:4]:
            if isinstance(lg, dict) and lg.get("description"):
                _push(tags, seen, lg["description"])
        ost = list(vi.get("on_screen_text") or [])
        for row in sorted(
            ost,
            key=lambda x: -float((x or {}).get("confidence") or 0) if isinstance(x, dict) else 0,
        )[:8]:
            if isinstance(row, dict):
                txt = str(row.get("text") or "").strip()
            else:
                txt = str(row).strip()
            if not txt or len(txt) < 3 or len(txt) > 28:
                continue
            # Prefer highway tokens from the line; skip digit-heavy HUD dumps.
            hwy_from_line: List[str] = []
            for pat in _HIGHWAY_PATTERNS:
                hwy_from_line.extend(str(m) for m in pat.findall(txt))
            if hwy_from_line:
                _take(hwy_from_line, 1)
                continue
            if re.search(r"\d", txt):
                continue
            if is_junk_hashtag_body(txt) or is_generic_vision_label(txt):
                continue
            if re.search(r"(?i)\b(?:mph|escort|blackvue|viofo|gps|am|pm)\b", txt):
                continue
            _push(tags, seen, vision_label_slug(txt)[:36] or txt[:36])

    # ── Trill score bucket (driving energy) ──────────────────────────────
    tr = ctx.trill or ctx.trill_score
    bucket = (getattr(tr, "bucket", "") if tr else "") or ""
    if bucket in _TRILL_TAGS:
        _take(_TRILL_TAGS[bucket], 3)

    # ── Speed-bucket tags (works even with no Trill score) ───────────────
    max_speed = 0.0
    if tel is not None:
        try:
            max_speed = float(getattr(tel, "max_speed_mph", 0.0) or 0.0)
        except (TypeError, ValueError):
            max_speed = 0.0
    if max_speed <= 0.0:
        # Fall back to OSD-only signal if telemetry has no speed yet.
        osd = (ctx.dashcam_osd_context or {}) if isinstance(ctx.dashcam_osd_context, dict) else {}
        try:
            max_speed = float(osd.get("max_speed_mph") or 0.0)
        except (TypeError, ValueError):
            max_speed = 0.0
    for thresh, sb_tags in _SPEED_BUCKETS:
        if max_speed >= thresh:
            _take(sb_tags, 2)
            break

    if len(tags) > max_extra:
        tags = tags[:max_extra]
    return tags


def merge_signal_hashtags_into_ctx(ctx: JobContext, *, max_extra: int = 12) -> List[str]:
    """Inject ``build_signal_hashtags(ctx)`` into legacy + per-platform lists.

    Behavior:
      * The signal tags are PREPENDED to ``ctx.ai_hashtags`` (so they survive
        the ``maxHashtags`` cap, and the LLM-only tags fill the remainder).
      * For every entry in ``ctx.m8_platform_hashtags`` we PREPEND the same
        signal tags, deduped — because per-platform variants from the M8
        engine bypass ``ctx.ai_hashtags`` in ``get_effective_hashtags``.
      * Returns the deduped signal list (also useful for logging).

    Idempotent: re-running on the same context does not duplicate tags.
    """
    extras = build_signal_hashtags(ctx, max_extra=max_extra)
    if not extras:
        logger.info("[signal_hashtags] no extra tags from current signals")
        return []

    extras_lower = {t.lower() for t in extras}

    # ── Legacy ai_hashtags (general fallback list) ───────────────────────
    existing_ai: List[str] = []
    seen_ai: set = set()
    for raw in (ctx.ai_hashtags or []):
        b = sanitize_hashtag_body(str(raw))
        if not b or b in seen_ai or b in extras_lower:
            continue
        seen_ai.add(b)
        existing_ai.append(b)
    ctx.ai_hashtags = list(extras) + existing_ai

    # ── Per-platform M8 hashtags ─────────────────────────────────────────
    m8_map = getattr(ctx, "m8_platform_hashtags", None) or {}
    if isinstance(m8_map, dict):
        for pl, raw_list in list(m8_map.items()):
            if not isinstance(raw_list, list):
                continue
            cur_seen: set = set()
            cur: List[str] = []
            for raw in raw_list:
                b = sanitize_hashtag_body(str(raw))
                if not b or b in cur_seen or b in extras_lower:
                    continue
                cur_seen.add(b)
                cur.append(b)
            m8_map[pl] = list(extras) + cur

    logger.info(
        "[signal_hashtags] injected %d signal tags: %s",
        len(extras),
        ", ".join(extras),
    )
    return extras


__all__ = [
    "build_signal_hashtags",
    "merge_signal_hashtags_into_ctx",
]
