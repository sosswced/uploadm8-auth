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
* Geo tags are separate city + full state name (``losangeles``, ``california``) —
  never city+abbr run-ons like ``losangelesCA``.
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

from core.helpers import (
    expand_geo_runon_hashtag,
    extract_highway_route_tokens,
    is_instructional_road_sign,
    music_track_hashtag_bodies,
    normalize_hashtag_bodies,
    sanitize_hashtag_body,
    split_hashtag_source_phrases,
)
from core.vision_labels import (
    HASHTAG_BODY_MAX_LEN,
    is_generic_vision_label,
    is_invented_person_hashtag,
    is_junk_hashtag_body,
    rare_env_hashtag_bodies,
    road_hashtag_tokens,
    vision_label_slug,
)
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

# Common state-name → 2-letter abbreviation (used only for display/mapping,
# never glued onto city slugs — discovery wants #lasvegas #nevada).
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

    # Trill bucket → caption weave only (not discovery hashtags).
    # Kept for reference / tests that import the map; build_signal_hashtags
    # no longer injects these lifestyle tags.
_TRILL_TAGS: Dict[str, List[str]] = {
    "gloryBoy": ["GloryBoyTour", "TrillScore100", "SendIt", "DashCam", "CarLife"],
    "euphoric": ["Euphoric", "TrillScore", "SpeedDemon", "DashCam"],
    "sendIt":   ["SendIt", "TrillScore", "Spirited", "DashCam"],
    "spirited": ["SpiritedDrive", "TrillScore", "DashCam"],
    "chill":    ["TrillScore", "CruiseControl", "DashCam"],
}

# Evidence-class weights for final truncation (higher survives max_extra first).
_TAG_CLASS_WEIGHT = {
    "landmark": 100,
    "place_sign": 95,
    "road": 90,
    "route": 88,
    "business": 85,
    "city": 80,
    "state": 75,
    "padus": 72,
    "music_artist": 70,
    "music_genre": 55,
    "music_title": 50,
    "logo": 45,
    "env": 40,
    "trill": 35,
    "speed": 30,
    "other": 20,
}

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


def _slug(raw: Any, *, max_len: int = HASHTAG_BODY_MAX_LEN) -> str:
    """Sanitize-then-clamp; ``sanitize_hashtag_body`` already strips noise."""
    return sanitize_hashtag_body(str(raw or ""), max_len=max_len)


def _tag_weight(body: str, *, tag_class: str = "other") -> int:
    w = int(_TAG_CLASS_WEIGHT.get(tag_class, _TAG_CLASS_WEIGHT["other"]))
    # Prefer longer specific place slugs over short crumbs when class ties.
    return w * 100 + min(len(body), 40)


def _push(
    tags: List[str],
    seen: set,
    candidate: Any,
    *,
    max_len: int = HASHTAG_BODY_MAX_LEN,
    tag_class: str = "other",
    weights: Optional[Dict[str, int]] = None,
) -> None:
    """Append candidate slug(s) if non-empty, not duplicate, not blocklisted.

    Multi-entity sources (``Artist|Artist``, ``City, ST``) and geo run-ons are
    expanded into separate tags before sanitize.
    """
    for phrase in split_hashtag_source_phrases(str(candidate or "")):
        for body in expand_geo_runon_hashtag(phrase, max_len=max_len):
            body = _slug(body, max_len=max_len)
            if not body or body in seen or body in _BLOCKED_META:
                continue
            if is_junk_hashtag_body(body):
                continue
            if is_instructional_road_sign(phrase) or is_instructional_road_sign(body):
                continue
            seen.add(body)
            tags.append(body)
            if weights is not None:
                weights[body] = max(weights.get(body, 0), _tag_weight(body, tag_class=tag_class))


def _rank_by_weight(tags: List[str], weights: Dict[str, int], *, max_extra: int) -> List[str]:
    """Stable class-weight sort so geo/route/artist beat weak leftovers under max_extra."""
    if len(tags) <= max_extra:
        return tags
    indexed = list(enumerate(tags))
    indexed.sort(key=lambda it: (-int(weights.get(it[1], 0)), it[0]))
    keep = {t for _, t in indexed[:max_extra]}
    return [t for t in tags if t in keep][:max_extra]


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
      5. Geo city                        (e.g. ``losangeles``)
      6. Geo state                       (e.g. ``california``) — never city+abbr
      7. PADUS unit / public-lands hint  (protected area name or ``publiclands``)
      8. Highway hits parsed from OCR    (e.g. ``i15``, ``181south``)
      9. Vision logos / brands           (when prominent on screen)
     10. Speed-bucket tags               (max_speed_mph thresholds)
    """
    tags: List[str] = []
    seen: set = set()
    weights: Dict[str, int] = {}

    # Cap individual buckets so one signal can't crowd out the others.
    def _take(
        items: Iterable[Any],
        n: int,
        *,
        max_len: int = HASHTAG_BODY_MAX_LEN,
        tag_class: str = "other",
    ) -> None:
        added = 0
        for item in items:
            if added >= n:
                break
            before = len(tags)
            _push(tags, seen, item, max_len=max_len, tag_class=tag_class, weights=weights)
            if len(tags) > before:
                added += 1

    pack_subject = ""
    subject_token_overlap = None  # type: ignore
    pack: Dict[str, Any] = {}
    try:
        from core.publish_pack import get_publish_pack, subject_token_overlap as _sto

        subject_token_overlap = _sto
        pack = get_publish_pack(ctx) or {}
        pack_subject = str(pack.get("subject") or "")
    except Exception:
        pack = {}
        subject_token_overlap = None  # type: ignore

    def _logo_allowed(desc: Any) -> bool:
        text = str(desc or "").strip()
        if not text:
            return False
        slug = sanitize_hashtag_body(text)
        ambient = {
            "uhaul",
            "uhaulinternational",
            "jordankuwaitbank",
            "kuwaitbank",
            "realunited",
            "klankosova",
            "klankoso",
            "maersk",
            "fedex",
            "ups",
            "dhl",
            "walmart",
            "costco",
            "shell",
            "chevron",
        }
        if slug in ambient or is_junk_hashtag_body(slug):
            return False
        if re.search(r"(?i)\b(?:freight|logistics|u[\-\s]?haul)\b", text):
            return False
        # Local credit unions are useful place signals; ban mega/national banks only.
        if re.search(r"(?i)\bcredit\s+union\b", text):
            pass
        elif re.search(
            r"(?i)\b(?:chase|wells\s*fargo|bank\s*of\s*america|citibank|capital\s*one)\b",
            text,
        ):
            return False
        elif re.search(r"(?i)\bbank\b", text) and len(text.split()) < 2:
            return False
        elif re.search(r"(?i)\b(?:jordan|kuwait).*\bbank\b|\bbank\b.*(?:jordan|kuwait)", text):
            return False
        try:
            from services.hydration_enforcer import _is_ambient_logo

            if _is_ambient_logo(text):
                return False
        except Exception:
            pass
        if pack_subject and subject_token_overlap is not None:
            return bool(subject_token_overlap(text, pack_subject))
        # Without a pack, keep legacy behavior but still drop ambient brands.
        return True

    def _vi_logo_duration_ok_local(lg: dict) -> bool:
        try:
            start = float(lg.get("start_s") or 0.0)
            end = float(lg.get("end_s") or 0.0)
        except (TypeError, ValueError):
            return False
        return end > start and (end - start) >= 1.5

    for seed in list(pack.get("hashtag_seeds") or [])[:8]:
        if _logo_allowed(seed):
            _push(tags, seen, seed, tag_class="landmark", weights=weights)

    # ── Vision: landmarks ────────────────────────────────────────────────
    vc = (ctx.vision_context or {}) if isinstance(ctx.vision_context, dict) else {}
    landmark_names = list(vc.get("landmark_names") or [])
    _take(landmark_names, 4, tag_class="landmark")

    # ── Welcome to / Entering roadside signs (Vision OCR + VI text) ─────
    try:
        from services.scene_fusion import collect_place_signs

        for sign in collect_place_signs(ctx)[:2]:
            _push(tags, seen, sign, tag_class="place_sign", weights=weights)
    except Exception:
        pass

    # ── ACR music identification ─────────────────────────────────────────
    # Artist (+ short track / genre). Long smashed track titles are dropped.
    ac = (ctx.audio_context or {}) if isinstance(ctx.audio_context, dict) else {}
    if ac.get("music_detected") or ac.get("music_artist") or ac.get("music_title"):
        artist_raw = str(ac.get("music_artist") or "")
        title_raw = str(ac.get("music_title") or "")
        genre_raw = str(ac.get("music_genre") or ac.get("genre") or "")
        artist_slug = sanitize_hashtag_body(artist_raw)
        genre_primary = re.split(r"[/|,;]", genre_raw)[0].strip().replace("&", "")
        genre_slug = sanitize_hashtag_body(genre_primary)
        for body in music_track_hashtag_bodies(artist_raw, title_raw, genre_raw):
            if body in seen or body in _BLOCKED_META or is_junk_hashtag_body(body):
                continue
            if body == artist_slug:
                cls = "music_artist"
            elif genre_slug and body == genre_slug:
                cls = "music_genre"
            else:
                cls = "music_title"
            seen.add(body)
            tags.append(body)
            weights[body] = max(weights.get(body, 0), _tag_weight(body, tag_class=cls))

    # ── Geo (telemetry / OSD backfill, after reverse-geocode) ────────────
    tel = ctx.telemetry or ctx.telemetry_data
    if tel is not None:
        road = getattr(tel, "location_road", None)
        city = getattr(tel, "location_city", None)
        state = getattr(tel, "location_state", None)
        gaz_place = getattr(tel, "gazetteer_place_name", None)
        for tok in road_hashtag_tokens(road):
            _push(tags, seen, tok, tag_class="road", weights=weights)
        if gaz_place:
            gz_body = _slug(gaz_place)
            city_body = _slug(city or "")
            if not city_body or gz_body != city_body:
                _push(tags, seen, gaz_place, tag_class="city", weights=weights)
        # Separate discovery tags only — never city+CA / city+NV run-ons.
        _push(tags, seen, city, tag_class="city", weights=weights)
        _push(tags, seen, state, tag_class="state", weights=weights)
        start_disp = getattr(tel, "location_start_display", None)
        if start_disp:
            # "Las Vegas, NV" → #lasvegas #nevada via split_hashtag_source_phrases.
            _push(tags, seen, start_disp, tag_class="city", weights=weights)
        pun = getattr(tel, "padus_unit_name", None)
        if pun:
            _push(tags, seen, pun, tag_class="padus", weights=weights)
        elif getattr(tel, "near_padus", False):
            _push(tags, seen, "publiclands", tag_class="padus", weights=weights)

    # ── Highway route hits from Vision OCR (never roadside businesses) ───
    ocr_text = ((vc.get("ocr_text") or "") if isinstance(vc, dict) else "")[:4000]
    _take(extract_highway_route_tokens(ocr_text, limit=4), 3, tag_class="route")
    # Local business boards (Salal Credit Union, etc.) stay caption-prose only.

    # ── Vision: logos (brands visible on screen) ─────────────────────────
    # Prefer durable VI logos; Vision still-shot logos only when not ambient freight.
    try:
        from services.hydration_enforcer import (
            _is_ambient_logo,
            _logo_ok_for_hashtag,
        )
    except Exception:
        _is_ambient_logo = lambda _t: False  # type: ignore
        _logo_ok_for_hashtag = lambda _t, **_k: bool(str(_t or "").strip())  # type: ignore

    durable_logos: List[str] = []
    vi = getattr(ctx, "video_intelligence", None) or getattr(ctx, "video_intelligence_context", None) or {}
    if isinstance(vi, dict):
        for lg in list(vi.get("logos") or [])[:8]:
            if not isinstance(lg, dict) or not lg.get("description"):
                continue
            if not _vi_logo_duration_ok_local(lg):
                continue
            conf = lg.get("confidence")
            try:
                conf_f = float(conf) if conf is not None else None
            except (TypeError, ValueError):
                conf_f = None
            desc = str(lg["description"])
            if _logo_ok_for_hashtag(desc, confidence=conf_f) and _logo_allowed(desc):
                durable_logos.append(desc)
    for name in list(vc.get("logo_names") or [])[:4]:
        if _logo_ok_for_hashtag(name) and not _is_ambient_logo(name) and _logo_allowed(name):
            durable_logos.append(str(name))
    _take(durable_logos, 2, tag_class="logo")

    # ── Video Intelligence selective on-screen text ──────────────────────
    # Never slugify raw HUD lines (speed/GPS/timestamps) into hashtags.
    if isinstance(vi, dict):
        ost = list(vi.get("on_screen_text") or [])
        for row in sorted(
            ost,
            key=lambda x: -float((x or {}).get("confidence") or 0) if isinstance(x, dict) else 0,
        )[:8]:
            if isinstance(row, dict):
                txt = str(row.get("text") or "").strip()
            else:
                txt = str(row).strip()
            if not txt or len(txt) < 3 or len(txt) > 40:
                continue
            if is_instructional_road_sign(txt):
                continue
            hwy_from_line = extract_highway_route_tokens(txt, limit=2)
            if hwy_from_line:
                _take(hwy_from_line, 1, tag_class="route")
                continue
            if re.search(r"\d", txt):
                continue
            if is_junk_hashtag_body(txt) or is_generic_vision_label(txt):
                continue
            if re.search(r"(?i)\b(?:mph|escort|blackvue|viofo|gps|am|pm)\b", txt):
                continue
            if _is_ambient_logo(txt):
                continue
            _push(
                tags,
                seen,
                vision_label_slug(txt)[:36] or txt[:36],
                tag_class="other",
                weights=weights,
            )

    # Trill bucket is caption-weave only — never mint CruiseControl / chill
    # lifestyle discovery tags from the energy score.

    # ── Speed-bucket tags (works even with no Trill score) ───────────────
    # High-confidence publishable peak only — HUD-only medium must not mint
    # #speeddemon / triple-digit tags from an uncorroborated OCR spike.
    max_speed = 0.0
    try:
        from core.speed_consensus import publishable_peak_mph

        max_speed = publishable_peak_mph(ctx)
    except Exception:
        max_speed = 0.0
    for thresh, sb_tags in _SPEED_BUCKETS:
        if max_speed >= thresh:
            _take(sb_tags, 2, tag_class="speed")
            break

    # 0–2 rare environment tags (snowfall, ferry, …) even when geo+music are strong.
    try:
        plants: List[Any] = []
        yamnet_top = ""
        ac_env = (ctx.audio_context or {}) if isinstance(ctx.audio_context, dict) else {}
        yamnet_top = str(ac_env.get("yamnet_top") or ac_env.get("top_label") or "")
        vu = getattr(ctx, "video_understanding", None) or {}
        if isinstance(vu, dict):
            plants = list((vu.get("recognition_entities") or {}).get("plants") or [])[:6]
        vision_labels = list(vc.get("labels") or vc.get("label_names") or [])[:12]
        for body in rare_env_hashtag_bodies(yamnet_top, vision_labels, plants, limit=2):
            _push(tags, seen, body, tag_class="env", weights=weights)
    except Exception:
        pass

    try:
        from core.upload_domain_plan import discovery_hashtags_for_upload

        for tag in discovery_hashtags_for_upload(ctx, limit=max_extra):
            if not _logo_allowed(tag):
                continue
            _push(tags, seen, tag, tag_class="other", weights=weights)
    except Exception:
        pass
    return _rank_by_weight(tags, weights, max_extra=max_extra)


def _unsquash_delimited_sources(tags: List[str], sources: Iterable[Any]) -> List[str]:
    """Replace smashed multi-entity slugs with their split forms.

    ``Destroy Lonely|Lil Uzi Vert`` slugifies to ``destroylonelyliluzivert``;
    when we still have the delimited source, expand back to separate tags.
    """
    drop: set = set()
    inject: List[str] = []
    tag_set = {t.lower() for t in tags}
    for src in sources:
        text = str(src or "").strip()
        if not text:
            continue
        phrases = split_hashtag_source_phrases(text)
        if len(phrases) < 2:
            continue
        smashed = sanitize_hashtag_body(text)
        parts = [sanitize_hashtag_body(p) for p in phrases]
        parts = [p for p in parts if p]
        if not smashed or not parts:
            continue
        if smashed in tag_set:
            drop.add(smashed)
            for p in parts:
                if p not in tag_set:
                    inject.append(p)
                    tag_set.add(p)
    if not drop and not inject:
        return tags
    out: List[str] = []
    seen: set = set()
    for t in list(inject) + list(tags):
        b = sanitize_hashtag_body(str(t))
        if not b or b in seen or b in drop:
            continue
        seen.add(b)
        out.append(b)
    return out


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
    us = getattr(ctx, "user_settings", None) or {}
    try:
        target = int(us.get("maxHashtags") or us.get("max_hashtags") or max_extra or 15)
    except (TypeError, ValueError):
        target = max_extra or 15
    target = max(1, min(30, target))
    signal_cap = max_extra
    try:
        from core.upload_domain_plan import discovery_hashtags_for_upload

        if discovery_hashtags_for_upload(ctx, limit=1):
            signal_cap = max(max_extra, target)
    except Exception:
        pass
    extras = build_signal_hashtags(ctx, max_extra=signal_cap)
    if not extras:
        logger.info("[signal_hashtags] no extra tags from current signals")
        return []

    extras_lower = {t.lower() for t in extras}
    ac = (ctx.audio_context or {}) if isinstance(ctx.audio_context, dict) else {}
    unsquash_sources = [
        ac.get("music_artist"),
        ac.get("music_title"),
        getattr(ctx.telemetry or ctx.telemetry_data, "location_start_display", None)
        if (ctx.telemetry or ctx.telemetry_data) is not None
        else None,
    ]

    # ── Legacy ai_hashtags (general fallback list) ───────────────────────
    existing_ai: List[str] = []
    seen_ai: set = set()
    for raw in (ctx.ai_hashtags or []):
        for b in normalize_hashtag_bodies([str(raw)]):
            if not b or b in seen_ai or b in extras_lower:
                continue
            if is_invented_person_hashtag(raw) or is_junk_hashtag_body(b):
                continue
            seen_ai.add(b)
            existing_ai.append(b)
    ctx.ai_hashtags = _unsquash_delimited_sources(
        list(extras) + existing_ai, unsquash_sources
    )

    # ── Per-platform M8 hashtags ─────────────────────────────────────────
    m8_map = getattr(ctx, "m8_platform_hashtags", None) or {}
    if isinstance(m8_map, dict):
        for pl, raw_list in list(m8_map.items()):
            if not isinstance(raw_list, list):
                continue
            cur_seen: set = set()
            cur: List[str] = []
            for raw in raw_list:
                for b in normalize_hashtag_bodies([str(raw)]):
                    if not b or b in cur_seen or b in extras_lower:
                        continue
                    if is_invented_person_hashtag(raw) or is_junk_hashtag_body(b):
                        continue
                    cur_seen.add(b)
                    cur.append(b)
            m8_map[pl] = _unsquash_delimited_sources(list(extras) + cur, unsquash_sources)

    # Normalize legacy list too (split any LLM geo run-ons that survived).
    ctx.ai_hashtags = normalize_hashtag_bodies(list(ctx.ai_hashtags or []))
    ctx.ai_hashtags = _unsquash_delimited_sources(ctx.ai_hashtags, unsquash_sources)

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
