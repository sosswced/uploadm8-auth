"""
Deterministic Hydration Enforcer
================================

Even with the M8 engine, the LLM occasionally returns category-seed boilerplate
("Cruise under vast skies! Endless horizons await." with #openroad / #travelvibes
/ #scenicdrive) that contains zero footage-derived evidence. That makes published
captions feel generic regardless of how rich the underlying analysis was.

This module is the LAST GATE in the caption pipeline. After:

    run_caption_stage(ctx)  →  merge_signal_hashtags_into_ctx(ctx)

we call ``enforce_hydration(ctx)`` to:

1. Build an *evidence pool* from every analyzed source on ``ctx``:
   Vision labels / OCR / landmarks / logos, ACR music, telemetry geo,
   dashcam OSD speed / driver / GPS, Trill bucket, Whisper transcript nouns,
   Twelve Labs scene description, Hume emotion.
2. Compute a structured *hydration report* (which signals were actually
   captured) and log it. If everything is empty, log WARNING + reasons so
   the operator can see why the post will fall back to generic copy.
3. **Force-rewrite captions** that mention zero evidence even though
   evidence exists, by appending a deterministic factual anchor phrase
   (e.g. "Captured at 46 MPH near Guadalupe, California with The Eagles —
   Hotel California playing.").
4. **Replace category-seed-only hashtags** with deterministic
   evidence-driven hashtags when real signals exist. Category seeds only
   survive when they're the only thing we have.
5. Apply identical treatment to every per-platform M8 variant
   (``ctx.m8_platform_captions`` / ``..._titles`` / ``..._hashtags``) plus
   the legacy ``ctx.ai_caption`` / ``ctx.ai_title`` / ``ctx.ai_hashtags``
   used by ``get_effective_*`` fallbacks.

When the evidence pool is empty, generic-pattern captions/titles are still
replaced using a deterministic **filename + category** anchor (and optional
**persona name** + **transcript fragment** when those exist on ``ctx``). When
evidence exists but the model copy ignores it, rewrites follow the same overlap
rules as before. Good non-generic output is left untouched. The pass is
deterministic and idempotent — running it twice produces the same artifacts.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from core.helpers import sanitize_hashtag_body
from core.vision_labels import (
    evidence_pool_has_strong_hashtag_signals,
    filter_vision_labels_for_hashtags,
    is_generic_vision_label,
    is_redundant_vision_label,
    resolve_ambient_profiles,
    vision_label_slug,
)
from stages.context import JobContext, build_hydration_story_text

logger = logging.getLogger("uploadm8-worker")

# ---------------------------------------------------------------------------
# Generic-content detectors
# ---------------------------------------------------------------------------

# Phrases that are pure boilerplate. Any caption matching one of these without
# referencing actual evidence is rewritten.
_GENERIC_CAPTION_PATTERNS: List[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in (
        # Original travel / dashcam clichés
        r"\bcruise (?:under|through|along)\b",
        r"\bvast skies\b",
        r"\bendless (?:horizons?|road|roads|highway|highways|sky|skies)(?:\b|\s+(?:ahead|await|beckons?))",
        r"\b(?:adventure|journey|destiny|moment|magic) (?:awaits?|unfolds?|begins?|calls?|beckons?)\b",
        r"\bopen road(?:\s+(?:odyssey|calls?|beckons?|symphony|dreams?|magic))?\b",
        r"\b(?:open|endless) road\b",
        r"\bopen road\s+(?:odyssey|symphony|dreams?|magic|calls?|adventures?)\b",
        r"\bscenic (?:vibes?|drive|views?|stop|route|roads?|beauty)\b",
        r"\b(?:travel|highway|cloud) (?:vibes?|watching|symphony|dreams?)\b",
        r"\bhighway (?:symphony|dreams?|magic|melody|odyssey|tales?|stories)\b",
        r"\b(?:colorful|vibrant|stunning|breathtaking) blooms?\b",
        r"\bblooms? (?:stun|meet|burst|dance)\b",
        r"\bblooming (?:roads?|highways?|paths?)\b",
        r"\b(?:purple|red|yellow|pink) blooms?\s+(?:stun|meet|burst|dance)\b",
        r"\bdesert sands?\b",
        r"\bgood vibes\b",
        r"\bbreath(?:e|taking) (?:in )?(?:the )?freedom\b",
        r"\b(?:explore|discover) more\b",
        r"\bnature(?:'s)? (?:beauty|symphony|call|magic|wonders?)\b",
        r"\bbuckle up\b",
        r"\bjoin me\b",
        r"\blet's dive\b",
        r"\byou won't believe\b",
        r"\bexciting moments?\b",
        r"\bunbelievable moments?\b",
        r"\bridin'? dirty\b",
        r"\bvibes? only\b",
        r"\bembrace the chaos\b",
        r"\bhidden gem\b",
        # New AI-cliché catches based on observed M8 output
        r"\b(?:watch|witness) (?:the road|the world|the sky|nature|magic) (?:transform|unfold|change)\b",
        r"\b(?:watch|witness) (?:serenity|magic|nature|beauty) (?:meet|meets) (?:motion|sky|road|nature)\b",
        r"\bserenity meet(?:s)? motion\b",
        r"\b(?:road|highway|drive|journey) ahead\.?\s*$",
        r"\b(?:where|when) (?:the )?road meets (?:the )?(?:sky|horizon|sunset|dreams?)\b",
        r"\b(?:tranquil|peaceful|serene) (?:drive|journey|road|moments?)\b",
        r"\b(?:road|highway) (?:tales?|stories|chronicles|poetry)\b",
        r"\b(?:every )?mile (?:tells a story|matters|counts)\b",
        r"\bjourney captured\b",
        r"\b(?:scenic|epic|legendary) (?:moments?|adventures?|stops?)\b",
        r"\b(?:unforgettable|magical|legendary) (?:journey|drive|ride|moments?)\b",
        r"\bon the open road\b",
        r"\b(?:roads?|highways?) less travel(?:l)?ed\b",
        r"\bwhere the road takes (?:me|us|you)\b",
        # Observed M8 filler that ignores timeline/hydration
        r"\bhigh[- ]energy,?\s+first[- ]person\s+dashcam\b",
        r"\bthe video is a\s+(?:high[- ]energy|tense|exciting)\b",
        r"\bfrom inside a moving vehicle\b",
        r"\bcapturing a tense and confrontational journey\b",
        r"\bdashcam recording from inside\b",
        # Extremely short titles that are pure mood (no proper noun, no number)
        r"^(?:road|drive|journey|adventure|highway|moment|vibes?|cruise|escape)\.?$",
    )
]

# Hashtags we should treat as "generic seed" placeholders. If we have
# evidence-driven tags, these get demoted/replaced.
_CATEGORY_SEED_TAGS = {
    # automotive / dashcam category seeds
    "dashcam", "roadtrip", "openroad", "highwayviews", "travelvibes",
    "scenicdrive", "exploremore", "carjourney", "naturedrive",
    "adventuretime", "landscapeviews", "journeyon", "cloudwatching",
    "carporn", "carsofinstagram", "carlife", "autolife", "drivingvlog",
    "carculture", "joyride", "speedlimit", "drive", "driving",
    "highwayjourney", "tranquildrive", "scenicroute", "ontheroad",
    "carride", "adventureawaits", "journeycaptured", "naturebeauty",
    "flowerfields", "desertdrive",
    # generic travel/lifestyle
    "travel", "travelgram", "traveltok", "wanderlust", "adventure",
    "solotravel", "travelvlog", "travelblogger", "vlog", "lifestylevlog",
    "morningroutine", "selfcare", "wellnesstok", "authenticlife",
    # platform-meta (already blocked elsewhere; mirror for safety)
    "viral", "fyp", "foryoupage", "trending", "mustwatch",
    # leaked QA / placeholder tags
    "tester", "qwe",
}

# Words that indicate the caption is a real evidence-grounded sentence.
# Used as a second-pass safety check before we rewrite.
_EVIDENCE_BEARING_HINTS = re.compile(
    r"\b(\d{1,3})\s*(?:mph|kmh|km/h|mi/h)\b|\bnear\b|\binterstate\b|\bhighway\s*\d|"
    r"\broute\s*\d|\bus[-\s]?\d{1,3}\b|\bi[-\s]?\d{1,3}\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Evidence model
# ---------------------------------------------------------------------------

# Common stop-words filtered when extracting nouns from transcript/OCR.
_STOP_WORDS = {
    "the", "and", "for", "with", "this", "that", "but", "have", "has", "had",
    "are", "was", "were", "into", "your", "from", "their", "them", "they",
    "you", "yes", "very", "just", "like", "want", "what", "when", "where",
    "there", "here", "then", "than", "him", "her", "his", "she", "its", "it's",
    "our", "us", "we", "be", "do", "does", "did", "is", "am", "an", "of",
    "to", "in", "on", "at", "by", "or", "if", "no", "so", "as", "up", "out",
    "off", "all", "any", "some", "now", "one", "two", "get", "got", "been",
    "around", "through", "thing", "things", "really", "going", "yeah",
    "good", "great", "nice", "cool",
}


@dataclass
class EvidencePool:
    """Structured evidence collected from ctx, used to drive deterministic copy."""

    # Geo (.map telemetry or HUD-backfilled OSD GPS, after reverse-geocode + gazetteer + PADUS)
    road: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    state_abbr: Optional[str] = None
    country: Optional[str] = None
    gazetteer_place: Optional[str] = None
    protected_area: Optional[str] = None
    near_protected_land: bool = False

    # Speed (canonical: .map telemetry > OSD HUD > Vision OCR — see collect_evidence)
    max_speed_mph: float = 0.0
    avg_speed_mph: float = 0.0
    speed_source: str = ""  # "telemetry" | "osd" | "vision_ocr"

    # OSD HUD facts
    driver_name: Optional[str] = None

    # Trill (driving energy bucket)
    trill_bucket: Optional[str] = None
    trill_score: float = 0.0

    # ACRCloud music ID
    music_artist: Optional[str] = None
    music_title: Optional[str] = None
    music_genre: Optional[str] = None

    # Whisper / transcript
    transcript_nouns: List[str] = field(default_factory=list)
    transcript_phrase: Optional[str] = None
    transcript_language: Optional[str] = None
    transcript_topics: List[str] = field(default_factory=list)
    transcript_questions: List[str] = field(default_factory=list)
    transcript_entities: Dict[str, List[str]] = field(default_factory=dict)
    transcript_speaker_turns: List[Dict[str, Any]] = field(default_factory=list)
    transcript_topic_timeline: List[Dict[str, Any]] = field(default_factory=list)

    # Vision (Google Cloud Vision)
    vision_labels: List[str] = field(default_factory=list)
    vision_landmarks: List[str] = field(default_factory=list)
    vision_logos: List[str] = field(default_factory=list)
    vision_ocr_tokens: List[str] = field(default_factory=list)
    vision_highways: List[str] = field(default_factory=list)
    has_faces: bool = False
    expressive_faces: bool = False

    # Google visual recognition buckets (vehicles, food, plants, fishing, …)
    recognition_entities: Dict[str, List[str]] = field(default_factory=dict)

    # Resolved ambient-redundancy profiles (automotive, dashcam, cooking, …)
    ambient_profiles: List[str] = field(default_factory=list)

    # Twelve Labs / Video Intelligence (legacy text + new structured tracks)
    video_understanding_phrase: Optional[str] = None
    video_summary_phrase: Optional[str] = None
    video_labels: List[str] = field(default_factory=list)
    vi_object_tracks: List[Dict[str, Any]] = field(default_factory=list)
    vi_text_detections: List[Dict[str, Any]] = field(default_factory=list)
    vi_person_segments: List[Dict[str, Any]] = field(default_factory=list)
    vi_logos: List[Dict[str, Any]] = field(default_factory=list)

    # Audio environment
    yamnet_top: Optional[str] = None

    def has_any_evidence(self) -> bool:
        return any(
            (
                self.road, self.city, self.gazetteer_place, self.protected_area,
                self.max_speed_mph > 0, self.driver_name, self.trill_bucket,
                self.music_artist, self.music_title, self.transcript_phrase,
                self.transcript_topics, self.transcript_entities,
                self.vision_labels, self.vision_landmarks, self.vision_logos,
                self.vision_ocr_tokens, self.vision_highways,
                self.video_understanding_phrase, self.video_summary_phrase,
                self.video_labels, self.yamnet_top,
                self.vi_object_tracks, self.vi_text_detections,
                self.vi_person_segments, self.vi_logos,
                self.recognition_entities,
                getattr(self, "place_beaches", None),
                getattr(self, "place_monuments", None),
                getattr(self, "place_stadiums", None),
                getattr(self, "license_plates", None),
                getattr(self, "sports_teams", None),
            )
        )

    def to_report(self) -> Dict[str, Any]:
        return {
            "geo": {
                "road": self.road,
                "city": self.city,
                "state": self.state,
                "gazetteer_place": self.gazetteer_place,
                "protected_area": self.protected_area,
                "near_protected_land": self.near_protected_land,
            },
            "speed": {
                "max_mph": self.max_speed_mph,
                "avg_mph": self.avg_speed_mph,
                "source": self.speed_source,
            },
            "osd": {"driver_name": self.driver_name},
            "trill": {"bucket": self.trill_bucket, "score": self.trill_score},
            "music": {
                "artist": self.music_artist,
                "title": self.music_title,
                "genre": self.music_genre,
            },
            "transcript": {
                "phrase": self.transcript_phrase,
                "language": self.transcript_language,
                "nouns": self.transcript_nouns[:8],
                "topics": self.transcript_topics[:8],
                "questions": self.transcript_questions[:4],
                "entities": {
                    k: v[:6] for k, v in (self.transcript_entities or {}).items()
                },
                "speaker_turns": self.transcript_speaker_turns[:4],
                "topic_timeline": self.transcript_topic_timeline[:6],
            },
            "video_intelligence": {
                "labels": self.video_labels[:12],
                "summary": self.video_summary_phrase,
                "object_track_count": len(self.vi_object_tracks),
                "object_tracks": self.vi_object_tracks[:6],
                "text_detection_count": len(self.vi_text_detections),
                "on_screen_text": self.vi_text_detections[:6],
                "person_segment_count": len(self.vi_person_segments),
                "logo_count": len(self.vi_logos),
                "logos": self.vi_logos[:6],
            },
            "vision": {
                "label_count": len(self.vision_labels),
                "labels": self.vision_labels[:6],
                "landmarks": self.vision_landmarks[:4],
                "logos": self.vision_logos[:4],
                "ocr_tokens": self.vision_ocr_tokens[:6],
                "highways": self.vision_highways[:4],
                "has_faces": self.has_faces,
                "recognition_entities": {
                    k: v[:6] for k, v in (self.recognition_entities or {}).items() if v
                },
            },
            "place_entities": {
                "beaches": list(getattr(self, "place_beaches", None) or [])[:6],
                "monuments": list(getattr(self, "place_monuments", None) or [])[:6],
                "stadiums": list(getattr(self, "place_stadiums", None) or [])[:6],
                "license_plates": list(getattr(self, "license_plates", None) or [])[:6],
                "sports_teams": list(getattr(self, "sports_teams", None) or [])[:6],
                "sources": list(getattr(self, "place_sources", None) or []),
            },
            "video_understanding": self.video_understanding_phrase,
            "yamnet_top": self.yamnet_top,
        }


# ---------------------------------------------------------------------------
# Evidence collection
# ---------------------------------------------------------------------------

# 2-letter US state abbreviations (mirror signal_hashtags._US_STATE_ABBR).
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

_HIGHWAY_PATTERNS = (
    re.compile(r"\b(I[-\s]?\d{1,3})\b", re.IGNORECASE),
    re.compile(r"\b(US[-\s]?\d{1,3})\b", re.IGNORECASE),
    re.compile(r"\b(SR[-\s]?\d{1,3})\b", re.IGNORECASE),
    re.compile(r"\b(HWY[-\s]?\d{1,3})\b", re.IGNORECASE),
    re.compile(r"\b(ROUTE[-\s]?\d{1,3})\b", re.IGNORECASE),
)


def _state_abbr(state: Optional[str], country: Optional[str]) -> Optional[str]:
    if state:
        key = state.strip().lower()
        if key in _US_STATE_ABBR:
            return _US_STATE_ABBR[key]
    if country:
        c = country.strip().upper()
        if 2 <= len(c) <= 3 and c.isalpha():
            return c
    return None


def _extract_transcript_nouns(transcript: str, *, limit: int = 12) -> List[str]:
    """Extract candidate noun-like tokens from a transcript or OCR blob."""
    if not transcript:
        return []
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9'-]{3,}", transcript)
    out: List[str] = []
    seen: set = set()
    for tok in tokens:
        low = tok.lower().strip("'-")
        if len(low) < 4 or low in _STOP_WORDS or low in seen:
            continue
        # Prefer tokens that are capitalized in the original (likely proper nouns)
        seen.add(low)
        out.append(low)
        if len(out) >= limit:
            break
    return out


def _first_sentence(text: str, *, max_chars: int = 120) -> Optional[str]:
    """Return the first non-trivial sentence from a transcript blob."""
    if not text:
        return None
    chunks = re.split(r"(?<=[.!?])\s+", text.strip())
    for c in chunks:
        c = c.strip()
        if len(c) >= 12 and not c.startswith("["):
            return c[:max_chars]
    snippet = text.strip()[:max_chars]
    return snippet or None


def _clean_video_label(raw: Any) -> str:
    """Normalize VI labels such as ``"flower (0.93)"`` to ``"flower"``."""
    text = re.sub(r"\s+", " ", str(raw or "").strip())
    text = re.sub(r"\s*\([0-9.]+\)\s*$", "", text).strip()
    return text


def _vision_ocr_peak_mph(ocr: str) -> float:
    """Best-effort peak speed from Vision OCR (fallback when .map and OSD lack HUD).

    Rejects roadside SPEED LIMIT copy and singleton OCR spikes (e.g. 154 when
    the HUD samples are 87–92) so false peaks never enter caption anchors.
    """
    if not ocr:
        return 0.0
    if re.search(r"speed\s*limit|maximum\s*speed|school\s*zone|work\s*zone", ocr, re.I):
        # Still allow HUD-like lines elsewhere in the blob; strip limit lines.
        ocr = re.sub(
            r"[^\n]*(?:speed\s*limit|maximum\s*speed|school\s*zone|work\s*zone)[^\n]*",
            " ",
            ocr,
            flags=re.I,
        )
    vals: List[float] = []
    for m in re.finditer(r"\b(\d{1,3})\s*(?:mph|mi/h)\b", ocr, re.IGNORECASE):
        try:
            v = float(m.group(1))
        except (TypeError, ValueError):
            continue
        if 5.0 <= v <= 200.0:
            vals.append(v)
    for m in re.finditer(r"\b(\d{1,3})\s*(?:kmh|km/h)\b", ocr, re.IGNORECASE):
        try:
            v = float(m.group(1)) * 0.621371
        except (TypeError, ValueError):
            continue
        if 5.0 <= v <= 200.0:
            vals.append(v)
    if not vals:
        return 0.0
    vals.sort()
    mid = vals[len(vals) // 2]
    # Drop singleton spikes far above the median of OCR speed hits.
    trusted = [v for v in vals if v <= mid + 45.0]
    if not trusted:
        trusted = [mid]
    return max(trusted)


def collect_evidence(ctx: JobContext) -> EvidencePool:
    """Build EvidencePool from every analyzed source on ctx. Cheap and pure."""
    pool = EvidencePool()

    # Speed — canonical resolver: .map telemetry > OSD HUD > Vision OCR text.
    tel = getattr(ctx, "telemetry", None) or getattr(ctx, "telemetry_data", None)
    tel_max = tel_avg = 0.0
    if tel is not None:
        pool.road = (getattr(tel, "location_road", None) or None)
        pool.city = (getattr(tel, "location_city", None) or None)
        pool.state = (getattr(tel, "location_state", None) or None)
        pool.country = (getattr(tel, "location_country", None) or None)
        pool.gazetteer_place = (getattr(tel, "gazetteer_place_name", None) or None)
        pool.protected_area = (getattr(tel, "padus_unit_name", None) or None)
        pool.near_protected_land = bool(getattr(tel, "near_padus", False))
        pool.state_abbr = _state_abbr(pool.state, pool.country)
        try:
            tel_max = float(getattr(tel, "max_speed_mph", 0) or 0)
            tel_avg = float(getattr(tel, "avg_speed_mph", 0) or 0)
        except (TypeError, ValueError):
            tel_max = tel_avg = 0.0

    vc_early = getattr(ctx, "vision_context", None) or {}
    ocr_blob = str(vc_early.get("ocr_text") or "") if isinstance(vc_early, dict) else ""
    vision_peak = _vision_ocr_peak_mph(ocr_blob)

    osd = getattr(ctx, "dashcam_osd_context", None) or {}
    osd_max = osd_avg = 0.0
    if isinstance(osd, dict) and osd and not osd.get("skipped"):
        try:
            osd_max = float(osd.get("max_speed_mph") or 0)
            osd_avg = float(osd.get("avg_speed_mph") or 0)
        except (TypeError, ValueError):
            osd_max = osd_avg = 0.0

    from core.caption_creative import osd_series_peak_mph, trusted_peak_speed_mph

    series_peak = osd_series_peak_mph(osd if isinstance(osd, dict) else None)
    peak, src = trusted_peak_speed_mph(
        telemetry_max=tel_max,
        osd_max=osd_max,
        series_peak=series_peak,
        vision_peak=vision_peak,
    )
    if peak >= 5:
        pool.max_speed_mph = peak
        pool.avg_speed_mph = (
            tel_avg if src == "telemetry" and tel_avg > 0
            else (osd_avg if src.startswith("osd") and osd_avg > 0 else peak)
        )
        pool.speed_source = src

    if isinstance(osd, dict) and osd and not osd.get("skipped"):
        drv = osd.get("driver_name")
        if drv and isinstance(drv, str) and drv.strip():
            pool.driver_name = drv.strip()

    # ── Trill bucket ────────────────────────────────────────────────────
    tr = getattr(ctx, "trill", None) or getattr(ctx, "trill_score", None)
    if tr:
        try:
            pool.trill_score = float(getattr(tr, "score", 0) or 0)
        except (TypeError, ValueError):
            pool.trill_score = 0.0
        bucket = getattr(tr, "bucket", None)
        if bucket and isinstance(bucket, str):
            pool.trill_bucket = bucket.strip() or None

    # ── ACRCloud music ID ───────────────────────────────────────────────
    ac = getattr(ctx, "audio_context", None) or {}
    if isinstance(ac, dict) and ac.get("music_detected"):
        pool.music_artist = (ac.get("music_artist") or "").strip() or None
        pool.music_title = (ac.get("music_title") or "").strip() or None
        pool.music_genre = (ac.get("music_genre") or "").strip() or None

    # ── Whisper / transcript ────────────────────────────────────────────
    transcript = (getattr(ctx, "ai_transcript", "") or "").strip()
    if not transcript and isinstance(ac, dict):
        transcript = (ac.get("transcript") or "").strip()
    if not transcript and isinstance(ac, dict):
        segs_merge = ac.get("transcript_segments") or []
        if isinstance(segs_merge, list) and segs_merge:
            parts: List[str] = []
            for seg in segs_merge[:48]:
                if isinstance(seg, dict):
                    t = str(seg.get("text") or "").strip()
                    if t:
                        parts.append(t)
                elif isinstance(seg, str) and seg.strip():
                    parts.append(seg.strip())
            if parts:
                transcript = " ".join(parts)[:12000]
    if transcript:
        pool.transcript_phrase = _first_sentence(transcript)
        pool.transcript_nouns = _extract_transcript_nouns(transcript, limit=8)
    structured = (
        ac.get("transcript_structured") if isinstance(ac, dict) else None
    ) or {}
    if isinstance(structured, dict) and structured:
        lang = structured.get("language") or (ac.get("transcript_language") if isinstance(ac, dict) else "")
        if lang:
            pool.transcript_language = str(lang)
        pool.transcript_topics = [str(x).strip() for x in (structured.get("topics") or []) if str(x).strip()][:8]
        pool.transcript_questions = [str(x).strip() for x in (structured.get("questions") or []) if str(x).strip()][:4]
        ne = structured.get("named_entities") or {}
        if isinstance(ne, dict):
            pool.transcript_entities = {
                k: [str(x).strip() for x in (ne.get(k) or []) if str(x).strip()][:6]
                for k in ("people", "places", "products", "organizations")
                if ne.get(k)
            }
        st = structured.get("speaker_turns") or []
        if isinstance(st, list):
            pool.transcript_speaker_turns = [s for s in st if isinstance(s, dict)][:4]
        # Topic-timeline can come from heuristic (topic_timestamps dict) or GPT (topic_timestamps_ai list)
        tt_ai = structured.get("topic_timestamps_ai") or []
        if isinstance(tt_ai, list) and tt_ai:
            pool.transcript_topic_timeline = [t for t in tt_ai if isinstance(t, dict)][:6]
        else:
            tt_h = structured.get("topic_timestamps") or {}
            if isinstance(tt_h, dict):
                pool.transcript_topic_timeline = [
                    {
                        "topic": k,
                        "start_seconds": v.get("first_seen"),
                        "end_seconds": v.get("last_seen"),
                        "mentions": v.get("mentions"),
                    }
                    for k, v in list(tt_h.items())[:6]
                    if isinstance(v, dict)
                ]
        if not pool.transcript_phrase and structured.get("key_phrase"):
            pool.transcript_phrase = str(structured["key_phrase"])[:140]

    # ── YAMNet top sound class (audio environment) ──────────────────────
    if isinstance(ac, dict):
        top = (ac.get("top_sound_class") or "").strip()
        if top:
            pool.yamnet_top = top.lower()

    # ── Vision (multi-frame fused) ──────────────────────────────────────
    vc = getattr(ctx, "vision_context", None) or {}
    if isinstance(vc, dict):
        pool.vision_labels = [
            str(x).strip().lower()
            for x in (vc.get("label_names") or [])
            if str(x).strip()
        ][:24]
        pool.vision_landmarks = [
            str(x).strip()
            for x in (vc.get("landmark_names") or [])
            if str(x).strip()
        ][:6]
        pool.vision_logos = [
            str(x).strip()
            for x in (vc.get("logo_names") or [])
            if str(x).strip()
        ][:6]
        pool.has_faces = bool(vc.get("has_faces"))
        pool.expressive_faces = bool(vc.get("expressive"))
        ocr = (vc.get("ocr_text") or "").strip()
        if ocr:
            for pat in _HIGHWAY_PATTERNS:
                for m in pat.findall(ocr):
                    s = re.sub(r"[\s\-]+", "", str(m)).upper()
                    if s and s not in pool.vision_highways:
                        pool.vision_highways.append(s)
            pool.vision_ocr_tokens = _extract_transcript_nouns(ocr, limit=8)
        rec_flat = vc.get("recognition_flat") or {}
        if not rec_flat:
            vr = getattr(ctx, "visual_recognition", None) or {}
            if isinstance(vr, dict):
                rec_flat = vr.get("flat") or {}
        if isinstance(rec_flat, dict) and rec_flat:
            pool.recognition_entities = {
                k: [str(x).strip() for x in v[:10] if str(x).strip()]
                for k, v in rec_flat.items()
                if isinstance(v, list) and v
            }
            for key in (
                "vehicles", "food", "plants", "animals", "brands",
                "outdoors", "sports", "art", "restaurants", "products",
            ):
                for name in (rec_flat.get(key) or [])[:6]:
                    low = str(name).strip().lower()
                    if low and low not in pool.vision_labels:
                        pool.vision_labels.append(low)

    # ── Twelve Labs / Video Intelligence narrative ──────────────────────
    vu = getattr(ctx, "video_understanding", None) or {}
    if isinstance(vu, dict):
        scene = (
            vu.get("scene_description")
            or vu.get("description")
            or vu.get("title_suggestion")
            or ""
        )
        if isinstance(scene, str) and scene.strip():
            pool.video_understanding_phrase = _first_sentence(scene, max_chars=140)

    # ── Video Intelligence structured tracks (object/text/person/logo) ──
    # Populated by stages.video_intelligence_stage when full feature set is on.
    vi = getattr(ctx, "video_intelligence", None) or {}
    if isinstance(vi, dict) and vi:
        label_sources: List[Any] = []
        for key in ("top_labels", "labels"):
            raw_labels = vi.get(key) or []
            if isinstance(raw_labels, list):
                label_sources.extend(raw_labels)
        for key in ("segment_labels", "shot_labels"):
            rows = vi.get(key) or []
            if isinstance(rows, list):
                label_sources.extend(
                    row.get("description") for row in rows if isinstance(row, dict)
                )
        for raw in label_sources:
            label = _clean_video_label(raw)
            if label and label.lower() not in {x.lower() for x in pool.video_labels}:
                pool.video_labels.append(label)
            if len(pool.video_labels) >= 16:
                break
        summary = vi.get("summary_text") or vi.get("summary") or ""
        if isinstance(summary, str) and summary.strip():
            pool.video_summary_phrase = _first_sentence(summary, max_chars=160)
        ot = vi.get("object_tracks") or []
        if isinstance(ot, list):
            pool.vi_object_tracks = [t for t in ot if isinstance(t, dict)][:12]
        td = vi.get("on_screen_text") or vi.get("text_detections") or []
        if isinstance(td, list):
            pool.vi_text_detections = [
                t if isinstance(t, dict) else {"text": str(t)}
                for t in td
            ][:12]
        ps = vi.get("person_segments") or []
        if isinstance(ps, list):
            pool.vi_person_segments = [s for s in ps if isinstance(s, dict)][:12]
        lg = vi.get("logos") or []
        if isinstance(lg, list):
            pool.vi_logos = [l for l in lg if isinstance(l, dict)][:8]

    # Older/legacy callers may only populate ``video_intelligence_context``.
    vic = getattr(ctx, "video_intelligence_context", None) or {}
    if isinstance(vic, dict) and vic and not vic.get("error"):
        if not pool.video_labels:
            label_sources = []
            raw_top = vic.get("top_labels") or []
            if isinstance(raw_top, list):
                label_sources.extend(raw_top)
            for key in ("segment_labels", "shot_labels"):
                rows = vic.get(key) or []
                if isinstance(rows, list):
                    label_sources.extend(
                        row.get("description") for row in rows if isinstance(row, dict)
                    )
            for raw in label_sources:
                label = _clean_video_label(raw)
                if label and label.lower() not in {x.lower() for x in pool.video_labels}:
                    pool.video_labels.append(label)
                if len(pool.video_labels) >= 16:
                    break
        if not pool.video_summary_phrase:
            summary = vic.get("summary_text") or ""
            if isinstance(summary, str) and summary.strip():
                pool.video_summary_phrase = _first_sentence(summary, max_chars=160)

    cat = str(getattr(ctx, "thumbnail_category", None) or "").strip().lower()
    if not cat:
        hp = getattr(ctx, "hydration_payload", None) or {}
        if isinstance(hp, dict):
            cat = str(hp.get("category") or "").strip().lower()
    if not cat:
        cat = "general"
    ambient = resolve_ambient_profiles(
        category=cat,
        filename=str(getattr(ctx, "filename", "") or ""),
        vision_label_names=pool.vision_labels,
    )
    pool.ambient_profiles = sorted(ambient)
    pool.vision_labels = [
        lbl
        for lbl in pool.vision_labels
        if not is_redundant_vision_label(lbl, ambient_profiles=ambient)
    ]
    pool.video_labels = [
        lbl
        for lbl in pool.video_labels
        if not is_redundant_vision_label(lbl, ambient_profiles=ambient)
    ]

    try:
        from services.place_evidence import merge_place_evidence_into_pool

        pe = getattr(ctx, "place_evidence", None)
        if not isinstance(pe, dict):
            arts = getattr(ctx, "output_artifacts", None) or {}
            if isinstance(arts, dict):
                pe = arts.get("place_evidence_v1")
        merge_place_evidence_into_pool(pool, pe if isinstance(pe, dict) else None)
    except Exception:
        pass

    return pool


# ---------------------------------------------------------------------------
# Deterministic phrase / hashtag builders
# ---------------------------------------------------------------------------


def _sanitize_anchor_fragment(text: str, *, max_chars: int = 88) -> str:
    if not text:
        return ""
    t = re.sub(r"\s+", " ", str(text).strip())
    t = re.sub(r"[\x00-\x1f\x7f]", "", t)
    if len(t) > max_chars:
        t = t[: max_chars - 1].rstrip() + "…"
    return t


def _transcript_fragment_for_anchor(ctx: JobContext) -> str:
    """Short spoken-text snippet for thin-evidence anchors (Whisper / merged transcript)."""
    raw = (getattr(ctx, "ai_transcript", None) or "").strip()
    ac = getattr(ctx, "audio_context", None) or {}
    if isinstance(ac, dict):
        if not raw:
            raw = str(ac.get("transcript") or "").strip()
        if not raw:
            ts = ac.get("transcript_structured") or {}
            if isinstance(ts, dict) and ts.get("key_phrase"):
                raw = str(ts["key_phrase"]).strip()
        if not raw:
            segs = ac.get("transcript_segments") or []
            if isinstance(segs, list) and segs:
                first = segs[0]
                if isinstance(first, dict):
                    raw = str(first.get("text") or "").strip()
    if not raw:
        return ""
    sent = _first_sentence(raw, max_chars=96)
    if sent:
        return _sanitize_anchor_fragment(sent, max_chars=88)
    if len(raw) >= 8:
        return _sanitize_anchor_fragment(raw, max_chars=88)
    return ""


def _base_file_anchor(ctx: JobContext, category: Optional[str] = None) -> str:
    """Category + filename-derived identity (no persona, no transcript).

    Same date/#seq parsing as the thin-evidence fallback so anchors stay
    consistent across ``build_anchor_phrase`` and ``enforce_hydration``.
    """
    cat_raw = (category or getattr(ctx, "thumbnail_category", None) or "general").strip().lower()
    noun = _FILE_CATEGORY_HINTS.get(cat_raw, _FILE_CATEGORY_HINTS["general"])
    fname = (getattr(ctx, "filename", None) or "").strip()
    if not fname:
        return f"{noun}."

    stem = re.sub(r"\.[A-Za-z0-9]{2,5}$", "", fname).strip()
    date_part: str = ""
    file_seq: str = ""
    for pat in _DASHCAM_FILENAME_PATTERNS:
        m = pat.search(stem)
        if not m:
            continue
        groups = [g for g in m.groups() if g]
        if len(groups) >= 4 and groups[0].isdigit() and len(groups[0]) == 4:
            date_part = f"{groups[0]}-{groups[1]}-{groups[2]}"
            file_seq = groups[3]
        elif groups:
            file_seq = groups[0]
        break

    bits: List[str] = [noun]
    if date_part:
        bits.append(f"from {date_part}")
    if file_seq:
        bits.append(f"#{file_seq}")
    if not date_part and not file_seq:
        bits.append(f"({stem[:48]})")
    return " ".join(bits).strip().rstrip(".") + "."


def _format_place(pool: EvidencePool) -> Optional[str]:
    """Compose a human place phrase like ``Guadalupe, CA`` or ``near I-15``."""
    parts: List[str] = []
    if pool.gazetteer_place and pool.state_abbr:
        return f"{pool.gazetteer_place}, {pool.state_abbr}"
    if pool.gazetteer_place:
        return pool.gazetteer_place
    if pool.city and pool.state_abbr:
        return f"{pool.city}, {pool.state_abbr}"
    if pool.city and pool.state:
        return f"{pool.city}, {pool.state}"
    if pool.city:
        return pool.city
    if pool.protected_area:
        return pool.protected_area
    if pool.road:
        return pool.road
    if pool.vision_highways:
        return pool.vision_highways[0]
    return " ".join(parts) if parts else None


def build_anchor_phrase(pool: EvidencePool, ctx: Optional[JobContext] = None) -> str:
    """Compose a single deterministic factual sentence summarizing evidence.

    Ordered for read-out fluency:
      "Captured at 46 MPH on Highway 1 near Guadalupe, CA — The Eagles
       'Hotel California' playing." — chunks omitted when their signal
       is missing. Returns "" when evidence is empty.

    When only speech / scene-understanding / ambience is present (no speed,
    geo, or music), pass ``ctx`` so the anchor still ties to the uploaded file
    (category + filename date/slug) for authentic grounding.
    """
    bits: List[str] = []

    # Speed leads when present (it's the highest-info short anchor).
    if pool.max_speed_mph and pool.max_speed_mph >= 5:
        bits.append(f"Captured at {int(round(pool.max_speed_mph))} MPH")
    elif pool.trill_bucket:
        bits.append(f"Trill bucket: {pool.trill_bucket}")
    elif pool.driver_name:
        bits.append(f"Driver {pool.driver_name}")

    # Place phrase.
    place = _format_place(pool)
    if place:
        prefix = "near" if bits else "Captured near"
        bits.append(f"{prefix} {place}")

    # Music when ACR matched.
    if pool.music_artist and pool.music_title:
        bits.append(f"with {pool.music_artist} — {pool.music_title} on the speakers")
    elif pool.music_title:
        bits.append(f"with {pool.music_title} on the speakers")
    elif pool.music_artist:
        bits.append(f"with {pool.music_artist} on the speakers")

    # Vision landmark/logo callout when no place phrase yet.
    if not place and pool.vision_landmarks:
        bits.append(f"near {pool.vision_landmarks[0]}")

    # Closing sound or transcript phrase.
    closing: Optional[str] = None
    if pool.transcript_phrase:
        closing = pool.transcript_phrase.rstrip(".!?")
    elif pool.video_understanding_phrase:
        closing = pool.video_understanding_phrase.rstrip(".!?")
    elif pool.video_summary_phrase:
        closing = pool.video_summary_phrase.rstrip(".!?")
    elif pool.video_labels:
        closing = "Video analysis detected " + ", ".join(pool.video_labels[:4])
    elif pool.yamnet_top:
        closing = f"ambient {pool.yamnet_top}"

    if not bits and not closing:
        # No evidence at all → caller should skip rewrite.
        return ""

    head = ", ".join(bits) if bits else ""
    if closing and head:
        return f"{head}. {closing}."
    if closing:
        if ctx is not None:
            tail = _base_file_anchor(ctx).rstrip(".").strip()
            if tail and tail.lower() not in closing.lower():
                return f"{closing}. {tail}."
        return f"{closing}."
    return f"{head}."


def build_evidence_hashtags(pool: EvidencePool, *, max_extra: int = 14) -> List[str]:
    """Deterministic, evidence-driven hashtag bodies (no leading '#').

    Priority encodes survival under maxHashtags truncation:
      1. landmark name(s)
      2. music artist + title
      3. road / highway / route
      4. gazetteer place + state composite
      5. city / state
      6. protected-area unit name
      7. driver name (HUD)
      8. trill bucket tags
      9. speed bucket tags
     10. vision label proper nouns (if not a known generic)
     11. transcript proper nouns (filtered)
    """
    out: List[str] = []
    seen: set = set()

    def _push(raw: Any) -> bool:
        body = sanitize_hashtag_body(str(raw or ""), max_len=36)
        if not body or body in seen:
            return False
        seen.add(body)
        out.append(body)
        return True

    for lm in pool.vision_landmarks[:3]:
        _push(lm)

    for beach in list(getattr(pool, "place_beaches", None) or [])[:2]:
        _push(beach)
    for mon in list(getattr(pool, "place_monuments", None) or [])[:2]:
        _push(mon)
    for team in list(getattr(pool, "sports_teams", None) or [])[:2]:
        _push(team)
    for stad in list(getattr(pool, "place_stadiums", None) or [])[:2]:
        _push(stad)

    if pool.music_artist:
        _push(pool.music_artist)
    if pool.music_title:
        _push(pool.music_title)
    if pool.music_genre:
        _push(pool.music_genre)

    if pool.road:
        _push(pool.road)
    for hwy in pool.vision_highways[:2]:
        _push(hwy)

    if pool.gazetteer_place:
        if pool.state_abbr:
            _push(f"{pool.gazetteer_place}{pool.state_abbr}")
        else:
            _push(pool.gazetteer_place)
    if pool.city:
        if pool.state_abbr:
            _push(f"{pool.city}{pool.state_abbr}")
        _push(pool.city)
    if pool.state:
        _push(pool.state)
    if pool.protected_area:
        _push(pool.protected_area)
    elif pool.near_protected_land:
        _push("publiclands")

    if pool.driver_name:
        _push(pool.driver_name)

    if pool.trill_bucket:
        bucket_tags = {
            "gloryBoy": ["GloryBoyTour", "TrillScore100", "SendIt"],
            "euphoric": ["Euphoric", "TrillScore", "SpeedDemon"],
            "sendIt":   ["SendIt", "TrillScore", "Spirited"],
            "spirited": ["SpiritedDrive", "TrillScore"],
            "chill":    ["TrillScore", "CruiseControl"],
        }.get(pool.trill_bucket, [])
        for t in bucket_tags[:3]:
            _push(t)

    if pool.max_speed_mph >= 130:
        for t in ("TripleDigits", "TopSpeed"):
            _push(t)
    elif pool.max_speed_mph >= 100:
        _push("TripleDigits")
    elif pool.max_speed_mph >= 80:
        _push("HighwaySpeed")
    elif pool.max_speed_mph >= 60:
        _push("FreewayDrive")

    # Vision logo brands.
    for logo in pool.vision_logos[:2]:
        _push(logo)

    # Vision / VI segment labels: only when no stronger geo/VI text/landmark signals,
    # and never generic / ambient-redundant detector slugs.
    strong_signals = evidence_pool_has_strong_hashtag_signals(pool)
    ambient = tuple(pool.ambient_profiles or ())
    if not strong_signals:
        for lbl in filter_vision_labels_for_hashtags(
            pool.vision_labels,
            min_specific_len=4,
            ambient_profiles=ambient or None,
        ):
            _push(vision_label_slug(lbl) or lbl)

    for lbl in pool.video_labels[:8]:
        if is_redundant_vision_label(lbl, ambient_profiles=ambient or None):
            continue
        slug = vision_label_slug(lbl)
        if slug:
            _push(slug)

    # Transcript STRUCTURED entities (people / places / products / orgs from GPT pass).
    for kind in ("places", "products", "organizations", "people"):
        for e in (pool.transcript_entities.get(kind) or [])[:3]:
            _push(e)

    # Transcript topics (compact noun phrases from GPT pass).
    for t in pool.transcript_topics[:3]:
        _push(t)

    # Transcript proper-noun-ish tokens (regex fallback).
    for n in pool.transcript_nouns[:4]:
        _push(n)

    # VI logos (Tesla, In-N-Out, etc.) – brand callouts beat generic labels.
    for lg in pool.vi_logos[:3]:
        if isinstance(lg, dict) and lg.get("description"):
            _push(str(lg["description"]))

    # VI object descriptions (Tesla Model 3, dog, bicycle, etc.) where
    # confidence is reasonable. Skip very generic tracks ("Vehicle").
    _GENERIC_VI = {"vehicle", "person", "object", "animal", "structure", "land vehicle"}
    for ot in pool.vi_object_tracks[:5]:
        if isinstance(ot, dict):
            desc = str(ot.get("description") or "").strip()
            if desc and desc.lower() not in _GENERIC_VI:
                _push(desc)

    if len(out) > max_extra:
        out = out[:max_extra]
    return out


# ---------------------------------------------------------------------------
# Generic detection on AI output
# ---------------------------------------------------------------------------


def _caption_uses_evidence(caption: str, pool: EvidencePool) -> bool:
    """True when ``caption`` mentions ANY token from the evidence pool."""
    if not caption:
        return False
    blob = caption.lower()
    candidates: List[str] = []
    for v in (
        pool.road, pool.city, pool.state, pool.gazetteer_place, pool.protected_area,
        pool.driver_name, pool.music_artist, pool.music_title, pool.music_genre,
        pool.trill_bucket, pool.video_understanding_phrase, pool.video_summary_phrase,
        pool.yamnet_top,
    ):
        if v:
            candidates.append(str(v).lower())
    candidates.extend(pool.vision_landmarks)
    candidates.extend(pool.vision_logos)
    candidates.extend(pool.vision_highways)
    candidates.extend(pool.video_labels)
    candidates.extend(pool.transcript_nouns)
    candidates.extend(pool.transcript_topics)
    for kind in ("places", "products", "organizations", "people"):
        candidates.extend(pool.transcript_entities.get(kind) or [])
    for ot in pool.vi_object_tracks:
        if isinstance(ot, dict) and ot.get("description"):
            candidates.append(str(ot.get("description")))
    for lg in pool.vi_logos:
        if isinstance(lg, dict) and lg.get("description"):
            candidates.append(str(lg.get("description")))
    for td in pool.vi_text_detections:
        if isinstance(td, dict) and td.get("text"):
            candidates.append(str(td.get("text")))
    candidates.extend(pool.vision_ocr_tokens)
    if pool.max_speed_mph >= 5:
        candidates.append(f"{int(round(pool.max_speed_mph))}")
        candidates.append(f"{int(round(pool.max_speed_mph))} mph")
    for c in candidates:
        if not c:
            continue
        c2 = c.lower().strip()
        if len(c2) < 3:
            continue
        if c2 in blob:
            return True
    if _EVIDENCE_BEARING_HINTS.search(caption):
        return True
    return False


def _is_generic_caption(caption: str) -> bool:
    if not caption or len(caption.strip()) < 12:
        return True
    for pat in _GENERIC_CAPTION_PATTERNS:
        if pat.search(caption):
            return True
    return False


def _hashtags_are_seed_only(tags: Iterable[str]) -> Tuple[bool, int, int]:
    """Return (seed_only, seed_count, total_count)."""
    total = 0
    seed = 0
    for t in tags or []:
        body = sanitize_hashtag_body(str(t))
        if not body:
            continue
        total += 1
        if body in _CATEGORY_SEED_TAGS:
            seed += 1
    if total == 0:
        return False, 0, 0
    return (seed >= max(3, total // 2)), seed, total


# ---------------------------------------------------------------------------
# Rewrite helpers
# ---------------------------------------------------------------------------


def _hydrate_caption(caption: str, anchor: str, *, max_chars: int = 520) -> str:
    """Append/insert anchor into a caption when it doesn't already cover it.

    Strategy:
      * If caption is empty → return anchor.
      * If caption is generic → REPLACE with anchor + lightly preserve any
        non-generic clause from the original.
      * Else → append anchor as a new sentence (bounded to ``max_chars``).
    """
    cap = (caption or "").strip()
    a = (anchor or "").strip()
    if not a:
        return cap[:max_chars]
    if not cap:
        return a[:max_chars]
    if _is_generic_caption(cap):
        return a[:max_chars]
    glue = " " if cap.endswith((".", "!", "?")) else ". "
    out = (cap + glue + a).strip()
    return out[:max_chars]


def _hydrate_title(title: str, anchor: str, *, max_chars: int = 100) -> str:
    """Replace title with anchor when it's empty or clearly generic. Keeps editorial titles."""
    t = (title or "").strip()
    a = (anchor or "").strip()
    if not a:
        return t[:max_chars]
    if not t:
        return a[:max_chars]
    # Long or subtitle-style headlines must not be replaced by a raw speed+music
    # anchor. ``_is_generic_caption`` is substring-based (e.g. ``open road`` matches
    # inside ``…Through the Open Road``) and was clobbering good LLM titles with
    # truncated anchors that pulled profanity from transcript-adjacent evidence.
    if len(t) >= 36 or (":" in t and len(t) >= 18):
        return t[:max_chars]
    if _is_generic_caption(t):
        return a[:max_chars]
    return t[:max_chars]


def _merge_hashtag_lists(*lists: Iterable[str], cap: Optional[int] = None) -> List[str]:
    """Merge in priority order, deduplicating by sanitized body. Optional cap."""
    out: List[str] = []
    seen: set = set()
    for lst in lists:
        for raw in lst or []:
            body = sanitize_hashtag_body(str(raw))
            if not body or body in seen:
                continue
            seen.add(body)
            out.append(body)
            if cap is not None and len(out) >= cap:
                return out
    return out


def _purge_seed_tags_when_evidence(
    tags: List[str], evidence_tags: List[str]
) -> List[str]:
    """Drop category-seed tags ONLY when we have evidence-driven tags to replace them."""
    if not evidence_tags:
        return tags
    evidence_lower = {t.lower() for t in evidence_tags}
    out: List[str] = []
    for raw in tags or []:
        body = sanitize_hashtag_body(str(raw))
        if not body:
            continue
        if body.lower() in evidence_lower:
            out.append(body)
            continue
        if body in _CATEGORY_SEED_TAGS:
            # Drop — we have a real signal that should take its slot.
            continue
        out.append(body)
    return out


# ---------------------------------------------------------------------------
# Filename / category fallback (used when evidence pool is empty)
# ---------------------------------------------------------------------------

# Hint: dashcam-style filenames the M8 cohort tends to upload (DDR, BlackVue,
# Viofo, ESCORT, INSTA360, Garmin etc.). When OSD/.map/Vision all stay empty
# we still want a *truthful* anchor that is at least specific to the file
# the user uploaded so the published copy stops reading like stock filler.
_DASHCAM_FILENAME_PATTERNS: List[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"(\d{4})(\d{2})(\d{2})[_-](\d{2,4})[_-]?(?:CAM[_-]?EVNT|EVNT|EVENT|REC|MOV|CLIP|DASH)?",
        r"GH(\d{6,})",                          # GoPro Hero
        r"DJI[_-]?(\d{4,})",                    # DJI / Osmo
        r"VID[_-]?(\d{8,})",                    # generic Android camera
        r"MVI[_-]?(\d{4,})",                    # Canon
        r"PXL[_-]?(\d{8,})",                    # Pixel
    )
]

_FILE_CATEGORY_HINTS: Dict[str, str] = {
    "automotive": "Dashcam clip",
    "travel":     "Travel clip",
    "sports":     "Action capture",
    "music":      "Performance clip",
    "tech":       "Gear demo clip",
    "fitness":    "Training clip",
    "gaming":     "Gameplay clip",
    "food":       "Cooking clip",
    "beauty":     "Beauty look clip",
    "fashion":    "Fit-check clip",
    "pets":       "Pet clip",
    "lifestyle":  "Day-in-life clip",
    "comedy":     "Skit clip",
    "education":  "Lesson clip",
    "real_estate":"Property tour",
    "general":    "Video clip",
}


def _persona_display_name(ctx: JobContext) -> str:
    """Pull the user's saved persona display name (e.g. "gloc") from settings.

    Populated by ``stages.db.merge_pikzels_thumbnail_persona_id`` from the
    ``creator_personas.name`` column. Used as a brand prefix in the fallback
    anchor so the user's "I have a persona set" intent visibly survives even
    when the evidence pool is empty.
    """
    us = getattr(ctx, "user_settings", None) or {}
    if not isinstance(us, dict):
        return ""
    raw = (
        us.get("thumbnail_persona_display_name")
        or us.get("thumbnailPersonaDisplayName")
        or ""
    )
    return str(raw or "").strip()[:40]


def _fallback_anchor_from_ctx(ctx: JobContext, category: Optional[str] = None) -> str:
    """Deterministic, truthful anchor for thin-evidence clips.

    Composes ``"[<persona> · ]<category + filename>`` plus an optional
    **transcript fragment** (Whisper / structured key phrase / first segment)
    when speech exists but did not surface in the evidence pool — so copy
    stays grounded in what was actually said, not only the file name.

    Returns ``""`` when there is no filename and no category tail (extremely rare).
    """
    persona = _persona_display_name(ctx)
    base = _base_file_anchor(ctx, category=category).rstrip(".").strip()
    if not base:
        return ""
    prefix = f"{persona} · " if persona else ""
    out = f"{prefix}{base}."
    frag = _transcript_fragment_for_anchor(ctx)
    if frag and frag.lower() not in out.lower():
        frag = _sanitize_anchor_fragment(frag, max_chars=max(24, 230 - len(out)))
        if frag:
            out = f"{out[:-1]} — heard: {frag}." if out.endswith(".") else f"{out} — heard: {frag}."
    return out


def _scrub_leaked_junk_hashtags(tags: Iterable[str]) -> List[str]:
    """Remove obvious QA/placeholder tokens from AI hashtag lists."""
    from core.helpers import sanitize_hashtag_body

    banned = frozenset({"tester", "qwe", "asdf", "foobar", "lorem", "ipsum"})
    out: List[str] = []
    for raw in tags or []:
        body = sanitize_hashtag_body(str(raw).strip().lstrip("#"))
        if not body:
            continue
        if body.lower() in banned:
            continue
        out.append(body)
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def enforce_hydration(
    ctx: JobContext,
    *,
    max_anchor_chars: int = 240,
    max_extra_hashtags: int = 14,
) -> Dict[str, Any]:
    """Final deterministic guarantor that captions/hashtags reflect evidence.

    Mutates ``ctx`` in-place. Returns a structured *report* dict (also logged):

        {
          "evidence_present": bool,
          "rewrote_caption": bool,
          "rewrote_title": bool,
          "purged_seed_tags": int,
          "anchor": str,
          "evidence": EvidencePool.to_report(),
          "warnings": [...],
        }

    The function is idempotent — re-running on the same ctx after it has
    already been hydrated does not re-rewrite (anchor matches, caption
    already mentions evidence, etc).
    """
    pool = collect_evidence(ctx)
    has_evidence = pool.has_any_evidence()
    anchor = build_anchor_phrase(pool, ctx) if has_evidence else ""
    fallback_anchor = ""
    used_fallback = False
    if not anchor:
        fallback_anchor = _fallback_anchor_from_ctx(ctx)
        anchor = fallback_anchor
        used_fallback = bool(fallback_anchor)
    if anchor and len(anchor) > max_anchor_chars:
        anchor = anchor[: max_anchor_chars - 1].rstrip() + "…"

    evidence_tags = build_evidence_hashtags(pool, max_extra=max_extra_hashtags)

    report: Dict[str, Any] = {
        "evidence_present": has_evidence,
        "used_fallback_anchor": used_fallback,
        "rewrote_caption": False,
        "rewrote_title": False,
        "purged_seed_tags": 0,
        "added_evidence_tags": 0,
        "anchor": anchor,
        "hydration_story": build_hydration_story_text(ctx, max_chars=900),
        "evidence_tags": list(evidence_tags),
        "evidence": pool.to_report(),
        "warnings": [],
    }

    # ── Loud warning when no signals at all ──────────────────────────────
    if not has_evidence:
        reasons: List[str] = []
        if not getattr(ctx, "vision_context", None):
            reasons.append("vision_context empty (Google Vision skipped, missing credentials, or 0 labels)")
        if not getattr(ctx, "audio_context", None):
            reasons.append("audio_context empty (audio stage skipped)")
        if not (getattr(ctx, "telemetry", None) or getattr(ctx, "telemetry_data", None)):
            reasons.append("telemetry not provided (no .map and OSD did not backfill)")
        if not getattr(ctx, "dashcam_osd_context", None):
            reasons.append("dashcam_osd_context empty (OSD disabled or no HUD detected)")
        vic = getattr(ctx, "video_intelligence_context", None) or {}
        vit = getattr(ctx, "video_intelligence", None) or {}
        vi_has_tracks = bool(
            (isinstance(vit, dict) and (vit.get("object_tracks") or vit.get("logos")))
            or (
                isinstance(vic, dict)
                and (vic.get("object_tracks") or vic.get("segment_labels") or vic.get("top_labels"))
                and not vic.get("error")
            )
        )
        if not vi_has_tracks:
            if isinstance(vic, dict) and vic.get("error"):
                reasons.append(
                    f"video_intelligence failed or empty ({str(vic.get('error'))[:100]})"
                )
            else:
                reasons.append(
                    "video_intelligence empty (GCP Video Intelligence missing credentials, limits, "
                    "or object/label tracks)"
                )
        vu = getattr(ctx, "video_understanding", None) or {}
        if not (isinstance(vu, dict) and str(vu.get("scene_description") or "").strip()):
            reasons.append(
                "video_understanding empty (Twelve Labs skipped: disabled in prefs, no API key, "
                "VI-rich cost gate, or indexing failed)"
            )
        tx = (getattr(ctx, "ai_transcript", None) or "").strip()
        ac_warn = getattr(ctx, "audio_context", None) or {}
        if not tx and isinstance(ac_warn, dict):
            tx = str(ac_warn.get("transcript") or "").strip()
        if not tx and isinstance(ac_warn, dict):
            segs_w = ac_warn.get("transcript_segments") or []
            if isinstance(segs_w, list) and segs_w:
                first_seg = segs_w[0]
                if isinstance(first_seg, dict):
                    tx = str(first_seg.get("text") or "").strip()
        if not tx:
            reasons.append(
                "no speech transcript (Whisper off in prefs, audio stage skipped tier/can_ai, "
                "OPENAI_API_KEY missing, or clip has no usable speech)"
            )
        report["warnings"].extend(reasons)
        logger.warning(
            "[hydration] NO evidence captured for upload %s — captions will be %s. Reasons: %s",
            getattr(ctx, "upload_id", "?"),
            "rewritten with filename/category fallback anchor" if used_fallback else "generic",
            "; ".join(reasons) if reasons else "unknown",
        )
        # Fall through (do not return) so generic-pattern captions still get
        # rewritten with the fallback anchor and seed-only hashtags still get
        # purged. Without this the M8 cliché survives unmodified.

    logger.info(
        "[hydration] evidence captured for upload %s: %s (fallback_anchor=%s)",
        getattr(ctx, "upload_id", "?"),
        report["evidence"],
        used_fallback,
    )

    # ── Per-platform M8 captions / titles ────────────────────────────────
    m8_captions = getattr(ctx, "m8_platform_captions", None) or {}
    m8_titles = getattr(ctx, "m8_platform_titles", None) or {}
    m8_hashtags = getattr(ctx, "m8_platform_hashtags", None) or {}

    def _maybe_rewrite_caption(cap_str: str) -> Optional[str]:
        """Return new caption when a rewrite is warranted, else None.

        Evidence-rich anchor → APPEND when caption already non-generic, REPLACE when generic.
        Fallback (filename/category) anchor → only REPLACE when caption is generic.
        Never append the thin fallback to good copy — that just adds noise.
        """
        if not anchor:
            return None
        if _caption_uses_evidence(cap_str, pool):
            return None
        if used_fallback:
            if not _is_generic_caption(cap_str):
                return None
            new = _hydrate_caption(cap_str, anchor)
            return new if new and new != cap_str else None
        new = _hydrate_caption(cap_str, anchor)
        return new if new and new != cap_str else None

    def _maybe_rewrite_title(ttl_str: str) -> Optional[str]:
        if not anchor:
            return None
        if _caption_uses_evidence(ttl_str, pool):
            return None
        if used_fallback:
            if not _is_generic_caption(ttl_str):
                return None
            new = _hydrate_title(ttl_str, anchor)
            return new if new and new != ttl_str else None
        new = _hydrate_title(ttl_str, anchor)
        return new if new and new != ttl_str else None

    if isinstance(m8_captions, dict):
        for pl, cap in list(m8_captions.items()):
            cap_str = str(cap or "")
            new = _maybe_rewrite_caption(cap_str)
            if new is not None:
                m8_captions[pl] = new
                report["rewrote_caption"] = True

    if isinstance(m8_titles, dict):
        for pl, ttl in list(m8_titles.items()):
            ttl_str = str(ttl or "")
            new = _maybe_rewrite_title(ttl_str)
            if new is not None:
                m8_titles[pl] = new
                report["rewrote_title"] = True

    # ── Per-platform M8 hashtags: replace seed-only with evidence ────────
    if isinstance(m8_hashtags, dict):
        for pl, raw_list in list(m8_hashtags.items()):
            tags = list(raw_list) if isinstance(raw_list, list) else []
            seed_only, _seed_n, _total_n = _hashtags_are_seed_only(tags)
            purged = _purge_seed_tags_when_evidence(tags, evidence_tags)
            cap = max(len(tags), len(evidence_tags))
            merged = _merge_hashtag_lists(evidence_tags, purged, cap=cap)
            if merged != tags:
                report["purged_seed_tags"] += len(tags) - len(purged)
                report["added_evidence_tags"] += max(0, len(merged) - len(purged))
                m8_hashtags[pl] = merged
            elif seed_only and evidence_tags:
                m8_hashtags[pl] = _merge_hashtag_lists(evidence_tags, tags, cap=cap)

    # ── Legacy ai_caption / ai_title / ai_hashtags fallbacks ────────────
    ai_caption = getattr(ctx, "ai_caption", "") or ""
    new = _maybe_rewrite_caption(ai_caption)
    if new is not None:
        ctx.ai_caption = new
        report["rewrote_caption"] = True

    ai_title = getattr(ctx, "ai_title", "") or ""
    new = _maybe_rewrite_title(ai_title)
    if new is not None:
        ctx.ai_title = new
        report["rewrote_title"] = True

    if evidence_tags:
        existing = list(getattr(ctx, "ai_hashtags", None) or [])
        seed_only, _s, _t = _hashtags_are_seed_only(existing)
        purged = _purge_seed_tags_when_evidence(existing, evidence_tags)
        cap = max(len(existing), len(evidence_tags))
        merged = _merge_hashtag_lists(evidence_tags, purged, cap=cap)
        if merged != existing:
            report["purged_seed_tags"] += len(existing) - len(purged)
            report["added_evidence_tags"] += max(0, len(merged) - len(purged))
            ctx.ai_hashtags = merged
        elif seed_only:
            ctx.ai_hashtags = _merge_hashtag_lists(evidence_tags, existing, cap=cap)

    ctx.ai_hashtags = _scrub_leaked_junk_hashtags(list(getattr(ctx, "ai_hashtags", None) or []))
    if isinstance(m8_hashtags, dict):
        for pl, raw_list in list(m8_hashtags.items()):
            m8_hashtags[pl] = _scrub_leaked_junk_hashtags(list(raw_list or []))

    # ── Metadata quality notes (surfaced in hydration_report / admin trace) ─
    qual: List[str] = []
    cap_all = f"{getattr(ctx, 'ai_title', '')} {getattr(ctx, 'ai_caption', '')}".strip()
    if has_evidence and cap_all and not used_fallback:
        if not _caption_uses_evidence(cap_all, pool):
            qual.append("combined_copy_missing_explicit_evidence_token")
        if _is_generic_caption(cap_all):
            qual.append("combined_copy_matches_generic_template")
    if qual:
        report["warnings"].extend(qual)
    report["metadata_quality"] = {"notes": qual}

    try:
        from services.grounding_eval import score_ctx_grounding

        grounding = score_ctx_grounding(ctx, pool, evidence_present=has_evidence)
        report["grounding"] = grounding
        report["grounding_score"] = grounding.get("grounding_score")
    except Exception as _ge:
        logger.debug("grounding_eval skipped: %s", _ge)
        report["grounding"] = {"status": "error"}
        report["grounding_score"] = None

    # Persist report for diagnostics if ctx supports it.
    try:
        if not isinstance(getattr(ctx, "output_artifacts", None), dict):
            ctx.output_artifacts = {}
        ctx.output_artifacts["hydration_story"] = report["hydration_story"]
        ctx.output_artifacts["hydration_report"] = {
            "evidence_present": report["evidence_present"],
            "rewrote_caption": report["rewrote_caption"],
            "rewrote_title": report["rewrote_title"],
            "purged_seed_tags": report["purged_seed_tags"],
            "added_evidence_tags": report["added_evidence_tags"],
            "anchor": report["anchor"],
            "hydration_story": report["hydration_story"],
            "evidence_tags": list(report["evidence_tags"]),
            "evidence": report["evidence"],
            "warnings": report["warnings"],
            "metadata_quality": report.get("metadata_quality") or {},
            "grounding_score": report.get("grounding_score"),
            "grounding": report.get("grounding") or {},
        }
        if report.get("grounding"):
            ctx.output_artifacts["grounding_score_v1"] = report["grounding"]
    except (AttributeError, TypeError):
        pass

    # Self-persist: hydration_report is the deterministic answer to "did the
    # hydration enforcer find evidence and rewrite anything for this upload?".
    # It must always reach the DB, even when caption_stage skipped before
    # save_generated_metadata could fire.
    try:
        from services.diag_persist import schedule_persist_artifact_now

        schedule_persist_artifact_now(
            ctx,
            "hydration_report",
            "hydration_story",
            "grounding_score_v1",
        )
    except Exception:
        pass

    logger.info(
        "[hydration] enforce_hydration upload=%s rewrote_caption=%s rewrote_title=%s "
        "purged_seed_tags=%s added_evidence_tags=%s anchor=%r",
        getattr(ctx, "upload_id", "?"),
        report["rewrote_caption"],
        report["rewrote_title"],
        report["purged_seed_tags"],
        report["added_evidence_tags"],
        anchor,
    )
    return report


__all__ = [
    "EvidencePool",
    "collect_evidence",
    "build_anchor_phrase",
    "build_evidence_hashtags",
    "enforce_hydration",
]
