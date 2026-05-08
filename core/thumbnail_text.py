from __future__ import annotations

import re
from typing import Any, Dict, FrozenSet


GENERIC_THUMBNAIL_HEADLINES = {
    "amazing moment",
    "amazing moments",
    "best moment",
    "big moment",
    "crazy moment",
    "crazy moments",
    "dont miss this",
    "epic clip",
    "epic moment",
    "epic moments",
    "exciting moment",
    "exciting moments",
    "insane moment",
    "insane moments",
    "must watch",
    "new upload",
    "unforgettable moment",
    "unbelievable moment",
    "unbelievable moments",
    "viral moment",
    "watch",
    "watch now",
    "watch this",
    "wow moment",
    "you wont believe",
}

# Evidence-empty defaults from thumbnail_stage._concrete_thumbnail_headline — still generic
# for Pikzels image text rendering (model hallucinates clichés when told to paint these).
CATEGORY_HEADLINE_FALLBACKS: Dict[str, str] = {
    "automotive": "ROAD HIGHLIGHT",
    "beauty": "FINAL LOOK",
    "food": "FINISHED DISH",
    "home_renovation": "FINAL REVEAL",
    "gardening": "GARDEN UPDATE",
    "fitness": "PEAK EFFORT",
    "fashion": "FIT CHECK",
    "gaming": "GAMEPLAY HIGHLIGHT",
    "travel": "SCENIC STOP",
    "pets": "PET CLOSEUP",
    "education": "KEY LESSON",
    "comedy": "REACTION SHOT",
    "tech": "GEAR CLOSEUP",
    "music": "PERFORMANCE SHOT",
    "real_estate": "PROPERTY FEATURE",
    "sports": "ACTION PLAY",
    "asmr": "TEXTURE CLOSEUP",
    "lifestyle": "DAY HIGHLIGHT",
    "general": "VIDEO HIGHLIGHT",
}

THUMBNAIL_CATEGORY_FALLBACK_HEADLINES: FrozenSet[str] = frozenset(
    {v.strip().upper() for v in CATEGORY_HEADLINE_FALLBACKS.values() if str(v).strip()}
)

_GENERIC_WORDS = {
    "exciting",
    "amazing",
    "epic",
    "crazy",
    "insane",
    "unbelievable",
    "unforgettable",
    "moment",
    "moments",
    "clip",
    "highlight",
    "highlights",
}


def thumbnail_headline_body(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def is_generic_thumbnail_headline(text: Any) -> bool:
    body = thumbnail_headline_body(text)
    if not body:
        return True
    if body in GENERIC_THUMBNAIL_HEADLINES:
        return True
    words = body.split()
    if len(words) <= 2 and all(w in _GENERIC_WORDS for w in words):
        return True
    for phrase in GENERIC_THUMBNAIL_HEADLINES:
        if " " in phrase and phrase in body:
            return True
    return False


def is_evidence_empty_fallback_headline(text: Any) -> bool:
    """True when headline equals a category default from lack of concrete evidence."""
    raw = str(text or "").strip().upper()
    return bool(raw) and raw in THUMBNAIL_CATEGORY_FALLBACK_HEADLINES


def clean_thumbnail_headline(text: Any, *, max_words: int = 5, max_chars: int = 34) -> str:
    raw = str(text or "").strip()
    raw = re.sub(r"https?://\S+", "", raw)
    raw = re.sub(r"#[\w-]+", "", raw)
    raw = re.sub(r"[\r\n\t]+", " ", raw)
    raw = re.sub(r"[^A-Za-z0-9 .,'&/+:-]+", " ", raw)
    words = [w.strip(" .,'&/+:-") for w in raw.split() if w.strip(" .,'&/+:-")]
    if not words:
        return ""
    return " ".join(words[:max_words]).upper()[:max_chars].strip()
