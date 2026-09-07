from __future__ import annotations

import re
from pathlib import Path
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


# Camera-roll / phone dump stems that must never become on-image thumbnail text.
_MEDIA_DUMP_STEM_RE = re.compile(
    r"^(?:"
    r"img[_-]?\d+"
    r"|vid[_-]?\d+"
    r"|dsc[_-]?\d+"
    r"|dscn?\d+"
    r"|mvi[_-]?\d+"
    r"|mov[_-]?\d+"
    r"|pxl[_-]?\d+"
    r"|photo[_-]?\d+"
    r"|video[_-]?\d+"
    r"|clip[_-]?\d+"
    r"|whatsapp[_ -]?(?:video|image)[_ -]?\d*"
    r"|screen[_-]?recording[_-]?\d*"
    r"|rpReplay_Final\d+"
    r"|gopr\d+"
    r"|gx\d+"
    r"|gh\d+"
    r")$",
    re.IGNORECASE,
)

# Internal pipeline labels that LLMs/Pikzels sometimes stamp onto the canvas.
_HYDRATION_META_PHRASES = frozenset(
    {
        "hydration story",
        "hydration_story",
        "fusion summary",
        "fusion_summary",
        "canonical geo",
        "canonical music",
        "canonical dashcam",
        "signal hashtags",
        "uploadm8 hydration",
        "no strong analysis",
        "use the actual frame",
        "filename only",
    }
)


def media_filename_stem(text: Any) -> str:
    """Basename stem without extension, lowercased alnum-normalized for dump checks."""
    raw = str(text or "").strip()
    if not raw:
        return ""
    name = Path(raw.replace("\\", "/")).name
    stem = re.sub(r"\.[A-Za-z0-9]{2,5}$", "", name).strip()
    return stem


def is_media_dump_filename(text: Any) -> bool:
    """True for phone/camera dump names (IMG_5135.MOV, VID_0001, etc.)."""
    stem = media_filename_stem(text)
    if not stem:
        return False
    compact = re.sub(r"[\s.]+", "", stem)
    if _MEDIA_DUMP_STEM_RE.match(stem) or _MEDIA_DUMP_STEM_RE.match(compact):
        return True
    # Bare "IMG 5135" / "IMG_5135" after clean_thumbnail_headline.
    body = thumbnail_headline_body(text)
    if _MEDIA_DUMP_STEM_RE.match(body.replace(" ", "")) or _MEDIA_DUMP_STEM_RE.match(
        body.replace(" ", "_")
    ):
        return True
    # Entire headline is only a dump stem (+ optional short extension token).
    words = body.split()
    if not words:
        return False
    joined = "".join(words)
    if _MEDIA_DUMP_STEM_RE.match(joined):
        return True
    if len(words) <= 3 and words[-1] in {"mov", "mp4", "m4v", "avi", "mkv", "hevc", "3gp"}:
        stemish = "".join(words[:-1])
        if _MEDIA_DUMP_STEM_RE.match(stemish):
            return True
    return False


def is_hydration_meta_headline(text: Any) -> bool:
    """True when headline carries internal hydration / brief meta labels."""
    body = thumbnail_headline_body(text)
    if not body:
        return False
    if body in _HYDRATION_META_PHRASES:
        return True
    for phrase in _HYDRATION_META_PHRASES:
        if phrase in body:
            return True
    # Leading "HYDRATION STORY …" after clean_thumbnail_headline.
    if body.startswith("hydration story"):
        return True
    if body.startswith("fusion summary"):
        return True
    return False


def is_empty_hydration_story_fallback(text: Any) -> bool:
    """True for the no-signal hydration_story string that must not feed image prompts."""
    raw = re.sub(r"\s+", " ", str(text or "").strip().lower())
    if not raw:
        return False
    if "no strong analysis signals" in raw:
        return True
    if "use the actual frame and filename" in raw:
        return True
    return False


# Any string that is literally a media path / extension — never paint or prompt.
_MEDIA_FILE_EXT_RE = re.compile(
    r"(?i)\.(?:mov|mp4|m4v|avi|mkv|hevc|3gp|webm|mts|m2ts|wmv|jpg|jpeg|png|heic)$"
)


def is_filename_like_thumbnail_text(text: Any, *, filename: str = "") -> bool:
    """True when text is (or equals) an upload/media filename — never for Pikzels.

    Hard invariant: camera dump names, bare stems matching the upload file, and
    any title that still carries a video/image extension must never become
    ``effective_title``, headline, badge, or Pikzels ``source_title``.
    """
    raw = str(text or "").strip()
    if not raw:
        return False
    if is_media_dump_filename(raw):
        return True
    # Explicit extension still present (holiday.mp4, clip.MOV).
    if _MEDIA_FILE_EXT_RE.search(raw.replace(" ", "")):
        return True
    # Matches the upload filename stem (with or without extension).
    fname = str(filename or "").strip()
    if fname:
        stem = media_filename_stem(fname)
        title_stem = media_filename_stem(raw) or re.sub(
            r"[^a-z0-9]+", "", raw.lower()
        )
        fname_key = re.sub(r"[^a-z0-9]+", "", stem.lower()) if stem else ""
        if fname_key and title_stem and fname_key == title_stem:
            return True
        # Exact basename match ignoring case/spaces.
        base = Path(fname.replace("\\", "/")).name
        if re.sub(r"\s+", "", raw.lower()) == re.sub(r"\s+", "", base.lower()):
            return True
    return False


def is_unusable_thumbnail_headline(text: Any, *, filename: str = "") -> bool:
    """Generic, category-fallback, filename, or hydration-meta — never paint on canvas."""
    if is_generic_thumbnail_headline(text):
        return True
    if is_evidence_empty_fallback_headline(text):
        return True
    if is_filename_like_thumbnail_text(text, filename=filename):
        return True
    if is_hydration_meta_headline(text):
        return True
    return False


def safe_thumbnail_prompt_title(
    text: Any,
    *,
    filename: str = "",
    fallback: str = "Video",
) -> str:
    """Title safe to put in a thumbnail brief / Pikzels prompt — never a filename."""
    raw = re.sub(r"\s+", " ", str(text or "").strip())
    if not raw or is_unusable_thumbnail_headline(raw, filename=filename):
        return fallback
    if is_filename_like_thumbnail_text(raw, filename=filename):
        return fallback
    return raw


def clean_thumbnail_headline(
    text: Any,
    *,
    max_words: int = 5,
    max_chars: int = 34,
    filename: str = "",
) -> str:
    raw = str(text or "").strip()
    raw = re.sub(r"https?://\S+", "", raw)
    raw = re.sub(r"#[\w-]+", "", raw)
    raw = re.sub(r"[\r\n\t]+", " ", raw)
    raw = re.sub(r"[^A-Za-z0-9 .,'&/+:-]+", " ", raw)
    words = [w.strip(" .,'&/+:-") for w in raw.split() if w.strip(" .,'&/+:-")]
    if not words:
        return ""
    cleaned = " ".join(words[:max_words]).upper()[:max_chars].strip()
    # Camera-roll stems / upload basenames must never become paintable cover text.
    if is_unusable_thumbnail_headline(cleaned, filename=filename) or is_unusable_thumbnail_headline(
        raw, filename=filename
    ):
        return ""
    return cleaned
