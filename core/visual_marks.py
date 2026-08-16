"""Open-vocab visual / spoken marks for downtown sports and landmarks.

Harvest whatever name Google or speech actually said. Do not grow a 200-team
enum — unknown marks still win when Vision/OCR/web named them.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Set

_NAME_LIKE_RE = re.compile(
    r"\b([A-Z][A-Za-z]{2,}(?:\s+[A-Z][A-Za-z&.]{2,}){0,3})\b"
)
_GENERIC_NAME_SLUGS = frozenset(
    {
        "the", "and", "with", "from", "this", "that", "photo", "tickets",
        "ticket", "party", "package", "video", "clip", "scene", "view",
        "outdoor", "indoor", "person", "people", "music", "game", "day",
        "best", "guess", "image", "stock",
    }
)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", _norm(text).lower())


def _iter_str_items(value: Any) -> List[str]:
    out: List[str] = []
    if isinstance(value, str) and value.strip():
        out.append(value.strip())
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
            elif isinstance(item, dict):
                for key in ("description", "name", "text", "label", "entity", "logo"):
                    v = item.get(key)
                    if isinstance(v, str) and v.strip():
                        out.append(v.strip())
                        break
    return out


def _name_like_tokens(blob: str) -> List[str]:
    found: List[str] = []
    seen: Set[str] = set()
    for m in _NAME_LIKE_RE.finditer(str(blob or "")):
        tok = _norm(m.group(1))
        slug = _slug(tok)
        if len(slug) < 3 or slug in _GENERIC_NAME_SLUGS or slug in seen:
            continue
        seen.add(slug)
        found.append(tok)
    return found


def _team_token_set() -> Set[str]:
    try:
        from services.place_evidence import _TEAM_TOKENS

        return {str(t).strip().lower() for t in _TEAM_TOKENS if str(t).strip()}
    except Exception:
        return set()


def _add_mark(
    out: List[Dict[str, Any]],
    seen: Set[str],
    text: str,
    source: str,
    score: float,
) -> None:
    cleaned = _norm(text)
    if not cleaned or len(cleaned) < 2:
        return
    key = _slug(cleaned)
    if not key or key in seen or key in _GENERIC_NAME_SLUGS:
        return
    seen.add(key)
    out.append({"text": cleaned[:80], "source": source, "score": float(score)})


def collect_visual_marks(ctx: Any) -> List[Dict[str, Any]]:
    """Ordered open-vocab marks: logos → OCR → web → landmarks → confirmed speech."""
    out: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    vc = getattr(ctx, "vision_context", None) or {}
    if not isinstance(vc, dict):
        vc = {}
    vi = (
        getattr(ctx, "video_intelligence_context", None)
        or getattr(ctx, "video_intelligence", None)
        or {}
    )
    if not isinstance(vi, dict):
        vi = {}

    for item in _iter_str_items(vc.get("logo_names") or vc.get("logos")):
        _add_mark(out, seen, item, "vision_logo", 1.0)
    for item in _iter_str_items(vi.get("logos") or vi.get("logo_names")):
        _add_mark(out, seen, item, "vi_logo", 0.95)

    ocr_blob = " ".join(
        [
            str(vc.get("ocr_text") or ""),
            " ".join(_iter_str_items(vi.get("on_screen_text") or vi.get("ocr_text"))),
        ]
    )
    for tok in _name_like_tokens(ocr_blob)[:12]:
        _add_mark(out, seen, tok, "ocr", 0.85)

    web_items = list(_iter_str_items(vc.get("web_entities"))) + list(
        _iter_str_items(vc.get("web_best_guess"))
    )
    for item in web_items[:12]:
        _add_mark(out, seen, item, "web", 0.8)

    for item in _iter_str_items(vc.get("landmark_names") or vc.get("landmarks")):
        _add_mark(out, seen, item, "landmark", 0.9)

    confirm_slugs = {_slug(x) for x in list(_iter_str_items(ocr_blob.split())) + web_items}
    confirm_slugs |= {_slug(m["text"]) for m in out}
    team_tokens = _team_token_set()

    speech_chunks: List[str] = []
    ac = getattr(ctx, "audio_context", None) or {}
    if isinstance(ac, dict):
        speech_chunks.append(str(ac.get("transcript") or ""))
        structured = ac.get("transcript_structured") or {}
        if isinstance(structured, dict):
            for topic in structured.get("topics") or []:
                speech_chunks.append(str(topic or ""))
            ne = structured.get("named_entities") or {}
            if isinstance(ne, dict):
                for key in ("organizations", "places"):
                    speech_chunks.extend(str(x) for x in (ne.get(key) or []) if x)
    speech_chunks.append(str(getattr(ctx, "ai_transcript", None) or ""))
    speech_blob = " ".join(speech_chunks)
    for tok in _name_like_tokens(speech_blob) + _speech_team_hits(speech_blob, team_tokens):
        slug = _slug(tok)
        low = tok.lower()
        overlaps = slug in confirm_slugs or any(
            slug and slug in _slug(c) for c in confirm_slugs
        )
        team_hit = any(t in low for t in team_tokens if len(t) >= 4) or low in team_tokens
        if overlaps or team_hit:
            _add_mark(out, seen, tok, "speech", 0.7)

    return out


def _speech_team_hits(blob: str, team_tokens: Iterable[str]) -> List[str]:
    low = str(blob or "").lower()
    hits: List[str] = []
    for team in team_tokens:
        t = str(team or "").strip().lower()
        if len(t) < 4:
            continue
        if t in low:
            hits.append(t.title() if " " not in t else " ".join(w.capitalize() for w in t.split()))
    return hits


def has_pixel_visual_marks(ctx: Any) -> bool:
    """True when Google/OCR/web/landmark named something (speech-only does not count)."""
    for mark in collect_visual_marks(ctx):
        if str(mark.get("source") or "") not in {"speech", "transcript"}:
            return True
    return False


__all__ = [
    "collect_visual_marks",
    "has_pixel_visual_marks",
]
