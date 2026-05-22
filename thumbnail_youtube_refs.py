"""Saved YouTube style references for Thumbnail Studio and upload-time Pikzels."""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from services.thumbnail_brief_pipeline import youtube_reference_still_url_from_watch_url
from services.thumbnail_studio import extract_youtube_video_id

_MAX_REFS = 12
_PREF_LIST_KEYS = (
    "thumbnailYoutubeReferences",
    "thumbnail_youtube_references",
)
_PREF_DEFAULT_URL_KEYS = (
    "thumbnailDefaultYoutubeUrl",
    "thumbnail_default_youtube_url",
)


def _prefs_list(prefs: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(prefs, dict):
        return []
    raw = None
    for k in _PREF_LIST_KEYS:
        if k in prefs and prefs[k] is not None:
            raw = prefs[k]
            break
    if not isinstance(raw, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in raw[:_MAX_REFS]:
        if not isinstance(item, dict):
            continue
        norm = normalize_reference_entry(item)
        if norm:
            out.append(norm)
    return out


def normalize_reference_entry(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    url = str(
        entry.get("youtube_url")
        or entry.get("youtubeUrl")
        or entry.get("url")
        or ""
    ).strip()
    if not url:
        return None
    vid = str(entry.get("youtube_video_id") or entry.get("youtubeVideoId") or "").strip()
    if not vid:
        vid = extract_youtube_video_id(url) or ""
    if not vid:
        return None
    label = str(entry.get("label") or entry.get("name") or "").strip()
    if not label:
        label = f"YouTube {vid[:8]}"
    ref_id = str(entry.get("id") or "").strip() or str(uuid.uuid4())
    return {
        "id": ref_id[:64],
        "label": label[:120],
        "youtube_url": url[:500],
        "youtube_video_id": vid[:32],
        "is_default": bool(entry.get("is_default") or entry.get("isDefault")),
    }


def list_youtube_references(prefs: Dict[str, Any]) -> List[Dict[str, Any]]:
    refs = _prefs_list(prefs)
    if not refs and isinstance(prefs, dict):
        for k in _PREF_DEFAULT_URL_KEYS:
            u = str(prefs.get(k) or "").strip()
            if u:
                vid = extract_youtube_video_id(u)
                if vid:
                    refs = [
                        {
                            "id": "legacy-default",
                            "label": "Default reference",
                            "youtube_url": u,
                            "youtube_video_id": vid,
                            "is_default": True,
                        }
                    ]
                break
    # Ensure single default
    seen_default = False
    for r in refs:
        if r.get("is_default"):
            if seen_default:
                r["is_default"] = False
            else:
                seen_default = True
    if refs and not seen_default:
        refs[0]["is_default"] = True
    return refs[:_MAX_REFS]


def default_youtube_url(prefs: Dict[str, Any]) -> str:
    for r in _prefs_list(prefs):
        if r.get("is_default"):
            return str(r.get("youtube_url") or "")
    if not isinstance(prefs, dict):
        return ""
    for k in _PREF_DEFAULT_URL_KEYS:
        u = str(prefs.get(k) or "").strip()
        if u:
            return u[:500]
    return ""


def default_youtube_video_id(prefs: Dict[str, Any]) -> str:
    for r in list_youtube_references(prefs):
        if r.get("is_default"):
            return str(r.get("youtube_video_id") or "")
    u = default_youtube_url(prefs)
    return extract_youtube_video_id(u) or ""


def support_image_url_from_prefs(prefs: Dict[str, Any]) -> str:
    u = default_youtube_url(prefs)
    return youtube_reference_still_url_from_watch_url(u)


def support_image_url_from_strategy(strategy: Dict[str, Any]) -> str:
    if not isinstance(strategy, dict):
        return ""
    u = str(strategy.get("reference_youtube_url") or strategy.get("youtube_url") or "").strip()
    if u:
        return youtube_reference_still_url_from_watch_url(u)
    vid = str(strategy.get("reference_youtube_video_id") or strategy.get("youtube_video_id") or "").strip()
    if vid:
        return youtube_reference_still_url_from_watch_url(f"https://www.youtube.com/watch?v={vid}")
    return ""


def merge_references_into_prefs(
    prefs: Dict[str, Any],
    references: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Return updated preferences dict with normalized reference list + default URL."""
    out = dict(prefs or {})
    normed: List[Dict[str, Any]] = []
    for item in references[:_MAX_REFS]:
        if not isinstance(item, dict):
            continue
        n = normalize_reference_entry(item)
        if n:
            normed.append(n)
    if normed and not any(r.get("is_default") for r in normed):
        normed[0]["is_default"] = True
    default_u = ""
    for r in normed:
        if r.get("is_default"):
            default_u = str(r.get("youtube_url") or "")
            break
    out["thumbnailYoutubeReferences"] = normed
    out["thumbnail_youtube_references"] = normed
    out["thumbnailDefaultYoutubeUrl"] = default_u
    out["thumbnail_default_youtube_url"] = default_u
    return out
