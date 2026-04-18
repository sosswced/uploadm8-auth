"""
Per-upload content generation attribution for ML rollups (settings → engagement).

Snapshots are stored on ``uploads.output_artifacts`` as JSON strings keyed by
``content_attribution_v1`` / ``content_attribution_key`` so ``upload_quality_scores_daily``
can group outcomes by the exact packaging the worker used.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

_SLUG_SAFE = re.compile(r"[^a-z0-9_.-]+", re.I)

# Omitted from strategy_key hash so tag sets do not explode daily rollup buckets.
_ATTRIBUTION_HASH_EXCLUDE = frozenset({"captured_at", "hashtag_slugs_used"})


def collect_hashtag_slugs_for_attribution(ctx: Any) -> List[str]:
    """
    Sanitized hashtag bodies (no #) unioned across publish targets — follows
    ``JobContext.get_effective_hashtags`` per platform (always → platform →
    upload → M8 → AI).
    """
    from core.helpers import sanitize_hashtag_body

    plats = [str(p).strip().lower() for p in (getattr(ctx, "platforms", None) or []) if str(p).strip()]
    if not plats:
        plats = ["tiktok"]
    seen: set = set()
    out: List[str] = []
    fn = getattr(ctx, "get_effective_hashtags", None)
    if not callable(fn):
        return []
    for pl in plats:
        try:
            merged = fn(platform=pl)
        except Exception:
            merged = []
        for h in merged or []:
            b = sanitize_hashtag_body(str(h).lstrip("#"))
            if not b or b in seen:
                continue
            seen.add(b)
            out.append(b)
    out.sort()
    return out[:50]


def normalize_thumbnail_selection_mode(user_settings: Optional[Dict[str, Any]] = None) -> str:
    """User intent for frame pick: AI compares candidates vs sharpest-only."""
    us = user_settings or {}
    v = str(
        us.get("thumbnail_selection_mode") or us.get("thumbnailSelectionMode") or "ai"
    ).lower().strip()
    return v if v in ("ai", "sharpness") else "ai"


def normalize_thumbnail_render_pipeline(user_settings: Optional[Dict[str, Any]] = None) -> str:
    """Order/focus for styled thumbnail compositing (MrBeast-style overlays)."""
    us = user_settings or {}
    v = str(
        us.get("thumbnail_render_pipeline") or us.get("thumbnailRenderPipeline") or "auto"
    ).lower().strip()
    allowed = frozenset(("auto", "studio_renderer", "ai_edit", "template", "none"))
    return v if v in allowed else "auto"


def _slug_segment(val: Any, max_len: int = 48) -> str:
    s = _SLUG_SAFE.sub("_", str(val or "").strip().lower())
    return s[:max_len] if s else "na"


def build_content_attribution_snapshot(
    *,
    user_settings: Dict[str, Any],
    strategy: Optional[Dict[str, Any]],
    category: str,
    used_m8_engine: bool,
    caption_style_ui: str,
    caption_tone_ui: str,
    caption_voice_ui: str,
    hashtag_style: str,
    hashtag_count: int,
    caption_frame_count: int,
    generate_hashtags: bool,
    output_artifacts: Dict[str, str],
    hashtag_slugs_used: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Canonical dict merged into output_artifacts (stringified JSON)."""
    outs = (strategy or {}).get("outputs") or {}
    tags = [str(t).strip().lower().lstrip("#") for t in (hashtag_slugs_used or []) if str(t).strip()]
    tags = [t for t in tags if 0 < len(t) < 60][:50]
    snap: Dict[str, Any] = {
        "v": 2,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "content_category": (category or "general").lower(),
        "caption_style": (caption_style_ui or "story").lower(),
        "caption_tone": (caption_tone_ui or "authentic").lower(),
        "caption_voice": (caption_voice_ui or "default").lower(),
        "m8_engine": bool(used_m8_engine),
        "effective_style": str(outs.get("caption_style") or ""),
        "effective_tone": str(outs.get("tone") or ""),
        "effective_persona": str(outs.get("voice_persona") or outs.get("persona") or ""),
        "hashtag_style": (hashtag_style or "mixed").lower(),
        "ai_hashtag_count": int(hashtag_count or 0),
        "ai_hashtags_enabled": bool(generate_hashtags),
        "caption_frame_count": int(caption_frame_count or 6),
        "styled_thumbnails": bool(
            user_settings.get("styledThumbnails")
            if user_settings.get("styledThumbnails") is not None
            else user_settings.get("styled_thumbnails", True)
        ),
        "auto_thumbnails": bool(
            user_settings.get("autoThumbnails")
            if user_settings.get("autoThumbnails") is not None
            else user_settings.get("auto_thumbnails", False)
        ),
        "thumbnail_selection_method": str(output_artifacts.get("thumbnail_selection_method") or ""),
        "thumbnail_render_method": str(output_artifacts.get("thumbnail_render_method") or ""),
        "thumbnail_category": str(output_artifacts.get("thumbnail_category") or ""),
        "thumbnail_selection_mode": normalize_thumbnail_selection_mode(user_settings),
        "thumbnail_render_pipeline": normalize_thumbnail_render_pipeline(user_settings),
        "hashtag_slugs_used": tags,
    }
    return snap


def content_attribution_strategy_key(snap: Dict[str, Any]) -> str:
    """
    Stable rollup key: human-readable segments + short hash for disambiguation.
    Keep length within 480 chars for varchar columns.
    """
    payload = {k: v for k, v in sorted(snap.items()) if k not in _ATTRIBUTION_HASH_EXCLUDE}
    sig = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]
    parts: List[str] = [
        "v1",
        f"cs={_slug_segment(snap.get('caption_style'))}",
        f"ct={_slug_segment(snap.get('caption_tone'))}",
        f"cv={_slug_segment(snap.get('caption_voice'))}",
        f"m8={'1' if snap.get('m8_engine') else '0'}",
        f"es={_slug_segment(snap.get('effective_style'))}",
        f"et={_slug_segment(snap.get('effective_tone'))}",
        f"ep={_slug_segment(snap.get('effective_persona'))}",
        f"tsm={_slug_segment(snap.get('thumbnail_selection_mode'))}",
        f"trp={_slug_segment(snap.get('thumbnail_render_pipeline'))}",
        f"tsel={_slug_segment(snap.get('thumbnail_selection_method'))}",
        f"trend={_slug_segment(snap.get('thumbnail_render_method'))}",
        f"sty={'1' if snap.get('styled_thumbnails') else '0'}",
        f"ah={'1' if snap.get('ai_hashtags_enabled') else '0'}",
        f"hc={int(snap.get('ai_hashtag_count') or 0)}",
        f"hs={_slug_segment(snap.get('hashtag_style'))}",
        f"nfc={int(snap.get('caption_frame_count') or 0)}",
        f"sig={sig}",
    ]
    key = "|".join(parts)
    return key[:480]


def parse_content_attribution_key(strategy_key: str) -> Dict[str, Any]:
    """Parse a key produced by ``content_attribution_strategy_key`` (for apply-optimized)."""
    out: Dict[str, Any] = {}
    if not strategy_key or not strategy_key.startswith("v1|"):
        return out
    for part in strategy_key.split("|")[1:]:
        if "=" not in part:
            continue
        k, _, v = part.partition("=")
        k = k.strip()
        v = v.strip()
        if k == "cs":
            out["caption_style"] = v
        elif k == "ct":
            out["caption_tone"] = v
        elif k == "cv":
            out["caption_voice"] = v
        elif k == "m8":
            out["m8_engine"] = v == "1"
        elif k == "sty":
            out["styled_thumbnails"] = v == "1"
        elif k == "ah":
            out["ai_hashtags_enabled"] = v == "1"
        elif k == "hc":
            try:
                out["ai_hashtag_count"] = int(v)
            except ValueError:
                pass
        elif k == "hs":
            out["ai_hashtag_style"] = v
        elif k == "nfc":
            try:
                out["caption_frame_count"] = int(v)
            except ValueError:
                pass
        elif k == "tsm" and v in ("ai", "sharpness"):
            out["thumbnail_selection_mode"] = v
        elif k == "trp":
            allowed = ("auto", "studio_renderer", "ai_edit", "template", "none")
            if v in allowed:
                out["thumbnail_render_pipeline"] = v
    return out


def preferences_patch_from_parsed_attribution(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """CamelCase patch suitable for ``save_user_content_preferences`` / users.preferences."""
    patch: Dict[str, Any] = {}
    if "caption_style" in parsed:
        patch["captionStyle"] = parsed["caption_style"]
    if "caption_tone" in parsed:
        patch["captionTone"] = parsed["caption_tone"]
    if "caption_voice" in parsed:
        patch["captionVoice"] = parsed["caption_voice"]
    if "styled_thumbnails" in parsed:
        patch["styledThumbnails"] = bool(parsed["styled_thumbnails"])
    if "ai_hashtags_enabled" in parsed:
        patch["aiHashtagsEnabled"] = bool(parsed["ai_hashtags_enabled"])
    if "ai_hashtag_count" in parsed:
        patch["aiHashtagCount"] = int(parsed["ai_hashtag_count"])
    if "ai_hashtag_style" in parsed:
        patch["aiHashtagStyle"] = parsed["ai_hashtag_style"]
    if "caption_frame_count" in parsed:
        patch["captionFrameCount"] = int(parsed["caption_frame_count"])
    if "thumbnail_selection_mode" in parsed:
        patch["thumbnailSelectionMode"] = str(parsed["thumbnail_selection_mode"])
    if "thumbnail_render_pipeline" in parsed:
        patch["thumbnailRenderPipeline"] = str(parsed["thumbnail_render_pipeline"])
    return patch
