"""
Shared thumbnail brief shaping for the worker pipeline and on-demand regenerate.

Keeps ``_apply_thumbnail_default_strategy`` → ``apply_hydration_payload_to_thumbnail_brief``
→ optional evidence-anchor fallbacks in one place so queue/regenerate paths do not drift.

YouTube watch links must not appear inside Pikzels ``prompt`` (public API: prompt cannot
contain URLs). When a published YouTube URL is known, we derive ``i.ytimg.com`` HTTPS
still URLs and pass them as ``support_image_url`` on ``/v2/thumbnail/image`` (pkz_4+).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Iterable, List, Optional

from stages.context import JobContext
from services.hydration_payload import apply_hydration_payload_to_thumbnail_brief

logger = logging.getLogger(__name__)

# Stripped before persisting ``thumbnail_brief_json`` / merged output_artifacts.
INTERNAL_BRIEF_KEYS: frozenset[str] = frozenset({
    "_uploadm8_pikzels_support_image_url",
    "_uploadm8_dashcam_pov",
})

_YT_WATCH_RE = re.compile(
    r"(?:youtube\.com/watch\?v=|youtu\.be/)([0-9A-Za-z_-]{11})",
    re.IGNORECASE,
)


def copy_brief_for_persistence(brief: Dict[str, Any]) -> Dict[str, Any]:
    """Return a JSON-safe brief without internal renderer-only keys."""
    out = dict(brief or {})
    for k in INTERNAL_BRIEF_KEYS:
        out.pop(k, None)
    return out


def _iter_platform_rows(platform_results: Any) -> Iterable[Dict[str, Any]]:
    pr = platform_results
    if pr is None:
        return
    if isinstance(pr, str) and pr.strip():
        try:
            pr = json.loads(pr)
        except Exception:
            return
    if not isinstance(pr, list):
        return
    for item in pr:
        if isinstance(item, dict):
            yield item
        elif hasattr(item, "platform"):
            yield {
                "platform": getattr(item, "platform", None),
                "platform_url": getattr(item, "platform_url", None),
                "url": getattr(item, "platform_url", None),
            }


def youtube_watch_url_from_platform_results(platform_results: Any) -> str:
    """First YouTube watch / youtu.be URL from publish results (page URL, not img CDN)."""
    for item in _iter_platform_rows(platform_results):
        if str(item.get("platform") or "").lower() != "youtube":
            continue
        for k in ("platform_url", "url", "video_url", "share_url", "permalink"):
            u = str(item.get(k) or "").strip()
            if "youtube.com" in u.lower() or "youtu.be" in u.lower():
                return u[:500]
    return ""


def youtube_reference_still_url_from_watch_url(watch_url: str) -> str:
    """
    Map a YouTube page URL to an HTTPS still on Google's CDN (valid for Pikzels
    ``support_image_url``). Empty if no video id found.
    """
    u = (watch_url or "").strip()
    if not u:
        return ""
    m = _YT_WATCH_RE.search(u)
    if not m:
        return ""
    vid = m.group(1)
    return f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg"


def pikzels_support_image_url_from_platform_results(platform_results: Any) -> str:
    """HTTPS reference still for Pikzels v2 ``support_image_url``, or empty."""
    w = youtube_watch_url_from_platform_results(platform_results)
    return youtube_reference_still_url_from_watch_url(w)


def merge_story_voice_youtube_into_brief(
    brief: Dict[str, Any],
    *,
    arts: Dict[str, Any],
    settings: Dict[str, Any],
    platform_results: Any,
) -> Dict[str, Any]:
    """
    Fold UI-facing narrative + prefs into the brief.

    YouTube page URLs are never appended to ``notes`` (would violate Pikzels prompt rules
    and often exceed the short-notes path). When resolvable, sets
    ``_uploadm8_pikzels_support_image_url`` for the image renderer.
    """
    b = dict(brief or {})
    extras: List[str] = []

    voice = str(
        settings.get("captionVoice")
        or settings.get("caption_voice")
        or settings.get("captionTone")
        or settings.get("caption_tone")
        or ""
    ).strip()
    if voice:
        extras.append(f"Caption voice / tone (user setting): {voice[:120]}")

    scene = arts.get("scene_story")
    if isinstance(scene, str) and scene.strip():
        ss = scene.strip()
        if len(ss) > 900:
            ss = ss[:899] + "…"
        if not str(b.get("hydration_story") or "").strip():
            b["hydration_story"] = ss[:700]
        extras.append(f"Scene story (UploadM8 UI narrative): {ss[:520]}")

    ts = arts.get("timeline_story")
    if isinstance(ts, str) and ts.strip():
        try:
            ts = json.loads(ts)
        except Exception:
            ts = None
    if isinstance(ts, list) and ts:
        bits: List[str] = []
        for ev in ts[:14]:
            if isinstance(ev, dict):
                lbl = str(ev.get("label") or ev.get("type") or ev.get("title") or "").strip()
                t0 = ev.get("t_seconds", ev.get("start", ev.get("at")))
                if lbl or t0 is not None:
                    bits.append(f"{t0}:{lbl[:56]}" if t0 is not None else lbl[:72])
            elif isinstance(ev, str) and ev.strip():
                bits.append(ev.strip()[:80])
        if bits:
            extras.append("Timeline story: " + "; ".join(bits)[:480])

    sup = pikzels_support_image_url_from_platform_results(platform_results)
    if sup:
        b["_uploadm8_pikzels_support_image_url"] = sup
        extras.append(
            "A separate reference image shows your channel's typical YouTube cover still "
            "(layout and energy only; do not copy third-party logos, watermarks, or typography)."
        )

    if extras:
        piece = " ".join(extras)
        cur = str(b.get("notes") or "").strip()
        merged = (cur + " | " + piece).strip() if cur else piece
        b["notes"] = merged[:900]
    return b


def minimal_thumbnail_brief(*, title: str) -> Dict[str, Any]:
    from core.thumbnail_text import clean_thumbnail_headline, is_generic_thumbnail_headline

    headline = (
        clean_thumbnail_headline((title or "").strip(), max_words=5, max_chars=24) or "VIDEO HIGHLIGHT"
    )
    if is_generic_thumbnail_headline(headline):
        headline = "VIDEO HIGHLIGHT"
    return {
        "selected_headline": headline,
        "headline_options": [],
        "badge_text": "",
        "badge_style": "red",
        "directional_element": "circle",
        "props": [],
        "emotion_cue": "excited",
        "color_mood": "red_black",
        "platform_plan": {
            "youtube": {"enabled": True, "canvas": "16:9"},
            "instagram": {"enabled": True, "canvas": "9:16", "safe_center_pct": 60},
            "facebook": {"enabled": True, "canvas": "9:16", "safe_center_pct": 60},
            "tiktok": {"enabled": True, "canvas": "9:16", "thumb_offset_seconds": 1.5},
        },
        "notes": "Regenerated thumbnail brief (minimal seed)",
    }


def finalize_styled_thumbnail_brief(
    ctx: JobContext,
    brief: Dict[str, Any],
    category: str,
    user_settings: Dict[str, Any],
    *,
    studio_render_report: Optional[Dict[str, Any]] = None,
    evidence_anchor_fallbacks: bool = False,
) -> Dict[str, Any]:
    """
    Apply default strategy, hydration merge, and optional evidence fallbacks (worker only).

    ``brief`` must already have been through ``_sanitize_thumbnail_brief`` at the call site.
    """
    from stages import thumbnail_stage as ts

    out = dict(brief or {})
    effective_category = ts.effective_thumbnail_category(user_settings, category)
    out = ts._apply_thumbnail_default_strategy(out, user_settings, category=effective_category)

    try:
        from services.hydration_enforcer import (
            build_anchor_phrase,
            build_evidence_hashtags,
            collect_evidence,
        )

        out = apply_hydration_payload_to_thumbnail_brief(ctx, out)
        hp_ctx = getattr(ctx, "hydration_payload", None)
        if isinstance(hp_ctx, dict) and str(hp_ctx.get("anchor_phrase") or "").strip():
            if studio_render_report is not None:
                studio_render_report["evidence_anchor"] = str(hp_ctx["anchor_phrase"])[:500]
        elif evidence_anchor_fallbacks:
            pool = collect_evidence(ctx)
            anchor = build_anchor_phrase(pool, ctx)
            if studio_render_report is not None:
                studio_render_report["evidence_anchor"] = anchor or None
            if anchor:
                _ph_notes = frozenset({"No AI — evidence-based brief", "Fallback brief"})
                _cur_notes = str(out.get("notes") or "").strip()
                if not _cur_notes or _cur_notes in _ph_notes:
                    out["notes"] = anchor[:200]
                if not str(out.get("speech_context") or "").strip() and pool.transcript_phrase:
                    out["speech_context"] = pool.transcript_phrase[:260]
                if not str(out.get("music_context") or "").strip() and (pool.music_artist or pool.music_title):
                    parts = [p for p in (pool.music_artist, pool.music_title) if p]
                    out["music_context"] = " — ".join(parts)[:220]
                if not str(out.get("geo_context") or "").strip():
                    geo_bits_fb: List[str] = []
                    if pool.road:
                        geo_bits_fb.append(pool.road)
                    if pool.gazetteer_place and pool.state_abbr:
                        geo_bits_fb.append(f"{pool.gazetteer_place}, {pool.state_abbr}")
                    elif pool.city and pool.state_abbr:
                        geo_bits_fb.append(f"{pool.city}, {pool.state_abbr}")
                    elif pool.protected_area:
                        geo_bits_fb.append(pool.protected_area)
                    if geo_bits_fb:
                        out["geo_context"] = "; ".join(geo_bits_fb)[:260]
                if not str(out.get("osd_context") or "").strip() and (pool.max_speed_mph or pool.driver_name):
                    osd_bits_fb: List[str] = []
                    if pool.max_speed_mph:
                        osd_bits_fb.append(f"max {int(round(pool.max_speed_mph))} MPH")
                    if pool.driver_name:
                        osd_bits_fb.append(f"driver {pool.driver_name}")
                    out["osd_context"] = "; ".join(osd_bits_fb)[:220]
                if not str(out.get("trill_context") or "").strip() and pool.trill_bucket:
                    out["trill_context"] = (
                        f"Trill bucket {pool.trill_bucket} (score {pool.trill_score:.0f})"
                    )[:220]
                evidence_tags = build_evidence_hashtags(pool, max_extra=8)
                if evidence_tags and not str(out.get("signal_hashtags") or "").strip():
                    out["signal_hashtags"] = ", ".join(evidence_tags[:8])[:180]
    except Exception as e:
        logger.debug("[thumb-brief-pipeline] hydration brief enrichment failed: %s", e)

    return out


def attach_youtube_support_image_from_ctx(
    brief: Dict[str, Any],
    ctx: JobContext,
    *,
    platform_results_override: Any = None,
    user_settings: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    If the brief has no support URL yet, resolve in order:

    1. Existing brief field
    2. Saved default YouTube reference (user preferences)
    3. Upload default strategy ``reference_youtube_url`` / ``youtube_url``
    4. Published YouTube URL on this upload (``platform_results``)

    Skipped when apply mode is ``strategy_only`` / ``pinned_cover``, or when
    ref/persona mode is ``face_brand`` (persona XOR YouTube support).
    """
    b = dict(brief or {})
    if str(b.get("_uploadm8_pikzels_support_image_url") or "").strip().startswith("https://"):
        return b

    us = user_settings if user_settings is not None else getattr(ctx, "user_settings", None)
    if isinstance(us, dict):
        from services.thumbnail_apply_mode import allow_youtube_support_image, resolve_ref_persona_mode

        if not allow_youtube_support_image(us):
            b["_uploadm8_support_image_skipped"] = resolve_ref_persona_mode(us)
            return b
        from services.thumbnail_youtube_refs import (
            support_image_url_from_prefs,
        )
        from stages.thumbnail_stage import _thumbnail_default_strategy

        sup_pref = support_image_url_from_prefs(us)
        if sup_pref:
            b["_uploadm8_pikzels_support_image_url"] = sup_pref
            return b
        strategy = _thumbnail_default_strategy(us)
        from services.thumbnail_youtube_refs import support_image_url_from_strategy

        sup_strat = support_image_url_from_strategy(strategy)
        if sup_strat:
            b["_uploadm8_pikzels_support_image_url"] = sup_strat
            return b

    src = platform_results_override
    if src is None:
        src = getattr(ctx, "platform_results", None)
    sup = pikzels_support_image_url_from_platform_results(src)
    if sup:
        b["_uploadm8_pikzels_support_image_url"] = sup
    return b
