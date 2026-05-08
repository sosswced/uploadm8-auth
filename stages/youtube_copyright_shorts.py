"""
YouTube Shorts + catalogue audio (ACR / Content ID style).

When a clip is long enough to trip YouTube's rule for copyright in Shorts (≥60s),
we surface a warning on the upload record and optionally re-encode the **YouTube**
deliverable to the first ~60s so Shorts + recognized music can stay viable.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from .context import JobContext
from .transcode_stage import (
    get_video_info,
    resolve_reframe_action,
    transcode_video,
)

logger = logging.getLogger("uploadm8-worker.youtube_copyright_shorts")

# Align with stages.publish_stage.YOUTUBE_SHORTS_COPYRIGHT_MAX_SEC
COPYRIGHT_SHORTS_MAX_SEC = 60.0


def _source_duration_sec(ctx: JobContext) -> float:
    try:
        return float((ctx.video_info or {}).get("duration") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _acr_catalog_copyright_signal(ac: Dict[str, Any]) -> bool:
    if ac.get("copyright_risk"):
        return True
    cs = ac.get("content_signals") or []
    if not isinstance(cs, list):
        return False
    if "acr_catalog_match" not in {str(x).strip() for x in cs if x}:
        return False
    return bool(ac.get("music_detected") or ac.get("music_title") or ac.get("music_artist"))


def youtube_copyright_shorts_acr_risk(ctx: JobContext) -> bool:
    """ACR-style catalogue hit + long clip + YouTube is a target platform."""
    if "youtube" not in [str(p).strip().lower() for p in (ctx.platforms or []) if p]:
        return False
    dur = _source_duration_sec(ctx)
    if dur < COPYRIGHT_SHORTS_MAX_SEC:
        return False
    ac = getattr(ctx, "audio_context", None) or {}
    if not isinstance(ac, dict):
        return False
    return _acr_catalog_copyright_signal(ac)


def _trim_pref_enabled(ctx: JobContext) -> bool:
    us = ctx.user_settings or {}
    v = us.get("youtube_shorts_copyright_trim")
    if v is None:
        v = us.get("youtubeShortsCopyrightTrim")
    if isinstance(v, str):
        return v.lower() not in ("false", "0", "no", "off", "")
    return bool(v)


def get_youtube_copyright_notice(ctx: JobContext) -> Optional[Dict[str, Any]]:
    raw = (getattr(ctx, "output_artifacts", None) or {}).get("youtube_copyright_shorts")
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            d = json.loads(raw)
            return d if isinstance(d, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def youtube_copyright_shorts_trim_applied(ctx: JobContext) -> bool:
    n = get_youtube_copyright_notice(ctx)
    return bool(n and n.get("trim_applied"))


async def _retrim_youtube_deliverable(ctx: JobContext, db_pool) -> bool:
    pv = getattr(ctx, "platform_videos", None) or {}
    yp = pv.get("youtube")
    if not yp or not Path(yp).exists():
        logger.warning("[%s] YouTube copyright trim: no youtube platform video path", ctx.upload_id)
        return False
    info = await get_video_info(Path(yp))
    if info.duration <= COPYRIGHT_SHORTS_MAX_SEC:
        return False
    if not ctx.temp_dir:
        return False
    out = Path(ctx.temp_dir) / "transcoded_youtube_copyright_shorts_trim.mp4"
    reframe = resolve_reframe_action(info, getattr(ctx, "reframe_mode", "auto") or "auto", "youtube")
    await transcode_video(
        Path(yp),
        out,
        "youtube",
        info,
        reframe,
        db_pool=db_pool,
        upload_id=str(ctx.upload_id) if ctx.upload_id else None,
        force_duration_trim_sec=COPYRIGHT_SHORTS_MAX_SEC,
    )
    ctx.platform_videos["youtube"] = out
    new_info = await get_video_info(out)
    logger.info(
        "[%s] YouTube copyright trim applied: %.2fs -> %.2fs",
        ctx.upload_id,
        info.duration,
        new_info.duration,
    )
    return True


async def apply_youtube_copyright_shorts_after_audio(ctx: JobContext, db_pool) -> None:
    """
    Persist a user-visible notice on the upload row; optionally replace the YouTube
    deliverable with a ≤60s head when the user enabled ``youtubeShortsCopyrightTrim``.
    """
    if not youtube_copyright_shorts_acr_risk(ctx):
        return

    from stages.pipeline_checkpoint import merge_output_artifacts_patch

    trim_pref = _trim_pref_enabled(ctx)
    dur = _source_duration_sec(ctx)
    ac = getattr(ctx, "audio_context", None) or {}
    notice: Dict[str, Any] = {
        "level": "warning",
        "source": "acr_catalog",
        "duration_sec": dur,
        "youtube_shorts_policy_sec": int(COPYRIGHT_SHORTS_MAX_SEC),
        "trim_pref_enabled": trim_pref,
        "trim_applied": False,
        "music_title": str(ac.get("music_title") or "")[:200],
        "music_artist": str(ac.get("music_artist") or "")[:200],
    }
    if trim_pref:
        notice["message"] = (
            "UploadM8 detected recognized music (ACR) that may be copyright-protected. "
            f"YouTube often blocks Shorts **{int(COPYRIGHT_SHORTS_MAX_SEC)} seconds or longer** when that audio is claimed. "
            "Because you enabled **Trim YouTube to 60s**, we are using only the first minute for your "
            "**YouTube** upload so Shorts + this track can stay within that policy. Other platforms are unchanged."
        )
    else:
        notice["message"] = (
            "UploadM8 detected recognized music (ACR) that may be copyright-protected. "
            f"YouTube may block or restrict Shorts **{int(COPYRIGHT_SHORTS_MAX_SEC)} seconds or longer** with this audio. "
            "We will publish your **YouTube** copy as a standard watch-page video (no #shorts) unless you enable "
            "**Trim YouTube to 60s** under Settings → Audio Intelligence. "
            "You can also use royalty-free audio, shorten the clip yourself, or resolve claims in YouTube Studio."
        )

    ctx.output_artifacts["youtube_copyright_shorts"] = json.dumps(notice)
    try:
        await merge_output_artifacts_patch(db_pool, str(ctx.upload_id), {"youtube_copyright_shorts": notice})
    except Exception as e:
        logger.warning("[%s] youtube_copyright_shorts notice merge failed: %s", ctx.upload_id, e)

    if trim_pref:
        try:
            if await _retrim_youtube_deliverable(ctx, db_pool):
                notice["trim_applied"] = True
                yp = (getattr(ctx, "platform_videos", None) or {}).get("youtube")
                if yp and Path(yp).exists():
                    notice["youtube_output_duration_sec"] = (await get_video_info(Path(yp))).duration
                notice["message"] = (
                    "UploadM8 detected recognized music (ACR) and **trimmed the YouTube file to the first ~60 seconds** "
                    "because you enabled **Trim YouTube to 60s** — this keeps Shorts-style delivery safer when a catalogue "
                    "match is present. TikTok / Instagram / Facebook files were not shortened."
                )
                ctx.output_artifacts["youtube_copyright_shorts"] = json.dumps(notice)
                await merge_output_artifacts_patch(db_pool, str(ctx.upload_id), {"youtube_copyright_shorts": notice})
        except Exception as e:
            logger.warning("[%s] YouTube copyright trim failed (continuing with full-length YouTube file): %s", ctx.upload_id, e)
