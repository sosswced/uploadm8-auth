"""Burn styled thumbnail pixels into TikTok MP4 at the cover frame timestamp."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Optional

from .ffmpeg_env import resolve_ffmpeg_executable

logger = logging.getLogger("uploadm8-worker.tiktok_cover_burn")

_BURN_FLASH_SEC = float(os.environ.get("TIKTOK_COVER_BURN_FLASH_SEC", "0.12") or "0.12")
_DEFAULT_W = 1080
_DEFAULT_H = 1920


def tiktok_burn_enabled(user_settings: Optional[dict]) -> bool:
    """User/env gate for burning styled covers into TikTok video."""
    if os.environ.get("TIKTOK_BURN_STYLED_COVER", "true").lower() in ("0", "false", "no", "off"):
        return False
    us = user_settings if isinstance(user_settings, dict) else {}
    for key in ("tiktokBurnStyledCover", "tiktok_burn_styled_cover"):
        if key in us:
            return bool(us[key])
    return True


def resolve_tiktok_cover_offset_sec(output_artifacts: Optional[dict]) -> float:
    arts = output_artifacts if isinstance(output_artifacts, dict) else {}
    for key in ("thumbnail_frame_offset_seconds", "tiktok_thumb_offset_seconds"):
        raw = arts.get(key)
        if raw is None or raw == "":
            continue
        try:
            return max(0.0, float(raw))
        except (TypeError, ValueError):
            continue
    return 1.5


def build_tiktok_cover_burn_command(
    video_path: Path,
    thumb_path: Path,
    output_path: Path,
    offset_sec: float,
    *,
    width: int = _DEFAULT_W,
    height: int = _DEFAULT_H,
    flash_sec: float = _BURN_FLASH_SEC,
) -> list:
    """FFmpeg command: overlay scaled thumb on video for a short window at offset_sec."""
    ff = resolve_ffmpeg_executable() or "ffmpeg"
    start = max(0.0, float(offset_sec))
    end = start + max(0.04, float(flash_sec))
    # Thumb is second input; loop 1 frame for still image overlay duration.
    filter_complex = (
        f"[1:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,format=yuva420p[thumb];"
        f"[0:v][thumb]overlay=0:0:enable='between(t,{start:.3f},{end:.3f})'[vout]"
    )
    return [
        ff,
        "-y",
        "-i",
        str(video_path),
        "-loop",
        "1",
        "-i",
        str(thumb_path),
        "-filter_complex",
        filter_complex,
        "-map",
        "[vout]",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-preset",
        os.environ.get("TRANSCODE_X264_PRESET", "medium"),
        "-crf",
        os.environ.get("TRANSCODE_X264_CRF", "20"),
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(output_path),
    ]


async def burn_tiktok_styled_cover(
    video_path: Path,
    thumb_path: Path,
    output_path: Path,
    offset_sec: float,
) -> bool:
    """Re-encode TikTok MP4 with styled thumb composited at offset_sec. Returns True on success."""
    if not video_path.exists() or not thumb_path.exists():
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = build_tiktok_cover_burn_command(video_path, thumb_path, output_path, offset_sec)
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0 or not output_path.exists():
            logger.warning(
                "TikTok cover burn failed (rc=%s): %s",
                proc.returncode,
                (stderr.decode(errors="replace") or "")[-400:],
            )
            return False
        return True
    except Exception as e:
        logger.warning("TikTok cover burn error: %s", e)
        return False
