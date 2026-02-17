"""
UploadM8 Watermark Stage
==========================
Apply tier-based watermark overlay to videos using FFmpeg.

Free tier: UploadM8 watermark applied.
Creator Pro+: No watermark (or custom white-label).

Exports: run_watermark_stage(ctx)
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Optional

from .errors import SkipStage, WatermarkError, ErrorCode
from .context import JobContext

logger = logging.getLogger("uploadm8-worker")

# Watermark text (configurable via env)
WATERMARK_TEXT = os.environ.get("WATERMARK_TEXT", "UploadM8")
WATERMARK_FONT_SIZE = int(os.environ.get("WATERMARK_FONT_SIZE", "18"))
WATERMARK_OPACITY = float(os.environ.get("WATERMARK_OPACITY", "0.5"))
WATERMARK_POSITION = os.environ.get("WATERMARK_POSITION", "bottom-right")


def _get_position_filter(position: str, font_size: int) -> str:
    """Convert position name to FFmpeg drawtext x:y coordinates."""
    pad = 20
    positions = {
        "top-left": f"x={pad}:y={pad}",
        "top-center": f"x=(w-text_w)/2:y={pad}",
        "top-right": f"x=w-text_w-{pad}:y={pad}",
        "bottom-left": f"x={pad}:y=h-text_h-{pad}",
        "bottom-center": f"x=(w-text_w)/2:y=h-text_h-{pad}",
        "bottom-right": f"x=w-text_w-{pad}:y=h-text_h-{pad}",
    }
    return positions.get(position, positions["bottom-right"])


async def run_watermark_stage(ctx: JobContext) -> JobContext:
    """
    Apply watermark to the video if required by tier.

    Logic:
    - If entitlements.can_watermark is False → no watermark (paid tier benefit).
    - If True → burn a text watermark onto the video.
    - On FFmpeg failure, skip gracefully (don't block the pipeline).
    """
    ctx.mark_stage("watermark")

    # Check if watermark is needed
    if ctx.entitlements and not ctx.entitlements.can_watermark:
        raise SkipStage("Watermark not required for this tier")

    # Find the video to watermark
    video_path = ctx.processed_video_path or ctx.local_video_path
    if not video_path or not video_path.exists():
        raise SkipStage("No video file for watermark")

    if not ctx.temp_dir:
        raise SkipStage("No temp directory available")

    logger.info(f"Applying watermark to upload {ctx.upload_id}")

    output_path = ctx.temp_dir / f"wm_{ctx.upload_id}.mp4"

    # Build FFmpeg drawtext filter
    position = _get_position_filter(WATERMARK_POSITION, WATERMARK_FONT_SIZE)
    alpha = WATERMARK_OPACITY

    drawtext_filter = (
        f"drawtext=text='{WATERMARK_TEXT}':"
        f"fontsize={WATERMARK_FONT_SIZE}:"
        f"fontcolor=white@{alpha}:"
        f"{position}:"
        f"shadowcolor=black@0.3:shadowx=1:shadowy=1"
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vf", drawtext_filter,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(output_path),
    ]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()

        if proc.returncode != 0 or not output_path.exists():
            error_snippet = stderr.decode()[-300:] if stderr else "unknown error"
            logger.warning(f"Watermark FFmpeg failed (non-fatal): {error_snippet}")
            raise SkipStage(f"Watermark FFmpeg failed: rc={proc.returncode}")

        ctx.processed_video_path = output_path
        ctx.output_artifacts["watermarked_video"] = str(output_path)
        logger.info(f"Watermark applied: {output_path} ({output_path.stat().st_size} bytes)")

    except SkipStage:
        raise
    except Exception as e:
        # Watermark failure should never crash the pipeline
        logger.warning(f"Watermark stage error (non-fatal): {e}")
        raise SkipStage(f"Watermark failed: {e}")

    return ctx
