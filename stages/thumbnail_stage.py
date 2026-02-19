"""
UploadM8 Thumbnail Stage
=========================
Generate video thumbnail using FFmpeg.

The capture timestamp is configurable via user settings:
  ctx.user_settings["thumbnail_offset"] = N  (seconds, default 1.0)

This runs AFTER transcode so the thumbnail reflects the final
processed/HUD/watermarked frame that viewers will actually see.

Exports: run_thumbnail_stage(ctx)
"""

import asyncio
import logging
from pathlib import Path

from .errors import SkipStage, ErrorCode, StageError
from .context import JobContext

logger = logging.getLogger("uploadm8-worker.thumbnail")

# Fallback offset if setting is missing/invalid
DEFAULT_THUMBNAIL_OFFSET = 1.0
# Max we'll honour (prevents absurd values on short clips)
MAX_THUMBNAIL_OFFSET = 300.0


def _resolve_offset(ctx: JobContext) -> float:
    """
    Read thumbnail_offset from user_settings.

    Priority: user_settings["thumbnail_offset"] -> DEFAULT_THUMBNAIL_OFFSET
    Value is clamped to [0, MAX_THUMBNAIL_OFFSET].
    """
    raw = (ctx.user_settings or {}).get("thumbnail_offset", DEFAULT_THUMBNAIL_OFFSET)
    try:
        offset = float(raw)
    except (TypeError, ValueError):
        logger.warning(
            f"Invalid thumbnail_offset '{raw}', using default {DEFAULT_THUMBNAIL_OFFSET}s"
        )
        offset = DEFAULT_THUMBNAIL_OFFSET

    return max(0.0, min(offset, MAX_THUMBNAIL_OFFSET))


async def _extract_frame(video_path: Path, output_path: Path, offset: float) -> bool:
    """
    Run FFmpeg to extract a single JPEG frame.
    Returns True on success.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", f"{offset:.3f}",
        "-i", str(video_path),
        "-vframes", "1",
        "-q:v", "2",
        "-vf", "scale=1080:-2",  # 1080px wide, proportional height (even)
        str(output_path),
    ]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()

    if proc.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
        return True

    logger.debug(
        f"FFmpeg thumb failed at {offset}s (rc={proc.returncode}): "
        f"{stderr.decode()[-200:]}"
    )
    return False


async def run_thumbnail_stage(ctx: JobContext) -> JobContext:
    """
    Generate a thumbnail from the final processed video.

    Strategy:
    1. Prefer ctx.processed_video_path (post-HUD/watermark/transcode).
    2. Fall back to ctx.local_video_path (original).
    3. Read capture offset from ctx.user_settings["thumbnail_offset"].
    4. If capture fails at requested offset, retry at 0s (handles very short clips).
    5. Sets ctx.thumbnail_path on success.
    """
    ctx.mark_stage("thumbnail")

    # Choose source video: prefer the fully-processed copy
    video_path = None
    for candidate in (ctx.processed_video_path, ctx.local_video_path):
        if candidate and Path(candidate).exists():
            video_path = Path(candidate)
            break

    if not video_path:
        raise SkipStage("No video file available for thumbnail generation")

    if not ctx.temp_dir:
        raise SkipStage("No temp directory available")

    output_path = ctx.temp_dir / f"thumb_{ctx.upload_id}.jpg"
    offset = _resolve_offset(ctx)

    logger.info(
        f"Extracting thumbnail from {video_path.name} at {offset}s "
        f"(user setting: {ctx.user_settings.get('thumbnail_offset', 'default')})"
    )

    try:
        success = await _extract_frame(video_path, output_path, offset)

        # Fall back to frame 0 if requested offset is beyond video length
        if not success and offset > 0:
            logger.info(f"Thumbnail at {offset}s failed, retrying at 0s")
            success = await _extract_frame(video_path, output_path, 0.0)

        if not success:
            logger.warning("Thumbnail generation failed (non-fatal) — no frame extracted")
            raise SkipStage("FFmpeg thumbnail extraction produced no output")

        ctx.thumbnail_path = output_path
        ctx.output_artifacts["thumbnail"] = str(output_path)
        sz_kb = output_path.stat().st_size / 1024
        logger.info(
            f"Thumbnail generated: {output_path.name} ({sz_kb:.1f} KB) at {offset:.1f}s"
        )

    except SkipStage:
        raise
    except Exception as e:
        logger.warning(f"Thumbnail stage error (non-fatal): {e}")
        raise SkipStage(f"Thumbnail generation failed: {e}")

    return ctx
