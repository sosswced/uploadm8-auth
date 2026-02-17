"""
UploadM8 Thumbnail Stage
=========================
Generate video thumbnails using FFmpeg.

Exports: run_thumbnail_stage(ctx)
"""

import asyncio
import logging
from pathlib import Path

from .errors import SkipStage, ErrorCode, StageError
from .context import JobContext

logger = logging.getLogger("uploadm8-worker")


async def run_thumbnail_stage(ctx: JobContext) -> JobContext:
    """
    Generate a thumbnail from the video at the 1-second mark.

    - If no video is available, skip gracefully.
    - Uses FFmpeg to extract a single frame.
    - Sets ctx.thumbnail_path on success.
    """
    ctx.mark_stage("thumbnail")

    video_path = ctx.processed_video_path or ctx.local_video_path
    if not video_path or not video_path.exists():
        raise SkipStage("No video file for thumbnail generation")

    if not ctx.temp_dir:
        raise SkipStage("No temp directory available")

    output_path = ctx.temp_dir / f"thumb_{ctx.upload_id}.jpg"

    # Extract frame at 1 second (or 0 if video is very short)
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-ss", "1",
        "-vframes", "1",
        "-q:v", "2",
        "-vf", "scale=640:-2",
        str(output_path),
    ]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()

        if proc.returncode != 0:
            # Try again at 0 seconds (very short video)
            cmd[cmd.index("1")] = "0"
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()

        if proc.returncode != 0 or not output_path.exists():
            logger.warning(f"Thumbnail generation failed (non-fatal): rc={proc.returncode}")
            raise SkipStage("FFmpeg thumbnail extraction failed")

        ctx.thumbnail_path = output_path
        ctx.output_artifacts["thumbnail"] = str(output_path)
        logger.info(f"Thumbnail generated: {output_path} ({output_path.stat().st_size} bytes)")

    except SkipStage:
        raise
    except Exception as e:
        # Thumbnail failure should never crash the pipeline
        logger.warning(f"Thumbnail stage error (non-fatal): {e}")
        raise SkipStage(f"Thumbnail generation failed: {e}")

    return ctx
