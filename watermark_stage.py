"""
UploadM8 Watermark Stage
========================
Burn UploadM8 watermark onto video for free/low tier users.

Features:
- Semi-transparent watermark in corner
- Tier-gated (free users get watermark, paid don't)
- Configurable position and opacity
"""

import os
import asyncio
import logging
from pathlib import Path
from typing import Optional

from .context import JobContext
from .errors import StageError, SkipStage, ErrorCode
from .entitlements import should_burn_watermark

logger = logging.getLogger("uploadm8-worker")

# Configuration
FFMPEG_PATH = os.environ.get("FFMPEG_PATH", "ffmpeg")
WATERMARK_IMAGE = os.environ.get("WATERMARK_IMAGE", "/app/assets/watermark.png")
WATERMARK_OPACITY = float(os.environ.get("WATERMARK_OPACITY", "0.3"))
WATERMARK_POSITION = os.environ.get("WATERMARK_POSITION", "bottom-right")  # top-left, top-right, bottom-left, bottom-right
WATERMARK_SCALE = float(os.environ.get("WATERMARK_SCALE", "0.15"))  # 15% of video width


async def run_watermark_stage(ctx: JobContext) -> JobContext:
    """
    Burn watermark onto video if required by tier.
    
    Process:
    1. Check if watermark is required (tier-based)
    2. Apply watermark using FFmpeg
    3. Update context with new video path
    """
    ctx.mark_stage("watermark")
    
    if not ctx.entitlements:
        raise SkipStage("No entitlements loaded", stage="watermark")
    
    # Check if watermark should be applied
    if not should_burn_watermark(ctx.entitlements):
        logger.info("Watermark not required for this tier")
        raise SkipStage("Watermark not required for tier", stage="watermark")
    
    # Get video to process (HUD output or original)
    input_video = ctx.processed_video_path or ctx.local_video_path
    
    if not input_video or not input_video.exists():
        raise SkipStage("No video file to watermark", stage="watermark")
    
    # Check if watermark image exists
    watermark_path = Path(WATERMARK_IMAGE)
    if not watermark_path.exists():
        logger.warning(f"Watermark image not found: {WATERMARK_IMAGE}")
        # Generate text watermark instead
        return await apply_text_watermark(ctx, input_video)
    
    try:
        output_path = ctx.temp_dir / f"watermarked_{ctx.upload_id}.mp4"
        
        # Build FFmpeg filter for watermark
        position_filter = get_position_filter(WATERMARK_POSITION)
        
        # FFmpeg command with watermark overlay
        cmd = [
            FFMPEG_PATH,
            "-i", str(input_video),
            "-i", str(watermark_path),
            "-filter_complex",
            f"[1:v]scale=iw*{WATERMARK_SCALE}:-1,format=rgba,colorchannelmixer=aa={WATERMARK_OPACITY}[wm];"
            f"[0:v][wm]overlay={position_filter}",
            "-c:a", "copy",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-y",
            str(output_path)
        ]
        
        logger.info(f"Applying watermark: {WATERMARK_POSITION}")
        
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        _, stderr = await proc.communicate()
        
        if proc.returncode != 0:
            logger.error(f"Watermark failed: {stderr.decode()[:500]}")
            raise StageError(
                ErrorCode.WATERMARK_FAILED,
                "FFmpeg watermark overlay failed",
                stage="watermark"
            )
        
        if not output_path.exists():
            raise StageError(
                ErrorCode.WATERMARK_FAILED,
                "Watermarked video not created",
                stage="watermark"
            )
        
        ctx.processed_video_path = output_path
        logger.info(f"Watermark applied: {output_path.stat().st_size} bytes")
        
        return ctx
        
    except SkipStage:
        raise
    except StageError:
        raise
    except Exception as e:
        raise StageError(
            ErrorCode.WATERMARK_FAILED,
            f"Watermark stage failed: {str(e)}",
            stage="watermark"
        )


async def apply_text_watermark(ctx: JobContext, input_video: Path) -> JobContext:
    """
    Apply text watermark if image not available.
    """
    try:
        output_path = ctx.temp_dir / f"watermarked_{ctx.upload_id}.mp4"
        
        # Position for text
        position = WATERMARK_POSITION.replace("-", "_")
        if position == "bottom_right":
            pos_x, pos_y = "W-tw-20", "H-th-20"
        elif position == "bottom_left":
            pos_x, pos_y = "20", "H-th-20"
        elif position == "top_right":
            pos_x, pos_y = "W-tw-20", "20"
        else:  # top_left
            pos_x, pos_y = "20", "20"
        
        # FFmpeg command with text overlay
        cmd = [
            FFMPEG_PATH,
            "-i", str(input_video),
            "-vf", f"drawtext=text='UploadM8':fontsize=24:fontcolor=white@{WATERMARK_OPACITY}:x={pos_x}:y={pos_y}",
            "-c:a", "copy",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-y",
            str(output_path)
        ]
        
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        _, stderr = await proc.communicate()
        
        if proc.returncode == 0 and output_path.exists():
            ctx.processed_video_path = output_path
            logger.info("Text watermark applied")
        else:
            logger.warning(f"Text watermark failed: {stderr.decode()[:200]}")
        
        return ctx
        
    except Exception as e:
        logger.warning(f"Text watermark error: {e}")
        return ctx


def get_position_filter(position: str) -> str:
    """Convert position name to FFmpeg overlay coordinates."""
    positions = {
        "top-left": "10:10",
        "top-right": "W-w-10:10",
        "bottom-left": "10:H-h-10",
        "bottom-right": "W-w-10:H-h-10",
        "center": "(W-w)/2:(H-h)/2",
    }
    return positions.get(position, positions["bottom-right"])
