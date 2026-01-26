"""
UploadM8 HUD Stage
==================
Generate speed HUD overlay on videos using FFmpeg.
"""

import os
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from .errors import HUDError, SkipStage, ErrorCode
from .context import JobContext


logger = logging.getLogger("uploadm8-worker")


JOB_TIMEOUT_SECONDS = int(os.environ.get("JOB_TIMEOUT_SECONDS", "600"))


def generate_srt_file(ctx: JobContext, output_path: Path) -> Path:
    """
    Generate SRT subtitle file with speed data.
    
    Args:
        ctx: Job context with telemetry data
        output_path: Path to write SRT file
        
    Returns:
        Path to generated SRT file
    """
    if not ctx.telemetry or not ctx.telemetry.data_points:
        raise HUDError(
            "No telemetry data for HUD",
            code=ErrorCode.HUD_GENERATION_FAILED
        )
    
    # Get speed unit preference
    speed_unit = ctx.user_settings.get("hud_speed_unit", "mph")
    
    def format_srt_time(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    
    with open(output_path, 'w') as f:
        data_points = ctx.telemetry.data_points
        for i, point in enumerate(data_points):
            start_time = point['timestamp']
            end_time = data_points[i + 1]['timestamp'] if i + 1 < len(data_points) else start_time + 0.5
            
            speed = point['speed_mph']
            if speed_unit == 'kmh':
                speed = speed * 1.60934
                unit_label = 'KM/H'
            else:
                unit_label = 'MPH'
            
            f.write(f"{i + 1}\n")
            f.write(f"{format_srt_time(start_time)} --> {format_srt_time(end_time)}\n")
            f.write(f"{int(speed)} {unit_label}\n\n")
    
    return output_path


def build_ffmpeg_command(
    input_path: Path,
    output_path: Path,
    srt_path: Path,
    settings: dict
) -> list:
    """Build FFmpeg command for HUD overlay."""
    # Get user settings
    hud_color = settings.get("hud_color", "#FFFFFF")
    font_family = settings.get("hud_font_family", "Arial")
    font_size = settings.get("hud_font_size", 24)
    position = settings.get("hud_position", "bottom-left")
    
    # Convert hex color to FFmpeg format (BGR with alpha)
    if hud_color.startswith('#'):
        hud_color = hud_color[1:]
    # Convert RGB to BGR for ASS format
    if len(hud_color) == 6:
        r, g, b = hud_color[0:2], hud_color[2:4], hud_color[4:6]
        hud_color = f"&H{b}{g}{r}&"
    
    # Map position to ASS alignment
    alignments = {
        'top-left': 7,
        'top-center': 8,
        'top-right': 9,
        'center-left': 4,
        'center': 5,
        'center-right': 6,
        'bottom-left': 1,
        'bottom-center': 2,
        'bottom-right': 3,
    }
    alignment = alignments.get(position, 1)
    
    # Build force_style for subtitles
    force_style = (
        f"FontName={font_family},"
        f"FontSize={font_size},"
        f"PrimaryColour={hud_color},"
        f"Alignment={alignment},"
        f"MarginV=20,"
        f"BorderStyle=3,"
        f"Outline=2,"
        f"Shadow=1"
    )
    
    return [
        'ffmpeg', '-y',
        '-i', str(input_path),
        '-vf', f"subtitles={str(srt_path)}:force_style='{force_style}'",
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '23',
        '-c:a', 'copy',
        str(output_path)
    ]


async def run_hud_stage(ctx: JobContext) -> JobContext:
    """
    Execute HUD overlay generation stage.
    
    Args:
        ctx: Job context
        
    Returns:
        Updated context with processed video path
        
    Raises:
        SkipStage: If HUD not enabled or no telemetry
        HUDError: If FFmpeg fails
    """
    # Check if HUD is enabled
    if not ctx.user_settings.get("hud_enabled", True):
        raise SkipStage("HUD disabled in user settings")
    
    # Check tier entitlements
    if not ctx.entitlements.can_burn_hud:
        raise SkipStage("HUD not available for this tier")
    
    # Check if we have telemetry data
    if not ctx.telemetry or not ctx.telemetry.data_points:
        raise SkipStage("No telemetry data for HUD overlay")
    
    # Check if we have video
    if not ctx.local_video_path or not ctx.local_video_path.exists():
        raise SkipStage("No video file for HUD overlay")
    
    logger.info(f"Generating HUD overlay for upload {ctx.upload_id}")
    
    # Create temp SRT file
    srt_path = ctx.temp_dir / "speed.srt"
    try:
        generate_srt_file(ctx, srt_path)
    except Exception as e:
        raise HUDError(
            f"Failed to generate SRT file: {e}",
            code=ErrorCode.HUD_GENERATION_FAILED,
            detail=str(e)
        )
    
    # Set output path
    output_filename = f"hud_{ctx.local_video_path.name}"
    output_path = ctx.temp_dir / output_filename
    
    # Build and run FFmpeg command
    cmd = build_ffmpeg_command(
        ctx.local_video_path,
        output_path,
        srt_path,
        ctx.user_settings
    )
    
    try:
        logger.debug(f"FFmpeg command: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=JOB_TIMEOUT_SECONDS
        )
        
        if result.returncode != 0:
            logger.error(f"FFmpeg stderr: {result.stderr}")
            raise HUDError(
                "FFmpeg encoding failed",
                code=ErrorCode.FFMPEG_FAILED,
                detail=result.stderr[:500]
            )
        
        if not output_path.exists():
            raise HUDError(
                "FFmpeg produced no output",
                code=ErrorCode.FFMPEG_FAILED
            )
        
        ctx.processed_video_path = output_path
        ctx.hud_applied = True
        logger.info(f"HUD overlay generated: {output_path}")
        
    except subprocess.TimeoutExpired:
        raise HUDError(
            "FFmpeg timed out",
            code=ErrorCode.TIMEOUT,
            recoverable=True
        )
    except FileNotFoundError:
        raise HUDError(
            "FFmpeg not installed",
            code=ErrorCode.FFMPEG_NOT_FOUND,
            detail="Install FFmpeg on the worker server"
        )
    finally:
        # Clean up SRT file
        try:
            srt_path.unlink(missing_ok=True)
        except Exception:
            pass
    
    return ctx
