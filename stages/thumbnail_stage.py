"""
UploadM8 Thumbnail Stage
========================
Generate thumbnails using FFmpeg screenshot extraction
and optional AI enhancement via OpenAI Vision.

Features:
- FFmpeg screenshot at user-defined timestamp (settings knob)
- Multiple sample extraction for best frame selection
- OpenAI Vision for "marketable thumbnail" generation
- Tier-gated AI enhancement
"""

import os
import asyncio
import logging
import base64
import json
from pathlib import Path
from typing import Optional, List
import httpx

from .context import JobContext
from .errors import StageError, SkipStage, ErrorCode
from .entitlements import should_generate_thumbnails

logger = logging.getLogger("uploadm8-worker")

# Configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
FFMPEG_PATH = os.environ.get("FFMPEG_PATH", "ffmpeg")
FFPROBE_PATH = os.environ.get("FFPROBE_PATH", "ffprobe")


async def run_thumbnail_stage(ctx: JobContext) -> JobContext:
    """
    Generate thumbnail for video.
    
    Process:
    1. Check entitlements
    2. Get video duration
    3. Extract screenshot(s) at configured interval
    4. If AI enabled, select best and optionally enhance
    5. Upload thumbnail to R2
    """
    ctx.mark_stage("thumbnail")
    
    # Check if thumbnail generation is enabled
    if not ctx.entitlements or not should_generate_thumbnails(ctx.entitlements):
        raise SkipStage("Thumbnail generation not enabled for tier", stage="thumbnail")
    
    if not ctx.local_video_path or not ctx.local_video_path.exists():
        raise SkipStage("No local video file", stage="thumbnail")
    
    # Get user settings for screenshot
    screenshot_interval = ctx.user_settings.get("ffmpeg_screenshot_interval", 5)
    auto_generate = ctx.user_settings.get("auto_generate_thumbnails", True)
    
    if not auto_generate:
        raise SkipStage("Auto thumbnail generation disabled in settings", stage="thumbnail")
    
    try:
        # Get video duration
        duration = await get_video_duration(ctx.local_video_path)
        if duration <= 0:
            raise StageError(ErrorCode.THUMBNAIL_FAILED, "Could not determine video duration", stage="thumbnail")
        
        # Calculate screenshot timestamps
        timestamps = calculate_screenshot_timestamps(duration, screenshot_interval)
        
        # Extract screenshots
        screenshots = await extract_screenshots(ctx.local_video_path, timestamps, ctx.temp_dir)
        
        if not screenshots:
            raise StageError(ErrorCode.THUMBNAIL_FAILED, "No screenshots extracted", stage="thumbnail")
        
        # Select best thumbnail
        if ctx.entitlements.ai_thumbnails_enabled and OPENAI_API_KEY:
            best_thumbnail = await ai_select_best_thumbnail(screenshots)
        else:
            # Simple heuristic: use frame from 1/3 into video
            best_thumbnail = screenshots[len(screenshots) // 3] if len(screenshots) > 2 else screenshots[0]
        
        ctx.thumbnail_path = best_thumbnail
        
        # Upload to R2
        from . import r2 as r2_stage
        thumbnail_key = f"thumbnails/{ctx.user_id}/{ctx.upload_id}.jpg"
        await r2_stage.upload_file(best_thumbnail, thumbnail_key, "image/jpeg")
        ctx.thumbnail_r2_key = thumbnail_key
        ctx.output_artifacts["thumbnail"] = thumbnail_key
        
        logger.info(f"Thumbnail generated: {thumbnail_key}")
        return ctx
        
    except SkipStage:
        raise
    except StageError:
        raise
    except Exception as e:
        raise StageError(
            ErrorCode.THUMBNAIL_FAILED,
            f"Thumbnail generation failed: {str(e)}",
            stage="thumbnail",
            retryable=True
        )


async def get_video_duration(video_path: Path) -> float:
    """Get video duration in seconds using ffprobe."""
    try:
        cmd = [
            FFPROBE_PATH,
            "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "json",
            str(video_path)
        ]
        
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await proc.communicate()
        
        data = json.loads(stdout)
        return float(data.get("format", {}).get("duration", 0))
        
    except Exception as e:
        logger.warning(f"Failed to get video duration: {e}")
        return 0


def calculate_screenshot_timestamps(duration: float, interval: int, max_samples: int = 10) -> List[float]:
    """
    Calculate timestamps for screenshot extraction.
    
    Args:
        duration: Video duration in seconds
        interval: Seconds between samples
        max_samples: Maximum number of samples
        
    Returns:
        List of timestamps in seconds
    """
    if duration <= 0:
        return [0]
    
    if interval <= 0:
        interval = 5
    
    timestamps = []
    current = interval
    
    while current < duration and len(timestamps) < max_samples:
        timestamps.append(current)
        current += interval
    
    # Always include a point at 1/3 and 2/3 of the video
    third = duration / 3
    two_thirds = duration * 2 / 3
    
    if third not in timestamps and third < duration:
        timestamps.append(third)
    if two_thirds not in timestamps and two_thirds < duration:
        timestamps.append(two_thirds)
    
    return sorted(set(timestamps))[:max_samples]


async def extract_screenshots(video_path: Path, timestamps: List[float], temp_dir: Path) -> List[Path]:
    """
    Extract screenshots at specified timestamps using FFmpeg.
    
    Args:
        video_path: Path to video file
        timestamps: List of timestamps in seconds
        temp_dir: Directory to save screenshots
        
    Returns:
        List of paths to screenshot files
    """
    screenshots = []
    
    for i, ts in enumerate(timestamps):
        output_path = temp_dir / f"thumb_{i:03d}.jpg"
        
        cmd = [
            FFMPEG_PATH,
            "-ss", str(ts),
            "-i", str(video_path),
            "-vframes", "1",
            "-q:v", "2",  # High quality JPEG
            "-y",
            str(output_path)
        ]
        
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            _, stderr = await proc.communicate()
            
            if proc.returncode == 0 and output_path.exists():
                screenshots.append(output_path)
            else:
                logger.warning(f"Screenshot extraction failed at {ts}s: {stderr.decode()[:200]}")
                
        except Exception as e:
            logger.warning(f"Screenshot extraction error at {ts}s: {e}")
    
    return screenshots


async def ai_select_best_thumbnail(screenshots: List[Path]) -> Path:
    """
    Use OpenAI Vision to select the best thumbnail from candidates.
    
    Args:
        screenshots: List of screenshot paths
        
    Returns:
        Path to best screenshot
    """
    if not OPENAI_API_KEY or len(screenshots) <= 1:
        return screenshots[0] if screenshots else None
    
    try:
        # Encode images to base64
        images_b64 = []
        for ss in screenshots[:5]:  # Limit to 5 for API cost
            with open(ss, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
                images_b64.append(b64)
        
        # Prepare message content
        content = [
            {
                "type": "text",
                "text": """Analyze these video thumbnail candidates and select the BEST one for maximum click-through rate.
                
Consider:
- Visual appeal and clarity
- Subject focus and composition  
- Color vibrancy
- Action/interest in the frame
- Avoid blurry, dark, or unfocused frames

Respond with ONLY the number (1-5) of the best thumbnail. Example: "3" """
            }
        ]
        
        for i, b64 in enumerate(images_b64, 1):
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}",
                    "detail": "low"  # Use low detail to reduce costs
                }
            })
        
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": content}],
                    "max_tokens": 10
                }
            )
            
            if response.status_code != 200:
                logger.warning(f"OpenAI thumbnail selection failed: {response.status_code}")
                return screenshots[len(screenshots) // 3]
            
            data = response.json()
            answer = data["choices"][0]["message"]["content"].strip()
            
            # Parse response
            try:
                idx = int(answer) - 1
                if 0 <= idx < len(screenshots):
                    logger.info(f"AI selected thumbnail #{idx + 1}")
                    return screenshots[idx]
            except:
                pass
            
            return screenshots[len(screenshots) // 3]
            
    except Exception as e:
        logger.warning(f"AI thumbnail selection error: {e}")
        return screenshots[len(screenshots) // 3] if screenshots else None


async def generate_ai_thumbnail(screenshot_path: Path, temp_dir: Path) -> Optional[Path]:
    """
    Generate an enhanced AI thumbnail using OpenAI image generation.
    This is a premium feature for high tiers.
    
    Note: This uses DALL-E for image-to-image enhancement which has additional costs.
    Currently returns the original screenshot as DALL-E doesn't support direct enhancement.
    """
    # For now, return the original screenshot
    # Future: Could use DALL-E 3 with image-to-image or other enhancement APIs
    return screenshot_path
