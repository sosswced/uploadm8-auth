"""
UploadM8 Transcode Stage - Converts videos to platform-specific formats

Ensures videos meet the requirements for:
- YouTube Shorts: H.264, AAC, 9:16 aspect ratio, max 60 seconds
- TikTok: H.264, AAC, 9:16 aspect ratio, max 10 minutes
- Instagram Reels: H.264, AAC, 9:16 aspect ratio, max 90 seconds
- Facebook Reels: H.264, AAC, 9:16 aspect ratio, max 90 seconds
"""

import asyncio
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from .context import JobContext
from .errors import StageError, SkipStage, ErrorCode

logger = logging.getLogger("uploadm8-worker.transcode")

# Platform-specific requirements
PLATFORM_SPECS = {
    "youtube": {
        "name": "YouTube Shorts",
        "max_duration": 60,           # seconds
        "preferred_aspect": (9, 16),  # vertical
        "min_aspect": (9, 16),
        "max_aspect": (16, 9),        # also supports horizontal
        "video_codec": "h264",
        "audio_codec": "aac",
        "max_width": 1080,
        "max_height": 1920,
        "max_fps": 60,
        "max_bitrate_video": "12M",
        "max_bitrate_audio": "192k",
        "sample_rate": 48000,
        "pixel_format": "yuv420p",
    },
    "tiktok": {
        "name": "TikTok",
        "max_duration": 600,          # 10 minutes
        "preferred_aspect": (9, 16),
        "min_aspect": (9, 16),
        "max_aspect": (16, 9),
        "video_codec": "h264",
        "audio_codec": "aac",
        "max_width": 1080,
        "max_height": 1920,
        "max_fps": 60,
        "max_bitrate_video": "10M",
        "max_bitrate_audio": "192k",
        "sample_rate": 44100,
        "pixel_format": "yuv420p",
    },
    "instagram": {
        "name": "Instagram Reels",
        "max_duration": 90,
        "preferred_aspect": (9, 16),
        "min_aspect": (4, 5),
        "max_aspect": (16, 9),
        "video_codec": "h264",
        "audio_codec": "aac",
        "max_width": 1080,
        "max_height": 1920,
        "max_fps": 30,
        "max_bitrate_video": "8M",
        "max_bitrate_audio": "128k",
        "sample_rate": 44100,
        "pixel_format": "yuv420p",
    },
    "facebook": {
        "name": "Facebook Reels",
        "max_duration": 90,
        "preferred_aspect": (9, 16),
        "min_aspect": (9, 16),
        "max_aspect": (16, 9),
        "video_codec": "h264",
        "audio_codec": "aac",
        "max_width": 1080,
        "max_height": 1920,
        "max_fps": 30,
        "max_bitrate_video": "8M",
        "max_bitrate_audio": "128k",
        "sample_rate": 44100,
        "pixel_format": "yuv420p",
    },
}


@dataclass
class VideoInfo:
    """Parsed video metadata from ffprobe"""
    width: int
    height: int
    duration: float
    fps: float
    video_codec: str
    audio_codec: Optional[str]
    video_bitrate: Optional[int]
    audio_bitrate: Optional[int]
    sample_rate: Optional[int]
    pixel_format: str
    rotation: int = 0


async def get_video_info(video_path: Path) -> VideoInfo:
    """Use ffprobe to get video metadata"""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(video_path)
    ]
    
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        
        if proc.returncode != 0:
            raise StageError(ErrorCode.TRANSCODE_FAILED, f"ffprobe failed: {stderr.decode()}")
        
        data = json.loads(stdout.decode())
        
        # Find video and audio streams
        video_stream = None
        audio_stream = None
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video" and not video_stream:
                video_stream = stream
            elif stream.get("codec_type") == "audio" and not audio_stream:
                audio_stream = stream
        
        if not video_stream:
            raise StageError(ErrorCode.TRANSCODE_FAILED, "No video stream found")
        
        # Parse video info
        width = int(video_stream.get("width", 0))
        height = int(video_stream.get("height", 0))
        
        # Handle rotation metadata (common in phone videos)
        rotation = 0
        if "tags" in video_stream:
            rotation = int(video_stream["tags"].get("rotate", 0))
        if "side_data_list" in video_stream:
            for side_data in video_stream["side_data_list"]:
                if "rotation" in side_data:
                    rotation = int(side_data["rotation"])
        
        # Swap width/height if rotated 90 or 270 degrees
        if rotation in (90, 270, -90, -270):
            width, height = height, width
        
        # Parse framerate
        fps_str = video_stream.get("r_frame_rate", "30/1")
        if "/" in fps_str:
            num, den = fps_str.split("/")
            fps = float(num) / float(den) if float(den) > 0 else 30.0
        else:
            fps = float(fps_str)
        
        # Parse duration
        duration = float(data.get("format", {}).get("duration", 0))
        if duration == 0:
            duration = float(video_stream.get("duration", 0))
        
        # Parse bitrates
        video_bitrate = int(video_stream.get("bit_rate", 0)) if video_stream.get("bit_rate") else None
        audio_bitrate = int(audio_stream.get("bit_rate", 0)) if audio_stream and audio_stream.get("bit_rate") else None
        sample_rate = int(audio_stream.get("sample_rate", 0)) if audio_stream and audio_stream.get("sample_rate") else None
        
        return VideoInfo(
            width=width,
            height=height,
            duration=duration,
            fps=fps,
            video_codec=video_stream.get("codec_name", "unknown"),
            audio_codec=audio_stream.get("codec_name") if audio_stream else None,
            video_bitrate=video_bitrate,
            audio_bitrate=audio_bitrate,
            sample_rate=sample_rate,
            pixel_format=video_stream.get("pix_fmt", "unknown"),
            rotation=rotation,
        )
        
    except json.JSONDecodeError as e:
        raise StageError(ErrorCode.TRANSCODE_FAILED, f"Failed to parse ffprobe output: {e}")
    except Exception as e:
        if isinstance(e, StageError):
            raise
        raise StageError(ErrorCode.TRANSCODE_FAILED, f"Failed to get video info: {e}")


def needs_transcode(info: VideoInfo, platform: str) -> Tuple[bool, list]:
    """Check if video needs transcoding for the target platform"""
    spec = PLATFORM_SPECS.get(platform)
    if not spec:
        return False, []
    
    reasons = []
    
    # Check codec
    if info.video_codec.lower() not in ("h264", "avc"):
        reasons.append(f"video codec is {info.video_codec}, needs h264")
    
    if info.audio_codec and info.audio_codec.lower() != "aac":
        reasons.append(f"audio codec is {info.audio_codec}, needs aac")
    
    # Check pixel format (must be yuv420p for compatibility)
    if info.pixel_format != "yuv420p":
        reasons.append(f"pixel format is {info.pixel_format}, needs yuv420p")
    
    # Check resolution
    if info.width > spec["max_width"] or info.height > spec["max_height"]:
        reasons.append(f"resolution {info.width}x{info.height} exceeds {spec['max_width']}x{spec['max_height']}")
    
    # Check fps
    if info.fps > spec["max_fps"]:
        reasons.append(f"fps {info.fps:.1f} exceeds max {spec['max_fps']}")
    
    # Check duration
    if info.duration > spec["max_duration"]:
        reasons.append(f"duration {info.duration:.1f}s exceeds max {spec['max_duration']}s")
    
    # Check rotation (needs to be applied)
    if info.rotation != 0:
        reasons.append(f"video has {info.rotation}° rotation that needs to be applied")
    
    return len(reasons) > 0, reasons


def build_ffmpeg_command(
    input_path: Path,
    output_path: Path,
    info: VideoInfo,
    platform: str,
    target_aspect: str = "auto"
) -> list:
    """Build FFmpeg command for transcoding"""
    spec = PLATFORM_SPECS.get(platform, PLATFORM_SPECS["youtube"])
    
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output
        "-i", str(input_path),
    ]
    
    # Video filters
    vf_filters = []
    
    # Handle rotation
    if info.rotation == 90:
        vf_filters.append("transpose=1")
    elif info.rotation == 180:
        vf_filters.append("transpose=1,transpose=1")
    elif info.rotation == 270 or info.rotation == -90:
        vf_filters.append("transpose=2")
    
    # Calculate scaling
    target_width = min(info.width, spec["max_width"])
    target_height = min(info.height, spec["max_height"])
    
    # Ensure dimensions are even (required for h264)
    target_width = target_width - (target_width % 2)
    target_height = target_height - (target_height % 2)
    
    # Add scaling filter
    vf_filters.append(f"scale={target_width}:{target_height}:force_original_aspect_ratio=decrease")
    vf_filters.append(f"pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2")
    
    # Add format conversion for pixel format
    vf_filters.append("format=yuv420p")
    
    if vf_filters:
        cmd.extend(["-vf", ",".join(vf_filters)])
    
    # Video codec settings
    cmd.extend([
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "23",
        "-profile:v", "high",
        "-level", "4.1",
        "-maxrate", spec["max_bitrate_video"],
        "-bufsize", spec["max_bitrate_video"].replace("M", "") + "M",
    ])
    
    # Frame rate limiting
    if info.fps > spec["max_fps"]:
        cmd.extend(["-r", str(spec["max_fps"])])
    
    # Duration limiting
    if info.duration > spec["max_duration"]:
        cmd.extend(["-t", str(spec["max_duration"])])
    
    # Audio codec settings
    if info.audio_codec:
        cmd.extend([
            "-c:a", "aac",
            "-b:a", spec["max_bitrate_audio"],
            "-ar", str(spec["sample_rate"]),
            "-ac", "2",  # Stereo
        ])
    else:
        # No audio - add silent audio track (some platforms require audio)
        cmd.extend([
            "-f", "lavfi",
            "-i", "anullsrc=r=44100:cl=stereo",
            "-shortest",
            "-c:a", "aac",
            "-b:a", "128k",
        ])
    
    # Output settings
    cmd.extend([
        "-movflags", "+faststart",  # Enable streaming
        "-pix_fmt", "yuv420p",
        str(output_path)
    ])
    
    return cmd


async def transcode_video(
    input_path: Path,
    output_path: Path,
    platform: str,
    info: Optional[VideoInfo] = None
) -> Path:
    """Transcode video to platform-specific format"""
    if not info:
        info = await get_video_info(input_path)
    
    cmd = build_ffmpeg_command(input_path, output_path, info, platform)
    
    logger.info(f"Transcoding for {platform}: {' '.join(cmd[:10])}...")
    
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        
        if proc.returncode != 0:
            error_msg = stderr.decode()[-500:]  # Last 500 chars of error
            raise StageError(ErrorCode.TRANSCODE_FAILED, f"FFmpeg failed: {error_msg}")
        
        if not output_path.exists():
            raise StageError(ErrorCode.TRANSCODE_FAILED, "Output file not created")
        
        logger.info(f"Transcode complete: {output_path.stat().st_size / 1024 / 1024:.1f}MB")
        return output_path
        
    except Exception as e:
        if isinstance(e, StageError):
            raise
        raise StageError(ErrorCode.TRANSCODE_FAILED, f"Transcode failed: {e}")


async def run_transcode_stage(ctx: JobContext) -> JobContext:
    """
    Run the transcode stage - creates platform-specific versions of the video
    
    This stage:
    1. Analyzes the input video
    2. Determines which platforms need transcoding
    3. Creates optimized versions for each platform
    """
    ctx.mark_stage("transcode")
    
    if not ctx.local_video_path or not ctx.local_video_path.exists():
        raise SkipStage("No video file to transcode")
    
    platforms = ctx.platforms or []
    if not platforms:
        raise SkipStage("No target platforms specified")
    
    logger.info(f"Analyzing video for platforms: {platforms}")
    
    # Get video info
    info = await get_video_info(ctx.local_video_path)
    logger.info(f"Video: {info.width}x{info.height}, {info.duration:.1f}s, {info.fps:.1f}fps, "
                f"codec={info.video_codec}, rotation={info.rotation}")
    
    # Store video info in context
    ctx.video_info = {
        "width": info.width,
        "height": info.height,
        "duration": info.duration,
        "fps": info.fps,
        "video_codec": info.video_codec,
        "audio_codec": info.audio_codec,
    }
    
    # Check each platform and transcode if needed
    ctx.platform_videos = {}
    transcoded_any = False
    
    for platform in platforms:
        needs_tc, reasons = needs_transcode(info, platform)
        
        if needs_tc:
            logger.info(f"{platform} needs transcode: {', '.join(reasons)}")
            
            output_path = ctx.temp_dir / f"transcoded_{platform}.mp4"
            await transcode_video(ctx.local_video_path, output_path, platform, info)
            
            ctx.platform_videos[platform] = output_path
            transcoded_any = True
        else:
            logger.info(f"{platform}: video is already compatible")
            ctx.platform_videos[platform] = ctx.local_video_path
    
    # If we transcoded, update the processed video path to the first transcoded version
    # (for backwards compatibility with single-output workflows)
    if transcoded_any:
        first_platform = platforms[0]
        if first_platform in ctx.platform_videos:
            ctx.processed_video_path = ctx.platform_videos[first_platform]
    
    return ctx


# Utility function to check if FFmpeg is available
async def check_ffmpeg_available() -> bool:
    """Check if FFmpeg is installed and accessible"""
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await proc.communicate()
        return proc.returncode == 0
    except Exception:
        return False
