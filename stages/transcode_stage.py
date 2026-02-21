"""
UploadM8 Transcode Stage - Platform-specific video transcoding

Creates separate, optimized video files for each target platform.
Each platform gets its own properly formatted MP4 with correct:
  - Aspect ratio (9:16 vertical via pad/crop based on reframe_mode)
  - Resolution (max 1080x1920)
  - Duration (trimmed to platform max)
  - Codec (H.264 High + AAC-LC)
  - Frame rate (capped per platform)
  - Pixel format (yuv420p)
  - Audio sample rate (per platform)

Reframe modes (ctx.reframe_mode):
  auto  - Landscape input -> pad to 9:16 for all platforms. Already vertical -> pass through.
  pad   - Always letterbox/pillarbox to 9:16 (black bars, no content lost)
  crop  - Center-crop to fill 9:16 (loses edges, no black bars)
  none  - Keep original aspect ratio, only fix codec/format/duration/fps

Platform specs verified February 2026.
"""

import asyncio
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .context import JobContext
from .errors import StageError, SkipStage, ErrorCode

logger = logging.getLogger("uploadm8-worker.transcode")


# -------------------------------------------------------------------------
# Platform specifications (verified Feb 2026)
# -------------------------------------------------------------------------
PLATFORM_SPECS = {
    "tiktok": {
        "name": "TikTok",
        "max_duration": 600,            # 10 min via Content Posting API
        "target_width": 1080,
        "target_height": 1920,
        "preferred_aspect": (9, 16),    # vertical
        "video_codec": "h264",
        "audio_codec": "aac",
        "max_fps": 30,                  # 60 accepted but heavily compressed; 30 safer
        "max_bitrate_video": "10M",
        "max_bitrate_audio": "256k",
        "sample_rate": 44100,
        "pixel_format": "yuv420p",
        "profile": "high",
        "level": "4.2",
        "max_file_mb": 500,             # API limit
    },
    "youtube": {
        "name": "YouTube Shorts",
        "max_duration": 180,            # 3 minutes (expanded Oct 2024)
        "target_width": 1080,
        "target_height": 1920,
        "preferred_aspect": (9, 16),
        "video_codec": "h264",
        "audio_codec": "aac",
        "max_fps": 60,
        "max_bitrate_video": "12M",
        "max_bitrate_audio": "192k",
        "sample_rate": 48000,
        "pixel_format": "yuv420p",
        "profile": "high",
        "level": "4.1",
        "max_file_mb": 256000,          # YouTube general limit (256GB)
    },
    "instagram": {
        "name": "Instagram Reels",
        "max_duration": 90,             # 90s via API (app supports up to 3 min / 15 min)
        "target_width": 1080,
        "target_height": 1920,
        "preferred_aspect": (9, 16),
        "video_codec": "h264",
        "audio_codec": "aac",
        "max_fps": 30,
        "max_bitrate_video": "5M",      # IG compresses aggressively; lower = less re-compression
        "max_bitrate_audio": "128k",
        "sample_rate": 44100,
        "pixel_format": "yuv420p",
        "profile": "high",
        "level": "4.0",
        "max_file_mb": 4096,            # 4GB
    },
    "facebook": {
        "name": "Facebook Reels",
        "max_duration": 0,              # 0 = no limit (June 2025: all videos are Reels)
        "target_width": 1080,
        "target_height": 1920,
        "preferred_aspect": (9, 16),
        "video_codec": "h264",
        "audio_codec": "aac",
        "max_fps": 30,
        "max_bitrate_video": "8M",
        "max_bitrate_audio": "128k",
        "sample_rate": 44100,
        "pixel_format": "yuv420p",
        "profile": "high",
        "level": "4.1",
        "max_file_mb": 4096,            # 4GB
    },
}


# -------------------------------------------------------------------------
# Video info
# -------------------------------------------------------------------------
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
    file_size: int = 0

    @property
    def is_landscape(self) -> bool:
        return self.width > self.height

    @property
    def is_portrait(self) -> bool:
        return self.height > self.width

    @property
    def is_square(self) -> bool:
        return self.width == self.height

    @property
    def aspect_ratio(self) -> float:
        if self.height == 0:
            return 0
        return self.width / self.height


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

        # Parse dimensions
        width = int(video_stream.get("width", 0))
        height = int(video_stream.get("height", 0))

        # Handle rotation metadata (common in phone videos)
        rotation = 0
        if "tags" in video_stream:
            rotation = int(video_stream["tags"].get("rotate", 0))
        if "side_data_list" in video_stream:
            for side_data in video_stream["side_data_list"]:
                if "rotation" in side_data:
                    rotation = abs(int(side_data["rotation"]))

        # Swap width/height if rotated 90 or 270 degrees
        if rotation in (90, 270):
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

        # File size
        file_size = int(data.get("format", {}).get("size", 0))
        if file_size == 0:
            try:
                file_size = video_path.stat().st_size
            except Exception:
                pass

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
            file_size=file_size,
        )

    except json.JSONDecodeError as e:
        raise StageError(ErrorCode.TRANSCODE_FAILED, f"Failed to parse ffprobe output: {e}")
    except Exception as e:
        if isinstance(e, StageError):
            raise
        raise StageError(ErrorCode.TRANSCODE_FAILED, f"Failed to get video info: {e}")


# -------------------------------------------------------------------------
# Transcode decision logic
# -------------------------------------------------------------------------
def resolve_reframe_action(info: VideoInfo, reframe_mode: str, platform: str) -> str:
    """
    Decide the actual reframe action for this video + platform + mode.

    Returns one of: "pad", "crop", "none"
    """
    mode = (reframe_mode or "auto").lower().strip()

    if mode == "none":
        return "none"

    if mode == "pad":
        return "pad"

    if mode == "crop":
        return "crop"

    # mode == "auto" (default)
    # If video is already vertical (9:16 or close), no reframing needed
    if info.is_portrait or info.is_square:
        return "none"

    # Landscape video going to a vertical platform -> pad (safe default, preserves all content)
    return "pad"


def needs_transcode(info: VideoInfo, platform: str, reframe_action: str) -> Tuple[bool, List[str]]:
    """Check if video needs transcoding for the target platform"""
    spec = PLATFORM_SPECS.get(platform)
    if not spec:
        return False, []

    reasons = []

    # Check if reframing is needed (aspect ratio change)
    if reframe_action in ("pad", "crop"):
        reasons.append(f"reframe={reframe_action} to {spec['target_width']}x{spec['target_height']}")

    # Check codec
    if info.video_codec.lower() not in ("h264", "avc"):
        reasons.append(f"video codec is {info.video_codec}, needs h264")

    if info.audio_codec and info.audio_codec.lower() != "aac":
        reasons.append(f"audio codec is {info.audio_codec}, needs aac")

    # Check pixel format
    if info.pixel_format != "yuv420p":
        reasons.append(f"pixel format is {info.pixel_format}, needs yuv420p")

    # Check resolution (even without reframe, may need downscale)
    if reframe_action == "none":
        if info.width > spec["target_width"] or info.height > spec["target_height"]:
            reasons.append(f"resolution {info.width}x{info.height} exceeds max")

    # Check fps
    if info.fps > spec["max_fps"]:
        reasons.append(f"fps {info.fps:.1f} exceeds max {spec['max_fps']}")

    # Check duration
    if spec["max_duration"] > 0 and info.duration > spec["max_duration"]:
        reasons.append(f"duration {info.duration:.1f}s exceeds max {spec['max_duration']}s")

    # Check audio sample rate
    if info.sample_rate and info.sample_rate != spec["sample_rate"]:
        reasons.append(f"sample rate {info.sample_rate} needs {spec['sample_rate']}")

    # Check rotation
    if info.rotation != 0:
        reasons.append(f"has {info.rotation} degree rotation to apply")

    return len(reasons) > 0, reasons


# -------------------------------------------------------------------------
# FFmpeg command builder
# -------------------------------------------------------------------------
def build_ffmpeg_command(
    input_path: Path,
    output_path: Path,
    info: VideoInfo,
    platform: str,
    reframe_action: str,
) -> list:
    """Build platform-specific FFmpeg command"""
    spec = PLATFORM_SPECS.get(platform, PLATFORM_SPECS["tiktok"])

    needs_silent_audio = not info.audio_codec

    cmd = [
        "ffmpeg",
        "-y",                           # Overwrite output
        "-i", str(input_path),
    ]

    # When there is no audio stream we must declare the silent source as a
    # second input BEFORE any output/codec flags — FFmpeg requires all -i
    # arguments to precede filter/codec/output arguments.
    if needs_silent_audio:
        cmd.extend([
            "-f", "lavfi",
            "-i", f"anullsrc=r={spec['sample_rate']}:cl=stereo",
        ])

    # -- Video filters --
    vf_filters = []

    # 1. Handle rotation (metadata rotation not auto-applied by all decoders)
    if info.rotation == 90:
        vf_filters.append("transpose=1")
    elif info.rotation == 180:
        vf_filters.append("transpose=1,transpose=1")
    elif info.rotation == 270:
        vf_filters.append("transpose=2")

    # 2. Reframe logic
    tw = spec["target_width"]
    th = spec["target_height"]

    if reframe_action == "pad":
        # Scale to fit inside target box, then pad with black bars
        # force_original_aspect_ratio=decrease -> shrink to fit
        # pad -> center in target box
        vf_filters.append(f"scale={tw}:{th}:force_original_aspect_ratio=decrease")
        vf_filters.append(f"pad={tw}:{th}:(ow-iw)/2:(oh-ih)/2")

    elif reframe_action == "crop":
        # Scale to fill target box (some edges clipped), then crop to exact size
        # force_original_aspect_ratio=increase -> scale up to fill
        # crop -> center-crop to target
        vf_filters.append(f"scale={tw}:{th}:force_original_aspect_ratio=increase")
        vf_filters.append(f"crop={tw}:{th}")

    elif reframe_action == "none":
        # Keep original aspect ratio but ensure within max bounds
        max_w = spec["target_width"]
        max_h = spec["target_height"]

        # For landscape videos with reframe=none, allow landscape dimensions
        if info.is_landscape:
            max_w = max(spec["target_width"], spec["target_height"])  # 1920
            max_h = min(spec["target_width"], spec["target_height"])  # 1080

        if info.width > max_w or info.height > max_h:
            vf_filters.append(f"scale={max_w}:{max_h}:force_original_aspect_ratio=decrease")

        # Ensure even dimensions (h264 requirement)
        vf_filters.append("pad=ceil(iw/2)*2:ceil(ih/2)*2")

    # 3. Force yuv420p (universal compatibility)
    vf_filters.append("format=yuv420p")

    if vf_filters:
        cmd.extend(["-vf", ",".join(vf_filters)])

    # -- Video codec settings --
    cmd.extend([
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "23",
        "-profile:v", spec.get("profile", "high"),
        "-level", spec.get("level", "4.1"),
        "-maxrate", spec["max_bitrate_video"],
        "-bufsize", str(int(spec["max_bitrate_video"].replace("M", "")) * 2) + "M",
    ])

    # -- Frame rate capping --
    if info.fps > spec["max_fps"]:
        cmd.extend(["-r", str(spec["max_fps"])])

    # -- Duration trimming --
    if spec["max_duration"] > 0 and info.duration > spec["max_duration"]:
        # Trim 0.5s before max to avoid edge-case frame overrun
        trim_to = max(1.0, spec["max_duration"] - 0.5)
        cmd.extend(["-t", f"{trim_to:.3f}"])

    # -- Audio settings --
    # The anullsrc input (input index 1) was already declared at the top of the
    # command when needs_silent_audio is True.  We now either encode the real
    # audio stream or map+encode the synthetic silent stream.
    if not needs_silent_audio:
        cmd.extend([
            "-c:a", "aac",
            "-b:a", spec["max_bitrate_audio"],
            "-ar", str(spec["sample_rate"]),
            "-ac", "2",                  # Force stereo
        ])
    else:
        # Map video from input 0, audio from anullsrc input 1
        cmd.extend([
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            "-c:a", "aac",
            "-b:a", "128k",
            "-ar", str(spec["sample_rate"]),
            "-ac", "2",
        ])

    # -- Output container settings --
    cmd.extend([
        "-movflags", "+faststart",       # Enable progressive download / streaming
        "-pix_fmt", "yuv420p",
        str(output_path)
    ])

    return cmd


# -------------------------------------------------------------------------
# Transcode execution
# -------------------------------------------------------------------------
async def transcode_video(
    input_path: Path,
    output_path: Path,
    platform: str,
    info: VideoInfo,
    reframe_action: str,
    db_pool=None,
    upload_id: str = None,
) -> Path:
    """Transcode a video to a platform-specific format.
    
    When db_pool and upload_id are provided, streams FFmpeg stderr in real-time
    and writes progress (0-100) to the uploads table every ~2 seconds so the
    frontend can display a live progress bar.
    """
    cmd = build_ffmpeg_command(input_path, output_path, info, platform, reframe_action)

    cmd_preview = " ".join(cmd[:15]) + "..."
    logger.info(
        f"Transcoding for {platform}: {info.width}x{info.height} -> {output_path.name}, "
        f"reframe={reframe_action}, cmd starts: {cmd_preview}"
    )

    # Duration in seconds — used to calculate percent complete from FFmpeg time output
    total_duration = info.duration if info.duration and info.duration > 0 else None

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stderr_lines = []
        last_progress_write = 0.0
        PROGRESS_INTERVAL = 2.0  # Write to DB at most once per 2 seconds

        if db_pool and upload_id and total_duration:
            # Stream stderr line-by-line so we can parse FFmpeg progress in real time
            # FFmpeg writes lines like: frame=  120 fps= 30 ... time=00:00:04.00 ...
            import re, time as _time
            time_pattern = re.compile(r"time=(\d+):(\d+):(\d+\.\d+)")

            async def _stream_stderr():
                nonlocal last_progress_write
                while True:
                    line_bytes = await proc.stderr.readline()
                    if not line_bytes:
                        break
                    line = line_bytes.decode("utf-8", errors="replace")
                    stderr_lines.append(line)
                    m = time_pattern.search(line)
                    if m:
                        h, mn, s = float(m.group(1)), float(m.group(2)), float(m.group(3))
                        encoded_secs = h * 3600 + mn * 60 + s
                        pct = int(min(99, (encoded_secs / total_duration) * 100))
                        now = _time.monotonic()
                        if now - last_progress_write >= PROGRESS_INTERVAL:
                            last_progress_write = now
                            try:
                                from stages import db as _db
                                await _db.update_upload_progress(
                                    db_pool, upload_id, pct,
                                    stage=f"transcoding:{platform}"
                                )
                            except Exception:
                                pass

            await asyncio.gather(proc.wait(), _stream_stderr())
            stderr_output = "".join(stderr_lines)
        else:
            # No db_pool — fall back to original blocking communicate()
            stdout_data, stderr_data = await proc.communicate()
            stderr_output = stderr_data.decode("utf-8", errors="replace")

        if proc.returncode != 0:
            error_msg = stderr_output[-500:]
            raise StageError(ErrorCode.TRANSCODE_FAILED, f"FFmpeg failed for {platform}: {error_msg}")

        if not output_path.exists():
            raise StageError(ErrorCode.TRANSCODE_FAILED, f"Output file not created for {platform}")

        out_size_mb = output_path.stat().st_size / 1024 / 1024
        logger.info(f"Transcode complete for {platform}: {out_size_mb:.1f}MB")

        # Warn if output exceeds platform file size limit
        spec = PLATFORM_SPECS.get(platform, {})
        max_mb = spec.get("max_file_mb", 0)
        if max_mb > 0 and out_size_mb > max_mb:
            logger.warning(
                f"{platform} WARNING: Output {out_size_mb:.1f}MB exceeds platform limit {max_mb}MB. "
                f"Upload may be rejected."
            )

        return output_path

    except Exception as e:
        if isinstance(e, StageError):
            raise
        raise StageError(ErrorCode.TRANSCODE_FAILED, f"Transcode failed for {platform}: {e}")


# -------------------------------------------------------------------------
# Stage entry point
# -------------------------------------------------------------------------
async def run_transcode_stage(ctx: JobContext, db_pool=None) -> JobContext:
    """
    Run the transcode stage - creates platform-specific video versions.

    Pipeline flow:
    1. Analyze input video (ffprobe)
    2. For each platform:
       a. Resolve reframe action (auto/pad/crop/none -> actual action)
       b. Check if transcoding is needed
       c. Build platform-specific FFmpeg command
       d. Execute transcode
    3. Store per-platform video paths in ctx.platform_videos
    """
    ctx.mark_stage("transcode")

    if not ctx.local_video_path or not ctx.local_video_path.exists():
        raise SkipStage("No video file to transcode")

    platforms = ctx.platforms or []
    if not platforms:
        raise SkipStage("No target platforms specified")

    reframe_mode = getattr(ctx, "reframe_mode", "auto") or "auto"

    # -- Step 1: Analyze input video --
    # Use HUD/watermarked video if available (pipeline reorder: HUD+watermark run before transcode)
    input_video = ctx.local_video_path
    if ctx.processed_video_path and ctx.processed_video_path.exists():
        input_video = ctx.processed_video_path
        logger.info(f"Using processed video as transcode input: {input_video.name}")

    info = await get_video_info(input_video)

    orientation = "landscape" if info.is_landscape else ("portrait" if info.is_portrait else "square")
    file_mb = info.file_size / 1024 / 1024 if info.file_size else 0

    logger.info(
        f"Video: {info.width}x{info.height}, {info.duration:.1f}s, {info.fps:.1f}fps, "
        f"codec={info.video_codec}, pix_fmt={info.pixel_format}, rotation={info.rotation}, "
        f"file_size={file_mb:.1f}MB"
    )
    logger.info(f"Analyzing video for platforms: {platforms}, reframe_mode={reframe_mode}")

    # Store video info in context
    ctx.video_info = {
        "width": info.width,
        "height": info.height,
        "duration": info.duration,
        "fps": info.fps,
        "video_codec": info.video_codec,
        "audio_codec": info.audio_codec,
        "orientation": orientation,
        "file_size": info.file_size,
    }

    # -- Step 2: Per-platform transcode --
    ctx.platform_videos = {}
    transcoded_any = False

    for platform in platforms:
        spec = PLATFORM_SPECS.get(platform)
        if not spec:
            logger.warning(f"Unknown platform '{platform}', skipping transcode")
            ctx.platform_videos[platform] = ctx.local_video_path
            continue

        # Resolve what reframe action to take for this platform
        reframe_action = resolve_reframe_action(info, reframe_mode, platform)

        # Log the reframe decision
        if reframe_action != "none":
            logger.info(
                f"{platform}: {reframe_mode} mode -> {reframe_action} to "
                f"{spec['target_width']}x{spec['target_height']}"
            )

        # Check if landscape video going to vertical platform (informational warning)
        if info.is_landscape and reframe_action == "none":
            logger.info(
                f"{platform} WARNING: Video is landscape {info.width}x{info.height}; "
                f"{spec['name']} prefers vertical 9:16. Video will be letterboxed by the platform."
            )

        # Check duration
        if spec["max_duration"] > 0 and info.duration > spec["max_duration"]:
            logger.info(
                f"{platform}: Video {info.duration:.1f}s exceeds max {spec['max_duration']}s, "
                f"will be trimmed to {spec['max_duration'] - 0.5:.1f}s"
            )

        # Determine if transcode is needed
        needs_tc, reasons = needs_transcode(info, platform, reframe_action)

        if needs_tc:
            for reason in reasons:
                logger.info(f"{platform} needs transcode: {reason}")

            output_path = ctx.temp_dir / f"transcoded_{platform}.mp4"
            await transcode_video(
                input_video, output_path, platform, info, reframe_action,
                db_pool=db_pool, upload_id=str(ctx.upload_id) if ctx.upload_id else None,
            )

            ctx.platform_videos[platform] = output_path
            transcoded_any = True
        else:
            logger.info(f"{platform}: video is already compatible, using original")
            ctx.platform_videos[platform] = input_video

    # -- Step 3: Set processed video path (backward compat) --
    if transcoded_any:
        first_platform = platforms[0]
        if first_platform in ctx.platform_videos:
            ctx.processed_video_path = ctx.platform_videos[first_platform]

    # Summary log
    summary_parts = []
    for p in platforms:
        vpath = ctx.platform_videos.get(p)
        if vpath and vpath != input_video:
            try:
                sz = vpath.stat().st_size / 1024 / 1024
                summary_parts.append(f"{p}={sz:.1f}MB")
            except Exception:
                summary_parts.append(f"{p}=transcoded")
        else:
            summary_parts.append(f"{p}=original")

    logger.info(f"Transcode summary: {', '.join(summary_parts)}")

    return ctx


# -------------------------------------------------------------------------
# Utility
# -------------------------------------------------------------------------
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
