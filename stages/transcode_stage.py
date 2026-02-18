"""
UploadM8 Transcode Stage
=========================
Converts videos to platform-specific formats using FFmpeg.

Ensures videos meet the requirements for:
- YouTube Shorts: H.264, AAC, 9:16 or 1:1, max 3 minutes  (LANDSCAPE REJECTED)
- TikTok: H.264 High, AAC-LC 256k/44.1kHz, max 10 min upload
- Instagram Reels: H.264, AAC, max 3 minutes (15 min via API)
- Facebook Reels: H.264, AAC, 3-90s (all videos = Reels since Jun 2025)

Exports:
  - run_transcode_stage(ctx)
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from .context import JobContext
from .errors import StageError, SkipStage, ErrorCode

logger = logging.getLogger("uploadm8-worker")

FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg")
FFPROBE_BIN = os.environ.get("FFPROBE_BIN", "ffprobe")

# ─────────────────────────────────────────────────────────────────────
# Platform-specific requirements  (last verified Feb 2026)
#
# Sources:
#   YouTube  – support.google.com/youtube/answer/15424877  (3-min Shorts since Oct 2024)
#   TikTok   – developers.tiktok.com  Content Posting API + partner docs (Sep 2025)
#   Instagram – help.instagram.com/1038071743007909  (3-min Reels since Jan 2025)
#   Facebook – postfa.st/sizes/facebook/reels  (all videos = Reels since Jun 2025)
#
# IMPORTANT — landscape_allowed:
#   YouTube Shorts REQUIRES vertical (9:16) or square (1:1) aspect ratio.
#   A landscape video will NOT qualify as a Short; it becomes a regular
#   YouTube upload.  TikTok, Instagram, and Facebook accept landscape
#   but strongly prefer vertical.  The flag controls whether the
#   transcoder should warn/block landscape uploads for a given platform.
#
# max_file_size is in bytes — the pipeline should reject files that
# exceed this BEFORE transcoding to avoid wasted compute.
# ─────────────────────────────────────────────────────────────────────

PLATFORM_SPECS = {
    "youtube": {
        "name": "YouTube Shorts",
        "max_duration": 180,              # 3 minutes (Oct 2024 update)
        "preferred_aspect": (9, 16),
        "landscape_allowed": False,       # Landscape → regular upload, NOT a Short
        "video_codec": "h264",
        "audio_codec": "aac",
        "max_width": 1080,
        "max_height": 1920,
        "max_fps": 60,
        "max_bitrate_video": "15M",       # YT compression is aggressive; higher source = better
        "max_bitrate_audio": "192k",
        "sample_rate": 48000,
        "pixel_format": "yuv420p",
        "max_file_size": 2 * 1024**3,     # 2 GB
        "container": "mp4",
    },
    "tiktok": {
        "name": "TikTok",
        "max_duration": 600,              # 10 min in-app record; 60 min upload (API safe: 10 min)
        "preferred_aspect": (9, 16),
        "landscape_allowed": True,        # Accepted, but vertical strongly preferred by algorithm
        "video_codec": "h264",            # H.264 High Profile, level 4.2
        "audio_codec": "aac",             # AAC-LC
        "max_width": 1080,
        "max_height": 1920,
        "max_fps": 60,                    # Supported; 30 fps recommended for less compression
        "max_bitrate_video": "15M",       # 8-15 Mbps VBR recommended; >20 gets flattened
        "max_bitrate_audio": "256k",      # AAC-LC 256 kbps per partner docs
        "sample_rate": 44100,             # 44.1 kHz confirmed
        "pixel_format": "yuv420p",
        "max_file_size": 500 * 1024**2,   # 500 MB (<3 min); 2 GB for 3-10 min
        "max_file_size_long": 2 * 1024**3,  # For videos > 3 min
        "container": "mp4",
    },
    "instagram": {
        "name": "Instagram Reels",
        "max_duration": 180,              # 3 minutes (Jan 2025 update); up to 15 min via API
        "preferred_aspect": (9, 16),
        "landscape_allowed": True,        # Accepted, but gets letterboxed
        "video_codec": "h264",
        "audio_codec": "aac",
        "max_width": 1080,
        "max_height": 1920,
        "max_fps": 60,                    # 30 fps minimum, 60 supported
        "max_bitrate_video": "10M",       # IG compression heavy; 10M is sweet spot
        "max_bitrate_audio": "192k",
        "sample_rate": 44100,
        "pixel_format": "yuv420p",
        "max_file_size": 650 * 1024**2,   # 650 MB for <10 min; 4 GB absolute max
        "container": "mp4",
    },
    "facebook": {
        "name": "Facebook Reels",
        "max_duration": 90,               # API still enforces 3-90s for Reels endpoint
        "preferred_aspect": (9, 16),
        "landscape_allowed": True,        # All videos are Reels since Jun 2025
        "video_codec": "h264",
        "audio_codec": "aac",
        "max_width": 1080,
        "max_height": 1920,
        "max_fps": 60,                    # 24-60 fps per Meta docs
        "max_bitrate_video": "10M",
        "max_bitrate_audio": "192k",
        "sample_rate": 44100,
        "pixel_format": "yuv420p",
        "max_file_size": 1 * 1024**3,     # 1 GB
        "container": "mp4",
    },
}


@dataclass
class VideoInfo:
    """Parsed video metadata from ffprobe."""
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
    """Use ffprobe to get video metadata."""
    cmd = [
        FFPROBE_BIN,
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            raise StageError(ErrorCode.TRANSCODE_FAILED, f"ffprobe failed: {stderr.decode()}")

        data = json.loads(stdout.decode())

        video_stream = None
        audio_stream = None
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video" and not video_stream:
                video_stream = stream
            elif stream.get("codec_type") == "audio" and not audio_stream:
                audio_stream = stream

        if not video_stream:
            raise StageError(ErrorCode.TRANSCODE_FAILED, "No video stream found")

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

        if rotation in (90, 270, -90, -270):
            width, height = height, width

        fps_str = video_stream.get("r_frame_rate", "30/1")
        if "/" in fps_str:
            num, den = fps_str.split("/")
            fps = float(num) / float(den) if float(den) > 0 else 30.0
        else:
            fps = float(fps_str)

        duration = float(data.get("format", {}).get("duration", 0))
        if duration == 0:
            duration = float(video_stream.get("duration", 0))

        video_bitrate = int(video_stream.get("bit_rate", 0)) if video_stream.get("bit_rate") else None
        audio_bitrate = int(audio_stream.get("bit_rate", 0)) if audio_stream and audio_stream.get("bit_rate") else None
        sample_rate = int(audio_stream.get("sample_rate", 0)) if audio_stream and audio_stream.get("sample_rate") else None

        return VideoInfo(
            width=width, height=height, duration=duration, fps=fps,
            video_codec=video_stream.get("codec_name", "unknown"),
            audio_codec=audio_stream.get("codec_name") if audio_stream else None,
            video_bitrate=video_bitrate, audio_bitrate=audio_bitrate,
            sample_rate=sample_rate,
            pixel_format=video_stream.get("pix_fmt", "unknown"),
            rotation=rotation,
        )

    except json.JSONDecodeError as e:
        raise StageError(ErrorCode.TRANSCODE_FAILED, f"Failed to parse ffprobe output: {e}")
    except StageError:
        raise
    except Exception as e:
        raise StageError(ErrorCode.TRANSCODE_FAILED, f"Failed to get video info: {e}")


def check_platform_blocked(info: VideoInfo, platform: str, reframe_mode: str = "none") -> Optional[str]:
    """Check for hard blockers that transcoding CANNOT fix.

    Returns a human-readable reason string if the video is fundamentally
    incompatible with the platform, or None if it can proceed.

    If reframe_mode is "blur_fill" or "center_crop", landscape videos are
    NOT blocked — they'll be reframed to vertical in build_ffmpeg_command().
    """
    spec = PLATFORM_SPECS.get(platform)
    if not spec:
        return None

    is_landscape = info.width > info.height

    # YouTube Shorts REQUIRES vertical or square — landscape becomes a regular upload
    # UNLESS the user opted into vertical reframing
    if is_landscape and not spec.get("landscape_allowed", True):
        if reframe_mode in ("blur_fill", "center_crop", "pad", "auto"):
            return None  # Reframe will handle it
        return (
            f"{spec['name']} requires vertical (9:16) or square (1:1) video. "
            f"Your {info.width}x{info.height} landscape video would be uploaded "
            f"as a regular YouTube video, NOT a Short. "
            f"Enable vertical reframe (blur_fill or center_crop) to auto-convert."
        )

    return None


# ─────────────────────────────────────────────────────────────────────
# Vertical reframe modes
#
# When a landscape video targets a vertical-only platform (YouTube Shorts),
# the transcoder can reframe it to 9:16 using one of these strategies:
#
#   blur_fill    – Full video centered in a 1080x1920 frame, with a
#                  blurred + zoomed copy filling the top/bottom gaps.
#                  No content lost. Looks native on Shorts/TikTok.
#                  ┌───────────────┐
#                  │ ░░░ BLUR ░░░░ │
#                  │ ┌───────────┐ │
#                  │ │ FULL VID  │ │
#                  │ └───────────┘ │
#                  │ ░░░ BLUR ░░░░ │
#                  └───────────────┘
#
#   center_crop  – Crops the center 9:16 slice out of the 16:9 frame.
#                  Loses ~65% of the frame width. Only useful if the
#                  subject is centered (talking head, not dashcam).
#                  ┌───────────────┐
#                  │               │
#                  │   CROPPED     │
#                  │   CENTER      │
#                  │               │
#                  └───────────────┘
#
#   none         – No reframe. Landscape videos are blocked on platforms
#                  that require vertical (YouTube Shorts).
# ─────────────────────────────────────────────────────────────────────

# "pad" = 1080x1920 canvas with black bars (no blur)
# "auto" = platform policy: pad for vertical platforms if landscape; YouTube also trims to 59.5s
VALID_REFRAME_MODES = ("none", "pad", "blur_fill", "center_crop", "auto")

YOUTUBE_SHORTS_TRIM_SECONDS = 59.5


def _build_blur_fill_filter(info: VideoInfo, target_w: int = 1080, target_h: int = 1920) -> str:
    """Build an FFmpeg filtergraph that reframes landscape → vertical with blur-fill.

    The filter creates two layers:
      1. Background: input scaled UP to cover the full target canvas,
         center-cropped, then heavily blurred.
      2. Foreground: input scaled DOWN to fit within target_w, keeping
         its original aspect ratio.
    The foreground is overlaid centered on the background.

    Example for 1920x1080 → 1080x1920:
      BG: scale to ~3413x1920 (covers canvas), crop 1080x1920, blur
      FG: scale to 1080x608 (fits width, maintains 16:9)
      Overlay: FG centered at y=(1920-608)/2 = 656
    """
    # Background: scale to cover, crop center, blur hard
    bg = (
        f"scale={target_w}:{target_h}:"
        f"force_original_aspect_ratio=increase,"
        f"crop={target_w}:{target_h},"
        f"boxblur=luma_radius=25:luma_power=2"
    )

    # Foreground: scale to fit width, auto-height (keep aspect, force even)
    fg = f"scale={target_w}:-2"

    # Combine with complex filtergraph
    filtergraph = (
        f"[0:v]{bg}[bg];"
        f"[0:v]{fg}[fg];"
        f"[bg][fg]overlay=0:(H-h)/2,format=yuv420p"
    )

    return filtergraph


def _build_center_crop_filter(info: VideoInfo, target_w: int = 1080, target_h: int = 1920) -> str:
    """Build an FFmpeg filtergraph that center-crops landscape → vertical 9:16.

    Crops the center vertical slice, then scales to target resolution.
    WARNING: For dashcam footage this loses most of the frame — blur_fill
    is almost always a better choice.
    """
    # From the input, crop a 9:16 slice centered horizontally
    # crop_width = input_height * (9/16)
    crop_w = int(info.height * target_w / target_h)
    crop_w -= crop_w % 2  # Even width for h264

    filtergraph = (
        f"crop={crop_w}:{info.height}:(iw-{crop_w})/2:0,"
        f"scale={target_w}:{target_h},"
        f"format=yuv420p"
    )

    return filtergraph


def _build_pad_filter(target_w: int = 1080, target_h: int = 1920) -> str:
    """
    Pad to a fixed vertical canvas (1080x1920) while preserving aspect ratio.
    This is the "letterbox to vertical" strategy (no blur, no crop).
    """
    return (
        f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
        f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2,"
        f"format=yuv420p"
    )


def needs_transcode(info: VideoInfo, platform: str) -> Tuple[bool, list]:
    """Check if video needs transcoding for the target platform.

    Resolution check is orientation-aware: a 1920x1080 landscape video
    is fine for platforms that list max_width=1080, max_height=1920
    (those are portrait-preferred dimensions, not hard per-axis limits).

    Note: call check_platform_blocked() FIRST to catch hard blockers
    that transcoding cannot fix (e.g. landscape on YouTube Shorts).
    """
    spec = PLATFORM_SPECS.get(platform)
    if not spec:
        return False, []

    reasons = []

    if info.video_codec.lower() not in ("h264", "avc"):
        reasons.append(f"video codec is {info.video_codec}, needs h264")
    if info.audio_codec and info.audio_codec.lower() != "aac":
        reasons.append(f"audio codec is {info.audio_codec}, needs aac")
    if info.pixel_format != "yuv420p":
        reasons.append(f"pixel format is {info.pixel_format}, needs yuv420p")

    # Orientation-aware resolution check:
    # max_long_side = larger of the two spec dimensions (1920)
    # max_short_side = smaller of the two spec dimensions (1080)
    # Compare against the input's long/short sides regardless of orientation
    max_long = max(spec["max_width"], spec["max_height"])
    max_short = min(spec["max_width"], spec["max_height"])
    input_long = max(info.width, info.height)
    input_short = min(info.width, info.height)

    if input_long > max_long or input_short > max_short:
        reasons.append(
            f"resolution {info.width}x{info.height} exceeds "
            f"{max_short}x{max_long} bounding box"
        )

    if info.fps > spec["max_fps"]:
        reasons.append(f"fps {info.fps:.1f} exceeds max {spec['max_fps']}")
    if info.duration > spec["max_duration"]:
        reasons.append(f"duration {info.duration:.1f}s exceeds max {spec['max_duration']}s")
    if info.rotation != 0:
        reasons.append(f"video has {info.rotation}° rotation")

    return len(reasons) > 0, reasons


def build_ffmpeg_command(
    input_path: Path,
    output_path: Path,
    info: VideoInfo,
    platform: str,
    reframe_mode: str = "none",
    trim_seconds: Optional[float] = None,
) -> list:
    """Build FFmpeg command for transcoding.

    Scaling logic:
    - Orientation-aware: matches the input's landscape/portrait orientation
      against the platform's max bounding box.
    - Proportional: scales both axes by the same factor to preserve aspect ratio.
    - No forced padding: platforms handle letterboxing in their own players.
      Baking black bars into the video file makes it look bad everywhere.

    Reframe modes:
    - "blur_fill": blurred background + sharp center (recommended for dashcam)
    - "center_crop": center 9:16 slice (loses ~65% width)
    - "pad": letterbox into a fixed 1080x1920 canvas (black bars)
    - "auto": policy-driven (applied in run_transcode_stage)
    - "none": no reframe, use normal scaling
    """
    spec = PLATFORM_SPECS.get(platform, PLATFORM_SPECS["youtube"])

    cmd = [FFMPEG_BIN, "-y", "-i", str(input_path)]

    is_landscape = info.width > info.height

    # Effective target canvas for vertical reframes
    target_w = min(spec["max_width"], spec["max_height"])   # 1080
    target_h = max(spec["max_width"], spec["max_height"])   # 1920

    # Reframe strategy selection:
    # - blur_fill / center_crop = only applied when platform disallows landscape
    # - pad = explicit; can be used even when the platform accepts landscape
    use_blur_or_crop = (
        reframe_mode in ("blur_fill", "center_crop")
        and is_landscape
        and not spec.get("landscape_allowed", True)
    )
    use_pad = (reframe_mode == "pad" and is_landscape)

    if use_blur_or_crop:
        # ── Vertical reframe path (blur_fill/center_crop) ────────────
        # Uses a complex filtergraph (-filter_complex) instead of -vf
        # because blur_fill needs to split the input into two streams.
        if reframe_mode == "blur_fill":
            filtergraph = _build_blur_fill_filter(info, target_w, target_h)
            logger.info(
                f"Reframing {info.width}x{info.height} → {target_w}x{target_h} "
                f"via blur_fill for {platform}"
            )
        else:
            filtergraph = _build_center_crop_filter(info, target_w, target_h)
            logger.info(
                f"Reframing {info.width}x{info.height} → {target_w}x{target_h} "
                f"via center_crop for {platform}"
            )

        # Handle rotation before reframe
        if info.rotation == 90:
            filtergraph = filtergraph.replace("[0:v]", "[0:v]transpose=1,", 1)
        elif info.rotation == 180:
            filtergraph = filtergraph.replace("[0:v]", "[0:v]transpose=1,transpose=1,", 1)
        elif info.rotation in (270, -90):
            filtergraph = filtergraph.replace("[0:v]", "[0:v]transpose=2,", 1)

        cmd.extend(["-filter_complex", filtergraph])

    else:
        # ── Normal / pad scaling path ────────────────────────────────
        vf_filters = []

        # Rotation
        if info.rotation == 90:
            vf_filters.append("transpose=1")
        elif info.rotation == 180:
            vf_filters.append("transpose=1,transpose=1")
        elif info.rotation in (270, -90):
            vf_filters.append("transpose=2")

        if use_pad:
            # Force vertical canvas output (1080x1920) with padding
            vf_filters.append(_build_pad_filter(target_w, target_h))
        else:
            # Orientation-aware bounding box (no forced padding)
            max_long = max(spec["max_width"], spec["max_height"])    # 1920
            max_short = min(spec["max_width"], spec["max_height"])   # 1080

            if is_landscape:
                box_w, box_h = max_long, max_short     # 1920 x 1080 for landscape
            else:
                box_w, box_h = max_short, max_long     # 1080 x 1920 for portrait

            # Proportional scale factor — shrink only, never upscale
            scale_w = box_w / info.width if info.width > box_w else 1.0
            scale_h = box_h / info.height if info.height > box_h else 1.0
            scale_factor = min(scale_w, scale_h)

            target_width = int(info.width * scale_factor)
            target_height = int(info.height * scale_factor)
            target_width -= target_width % 2
            target_height -= target_height % 2

            needs_scale = (target_width != info.width or target_height != info.height)

            if needs_scale:
                vf_filters.append(f"scale={target_width}:{target_height}")

            vf_filters.append("format=yuv420p")

        if vf_filters:
            cmd.extend(["-vf", ",".join(vf_filters)])

    # Video codec
    cmd.extend([
        "-c:v", "libx264", "-preset", "medium", "-crf", "23",
        "-profile:v", "high", "-level", "4.1",
        "-maxrate", spec["max_bitrate_video"],
        "-bufsize", spec["max_bitrate_video"].replace("M", "") + "M",
    ])

    if info.fps > spec["max_fps"]:
        cmd.extend(["-r", str(spec["max_fps"])])

    # Duration trim: platform max_duration OR explicit trim_seconds (used by auto policy for YouTube Shorts)
    effective_trim = None
    if trim_seconds is not None:
        effective_trim = float(trim_seconds)
    elif info.duration > spec["max_duration"]:
        effective_trim = float(spec["max_duration"])

    if effective_trim is not None and info.duration > effective_trim:
        cmd.extend(["-t", f"{effective_trim:.3f}"])

    # Audio
    if info.audio_codec:
        cmd.extend([
            "-c:a", "aac", "-b:a", spec["max_bitrate_audio"],
            "-ar", str(spec["sample_rate"]), "-ac", "2",
        ])
    else:
        cmd.extend([
            "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
            "-shortest", "-c:a", "aac", "-b:a", "128k",
        ])

    cmd.extend(["-movflags", "+faststart", "-pix_fmt", "yuv420p", str(output_path)])
    return cmd


async def transcode_video(
    input_path: Path,
    output_path: Path,
    platform: str,
    info: Optional[VideoInfo] = None,
    reframe_mode: str = "none",
    trim_seconds: Optional[float] = None,
) -> Path:
    """Transcode video to platform-specific format."""
    if not info:
        info = await get_video_info(input_path)

    cmd = build_ffmpeg_command(input_path, output_path, info, platform, reframe_mode, trim_seconds=trim_seconds)
    logger.info(f"Transcoding for {platform}: {info.width}x{info.height} -> "
                f"{output_path.name}, reframe={reframe_mode}, "
                f"cmd starts: {' '.join(cmd[:12])}...")

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            error_msg = stderr.decode()[-500:]
            raise StageError(ErrorCode.TRANSCODE_FAILED, f"FFmpeg failed: {error_msg}")

        if not output_path.exists():
            raise StageError(ErrorCode.TRANSCODE_FAILED, "Output file not created")

        logger.info(f"Transcode complete: {output_path.stat().st_size / 1024 / 1024:.1f}MB")
        return output_path

    except StageError:
        raise
    except Exception as e:
        raise StageError(ErrorCode.TRANSCODE_FAILED, f"Transcode failed: {e}")


async def run_transcode_stage(ctx: JobContext) -> JobContext:
    """Run the transcode stage — creates platform-specific versions of the video.

    Pipeline:
      1. Analyze input video with ffprobe
      2. Check file size against platform limits
      3. Check for hard blockers (respects reframe_mode setting)
      4. Determine which platforms need transcoding or reframing
      5. Create optimized versions for each platform
      6. Verify output file sizes are within platform limits

    Reframe mode is read from ctx.reframe_mode (set by the API/frontend).
    Valid values: "auto", "pad", "blur_fill", "center_crop", "none" (default).
    When set, landscape videos can be auto-converted to vertical for
    platforms that require it (YouTube Shorts).
    """
    ctx.mark_stage("transcode")

    if not ctx.local_video_path or not ctx.local_video_path.exists():
        raise SkipStage("No video file to transcode")

    platforms = ctx.platforms or []
    if not platforms:
        raise SkipStage("No target platforms specified")

    # Read reframe preference from job context (set by user at upload time)
    reframe_mode = getattr(ctx, "reframe_mode", "none") or "none"
    if reframe_mode not in VALID_REFRAME_MODES:
        logger.warning(f"Invalid reframe_mode '{reframe_mode}', falling back to 'none'")
        reframe_mode = "none"

    logger.info(f"Analyzing video for platforms: {platforms}, reframe_mode={reframe_mode}")

    info = await get_video_info(ctx.local_video_path)
    input_file_size = ctx.local_video_path.stat().st_size
    logger.info(
        f"Video: {info.width}x{info.height}, {info.duration:.1f}s, {info.fps:.1f}fps, "
        f"codec={info.video_codec}, pix_fmt={info.pixel_format}, rotation={info.rotation}, "
        f"file_size={input_file_size / 1024 / 1024:.1f}MB"
    )

    ctx.video_info = {
        "width": info.width, "height": info.height,
        "duration": info.duration, "fps": info.fps,
        "video_codec": info.video_codec, "audio_codec": info.audio_codec,
        "orientation": "landscape" if info.width > info.height else (
            "portrait" if info.height > info.width else "square"
        ),
    }

    ctx.platform_videos = {}
    ctx.platform_blocked = {}       # Platforms that can't proceed (hard blockers)
    ctx.platform_warnings = {}      # Non-blocking warnings (e.g. landscape on TikTok)
    ctx.platform_reframed = {}      # Platforms where reframe was applied
    transcoded_any = False

    is_landscape = info.width > info.height
    user_reframe_mode = reframe_mode  # preserve original user intent

    for platform in platforms:
        spec = PLATFORM_SPECS.get(platform, {})

        # ── Hard blocker check (respects reframe_mode) ──────────────
        blocked_reason = check_platform_blocked(info, platform, reframe_mode)
        if blocked_reason:
            logger.warning(f"{platform} BLOCKED: {blocked_reason}")
            ctx.platform_blocked[platform] = blocked_reason
            continue

        # ── Determine if this platform needs a reframe / trim policy ─────────
        platform_reframe = "none"
        platform_trim_seconds: Optional[float] = None

        if user_reframe_mode == "auto":
            # Auto rules:
            # - TikTok/IG/FB/YouTube: pad to 1080x1920 if landscape
            # - YouTube: also trim to 59.5s (Shorts-safe)
            if is_landscape and platform in ("tiktok", "instagram", "facebook", "youtube"):
                platform_reframe = "pad"
                ctx.platform_reframed[platform] = "pad"
                logger.info(f"{platform}: auto mode → pad to {min(spec.get('max_width',1080), spec.get('max_height',1920))}x{max(spec.get('max_width',1080), spec.get('max_height',1920))}")

            if platform == "youtube":
                platform_trim_seconds = YOUTUBE_SHORTS_TRIM_SECONDS

        else:
            # Manual behavior (existing): only reframe when platform disallows landscape
            if is_landscape and not spec.get("landscape_allowed", True) and user_reframe_mode != "none":
                platform_reframe = user_reframe_mode
                ctx.platform_reframed[platform] = user_reframe_mode
                logger.info(
                    f"{platform}: landscape video will be reframed to vertical "
                    f"via {user_reframe_mode}"
                )

        # ── Landscape warning (for platforms that accept but don't prefer it)
        if is_landscape and spec.get("landscape_allowed", True) and spec.get("preferred_aspect") == (9, 16):
            warning = (
                f"Video is landscape {info.width}x{info.height}; "
                f"{spec.get('name', platform)} prefers vertical 9:16. "
                f"Video will be letterboxed by the platform."
            )
            logger.info(f"{platform} WARNING: {warning}")
            ctx.platform_warnings[platform] = warning

        # ── Input file size check ───────────────────────────────────
        max_size = spec.get("max_file_size")
        if platform == "tiktok" and info.duration > 180:
            max_size = spec.get("max_file_size_long", max_size)

        if max_size and input_file_size > max_size:
            logger.info(
                f"{platform}: input file {input_file_size / 1024 / 1024:.0f}MB "
                f"exceeds {max_size / 1024 / 1024:.0f}MB limit — will transcode to compress"
            )

        # ── Transcode / reframe check ───────────────────────────────
        needs_tc, reasons = needs_transcode(info, platform)

        # Auto-mode YouTube Shorts trim triggers a transcode even if within platform max_duration
        if platform == "youtube" and platform_trim_seconds is not None and info.duration > platform_trim_seconds:
            if not needs_tc:
                needs_tc = True
            reasons.append(f"trim to {platform_trim_seconds:.1f}s for Shorts-safe upload")

        # Force transcode if reframe is needed (even if video is otherwise compatible)
        if platform_reframe != "none" and not needs_tc:
            needs_tc = True
            reasons.append(f"landscape → vertical reframe ({platform_reframe})")

        if needs_tc:
            logger.info(f"{platform} needs transcode: {', '.join(reasons)}")
            output_path = ctx.temp_dir / f"transcoded_{platform}.mp4"
            await transcode_video(
                ctx.local_video_path,
                output_path,
                platform,
                info,
                platform_reframe,
                trim_seconds=platform_trim_seconds,
            )

            # Verify output size is within platform limit
            output_size = output_path.stat().st_size
            if max_size and output_size > max_size:
                logger.warning(
                    f"{platform}: transcoded file {output_size / 1024 / 1024:.0f}MB "
                    f"still exceeds {max_size / 1024 / 1024:.0f}MB limit"
                )

            ctx.platform_videos[platform] = output_path
            transcoded_any = True
        else:
            logger.info(f"{platform}: video is already compatible")
            ctx.platform_videos[platform] = ctx.local_video_path

    # If ALL platforms were blocked, raise an error
    if not ctx.platform_videos:
        blocked_summary = "; ".join(
            f"{p}: {r}" for p, r in ctx.platform_blocked.items()
        )
        raise StageError(
            ErrorCode.TRANSCODE_FAILED,
            f"No compatible platforms: {blocked_summary}"
        )

    if ctx.platform_blocked:
        logger.warning(
            f"Skipped {len(ctx.platform_blocked)} blocked platform(s): "
            f"{', '.join(ctx.platform_blocked.keys())}"
        )

    if ctx.platform_reframed:
        logger.info(
            f"Reframed for {len(ctx.platform_reframed)} platform(s): "
            f"{', '.join(f'{p} ({m})' for p, m in ctx.platform_reframed.items())}"
        )

    if transcoded_any:
        for p in platforms:
            if p in ctx.platform_videos:
                ctx.processed_video_path = ctx.platform_videos[p]
                break

    return ctx


async def check_ffmpeg_available() -> bool:
    """Check if FFmpeg is installed and accessible."""
    try:
        proc = await asyncio.create_subprocess_exec(
            FFMPEG_BIN, "-version",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
        return proc.returncode == 0
    except Exception:
        return False
