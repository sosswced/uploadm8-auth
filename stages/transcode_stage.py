"""
UploadM8 Transcode Stage - Platform-specific video transcoding

Creates separate, optimized video files for each target platform.
Each platform gets its own properly formatted MP4 with correct:
  - Aspect ratio (9:16 vertical via pad/crop based on reframe_mode)
  - Resolution (1080x1920 default; 2160x3840 when source is 4K-class and TRANSCODE_PRESERVE_HD_TIER is on)
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

# Cap FFmpeg CPU/RAM per invocation so concurrent jobs don't OOM the box.
# Default scales with WORKER_CONCURRENCY; on Render clamp harder (2 GB workers).
_FFMPEG_THREADS_DEFAULT = max(
    1,
    (os.cpu_count() or 2) // max(1, int(os.environ.get("WORKER_CONCURRENCY", "2"))),
)
if os.environ.get("RENDER") and "FFMPEG_THREADS" not in os.environ:
    _FFMPEG_THREADS_DEFAULT = min(_FFMPEG_THREADS_DEFAULT, 2)
FFMPEG_THREADS = int(os.environ.get("FFMPEG_THREADS", str(_FFMPEG_THREADS_DEFAULT)))

# Encode quality (override without code changes). Preset affects encoder CPU time, not target resolution.
# Hybrid default: preset fast + CRF 19 ≈ medium + CRF 20 visual quality at ~25–35% faster encode (good for scaled workers).
TRANSCODE_X264_CRF = (os.environ.get("TRANSCODE_X264_CRF", "19").strip() or "19")
TRANSCODE_X264_PRESET = (os.environ.get("TRANSCODE_X264_PRESET", "fast").strip() or "fast")

# HD tier preservation: 1080p+ sources stay at 1080x1920 minimum; 4K-class sources use 2160x3840.
# TRANSCODE_YOUTUBE_4K: auto (default) | true | false — auto enables 4K YouTube when source qualifies.
# TRANSCODE_VERTICAL_4K_ALL_PLATFORMS: when true (default), 4K-class sources use 2160x3840 on all vertical platforms.

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .context import JobContext
from .errors import StageError, SkipStage, ErrorCode
from .ffmpeg_env import resolve_ffmpeg_executable

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
        "max_bitrate_video": "12M",
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
        "max_bitrate_video": "16M",
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
        "max_bitrate_video": "10M",
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
        "max_bitrate_video": "12M",
        "max_bitrate_audio": "128k",
        "sample_rate": 44100,
        "pixel_format": "yuv420p",
        "profile": "high",
        "level": "4.1",
        "max_file_mb": 4096,            # 4GB
    },
}

_YOUTUBE_4K_OVERRIDES = {
    "target_width": 2160,
    "target_height": 3840,
    "max_bitrate_video": "40M",
    "profile": "high",
    "level": "5.1",
}

_PLATFORM_4K_VIDEO_BITRATE = {
    "youtube": "40M",
    "tiktok": "20M",
    "instagram": "20M",
    "facebook": "20M",
}

# 1080p-class dashcam / phone footage — higher caps preserve OSD text and road detail.
_PLATFORM_1080_VIDEO_BITRATE = {
    "youtube": "18M",
    "tiktok": "16M",
    "instagram": "14M",
    "facebook": "14M",
}


def _env_truthy(name: str, default: str = "true") -> bool:
    v = (os.environ.get(name) or default).strip().lower()
    return v in ("1", "true", "yes", "on")


def _youtube_4k_mode() -> str:
    return (os.environ.get("TRANSCODE_YOUTUBE_4K") or "auto").strip().lower()


PRESERVE_HD_TIER = _env_truthy("TRANSCODE_PRESERVE_HD_TIER", "true")


def _apply_vertical_4k_spec(base: dict, platform: str) -> dict:
    return {
        **base,
        **_YOUTUBE_4K_OVERRIDES,
        "max_bitrate_video": _PLATFORM_4K_VIDEO_BITRATE.get(platform, "20M"),
    }


def _scale_filter(
    width: int,
    height: int,
    *,
    decrease: bool = False,
    increase: bool = False,
) -> str:
    """Lanczos scale for sharper down/up-scales."""
    flags = "flags=lanczos"
    if decrease:
        return f"scale={width}:{height}:{flags}:force_original_aspect_ratio=decrease"
    if increase:
        return f"scale={width}:{height}:{flags}:force_original_aspect_ratio=increase"
    return f"scale={width}:{height}:{flags}"


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


def source_hd_tier(info: VideoInfo) -> str:
    """Classify source resolution: 4k (2160+ long edge), 1080p, or standard."""
    short = min(info.width, info.height)
    long = max(info.width, info.height)
    if long >= 2160 and short >= 1080:
        return "4k"
    if short >= 1080:
        return "1080p"
    return "standard"


def get_platform_spec(platform: str, info: Optional[VideoInfo] = None) -> dict:
    """Return platform transcode spec with source-aware 1080p / 4K targets and bitrates."""
    base = PLATFORM_SPECS.get(platform) or PLATFORM_SPECS["tiktok"]
    if info is None:
        if platform == "youtube" and _youtube_4k_mode() in ("1", "true", "yes", "on"):
            return {**base, **_YOUTUBE_4K_OVERRIDES}
        return base

    tier = source_hd_tier(info)
    spec = dict(base)

    if PRESERVE_HD_TIER and tier == "1080p":
        boosted = _PLATFORM_1080_VIDEO_BITRATE.get(platform)
        if boosted:
            spec["max_bitrate_video"] = boosted

    if not PRESERVE_HD_TIER:
        return spec

    if tier != "4k":
        return spec

    yt_mode = _youtube_4k_mode()
    use_4k = False
    if platform == "youtube":
        if yt_mode not in ("0", "false", "no", "off"):
            use_4k = True
    elif _env_truthy("TRANSCODE_VERTICAL_4K_ALL_PLATFORMS", "true"):
        use_4k = True

    if use_4k:
        return _apply_vertical_4k_spec(spec, platform)
    return spec


def resolve_x264_encode_params(info: VideoInfo) -> Tuple[str, str, Optional[str]]:
    """
    Source-tier x264 settings: sharper CRF + fast preset for 1080p/4K dashcam-class sources;
    faster preset for smaller phone clips on scaled worker fleets.
    """
    tier = source_hd_tier(info)
    preset = TRANSCODE_X264_PRESET
    crf = TRANSCODE_X264_CRF
    tune_raw = (os.environ.get("TRANSCODE_X264_TUNE") or "").strip()
    tune: Optional[str] = tune_raw or None

    if tier == "4k":
        crf = (os.environ.get("TRANSCODE_X264_CRF_4K") or "18").strip() or "18"
        preset = (os.environ.get("TRANSCODE_X264_PRESET_4K") or preset).strip() or preset
        if not tune and _env_truthy("TRANSCODE_DASHCAM_TUNE_FILM", "true"):
            tune = "film"
    elif tier == "1080p":
        crf = (os.environ.get("TRANSCODE_X264_CRF_1080") or "19").strip() or "19"
        preset = (os.environ.get("TRANSCODE_X264_PRESET_1080") or preset).strip() or preset
        if not tune and _env_truthy("TRANSCODE_DASHCAM_TUNE_FILM", "true"):
            tune = "film"
    else:
        crf = (os.environ.get("TRANSCODE_X264_CRF_STANDARD") or crf).strip() or crf
        preset = (os.environ.get("TRANSCODE_X264_PRESET_STANDARD") or "faster").strip() or "faster"

    return preset, crf, tune


def analyze_transcode_needs(
    info: VideoInfo,
    platform: str,
    reframe_action: str,
) -> Tuple[List[str], List[str]]:
    """Split reasons into video vs audio so we can stream-copy video when only audio needs work."""
    spec = get_platform_spec(platform, info) if platform in PLATFORM_SPECS else None
    if not spec:
        return [], []

    video_reasons: List[str] = []
    audio_reasons: List[str] = []

    if reframe_action in ("pad", "crop"):
        video_reasons.append(f"reframe={reframe_action} to {spec['target_width']}x{spec['target_height']}")

    if info.video_codec.lower() not in ("h264", "avc"):
        video_reasons.append(f"video codec is {info.video_codec}, needs h264")

    if info.pixel_format != "yuv420p":
        video_reasons.append(f"pixel format is {info.pixel_format}, needs yuv420p")

    if reframe_action == "none":
        max_w = spec["target_width"]
        max_h = spec["target_height"]
        if info.is_landscape:
            max_w = max(spec["target_width"], spec["target_height"])
            max_h = min(spec["target_width"], spec["target_height"])
        if info.width > max_w or info.height > max_h:
            video_reasons.append(f"resolution {info.width}x{info.height} exceeds max {max_w}x{max_h}")

    if info.fps > spec["max_fps"]:
        video_reasons.append(f"fps {info.fps:.1f} exceeds max {spec['max_fps']}")

    if info.rotation != 0:
        video_reasons.append(f"has {info.rotation} degree rotation to apply")

    if not info.audio_codec:
        audio_reasons.append("no audio stream — mux silent stereo")
    elif info.audio_codec.lower() != "aac":
        audio_reasons.append(f"audio codec is {info.audio_codec}, needs aac")
    elif info.sample_rate and info.sample_rate != spec["sample_rate"]:
        audio_reasons.append(f"sample rate {info.sample_rate} needs {spec['sample_rate']}")

    if spec["max_duration"] > 0 and info.duration > spec["max_duration"]:
        # -t trim works with stream copy; not a video reencode reason by itself.
        pass

    return video_reasons, audio_reasons


async def get_video_info(video_path: Path) -> VideoInfo:
    """Use ffprobe to get video metadata"""
    cmd = [
        resolve_ffmpeg_executable("ffprobe") or "ffprobe",
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


def needs_transcode(
    info: VideoInfo,
    platform: str,
    reframe_action: str,
) -> Tuple[bool, List[str]]:
    """Check if any transcode/mux work is needed for the target platform."""
    v_reasons, a_reasons = analyze_transcode_needs(info, platform, reframe_action)
    reasons = list(v_reasons) + list(a_reasons)
    spec = get_platform_spec(platform, info) if platform in PLATFORM_SPECS else None
    if spec and spec["max_duration"] > 0 and info.duration > spec["max_duration"]:
        reasons.append(f"duration {info.duration:.1f}s exceeds max {spec['max_duration']}s")
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
    *,
    force_duration_trim_sec: Optional[float] = None,
    tiktok_cover_offset_sec: Optional[float] = None,
    watermark_vf: Optional[str] = None,
) -> list:
    """Build platform-specific FFmpeg command"""
    spec = get_platform_spec(platform, info)

    needs_silent_audio = not info.audio_codec

    ff = resolve_ffmpeg_executable() or "ffmpeg"
    cmd = [
        ff,
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
        vf_filters.append(_scale_filter(tw, th, decrease=True))
        vf_filters.append(f"pad={tw}:{th}:(ow-iw)/2:(oh-ih)/2")

    elif reframe_action == "crop":
        vf_filters.append(_scale_filter(tw, th, increase=True))
        vf_filters.append(f"crop={tw}:{th}")

    elif reframe_action == "none":
        max_w = spec["target_width"]
        max_h = spec["target_height"]

        if info.is_landscape:
            max_w = max(spec["target_width"], spec["target_height"])
            max_h = min(spec["target_width"], spec["target_height"])

        if info.width > max_w or info.height > max_h:
            vf_filters.append(_scale_filter(max_w, max_h, decrease=True))

        vf_filters.append("pad=ceil(iw/2)*2:ceil(ih/2)*2")

    if watermark_vf:
        vf_filters.append(watermark_vf)

    vf_filters.append("format=yuv420p")

    v_reasons, a_reasons = analyze_transcode_needs(info, platform, reframe_action)
    trim_needed = (
        force_duration_trim_sec is not None
        and force_duration_trim_sec > 0
        and info.duration > force_duration_trim_sec
    ) or (
        spec["max_duration"] > 0 and info.duration > spec["max_duration"]
    )
    video_copy = (
        not v_reasons
        and not needs_silent_audio
        and (bool(a_reasons) or trim_needed)
    )

    if vf_filters and not video_copy:
        cmd.extend(["-vf", ",".join(vf_filters)])

    # -- Video codec settings --
    if video_copy:
        cmd.extend(["-c:v", "copy"])
        if a_reasons:
            logger.info(f"{platform}: stream-copying video (audio re-encode only)")
        else:
            logger.info(f"{platform}: stream-copying video and audio (trim/mux only)")
    else:
        x264_preset, x264_crf, x264_tune = resolve_x264_encode_params(info)
        cmd.extend([
            "-c:v", "libx264",
            "-threads", str(FFMPEG_THREADS),
            "-preset", x264_preset,
            "-crf", x264_crf,
            "-profile:v", spec.get("profile", "high"),
            "-level", spec.get("level", "4.1"),
            "-maxrate", spec["max_bitrate_video"],
            "-bufsize", str(int(spec["max_bitrate_video"].replace("M", "")) * 2) + "M",
        ])
        if x264_tune:
            cmd.extend(["-tune", x264_tune])

    if platform == "tiktok" and tiktok_cover_offset_sec is not None and not video_copy:
        from stages.tiktok_cover_burn import extend_tiktok_transcode_x264_args

        extend_tiktok_transcode_x264_args(
            cmd,
            cover_offset_sec=float(tiktok_cover_offset_sec),
            fps=float(info.fps or 30.0),
        )

    # -- Frame rate capping (re-encode only) --
    if not video_copy and info.fps > spec["max_fps"]:
        cmd.extend(["-r", str(spec["max_fps"])])

    # -- Duration trimming (works with stream copy) --
    trim_to: Optional[float] = None
    if force_duration_trim_sec and force_duration_trim_sec > 0 and info.duration > force_duration_trim_sec:
        trim_to = max(1.0, force_duration_trim_sec - 0.5)
    elif spec["max_duration"] > 0 and info.duration > spec["max_duration"]:
        trim_to = max(1.0, spec["max_duration"] - 0.5)
    if trim_to is not None:
        cmd.extend(["-t", f"{trim_to:.3f}"])

    # -- Audio settings --
    # The anullsrc input (input index 1) was already declared at the top of the
    # command when needs_silent_audio is True.  We now either encode the real
    # audio stream or map+encode the synthetic silent stream.
    if not needs_silent_audio:
        if not a_reasons:
            cmd.extend(["-c:a", "copy"])
        else:
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
    out_tail = ["-movflags", "+faststart", str(output_path)]
    if not video_copy:
        out_tail = ["-pix_fmt", "yuv420p"] + out_tail
    cmd.extend(out_tail)

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
    *,
    force_duration_trim_sec: Optional[float] = None,
    tiktok_cover_offset_sec: Optional[float] = None,
    watermark_vf: Optional[str] = None,
) -> Path:
    """Transcode a video to a platform-specific format.
    
    When db_pool and upload_id are provided, streams FFmpeg stderr in real-time
    and writes progress (0-100) to the uploads table every ~2 seconds so the
    frontend can display a live progress bar.
    """
    cmd = build_ffmpeg_command(
        input_path,
        output_path,
        info,
        platform,
        reframe_action,
        force_duration_trim_sec=force_duration_trim_sec,
        tiktok_cover_offset_sec=tiktok_cover_offset_sec,
        watermark_vf=watermark_vf,
    )

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
        spec = get_platform_spec(platform, info)
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
    # Use watermarked source if watermark stage ran before transcode (legacy path).
    # WATERMARK_SINGLE_PASS burns drawtext inside each platform transcode instead.
    input_video = ctx.local_video_path
    watermark_vf: Optional[str] = None
    single_pass_wm = bool(getattr(ctx, "watermark_single_pass", False))
    if (
        single_pass_wm
        and ctx.entitlements
        and ctx.entitlements.can_watermark
    ):
        try:
            from stages.watermark_stage import build_watermark_vf_for_transcode

            watermark_vf = await build_watermark_vf_for_transcode(ctx, input_video)
            if watermark_vf:
                logger.info("[%s] WATERMARK_SINGLE_PASS: burn-in during transcode", ctx.upload_id)
        except Exception as e:
            logger.warning("[%s] single-pass watermark filter skipped: %s", ctx.upload_id, e)
    elif ctx.processed_video_path and ctx.processed_video_path.exists():
        input_video = ctx.processed_video_path
        logger.info(f"Using processed video as transcode input: {input_video.name}")

    info = await get_video_info(input_video)

    orientation = "landscape" if info.is_landscape else ("portrait" if info.is_portrait else "square")
    file_mb = info.file_size / 1024 / 1024 if info.file_size else 0

    logger.info(
        f"Video: {info.width}x{info.height}, {info.duration:.1f}s, {info.fps:.1f}fps, "
        f"codec={info.video_codec}, pix_fmt={info.pixel_format}, rotation={info.rotation}, "
        f"file_size={file_mb:.1f}MB, hd_tier={source_hd_tier(info)}"
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
        "hd_tier": source_hd_tier(info),
    }

    # -- Step 2: Per-platform transcode --
    ctx.platform_videos = {}
    transcoded_any = False

    for platform in platforms:
        if platform not in PLATFORM_SPECS:
            logger.warning(f"Unknown platform '{platform}', skipping transcode")
            ctx.platform_videos[platform] = ctx.local_video_path
            continue

        spec = get_platform_spec(platform, info)

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
            v_r, a_r = analyze_transcode_needs(info, platform, reframe_action)
            trim_only = not v_r and not a_r
            if trim_only:
                logger.info(f"{platform}: stream-copy trim/mux (preserving source pixels)")
            elif v_r:
                preset, crf, tune = resolve_x264_encode_params(info)
                logger.info(
                    f"{platform} encode profile: preset={preset} crf={crf} tune={tune or 'default'} "
                    f"hd_tier={source_hd_tier(info)} maxrate={spec.get('max_bitrate_video')}"
                )

            output_path = ctx.temp_dir / f"transcoded_{platform}.mp4"
            tiktok_cover_off = None
            if platform == "tiktok":
                from stages.tiktok_cover_burn import resolve_tiktok_cover_offset_sec

                tiktok_cover_off = resolve_tiktok_cover_offset_sec(ctx.output_artifacts)
            await transcode_video(
                input_video, output_path, platform, info, reframe_action,
                db_pool=db_pool, upload_id=str(ctx.upload_id) if ctx.upload_id else None,
                tiktok_cover_offset_sec=tiktok_cover_off,
                watermark_vf=watermark_vf,
            )

            ctx.platform_videos[platform] = output_path
            transcoded_any = True
            if platform == "tiktok" and tiktok_cover_off is not None:
                from stages.tiktok_cover_burn import store_tiktok_transcode_keyframe_artifacts

                store_tiktok_transcode_keyframe_artifacts(
                    ctx.output_artifacts,
                    offset_sec=tiktok_cover_off,
                    fps=float(info.fps or 30.0),
                    duration_sec=float(info.duration or 0.0),
                )
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
    exe = resolve_ffmpeg_executable() or "ffmpeg"
    try:
        proc = await asyncio.create_subprocess_exec(
            exe, "-version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await proc.communicate()
        return proc.returncode == 0
    except Exception:
        return False
