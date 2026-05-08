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

from .errors import SkipStage
from .context import JobContext
from .ffmpeg_env import resolve_ffmpeg_executable

logger = logging.getLogger("uploadm8-worker")

# Fallback when ctx.watermark_text is unset (env override for worker-only deploys)
WATERMARK_TEXT = os.environ.get("WATERMARK_TEXT", "Upload M8")
WATERMARK_FONT_SIZE = int(os.environ.get("WATERMARK_FONT_SIZE", "18"))
WATERMARK_OPACITY = float(os.environ.get("WATERMARK_OPACITY", "0.5"))
WATERMARK_POSITION = os.environ.get("WATERMARK_POSITION", "bottom-right")
# Optional explicit override; otherwise we auto-detect a system font below.
WATERMARK_FONT_FILE = os.environ.get("WATERMARK_FONT_FILE", "").strip() or None

# Intermediate encode only — transcode_stage re-encodes for each platform, so a
# faster preset here cuts queue time without affecting shipped quality much.
WATERMARK_X264_PRESET = (
    os.environ.get("WATERMARK_X264_PRESET", "veryfast").strip() or "veryfast"
)
WATERMARK_X264_CRF = os.environ.get("WATERMARK_X264_CRF", "23").strip() or "23"

# Candidate fontfile paths in order of preference. drawtext without a fontfile
# relies on fontconfig finding *some* font; on slim Linux containers no font
# may be installed and the filter will fail at runtime. We always try to pass
# an explicit fontfile=... when one is available.
_FONT_CANDIDATES = (
    # Linux (Debian/Ubuntu — install via fonts-dejavu-core)
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    # Alpine
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    # macOS
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
    # Windows
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/arial.ttf",
)


def _resolve_fontfile() -> Optional[str]:
    """Return the first existing fontfile path, or None if nothing is found."""
    if WATERMARK_FONT_FILE and Path(WATERMARK_FONT_FILE).exists():
        return WATERMARK_FONT_FILE
    for p in _FONT_CANDIDATES:
        try:
            if Path(p).exists():
                return p
        except OSError:
            continue
    return None


def _watermark_skip_hint_from_ffmpeg_stderr(stderr_text: str) -> str:
    """Short remediation hint for SkipStage / Sentry when FFmpeg stderr is available."""
    if not stderr_text:
        return ""
    low = stderr_text.lower()
    if "drawtext" in low or "fontconfig" in low or "no font" in low or "font file" in low:
        return (
            " [drawtext/fonts: install fonts (e.g. fonts-dejavu-core in Docker) "
            "or set WATERMARK_FONT_FILE to a .ttf path]"
        )
    if "unknown encoder" in low and "libx264" in low:
        return " [ffmpeg build missing libx264]"
    if "not recognized as an internal or external command" in low:
        return " [ffmpeg not executable]"
    if "winerror 2" in low or ("errno 2" in low and "ffmpeg" in low):
        return " [ffmpeg not found at run time — set FFMPEG_BIN to ffmpeg.exe]"
    return ""


def _escape_drawtext(text: str) -> str:
    """Escape a string for use inside a drawtext text='...' value.

    drawtext is sensitive to single quotes, backslashes, colons and percent
    signs because the filter uses ':' as a key separator and '%' for
    expressions. See ffmpeg-filters docs.
    """
    # Order matters: backslash first.
    return (
        text.replace("\\", "\\\\")
        .replace("'", r"\'")
        .replace(":", r"\:")
        .replace("%", r"\%")
    )


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


def resolve_watermark_display_text(ctx: JobContext) -> str:
    """Prefer job context (admin_settings / worker), then env WATERMARK_TEXT."""
    explicit = (getattr(ctx, "watermark_text", None) or "").strip()
    if explicit:
        return explicit
    env_fallback = (WATERMARK_TEXT or "").strip()
    return env_fallback or "Upload M8"


def build_drawtext_filter(
    text: str = WATERMARK_TEXT,
    font_size: int = WATERMARK_FONT_SIZE,
    opacity: float = WATERMARK_OPACITY,
    position: str = WATERMARK_POSITION,
    fontfile: Optional[str] = None,
) -> str:
    """Build the ffmpeg drawtext filter string used by the watermark stage.

    Exposed so tests (and callers) can verify the exact filter that gets sent
    to ffmpeg without spinning up a subprocess.
    """
    fontfile = fontfile if fontfile is not None else _resolve_fontfile()
    pos = _get_position_filter(position, font_size)
    parts = [f"drawtext=text='{_escape_drawtext(text)}'"]
    if fontfile:
        # ffmpeg wants forward slashes and the colon in C:/ escaped.
        ff = fontfile.replace("\\", "/").replace(":", r"\:")
        parts.append(f"fontfile='{ff}'")
    parts.append(f"fontsize={font_size}")
    parts.append(f"fontcolor=white@{opacity}")
    parts.append(pos)
    parts.append("box=1:boxcolor=black@0.35:boxborderw=6")
    parts.append("shadowcolor=black@0.6:shadowx=1:shadowy=1")
    return ":".join(parts)


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

    ffmpeg_bin = resolve_ffmpeg_executable()
    if not ffmpeg_bin:
        logger.warning(
            "Watermark skipped: ffmpeg not found (set FFMPEG_BIN or add ffmpeg to PATH)"
        )
        raise SkipStage(
            "ffmpeg binary not available (install ffmpeg; on Windows try: "
            "winget install FFmpeg, or set FFMPEG_BIN=C:\\\\path\\\\to\\\\ffmpeg.exe)"
        )

    fontfile = _resolve_fontfile()
    if not fontfile:
        # Without a fontfile drawtext often fails on slim Linux containers
        # because no fontconfig fonts are installed. Surface this loudly so
        # ops adds fonts-dejavu-core (or sets WATERMARK_FONT_FILE) to the image.
        logger.error(
            "Watermark cannot be applied: no usable fontfile found. "
            "Install fonts-dejavu-core in the worker image or set "
            "WATERMARK_FONT_FILE=/path/to/font.ttf"
        )
        raise SkipStage("No fontfile available for drawtext")

    display_text = resolve_watermark_display_text(ctx)
    logger.info(
        "Applying '%s' watermark to upload %s using font %s",
        display_text, ctx.upload_id, fontfile,
    )

    output_path = ctx.temp_dir / f"wm_{ctx.upload_id}.mp4"

    drawtext_filter = build_drawtext_filter(text=display_text, fontfile=fontfile)

    cmd = [
        ffmpeg_bin,
        "-y",
        "-i", str(video_path),
        "-vf", drawtext_filter,
        "-c:v", "libx264",
        "-preset", WATERMARK_X264_PRESET,
        "-crf", WATERMARK_X264_CRF,
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(output_path),
    ]

    logger.debug("Watermark ffmpeg cmd: %s", " ".join(cmd))

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()

        if proc.returncode != 0 or not output_path.exists():
            err_raw = stderr.decode(errors="replace") if stderr else ""
            error_snippet = err_raw[-500:] if err_raw else "unknown error"
            hint = _watermark_skip_hint_from_ffmpeg_stderr(err_raw)
            logger.warning(
                "Watermark FFmpeg failed (non-fatal) rc=%s hint=%r: %s",
                proc.returncode, hint.strip() or None, error_snippet,
            )
            raise SkipStage(
                f"Watermark FFmpeg failed rc={proc.returncode}{hint}; stderr_tail={error_snippet!r}"
            )

        size = output_path.stat().st_size
        if size <= 0:
            raise SkipStage("Watermark FFmpeg produced empty output")

        ctx.processed_video_path = output_path
        ctx.output_artifacts["watermarked_video"] = str(output_path)
        logger.info("Watermark applied: %s (%d bytes)", output_path, size)

    except SkipStage:
        raise
    except Exception as e:
        # Watermark failure should never crash the pipeline
        logger.warning("Watermark stage error (non-fatal): %s", e)
        raise SkipStage(f"Watermark failed: {e}")

    return ctx
