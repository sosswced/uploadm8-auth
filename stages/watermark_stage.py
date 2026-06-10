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
from typing import Any, Dict, Optional

# Cap FFmpeg CPU per invocation so concurrent jobs don't pin all cores.
# Default scales with WORKER_CONCURRENCY: cpu_count // concurrency, min 1.
_FFMPEG_THREADS_DEFAULT = max(
    1,
    (os.cpu_count() or 2) // max(1, int(os.environ.get("WORKER_CONCURRENCY", "3"))),
)
FFMPEG_THREADS = int(os.environ.get("FFMPEG_THREADS", str(_FFMPEG_THREADS_DEFAULT)))

from .db import normalize_watermark_settings
from .errors import SkipStage
from .context import JobContext
from .ffmpeg_env import resolve_ffmpeg_executable
from .transcode_stage import get_video_info

logger = logging.getLogger("uploadm8-worker")

# Fallback when ctx.watermark_settings is unset (env override for worker-only deploys)
WATERMARK_TEXT = os.environ.get("WATERMARK_TEXT", "Upload M8")
WATERMARK_SIZE_SCALE = int(os.environ.get("WATERMARK_SIZE_SCALE", "100"))
WATERMARK_OPACITY = float(os.environ.get("WATERMARK_OPACITY", "0.85"))
WATERMARK_POSITION = os.environ.get("WATERMARK_POSITION", "bottom-right")
# Legacy fixed size — only used when ffprobe fails and no dimensions are known.
WATERMARK_FONT_SIZE = int(os.environ.get("WATERMARK_FONT_SIZE", "42"))
# Resolution-scaled baseline: ~42px on a 1080p short edge at 100% admin scale.
_WATERMARK_REF_SHORT_EDGE = 1080
_WATERMARK_BASE_FONT_AT_REF = 42
_WATERMARK_MIN_FONT = 20
_WATERMARK_MAX_FONT = 120
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
_FONT_CANDIDATES_BOLD = (
    # Linux (Debian/Ubuntu — install via fonts-dejavu-core)
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    # Alpine
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    # macOS
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    # Windows
    "C:/Windows/Fonts/arialbd.ttf",
)
_FONT_CANDIDATES_REGULAR = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
    "C:/Windows/Fonts/arial.ttf",
)


def _hex_to_ffmpeg_color(hex_color: str) -> str:
    s = (hex_color or "").strip().lstrip("#")
    if len(s) == 6 and all(c in "0123456789abcdefABCDEF" for c in s):
        return f"0x{s.upper()}"
    return "white"


def _resolve_fontfile(bold: bool = True) -> Optional[str]:
    """Return the first existing fontfile path, or None if nothing is found."""
    if WATERMARK_FONT_FILE and Path(WATERMARK_FONT_FILE).exists():
        return WATERMARK_FONT_FILE
    candidates = _FONT_CANDIDATES_BOLD if bold else _FONT_CANDIDATES_REGULAR
    for p in candidates:
        try:
            if Path(p).exists():
                return p
        except OSError:
            continue
    # Fallback: any readable font from the other weight list
    for p in (_FONT_CANDIDATES_REGULAR if bold else _FONT_CANDIDATES_BOLD):
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
    pad = max(12, int(round(font_size * 0.45)))
    positions = {
        "top-left": f"x={pad}:y={pad}",
        "top-center": f"x=(w-text_w)/2:y={pad}",
        "top-right": f"x=w-text_w-{pad}:y={pad}",
        "bottom-left": f"x={pad}:y=h-text_h-{pad}",
        "bottom-center": f"x=(w-text_w)/2:y=h-text_h-{pad}",
        "bottom-right": f"x=w-text_w-{pad}:y=h-text_h-{pad}",
    }
    return positions.get(position, positions["bottom-right"])


def compute_scaled_watermark_font_size(
    width: int,
    height: int,
    *,
    size_scale: int = 100,
) -> int:
    """Scale watermark text to the video's short edge with an admin multiplier."""
    short_edge = max(1, min(int(width or 0), int(height or 0)))
    scale = max(0.5, min(2.0, float(size_scale) / 100.0))
    raw = _WATERMARK_BASE_FONT_AT_REF * (short_edge / _WATERMARK_REF_SHORT_EDGE) * scale
    return max(_WATERMARK_MIN_FONT, min(_WATERMARK_MAX_FONT, int(round(raw))))


def resolve_watermark_settings(ctx: JobContext) -> Dict[str, Any]:
    """Prefer job context (admin_settings / worker), then env fallbacks."""
    raw = dict(getattr(ctx, "watermark_settings", None) or {})
    explicit_text = (getattr(ctx, "watermark_text", None) or "").strip()
    if explicit_text:
        raw.setdefault("text", explicit_text)
    if not raw.get("text"):
        raw["text"] = (WATERMARK_TEXT or "").strip() or "Upload M8"
    if "size_scale" not in raw:
        raw["size_scale"] = WATERMARK_SIZE_SCALE
    if "opacity" not in raw:
        raw["opacity"] = WATERMARK_OPACITY
    if "position" not in raw:
        raw["position"] = WATERMARK_POSITION
    return normalize_watermark_settings(raw)


def resolve_watermark_display_text(ctx: JobContext) -> str:
    return resolve_watermark_settings(ctx)["text"]


async def build_watermark_vf_for_transcode(ctx: JobContext, video_path: Path) -> Optional[str]:
    """Build drawtext vf fragment for single-pass watermark burn during transcode."""
    if ctx.entitlements and not ctx.entitlements.can_watermark:
        return None
    wm_settings = resolve_watermark_settings(ctx)
    font_weight = wm_settings.get("font_weight") or "bold"
    bold = str(font_weight).strip().lower() not in ("normal", "regular", "400")
    fontfile = _resolve_fontfile(bold=bold)
    if not fontfile:
        return None
    font_size = WATERMARK_FONT_SIZE
    try:
        info = await get_video_info(video_path)
        font_size = compute_scaled_watermark_font_size(
            info.width,
            info.height,
            size_scale=wm_settings["size_scale"],
        )
    except Exception:
        pass
    return build_drawtext_filter(
        text=wm_settings["text"],
        font_size=font_size,
        opacity=wm_settings["opacity"],
        position=wm_settings["position"],
        fontfile=fontfile,
        text_color=wm_settings.get("text_color") or "#ffffff",
        font_weight=font_weight,
    )


def build_drawtext_filter(
    text: str = WATERMARK_TEXT,
    font_size: int = WATERMARK_FONT_SIZE,
    opacity: float = WATERMARK_OPACITY,
    position: str = WATERMARK_POSITION,
    fontfile: Optional[str] = None,
    text_color: str = "#ffffff",
    font_weight: str = "bold",
) -> str:
    """Build the ffmpeg drawtext filter string used by the watermark stage.

    Exposed so tests (and callers) can verify the exact filter that gets sent
    to ffmpeg without spinning up a subprocess.
    """
    bold = str(font_weight or "bold").strip().lower() not in ("normal", "regular", "400")
    fontfile = fontfile if fontfile is not None else _resolve_fontfile(bold=bold)
    pos = _get_position_filter(position, font_size)
    parts = [f"drawtext=text='{_escape_drawtext(text)}'"]
    if fontfile:
        # ffmpeg wants forward slashes and the colon in C:/ escaped.
        ff = fontfile.replace("\\", "/").replace(":", r"\:")
        parts.append(f"fontfile='{ff}'")
    box_border = max(4, int(round(font_size * 0.14)))
    shadow = max(1, int(round(font_size * 0.05)))
    parts.append(f"fontsize={font_size}")
    parts.append(f"fontcolor={_hex_to_ffmpeg_color(text_color)}@{opacity}")
    parts.append(pos)
    parts.append(f"box=1:boxcolor=black@0.45:boxborderw={box_border}")
    parts.append(f"shadowcolor=black@0.7:shadowx={shadow}:shadowy={shadow}")
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

    wm_settings = resolve_watermark_settings(ctx)
    font_weight = wm_settings.get("font_weight") or "bold"
    bold = str(font_weight).strip().lower() not in ("normal", "regular", "400")
    fontfile = _resolve_fontfile(bold=bold)
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
    display_text = wm_settings["text"]
    font_size = WATERMARK_FONT_SIZE
    try:
        info = await get_video_info(video_path)
        font_size = compute_scaled_watermark_font_size(
            info.width,
            info.height,
            size_scale=wm_settings["size_scale"],
        )
    except Exception as e:
        logger.warning(
            "Watermark font size fallback for upload %s (ffprobe failed: %s)",
            ctx.upload_id,
            e,
        )

    logger.info(
        "Applying '%s' watermark to upload %s (fontsize=%s, opacity=%s, position=%s) using font %s",
        display_text,
        ctx.upload_id,
        font_size,
        wm_settings["opacity"],
        wm_settings["position"],
        fontfile,
    )

    output_path = ctx.temp_dir / f"wm_{ctx.upload_id}.mp4"

    drawtext_filter = build_drawtext_filter(
        text=display_text,
        font_size=font_size,
        opacity=wm_settings["opacity"],
        position=wm_settings["position"],
        fontfile=fontfile,
        text_color=wm_settings.get("text_color") or "#ffffff",
        font_weight=font_weight,
    )

    cmd = [
        ffmpeg_bin,
        "-y",
        "-i", str(video_path),
        "-vf", drawtext_filter,
        "-c:v", "libx264",
        "-threads", str(FFMPEG_THREADS),
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
