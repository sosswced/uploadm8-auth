"""
UploadM8 Watermark Stage
==========================
Apply tier-based watermark overlay to videos using FFmpeg.

Free tier (can_watermark): UploadM8 text/logo burned in.
Paid tiers: no burn unless sponsorWatermarkOptIn (UploadM8 sponsorship branding).

Exports: run_watermark_stage(ctx)
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Cap FFmpeg CPU/RAM per invocation so concurrent jobs don't OOM the box.
_FFMPEG_THREADS_DEFAULT = max(
    1,
    (os.cpu_count() or 2) // max(1, int(os.environ.get("WORKER_CONCURRENCY", "2"))),
)
if os.environ.get("RENDER") and "FFMPEG_THREADS" not in os.environ:
    _FFMPEG_THREADS_DEFAULT = min(_FFMPEG_THREADS_DEFAULT, 2)
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
_WATERMARK_BASE_LOGO_AT_REF = 180
_WATERMARK_MIN_LOGO = 48
_WATERMARK_MAX_LOGO = 480
# Optional explicit override; otherwise we auto-detect a system font below.
WATERMARK_FONT_FILE = os.environ.get("WATERMARK_FONT_FILE", "").strip() or None

# Intermediate encode only — transcode_stage re-encodes for each platform, so a
# faster preset here cuts queue time without affecting shipped quality much.
WATERMARK_X264_PRESET = (
    os.environ.get("WATERMARK_X264_PRESET", "veryfast").strip() or "veryfast"
)
WATERMARK_X264_CRF = os.environ.get("WATERMARK_X264_CRF", "23").strip() or "23"

# font_family → (bold candidates, regular candidates)
_FONT_FAMILY_CANDIDATES: Dict[str, Tuple[Tuple[str, ...], Tuple[str, ...]]] = {
    "dejavu": (
        (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
            "C:/Windows/Fonts/DejaVuSans-Bold.ttf",
        ),
        (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
            "C:/Windows/Fonts/DejaVuSans.ttf",
        ),
    ),
    "liberation": (
        (
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/liberation/LiberationSans-Bold.ttf",
            "C:/Windows/Fonts/LiberationSans-Bold.ttf",
        ),
        (
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
            "C:/Windows/Fonts/LiberationSans-Regular.ttf",
        ),
    ),
    "arial": (
        (
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/Library/Fonts/Arial Bold.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
        ),
        (
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/Arial.ttf",
            "C:/Windows/Fonts/arial.ttf",
        ),
    ),
}

# Fallback across families when preferred family is missing on the worker image.
_FONT_CANDIDATES_BOLD = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
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


def _first_existing_font(candidates: Tuple[str, ...]) -> Optional[str]:
    for p in candidates:
        try:
            if Path(p).exists():
                return p
        except OSError:
            continue
    return None


def _resolve_fontfile(bold: bool = True, font_family: str = "dejavu") -> Optional[str]:
    """Return the first existing fontfile path, or None if nothing is found."""
    if WATERMARK_FONT_FILE and Path(WATERMARK_FONT_FILE).exists():
        return WATERMARK_FONT_FILE
    fam = str(font_family or "dejavu").strip().lower()
    pair = _FONT_FAMILY_CANDIDATES.get(fam)
    if pair:
        preferred = pair[0] if bold else pair[1]
        found = _first_existing_font(preferred)
        if found:
            return found
        # Same family, other weight
        found = _first_existing_font(pair[1] if bold else pair[0])
        if found:
            return found
    fallback = _FONT_CANDIDATES_BOLD if bold else _FONT_CANDIDATES_REGULAR
    found = _first_existing_font(fallback)
    if found:
        return found
    return _first_existing_font(_FONT_CANDIDATES_REGULAR if bold else _FONT_CANDIDATES_BOLD)


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
    if "overlay" in low and ("invalid" in low or "error" in low):
        return " [logo overlay failed — check PNG/JPEG watermark logo asset]"
    if "unknown encoder" in low and "libx264" in low:
        return " [ffmpeg build missing libx264]"
    if "not recognized as an internal or external command" in low:
        return " [ffmpeg not executable]"
    if "winerror 2" in low or ("errno 2" in low and "ffmpeg" in low):
        return " [ffmpeg not found at run time — set FFMPEG_BIN to ffmpeg.exe]"
    return ""


def _escape_drawtext(text: str) -> str:
    """Escape a string for use inside a drawtext text='...' value."""
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


def _get_overlay_xy(position: str, pad: int = 16) -> str:
    """FFmpeg overlay x/y expressions for a corner/center layout."""
    pad = max(8, int(pad))
    positions = {
        "top-left": f"x={pad}:y={pad}",
        "top-center": f"x=(main_w-overlay_w)/2:y={pad}",
        "top-right": f"x=main_w-overlay_w-{pad}:y={pad}",
        "bottom-left": f"x={pad}:y=main_h-overlay_h-{pad}",
        "bottom-center": f"x=(main_w-overlay_w)/2:y=main_h-overlay_h-{pad}",
        "bottom-right": f"x=main_w-overlay_w-{pad}:y=main_h-overlay_h-{pad}",
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


def compute_scaled_logo_width(
    width: int,
    height: int,
    *,
    size_scale: int = 100,
) -> int:
    """Scale logo width to the video's short edge with an admin multiplier."""
    short_edge = max(1, min(int(width or 0), int(height or 0)))
    scale = max(0.5, min(2.0, float(size_scale) / 100.0))
    raw = _WATERMARK_BASE_LOGO_AT_REF * (short_edge / _WATERMARK_REF_SHORT_EDGE) * scale
    return max(_WATERMARK_MIN_LOGO, min(_WATERMARK_MAX_LOGO, int(round(raw))))


def format_watermark_display_text(settings: Dict[str, Any]) -> str:
    """Apply optional 'Sponsored by' prefix to the burn-in label."""
    text = str(settings.get("text") or WATERMARK_TEXT or "Upload M8").strip() or "Upload M8"
    if settings.get("sponsored_prefix"):
        prefix = str(settings.get("sponsored_prefix_text") or "Sponsored by").strip() or "Sponsored by"
        # Avoid double-prefix if admin already typed it into the label.
        low = text.lower()
        if not low.startswith(prefix.lower()):
            text = f"{prefix} {text}".strip()
    return text[:120]


def watermark_requires_logo_prepass(settings: Optional[Dict[str, Any]] = None) -> bool:
    """Logo overlay needs a dedicated FFmpeg pass (cannot use text-only single-pass vf)."""
    s = settings or {}
    mode = str(s.get("mode") or "text").strip().lower()
    return mode in ("logo", "both") and bool(str(s.get("logo_r2_key") or "").strip())


def should_apply_watermark(ctx: JobContext) -> bool:
    """Free-tier burn-in, or paid opt-in for UploadM8 sponsorship branding."""
    explicit = getattr(ctx, "apply_watermark", None)
    if explicit is not None:
        return bool(explicit)
    if ctx.entitlements and getattr(ctx.entitlements, "can_watermark", False):
        return True
    us = getattr(ctx, "user_settings", None) or {}
    if isinstance(us, dict):
        return bool(us.get("sponsorWatermarkOptIn") or us.get("sponsor_watermark_opt_in"))
    return False


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
    if "position" not in raw and "text_position" not in raw:
        raw["position"] = WATERMARK_POSITION
    return normalize_watermark_settings(raw)


def resolve_watermark_display_text(ctx: JobContext) -> str:
    return format_watermark_display_text(resolve_watermark_settings(ctx))


async def build_watermark_vf_for_transcode(ctx: JobContext, video_path: Path) -> Optional[str]:
    """Build drawtext vf fragment for single-pass watermark burn during transcode.

    Returns None when logo overlay is required (caller must run watermark pre-pass)
    or when burn-in is not applicable.
    """
    if not should_apply_watermark(ctx):
        return None
    wm_settings = resolve_watermark_settings(ctx)
    if watermark_requires_logo_prepass(wm_settings):
        return None
    if wm_settings.get("mode") == "logo":
        return None
    font_weight = wm_settings.get("font_weight") or "bold"
    bold = str(font_weight).strip().lower() not in ("normal", "regular", "400")
    fontfile = _resolve_fontfile(
        bold=bold,
        font_family=str(wm_settings.get("font_family") or "dejavu"),
    )
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
        text=format_watermark_display_text(wm_settings),
        font_size=font_size,
        opacity=wm_settings["opacity"],
        position=wm_settings.get("text_position") or wm_settings.get("position") or WATERMARK_POSITION,
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
    font_family: str = "dejavu",
) -> str:
    """Build the ffmpeg drawtext filter string used by the watermark stage."""
    bold = str(font_weight or "bold").strip().lower() not in ("normal", "regular", "400")
    fontfile = fontfile if fontfile is not None else _resolve_fontfile(bold=bold, font_family=font_family)
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


def build_logo_overlay_chain(
    *,
    logo_width: int,
    opacity: float,
    position: str,
    include_drawtext: Optional[str] = None,
) -> str:
    """Build filter_complex for logo scale/opacity + overlay (+ optional drawtext)."""
    op = max(0.05, min(1.0, float(opacity)))
    xy = _get_overlay_xy(position, pad=max(12, int(round(logo_width * 0.08))))
    # Input 0 = video, input 1 = logo image
    chain = (
        f"[1:v]scale={int(logo_width)}:-1,format=rgba,"
        f"colorchannelmixer=aa={op:.2f}[lg];"
        f"[0:v][lg]overlay={xy}"
    )
    if include_drawtext:
        chain += f"[v1];[v1]{include_drawtext}"
    return chain


async def run_watermark_stage(ctx: JobContext) -> JobContext:
    """
    Apply watermark to the video if required by tier or paid sponsorship opt-in.

    Logic:
    - If should_apply_watermark is False → skip (paid tier without opt-in).
    - mode text → drawtext only
    - mode logo → overlay only
    - mode both → overlay + drawtext
    - On FFmpeg failure, skip gracefully (don't block the pipeline).
    """
    ctx.mark_stage("watermark")

    if not should_apply_watermark(ctx):
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
    mode = str(wm_settings.get("mode") or "text").strip().lower()
    logo_path = getattr(ctx, "watermark_logo_local_path", None)
    if isinstance(logo_path, str):
        logo_path = Path(logo_path)
    has_logo = bool(logo_path and Path(logo_path).exists())
    if mode in ("logo", "both") and not has_logo:
        if mode == "logo":
            raise SkipStage("Watermark logo mode but logo file missing")
        mode = "text"

    want_text = mode in ("text", "both")
    want_logo = mode in ("logo", "both") and has_logo

    font_weight = wm_settings.get("font_weight") or "bold"
    bold = str(font_weight).strip().lower() not in ("normal", "regular", "400")
    font_family = str(wm_settings.get("font_family") or "dejavu")
    fontfile = _resolve_fontfile(bold=bold, font_family=font_family) if want_text else None
    if want_text and not fontfile:
        logger.error(
            "Watermark cannot be applied: no usable fontfile found. "
            "Install fonts-dejavu-core in the worker image or set "
            "WATERMARK_FONT_FILE=/path/to/font.ttf"
        )
        if not want_logo:
            raise SkipStage("No fontfile available for drawtext")
        want_text = False
        mode = "logo"

    display_text = format_watermark_display_text(wm_settings)
    font_size = WATERMARK_FONT_SIZE
    logo_width = _WATERMARK_BASE_LOGO_AT_REF
    try:
        info = await get_video_info(video_path)
        font_size = compute_scaled_watermark_font_size(
            info.width,
            info.height,
            size_scale=wm_settings["size_scale"],
        )
        logo_width = compute_scaled_logo_width(
            info.width,
            info.height,
            size_scale=int(wm_settings.get("logo_size_scale") or 100),
        )
    except Exception as e:
        logger.warning(
            "Watermark size fallback for upload %s (ffprobe failed: %s)",
            ctx.upload_id,
            e,
        )

    logger.info(
        "Applying watermark to upload %s (mode=%s, text=%r, fontsize=%s, logo_w=%s)",
        ctx.upload_id,
        mode,
        display_text if want_text else None,
        font_size if want_text else None,
        logo_width if want_logo else None,
    )

    output_path = ctx.temp_dir / f"wm_{ctx.upload_id}.mp4"
    drawtext_filter = None
    if want_text:
        drawtext_filter = build_drawtext_filter(
            text=display_text,
            font_size=font_size,
            opacity=wm_settings["opacity"],
            position=wm_settings.get("text_position") or wm_settings.get("position") or WATERMARK_POSITION,
            fontfile=fontfile,
            text_color=wm_settings.get("text_color") or "#ffffff",
            font_weight=font_weight,
            font_family=font_family,
        )

    cmd = [ffmpeg_bin, "-y", "-i", str(video_path)]
    if want_logo:
        cmd.extend(["-i", str(logo_path)])
        filter_complex = build_logo_overlay_chain(
            logo_width=logo_width,
            opacity=float(wm_settings.get("logo_opacity") or 0.9),
            position=str(wm_settings.get("logo_position") or "bottom-left"),
            include_drawtext=drawtext_filter if want_text else None,
        )
        cmd.extend(["-filter_complex", filter_complex])
    elif drawtext_filter:
        cmd.extend(["-vf", drawtext_filter])
    else:
        raise SkipStage("Watermark mode produced no filters")

    cmd.extend(
        [
            "-c:v", "libx264",
            "-threads", str(FFMPEG_THREADS),
            "-preset", WATERMARK_X264_PRESET,
            "-crf", WATERMARK_X264_CRF,
            "-c:a", "copy",
            "-movflags", "+faststart",
            str(output_path),
        ]
    )

    logger.debug("Watermark ffmpeg cmd: %s", " ".join(cmd))

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, stderr = await proc.communicate()
        except asyncio.CancelledError:
            from stages.ffmpeg_progress import kill_process_quietly

            await kill_process_quietly(proc)
            raise

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
