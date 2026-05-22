"""
Local sticker compositor — crop → rembg → stroke → paste on 1280×720 (or 9:16).

Uses real frame crops only (no invented graphics). Layout follows the user's
``default_strategy`` (layout_pattern, text_position) and YouTube-ref-style slots.
Pikzels may still run afterward for YouTube-reference polish when configured.
"""

from __future__ import annotations

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.thumbnail_text import clean_thumbnail_headline, is_generic_thumbnail_headline
from services.thumbnail_sticker_pack import StickerSpec

logger = logging.getLogger("uploadm8-worker.thumbnail_sticker")

_sticker_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="thumb-sticker")

try:
    from rembg import remove as rembg_remove
    from rembg import new_session as rembg_new_session

    _rembg_session = rembg_new_session("isnet-general-use")
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    _rembg_session = None


def sticker_composite_enabled() -> bool:
    v = (os.environ.get("THUMBNAIL_STICKER_COMPOSITE") or "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _canvas_size(platform: str) -> Tuple[int, int]:
    if (platform or "").lower() == "youtube":
        return 1280, 720
    return 720, 1280


def _load_fonts():
    from PIL import ImageFont

    for path in (
        "arialbd.ttf",
        "Arial Bold.ttf",
        "arial.ttf",
        "Arial.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        try:
            return (
                ImageFont.truetype(path, 64),
                ImageFont.truetype(path, 32),
                ImageFont.truetype(path, 24),
            )
        except OSError:
            continue
    d = ImageFont.load_default()
    return d, d, d


def _fit_canvas(img, target_w: int, target_h: int):
    from PIL import Image, ImageEnhance

    iw, ih = img.size
    scale = max(target_w / iw, target_h / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    img = img.resize((nw, nh), Image.Resampling.LANCZOS)
    x0, y0 = (nw - target_w) // 2, (nh - target_h) // 2
    base = img.crop((x0, y0, x0 + target_w, y0 + target_h))
    base = ImageEnhance.Color(base).enhance(1.18)
    base = ImageEnhance.Contrast(base).enhance(1.12)
    return base.convert("RGBA")


def _crop_normalized(img, box: Dict[str, float]):
    from PIL import Image

    w, h = img.size
    left = max(0, min(w - 2, int(float(box.get("left", 0)) * w)))
    top = max(0, min(h - 2, int(float(box.get("top", 0)) * h)))
    right = max(left + 2, min(w, int(float(box.get("right", 1)) * w)))
    bottom = max(top + 2, min(h, int(float(box.get("bottom", 1)) * h)))
    if right - left < 24 or bottom - top < 24:
        return None
    return img.crop((left, top, right, bottom))


def _rembg_bytes(png_bytes: bytes) -> Optional[bytes]:
    if not REMBG_AVAILABLE or not _rembg_session:
        return None
    try:
        return rembg_remove(png_bytes, session=_rembg_session, force_return_bytes=True)
    except Exception as e:
        logger.debug("[sticker] rembg failed: %s", e)
        return None


def _add_stroke(rgba, *, stroke: int = 4, color=(255, 255, 255, 255)):
    from PIL import Image, ImageFilter

    if rgba.mode != "RGBA":
        rgba = rgba.convert("RGBA")
    alpha = rgba.split()[3]
    expanded = alpha.filter(ImageFilter.MaxFilter(stroke * 2 + 1))
    stroke_layer = Image.new("RGBA", rgba.size, color)
    stroke_layer.putalpha(expanded)
    out = Image.new("RGBA", rgba.size, (0, 0, 0, 0))
    out = Image.alpha_composite(out, stroke_layer)
    out = Image.alpha_composite(out, rgba)
    return out


def _layout_slots(
    target_w: int,
    target_h: int,
    count: int,
    *,
    layout_pattern: str = "",
    text_position: str = "",
) -> List[Tuple[int, int, float]]:
    """Return (x, y, scale) paste anchors for image stickers."""
    pattern = (layout_pattern or "").lower()
    margin = int(min(target_w, target_h) * 0.05)
    slots: List[Tuple[int, int, float]] = []

    if "split" in pattern or "reveal" in pattern:
        base_x = int(target_w * 0.58)
        base_y = margin
        step = int(target_h * 0.22)
        scales = [0.34, 0.30, 0.28, 0.26]
        for i in range(count):
            slots.append((base_x, base_y + i * step, scales[i % len(scales)]))
        return slots

    if "speed" in pattern or "streak" in pattern:
        scales = [0.38, 0.28, 0.24]
        slots.append((target_w - margin - int(target_w * 0.32), target_h - margin - int(target_h * 0.35), scales[0]))
        slots.append((margin, margin, scales[1]))
        for i in range(2, count):
            slots.append((target_w - margin - int(target_w * 0.28), margin + i * 80, scales[2]))
        return slots[:count]

    # Default fan — upper-right stack (YouTube collage style)
    upper = "upper" in (text_position or "").lower() or "top" in (text_position or "").lower()
    base_x = target_w - margin - int(target_w * 0.30)
    base_y = margin if upper else int(target_h * 0.12)
    scales = [0.36, 0.30, 0.26, 0.22]
    for i in range(count):
        slots.append((base_x, base_y + i * int(target_h * 0.18), scales[i % len(scales)]))
    return slots


def _text_badge_positions(target_w: int, target_h: int, count: int) -> List[Tuple[int, int]]:
    margin = int(min(target_w, target_h) * 0.05)
    positions = [
        (margin, margin),
        (margin, margin + 52),
        (target_w - margin - 220, margin),
    ]
    return positions[:count]


def _draw_text_badge(draw, xy, text: str, font, *, fill="#ffffff", bg="#e53935"):
    from PIL import ImageFont

    if not isinstance(font, ImageFont.FreeTypeFont):
        font = font
    pad_x, pad_y = 10, 6
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x, y = xy
    rect = [x, y, x + tw + pad_x * 2, y + th + pad_y * 2]
    draw.rounded_rectangle(rect, radius=8, fill=bg, outline="#ffffff", width=2)
    draw.text((x + pad_x, y + pad_y), text, fill=fill, font=font)


def _render_sync(
    base_path: Path,
    brief: Dict[str, Any],
    platform: str,
    output_path: Path,
    stickers: Sequence[StickerSpec],
    *,
    platform_color: Optional[str] = None,
    accent_color: Optional[str] = None,
) -> bool:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        logger.warning("[sticker] Pillow not installed")
        return False

    if not stickers:
        return False

    target_w, target_h = _canvas_size(platform)
    try:
        src = Image.open(base_path).convert("RGB")
    except Exception as e:
        logger.warning("[sticker] could not open base frame: %s", e)
        return False

    canvas = _fit_canvas(src, target_w, target_h)
    strategy = brief.get("default_strategy") if isinstance(brief.get("default_strategy"), dict) else {}
    layout_pattern = str(strategy.get("layout_pattern") or "")
    text_position = str(strategy.get("text_position") or "")

    image_stickers = [s for s in stickers if s.box and not s.text_only]
    text_stickers = [s for s in stickers if s.text_only or not s.box]

    slots = _layout_slots(
        target_w,
        target_h,
        len(image_stickers),
        layout_pattern=layout_pattern,
        text_position=text_position,
    )

    for sticker, (sx, sy, scale) in zip(image_stickers, slots):
        crop = _crop_normalized(src, sticker.box or {})
        if crop is None:
            continue
        buf = BytesIO()
        crop.save(buf, format="PNG")
        cutout_bytes = _rembg_bytes(buf.getvalue()) or buf.getvalue()
        try:
            sticker_img = Image.open(BytesIO(cutout_bytes)).convert("RGBA")
        except Exception:
            sticker_img = crop.convert("RGBA")
        sticker_img = _add_stroke(sticker_img)
        max_h = int(target_h * scale)
        ratio = max_h / max(1, sticker_img.height)
        nw = max(24, int(sticker_img.width * ratio))
        nh = max(24, int(sticker_img.height * ratio))
        sticker_img = sticker_img.resize((nw, nh), Image.Resampling.LANCZOS)
        canvas.paste(sticker_img, (sx, sy), sticker_img)

    draw = ImageDraw.Draw(canvas)
    font_lg, font_md, font_sm = _load_fonts()
    badge_bg = platform_color or "#e53935"
    accent = accent_color or "#FFFFFF"

    for i, sticker in enumerate(text_stickers[:3]):
        pos = _text_badge_positions(target_w, target_h, 3)[i % 3]
        _draw_text_badge(draw, pos, sticker.label[:42].upper(), font_sm, bg=badge_bg)

    headline = clean_thumbnail_headline(
        brief.get("selected_headline") or (brief.get("headline_options") or [""])[0],
        max_words=5,
        max_chars=32,
    )
    if headline and not is_generic_thumbnail_headline(headline):
        bbox = draw.textbbox((0, 0), headline, font=font_lg)
        tw = bbox[2] - bbox[0]
        margin = int(min(target_w, target_h) * 0.06)
        tx = (target_w - tw) // 2
        ty = target_h - margin - (bbox[3] - bbox[1]) - 12
        for dx, dy in [(-2, -2), (2, -2), (-2, 2), (2, 2)]:
            draw.text((tx + dx, ty + dy), headline, fill="#000000", font=font_lg)
        draw.text((tx, ty), headline, fill=accent, font=font_lg)

    badge = clean_thumbnail_headline(brief.get("badge_text"), max_words=2)[:12].upper()
    if badge:
        _draw_text_badge(draw, (int(target_w * 0.05), int(target_h * 0.08)), badge, font_md, bg=badge_bg)

    out_rgb = canvas.convert("RGB")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_rgb.save(output_path, format="JPEG", quality=92, optimize=True)
    return output_path.exists() and output_path.stat().st_size > 2048


async def render_sticker_composite(
    base_path: Path,
    brief: Dict[str, Any],
    platform: str,
    output_path: Path,
    stickers: Sequence[StickerSpec],
    *,
    platform_color: Optional[str] = None,
    accent_color: Optional[str] = None,
) -> bool:
    """Async wrapper around PIL/rembg compositing."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _sticker_executor,
        lambda: _render_sync(
            base_path,
            brief,
            platform,
            output_path,
            stickers,
            platform_color=platform_color,
            accent_color=accent_color,
        ),
    )
