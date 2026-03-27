"""
Dominant color extraction from thumbnail frames (Pillow — no extra deps).
Used for brief hints + optional Playwright/CSS accent selection.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger("uploadm8-worker")


def _rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    r, g, b = rgb
    return f"#{r:02x}{g:02x}{b:02x}"


def _complement(rgb: Tuple[int, int, int]) -> str:
    return _rgb_to_hex((255 - rgb[0], 255 - rgb[1], 255 - rgb[2]))


def extract_palette_from_image(image_path: Path, n: int = 5) -> Dict[str, Any]:
    """
    Extract dominant colors from a JPEG/PNG using median-cut quantization.
    Returns hex strings + category hint for thumbnail mood (warm/cool/neutral).
    """
    try:
        from PIL import Image
    except ImportError:
        return {}

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        logger.debug("[color_palette] open failed: %s", e)
        return {}

    w, h = img.size
    img = img.resize((max(1, min(120, w)), max(1, min(120, h))))
    q = img.quantize(colors=min(n + 2, 32), method=Image.Quantize.MEDIANCUT)
    pal = q.getpalette()
    if not pal:
        return {}
    counts = q.getcolors() or []
    counts.sort(key=lambda x: -x[0])
    top_rgb: List[Tuple[int, int, int]] = []
    for cnt, idx in counts[:n]:
        if len(top_rgb) >= n:
            break
        base = idx * 3
        if base + 2 < len(pal):
            top_rgb.append((pal[base], pal[base + 1], pal[base + 2]))

    if not top_rgb:
        return {}

    primary = top_rgb[0]
    accent = top_rgb[1] if len(top_rgb) > 1 else primary
    warm_score = sum(1 for c in top_rgb if c[0] + c[1] > 2 * c[2] + 80)
    cool_score = sum(1 for c in top_rgb if c[2] > c[0] + 30 and c[2] > c[1] + 30)
    if warm_score >= cool_score + 1:
        mood = "warm"
    elif cool_score >= warm_score + 1:
        mood = "cool"
    else:
        mood = "neutral"

    return {
        "primary_hex": _rgb_to_hex(primary),
        "accent_hex": _rgb_to_hex(accent),
        "complement_hex": _complement(primary),
        "all_hex": [_rgb_to_hex(c) for c in top_rgb],
        "mood_hint": mood,
    }
