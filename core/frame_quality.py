"""Reject unusable Vision stills (tiny / near-black HEVC keyframe misses)."""

from __future__ import annotations

from pathlib import Path
from typing import Union

PathLike = Union[str, Path]

_MIN_BYTES = 1000
_NEAR_BLACK_MEAN = 18.0


def jpeg_is_unusable(
    path: PathLike,
    *,
    min_bytes: int = _MIN_BYTES,
    max_mean: float = _NEAR_BLACK_MEAN,
) -> bool:
    """True when the JPEG is missing, tiny, or nearly all black."""
    p = Path(path)
    try:
        if not p.exists() or not p.is_file():
            return True
        size = p.stat().st_size
    except OSError:
        return True
    if size < max(1, int(min_bytes)):
        return True
    try:
        from PIL import Image

        with Image.open(p) as im:
            gray = im.convert("L")
            gray.thumbnail((64, 64))
            pixels = list(getattr(gray, "get_flattened_data", gray.getdata)())
        if not pixels:
            return True
        mean = sum(pixels) / float(len(pixels))
        return mean < float(max_mean)
    except Exception:
        return size < max(int(min_bytes), 4000)


__all__ = ["jpeg_is_unusable"]
