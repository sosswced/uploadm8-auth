"""Image format helpers for platform APIs (Meta cover_url requires real JPEG)."""

from __future__ import annotations

import io
import logging
from pathlib import Path

logger = logging.getLogger("uploadm8-worker")

# Instagram Reels cover photo limit (Meta docs)
META_COVER_JPEG_MAX_BYTES = 8 * 1024 * 1024


def sniff_image_format(path: Path) -> str:
    """Return ``jpeg``, ``png``, ``webp``, or ``unknown`` from file magic bytes."""
    try:
        head = path.read_bytes()[:16]
    except OSError:
        return "unknown"
    if len(head) >= 3 and head[0:3] == b"\xff\xd8\xff":
        return "jpeg"
    if len(head) >= 8 and head[0:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if len(head) >= 12 and head[0:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    return "unknown"


def ensure_jpeg_file(
    path: Path,
    *,
    quality: int = 90,
    max_bytes: int = META_COVER_JPEG_MAX_BYTES,
) -> Path:
    """Ensure ``path`` is a JPEG file (rewrite in place when PNG/WebP).

    Meta Instagram ``cover_url`` rejects PNG even when the key ends in ``.jpg``.
    Pikzels v2 often returns PNG bytes written to a ``.jpg`` path.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if sniff_image_format(path) == "jpeg":
        if path.stat().st_size <= max_bytes:
            return path
        # Re-encode oversized JPEG down.
    from PIL import Image

    with Image.open(path) as im:
        rgb = im.convert("RGB")
        buf = io.BytesIO()
        q = max(50, min(95, int(quality)))
        rgb.save(buf, format="JPEG", quality=q, optimize=True)
        while buf.tell() > max_bytes and q > 50:
            q -= 5
            buf = io.BytesIO()
            rgb.save(buf, format="JPEG", quality=q, optimize=True)
        if buf.tell() > max_bytes:
            logger.warning(
                "ensure_jpeg_file: %s still %s bytes after quality reduction",
                path.name,
                buf.tell(),
            )
        path.write_bytes(buf.getvalue())
    return path
