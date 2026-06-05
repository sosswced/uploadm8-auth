"""Mirror external platform images into R2 for browser-safe delivery."""

from __future__ import annotations

import asyncio
import logging
import secrets
from typing import Optional
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

# Meta / TikTok CDNs reject cross-origin browser loads (403) even when URLs look public.
HOTLINK_BLOCKED_IMAGE_HOST_SUFFIXES = (
    "cdninstagram.com",
    "fbcdn.net",
    "tiktokcdn.com",
    "tiktokcdn-us.com",
    "tiktokv.com",
)


def is_hotlink_blocked_image_url(url: str | None) -> bool:
    """True when the URL should not be used as a browser <img src> (expect 403)."""
    raw = str(url or "").strip()
    if not raw.startswith(("http://", "https://")):
        return False
    try:
        host = (urlparse(raw).hostname or "").lower()
    except Exception:
        return False
    return any(host == sfx or host.endswith("." + sfx) for sfx in HOTLINK_BLOCKED_IMAGE_HOST_SUFFIXES)


async def mirror_external_image_to_r2(key_prefix: str, image_url: str) -> Optional[str]:
    """
    Fetch an image server-side and store in R2. Returns the object key or None.
    """
    from core.config import R2_BUCKET_NAME
    from core.r2 import put_object_bytes

    url = str(image_url or "").strip()
    if not url.startswith(("http://", "https://")):
        return None
    if not (R2_BUCKET_NAME or "").strip():
        return None
    prefix = "/".join(p for p in str(key_prefix or "").strip("/").split("/") if p)
    if not prefix:
        prefix = "external-images"
    try:
        async with httpx.AsyncClient(timeout=18.0, follow_redirects=True) as client:
            resp = await client.get(
                url,
                headers={
                    "User-Agent": "UploadM8/1.0 (media-mirror)",
                    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
                },
            )
        if resp.status_code >= 400:
            logger.debug("mirror_external_image HTTP %s url=%s", resp.status_code, url[:100])
            return None
        raw = resp.content or b""
        if not raw or len(raw) > 8 * 1024 * 1024:
            return None
        ct = (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
        ext = "jpg"
        if "png" in ct or raw[:8] == b"\x89PNG\r\n\x1a\n":
            ext = "png"
        elif "webp" in ct:
            ext = "webp"
        elif "gif" in ct or raw[:6] in (b"GIF87a", b"GIF89a"):
            ext = "gif"
        elif raw[:3] == b"\xff\xd8\xff":
            ext = "jpg"
        elif ct and not (ct.startswith("image/") or ct == "application/octet-stream"):
            return None
        upload_ct = ct if ct.startswith("image/") else f"image/{ext}"
        key = f"{prefix}/{secrets.token_hex(10)}.{ext}"
        await asyncio.to_thread(put_object_bytes, key, bytes(raw), upload_ct)
        return key
    except Exception as e:
        logger.debug("mirror_external_image_to_r2 failed url=%s err=%s", url[:100], e)
        return None
