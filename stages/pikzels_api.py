"""
Optional external thumbnail render client.

This integration is intentionally defensive:
- Fully opt-in via env vars
- Accepts multiple response shapes
- Falls back cleanly when unavailable
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger("uploadm8-worker.thumb_renderer")

# Legacy app.pikzels.com single-shot renderer (Bearer + X-API-Key). Precedence: THUMB_* then PIKZELS_*.
# Public OpenAPI v2 (api.pikzels.com, X-Api-Key) uses services.pikzels_v2.resolve_public_api_key — PIKZELS_* first.
THUMB_RENDER_API_KEY = (
    os.environ.get("THUMB_RENDER_API_KEY")
    or os.environ.get("PIKZELS_API_KEY")
    or ""
).strip()
THUMB_RENDER_API_URL = (
    os.environ.get("THUMB_RENDER_API_URL")
    or os.environ.get("PIKZELS_API_URL")
    or "https://app.pikzels.com/platform/api/thumbnail"
).strip()
THUMB_RENDER_TIMEOUT_SECONDS = float(
    os.environ.get("THUMB_RENDER_TIMEOUT_SECONDS")
    or os.environ.get("PIKZELS_TIMEOUT_SECONDS", "60")
    or 60
)
MIN_THUMB_SIZE = 2048


def studio_renderer_enabled() -> bool:
    return bool(THUMB_RENDER_API_KEY and THUMB_RENDER_API_URL)


def _decode_response_image(data: Dict[str, Any]) -> Optional[bytes]:
    """
    Accept several common payload shapes:
    - {"image_base64": "..."}
    - {"b64_json": "..."}
    - {"data":[{"b64_json":"..."}]}
    - {"image_url":"https://..."} / {"url":"https://..."} (downloaded by caller)
    """
    b64_value = (
        data.get("image_base64")
        or data.get("b64_json")
        or data.get("image_b64")
    )
    if not b64_value and isinstance(data.get("data"), list) and data["data"]:
        first = data["data"][0] if isinstance(data["data"][0], dict) else {}
        b64_value = first.get("b64_json") or first.get("image_base64")
    if not b64_value:
        return None
    try:
        return base64.b64decode(str(b64_value))
    except (binascii.Error, TypeError, ValueError):
        return None


async def render_thumbnail_with_studio_renderer(
    base_frame_path: Path,
    brief: Dict[str, Any],
    platform: str,
    output_path: Path,
    *,
    upload_id: str = "",
    category: str = "",
    persona: Optional[Dict[str, Any]] = None,
    options: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Render a styled thumbnail using external render API.

    Returns True only when a valid output image is saved.
    """
    if not studio_renderer_enabled():
        return False
    try:
        frame_b64 = base64.b64encode(base_frame_path.read_bytes()).decode("utf-8")
    except (OSError, PermissionError, TypeError, ValueError) as e:
        logger.warning("[thumb-renderer] failed reading source frame: %s", e)
        return False

    payload = {
        "platform": str(platform or "").lower(),
        "upload_id": str(upload_id or ""),
        "category": str(category or ""),
        "brief": brief or {},
        "image_base64": frame_b64,
        "image_mime_type": "image/jpeg",
    }
    if isinstance(persona, dict) and persona:
        payload["persona"] = persona
    if isinstance(options, dict) and options:
        payload["options"] = options
    headers = {
        "Authorization": f"Bearer {THUMB_RENDER_API_KEY}",
        "X-API-Key": THUMB_RENDER_API_KEY,
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=THUMB_RENDER_TIMEOUT_SECONDS) as client:
            res = await client.post(THUMB_RENDER_API_URL, headers=headers, json=payload)
            if res.status_code != 200:
                logger.warning("[thumb-renderer] render HTTP %s: %s", res.status_code, res.text[:240])
                return False
            data = res.json() if res.content else {}

            image_bytes = _decode_response_image(data)
            if not image_bytes:
                image_url = data.get("image_url") or data.get("url")
                if isinstance(data.get("data"), list) and data["data"]:
                    first = data["data"][0] if isinstance(data["data"][0], dict) else {}
                    image_url = image_url or first.get("url")
                if image_url:
                    dl = await client.get(str(image_url))
                    if dl.status_code == 200:
                        image_bytes = dl.content

            if not image_bytes:
                logger.warning("[thumb-renderer] no image returned in response")
                return False

            output_path.write_bytes(image_bytes)
            if output_path.exists() and output_path.stat().st_size >= MIN_THUMB_SIZE:
                return True
            logger.warning("[thumb-renderer] output file too small or missing")
            return False
    except asyncio.CancelledError:
        raise
    except (httpx.HTTPError, json.JSONDecodeError, OSError, TypeError, ValueError) as e:
        logger.warning("[thumb-renderer] render failed (non-fatal): %s", e)
        return False


# Legacy aliases kept for compatibility with older imports.
pikzels_enabled = studio_renderer_enabled
render_thumbnail_with_pikzels = render_thumbnail_with_studio_renderer
