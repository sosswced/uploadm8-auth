"""
Optional Figma API integration for thumbnail text layers.

UploadM8 renders thumbnails with Playwright HTML/CSS and PIL templates by default.
To use Figma exports instead, set FIGMA_ACCESS_TOKEN + FIGMA_FILE_KEY + FIGMA_NODE_ID
and extend the thumbnail stage to call fetch_figma_node_png before compositing.

This module is a typed stub so credentials never block the pipeline.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger("uploadm8-worker.figma")

FIGMA_ACCESS_TOKEN = (os.environ.get("FIGMA_ACCESS_TOKEN") or "").strip()
FIGMA_FILE_KEY = (os.environ.get("FIGMA_FILE_KEY") or "").strip()
FIGMA_NODE_ID = (os.environ.get("FIGMA_NODE_ID") or "").strip()


def fetch_figma_node_png(scale: int = 2) -> Optional[bytes]:
    """
    GET /v1/images/{file_key} — returns PNG bytes for NODE_ID, or None if not configured.
    Not invoked automatically; wire from thumbnail_stage when you need brand-locked layouts.
    """
    if not (FIGMA_ACCESS_TOKEN and FIGMA_FILE_KEY and FIGMA_NODE_ID):
        return None
    try:
        import httpx

        url = f"https://api.figma.com/v1/images/{FIGMA_FILE_KEY}"
        with httpx.Client(timeout=60.0) as client:
            r = client.get(
                url,
                params={"ids": FIGMA_NODE_ID, "format": "png", "scale": scale},
                headers={"X-Figma-Token": FIGMA_ACCESS_TOKEN},
            )
        if r.status_code != 200:
            logger.warning("[figma] images API HTTP %s", r.status_code)
            return None
        data = r.json()
        images = data.get("images") or {}
        img_url = images.get(FIGMA_NODE_ID)
        if not img_url:
            return None
        with httpx.Client(timeout=120.0) as c2:
            ir = c2.get(img_url)
        if ir.status_code == 200:
            return ir.content
    except Exception as e:
        logger.warning("[figma] export failed: %s", e)
    return None
