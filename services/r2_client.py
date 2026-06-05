"""
Canonical async R2 upload helper for scripts and marketing image generation.

Wraps ``core.r2.put_object_bytes`` so callers can ``await r2_put_object_bytes(...)``
without blocking the event loop.
"""

from __future__ import annotations

import asyncio

from core.r2 import put_object_bytes


async def r2_put_object_bytes(*, key: str, body: bytes, content_type: str) -> str:
    """Upload bytes to R2 and return the normalized object key."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: put_object_bytes(key, body, content_type),
    )
