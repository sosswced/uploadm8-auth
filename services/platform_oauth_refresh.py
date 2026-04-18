"""Proactive OAuth refresh before platform Content/Data API calls.

TikTok access tokens expire ~24h; YouTube ~1h; Meta long-lived tokens still need renewal.
Used by analytics, per-upload sync, worker analytics loop, and catalog (catalog also calls
publish_stage refresh directly with token_row_id).
"""
from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger("uploadm8-api")


async def refresh_decrypted_token_for_row(
    platform: str,
    decrypted: Dict[str, Any],
    *,
    db_pool,
    user_id: str,
    token_row_id: str,
) -> Dict[str, Any]:
    """
    Refresh stored OAuth token for one platform_tokens row and persist when possible.
    Returns updated plaintext dict (or original on failure / unsupported platform).
    """
    if not decrypted or not db_pool or not user_id or not token_row_id:
        return decrypted

    plat = str(platform or "").lower()
    if plat not in ("tiktok", "youtube", "instagram", "facebook"):
        return decrypted

    try:
        from stages.publish_stage import (
            _refresh_meta_token,
            _refresh_tiktok_token,
            _refresh_youtube_token,
        )

        if plat == "tiktok":
            return await _refresh_tiktok_token(
                dict(decrypted),
                db_pool=db_pool,
                user_id=str(user_id),
                token_row_id=str(token_row_id),
            )
        if plat == "youtube":
            return await _refresh_youtube_token(
                dict(decrypted),
                db_pool=db_pool,
                user_id=str(user_id),
                token_row_id=str(token_row_id),
            )
        if plat in ("instagram", "facebook"):
            return await _refresh_meta_token(
                dict(decrypted),
                platform=plat,
                db_pool=db_pool,
                user_id=str(user_id),
                token_row_id=str(token_row_id),
            )
    except Exception as e:
        logger.debug("[oauth-refresh] %s row=%s: %s", plat, token_row_id[:8] if token_row_id else "", e)

    return decrypted
