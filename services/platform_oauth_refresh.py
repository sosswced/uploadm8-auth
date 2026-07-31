"""Platform OAuth refresh — gated by expiry + worker keepalive sweep.

TikTok access tokens expire ~24h; YouTube ~1h; Meta long-lived / page tokens
still need periodic renewal. Call sites (analytics, publish, catalog) use
``refresh_decrypted_token_for_row``; the worker runs ``run_platform_token_keepalive_loop``.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

from core.platform_token_expiry import should_refresh_access_token

logger = logging.getLogger("uploadm8-api")

# How often the worker sweeps all connected accounts.
TOKEN_KEEPALIVE_INTERVAL_SEC = max(
    60,
    int(os.environ.get("PLATFORM_TOKEN_KEEPALIVE_INTERVAL_SEC") or 300),
)
TOKEN_KEEPALIVE_BATCH = max(
    10,
    int(os.environ.get("PLATFORM_TOKEN_KEEPALIVE_BATCH") or 80),
)


async def refresh_decrypted_token_for_row(
    platform: str,
    decrypted: Dict[str, Any],
    *,
    db_pool,
    user_id: str,
    token_row_id: str,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Refresh stored OAuth token for one platform_tokens row and persist when possible.

    Skips the provider call when the access token is still outside the platform
    lead-time window (unless ``force=True``).
    """
    if not decrypted or not db_pool or not user_id or not token_row_id:
        return decrypted

    plat = str(platform or "").lower()
    if plat not in ("tiktok", "youtube", "instagram", "facebook"):
        return decrypted

    if not should_refresh_access_token(plat, decrypted, force=force):
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
                force=True,
            )
        if plat == "youtube":
            return await _refresh_youtube_token(
                dict(decrypted),
                db_pool=db_pool,
                user_id=str(user_id),
                token_row_id=str(token_row_id),
                force=True,
            )
        if plat in ("instagram", "facebook"):
            return await _refresh_meta_token(
                dict(decrypted),
                platform=plat,
                db_pool=db_pool,
                user_id=str(user_id),
                token_row_id=str(token_row_id),
                force=True,
            )
    except Exception as e:
        logger.debug(
            "[oauth-refresh] %s row=%s: %s",
            plat,
            token_row_id[:8] if token_row_id else "",
            e,
        )

    return decrypted


async def sweep_platform_token_keepalive(
    db_pool,
    *,
    limit: int = TOKEN_KEEPALIVE_BATCH,
) -> Dict[str, Any]:
    """
    Decrypt + refresh tokens that are within their platform lead window.

    Returns counts for logging / metrics. Never raises to the supervisor.
    """
    stats = {
        "scanned": 0,
        "due": 0,
        "refreshed": 0,
        "skipped": 0,
        "errors": 0,
    }
    if not db_pool:
        return stats

    try:
        from core.auth import decrypt_blob, init_enc_keys
        from core.sql_allowlist import (
            OAUTH_TOKEN_STORAGE_TABLES,
            OAUTH_TOKEN_STORAGE_TABLES_ORDERED,
            assert_relation_name,
        )

        init_enc_keys()
    except Exception as e:
        logger.warning("[token-keepalive] enc init failed: %s", e)
        stats["errors"] += 1
        return stats

    rows: List[Any] = []
    try:
        async with db_pool.acquire() as conn:
            for raw_table in OAUTH_TOKEN_STORAGE_TABLES_ORDERED:
                table = assert_relation_name(raw_table, OAUTH_TOKEN_STORAGE_TABLES)
                try:
                    fetched = await conn.fetch(
                        f"""
                        SELECT id, user_id, platform, token_blob
                          FROM {table}
                         WHERE revoked_at IS NULL
                         ORDER BY updated_at ASC NULLS FIRST
                         LIMIT $1
                        """,
                        int(limit),
                    )
                except Exception:
                    continue
                if fetched:
                    rows = list(fetched)
                    break
    except Exception as e:
        logger.warning("[token-keepalive] list failed: %s", e)
        stats["errors"] += 1
        return stats

    for row in rows:
        stats["scanned"] += 1
        plat = str(row.get("platform") or "").lower()
        uid = str(row.get("user_id") or "")
        rid = str(row.get("id") or "")
        if not plat or not uid or not rid:
            continue

        raw_blob = row.get("token_blob")
        decrypted: Optional[Dict[str, Any]] = None
        try:
            import json

            enc = raw_blob
            if isinstance(enc, str):
                enc = json.loads(enc)
                if isinstance(enc, str):
                    enc = json.loads(enc)
            if isinstance(enc, dict):
                decrypted = decrypt_blob(enc) or {}
        except Exception as e:
            logger.debug("[token-keepalive] decrypt failed row=%s: %s", rid[:8], e)
            stats["errors"] += 1
            continue

        if not decrypted:
            stats["errors"] += 1
            continue

        if not should_refresh_access_token(plat, decrypted, force=False):
            stats["skipped"] += 1
            continue

        stats["due"] += 1
        before = str(decrypted.get("access_token") or "")
        try:
            after_blob = await refresh_decrypted_token_for_row(
                plat,
                decrypted,
                db_pool=db_pool,
                user_id=uid,
                token_row_id=rid,
                force=True,
            )
            after = str((after_blob or {}).get("access_token") or "")
            if after and after != before:
                stats["refreshed"] += 1
            elif after:
                # Provider may return same token but we stamped expiry — count as refreshed.
                stats["refreshed"] += 1
        except Exception as e:
            logger.warning(
                "[token-keepalive] refresh failed %s row=%s: %s",
                plat,
                rid[:8],
                e,
            )
            stats["errors"] += 1

    return stats


async def run_platform_token_keepalive_loop(db_pool, shutdown_event: asyncio.Event) -> None:
    """Worker background loop: proactive refresh before platform access expiry."""
    logger.info(
        "[token-keepalive] loop start interval=%ss batch=%s",
        TOKEN_KEEPALIVE_INTERVAL_SEC,
        TOKEN_KEEPALIVE_BATCH,
    )
    # Small stagger so pool is warm after boot.
    try:
        await asyncio.wait_for(shutdown_event.wait(), timeout=15.0)
        return
    except asyncio.TimeoutError:
        pass

    while not shutdown_event.is_set():
        try:
            stats = await sweep_platform_token_keepalive(db_pool)
            if stats.get("due") or stats.get("errors"):
                logger.info("[token-keepalive] sweep %s", stats)
            else:
                logger.debug("[token-keepalive] sweep %s", stats)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("[token-keepalive] sweep error: %s", e)

        try:
            await asyncio.wait_for(
                shutdown_event.wait(),
                timeout=float(TOKEN_KEEPALIVE_INTERVAL_SEC),
            )
            return
        except asyncio.TimeoutError:
            continue
