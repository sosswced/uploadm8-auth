"""Platform OAuth refresh — gated by expiry + worker keepalive sweep.

TikTok access tokens expire ~24h; YouTube ~1h; Meta long-lived / page tokens
still need periodic renewal. Call sites (analytics, publish, catalog) use
``refresh_decrypted_token_for_row``; the worker runs ``run_platform_token_keepalive_loop``.

Keepalive rotates through *all* connected accounts (bumps ``updated_at`` after each
scan) so a batch of 80 cannot starve the rest. Hard refresh failures stamp
``oauth_health='needs_reconnection'`` for Connected Accounts UI.
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

_OAUTH_TABLES_CACHE: Optional[List[str]] = None


def _oauth_tables() -> List[str]:
    global _OAUTH_TABLES_CACHE
    if _OAUTH_TABLES_CACHE is None:
        from core.sql_allowlist import (
            OAUTH_TOKEN_STORAGE_TABLES,
            OAUTH_TOKEN_STORAGE_TABLES_ORDERED,
            assert_relation_name,
        )

        _OAUTH_TABLES_CACHE = [
            assert_relation_name(t, OAUTH_TOKEN_STORAGE_TABLES)
            for t in OAUTH_TOKEN_STORAGE_TABLES_ORDERED
        ]
    return _OAUTH_TABLES_CACHE


async def mark_platform_oauth_health(
    db_pool,
    token_row_id: str,
    health: str,
) -> None:
    """Persist ``ok`` / ``needs_reconnection`` on platform_tokens.oauth_health."""
    if not db_pool or not token_row_id:
        return
    h = str(health or "").strip().lower()
    if h not in ("ok", "needs_reconnection"):
        return
    try:
        async with db_pool.acquire() as conn:
            for table in _oauth_tables():
                try:
                    result = await conn.execute(
                        f"""
                        UPDATE {table}
                           SET oauth_health = $1,
                               updated_at = NOW()
                         WHERE id = $2::uuid
                        """,
                        h,
                        token_row_id,
                    )
                    if result and result != "UPDATE 0":
                        return
                except Exception:
                    continue
    except Exception as e:
        logger.debug("[oauth-health] mark failed row=%s: %s", str(token_row_id)[:8], e)


async def touch_platform_token_row(db_pool, token_row_id: str) -> None:
    """Rotate keepalive cursor: bump updated_at without changing the token blob."""
    if not db_pool or not token_row_id:
        return
    try:
        async with db_pool.acquire() as conn:
            for table in _oauth_tables():
                try:
                    result = await conn.execute(
                        f"UPDATE {table} SET updated_at = NOW() WHERE id = $1::uuid",
                        token_row_id,
                    )
                    if result and result != "UPDATE 0":
                        return
                except Exception:
                    continue
    except Exception as e:
        logger.debug("[token-keepalive] touch failed row=%s: %s", str(token_row_id)[:8], e)


def _refresh_hard_failed(
    platform: str,
    before: Dict[str, Any],
    after: Dict[str, Any],
) -> bool:
    """True when a due refresh left the access token unchanged (provider / missing RT)."""
    plat = str(platform or "").lower()
    if plat in ("tiktok", "youtube"):
        if not str(before.get("refresh_token") or "").strip():
            return True
    b_at = str(before.get("access_token") or "")
    a_at = str((after or {}).get("access_token") or "")
    b_exp = before.get("expires_at")
    a_exp = (after or {}).get("expires_at")
    b_obt = before.get("access_obtained_at")
    a_obt = (after or {}).get("access_obtained_at")
    if a_at and a_at != b_at:
        return False
    if a_exp is not None and a_exp != b_exp:
        return False
    if a_obt is not None and a_obt != b_obt:
        return False
    # Meta long-lived: unchanged after force still usually means soft skip / failure.
    return True


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

    before = dict(decrypted)
    if plat in ("tiktok", "youtube") and not str(before.get("refresh_token") or "").strip():
        await mark_platform_oauth_health(db_pool, token_row_id, "needs_reconnection")
        return decrypted

    try:
        from stages.publish_stage import (
            _refresh_meta_token,
            _refresh_tiktok_token,
            _refresh_youtube_token,
        )

        if plat == "tiktok":
            after = await _refresh_tiktok_token(
                dict(decrypted),
                db_pool=db_pool,
                user_id=str(user_id),
                token_row_id=str(token_row_id),
                force=True,
            )
        elif plat == "youtube":
            after = await _refresh_youtube_token(
                dict(decrypted),
                db_pool=db_pool,
                user_id=str(user_id),
                token_row_id=str(token_row_id),
                force=True,
            )
        else:
            after = await _refresh_meta_token(
                dict(decrypted),
                platform=plat,
                db_pool=db_pool,
                user_id=str(user_id),
                token_row_id=str(token_row_id),
                force=True,
            )

        if _refresh_hard_failed(plat, before, after or {}):
            await mark_platform_oauth_health(db_pool, token_row_id, "needs_reconnection")
        else:
            await mark_platform_oauth_health(db_pool, token_row_id, "ok")
        return after or decrypted
    except Exception as e:
        logger.debug(
            "[oauth-refresh] %s row=%s: %s",
            plat,
            token_row_id[:8] if token_row_id else "",
            e,
        )
        await mark_platform_oauth_health(db_pool, token_row_id, "needs_reconnection")

    return decrypted


async def sweep_platform_token_keepalive(
    db_pool,
    *,
    limit: int = TOKEN_KEEPALIVE_BATCH,
) -> Dict[str, Any]:
    """
    Decrypt + refresh tokens that are within their platform lead window.

    Always rotates ``updated_at`` after each scanned row so the ORDER BY
    ``updated_at ASC`` batch walks the full connected-account set over time.

    Returns counts for logging / metrics. Never raises to the supervisor.
    """
    stats = {
        "scanned": 0,
        "due": 0,
        "refreshed": 0,
        "skipped": 0,
        "errors": 0,
        "needs_reconnect": 0,
    }
    if not db_pool:
        return stats

    try:
        from core.auth import decrypt_blob, init_enc_keys

        init_enc_keys()
    except Exception as e:
        logger.warning("[token-keepalive] enc init failed: %s", e)
        stats["errors"] += 1
        return stats

    rows: List[Any] = []
    try:
        async with db_pool.acquire() as conn:
            for table in _oauth_tables():
                try:
                    fetched = await conn.fetch(
                        f"""
                        SELECT id, user_id, platform, token_blob
                          FROM {table}
                         WHERE revoked_at IS NULL
                         ORDER BY updated_at ASC NULLS FIRST, id ASC
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
            await mark_platform_oauth_health(db_pool, rid, "needs_reconnection")
            stats["needs_reconnect"] += 1
            continue

        if not decrypted:
            stats["errors"] += 1
            await mark_platform_oauth_health(db_pool, rid, "needs_reconnection")
            stats["needs_reconnect"] += 1
            continue

        if not should_refresh_access_token(plat, decrypted, force=False):
            stats["skipped"] += 1
            # Rotate out of the oldest-N batch so other accounts get scanned.
            await touch_platform_token_row(db_pool, rid)
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
            elif after and not _refresh_hard_failed(plat, decrypted, after_blob or {}):
                stats["refreshed"] += 1
            else:
                stats["needs_reconnect"] += 1
                # Ensure cursor rotates even when save_refreshed_token did not run.
                await touch_platform_token_row(db_pool, rid)
        except Exception as e:
            logger.warning(
                "[token-keepalive] refresh failed %s row=%s: %s",
                plat,
                rid[:8],
                e,
            )
            stats["errors"] += 1
            await mark_platform_oauth_health(db_pool, rid, "needs_reconnection")
            stats["needs_reconnect"] += 1
            await touch_platform_token_row(db_pool, rid)

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
            if stats.get("due") or stats.get("errors") or stats.get("needs_reconnect"):
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
