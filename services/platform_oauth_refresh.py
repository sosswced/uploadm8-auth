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
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from core.platform_token_expiry import (
    is_permanent_oauth_error,
    parse_access_expires_at,
    parse_refresh_expires_at,
    should_refresh_access_token,
)

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

# A provider outage must not read as a dead connection. Only declare
# ``needs_reconnection`` once the provider says the grant is gone, the refresh
# token has lapsed, or we have failed this many times *and* the access token has
# already expired (i.e. the connection really is unusable, not just unreachable).
OAUTH_FAIL_GRACE_ATTEMPTS = max(
    2,
    int(os.environ.get("PLATFORM_TOKEN_FAIL_GRACE_ATTEMPTS") or 6),
)
# Exponential backoff between retries of a failing row, so one broken account
# cannot consume the sweep budget every cycle.
OAUTH_RETRY_BACKOFF_BASE_SEC = 300
OAUTH_RETRY_BACKOFF_MAX_SEC = 4 * 3600

# Applied by the OAuth connect/reconnect path. Clears keepalive failure state so
# a freshly reconnected account is not left in backoff or reported as degraded,
# and drops the expiry mirrors so the next sweep re-stamps them from the new
# token (NULL mirrors sort first in the deadline cursor).
OAUTH_RECONNECT_RESET_SQL = (
    "oauth_health = 'ok', oauth_fail_count = 0, oauth_last_error = NULL, "
    "oauth_next_retry_at = NULL, oauth_reconnect_alert_at = NULL, "
    "access_expires_at = NULL, refresh_expires_at = NULL"
)

_OAUTH_TABLES_CACHE: Optional[List[str]] = None
# Per-table capability probe: legacy installs may predate migration 1105.
_OAUTH_TABLE_HAS_KEEPALIVE_COLS: Dict[str, bool] = {}


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


async def _patch_token_row(
    db_pool,
    token_row_id: str,
    assignments: Dict[str, Any],
    *,
    bump_fail_count: bool = False,
) -> bool:
    """
    Write allowlisted keepalive bookkeeping onto whichever table owns the row.

    Returns False when the columns do not exist yet (pre-migration install), so
    callers can fall back to the legacy ``updated_at`` touch.
    """
    if not db_pool or not token_row_id or (not assignments and not bump_fail_count):
        return False

    from core.sql_allowlist import (
        OAUTH_KEEPALIVE_PATCH_COLUMNS,
        assert_set_fragments_columns,
    )

    sets: List[str] = []
    values: List[Any] = []
    for col, val in assignments.items():
        sets.append(f"{col} = ${len(values) + 1}")
        values.append(val)
    if bump_fail_count:
        sets.append("oauth_fail_count = COALESCE(oauth_fail_count, 0) + 1")
    sets.append("updated_at = NOW()")
    assert_set_fragments_columns(sets, OAUTH_KEEPALIVE_PATCH_COLUMNS)

    try:
        async with db_pool.acquire() as conn:
            for table in _oauth_tables():
                if _OAUTH_TABLE_HAS_KEEPALIVE_COLS.get(table) is False:
                    continue
                try:
                    result = await conn.execute(
                        f"UPDATE {table} SET {', '.join(sets)} "
                        f"WHERE id = ${len(values) + 1}::uuid",
                        *values,
                        token_row_id,
                    )
                except Exception as e:
                    if "column" in str(e).lower():
                        _OAUTH_TABLE_HAS_KEEPALIVE_COLS[table] = False
                    continue
                _OAUTH_TABLE_HAS_KEEPALIVE_COLS[table] = True
                if result and result != "UPDATE 0":
                    return True
    except Exception as e:
        logger.debug("[oauth-keepalive] patch failed row=%s: %s", str(token_row_id)[:8], e)
    return False


def _retry_backoff_seconds(fail_count: int) -> int:
    """Exponential backoff, capped, from the number of consecutive failures."""
    n = max(1, int(fail_count or 1))
    return min(OAUTH_RETRY_BACKOFF_MAX_SEC, OAUTH_RETRY_BACKOFF_BASE_SEC * (2 ** (n - 1)))


def _should_declare_dead(
    *,
    platform: str,
    blob: Optional[Dict[str, Any]],
    fail_count: int,
    error: Any,
    now: Optional[datetime] = None,
) -> bool:
    """
    Is this connection genuinely unusable, or just temporarily unreachable?

    Declares death only on an explicit provider rejection, a lapsed refresh
    token, or repeated failure *after* the access token has already expired.
    """
    if is_permanent_oauth_error(error):
        return True

    now = now or datetime.now(timezone.utc)
    refresh_exp = parse_refresh_expires_at(blob or {}, platform=platform, now=now)
    if refresh_exp is not None and now >= refresh_exp:
        return True

    if int(fail_count or 0) < OAUTH_FAIL_GRACE_ATTEMPTS:
        return False

    # Repeated failures while the access token is still valid usually means the
    # provider is unreachable, not that the user revoked us.
    access_exp = parse_access_expires_at(blob or {}, now=now)
    if access_exp is None:
        return True
    return now >= access_exp


async def record_keepalive_scan(
    db_pool,
    token_row_id: str,
    *,
    platform: str = "",
    blob: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Row was read and decrypted fine but is not due for refresh yet.

    Refreshes the expiry mirrors (which also advances the deadline cursor) and
    records that the sweep saw it. Deliberately leaves ``oauth_health`` and the
    failure tally alone: a revoked connection can still hold a valid access
    token for a while, and must not be marked healthy on that basis.
    """
    now = datetime.now(timezone.utc)
    patch: Dict[str, Any] = {"oauth_last_verified_at": now}
    patch.update(_expiry_mirror_patch(platform, blob, now=now))
    if not await _patch_token_row(db_pool, token_row_id, patch):
        await touch_platform_token_row(db_pool, token_row_id)


async def record_keepalive_success(
    db_pool,
    token_row_id: str,
    *,
    platform: str = "",
    blob: Optional[Dict[str, Any]] = None,
) -> None:
    """Provider confirmed the grant: clear failure state and refresh mirrors."""
    now = datetime.now(timezone.utc)
    patch: Dict[str, Any] = {
        "oauth_health": "ok",
        "oauth_fail_count": 0,
        "oauth_last_error": None,
        "oauth_next_retry_at": None,
        "oauth_last_verified_at": now,
    }
    patch.update(_expiry_mirror_patch(platform, blob, now=now))
    if not await _patch_token_row(db_pool, token_row_id, patch):
        await mark_platform_oauth_health(db_pool, token_row_id, "ok")


async def record_keepalive_failure(
    db_pool,
    token_row_id: str,
    *,
    platform: str,
    blob: Optional[Dict[str, Any]] = None,
    error: Any = None,
    fail_count: int = 0,
) -> bool:
    """
    Record one failed refresh. Returns True when the connection is now dead.

    Transient failures schedule a backoff retry and leave health alone so the
    user is not told to reconnect over a provider blip.
    """
    now = datetime.now(timezone.utc)
    attempts = int(fail_count or 0) + 1
    dead = _should_declare_dead(
        platform=platform, blob=blob, fail_count=attempts, error=error, now=now
    )

    patch: Dict[str, Any] = {
        "oauth_last_failure_at": now,
        "oauth_last_error": (str(error)[:200] if error is not None else None),
        "oauth_next_retry_at": now + timedelta(seconds=_retry_backoff_seconds(attempts)),
    }
    if dead:
        patch["oauth_health"] = "needs_reconnection"
    patch.update(_expiry_mirror_patch(platform, blob, now=now))

    if not await _patch_token_row(db_pool, token_row_id, patch, bump_fail_count=True):
        # Pre-migration install: preserve the old behaviour rather than silently
        # dropping the signal, but still only when the grant is really gone.
        if dead:
            await mark_platform_oauth_health(db_pool, token_row_id, "needs_reconnection")
        else:
            await touch_platform_token_row(db_pool, token_row_id)
    return dead


def _expiry_mirror_patch(
    platform: str,
    blob: Optional[Dict[str, Any]],
    *,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Plaintext expiry columns derived from the decrypted blob."""
    if not blob:
        return {}
    now = now or datetime.now(timezone.utc)
    out: Dict[str, Any] = {
        "access_expires_at": parse_access_expires_at(blob, now=now),
        "refresh_expires_at": parse_refresh_expires_at(blob, platform=platform, now=now),
    }
    if blob.get("access_non_expiring") is True:
        out["access_non_expiring"] = True
    elif blob.get("expires_at") not in (None, ""):
        out["access_non_expiring"] = False
    return out


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
    fail_count: int = 0,
) -> Dict[str, Any]:
    """
    Refresh stored OAuth token for one platform_tokens row and persist when possible.

    Skips the provider call when the access token is still outside the platform
    lead-time window (unless ``force=True``).

    ``fail_count`` is the row's consecutive-failure tally; only the keepalive
    sweep passes it, because only the sweep may retire a connection by count.
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
            await record_keepalive_failure(
                db_pool,
                token_row_id,
                platform=plat,
                blob=after or before,
                error="refresh_returned_unchanged_token",
                fail_count=fail_count,
            )
        else:
            await record_keepalive_success(
                db_pool, token_row_id, platform=plat, blob=after or before
            )
        return after or decrypted
    except Exception as e:
        logger.debug(
            "[oauth-refresh] %s row=%s: %s",
            plat,
            token_row_id[:8] if token_row_id else "",
            e,
        )
        # Transient provider faults must not look like a revoked grant.
        await record_keepalive_failure(
            db_pool,
            token_row_id,
            platform=plat,
            blob=before,
            error=e,
            fail_count=fail_count,
        )

    return decrypted


async def _fetch_keepalive_candidates(conn, table: str, limit: int) -> List[Any]:
    """
    Rows most in need of refresh, soonest deadline first.

    Access expiry lives inside the encrypted blob, so the sweep keeps a plaintext
    ``access_expires_at`` mirror to order by. Rows with no mirror yet sort first
    and get stamped, which backfills the column without a data migration. Rows
    inside their retry backoff are skipped so a broken account cannot consume the
    whole batch on every cycle.
    """
    if _OAUTH_TABLE_HAS_KEEPALIVE_COLS.get(table) is not False:
        try:
            fetched = await conn.fetch(
                f"""
                SELECT id, user_id, platform, token_blob,
                       COALESCE(oauth_fail_count, 0) AS oauth_fail_count,
                       oauth_health
                  FROM {table}
                 WHERE revoked_at IS NULL
                   AND (oauth_next_retry_at IS NULL OR oauth_next_retry_at <= NOW())
                 ORDER BY access_expires_at ASC NULLS FIRST, updated_at ASC, id ASC
                 LIMIT $1
                """,
                int(limit),
            )
            _OAUTH_TABLE_HAS_KEEPALIVE_COLS[table] = True
            return list(fetched or [])
        except Exception as e:
            if "column" in str(e).lower():
                _OAUTH_TABLE_HAS_KEEPALIVE_COLS[table] = False
            else:
                return []

    # Pre-migration / legacy table: oldest-cursor rotation.
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
        return list(fetched or [])
    except Exception:
        return []


async def sweep_platform_token_keepalive(
    db_pool,
    *,
    limit: int = TOKEN_KEEPALIVE_BATCH,
) -> Dict[str, Any]:
    """
    Decrypt + refresh tokens that are within their platform lead window.

    Sweeps every token table, soonest access-expiry first, and stamps the
    plaintext expiry mirrors as it goes. Failures back off and only retire a
    connection once it is genuinely unusable — a provider outage must not tell
    users to reconnect.

    Returns counts for logging / metrics. Never raises to the supervisor.
    """
    stats = {
        "scanned": 0,
        "due": 0,
        "refreshed": 0,
        "skipped": 0,
        "errors": 0,
        "needs_reconnect": 0,
        "refresh_expired": 0,
        "deferred": 0,
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
            # Every token table gets swept — a row parked in a legacy table would
            # otherwise never refresh and would simply expire.
            for table in _oauth_tables():
                remaining = int(limit) - len(rows)
                if remaining <= 0:
                    break
                fetched = await _fetch_keepalive_candidates(conn, table, remaining)
                if fetched:
                    rows.extend(fetched)
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

        fail_count = int(row.get("oauth_fail_count") or 0)
        raw_blob = row.get("token_blob")
        decrypted: Optional[Dict[str, Any]] = None
        decrypt_error: Any = None
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
            decrypt_error = e
            logger.debug("[token-keepalive] decrypt failed row=%s: %s", rid[:8], e)

        if not decrypted:
            # Undecryptable blob may be a key-rotation problem rather than a
            # revoked grant, so it goes through the same grace path.
            stats["errors"] += 1
            if await record_keepalive_failure(
                db_pool,
                rid,
                platform=plat,
                blob=None,
                error=decrypt_error or "token_blob_unreadable",
                fail_count=fail_count,
            ):
                stats["needs_reconnect"] += 1
            else:
                stats["deferred"] += 1
            continue

        # Refresh-token death is terminal: no sweep can recover it, only the user.
        refresh_exp = parse_refresh_expires_at(decrypted, platform=plat)
        if refresh_exp is not None and datetime.now(timezone.utc) >= refresh_exp:
            stats["refresh_expired"] += 1
            stats["needs_reconnect"] += 1
            await record_keepalive_failure(
                db_pool,
                rid,
                platform=plat,
                blob=decrypted,
                error="refresh_token_expired",
                fail_count=fail_count,
            )
            continue

        if not should_refresh_access_token(plat, decrypted, force=False):
            stats["skipped"] += 1
            # Stamp the expiry mirrors so the next sweep can order by deadline
            # instead of decrypting this row again.
            await record_keepalive_scan(db_pool, rid, platform=plat, blob=decrypted)
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
                fail_count=fail_count,
            )
            after = str((after_blob or {}).get("access_token") or "")
            if after and after != before:
                stats["refreshed"] += 1
            elif after and not _refresh_hard_failed(plat, decrypted, after_blob or {}):
                stats["refreshed"] += 1
            elif (row.get("oauth_health") or "") == "needs_reconnection":
                stats["needs_reconnect"] += 1
            else:
                stats["deferred"] += 1
        except Exception as e:
            logger.warning(
                "[token-keepalive] refresh failed %s row=%s: %s",
                plat,
                rid[:8],
                e,
            )
            stats["errors"] += 1
            if await record_keepalive_failure(
                db_pool,
                rid,
                platform=plat,
                blob=decrypted,
                error=e,
                fail_count=fail_count,
            ):
                stats["needs_reconnect"] += 1
            else:
                stats["deferred"] += 1

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
