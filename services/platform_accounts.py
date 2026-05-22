"""Shared helpers for platform token listing and group upload aggregates."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import asyncpg

from core.r2 import resolve_stored_account_avatar_url

# Publish / OAuth error codes that mean the user should reconnect the account.
AUTH_RECONNECT_ERROR_CODES = frozenset(
    {
        "PLATFORM_AUTH_FAILED",
        "AUTH_FAILED",
        "TOKEN_EXPIRED",
        "NOT_CONNECTED",
        "NO_TOKEN",
    }
)

_PLATFORM_TOKEN_SELECT = """
    SELECT DISTINCT ON (platform, account_id, COALESCE(account_username,''), COALESCE(account_name,''))
        id,
        platform,
        account_id,
        account_name,
        account_username,
        account_avatar,
        is_primary,
        created_at,
        last_oauth_reconnect_at,
        last_used_at
    FROM platform_tokens
    WHERE user_id = $1
      AND revoked_at IS NULL
      AND account_id IS NOT NULL AND account_id <> ''
      AND (COALESCE(account_username,'') <> '' OR COALESCE(account_name,'') <> '')
    ORDER BY platform, account_id, COALESCE(account_username,''), COALESCE(account_name,''), created_at DESC
"""


def account_status(token_id: str, auth_error_by_token: Mapping[str, str]) -> str:
    code = (auth_error_by_token.get(str(token_id)) or "").strip()
    if code in AUTH_RECONNECT_ERROR_CODES:
        return "needs_reconnection"
    return "active"


def serialize_platform_account(
    row: asyncpg.Record | Mapping[str, Any],
    *,
    auth_error_by_token: Optional[Mapping[str, str]] = None,
) -> dict:
    """API shape for one connected platform_tokens row."""
    auth_error_by_token = auth_error_by_token or {}
    tid = str(row["id"])
    created = row.get("created_at")
    reconnect = row.get("last_oauth_reconnect_at")
    last_used = row.get("last_used_at")
    status = account_status(tid, auth_error_by_token)
    created_iso = created.isoformat() if created else None
    return {
        "id": tid,
        "account_id": row.get("account_id"),
        "name": row.get("account_name"),
        "username": row.get("account_username"),
        "avatar": resolve_stored_account_avatar_url(row.get("account_avatar")),
        "is_primary": row.get("is_primary"),
        "status": status,
        "connected_at": created_iso,
        "first_connected_at": created_iso,
        "last_reconnected_at": reconnect.isoformat() if reconnect else None,
        "last_used_at": last_used.isoformat() if last_used else None,
    }


def serialize_platform_account_flat(
    row: asyncpg.Record | Mapping[str, Any],
    platform: str,
    *,
    auth_error_by_token: Optional[Mapping[str, str]] = None,
) -> dict:
    base = serialize_platform_account(row, auth_error_by_token=auth_error_by_token)
    return {
        "id": base["id"],
        "platform": platform,
        "account_id": base["account_id"],
        "account_name": base["name"],
        "account_username": base["username"],
        "account_avatar_url": base["avatar"],
        "is_primary": base["is_primary"],
        "status": base["status"],
        "connected_at": base["connected_at"],
        "first_connected_at": base["first_connected_at"],
        "last_reconnected_at": base["last_reconnected_at"],
        "last_used_at": base["last_used_at"],
    }


async def fetch_auth_errors_by_token(conn: asyncpg.Connection, user_id: str) -> Dict[str, str]:
    """
    Latest publish failure per platform_tokens.id from uploads.platform_results.
    """
    rows = await conn.fetch(
        """
        WITH elems AS (
            SELECT
                u.updated_at,
                elem->>'token_row_id' AS token_id,
                elem->>'error_code' AS error_code,
                COALESCE((elem->>'success')::boolean, false) AS success
            FROM uploads u
            CROSS JOIN LATERAL jsonb_array_elements(
                CASE
                    WHEN u.platform_results IS NULL THEN '[]'::jsonb
                    WHEN jsonb_typeof(u.platform_results) = 'array' THEN u.platform_results
                    ELSE '[]'::jsonb
                END
            ) AS elem
            WHERE u.user_id = $1
              AND elem->>'token_row_id' IS NOT NULL
              AND elem->>'token_row_id' <> ''
        )
        SELECT DISTINCT ON (token_id) token_id, error_code, success
        FROM elems
        ORDER BY token_id, updated_at DESC NULLS LAST
        """,
        user_id,
    )
    out: Dict[str, str] = {}
    for r in rows:
        if r["success"]:
            continue
        code = (r["error_code"] or "").strip()
        if code in AUTH_RECONNECT_ERROR_CODES:
            out[str(r["token_id"])] = code
    return out


async def fetch_group_upload_counts(conn: asyncpg.Connection, user_id: str) -> Dict[str, int]:
    """Completed/succeeded uploads whose target_accounts overlap each group's account_ids."""
    rows = await conn.fetch(
        """
        SELECT ag.id::text AS group_id, COUNT(DISTINCT u.id)::int AS uploads_count
        FROM account_groups ag
        LEFT JOIN uploads u
          ON u.user_id = ag.user_id
         AND COALESCE(cardinality(u.target_accounts), 0) > 0
         AND u.target_accounts && ag.account_ids
         AND u.status IN ('completed', 'succeeded', 'partial')
        WHERE ag.user_id = $1
        GROUP BY ag.id
        """,
        user_id,
    )
    return {str(r["group_id"]): int(r["uploads_count"] or 0) for r in rows}
