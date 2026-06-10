"""TikTok post-settings validation for upload presign/complete."""

from __future__ import annotations

import json
from typing import Any, List

from fastapi import HTTPException

from services.tiktok_api import validate_tiktok_post_settings


async def _tiktok_target_account_ids(conn, bill_id: str, data: Any) -> List[str]:
    """Resolve platform_tokens.id values for TikTok publish targets on this upload."""
    platforms = [str(p).lower() for p in (getattr(data, "platforms", None) or [])]
    if "tiktok" not in platforms:
        return []
    target_accounts = list(getattr(data, "target_accounts", None) or [])
    if target_accounts:
        rows = await conn.fetch(
            """
            SELECT id::text AS id
            FROM platform_tokens
            WHERE user_id = $1
              AND platform = 'tiktok'
              AND revoked_at IS NULL
              AND id = ANY($2::uuid[])
            """,
            bill_id,
            target_accounts,
        )
        return [str(r["id"]) for r in rows]
    rows = await conn.fetch(
        """
        SELECT id::text AS id
        FROM platform_tokens
        WHERE user_id = $1
          AND platform = 'tiktok'
          AND revoked_at IS NULL
        ORDER BY is_primary DESC NULLS LAST, updated_at DESC
        """,
        bill_id,
    )
    return [str(r["id"]) for r in rows]


def _validate_tiktok_post_settings_for_upload(data: Any, tiktok_account_ids: List[str]) -> None:
    if "tiktok" not in [str(p).lower() for p in (getattr(data, "platforms", None) or [])]:
        return
    raw = getattr(data, "tiktok_post_settings", None) or getattr(data, "tiktokPostSettings", None)
    _validate_tiktok_by_account_settings(raw, tiktok_account_ids)


def _validate_tiktok_by_account_settings(raw: Any, tiktok_account_ids: List[str]) -> None:
    if not isinstance(raw, dict) or not raw:
        raise HTTPException(
            400,
            detail={
                "code": "tiktok_settings_required",
                "message": "TikTok posting settings are required. Complete the Post to TikTok section before uploading.",
            },
        )
    by_account = raw.get("by_account")
    if not isinstance(by_account, dict):
        by_account = {}
    missing: List[str] = []
    for acc_id in tiktok_account_ids:
        entry = by_account.get(str(acc_id))
        if not isinstance(entry, dict):
            missing.append(acc_id)
            continue
        errs = validate_tiktok_post_settings(entry)
        if errs:
            raise HTTPException(
                400,
                detail={
                    "code": "tiktok_settings_invalid",
                    "message": f"TikTok export settings invalid: {errs[0]}",
                },
            )
    if missing:
        raise HTTPException(
            400,
            detail={
                "code": "tiktok_settings_incomplete",
                "message": "Complete Post to TikTok settings for every selected TikTok account.",
            },
        )


async def _tiktok_target_account_ids_from_upload(conn, upload: dict) -> List[str]:
    """Resolve TikTok platform_tokens.id values from a persisted upload row."""
    platforms = [str(p).lower() for p in (upload.get("platforms") or [])]
    if "tiktok" not in platforms:
        return []
    bill_id = str(upload.get("user_id") or "")
    target_accounts = list(upload.get("target_accounts") or [])
    if target_accounts:
        rows = await conn.fetch(
            """
            SELECT id::text AS id
            FROM platform_tokens
            WHERE user_id = $1
              AND platform = 'tiktok'
              AND revoked_at IS NULL
              AND id = ANY($2::uuid[])
            """,
            bill_id,
            target_accounts,
        )
        return [str(r["id"]) for r in rows]
    rows = await conn.fetch(
        """
        SELECT id::text AS id
        FROM platform_tokens
        WHERE user_id = $1
          AND platform = 'tiktok'
          AND revoked_at IS NULL
        ORDER BY is_primary DESC NULLS LAST, updated_at DESC
        """,
        bill_id,
    )
    return [str(r["id"]) for r in rows]


async def validate_upload_row_tiktok_settings(conn, upload: dict) -> None:
    """Defense-in-depth: block complete/enqueue if TikTok lacks export settings."""
    platforms = [str(p).lower() for p in (upload.get("platforms") or [])]
    if "tiktok" not in platforms:
        return
    account_ids = await _tiktok_target_account_ids_from_upload(conn, upload)
    user_prefs = upload.get("user_preferences") or {}
    if isinstance(user_prefs, str):
        try:
            user_prefs = json.loads(user_prefs)
        except (json.JSONDecodeError, TypeError):
            user_prefs = {}
    if not isinstance(user_prefs, dict):
        user_prefs = {}
    raw = user_prefs.get("tiktok_post_settings") or user_prefs.get("tiktokPostSettings")
    _validate_tiktok_by_account_settings(raw, account_ids)
