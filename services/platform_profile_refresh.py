"""Re-fetch provider profile (name, username, avatar) for an existing platform_tokens row."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import httpx

from core.auth import decrypt_blob
from core.oauth import mirror_oauth_profile_image_to_r2
from services.tiktok_api import fetch_tiktok_user_profile_for_oauth

logger = logging.getLogger("uploadm8-api.platform_profile_refresh")


def is_degraded_tiktok_identity(
    account_name: Optional[str],
    account_username: Optional[str],
    account_avatar: Optional[str],
) -> bool:
    """True when OAuth stored a synthetic label and no avatar (ui-avatars fallback in UI)."""
    nm = str(account_name or "").strip()
    un = str(account_username or "").strip()
    av = str(account_avatar or "").strip()
    if un and av:
        return False
    if av.startswith("platform-avatars/"):
        return not un and nm.startswith("TikTok ")
    if nm.startswith("TikTok ") and nm not in ("TikTok User", "TikTok"):
        return not un or not av
    if nm in ("TikTok User", "") and not av:
        return True
    return not av and not un


def _access_token_from_blob(token_blob: Any) -> str:
    tok = token_blob if isinstance(token_blob, dict) else {}
    if "kid" in tok and "ciphertext" in tok:
        try:
            tok = decrypt_blob(tok)
        except Exception:
            return ""
    if not isinstance(tok, dict):
        return ""
    return str(tok.get("access_token") or "").strip()


async def refresh_tiktok_token_profile(
    conn,
    *,
    user_id: str,
    token_row: Dict[str, Any],
) -> Optional[Dict[str, str]]:
    """Call TikTok user/info and UPDATE platform_tokens. Returns new identity fields or None."""
    access = _access_token_from_blob(token_row.get("token_blob"))
    if not access:
        return None

    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        prof = await fetch_tiktok_user_profile_for_oauth(client, access)

    account_id = str(prof.get("account_id") or token_row.get("account_id") or "").strip()
    account_name = str(prof.get("account_name") or "").strip()
    account_username = str(prof.get("account_username") or "").strip()
    account_avatar = str(prof.get("account_avatar") or "").strip()

    if not account_name and account_username:
        account_name = account_username
    if account_name in ("", "TikTok User") and account_id:
        account_name = f"TikTok {account_id[-6:]}" if len(account_id) > 6 else f"TikTok {account_id}"

    if not (account_name or account_username or account_avatar):
        return None

    if account_avatar.startswith("http"):
        try:
            mirrored = await mirror_oauth_profile_image_to_r2(user_id, "tiktok", account_avatar)
            if mirrored:
                account_avatar = mirrored
        except Exception as e:
            logger.debug("refresh_tiktok avatar mirror skipped: %s", e)

    await conn.execute(
        """
        UPDATE platform_tokens
        SET account_name = COALESCE(NULLIF($1, ''), account_name),
            account_username = COALESCE(NULLIF($2, ''), account_username),
            account_avatar = COALESCE(NULLIF($3, ''), account_avatar),
            updated_at = NOW()
        WHERE id = $4 AND user_id = $5
        """,
        account_name,
        account_username,
        account_avatar,
        token_row["id"],
        user_id,
    )
    return {
        "account_name": account_name,
        "account_username": account_username,
        "account_avatar": account_avatar,
    }


async def refresh_platform_token_profile(conn, *, user_id: str, token_id: str) -> Optional[dict]:
    row = await conn.fetchrow(
        """
        SELECT id, platform, account_id, account_name, account_username, account_avatar, token_blob
        FROM platform_tokens
        WHERE id = $1 AND user_id = $2 AND revoked_at IS NULL
        """,
        token_id,
        user_id,
    )
    if not row:
        return None
    plat = str(row["platform"] or "").strip().lower()
    if plat == "tiktok":
        return await refresh_tiktok_token_profile(conn, user_id=user_id, token_row=dict(row))
    return None
