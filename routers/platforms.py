"""
UploadM8 Platform routes — linked accounts listing and disconnect.
OAuth flows live in routers.oauth (mounted from app.py).
"""

import logging

import httpx
from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel, Field

import core.state
from core.deps import get_current_user, get_current_user_readonly
from core.oauth import _revoke_platform_token
from core.audit import log_system_event
from core.helpers import get_plan
from services.workspace import require_can_manage_platforms, resolve_billing_user_id
from services.platform_accounts import (
    _PLATFORM_TOKEN_SELECT,
    fetch_auth_errors_by_token,
    serialize_platform_account,
    serialize_platform_account_flat,
)
from services.platform_profile_refresh import refresh_platform_token_profile
from services.accounts_hub_insights import build_accounts_hub_insights
from services.tiktok_api import (
    TIKTOK_BRANDED_CONTENT_POLICY_URL,
    TIKTOK_MUSIC_USAGE_CONFIRMATION_URL,
    fetch_tiktok_creator_info,
    tiktok_unaudited_mode,
)
from stages import db as db_stage
from stages.publish_stage import decrypt_token

logger = logging.getLogger("uploadm8-api")

router = APIRouter(tags=["platforms"])


class TikTokCreatorInfoRequest(BaseModel):
    account_id: str = Field(..., min_length=1, description="platform_tokens.id UUID")


@router.post("/api/tiktok/creator-info")
async def tiktok_creator_info(
    body: TikTokCreatorInfoRequest,
    user: dict = Depends(get_current_user_readonly),
):
    """
    TikTok Content Posting API: query creator info before rendering export UI.
    See https://developers.tiktok.com/doc/content-sharing-guidelines
    """
    user_id = resolve_billing_user_id(user)
    token_data, identity = await db_stage.load_platform_token_with_identity(
        core.state.db_pool,
        user_id,
        "tiktok",
        token_row_id=str(body.account_id).strip(),
    )
    if not token_data:
        raise HTTPException(404, "TikTok account not found")

    if isinstance(token_data, dict) and token_data.get("kid"):
        try:
            token_data = decrypt_token(token_data)
        except Exception as e:
            raise HTTPException(400, f"Could not decrypt TikTok token: {e}") from e

    access_token = (token_data or {}).get("access_token") if isinstance(token_data, dict) else None
    if not access_token:
        raise HTTPException(400, "TikTok account is missing an access token — reconnect and try again.")

    async with httpx.AsyncClient(timeout=30) as client:
        info, err = await fetch_tiktok_creator_info(client, str(access_token))
    if err:
        raise HTTPException(502, err)

    unaudited = tiktok_unaudited_mode()

    return {
        "ok": True,
        "account_id": str(body.account_id).strip(),
        "identity": identity or {},
        "creator_info": info,
        "unaudited_mode": unaudited,
        "links": {
            "music_usage_confirmation": TIKTOK_MUSIC_USAGE_CONFIRMATION_URL,
            "branded_content_policy": TIKTOK_BRANDED_CONTENT_POLICY_URL,
        },
    }


async def _load_user_platform_accounts(conn, user_id: str):
    auth_errors = await fetch_auth_errors_by_token(conn, user_id)
    rows = await conn.fetch(_PLATFORM_TOKEN_SELECT, user_id)
    return rows, auth_errors


@router.get("/api/platforms")
async def get_platforms(user: dict = Depends(get_current_user_readonly)):
    async with core.state.db_pool.acquire() as conn:
        accounts, auth_errors = await _load_user_platform_accounts(conn, resolve_billing_user_id(user))

    platforms = {}
    for acc in accounts:
        p = acc["platform"]
        if p not in platforms:
            platforms[p] = []
        platforms[p].append(serialize_platform_account(acc, auth_error_by_token=auth_errors))

    plan = get_plan(user.get("subscription_tier", "free"))
    total = sum(len(v) for v in platforms.values())
    return {
        "platforms": platforms,
        "total_accounts": total,
        "max_accounts": plan.get("max_accounts", 1),
        "can_add_more": total < plan.get("max_accounts", 1),
    }


@router.get("/api/accounts/hub-insights")
async def get_accounts_hub_insights(user: dict = Depends(get_current_user_readonly)):
    """ML/AI bundle for Account Groups and Connected Accounts pages."""
    try:
        return await build_accounts_hub_insights(core.state.db_pool, resolve_billing_user_id(user))
    except Exception as e:
        logger.warning("accounts hub-insights failed user_id=%s: %s", user.get("id"), e)
        return {
            "ok": False,
            "m8_engine": {},
            "platform_engagement": [],
            "group_suggestions": [],
            "platform_metrics": {},
            "per_account_metrics": {},
            "per_account_uploads": {},
            "meta_hints": [],
        }


@router.get("/api/platform-accounts")
async def get_platform_accounts(user: dict = Depends(get_current_user_readonly)):
    """Returns flat list of accounts for frontend compatibility"""
    async with core.state.db_pool.acquire() as conn:
        accounts, auth_errors = await _load_user_platform_accounts(conn, resolve_billing_user_id(user))

    return {
        "accounts": [
            serialize_platform_account_flat(acc, acc["platform"], auth_error_by_token=auth_errors)
            for acc in accounts
        ]
    }


@router.get("/api/accounts")
async def get_accounts_simple(user: dict = Depends(get_current_user_readonly)):
    """Simple accounts list for dashboard"""
    async with core.state.db_pool.acquire() as conn:
        accounts, auth_errors = await _load_user_platform_accounts(conn, resolve_billing_user_id(user))
    return [
        {
            "id": str(a["id"]),
            "platform": a["platform"],
            "name": a["account_name"],
            "username": a["account_username"],
            "avatar": serialize_platform_account(a, auth_error_by_token=auth_errors)["avatar"],
            "status": serialize_platform_account(a, auth_error_by_token=auth_errors)["status"],
        }
        for a in accounts
    ]


@router.post("/api/platform-accounts/{account_id}/refresh-profile")
async def refresh_account_profile(
    account_id: str,
    user: dict = Depends(get_current_user),
):
    """Re-fetch provider display name / avatar (fixes TikTok ui-avatars placeholders)."""
    token_owner = resolve_billing_user_id(user)
    async with core.state.db_pool.acquire() as conn:
        updated = await refresh_platform_token_profile(conn, user_id=token_owner, token_id=account_id)
        if not updated:
            raise HTTPException(404, "Account not found or profile could not be refreshed")
        row = await conn.fetchrow(
            "SELECT * FROM platform_tokens WHERE id = $1 AND user_id = $2",
            account_id,
            token_owner,
        )
        auth_errors = await fetch_auth_errors_by_token(conn, token_owner)
    return serialize_platform_account_flat(row, row["platform"], auth_error_by_token=auth_errors)


@router.delete("/api/platforms/{platform}/accounts/{account_id}")
async def disconnect_account(
    platform: str,
    account_id: str,
    request: Request,
    user: dict = Depends(get_current_user),
):
    """
    Disconnect a linked platform account.
    1. Revokes the access token at the provider.
    2. Hard-deletes the platform_tokens row.
    3. Writes a platform_disconnect_log record.
    """
    require_can_manage_platforms(user)
    token_owner = resolve_billing_user_id(user)
    actor_id = str(user["id"])
    ip_addr = request.headers.get("X-Forwarded-For", request.client.host if request.client else None)
    async with core.state.db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, platform, account_id, account_name, token_blob FROM platform_tokens WHERE id = $1 AND user_id = $2",
            account_id,
            token_owner,
        )
        if not row:
            raise HTTPException(404, "Account not found")

        ok, err = await _revoke_platform_token(row["platform"], row["token_blob"])

        await conn.execute(
            "UPDATE platform_tokens SET revoked_at = NOW() WHERE id = $1", row["id"]
        )
        await conn.execute("DELETE FROM platform_tokens WHERE id = $1", row["id"])

        await conn.execute(
            """
            INSERT INTO platform_disconnect_log
                (user_id, platform, account_id, account_name,
                 revoked_at_provider, provider_revoke_error, initiated_by, ip_address)
            VALUES ($1,$2,$3,$4,$5,$6,'self',$7)
            """,
            token_owner,
            row["platform"],
            row["account_id"],
            row["account_name"],
            ok,
            err or None,
            ip_addr,
        )
        await log_system_event(
            conn,
            user_id=token_owner,
            action="PLATFORM_DISCONNECTED",
            event_category="PLATFORM",
            resource_type="platform",
            resource_id=f"{row['platform']}:{row['account_id']}",
            details={
                "platform": row["platform"],
                "account_name": row["account_name"],
                "provider_revoked": ok,
                "provider_error": err,
                "actor_user_id": actor_id,
            },
            severity="WARNING",
        )

    return {"status": "disconnected", "provider_revoked": ok}


@router.delete("/api/platform-accounts/{account_id}")
async def disconnect_account_by_id(
    account_id: str,
    request: Request,
    user: dict = Depends(get_current_user),
):
    """Alias for the disconnect endpoint (used by older frontend code)."""
    return await disconnect_account(
        platform="",
        account_id=account_id,
        request=request,
        user=user,
    )


# OAuth start/callback: see routers.oauth (mounted from app.py).
