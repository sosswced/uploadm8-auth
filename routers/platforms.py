"""
UploadM8 Platform routes — linked accounts listing and disconnect.
OAuth flows live in routers.oauth (mounted from app.py).
"""

import logging

from fastapi import APIRouter, HTTPException, Depends, Request

import core.state
from core.deps import get_current_user
from core.oauth import _revoke_platform_token
from core.audit import log_system_event
from core.helpers import get_plan
from core.r2 import resolve_stored_account_avatar_url

logger = logging.getLogger("uploadm8-api")

router = APIRouter(tags=["platforms"])


@router.get("/api/platforms")
async def get_platforms(user: dict = Depends(get_current_user)):
    async with core.state.db_pool.acquire() as conn:
        accounts = await conn.fetch("""
            SELECT DISTINCT ON (platform, account_id, COALESCE(account_username,''), COALESCE(account_name,''))
                id, platform, account_id, account_name, account_username, account_avatar, is_primary, created_at
            FROM platform_tokens
            WHERE user_id = $1
              AND revoked_at IS NULL
              AND account_id IS NOT NULL AND account_id <> ''
              AND (COALESCE(account_username,'') <> '' OR COALESCE(account_name,'') <> '')
            ORDER BY platform, account_id, COALESCE(account_username,''), COALESCE(account_name,''), created_at DESC
        """, user["id"])

    platforms = {}
    for acc in accounts:
        p = acc["platform"]
        if p not in platforms: platforms[p] = []
        platforms[p].append(
            {
                "id": str(acc["id"]),
                "account_id": acc["account_id"],
                "name": acc["account_name"],
                "username": acc["account_username"],
                "avatar": resolve_stored_account_avatar_url(acc["account_avatar"]),
                "is_primary": acc["is_primary"],
                "status": "active",
                "connected_at": acc["created_at"].isoformat() if acc["created_at"] else None,
            }
        )

    plan = get_plan(user.get("subscription_tier", "free"))
    total = sum(len(v) for v in platforms.values())
    return {"platforms": platforms, "total_accounts": total, "max_accounts": plan.get("max_accounts", 1), "can_add_more": total < plan.get("max_accounts", 1)}

# Alias endpoint for frontend compatibility
@router.get("/api/platform-accounts")
async def get_platform_accounts(user: dict = Depends(get_current_user)):
    """Returns flat list of accounts for frontend compatibility"""
    async with core.state.db_pool.acquire() as conn:
        accounts = await conn.fetch("""
            SELECT DISTINCT ON (platform, account_id, COALESCE(account_username,''), COALESCE(account_name,''))
                id, platform, account_id, account_name, account_username, account_avatar, is_primary, created_at
            FROM platform_tokens
            WHERE user_id = $1
              AND revoked_at IS NULL
              AND account_id IS NOT NULL AND account_id <> ''
              AND (COALESCE(account_username,'') <> '' OR COALESCE(account_name,'') <> '')
            ORDER BY platform, account_id, COALESCE(account_username,''), COALESCE(account_name,''), created_at DESC
        """, user["id"])

    result = []
    for acc in accounts:
        result.append({
            "id": str(acc["id"]),
            "platform": acc["platform"],
            "account_id": acc["account_id"],
            "account_name": acc["account_name"],
            "account_username": acc["account_username"],
            "account_avatar_url": resolve_stored_account_avatar_url(acc["account_avatar"]),
            "is_primary": acc["is_primary"],
            "status": "active",
            "connected_at": acc["created_at"].isoformat() if acc["created_at"] else None,
        })
    return {"accounts": result}

@router.get("/api/accounts")
async def get_accounts_simple(user: dict = Depends(get_current_user)):
    """Simple accounts list for dashboard"""
    async with core.state.db_pool.acquire() as conn:
        accounts = await conn.fetch("SELECT DISTINCT ON (platform, account_id) id, platform, account_name, account_username, account_avatar FROM platform_tokens WHERE user_id = $1 AND revoked_at IS NULL AND account_id IS NOT NULL AND account_id <> '' AND (COALESCE(account_username,'') <> '' OR COALESCE(account_name,'') <> '') ORDER BY platform, account_id, created_at DESC", user["id"])
    return [
        {
            "id": str(a["id"]),
            "platform": a["platform"],
            "name": a["account_name"],
            "username": a["account_username"],
            "avatar": resolve_stored_account_avatar_url(a["account_avatar"]),
            "status": "active",
        }
        for a in accounts
    ]

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
    ip_addr = request.headers.get("X-Forwarded-For", request.client.host if request.client else None)
    async with core.state.db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, platform, account_id, account_name, token_blob FROM platform_tokens WHERE id = $1 AND user_id = $2",
            account_id,
            user["id"],
        )
        if not row:
            raise HTTPException(404, "Account not found")

        # Revoke token at provider
        ok, err = await _revoke_platform_token(row["platform"], row["token_blob"])

        # Hard-delete (mark revoked_at first to satisfy the partial unique index,
        # then delete so no stale token lingers)
        await conn.execute(
            "UPDATE platform_tokens SET revoked_at = NOW() WHERE id = $1", row["id"]
        )
        await conn.execute("DELETE FROM platform_tokens WHERE id = $1", row["id"])

        # Audit log — platform_disconnect_log (existing) + system_event_log (new)
        await conn.execute(
            """
            INSERT INTO platform_disconnect_log
                (user_id, platform, account_id, account_name,
                 revoked_at_provider, provider_revoke_error, initiated_by, ip_address)
            VALUES ($1,$2,$3,$4,$5,$6,'self',$7)
            """,
            str(user["id"]),
            row["platform"],
            row["account_id"],
            row["account_name"],
            ok,
            err or None,
            ip_addr,
        )
        await log_system_event(conn, user_id=str(user["id"]), action="PLATFORM_DISCONNECTED",
                               event_category="PLATFORM", resource_type="platform",
                               resource_id=f"{row['platform']}:{row['account_id']}",
                               details={"platform": row["platform"], "account_name": row["account_name"],
                                        "provider_revoked": ok, "provider_error": err},
                               severity="WARNING")

    return {"status": "disconnected", "provider_revoked": ok}


@router.delete("/api/platform-accounts/{account_id}")
async def disconnect_account_by_id(
    account_id: str,
    request: Request,
    user: dict = Depends(get_current_user),
):
    """Alias for the disconnect endpoint (used by older frontend code)."""
    return await disconnect_account(
        platform="",      # platform is looked up from the DB row
        account_id=account_id,
        request=request,
        user=user,
    )


# OAuth start/callback: see routers.oauth (mounted from app.py).
