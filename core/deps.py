"""
UploadM8 FastAPI dependencies — extracted from app.py.
get_current_user, require_admin, require_master_admin.
"""

from typing import Optional, Tuple

from fastapi import HTTPException, Request, Header, Depends

import core.state
from core.auth import verify_access_jwt
from core.cookie_auth import access_token_from_cookie
from core.wallet import get_wallet, daily_refill


def _resolve_user_id_from_session(authorization: Optional[str], cookies: dict) -> Tuple[Optional[str], str]:
    """
    Prefer a valid Bearer JWT, then fall back to the access cookie.

    If the client sends an expired Bearer (common after long R2 uploads in cross-host
    dev with um8_send_bearer) but still has a fresh HttpOnly cookie from refresh,
    accepting the cookie avoids spurious 401s on GET /api/uploads/{id} polling.
    """
    bearer = authorization[7:].strip() if authorization and authorization.startswith("Bearer ") else None
    cookie_tok = access_token_from_cookie(cookies)

    if bearer:
        uid = verify_access_jwt(bearer)
        if uid:
            return uid, ""
    if cookie_tok:
        uid = verify_access_jwt(cookie_tok)
        if uid:
            return uid, ""

    if not bearer and not cookie_tok:
        return None, "missing"
    return None, "invalid"


async def get_current_user(request: Request, authorization: Optional[str] = Header(None)):
    user_id, reason = _resolve_user_id_from_session(authorization, request.cookies)
    if not user_id:
        if reason == "missing":
            raise HTTPException(401, "Missing authorization")
        raise HTTPException(401, "Invalid token")

    async with core.state.db_pool.acquire() as conn:
        user = await conn.fetchrow("SELECT * FROM users WHERE id = $1", user_id)
        if not user: raise HTTPException(401, "User not found")
        if user["status"] == "banned": raise HTTPException(403, "Account suspended")
        if user.get("email_verified") is False:
            raise HTTPException(
                status_code=403,
                detail={
                    "message": "Please verify your email to use the app.",
                    "code": "email_not_verified",
                },
            )
        await conn.execute("UPDATE users SET last_active_at = NOW() WHERE id = $1", user_id)
        # Daily token refill
        await daily_refill(conn, user_id, user["subscription_tier"])
        wallet = await get_wallet(conn, user_id)
        return {**dict(user), "wallet": wallet}


async def get_current_user_readonly(request: Request, authorization: Optional[str] = Header(None)):
    """
    Same auth/validation as get_current_user but skips last_active_at write and daily_refill
    (for read-heavy endpoints like catalog lists).
    """
    user_id, reason = _resolve_user_id_from_session(authorization, request.cookies)
    if not user_id:
        if reason == "missing":
            raise HTTPException(401, "Missing authorization")
        raise HTTPException(401, "Invalid token")

    async with core.state.db_pool.acquire() as conn:
        user = await conn.fetchrow("SELECT * FROM users WHERE id = $1", user_id)
        if not user:
            raise HTTPException(401, "User not found")
        if user["status"] == "banned":
            raise HTTPException(403, "Account suspended")
        if user.get("email_verified") is False:
            raise HTTPException(
                status_code=403,
                detail={
                    "message": "Please verify your email to use the app.",
                    "code": "email_not_verified",
                },
            )
        wallet = await get_wallet(conn, user_id)
        return {**dict(user), "wallet": wallet}


async def require_admin(user: dict = Depends(get_current_user)):
    if user.get("role") not in ("admin", "master_admin"): raise HTTPException(403, "Admin required")
    return user

async def require_master_admin(user: dict = Depends(get_current_user)):
    if user.get("role") != "master_admin": raise HTTPException(403, "Master admin required")
    return user
