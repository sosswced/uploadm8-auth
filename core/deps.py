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
        # Single wallet fetch shared with daily_refill — eliminates the duplicate
        # SELECT * FROM wallets that Sentry flagged as "Consecutive DB Queries".
        # daily_refill returns the (possibly updated) wallet, or None when it
        # short-circuited for a paid/internal tier without touching the DB.
        wallet = await get_wallet(conn, user_id)
        refreshed = await daily_refill(conn, user_id, user["subscription_tier"], wallet=wallet)
        if refreshed is not None:
            wallet = refreshed
        return {**dict(user), "wallet": wallet}


async def get_current_user_readonly(request: Request, authorization: Optional[str] = Header(None)):
    """
    Same auth/validation as get_current_user but skips last_active_at write and daily_refill.
    Use for read-heavy or frequently-polled routes (catalog, platform account lists, analytics cache,
    Thumbnail Studio list/job GETs, CDN preview). ``GET /api/uploads/{id}`` uses ``get_verified_user_id``
    plus a single-connection fetch instead.
    Still loads wallet once for responses that need balances / plan checks.
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


async def get_current_user_readonly_no_wallet(
    request: Request, authorization: Optional[str] = Header(None)
):
    """
    Same auth gates as ``get_current_user_readonly`` but skips ``SELECT`` on ``wallets``.

    Use when the route only needs ``user["id"]`` (or other ``users`` columns) — e.g. hot
    read paths that open their own connection for heavy queries (``GET /api/catalog/content``).
    ``wallet`` is set to ``None`` for compatibility with code that checks ``user.get("wallet")``.
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
        return {**dict(user), "wallet": None}


async def require_admin(user: dict = Depends(get_current_user)):
    if user.get("role") not in ("admin", "master_admin"): raise HTTPException(403, "Admin required")
    return user

async def require_master_admin(user: dict = Depends(get_current_user)):
    if user.get("role") != "master_admin": raise HTTPException(403, "Master admin required")
    return user


async def get_verified_user_id(request: Request, authorization: Optional[str] = Header(None)) -> str:
    """
    Resolve and verify the access JWT (Bearer or cookie) without touching the database.

    Use when the handler performs its own user row checks on the same connection as other
    queries (e.g. ``GET /api/uploads/{id}``) to avoid an extra pool checkout and duplicate
    ``pg_advisory_unlock_all`` on connection release.
    """
    user_id, reason = _resolve_user_id_from_session(authorization, request.cookies)
    if not user_id:
        if reason == "missing":
            raise HTTPException(401, "Missing authorization")
        raise HTTPException(401, "Invalid token")
    return user_id
