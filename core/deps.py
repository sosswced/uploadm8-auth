"""
UploadM8 FastAPI dependencies — extracted from app.py.
get_current_user, require_admin, require_master_admin.
"""

from typing import Optional, Tuple

from fastapi import HTTPException, Request, Header, Depends

import core.state
from core.auth import verify_access_jwt
from core.cookie_auth import access_token_from_cookie
from core.db_pool import acquire_db
from core.wallet import get_wallet, daily_refill
from services.workspace import get_workspace_for_user, billing_user_id


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


async def _attach_workspace_context(conn, user: dict, request: Optional[Request]) -> dict:
    """Resolve workspace membership and owner billing wallet."""
    user_id = str(user["id"])
    ws_header = None
    if request is not None:
        ws_header = request.headers.get("X-Workspace-Id") or request.headers.get("x-workspace-id")
    active_ws = user.get("active_workspace_id")
    ws_id = ws_header or (str(active_ws) if active_ws else None)
    ctx = None
    try:
        ctx = await get_workspace_for_user(conn, user_id, ws_id)
    except HTTPException:
        raise
    except Exception:
        ctx = None

    billing_id = billing_user_id(ctx, user_id)
    wallet = await get_wallet(conn, billing_id)
    out = {**user, "wallet": wallet}
    if ctx:
        owner = ctx.owner_row
        ws_name = await conn.fetchval(
            "SELECT name FROM workspaces WHERE id = $1::uuid",
            ctx.workspace_id,
        )
        out["workspace"] = {
            "id": ctx.workspace_id,
            "owner_user_id": ctx.owner_user_id,
            "role": ctx.role,
            "name": (ws_name or "My Workspace").strip() or "My Workspace",
        }
        out["billing_user_id"] = billing_id
        if billing_id != user_id:
            out["subscription_tier"] = owner.get("subscription_tier")
            out["flex_enabled"] = owner.get("flex_enabled")
            from services.workspace import apply_owner_billing_profile

            out = apply_owner_billing_profile(out, owner)
    else:
        out["billing_user_id"] = user_id
    return out


async def get_current_user(request: Request, authorization: Optional[str] = Header(None)):
    user_id, reason = _resolve_user_id_from_session(authorization, request.cookies)
    if not user_id:
        if reason == "missing":
            raise HTTPException(401, "Missing authorization")
        raise HTTPException(401, "Invalid token")

    async with acquire_db(core.state.db_pool) as conn:
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
        user_dict = dict(user)
        user_dict = await _attach_workspace_context(conn, user_dict, request)
        billing_id = user_dict.get("billing_user_id") or user_id
        refreshed = await daily_refill(
            conn, billing_id, user_dict.get("subscription_tier") or user["subscription_tier"],
            wallet=user_dict.get("wallet"),
        )
        if refreshed is not None:
            user_dict["wallet"] = refreshed
        return user_dict


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

    async with acquire_db(core.state.db_pool) as conn:
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
        user_dict = dict(user)
        user_dict = await _attach_workspace_context(conn, user_dict, request)
        return user_dict


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

    async with acquire_db(core.state.db_pool) as conn:
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
        user_dict = dict(user)
        user_dict = await _attach_workspace_context(conn, user_dict, request)
        return {**user_dict, "wallet": None}


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


async def require_verified_user_on_conn(conn, user_id: str) -> dict:
    """
    Same auth gates as ``get_current_user_readonly`` on an existing asyncpg connection.

    Pair with ``get_verified_user_id`` for read-heavy handlers that only need ``users``
    row validation (no wallet / daily_refill).
    """
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
    return dict(user)


async def require_master_admin_on_conn(conn, user_id: str) -> dict:
    """
    Same auth gates as ``require_master_admin`` on an existing asyncpg connection.

    Pair with ``get_verified_user_id`` so admin handlers use one pool checkout for auth
    and business SQL (Sentry: consecutive ``pg_advisory_unlock_all`` spans).
    """
    user = await conn.fetchrow(
        "SELECT id, email, name, role, status, email_verified FROM users WHERE id = $1",
        user_id,
    )
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
    if user.get("role") != "master_admin":
        raise HTTPException(403, "Master admin required")
    return dict(user)
