"""
Password registration, login, and refresh-token rotation (DB + JWT).

Routers attach cookies / JSONResponse; this module only performs data work.
"""

from __future__ import annotations

import uuid
from typing import Optional, Tuple

from fastapi import HTTPException

from core.auth import (
    create_access_jwt,
    create_refresh_token,
    hash_password,
    rotate_refresh_token,
    verify_password,
)
from core.models import UserCreate, UserLogin
from core.wallet import ledger_entry
from stages.entitlements import get_entitlements_for_tier


async def register_user(
    conn,
    data: UserCreate,
    country_code: Optional[str],
    *,
    issue_tokens: bool = True,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Create user row, settings, wallet, signup bonuses.
    When issue_tokens is True, return (access_jwt, refresh_token); otherwise (None, None).
    """
    if await conn.fetchrow("SELECT id FROM users WHERE LOWER(email) = $1", data.email.lower()):
        raise HTTPException(409, "Email already registered")

    user_id = str(uuid.uuid4())
    cc = country_code
    if cc in ("XX", "T1", ""):
        cc = None

    await conn.execute(
        "INSERT INTO users (id, email, password_hash, name, country) VALUES ($1, $2, $3, $4, $5)",
        user_id,
        data.email.lower(),
        hash_password(data.password),
        data.name,
        cc,
    )
    await conn.execute("INSERT INTO user_settings (user_id) VALUES ($1)", user_id)

    ent = get_entitlements_for_tier("free")
    signup_put = max(ent.put_monthly, 80)
    signup_aic = max(ent.aic_monthly, 50)
    await conn.execute(
        "INSERT INTO wallets (user_id, put_balance, aic_balance) VALUES ($1, $2, $3)",
        user_id,
        signup_put,
        signup_aic,
    )
    await ledger_entry(conn, user_id, "put", signup_put, "signup_bonus")
    await ledger_entry(conn, user_id, "aic", signup_aic, "signup_bonus")

    if not issue_tokens:
        return None, None
    access = create_access_jwt(user_id)
    refresh = await create_refresh_token(conn, user_id)
    return access, refresh


async def login_user(conn, data: UserLogin) -> Tuple[str, str]:
    """Validate credentials; return (access_jwt, refresh_token)."""
    user = await conn.fetchrow(
        """
        SELECT id, password_hash, status, email_verified
        FROM users WHERE LOWER(email) = $1
        """,
        data.email.lower(),
    )
    if not user or not verify_password(data.password, user["password_hash"]):
        raise HTTPException(401, "Invalid credentials")
    if user["status"] == "banned":
        raise HTTPException(403, "Account suspended")
    if user.get("email_verified") is False:
        raise HTTPException(
            status_code=403,
            detail={
                "message": "Please verify your email before signing in.",
                "code": "email_not_verified",
            },
        )

    uid = str(user["id"])
    access = create_access_jwt(uid)
    refresh = await create_refresh_token(conn, uid)
    return access, refresh


async def refresh_session(conn, refresh_token: str) -> Tuple[str, str]:
    """Rotate refresh; return (new_access_jwt, new_refresh_token)."""
    access, new_refresh = await rotate_refresh_token(conn, refresh_token)
    return access, new_refresh
