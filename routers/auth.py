"""
UploadM8 Auth routes — extracted from app.py.
Registration, login, token refresh, logout, password reset/change.
"""

import uuid
import secrets
import hashlib
import logging
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Request

import core.state
from core.auth import (
    hash_password,
    verify_password,
    create_access_jwt,
    create_refresh_token,
    rotate_refresh_token,
)
from core.helpers import _now_utc, _sha256_hex
from core.notifications import notify_signup
from core.audit import log_system_event
from core.wallet import get_wallet, daily_refill, credit_wallet, ledger_entry
from core.deps import get_current_user
from core.config import FRONTEND_URL
from core.models import (
    UserCreate,
    UserLogin,
    RefreshRequest,
    ForgotPasswordRequest,
    ResetPasswordRequest,
    PasswordChange,
)
from stages.emails import (
    send_welcome_email,
    send_password_reset_email,
    send_password_changed_email,
)
from stages.entitlements import get_entitlements_for_tier

logger = logging.getLogger("uploadm8-api")

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/register")
async def register(data: UserCreate, background_tasks: BackgroundTasks, request: Request):
    async with core.state.db_pool.acquire() as conn:
        if await conn.fetchrow("SELECT id FROM users WHERE LOWER(email) = $1", data.email.lower()):
            raise HTTPException(409, "Email already registered")
        user_id = str(uuid.uuid4())
        # Capture country from Cloudflare header if present
        country_code = (request.headers.get("CF-IPCountry") or "")[:2].upper() or None
        if country_code in ("XX", "T1", ""):
            country_code = None
        await conn.execute(
            "INSERT INTO users (id, email, password_hash, name, country) VALUES ($1, $2, $3, $4, $5)",
            user_id, data.email.lower(), hash_password(data.password), data.name, country_code
        )
        await conn.execute("INSERT INTO user_settings (user_id) VALUES ($1)", user_id)
        # Default credits from free tier entitlements — enough to try uploads + AI features
        ent = get_entitlements_for_tier("free")
        signup_put = max(ent.put_monthly, 80)   # full free monthly or 80 min
        signup_aic = max(ent.aic_monthly, 50)  # full free monthly or 50 min
        await conn.execute(
            "INSERT INTO wallets (user_id, put_balance, aic_balance) VALUES ($1, $2, $3)",
            user_id, signup_put, signup_aic
        )
        await ledger_entry(conn, user_id, "put", signup_put, "signup_bonus")
        await ledger_entry(conn, user_id, "aic", signup_aic, "signup_bonus")
        access = create_access_jwt(user_id)
        refresh = await create_refresh_token(conn, user_id)
    background_tasks.add_task(notify_signup, data.email, data.name)
    background_tasks.add_task(send_welcome_email, data.email, data.name)
    return {"access_token": access, "refresh_token": refresh, "token_type": "bearer"}

@router.post("/login")
async def login(data: UserLogin):
    async with core.state.db_pool.acquire() as conn:
        user = await conn.fetchrow("SELECT id, password_hash, status FROM users WHERE LOWER(email) = $1", data.email.lower())
        if not user or not verify_password(data.password, user["password_hash"]): raise HTTPException(401, "Invalid credentials")
        if user["status"] == "banned": raise HTTPException(403, "Account suspended")
        return {"access_token": create_access_jwt(str(user["id"])), "refresh_token": await create_refresh_token(conn, str(user["id"])), "token_type": "bearer"}

@router.post("/refresh")
async def refresh(data: RefreshRequest):
    async with core.state.db_pool.acquire() as conn:
        access, refresh = await rotate_refresh_token(conn, data.refresh_token)
    return {"access_token": access, "refresh_token": refresh, "token_type": "bearer"}

@router.post("/logout")
async def logout(user: dict = Depends(get_current_user)):
    async with core.state.db_pool.acquire() as conn:
        await conn.execute("UPDATE refresh_tokens SET revoked_at = NOW() WHERE user_id = $1 AND revoked_at IS NULL", user["id"])
    return {"status": "logged_out"}

@router.post("/logout-all")
async def logout_all(user: dict = Depends(get_current_user)):
    """Revoke all refresh tokens for current user (log out all devices)."""
    async with core.state.db_pool.acquire() as conn:
        await conn.execute(
            "UPDATE refresh_tokens SET revoked_at = NOW() WHERE user_id = $1 AND revoked_at IS NULL",
            user["id"],
        )
    return {"status": "logged_out_all"}



@router.post("/forgot-password")
async def forgot_password(payload: ForgotPasswordRequest, background: BackgroundTasks):
    """Initiate password reset. Always returns OK to prevent account enumeration."""
    email = payload.email.lower()
    async with core.state.db_pool.acquire() as conn:
        user_row = await conn.fetchrow("SELECT id, email, status FROM users WHERE LOWER(email)=$1", email)
        if user_row and user_row["status"] != "disabled":
            token = secrets.token_urlsafe(32)
            token_hash = hashlib.sha256(token.encode()).hexdigest()
            expires_at = datetime.now(timezone.utc) + timedelta(hours=1)

            # Invalidate prior unused tokens for this user
            await conn.execute(
                "UPDATE password_resets SET used_at = NOW() WHERE user_id=$1 AND used_at IS NULL",
                user_row["id"],
            )
            await conn.execute(
                "INSERT INTO password_resets (user_id, token_hash, expires_at) VALUES ($1,$2,$3)",
                user_row["id"], token_hash, expires_at
            )

            reset_link = f"{FRONTEND_URL.rstrip('/')}/reset-password?token={quote(token)}"
            background.add_task(send_password_reset_email, user_row["email"], reset_link)

    return {"ok": True}

@router.post("/reset-password")
async def reset_password(payload: ResetPasswordRequest, background: BackgroundTasks):
    token_hash = hashlib.sha256(payload.token.encode()).hexdigest()
    async with core.state.db_pool.acquire() as conn:
        pr = await conn.fetchrow(
            """
            SELECT id, user_id, expires_at, used_at
            FROM password_resets
            WHERE token_hash=$1
            ORDER BY created_at DESC
            LIMIT 1
            """,
            token_hash
        )
        if not pr or pr["used_at"] is not None:
            raise HTTPException(status_code=400, detail="Invalid or used reset token")
        if pr["expires_at"] < datetime.now(timezone.utc):
            raise HTTPException(status_code=400, detail="Reset token expired")

        new_hash = hash_password(payload.new_password)

        await conn.execute(
            "UPDATE users SET password_hash=$1, updated_at=NOW() WHERE id=$2",
            new_hash, pr["user_id"]
        )
        await conn.execute("UPDATE password_resets SET used_at=NOW() WHERE id=$1", pr["id"])

        # Force logout across devices/sessions
        await conn.execute("UPDATE refresh_tokens SET revoked_at = NOW() WHERE user_id=$1 AND revoked_at IS NULL", pr["user_id"])

        # Fetch email+name for the security confirmation email
        _u = await conn.fetchrow("SELECT email, name FROM users WHERE id = $1", pr["user_id"])

    if _u:
        background.add_task(send_password_changed_email, _u["email"], _u["name"] or "there")

    return {"ok": True}

@router.post("/change-password")
async def change_password(data: PasswordChange, background: BackgroundTasks, user: dict = Depends(get_current_user)):
    """Change user password"""
    async with core.state.db_pool.acquire() as conn:
        # Verify current password
        user_row = await conn.fetchrow("SELECT password_hash FROM users WHERE id = $1", user["id"])
        if not user_row or not verify_password(data.current_password, user_row["password_hash"]):
            raise HTTPException(401, "Current password is incorrect")

        # Update to new password
        new_hash = hash_password(data.new_password)
        await conn.execute("UPDATE users SET password_hash = $1, updated_at = NOW() WHERE id = $2", new_hash, user["id"])

        # Optionally invalidate other sessions (refresh tokens)
        await conn.execute("DELETE FROM refresh_tokens WHERE user_id = $1", user["id"])

    logger.info(f"Password changed for user {user['id']}")
    background.add_task(send_password_changed_email, user["email"], user.get("name") or "there")
    return {"status": "password_changed"}
