"""
UploadM8 Auth routes — extracted from app.py.
Registration, login, token refresh, logout, password reset/change.
"""

import secrets
import hashlib
import logging
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Request, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import core.state
from core.cookie_auth import clear_auth_cookies, refresh_token_from_cookie, set_auth_cookies
from core.auth import hash_password, verify_password, create_access_jwt, create_refresh_token
from core.notifications import notify_signup
from core.deps import get_current_user
from core.config import FRONTEND_URL, SIGNUP_EMAIL_VERIFICATION
from core.helpers import _sha256_hex
from core.models import (
    UserCreate,
    UserLogin,
    ForgotPasswordRequest,
    ResetPasswordRequest,
    PasswordChange,
)
from stages.emails import (
    send_welcome_email,
    send_signup_confirmation_email,
    send_post_verification_welcome_email,
    send_password_reset_email,
    send_password_changed_email,
)
from services.auth_credentials import login_user, refresh_session, register_user

logger = logging.getLogger("uploadm8-api")

router = APIRouter(prefix="/api/auth", tags=["auth"])


class ResendConfirmationBody(BaseModel):
    email: str = Field(..., min_length=3, max_length=320)


class UpdatePendingEmailBody(BaseModel):
    current_email: str = Field(..., min_length=3, max_length=320)
    new_email: str = Field(..., min_length=3, max_length=320)


@router.post("/register")
async def register(data: UserCreate, background_tasks: BackgroundTasks, request: Request):
    country_code = (request.headers.get("CF-IPCountry") or "")[:2].upper() or None
    email_lc = data.email.lower().strip()
    async with core.state.db_pool.acquire() as conn:
        access, refresh = await register_user(
            conn,
            data,
            country_code,
            issue_tokens=not SIGNUP_EMAIL_VERIFICATION,
        )
        if SIGNUP_EMAIL_VERIFICATION:
            uid = await conn.fetchval("SELECT id FROM users WHERE LOWER(email)=$1", email_lc)
            raw = secrets.token_urlsafe(32)
            token_hash = hashlib.sha256(raw.encode()).hexdigest()
            expires_at = datetime.now(timezone.utc) + timedelta(hours=48)
            await conn.execute("DELETE FROM signup_verifications WHERE user_id = $1", uid)
            await conn.execute(
                """
                INSERT INTO signup_verifications (user_id, token_hash, expires_at)
                VALUES ($1, $2, $3)
                """,
                uid,
                token_hash,
                expires_at,
            )
            await conn.execute("UPDATE users SET email_verified = false WHERE id = $1", uid)
            confirm_link = f"{FRONTEND_URL.rstrip('/')}/confirm-email.html?token={quote(raw)}"
            background_tasks.add_task(send_signup_confirmation_email, email_lc, data.name, confirm_link)
        else:
            background_tasks.add_task(send_welcome_email, email_lc, data.name)
    background_tasks.add_task(notify_signup, email_lc, data.name)
    if SIGNUP_EMAIL_VERIFICATION:
        return JSONResponse(content={"status": "pending_verification", "email": email_lc})
    resp = JSONResponse(
        content={"access_token": access, "refresh_token": refresh, "token_type": "bearer"}
    )
    set_auth_cookies(resp, access, refresh)
    return resp

@router.post("/login")
async def login(data: UserLogin):
    async with core.state.db_pool.acquire() as conn:
        access, refresh = await login_user(conn, data)
    resp = JSONResponse(
        content={"access_token": access, "refresh_token": refresh, "token_type": "bearer"}
    )
    set_auth_cookies(resp, access, refresh)
    return resp

@router.post("/refresh")
async def refresh(request: Request):
    """Rotate session. Refresh token from JSON body or HttpOnly cookie."""
    body: dict = {}
    try:
        raw = await request.json()
        if isinstance(raw, dict):
            body = raw
    except Exception:
        pass
    rt = (
        (body.get("refresh_token") or body.get("refreshToken") or "").strip()
        or refresh_token_from_cookie(request.cookies)
    )
    if not rt:
        raise HTTPException(401, "Missing refresh token")
    async with core.state.db_pool.acquire() as conn:
        access, new_refresh = await refresh_session(conn, rt)
    resp = JSONResponse(
        content={"access_token": access, "refresh_token": new_refresh, "token_type": "bearer"}
    )
    set_auth_cookies(resp, access, new_refresh)
    return resp

@router.post("/logout")
async def logout(user: dict = Depends(get_current_user)):
    async with core.state.db_pool.acquire() as conn:
        await conn.execute("UPDATE refresh_tokens SET revoked_at = NOW() WHERE user_id = $1 AND revoked_at IS NULL", user["id"])
    resp = JSONResponse(content={"status": "logged_out"})
    clear_auth_cookies(resp)
    return resp

@router.post("/logout-all")
async def logout_all(user: dict = Depends(get_current_user)):
    """Revoke all refresh tokens for current user (log out all devices)."""
    async with core.state.db_pool.acquire() as conn:
        await conn.execute(
            "UPDATE refresh_tokens SET revoked_at = NOW() WHERE user_id = $1 AND revoked_at IS NULL",
            user["id"],
        )
    resp = JSONResponse(content={"status": "logged_out_all"})
    clear_auth_cookies(resp)
    return resp


@router.post("/logout-other-sessions")
async def logout_other_sessions(request: Request, user: dict = Depends(get_current_user)):
    """
    Revoke every refresh token for this user except the one used on this request.
    Keeps the current browser/session signed in; use this for Settings → “other devices”.
    """
    rt = (refresh_token_from_cookie(request.cookies) or "").strip()
    if not rt:
        body: dict = {}
        try:
            raw = await request.json()
            if isinstance(raw, dict):
                body = raw
        except Exception:
            pass
        rt = (body.get("refresh_token") or body.get("refreshToken") or "").strip()
    if not rt:
        raise HTTPException(
            400,
            "No refresh token available (cookie or JSON). Sign out and sign in again, or use full Log out.",
        )
    h = _sha256_hex(rt)
    async with core.state.db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id FROM refresh_tokens
            WHERE token_hash = $1 AND user_id = $2 AND revoked_at IS NULL
            """,
            h,
            user["id"],
        )
        if not row:
            raise HTTPException(
                401,
                "This session's refresh token is invalid or expired. Sign out completely and sign in again.",
            )
        revoked_rows = await conn.fetch(
            """
            UPDATE refresh_tokens SET revoked_at = NOW()
            WHERE user_id = $1 AND revoked_at IS NULL AND id <> $2
            RETURNING id
            """,
            user["id"],
            row["id"],
        )
    return {"status": "other_sessions_revoked", "sessions_revoked": len(revoked_rows)}


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
    resp = JSONResponse(content={"status": "password_changed"})
    clear_auth_cookies(resp)
    return resp


async def _issue_auth_pair(conn, user_id: str) -> tuple[str, str]:
    access = create_access_jwt(user_id)
    refresh = await create_refresh_token(conn, user_id)
    return access, refresh


@router.get("/confirm-email")
async def confirm_email(
    background_tasks: BackgroundTasks,
    token: str = Query(..., min_length=8),
):
    """Consume signup_verifications token; returns session tokens on success."""
    now = datetime.now(timezone.utc)
    th = hashlib.sha256(token.encode()).hexdigest()
    async with core.state.db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, user_id, expires_at, used_at FROM signup_verifications WHERE token_hash = $1",
            th,
        )
        if not row:
            raise HTTPException(status_code=400, detail="Invalid confirmation link")
        if row["expires_at"] < now:
            raise HTTPException(status_code=400, detail="Confirmation link expired")
        uid = str(row["user_id"])
        if row["used_at"] is not None:
            u = await conn.fetchrow(
                "SELECT email_verified FROM users WHERE id = $1",
                row["user_id"],
            )
            if u and u["email_verified"] is True:
                access, refresh = await _issue_auth_pair(conn, uid)
                return JSONResponse(
                    status_code=400,
                    content={
                        "detail": "Email already confirmed",
                        "access_token": access,
                        "refresh_token": refresh,
                        "token_type": "bearer",
                    },
                )
            raise HTTPException(status_code=400, detail="Invalid confirmation link")

        await conn.execute("UPDATE signup_verifications SET used_at = NOW() WHERE id = $1", row["id"])
        await conn.execute(
            "UPDATE users SET email_verified = true, updated_at = NOW() WHERE id = $1",
            row["user_id"],
        )
        urow = await conn.fetchrow(
            "SELECT email, name FROM users WHERE id = $1",
            row["user_id"],
        )
        access, refresh = await _issue_auth_pair(conn, uid)

    if urow:
        background_tasks.add_task(
            send_post_verification_welcome_email,
            urow["email"],
            (urow["name"] or "there"),
        )
    return {
        "access_token": access,
        "refresh_token": refresh,
        "token_type": "bearer",
    }


@router.get("/verify-email")
async def verify_email_change(token: str = Query(..., min_length=8)):
    """Complete admin-initiated email change using plaintext token in email_changes."""
    async with core.state.db_pool.acquire() as conn:
        ec = await conn.fetchrow(
            """
            SELECT id, user_id, new_email
            FROM email_changes
            WHERE verification_token = $1
            ORDER BY created_at DESC
            LIMIT 1
            """,
            token,
        )
        if not ec:
            raise HTTPException(status_code=404, detail="Invalid or expired verification link")
        await conn.execute(
            "UPDATE users SET email_verified = true, updated_at = NOW() WHERE id = $1",
            ec["user_id"],
        )
        await conn.execute("UPDATE email_changes SET verification_token = NULL WHERE id = $1", ec["id"])
        email = ec["new_email"]
    return {"email": email, "new_email": email}


@router.post("/resend-confirmation")
async def resend_confirmation(body: ResendConfirmationBody, background_tasks: BackgroundTasks):
    """Resend signup confirmation (no account enumeration)."""
    email = body.email.lower().strip()
    async with core.state.db_pool.acquire() as conn:
        u = await conn.fetchrow(
            "SELECT id, name, email_verified FROM users WHERE LOWER(email) = $1",
            email,
        )
        if not u or u["email_verified"] is not False:
            return {"ok": True}
        uid = u["id"]
        raw = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(raw.encode()).hexdigest()
        expires_at = datetime.now(timezone.utc) + timedelta(hours=48)
        await conn.execute("DELETE FROM signup_verifications WHERE user_id = $1", uid)
        await conn.execute(
            """
            INSERT INTO signup_verifications (user_id, token_hash, expires_at)
            VALUES ($1, $2, $3)
            """,
            uid,
            token_hash,
            expires_at,
        )
    confirm_link = f"{FRONTEND_URL.rstrip('/')}/confirm-email.html?token={quote(raw)}"
    background_tasks.add_task(
        send_signup_confirmation_email,
        email,
        (u["name"] or "there"),
        confirm_link,
    )
    return {"ok": True}


@router.post("/update-pending-email")
async def update_pending_email(body: UpdatePendingEmailBody, background_tasks: BackgroundTasks):
    """While signup verification is pending, switch the email on the account and re-send confirm."""
    cur = body.current_email.lower().strip()
    new_email = body.new_email.lower().strip()
    if cur == new_email:
        raise HTTPException(400, "New email must differ from current")
    async with core.state.db_pool.acquire() as conn:
        u = await conn.fetchrow(
            "SELECT id, name, email_verified FROM users WHERE LOWER(email) = $1",
            cur,
        )
        if not u:
            raise HTTPException(404, "User not found")
        if u["email_verified"] is not False:
            raise HTTPException(400, "Email is not pending verification")
        taken = await conn.fetchval(
            "SELECT 1 FROM users WHERE LOWER(email) = $1 AND id <> $2",
            new_email,
            u["id"],
        )
        if taken:
            raise HTTPException(409, "Email already in use")
        await conn.execute(
            "UPDATE users SET email = $1, updated_at = NOW() WHERE id = $2",
            new_email,
            u["id"],
        )
        await conn.execute("DELETE FROM signup_verifications WHERE user_id = $1", u["id"])
        raw = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(raw.encode()).hexdigest()
        expires_at = datetime.now(timezone.utc) + timedelta(hours=48)
        await conn.execute(
            """
            INSERT INTO signup_verifications (user_id, token_hash, expires_at)
            VALUES ($1, $2, $3)
            """,
            u["id"],
            token_hash,
            expires_at,
        )
    confirm_link = f"{FRONTEND_URL.rstrip('/')}/confirm-email.html?token={quote(raw)}"
    background_tasks.add_task(
        send_signup_confirmation_email,
        new_email,
        (u["name"] or "there"),
        confirm_link,
    )
    return {"ok": True, "email": new_email}
