"""
UploadM8 Support & Activity Log routes — extracted from app.py.
"""

import asyncio
import logging
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

import core.state
from core.deps import get_current_user
from core.audit import log_system_event
from core.config import ADMIN_DISCORD_WEBHOOK_URL
from core.notifications import discord_notify
from core.models import SupportContactRequest, ActivityLogIn, ClientErrorIn
from core.r2 import put_object_bytes
from services.ops_incidents import record_operational_incident

logger = logging.getLogger("uploadm8-api")

router = APIRouter(tags=["support"])


@router.post("/api/support/contact")
async def support_contact(payload: SupportContactRequest, user: dict = Depends(get_current_user)):
    """Create a support ticket/message from the app."""
    async with core.state.db_pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO support_messages (user_id, name, email, subject, message)
            VALUES ($1, $2, $3, $4, $5)
            """,
            user["id"],
            (payload.name or user.get("name") or "").strip() or None,
            (payload.email or user.get("email") or "").strip() or None,
            payload.subject.strip(),
            payload.message.strip(),
        )

    # Optional admin notification
    if ADMIN_DISCORD_WEBHOOK_URL:
        await discord_notify(
            ADMIN_DISCORD_WEBHOOK_URL,
            embeds=[{
                "title": "\U0001f198 Support Message",
                "color": 0xf97316,
                "fields": [
                    {"name": "User", "value": f"{user.get('email','')} ({user.get('id','')})"},
                    {"name": "Subject", "value": payload.subject[:256]},
                    {"name": "Message", "value": (payload.message[:900] + "\u2026") if len(payload.message) > 900 else payload.message},
                ],
            }],
        )

    return {"status": "received"}


@router.post("/api/support/bug-report")
async def support_bug_report(
    request: Request,
    user: dict = Depends(get_current_user),
    message: str = Form(...),
    upload_id: str | None = Form(None),
    page_url: str | None = Form(None),
    screenshot: UploadFile | None = File(None),
):
    """
    User-submitted bug ticket with optional screenshot (PNG/JPEG/WebP, max 4MB).
    Creates operational_incidents row and alerts ops email + Discord.
    """
    msg = (message or "").strip()
    if len(msg) < 8:
        raise HTTPException(status_code=400, detail="Please describe the issue (at least 8 characters).")
    if len(msg) > 8000:
        raise HTTPException(status_code=400, detail="Message too long.")

    uid = str(user["id"])
    up_raw = (upload_id or "").strip() or None
    if up_raw:
        try:
            uuid.UUID(up_raw)
        except ValueError as e:
            raise HTTPException(status_code=400, detail="Invalid upload ID") from e
        async with core.state.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id FROM uploads WHERE id = $1::uuid AND user_id = $2::uuid",
                up_raw,
                uid,
            )
            if not row:
                raise HTTPException(status_code=400, detail="Upload not found for your account")

    shot_key = None
    if screenshot and getattr(screenshot, "filename", None):
        raw = await screenshot.read()
        if len(raw) > 4 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Screenshot too large (max 4MB).")
        ct = (screenshot.content_type or "").split(";")[0].strip().lower()
        ext_map = {"image/png": "png", "image/jpeg": "jpg", "image/jpg": "jpg", "image/webp": "webp"}
        if ct not in ext_map:
            raise HTTPException(status_code=400, detail="Screenshot must be PNG, JPEG, or WebP.")
        key = f"bug-reports/{uid}/{uuid.uuid4()}.{ext_map[ct]}"
        try:
            await asyncio.to_thread(put_object_bytes, key, raw, ct)
            shot_key = key
        except Exception as e:
            logger.warning("bug-report R2 upload failed: %s", e)
            raise HTTPException(status_code=500, detail="Could not store screenshot.") from e

    details = {
        "page_url": (page_url or "")[:2000],
        "user_email": user.get("email"),
        "user_agent": (request.headers.get("user-agent") or "")[:512],
    }
    inc = await record_operational_incident(
        core.state.db_pool,
        source="web",
        incident_type="user_bug_report",
        subject=f"Bug report from {user.get('email', uid)}",
        body=msg,
        details={**details, "upload_id": up_raw, "screenshot_r2_key": shot_key},
        user_id=uid,
        upload_id=up_raw,
        screenshot_r2_key=shot_key,
        alert_discord=False,
    )

    if ADMIN_DISCORD_WEBHOOK_URL:
        await discord_notify(
            ADMIN_DISCORD_WEBHOOK_URL,
            embeds=[
                {
                    "title": "Bug report",
                    "color": 0xF97316,
                    "fields": [
                        {"name": "User", "value": f"{user.get('email','')} ({uid})", "inline": False},
                        {"name": "Upload", "value": up_raw or "—", "inline": True},
                        {"name": "Incident", "value": str(inc or "—"), "inline": True},
                        {"name": "Message", "value": msg[:900] + ("…" if len(msg) > 900 else ""), "inline": False},
                    ],
                }
            ],
        )

    return {"ok": True, "incident_id": inc}


@router.post("/api/support/client-error")
async def support_client_error(payload: ClientErrorIn, request: Request, user: dict = Depends(get_current_user)):
    """Capture unexpected frontend errors for the operational incident log."""
    await log_system_event(
        user_id=str(user["id"]),
        action="client_js_error",
        event_category="CLIENT_ERROR",
        resource_type="page",
        resource_id=(payload.page_url or "")[:200],
        details={
            "message": payload.message[:4000],
            "stack": (payload.stack or "")[:8000],
            "upload_id": payload.upload_id,
        },
        request=request,
        severity="ERROR",
        outcome="FAILURE",
    )
    await record_operational_incident(
        core.state.db_pool,
        source="web",
        incident_type="client_js_error",
        subject=f"JS error: {payload.message[:120]}",
        body=payload.message[:8000],
        details={
            "stack": (payload.stack or "")[:8000],
            "page_url": (payload.page_url or "")[:2000],
            "upload_id": payload.upload_id,
            "user_email": user.get("email"),
        },
        user_id=str(user["id"]),
        upload_id=payload.upload_id,
    )
    return {"ok": True}


@router.post("/api/activity/log")
async def log_activity(data: ActivityLogIn, request: Request, user: dict = Depends(get_current_user)):
    """
    Frontend button-click and UI action audit trail.
    Called by JavaScript whenever a significant user action occurs.
    Stored in system_event_log with full context.
    """
    allowed_categories = {"UI_ACTION", "UPLOAD", "PLATFORM", "AUTH", "NAVIGATION", "CLIENT_ERROR"}
    category = data.event_category if data.event_category in allowed_categories else "UI_ACTION"

    # Sanitize — prevent log injection
    action_safe = str(data.action or "")[:100].strip()
    if not action_safe:
        return {"ok": False, "error": "action required"}

    await log_system_event(
        user_id=str(user["id"]),
        action=action_safe,
        event_category=category,
        resource_type=data.resource_type,
        resource_id=data.resource_id,
        details={**(data.details or {}), "session_id": data.session_id},
        request=request,
    )
    return {"ok": True}
