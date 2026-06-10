"""
Workspace / team seat API routes.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field

import core.state
from core.config import FRONTEND_URL
from core.deps import get_current_user
from services.workspace import (
    accept_workspace_invite,
    count_active_seats,
    count_pending_invites,
    create_workspace_invite,
    ensure_personal_workspace,
    get_workspace_for_user,
    list_pending_workspace_invites,
    list_workspace_members,
    rename_workspace,
    resend_workspace_invite,
    revoke_workspace_invite,
    revoke_workspace_member,
    switch_active_workspace,
    update_workspace_member_role,
)
from stages.entitlements import get_entitlements_from_user

logger = logging.getLogger("uploadm8-api")

router = APIRouter(prefix="/api/workspace", tags=["workspace"])


class InviteRequest(BaseModel):
    email: str = Field(min_length=3, max_length=255)
    role: str = Field(default="editor")


class AcceptInviteRequest(BaseModel):
    token: str = Field(min_length=8)


class SwitchWorkspaceRequest(BaseModel):
    workspace_id: str


class RenameWorkspaceRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)


class MemberRoleRequest(BaseModel):
    role: str = Field(min_length=3, max_length=20)


async def _send_invite_email(
    bg: BackgroundTasks,
    to_email: str,
    inviter_name: str,
    raw_token: str,
    workspace_name: str,
    owner_user_id: str,
):
    try:
        from stages.emails.workspace import send_workspace_invite_email

        link = f"{FRONTEND_URL.rstrip('/')}/accept-invite.html?token={raw_token}"
        bg.add_task(
            send_workspace_invite_email,
            to_email,
            inviter_name,
            workspace_name,
            link,
            owner_user_id,
            core.state.db_pool,
        )
    except Exception as e:
        logger.warning("workspace invite email skipped: %s", e)


async def _resolve_workspace_ctx(conn, user: dict):
    uid = str(user["id"])
    ctx = await get_workspace_for_user(conn, uid)
    if ctx:
        return ctx
    ws_id = await ensure_personal_workspace(
        conn, uid, user.get("name") or user.get("email") or "My Workspace"
    )
    return await get_workspace_for_user(conn, uid, ws_id)


@router.get("")
async def get_workspace(user: dict = Depends(get_current_user)):
    async with core.state.db_pool.acquire() as conn:
        ctx = await _resolve_workspace_ctx(conn, user)
        ws_row = await conn.fetchrow(
            "SELECT id::text AS id, name, owner_user_id::text AS owner_user_id FROM workspaces WHERE id = $1::uuid",
            ctx.workspace_id,
        )
        members = await list_workspace_members(conn, ctx.workspace_id)
        pending = await list_pending_workspace_invites(conn, ctx.workspace_id)
        ent = get_entitlements_from_user(ctx.owner_row)
        seats_used = await count_active_seats(conn, ctx.workspace_id)
        pending_invites = await count_pending_invites(conn, ctx.workspace_id)
    return {
        "workspace": {
            "id": ws_row["id"],
            "name": ws_row["name"],
            "owner_user_id": ws_row["owner_user_id"],
            "role": ctx.role,
        },
        "members": members,
        "pending_invite_list": pending,
        "seats": {
            "used": seats_used,
            "limit": int(ent.team_seats or 1),
            "pending_invites": pending_invites,
        },
    }


@router.patch("")
async def patch_workspace(body: RenameWorkspaceRequest, user: dict = Depends(get_current_user)):
    async with core.state.db_pool.acquire() as conn:
        ctx = await _resolve_workspace_ctx(conn, user)
        name = await rename_workspace(conn, ctx, body.name)
    return {"status": "ok", "name": name}


@router.post("/invites")
async def invite_member(body: InviteRequest, background_tasks: BackgroundTasks, user: dict = Depends(get_current_user)):
    async with core.state.db_pool.acquire() as conn:
        ctx = await _resolve_workspace_ctx(conn, user)
        raw, invite_id = await create_workspace_invite(conn, ctx, body.email, body.role)
        ws_name = await conn.fetchval("SELECT name FROM workspaces WHERE id = $1::uuid", ctx.workspace_id)
    inviter = user.get("name") or user.get("email") or "A teammate"
    await _send_invite_email(
        background_tasks,
        body.email.strip().lower(),
        inviter,
        raw,
        ws_name or "Workspace",
        ctx.owner_user_id,
    )
    return {"status": "sent", "invite_id": invite_id}


@router.post("/invites/{invite_id}/resend")
async def resend_invite(invite_id: str, background_tasks: BackgroundTasks, user: dict = Depends(get_current_user)):
    async with core.state.db_pool.acquire() as conn:
        ctx = await _resolve_workspace_ctx(conn, user)
        raw, _ = await resend_workspace_invite(conn, ctx, invite_id)
        row = await conn.fetchrow(
            "SELECT email FROM workspace_invites WHERE id = $1::uuid",
            invite_id,
        )
        ws_name = await conn.fetchval("SELECT name FROM workspaces WHERE id = $1::uuid", ctx.workspace_id)
    inviter = user.get("name") or user.get("email") or "A teammate"
    if row:
        await _send_invite_email(
            background_tasks,
            row["email"],
            inviter,
            raw,
            ws_name or "Workspace",
            ctx.owner_user_id,
        )
    return {"status": "resent", "invite_id": invite_id}


@router.post("/invites/accept")
async def accept_invite(body: AcceptInviteRequest, user: dict = Depends(get_current_user)):
    async with core.state.db_pool.acquire() as conn:
        result = await accept_workspace_invite(conn, user, body.token)
    return {"status": "accepted", **result}


@router.delete("/invites/{invite_id}")
async def cancel_invite(invite_id: str, user: dict = Depends(get_current_user)):
    async with core.state.db_pool.acquire() as conn:
        ctx = await _resolve_workspace_ctx(conn, user)
        await revoke_workspace_invite(conn, ctx, invite_id)
    return {"status": "cancelled"}


@router.patch("/members/{member_user_id}/role")
async def change_member_role(
    member_user_id: str,
    body: MemberRoleRequest,
    user: dict = Depends(get_current_user),
):
    async with core.state.db_pool.acquire() as conn:
        ctx = await _resolve_workspace_ctx(conn, user)
        await update_workspace_member_role(conn, ctx, member_user_id, body.role)
    return {"status": "ok", "role": body.role.lower()}


@router.delete("/members/{member_user_id}")
async def remove_member(member_user_id: str, user: dict = Depends(get_current_user)):
    async with core.state.db_pool.acquire() as conn:
        ctx = await _resolve_workspace_ctx(conn, user)
        await revoke_workspace_member(conn, ctx, member_user_id)
    return {"status": "removed"}


@router.post("/switch")
async def switch_workspace(body: SwitchWorkspaceRequest, user: dict = Depends(get_current_user)):
    async with core.state.db_pool.acquire() as conn:
        await switch_active_workspace(conn, str(user["id"]), body.workspace_id)
    return {"status": "ok", "workspace_id": body.workspace_id}
