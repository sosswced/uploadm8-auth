"""
Workspace / team seat helpers — shared billing context for multi-user accounts.
"""

from __future__ import annotations

import hashlib
import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import HTTPException

from core.user_columns import (
    USERS_MEMBER_COLUMNS,
    USERS_OWNER_BILLING_COLUMNS,
    users_select_sql,
)
from stages.entitlements import get_entitlements_from_user

INVITE_TTL_HOURS = 168
VALID_MEMBER_ROLES = frozenset({"owner", "admin", "editor", "viewer"})
VALID_INVITE_ROLES = frozenset({"editor", "viewer", "admin"})


@dataclass
class WorkspaceContext:
    workspace_id: str
    owner_user_id: str
    member_user_id: str
    role: str
    owner_row: dict
    member_row: dict


async def ensure_personal_workspace(conn, user_id: str, name: str = "My Workspace") -> str:
    """Ensure the user has an active workspace (create personal workspace if missing)."""
    uid = str(user_id)
    ctx = await get_workspace_for_user(conn, uid)
    if ctx:
        active = await conn.fetchval("SELECT active_workspace_id::text FROM users WHERE id = $1", uid)
        if not active:
            await conn.execute(
                "UPDATE users SET active_workspace_id = $1::uuid WHERE id = $2::uuid",
                ctx.workspace_id,
                uid,
            )
        return ctx.workspace_id
    return await create_personal_workspace(conn, uid, name)


async def create_personal_workspace(conn, user_id: str, name: str = "My Workspace") -> str:
    """Create default workspace for a new user (owner membership)."""
    ws_id = str(uuid.uuid4())
    await conn.execute(
        """
        INSERT INTO workspaces (id, owner_user_id, name, created_at)
        VALUES ($1, $2, $3, NOW())
        """,
        ws_id,
        user_id,
        (name or "My Workspace").strip() or "My Workspace",
    )
    await conn.execute(
        """
        INSERT INTO workspace_members (workspace_id, user_id, role, status, joined_at)
        VALUES ($1, $2, 'owner', 'active', NOW())
        ON CONFLICT (workspace_id, user_id) DO NOTHING
        """,
        ws_id,
        user_id,
    )
    await conn.execute(
        "UPDATE users SET active_workspace_id = $1 WHERE id = $2 AND active_workspace_id IS NULL",
        ws_id,
        user_id,
    )
    return ws_id


async def get_workspace_for_user(conn, user_id: str, workspace_id: Optional[str] = None) -> Optional[WorkspaceContext]:
    """Resolve active workspace membership for a user."""
    uid = str(user_id)
    ws_id = workspace_id
    if not ws_id:
        ws_id = await conn.fetchval("SELECT active_workspace_id::text FROM users WHERE id = $1", uid)
    if not ws_id:
        row = await conn.fetchrow(
            """
            SELECT w.id::text AS workspace_id, w.owner_user_id::text AS owner_user_id, wm.role
            FROM workspace_members wm
            JOIN workspaces w ON w.id = wm.workspace_id
            WHERE wm.user_id = $1 AND wm.status = 'active'
            ORDER BY CASE WHEN wm.role = 'owner' THEN 0 ELSE 1 END, w.created_at ASC
            LIMIT 1
            """,
            uid,
        )
        if not row:
            return None
        ws_id = row["workspace_id"]
    else:
        row = await conn.fetchrow(
            """
            SELECT w.id::text AS workspace_id, w.owner_user_id::text AS owner_user_id, wm.role
            FROM workspace_members wm
            JOIN workspaces w ON w.id = wm.workspace_id
            WHERE wm.user_id = $1 AND wm.workspace_id = $2::uuid AND wm.status = 'active'
            """,
            uid,
            ws_id,
        )
        if not row:
            raise HTTPException(403, "Not a member of this workspace")
    owner = await conn.fetchrow(users_select_sql(USERS_OWNER_BILLING_COLUMNS), row["owner_user_id"])
    member = await conn.fetchrow(users_select_sql(USERS_MEMBER_COLUMNS), uid)
    if not owner or not member:
        raise HTTPException(403, "Workspace user not found")
    return WorkspaceContext(
        workspace_id=str(row["workspace_id"]),
        owner_user_id=str(row["owner_user_id"]),
        member_user_id=uid,
        role=str(row["role"]),
        owner_row=dict(owner),
        member_row=dict(member),
    )


def billing_user_id(ctx: Optional[WorkspaceContext], fallback_user_id: str) -> str:
    if ctx:
        return ctx.owner_user_id
    return str(fallback_user_id)


def resolve_billing_user_id(user: dict) -> str:
    """Owner wallet / platform token row for workspace members."""
    return str(user.get("billing_user_id") or user["id"])


def workspace_role_from_user(user: dict) -> str:
    ws = user.get("workspace")
    if isinstance(ws, dict) and ws.get("role"):
        return str(ws["role"]).lower()
    return "owner"


def can_upload_in_workspace(ctx: Optional[WorkspaceContext]) -> bool:
    if not ctx:
        return True
    return ctx.role in ("owner", "admin", "editor")


def can_upload_from_user(user: dict) -> bool:
    role = workspace_role_from_user(user)
    return role in ("owner", "admin", "editor")


def can_manage_platforms(user: dict) -> bool:
    return workspace_role_from_user(user) in ("owner", "admin")


def can_manage_billing(user: dict) -> bool:
    return workspace_role_from_user(user) in ("owner", "admin")


def can_manage_team(user: dict) -> bool:
    return workspace_role_from_user(user) in ("owner", "admin")


def can_edit_account_settings(user: dict) -> bool:
    return workspace_role_from_user(user) in ("owner", "admin", "editor")


def workspace_capabilities(user: dict) -> dict[str, bool]:
    role = workspace_role_from_user(user)
    return {
        "role": role,
        "can_upload": role in ("owner", "admin", "editor"),
        "can_manage_platforms": role in ("owner", "admin"),
        "can_manage_billing": role in ("owner", "admin"),
        "can_manage_team": role in ("owner", "admin"),
        "can_edit_settings": role in ("owner", "admin", "editor"),
    }


def require_can_manage_platforms(user: dict) -> None:
    if not can_manage_platforms(user):
        raise HTTPException(
            403,
            {
                "code": "workspace_role_forbidden",
                "message": "Only workspace owners and admins can connect or disconnect platform accounts.",
            },
        )


def require_can_manage_billing(user: dict) -> None:
    if not can_manage_billing(user):
        raise HTTPException(
            403,
            {
                "code": "workspace_role_forbidden",
                "message": "Only workspace owners and admins can manage billing.",
            },
        )


def require_can_edit_settings(user: dict) -> None:
    if not can_edit_account_settings(user):
        raise HTTPException(
            403,
            {
                "code": "workspace_role_forbidden",
                "message": "Viewers have read-only access to workspace settings.",
            },
        )


def apply_owner_billing_profile(user_dict: dict, owner_row: dict) -> dict:
    """Stripe/subscription fields follow workspace owner for members."""
    bill_id = str(user_dict.get("billing_user_id") or user_dict["id"])
    if bill_id == str(user_dict["id"]):
        return user_dict
    for key in (
        "stripe_customer_id",
        "stripe_subscription_id",
        "subscription_status",
        "subscription_tier",
        "flex_enabled",
        "current_period_end",
        "trial_end",
    ):
        if key in owner_row:
            user_dict[key] = owner_row[key]
    return user_dict


async def count_active_seats(conn, workspace_id: str) -> int:
    return int(
        await conn.fetchval(
            """
            SELECT COUNT(*)::int FROM workspace_members
            WHERE workspace_id = $1::uuid AND status = 'active'
            """,
            workspace_id,
        )
        or 0
    )


async def count_pending_invites(conn, workspace_id: str) -> int:
    return int(
        await conn.fetchval(
            """
            SELECT COUNT(*)::int FROM workspace_invites
            WHERE workspace_id = $1::uuid
              AND accepted_at IS NULL
              AND expires_at > NOW()
            """,
            workspace_id,
        )
        or 0
    )


async def assert_seat_available(conn, owner_user: dict, workspace_id: str, *, include_pending: bool = False) -> None:
    ent = get_entitlements_from_user(owner_user)
    limit = int(ent.team_seats or 1)
    active = await count_active_seats(conn, workspace_id)
    used = active + (await count_pending_invites(conn, workspace_id) if include_pending else 0)
    if used >= limit:
        raise HTTPException(
            403,
            {
                "code": "team_seat_limit",
                "message": f"Team seat limit reached ({active}/{limit}). Upgrade or remove a member.",
            },
        )


async def list_workspace_members(conn, workspace_id: str) -> List[Dict[str, Any]]:
    rows = await conn.fetch(
        """
        SELECT wm.user_id::text AS user_id, wm.role, wm.status, wm.joined_at,
               u.email, u.name
        FROM workspace_members wm
        JOIN users u ON u.id = wm.user_id
        WHERE wm.workspace_id = $1::uuid AND wm.status = 'active'
        ORDER BY CASE wm.role WHEN 'owner' THEN 0 WHEN 'admin' THEN 1 ELSE 2 END, wm.joined_at ASC
        """,
        workspace_id,
    )
    out = []
    for r in rows:
        out.append(
            {
                "user_id": r["user_id"],
                "email": r["email"],
                "name": r["name"],
                "role": r["role"],
                "status": r["status"],
                "joined_at": r["joined_at"].isoformat() if r.get("joined_at") else None,
            }
        )
    return out


def _hash_invite_token(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


async def create_workspace_invite(
    conn,
    ctx: WorkspaceContext,
    email: str,
    role: str = "editor",
) -> tuple[str, str]:
    if ctx.role not in ("owner", "admin"):
        raise HTTPException(403, "Only workspace owners and admins can invite members")
    role = (role or "editor").lower()
    if role not in VALID_INVITE_ROLES:
        raise HTTPException(400, f"Invalid invite role: {role}")
    await assert_seat_available(conn, ctx.owner_row, ctx.workspace_id, include_pending=True)
    email_lc = email.strip().lower()
    if not email_lc or "@" not in email_lc:
        raise HTTPException(400, "Valid email required")

    existing = await conn.fetchrow(
        """
        SELECT 1 FROM workspace_members wm
        JOIN users u ON u.id = wm.user_id
        WHERE wm.workspace_id = $1::uuid AND LOWER(u.email) = $2 AND wm.status = 'active'
        """,
        ctx.workspace_id,
        email_lc,
    )
    if existing:
        raise HTTPException(400, "User is already a workspace member")

    pending = await conn.fetchrow(
        """
        SELECT id::text AS id FROM workspace_invites
        WHERE workspace_id = $1::uuid AND LOWER(email) = $2
          AND accepted_at IS NULL AND expires_at > NOW()
        LIMIT 1
        """,
        ctx.workspace_id,
        email_lc,
    )
    if pending:
        raise HTTPException(
            400,
            {
                "code": "invite_already_pending",
                "message": "An invite is already pending for this email. Resend or cancel it first.",
                "invite_id": pending["id"],
            },
        )

    raw = secrets.token_urlsafe(32)
    token_hash = _hash_invite_token(raw)
    expires = datetime.now(timezone.utc) + timedelta(hours=INVITE_TTL_HOURS)
    invite_id = str(uuid.uuid4())
    await conn.execute(
        """
        INSERT INTO workspace_invites (id, workspace_id, email, token_hash, role, expires_at)
        VALUES ($1, $2, $3, $4, $5, $6)
        """,
        invite_id,
        ctx.workspace_id,
        email_lc,
        token_hash,
        role,
        expires,
    )
    return raw, invite_id


async def accept_workspace_invite(conn, user: dict, raw_token: str) -> Dict[str, Any]:
    token_hash = _hash_invite_token(raw_token.strip())
    inv = await conn.fetchrow(
        """
        SELECT id::text AS id, workspace_id::text AS workspace_id, email, role, expires_at, accepted_at
        FROM workspace_invites WHERE token_hash = $1
        """,
        token_hash,
    )
    if not inv:
        raise HTTPException(404, "Invalid or expired invite")
    if inv["accepted_at"]:
        raise HTTPException(400, "Invite already accepted")
    if inv["expires_at"] and inv["expires_at"] < datetime.now(timezone.utc):
        raise HTTPException(400, "Invite expired")

    user_email = (user.get("email") or "").strip().lower()
    invite_email = (inv["email"] or "").strip().lower()
    if user_email != invite_email:
        raise HTTPException(
            403,
            {
                "code": "invite_email_mismatch",
                "message": f"Sign in with {invite_email} to accept this invite.",
                "expected_email": invite_email,
            },
        )

    owner = await conn.fetchrow(
        "SELECT * FROM users WHERE id = (SELECT owner_user_id FROM workspaces WHERE id = $1::uuid)",
        inv["workspace_id"],
    )
    if not owner:
        raise HTTPException(404, "Workspace not found")
    await assert_seat_available(conn, dict(owner), inv["workspace_id"])

    uid = str(user["id"])
    await conn.execute(
        """
        INSERT INTO workspace_members (workspace_id, user_id, role, status, invited_at, joined_at)
        VALUES ($1::uuid, $2::uuid, $3, 'active', NOW(), NOW())
        ON CONFLICT (workspace_id, user_id) DO UPDATE SET
            role = EXCLUDED.role, status = 'active', joined_at = NOW()
        """,
        inv["workspace_id"],
        uid,
        inv["role"],
    )
    await conn.execute(
        "UPDATE workspace_invites SET accepted_at = NOW() WHERE id = $1::uuid",
        inv["id"],
    )
    await conn.execute(
        "UPDATE users SET active_workspace_id = $1::uuid WHERE id = $2::uuid",
        inv["workspace_id"],
        uid,
    )
    return {"workspace_id": inv["workspace_id"], "role": inv["role"]}


async def list_pending_workspace_invites(conn, workspace_id: str) -> List[Dict[str, Any]]:
    rows = await conn.fetch(
        """
        SELECT id::text AS id, email, role, expires_at, created_at
        FROM workspace_invites
        WHERE workspace_id = $1::uuid
          AND accepted_at IS NULL
          AND expires_at > NOW()
        ORDER BY created_at DESC
        """,
        workspace_id,
    )
    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "id": r["id"],
                "email": r["email"],
                "role": r["role"],
                "expires_at": r["expires_at"].isoformat() if r.get("expires_at") else None,
                "created_at": r["created_at"].isoformat() if r.get("created_at") else None,
            }
        )
    return out


async def revoke_workspace_invite(conn, ctx: WorkspaceContext, invite_id: str) -> None:
    if ctx.role not in ("owner", "admin"):
        raise HTTPException(403, "Only workspace owners and admins can cancel invites")
    result = await conn.execute(
        """
        DELETE FROM workspace_invites
        WHERE id = $1::uuid AND workspace_id = $2::uuid AND accepted_at IS NULL
        """,
        invite_id,
        ctx.workspace_id,
    )
    if result == "DELETE 0":
        raise HTTPException(404, "Invite not found or already accepted")


async def revoke_workspace_member(conn, ctx: WorkspaceContext, member_user_id: str) -> None:
    if ctx.role not in ("owner", "admin"):
        raise HTTPException(403, "Only workspace owners and admins can remove members")
    if str(member_user_id) == ctx.owner_user_id:
        raise HTTPException(400, "Cannot remove workspace owner")
    await conn.execute(
        """
        UPDATE workspace_members SET status = 'removed'
        WHERE workspace_id = $1::uuid AND user_id = $2::uuid AND role <> 'owner'
        """,
        ctx.workspace_id,
        member_user_id,
    )


async def update_workspace_member_role(
    conn,
    ctx: WorkspaceContext,
    member_user_id: str,
    role: str,
) -> None:
    if ctx.role not in ("owner", "admin"):
        raise HTTPException(403, "Only workspace owners and admins can change member roles")
    role = (role or "").lower()
    if role not in VALID_INVITE_ROLES:
        raise HTTPException(400, f"Invalid role: {role}")
    if str(member_user_id) == ctx.owner_user_id:
        raise HTTPException(400, "Cannot change the workspace owner role")
    if ctx.role == "admin" and role == "admin":
        target = await conn.fetchval(
            "SELECT role FROM workspace_members WHERE workspace_id = $1::uuid AND user_id = $2::uuid",
            ctx.workspace_id,
            member_user_id,
        )
        if str(target) == "admin" and str(member_user_id) != ctx.member_user_id:
            raise HTTPException(403, "Admins cannot change other admins' roles")
    updated = await conn.execute(
        """
        UPDATE workspace_members SET role = $3
        WHERE workspace_id = $1::uuid AND user_id = $2::uuid AND role <> 'owner' AND status = 'active'
        """,
        ctx.workspace_id,
        member_user_id,
        role,
    )
    if updated == "UPDATE 0":
        raise HTTPException(404, "Member not found")


async def rename_workspace(conn, ctx: WorkspaceContext, name: str) -> str:
    if ctx.role not in ("owner", "admin"):
        raise HTTPException(403, "Only workspace owners and admins can rename the workspace")
    clean = (name or "").strip()[:120] or "My Workspace"
    await conn.execute(
        "UPDATE workspaces SET name = $2 WHERE id = $1::uuid",
        ctx.workspace_id,
        clean,
    )
    return clean


async def resend_workspace_invite(conn, ctx: WorkspaceContext, invite_id: str) -> tuple[str, str]:
    if ctx.role not in ("owner", "admin"):
        raise HTTPException(403, "Only workspace owners and admins can resend invites")
    row = await conn.fetchrow(
        """
        SELECT id::text AS id, email, role
        FROM workspace_invites
        WHERE id = $1::uuid AND workspace_id = $2::uuid
          AND accepted_at IS NULL AND expires_at > NOW()
        """,
        invite_id,
        ctx.workspace_id,
    )
    if not row:
        raise HTTPException(404, "Invite not found or expired")
    raw = secrets.token_urlsafe(32)
    token_hash = _hash_invite_token(raw)
    expires = datetime.now(timezone.utc) + timedelta(hours=INVITE_TTL_HOURS)
    await conn.execute(
        """
        UPDATE workspace_invites
        SET token_hash = $3, expires_at = $4
        WHERE id = $1::uuid AND workspace_id = $2::uuid
        """,
        invite_id,
        ctx.workspace_id,
        token_hash,
        expires,
    )
    return raw, str(row["id"])


async def switch_active_workspace(conn, user_id: str, workspace_id: str) -> None:
    ctx = await get_workspace_for_user(conn, user_id, workspace_id)
    if not ctx:
        raise HTTPException(403, "Not a member of this workspace")
    await conn.execute(
        "UPDATE users SET active_workspace_id = $1::uuid WHERE id = $2::uuid",
        workspace_id,
        user_id,
    )
