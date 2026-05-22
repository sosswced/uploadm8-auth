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
    owner = await conn.fetchrow("SELECT * FROM users WHERE id = $1", row["owner_user_id"])
    member = await conn.fetchrow("SELECT * FROM users WHERE id = $1", uid)
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


def can_upload_in_workspace(ctx: Optional[WorkspaceContext]) -> bool:
    if not ctx:
        return True
    return ctx.role in ("owner", "admin", "editor")


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
    if user_email != (inv["email"] or "").strip().lower():
        raise HTTPException(403, "Invite email does not match your account")

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


async def switch_active_workspace(conn, user_id: str, workspace_id: str) -> None:
    ctx = await get_workspace_for_user(conn, user_id, workspace_id)
    if not ctx:
        raise HTTPException(403, "Not a member of this workspace")
    await conn.execute(
        "UPDATE users SET active_workspace_id = $1::uuid WHERE id = $2::uuid",
        workspace_id,
        user_id,
    )
