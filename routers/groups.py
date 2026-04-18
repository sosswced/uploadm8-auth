"""
UploadM8 Account Groups routes — extracted from app.py.
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException

import core.state
from core.deps import get_current_user
from core.helpers import _safe_col
from core.sql_allowlist import ACCOUNT_GROUPS_UPDATE_COLUMNS, assert_set_fragments_columns
from core.models import AccountGroupIn, AccountGroupUpdate, GroupUpsert

router = APIRouter(prefix="/api/groups", tags=["groups"])


@router.get("")
async def get_groups(user: dict = Depends(get_current_user)):
    async with core.state.db_pool.acquire() as conn:
        groups = await conn.fetch(
            """
            SELECT id, user_id, name, description, account_ids, color, created_at, updated_at
            FROM account_groups
            WHERE user_id = $1
            ORDER BY created_at DESC
            """,
            user["id"],
        )
    return [
        {
            "id": str(g["id"]),
            "name": g["name"],
            "description": g["description"],
            "account_ids": g["account_ids"] or [],
            "members": g["account_ids"] or [],
            "color": g["color"],
            "created_at": g["created_at"].isoformat() if g["created_at"] else None,
            "updated_at": g["updated_at"].isoformat() if g["updated_at"] else None,
        }
        for g in groups
    ]


@router.get("/{group_id}")
async def get_group(group_id: str, user: dict = Depends(get_current_user)):
    async with core.state.db_pool.acquire() as conn:
        g = await conn.fetchrow(
            """
            SELECT id, user_id, name, description, account_ids, color, created_at, updated_at
            FROM account_groups
            WHERE id = $1 AND user_id = $2
            """,
            group_id,
            user["id"],
        )
    if not g:
        raise HTTPException(404, "Group not found")
    return {
        "id": str(g["id"]),
        "name": g["name"],
        "description": g["description"],
        "account_ids": g["account_ids"] or [],
        "members": g["account_ids"] or [],
        "color": g["color"],
        "created_at": g["created_at"].isoformat() if g["created_at"] else None,
        "updated_at": g["updated_at"].isoformat() if g["updated_at"] else None,
    }


@router.post("")
async def create_group(payload: GroupUpsert, user: dict = Depends(get_current_user)):
    name = (payload.name or "").strip()
    if not name:
        raise HTTPException(400, "name is required")
    color = payload.color or "#3b82f6"
    account_ids = payload.account_ids if payload.account_ids is not None else (payload.members or [])
    group_id = str(uuid.uuid4())

    async with core.state.db_pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO account_groups (id, user_id, name, description, account_ids, color, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, NOW(), NOW())
            """,
            group_id,
            user["id"],
            name,
            payload.description,
            account_ids,
            color,
        )

    return {"id": group_id, "name": name, "description": payload.description, "color": color, "account_ids": account_ids, "members": account_ids}


@router.put("/{group_id}")
async def update_group(group_id: str, payload: GroupUpsert, user: dict = Depends(get_current_user)):
    _GROUP_COLS = ACCOUNT_GROUPS_UPDATE_COLUMNS
    updates = []
    params = [group_id, user["id"]]

    if payload.name is not None:
        updates.append(f"{_safe_col('name', _GROUP_COLS)} = ${len(params)+1}")
        params.append(payload.name.strip())

    if payload.description is not None:
        updates.append(f"{_safe_col('description', _GROUP_COLS)} = ${len(params)+1}")
        params.append(payload.description)

    if payload.color is not None:
        updates.append(f"{_safe_col('color', _GROUP_COLS)} = ${len(params)+1}")
        params.append(payload.color)

    # accept either account_ids or members
    if payload.account_ids is not None or payload.members is not None:
        ids = payload.account_ids if payload.account_ids is not None else (payload.members or [])
        updates.append(f"{_safe_col('account_ids', _GROUP_COLS)} = ${len(params)+1}")
        params.append(ids)

    updates.append(f"{_safe_col('updated_at', _GROUP_COLS)} = NOW()")

    async with core.state.db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id FROM account_groups WHERE id = $1 AND user_id = $2",
            group_id,
            user["id"],
        )
        if not row:
            raise HTTPException(404, "Group not found")

        if updates:
            assert_set_fragments_columns(updates, ACCOUNT_GROUPS_UPDATE_COLUMNS)
            await conn.execute(
                f"UPDATE account_groups SET {', '.join(updates)} WHERE id = $1 AND user_id = $2",
                *params,
            )

    return {"status": "updated"}


@router.delete("/{group_id}")
async def delete_group(group_id: str, user: dict = Depends(get_current_user)):
    async with core.state.db_pool.acquire() as conn:
        await conn.execute("DELETE FROM account_groups WHERE id = $1 AND user_id = $2", group_id, user["id"])
    return {"status": "deleted"}
