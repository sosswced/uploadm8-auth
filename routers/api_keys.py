"""User API keys (integrations) — thin handlers; storage in ``api_keys`` table."""

from __future__ import annotations

import secrets
import uuid
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

import core.state
from core.deps import get_current_user
from core.helpers import _sha256_hex

router = APIRouter(tags=["api-keys"])


class ApiKeyCreate(BaseModel):
    name: str = Field(default="Default", max_length=255)


def _hash_secret(raw: str) -> str:
    return _sha256_hex(raw)


@router.get("/api/keys")
async def list_api_keys(user: dict = Depends(get_current_user)):
    async with core.state.db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, key_prefix, name, scopes, rate_limit, last_used_at, expires_at, created_at
            FROM api_keys
            WHERE user_id = $1 AND revoked_at IS NULL
            ORDER BY created_at DESC
            """,
            user["id"],
        )
    return {
        "keys": [
            {
                "id": str(r["id"]),
                "key_prefix": r["key_prefix"],
                "name": r["name"],
                "scopes": list(r["scopes"] or []),
                "rate_limit": r["rate_limit"],
                "last_used_at": r["last_used_at"].isoformat() if r.get("last_used_at") else None,
                "expires_at": r["expires_at"].isoformat() if r.get("expires_at") else None,
                "created_at": r["created_at"].isoformat() if r.get("created_at") else None,
            }
            for r in rows
        ]
    }


@router.post("/api/keys")
async def create_api_key(body: ApiKeyCreate, user: dict = Depends(get_current_user)):
    raw = f"um8_{secrets.token_urlsafe(32)}"
    prefix = raw[:12]
    kid = uuid.uuid4()
    key_hash = _hash_secret(raw)
    async with core.state.db_pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO api_keys (id, user_id, key_hash, key_prefix, name, scopes, rate_limit)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
            kid,
            user["id"],
            key_hash,
            prefix,
            body.name.strip() or "Default",
            ["read"],
            100,
        )
    return {
        "id": str(kid),
        "key_prefix": prefix,
        "key": raw,
        "warning": "Copy this key now; it will not be shown again.",
    }


@router.delete("/api/keys/{key_id}")
async def revoke_api_key(key_id: str, user: dict = Depends(get_current_user)):
    try:
        kid = uuid.UUID(key_id)
    except ValueError:
        raise HTTPException(400, "Invalid key id")
    async with core.state.db_pool.acquire() as conn:
        res = await conn.execute(
            """
            UPDATE api_keys SET revoked_at = NOW()
            WHERE id = $1 AND user_id = $2 AND revoked_at IS NULL
            """,
            kid,
            user["id"],
        )
    if res == "UPDATE 0":
        raise HTTPException(404, "Key not found")
    return {"status": "revoked"}
