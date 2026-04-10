"""
UploadM8 FastAPI dependencies — extracted from app.py.
get_current_user, require_admin, require_master_admin.
"""

from typing import Optional

from fastapi import HTTPException, Request, Header, Depends

import core.state
from core.auth import verify_access_jwt
from core.wallet import get_wallet, daily_refill


async def get_current_user(request: Request, authorization: Optional[str] = Header(None)):
    auth_token = authorization[7:] if authorization and authorization.startswith("Bearer ") else None
    if not auth_token:
        raise HTTPException(401, "Missing authorization")
    user_id = verify_access_jwt(auth_token)
    if not user_id: raise HTTPException(401, "Invalid token")

    async with core.state.db_pool.acquire() as conn:
        user = await conn.fetchrow("SELECT * FROM users WHERE id = $1", user_id)
        if not user: raise HTTPException(401, "User not found")
        if user["status"] == "banned": raise HTTPException(403, "Account suspended")
        await conn.execute("UPDATE users SET last_active_at = NOW() WHERE id = $1", user_id)
        # Daily token refill
        await daily_refill(conn, user_id, user["subscription_tier"])
        wallet = await get_wallet(conn, user_id)
        return {**dict(user), "wallet": wallet}

async def require_admin(user: dict = Depends(get_current_user)):
    if user.get("role") not in ("admin", "master_admin"): raise HTTPException(403, "Admin required")
    return user

async def require_master_admin(user: dict = Depends(get_current_user)):
    if user.get("role") != "master_admin": raise HTTPException(403, "Master admin required")
    return user
