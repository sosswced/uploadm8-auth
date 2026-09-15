"""
UploadM8 authentication & encryption helpers — extracted from app.py.
Encryption keys, JWT, password hashing, refresh tokens.
"""

import json
import base64
import secrets
import logging
from datetime import timedelta
from typing import Optional

import bcrypt
import jwt
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from fastapi import HTTPException

import core.state
from core.config import (
    TOKEN_ENC_KEYS,
    JWT_SECRET,
    JWT_ISSUER,
    JWT_AUDIENCE,
    ACCESS_TOKEN_MINUTES,
    REFRESH_TOKEN_DAYS,
    REFRESH_TOKEN_DAYS_ADMIN,
    REFRESH_TOKEN_DAYS_SESSION,
)
from core.helpers import _now_utc, _sha256_hex

logger = logging.getLogger("uploadm8-api")


def refresh_ttl_days(
    *,
    remember: bool = True,
    role: Optional[str] = None,
    tier: Optional[str] = None,
) -> int:
    """
    Meta-like session policy:
    - remember=True (default): REFRESH_TOKEN_DAYS (30)
    - remember=False (café / shared machine): REFRESH_TOKEN_DAYS_SESSION (1)
    - admin / master_admin: capped at REFRESH_TOKEN_DAYS_ADMIN (7)
    """
    days = REFRESH_TOKEN_DAYS if remember else REFRESH_TOKEN_DAYS_SESSION
    role_l = str(role or "").strip().lower()
    tier_l = str(tier or "").strip().lower()
    if role_l in ("admin", "master_admin") or tier_l == "master_admin":
        days = min(int(days), int(REFRESH_TOKEN_DAYS_ADMIN))
    return max(1, int(days))

def parse_enc_keys():
    if not TOKEN_ENC_KEYS:
        raise RuntimeError("Missing required env var: TOKEN_ENC_KEYS")
    keys = {}
    for part in TOKEN_ENC_KEYS.replace("\\n", "").split(","):
        if ":" in part:
            kid, b64 = part.split(":", 1)
            keys[kid.strip()] = base64.b64decode(b64.strip())
    if not keys:
        raise RuntimeError("TOKEN_ENC_KEYS is set but no valid keys were parsed")
    return keys

def init_enc_keys():
    core.state.ENC_KEYS = parse_enc_keys()
    core.state.CURRENT_KEY_ID = list(core.state.ENC_KEYS.keys())[-1]

def encrypt_blob(data: dict) -> dict:
    key = core.state.ENC_KEYS[core.state.CURRENT_KEY_ID]
    aesgcm = AESGCM(key)
    nonce = secrets.token_bytes(12)
    ct = aesgcm.encrypt(nonce, json.dumps(data).encode(), None)
    return {"kid": core.state.CURRENT_KEY_ID, "nonce": base64.b64encode(nonce).decode(), "ciphertext": base64.b64encode(ct).decode()}

def decrypt_blob(blob):
    if isinstance(blob, str): blob = json.loads(blob)
    key = core.state.ENC_KEYS.get(blob.get("kid", "v1"))
    if not key: raise ValueError("Unknown key")
    aesgcm = AESGCM(key)
    return json.loads(aesgcm.decrypt(base64.b64decode(blob["nonce"]), base64.b64decode(blob["ciphertext"]), None))

def hash_password(pw: str) -> str:
    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt(12)).decode()

def verify_password(pw: str, hashed: str) -> bool:
    try: return bcrypt.checkpw(pw.encode(), hashed.encode())
    except Exception: return False

def create_access_jwt(user_id: str) -> str:
    now = _now_utc()
    return jwt.encode({"sub": user_id, "iat": int(now.timestamp()), "exp": int((now + timedelta(minutes=ACCESS_TOKEN_MINUTES)).timestamp()), "iss": JWT_ISSUER, "aud": JWT_AUDIENCE}, JWT_SECRET, algorithm="HS256")

def verify_access_jwt(token: str) -> Optional[str]:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"], audience=JWT_AUDIENCE, issuer=JWT_ISSUER)
        return payload.get("sub")
    except jwt.ExpiredSignatureError:
        logger.warning("JWT token expired")
        return None
    except (jwt.InvalidAudienceError, jwt.InvalidIssuerError) as e:
        logger.warning(f"JWT verification failed: {type(e).__name__}")
        return None
    except Exception as e:
        logger.warning(f"JWT verification failed: {e}")
        return None

async def create_refresh_token(conn, user_id: str, *, days: Optional[int] = None, expires_at=None) -> str:
    token = secrets.token_urlsafe(64)
    if expires_at is None:
        ttl_days = int(days) if days is not None else int(REFRESH_TOKEN_DAYS)
        expires_at = _now_utc() + timedelta(days=max(1, ttl_days))
    await conn.execute(
        "INSERT INTO refresh_tokens (user_id, token_hash, expires_at) VALUES ($1, $2, $3)",
        user_id,
        _sha256_hex(token),
        expires_at,
    )
    return token

async def rotate_refresh_token(conn, old_token: str):
    h = _sha256_hex(old_token)
    row = await conn.fetchrow(
        "SELECT id, user_id, expires_at, revoked_at, created_at FROM refresh_tokens WHERE token_hash=$1",
        h,
    )
    if not row: raise HTTPException(401, "Invalid")
    if row["revoked_at"]:
        await conn.execute("UPDATE refresh_tokens SET revoked_at=NOW() WHERE user_id=$1 AND revoked_at IS NULL", row["user_id"])
        raise HTTPException(401, "Reuse detected")
    if row["expires_at"] < _now_utc(): raise HTTPException(401, "Expired")
    u = await conn.fetchrow(
        "SELECT email_verified, status, role, subscription_tier FROM users WHERE id = $1",
        row["user_id"],
    )
    if not u or u["status"] == "banned":
        await conn.execute(
            "UPDATE refresh_tokens SET revoked_at=NOW() WHERE user_id=$1 AND revoked_at IS NULL",
            row["user_id"],
        )
        raise HTTPException(401, "Invalid")
    if u.get("email_verified") is False:
        raise HTTPException(
            status_code=403,
            detail={
                "message": "Please verify your email to continue.",
                "code": "email_not_verified",
            },
        )
    await conn.execute("UPDATE refresh_tokens SET revoked_at=NOW() WHERE id=$1", row["id"])
    role_cap_days = refresh_ttl_days(
        remember=True,
        role=u.get("role"),
        tier=u.get("subscription_tier"),
    )
    now = _now_utc()
    created = row.get("created_at") or now
    lifetime = row["expires_at"] - created
    # Short café sessions (Remember me off): do not slide past the original absolute expiry.
    session_bound = timedelta(days=int(REFRESH_TOKEN_DAYS_SESSION) + 1)
    if lifetime <= session_bound:
        new_expires = row["expires_at"]
    else:
        new_expires = now + timedelta(days=role_cap_days)
    if new_expires <= now:
        raise HTTPException(401, "Expired")
    new_refresh = await create_refresh_token(conn, row["user_id"], expires_at=new_expires)
    refresh_max_age = max(60, int((new_expires - now).total_seconds()))
    return create_access_jwt(str(row["user_id"])), new_refresh, refresh_max_age
