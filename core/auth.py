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
)
from core.helpers import _now_utc, _sha256_hex

logger = logging.getLogger("uploadm8-api")

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

async def create_refresh_token(conn, user_id: str) -> str:
    token = secrets.token_urlsafe(64)
    await conn.execute("INSERT INTO refresh_tokens (user_id, token_hash, expires_at) VALUES ($1, $2, $3)", user_id, _sha256_hex(token), _now_utc() + timedelta(days=REFRESH_TOKEN_DAYS))
    return token

async def rotate_refresh_token(conn, old_token: str):
    h = _sha256_hex(old_token)
    row = await conn.fetchrow("SELECT id, user_id, expires_at, revoked_at FROM refresh_tokens WHERE token_hash=$1", h)
    if not row: raise HTTPException(401, "Invalid")
    if row["revoked_at"]:
        await conn.execute("UPDATE refresh_tokens SET revoked_at=NOW() WHERE user_id=$1 AND revoked_at IS NULL", row["user_id"])
        raise HTTPException(401, "Reuse detected")
    if row["expires_at"] < _now_utc(): raise HTTPException(401, "Expired")
    await conn.execute("UPDATE refresh_tokens SET revoked_at=NOW() WHERE id=$1", row["id"])
    return create_access_jwt(str(row["user_id"])), await create_refresh_token(conn, row["user_id"])
