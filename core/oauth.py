"""
UploadM8 OAuth helpers — token revocation, state storage, redirect URIs.
Extracted from app.py; uses core.state for Redis, core.config for credentials.
"""

import base64
import hashlib
import json
import logging
import secrets
from typing import Optional
from urllib.parse import urlparse

import httpx
from fastapi import HTTPException

import core.state
from core.config import BASE_URL, TIKTOK_CLIENT_KEY, TIKTOK_CLIENT_SECRET
from core.auth import decrypt_blob

logger = logging.getLogger("uploadm8-api")


async def _revoke_tiktok_token(access_token: str) -> bool:
    """Revoke a TikTok access token at the provider."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                "https://open.tiktokapis.com/v2/oauth/revoke/",
                data={"client_key": TIKTOK_CLIENT_KEY, "client_secret": TIKTOK_CLIENT_SECRET, "token": access_token},
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            return resp.status_code < 300
    except Exception as e:
        logger.warning(f"TikTok token revoke failed: {e}")
        return False


async def _revoke_google_token(access_token: str) -> bool:
    """Revoke a Google/YouTube access token at the provider."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                "https://oauth2.googleapis.com/revoke",
                params={"token": access_token},
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            return resp.status_code < 300
    except Exception as e:
        logger.warning(f"Google token revoke failed: {e}")
        return False


async def _revoke_meta_token(access_token: str) -> bool:
    """Revoke a Facebook/Instagram access token at the provider."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.delete(
                "https://graph.facebook.com/me/permissions",
                params={"access_token": access_token},
            )
            return resp.status_code < 300
    except Exception as e:
        logger.warning(f"Meta token revoke failed: {e}")
        return False


async def _revoke_platform_token(platform: str, token_blob: dict) -> tuple[bool, str]:
    """
    Attempt to revoke `token_blob` at the platform.
    Returns (success: bool, error_msg: str).
    """
    try:
        tok = token_blob if isinstance(token_blob, dict) else {}
        if isinstance(tok, str):
            try:
                tok = json.loads(tok)
            except Exception:
                tok = {}
        # Encrypted blobs: decrypt first
        if "kid" in tok and "ciphertext" in tok:
            try:
                tok = decrypt_blob(tok)
            except Exception:
                return False, "blob-decrypt-failed"
        access_token = tok.get("access_token", "")
        if not access_token:
            return False, "no-access-token"

        if platform == "youtube":
            ok = await _revoke_google_token(access_token)
        elif platform in ("facebook", "instagram"):
            ok = await _revoke_meta_token(access_token)
        elif platform == "tiktok":
            ok = await _revoke_tiktok_token(access_token)
        else:
            return False, f"unsupported-platform:{platform}"

        return ok, ("" if ok else "provider-rejected")
    except Exception as e:
        return False, str(e)


# OAuth state storage -- Redis required for multi-instance safety
_OAUTH_STATE_TTL = 600  # 10 minutes


async def _store_oauth_state(state: str, data: dict):
    """Store OAuth CSRF state in Redis. Raises 503 if Redis is unavailable."""
    if not core.state.redis_client:
        raise HTTPException(503, "OAuth temporarily unavailable (cache offline)")
    await core.state.redis_client.setex(f"oauth_state:{state}", _OAUTH_STATE_TTL, json.dumps(data))


async def _pop_oauth_state(state: str) -> Optional[dict]:
    """Retrieve and delete OAuth CSRF state from Redis. Returns None if missing/expired."""
    if not core.state.redis_client:
        return None
    raw = await core.state.redis_client.get(f"oauth_state:{state}")
    if not raw:
        return None
    await core.state.redis_client.delete(f"oauth_state:{state}")
    return json.loads(raw)


def get_oauth_redirect_uri(platform: str) -> str:
    return f"{BASE_URL}/api/oauth/{platform}/callback"


def sanitize_oauth_parent_origin(origin: Optional[str]) -> str:
    """postMessage target: must match opener origin. Defaults to FRONTEND_URL."""
    from core.config import FRONTEND_URL

    default = FRONTEND_URL.rstrip("/")
    if not origin or not str(origin).strip():
        return default
    try:
        p = urlparse(str(origin).strip())
    except Exception:
        return default
    if p.scheme not in ("http", "https") or not p.netloc:
        return default
    host = (p.hostname or "").lower()
    if not host:
        return default
    fe = urlparse(FRONTEND_URL)
    fe_host = (fe.hostname or "").lower()
    if host in ("localhost", "127.0.0.1") or host.endswith(".localhost"):
        return f"{p.scheme}://{p.netloc.split('/')[0]}"
    if fe_host and (host == fe_host or host.endswith("." + fe_host) or fe_host.endswith("." + host)):
        return f"{p.scheme}://{p.netloc.split('/')[0]}"
    if host.endswith(".uploadm8.com") or host == "uploadm8.com":
        return f"{p.scheme}://{p.netloc.split('/')[0]}"
    return default


def tiktok_pkce_verifier_and_challenge() -> tuple[str, str]:
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode("ascii").rstrip("=")
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest())
        .decode("ascii")
        .rstrip("=")
    )
    return verifier, challenge


async def mirror_oauth_profile_image_to_r2(user_id: str, platform: str, avatar_url: str) -> Optional[str]:
    """Optional: copy provider CDN avatar into R2 for stable hotlinking. Not implemented yet."""
    return None
