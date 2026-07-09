"""
Meta (Facebook / Instagram) compliance callbacks.

Mounted under /api/webhooks so Meta App Dashboard URLs stay stable:
  POST /api/webhooks/facebook/data-deletion
  GET  /api/webhooks/facebook/data-deletion/status
  POST /api/webhooks/facebook/deauthorize
"""

from __future__ import annotations

import base64
import hashlib
import hmac as _hmac
import json
import logging
import secrets
from typing import Optional
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse

import core.state
from core.config import FRONTEND_URL, META_APP_SECRET

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/webhooks", tags=["webhooks", "meta-compliance"])


def _b64url_decode(raw: str) -> bytes:
    pad = "=" * ((4 - len(raw) % 4) % 4)
    return base64.urlsafe_b64decode((raw + pad).encode("utf-8"))


def parse_meta_signed_request(signed_request: str, app_secret: str) -> Optional[dict]:
    """
    Parse and verify a Meta signed_request (HMAC-SHA256).
    Returns the payload dict on success, or None if invalid.
    """
    if not signed_request or not app_secret or "." not in signed_request:
        return None
    try:
        encoded_sig, payload = signed_request.split(".", 1)
        sig = _b64url_decode(encoded_sig)
        data = json.loads(_b64url_decode(payload).decode("utf-8"))
        expected = _hmac.new(
            app_secret.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        if not _hmac.compare_digest(sig, expected):
            logger.warning("[meta-data-deletion] signed_request signature mismatch")
            return None
        return data if isinstance(data, dict) else None
    except Exception as exc:
        logger.warning("[meta-data-deletion] signed_request parse failed: %s", exc)
        return None


async def _purge_meta_tokens_for_facebook_user(
    facebook_user_id: str,
    *,
    initiated_by: str = "meta_data_deletion_callback",
) -> tuple[int, str]:
    """
    Revoke + delete facebook/instagram platform_tokens tied to this Meta ASID.
    Matching strategy (in order):
      1) token_blob.facebook_user_id == ASID (stored on connect since this change)
      2) account_id == ASID (legacy / rare)
    Returns (tokens_purged, notes).
    """
    if core.state.db_pool is None:
        return 0, "db-unavailable"

    from core.auth import decrypt_blob
    from core.oauth import _revoke_platform_token

    purged = 0
    notes_parts: list[str] = []
    async with core.state.db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, user_id, platform, account_id, account_name, token_blob
            FROM platform_tokens
            WHERE platform IN ('facebook', 'instagram')
              AND revoked_at IS NULL
            """
        )
        for row in rows:
            match = False
            try:
                blob = decrypt_blob(row["token_blob"])
                if str(blob.get("facebook_user_id") or "") == str(facebook_user_id):
                    match = True
            except Exception:
                pass
            if not match and str(row["account_id"] or "") == str(facebook_user_id):
                match = True
            if not match:
                continue

            ok, err = await _revoke_platform_token(row["platform"], row["token_blob"])
            await conn.execute(
                "UPDATE platform_tokens SET revoked_at = NOW() WHERE id = $1",
                row["id"],
            )
            await conn.execute("DELETE FROM platform_tokens WHERE id = $1", row["id"])
            await conn.execute(
                """
                INSERT INTO platform_disconnect_log
                    (user_id, platform, account_id, account_name,
                     revoked_at_provider, provider_revoke_error, initiated_by)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                str(row["user_id"]),
                row["platform"],
                row["account_id"],
                row["account_name"],
                bool(ok),
                (err or None),
                initiated_by,
            )
            purged += 1
            notes_parts.append(f"{row['platform']}:{row['id']}")

    return purged, ("purged=" + ",".join(notes_parts)) if notes_parts else "no-matching-tokens"


@router.post("/facebook/deauthorize")
async def facebook_deauthorize_callback(request: Request):
    """
    Meta Deauthorize Callback — fired when a user removes the app in Facebook settings.
    Configure: App Dashboard → Facebook Login → Settings → Deauthorize Callback URL
      https://auth.uploadm8.com/api/webhooks/facebook/deauthorize
    """
    form = await request.form()
    signed_request = form.get("signed_request") or form.get("signedRequest") or ""
    if isinstance(signed_request, bytes):
        signed_request = signed_request.decode("utf-8", errors="replace")
    signed_request = str(signed_request).strip()

    if not META_APP_SECRET:
        logger.error("[meta-deauthorize] META_APP_SECRET not configured")
        raise HTTPException(503, "Meta app secret not configured")

    data = parse_meta_signed_request(signed_request, META_APP_SECRET)
    if not data:
        raise HTTPException(400, "Invalid signed_request")

    facebook_user_id = str(data.get("user_id") or "").strip()
    if not facebook_user_id:
        raise HTTPException(400, "signed_request missing user_id")

    try:
        purged, notes = await _purge_meta_tokens_for_facebook_user(
            facebook_user_id,
            initiated_by="meta_deauthorize_callback",
        )
        logger.info(
            "[meta-deauthorize] fb_user=%s purged=%s notes=%s",
            facebook_user_id,
            purged,
            notes,
        )
    except Exception as exc:
        logger.error("[meta-deauthorize] purge failed: %s", exc, exc_info=True)

    return JSONResponse({"status": "ok"})


@router.post("/facebook/data-deletion")
async def facebook_data_deletion_callback(request: Request):
    """
    Meta Data Deletion Request Callback.

    Meta POSTs application/x-www-form-urlencoded with signed_request.
    We must return JSON: { "url": "<status page>", "confirmation_code": "<code>" }.

    Configure: Settings → Basic → Data Deletion Request URL
      https://auth.uploadm8.com/api/webhooks/facebook/data-deletion
    """
    form = await request.form()
    signed_request = form.get("signed_request") or form.get("signedRequest") or ""
    if isinstance(signed_request, bytes):
        signed_request = signed_request.decode("utf-8", errors="replace")
    signed_request = str(signed_request).strip()

    if not META_APP_SECRET:
        logger.error("[meta-data-deletion] META_APP_SECRET not configured")
        raise HTTPException(503, "Meta app secret not configured")

    data = parse_meta_signed_request(signed_request, META_APP_SECRET)
    if not data:
        raise HTTPException(400, "Invalid signed_request")

    facebook_user_id = str(data.get("user_id") or "").strip()
    if not facebook_user_id:
        raise HTTPException(400, "signed_request missing user_id")

    confirmation_code = secrets.token_urlsafe(12)
    status = "received"
    tokens_purged = 0
    notes = ""

    try:
        tokens_purged, notes = await _purge_meta_tokens_for_facebook_user(facebook_user_id)
        status = "completed" if tokens_purged else "no_data"
    except Exception as exc:
        status = "failed"
        notes = f"ERROR:{type(exc).__name__}:{exc}"
        logger.error("[meta-data-deletion] purge failed: %s", exc, exc_info=True)

    if core.state.db_pool is not None:
        try:
            async with core.state.db_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO meta_data_deletion_requests
                        (confirmation_code, facebook_user_id, status, tokens_purged, notes, completed_at)
                    VALUES ($1, $2, $3, $4, $5, CASE WHEN $3 IN ('completed','no_data') THEN NOW() ELSE NULL END)
                    """,
                    confirmation_code,
                    facebook_user_id,
                    status,
                    tokens_purged,
                    notes[:4000] if notes else None,
                )
        except Exception as exc:
            logger.error("[meta-data-deletion] audit insert failed: %s", exc)

    status_url = (
        f"{FRONTEND_URL.rstrip('/')}/data-deletion.html"
        f"?code={quote(confirmation_code)}"
    )
    logger.info(
        "[meta-data-deletion] fb_user=%s status=%s purged=%s code=%s",
        facebook_user_id,
        status,
        tokens_purged,
        confirmation_code,
    )
    return JSONResponse(
        {
            "url": status_url,
            "confirmation_code": confirmation_code,
        }
    )


@router.get("/facebook/data-deletion/status")
async def facebook_data_deletion_status(code: str = Query(..., min_length=4, max_length=128)):
    """Public status lookup for Meta deletion confirmation codes (data-deletion.html?code=...)."""
    code = (code or "").strip()
    if not code or core.state.db_pool is None:
        raise HTTPException(404, "Unknown confirmation code")

    async with core.state.db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT confirmation_code, status, tokens_purged, requested_at, completed_at
            FROM meta_data_deletion_requests
            WHERE confirmation_code = $1
            """,
            code,
        )
    if not row:
        raise HTTPException(404, "Unknown confirmation code")

    return {
        "confirmation_code": row["confirmation_code"],
        "status": row["status"],
        "tokens_purged": row["tokens_purged"],
        "requested_at": row["requested_at"].isoformat() if row["requested_at"] else None,
        "completed_at": row["completed_at"].isoformat() if row["completed_at"] else None,
        "message": {
            "received": "We received your Meta data deletion request and are processing it.",
            "completed": "Connected Facebook/Instagram tokens for this request have been removed from UploadM8.",
            "no_data": "We found no active Facebook/Instagram connections for this request. Nothing further to delete on our side.",
            "failed": "We hit an error processing this request. Email privacy@uploadm8.com with this confirmation code.",
        }.get(row["status"], "Request recorded."),
    }
