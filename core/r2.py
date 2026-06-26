"""
UploadM8 R2/S3 helpers — presigned URLs, object management.
Extracted from app.py; uses core.config for all R2 credentials.
"""

from __future__ import annotations

import asyncio
import logging

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from core.config import (
    R2_ACCOUNT_ID,
    R2_ACCESS_KEY_ID,
    R2_SECRET_ACCESS_KEY,
    R2_BUCKET_NAME,
    R2_ENDPOINT_URL,
    R2_PRESIGN_UPLOAD_TTL,
    R2_PRESIGN_PUT_UNSIGNED_CONTENT,
)

logger = logging.getLogger("uploadm8-api")


def ensure_r2_presign_configured() -> None:
    """
    Fail fast with a clear message when local/dev env cannot presign PUT URLs.
    Without this, boto3 often raises opaque errors after the upload row is already inserted.
    """
    if not (R2_BUCKET_NAME or "").strip():
        raise RuntimeError("R2_BUCKET_NAME is empty; cannot presign uploads.")
    if not (R2_ACCOUNT_ID or "").strip():
        raise RuntimeError("R2_ACCOUNT_ID is not set; cannot presign uploads.")
    if not (R2_ACCESS_KEY_ID or "").strip() or not (R2_SECRET_ACCESS_KEY or "").strip():
        raise RuntimeError(
            "R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY are not set; cannot presign uploads. "
            "Add R2 API tokens to .env (see Cloudflare R2 S3 API credentials)."
        )


def _normalize_r2_key(key: str) -> str:
    """Normalize object keys to prevent bucket/bucket/... poisoning and signature mismatches."""
    if not key:
        return ""
    k = str(key).lstrip("/")
    bucket = (R2_BUCKET_NAME or "").strip()
    if bucket:
        prefix = bucket + "/"
        # Strip duplicated bucket prefixes (e.g., bucket/bucket/key or bucket/key)
        while k.startswith(prefix):
            k = k[len(prefix):]
    # Collapse accidental double slashes
    while "//" in k:
        k = k.replace("//", "/")
    return k


def platform_avatar_redirect_path(stored_key: str) -> str:
    """Same-origin path for authenticated redirect when presign is unavailable."""
    k = (stored_key or "").strip().lstrip("/")
    if not k.startswith("platform-avatars/"):
        return ""
    parts = k.split("/")
    if len(parts) < 4:
        return ""
    _prefix, uid, plat, filename = parts[0], parts[1], parts[2], parts[3]
    if not uid or not plat or not filename:
        return ""
    return f"/platform-avatars/{uid}/{plat}/{filename}"


def user_avatar_redirect_path(stored_key: str | None) -> str:
    """
    Same-origin path for the signed-in user's profile avatar (R2 key ``avatars/{user_id}/{file}``).
    Presign runs on GET /user-avatars/... so GET /api/me stays fast.
    """
    k = (stored_key or "").strip().lstrip("/")
    if not k.startswith("avatars/"):
        return ""
    parts = k.split("/")
    if len(parts) < 3:
        return ""
    _prefix, uid, filename = parts[0], parts[1], parts[2]
    if not uid or not filename or ".." in filename or "/" in filename:
        return ""
    return f"/user-avatars/{uid}/{filename}"


def resolve_user_profile_avatar_url(avatar_r2_key: str | None, *, presign: bool = False) -> str:
    """
    Profile avatar URL for API responses.

    ``presign=False`` (default): same-origin ``/user-avatars/...`` redirect path.
    ``presign=True``: short-lived R2 GET URL for ``<img src>`` (bearer-only / E2E sessions
    cannot send Authorization on image requests).
    """
    if presign:
        k = (avatar_r2_key or "").strip()
        if not k.startswith("avatars/"):
            return ""
        try:
            return generate_presigned_download_url(k, ttl=3600) or ""
        except Exception:
            return ""
    return user_avatar_redirect_path(avatar_r2_key)


def resolve_stored_account_avatar_url(stored: str | None, *, ttl: int = 86_400, presign: bool = True) -> str:
    """
    platform_tokens.account_avatar may be:
      - https?://... (provider CDN — return as-is)
      - R2 object key (e.g. platform-avatars/{user_id}/{platform}/{id}.jpg) — presign or redirect
      - when ``presign=False`` (bulk list/bootstrap), use same-origin /platform-avatars/... only
    """
    if stored is None:
        return ""
    s = str(stored).strip()
    if not s:
        return ""
    if s.startswith("//"):
        s = "https:" + s
    low = s.lower()
    if low.startswith("http://") or low.startswith("https://"):
        return s
    if not presign:
        return platform_avatar_redirect_path(s)
    try:
        presigned = generate_presigned_download_url(s, ttl=int(ttl)) or ""
        if presigned:
            return presigned
    except Exception as e:
        logger.debug("resolve_stored_account_avatar_url: presign failed for key=%s err=%s", s[:120], e)
    return platform_avatar_redirect_path(s)


def generate_presigned_download_url(key: str, ttl: int = 3600) -> str:
    """Generate a short-lived signed GET URL for a private R2 object."""
    k = _normalize_r2_key(key)
    if not k:
        return ""
    if not R2_BUCKET_NAME:
        raise RuntimeError("Missing R2_BUCKET_NAME env var")
    s3 = get_s3_client()
    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": R2_BUCKET_NAME, "Key": k},
        ExpiresIn=int(ttl),
    )


_s3_client = None


def get_s3_client():
    """Reused boto3 client — avoids constructing a new client on every presign/head."""
    global _s3_client
    if _s3_client is not None:
        return _s3_client
    endpoint = R2_ENDPOINT_URL or f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
    _s3_client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )
    return _s3_client


def r2_presign_get_url(r2_key: str, expires_in: int = 3600) -> str:
    """Generate a short-lived signed URL for a private R2 object."""
    return generate_presigned_download_url(r2_key, ttl=int(expires_in))


def generate_presigned_upload_url(key: str, content_type: str, ttl: int = None) -> str:
    ensure_r2_presign_configured()
    ttl = int(ttl) if ttl is not None else R2_PRESIGN_UPLOAD_TTL
    key = _normalize_r2_key(key)
    s3 = get_s3_client()
    # Binding ContentType in the signature requires the browser to send the exact same header; mismatches -> 403 -> client often reports "network error".
    if R2_PRESIGN_PUT_UNSIGNED_CONTENT:
        params = {"Bucket": R2_BUCKET_NAME, "Key": key}
        logger.info(f"Presigned upload URL (unsigned Content-Type) for key={key[:80]}{'...' if len(key) > 80 else ''} ttl={ttl}s")
    else:
        params = {"Bucket": R2_BUCKET_NAME, "Key": key, "ContentType": content_type}
        logger.info(f"Presigned upload URL generated for key={key[:80]}{'...' if len(key) > 80 else ''} ttl={ttl}s content_type={content_type}")
    url = s3.generate_presigned_url("put_object", Params=params, ExpiresIn=ttl)
    return url


def copy_r2_object_within_bucket(src_key: str, dest_key: str) -> str:
    """
    Server-side copy within ``R2_BUCKET_NAME`` (no download/re-upload).

    Used for durable notification preview keys (e.g. ``notifications/upload-previews/…``)
    so presigned GET URLs stay stable while marketing the same logical object.
    """
    sk = _normalize_r2_key(src_key)
    dk = _normalize_r2_key(dest_key)
    if not sk or not dk:
        raise ValueError("copy_r2_object_within_bucket: empty key")
    if sk == dk:
        return dk
    if not R2_BUCKET_NAME:
        raise RuntimeError("Missing R2_BUCKET_NAME env var")
    s3 = get_s3_client()
    s3.copy_object(
        Bucket=R2_BUCKET_NAME,
        CopySource={"Bucket": R2_BUCKET_NAME, "Key": sk},
        Key=dk,
        MetadataDirective="COPY",
    )
    return dk


def r2_object_exists(key: str) -> bool:
    """Return True only when HeadObject confirms the object exists."""
    return r2_object_head_status(key) == "present"


def r2_object_head_status(key: str) -> str:
    """
    HeadObject result for ``key``: ``present``, ``missing``, or ``unknown``.

    ``unknown`` means R2/network/API error — callers must not treat as missing.
    """
    k = _normalize_r2_key(key or "")
    if not k or not R2_BUCKET_NAME:
        return "missing"
    s3 = get_s3_client()
    try:
        s3.head_object(Bucket=R2_BUCKET_NAME, Key=k)
        return "present"
    except ClientError as e:
        err = (e.response or {}).get("Error") or {}
        code = str(err.get("Code") or "")
        if code in ("404", "NoSuchKey", "NotFound"):
            return "missing"
        logger.warning("r2_object_head_status: head_object failed key=%s code=%s", k[:120], code)
        return "unknown"
    except Exception as e:
        logger.warning("r2_object_head_status: unexpected error key=%s err=%s", k[:120], e)
        return "unknown"


def put_object_bytes(key: str, body: bytes, content_type: str) -> str:
    """Server-side upload of small objects (e.g. bug-report screenshots). Returns normalized key."""
    if not R2_BUCKET_NAME:
        raise RuntimeError("Missing R2_BUCKET_NAME env var")
    k = _normalize_r2_key(key)
    if not k:
        raise ValueError("empty R2 key")
    s3 = get_s3_client()
    s3.put_object(
        Bucket=R2_BUCKET_NAME,
        Key=k,
        Body=body,
        ContentType=content_type or "application/octet-stream",
    )
    return k


async def _delete_r2_objects(keys: list[str]) -> int:
    """
    Delete a list of R2 object keys.  Runs in a thread-pool executor so it
    doesn't block the event loop.  Returns the number of objects deleted.
    """
    if not keys or not R2_BUCKET_NAME:
        return 0
    loop = asyncio.get_event_loop()

    def _bulk_delete(chunk):
        s3 = get_s3_client()
        objects = [{"Key": _normalize_r2_key(k)} for k in chunk if k]
        if not objects:
            return 0
        resp = s3.delete_objects(Bucket=R2_BUCKET_NAME, Delete={"Objects": objects, "Quiet": True})
        errors = resp.get("Errors", [])
        if errors:
            logger.warning(f"R2 delete_objects errors: {errors}")
        return len(objects) - len(errors)

    deleted = 0
    # S3 delete_objects accepts up to 1 000 keys per call
    for i in range(0, len(keys), 1000):
        chunk = keys[i : i + 1000]
        deleted += await loop.run_in_executor(None, _bulk_delete, chunk)
    return deleted
