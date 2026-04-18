"""
UploadM8 R2/S3 helpers — presigned URLs, object management.
Extracted from app.py; uses core.config for all R2 credentials.
"""

from __future__ import annotations

import asyncio
import logging

import boto3
from botocore.config import Config

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


def resolve_stored_account_avatar_url(stored: str | None, *, ttl: int = 86_400) -> str:
    """
    platform_tokens.account_avatar may be:
      - https?://... (provider CDN — return as-is)
      - R2 object key (e.g. platform-avatars/{user_id}/{platform}/{id}.jpg) — presign for <img src>

    Without presigning, browsers request same-origin /platform-avatars/... and get 404.
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
    try:
        return generate_presigned_download_url(s, ttl=int(ttl)) or ""
    except Exception as e:
        logger.debug("resolve_stored_account_avatar_url: presign failed for key=%s err=%s", s[:120], e)
        return ""


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


def get_s3_client():
    endpoint = R2_ENDPOINT_URL or f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
    return boto3.client("s3", endpoint_url=endpoint, aws_access_key_id=R2_ACCESS_KEY_ID, aws_secret_access_key=R2_SECRET_ACCESS_KEY, config=Config(signature_version="s3v4"), region_name="auto")


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
