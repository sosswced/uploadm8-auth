"""
UploadM8 R2 Storage Stage
==========================
Cloudflare R2 object storage operations for the worker pipeline.

FIX v2 — Connection resilience for large video files:

  PROBLEM:
    boto3 was caching a single S3 client globally (_s3_client singleton).
    Cloudflare R2 closes idle TCP connections after ~90 seconds.
    When a 191MB video download starts on a stale connection, R2 drops
    it with RemoteDisconnected before sending a single byte.
    The cached client has no way to know the socket is dead until it tries.

  SOLUTION:
    1. No more global singleton — create a fresh client per operation.
       Client creation is cheap (no network call), so this is safe.
    2. TransferConfig with multipart_threshold=32MB and multipart_chunksize=16MB
       so large files use parallel chunked downloads, each chunk on its own
       fresh connection. A single dropped chunk retries independently.
    3. Explicit retry loop (3 attempts) with exponential backoff on
       ConnectionClosedError and ProtocolError — the two errors in the logs.
    4. max_pool_connections=20 to support concurrent job workers.

Exports:
  - download_file(r2_key, local_path)
  - upload_file(local_path, r2_key, content_type)
  - object_exists(r2_key)
  - delete_file(r2_key)
  - generate_presigned_url(r2_key, expires)
  - get_public_url(r2_key)
"""

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("uploadm8-worker")

# ── R2 credentials from environment ──────────────────────────────────────────
R2_ACCOUNT_ID       = os.environ.get("R2_ACCOUNT_ID", "")
R2_ACCESS_KEY_ID    = os.environ.get("R2_ACCESS_KEY_ID", "")
R2_SECRET_ACCESS_KEY = os.environ.get("R2_SECRET_ACCESS_KEY", "")
R2_BUCKET_NAME      = os.environ.get("R2_BUCKET_NAME", "uploadm8-media")
R2_ENDPOINT         = os.environ.get("R2_ENDPOINT", "")
R2_PUBLIC_URL       = os.environ.get("R2_PUBLIC_URL", "")

# ── Retry configuration ───────────────────────────────────────────────────────
_MAX_RETRIES   = int(os.environ.get("R2_MAX_RETRIES", "4"))
_RETRY_DELAY   = float(os.environ.get("R2_RETRY_DELAY_S", "3.0"))

# ── Multipart thresholds ──────────────────────────────────────────────────────
# Files above 32 MB use parallel multipart download/upload.
# Each 16 MB chunk is independent — a dropped connection only retries that chunk.
_MULTIPART_THRESHOLD  = int(os.environ.get("R2_MULTIPART_THRESHOLD_MB", "32")) * 1024 * 1024
_MULTIPART_CHUNKSIZE  = int(os.environ.get("R2_MULTIPART_CHUNKSIZE_MB", "16")) * 1024 * 1024
_MAX_CONCURRENCY      = int(os.environ.get("R2_CONCURRENCY", "4"))


def _make_client():
    """
    Create a fresh boto3 S3 client for Cloudflare R2.

    NOT cached — called per operation so stale TCP connections are never reused.
    Client creation has zero network overhead; it's just a Python object.
    """
    import boto3
    from botocore.config import Config

    endpoint = R2_ENDPOINT or f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        config=Config(
            signature_version="s3v4",
            retries={"max_attempts": 3, "mode": "adaptive"},
            max_pool_connections=20,
            # Increase read timeout for large video files on slow links
            read_timeout=300,
            connect_timeout=30,
        ),
        region_name="auto",
    )


def _make_transfer_config():
    """
    boto3 TransferConfig for multipart chunked transfers.
    Large files (>32MB) split into 16MB chunks downloaded in parallel.
    Each chunk retries independently on connection errors.
    """
    from boto3.s3.transfer import TransferConfig
    return TransferConfig(
        multipart_threshold=_MULTIPART_THRESHOLD,
        multipart_chunksize=_MULTIPART_CHUNKSIZE,
        max_concurrency=_MAX_CONCURRENCY,
        use_threads=True,
    )


def _is_connection_error(exc: Exception) -> bool:
    """Return True for transient connection errors worth retrying."""
    msg = str(exc).lower()
    retryable = (
        "connection was closed",
        "remote end closed",
        "connection aborted",
        "connection reset",
        "broken pipe",
        "protocol error",
        "timeout",
        "timed out",
        "econnreset",
        "connectionerror",
        "endpoint url",
    )
    return any(t in msg for t in retryable)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

async def download_file(r2_key: str, local_path: Path) -> Path:
    """
    Download a file from R2 to a local path.

    Retries up to R2_MAX_RETRIES times on transient connection errors.
    Uses multipart chunked download for files > 32 MB.

    Args:
        r2_key:     Object key in the R2 bucket.
        local_path: Local filesystem path to write to.

    Returns:
        local_path on success.

    Raises:
        Exception if all retries are exhausted.
    """
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"R2 download: {r2_key} -> {local_path}")

    last_exc: Optional[Exception] = None

    for attempt in range(1, _MAX_RETRIES + 1):
        # Remove partial file from a previous failed attempt
        if local_path.exists():
            local_path.unlink(missing_ok=True)

        def _do_download():
            client = _make_client()
            config = _make_transfer_config()
            client.download_file(
                R2_BUCKET_NAME,
                r2_key,
                str(local_path),
                Config=config,
            )

        try:
            await asyncio.get_event_loop().run_in_executor(None, _do_download)

            if not local_path.exists():
                raise FileNotFoundError(f"R2 download produced no file: {r2_key}")

            size = local_path.stat().st_size
            logger.info(f"R2 download complete: {r2_key} ({size:,} bytes) attempt={attempt}")
            return local_path

        except Exception as exc:
            last_exc = exc
            if _is_connection_error(exc) and attempt < _MAX_RETRIES:
                wait = _RETRY_DELAY * (2 ** (attempt - 1))  # 3s, 6s, 12s
                logger.warning(
                    f"R2 download connection error (attempt {attempt}/{_MAX_RETRIES}), "
                    f"retrying in {wait:.0f}s: {exc}"
                )
                await asyncio.sleep(wait)
                continue
            # Non-retryable or out of retries
            logger.error(f"R2 download failed after {attempt} attempt(s): {r2_key} — {exc}")
            raise

    raise RuntimeError(f"R2 download exhausted {_MAX_RETRIES} retries: {r2_key}") from last_exc


async def upload_file(
    local_path: Path,
    r2_key: str,
    content_type: str = "application/octet-stream",
) -> str:
    """
    Upload a local file to R2.

    Retries on transient connection errors.
    Uses multipart chunked upload for files > 32 MB.

    Args:
        local_path:   Path to the local file.
        r2_key:       Destination key in the R2 bucket.
        content_type: MIME type.

    Returns:
        r2_key on success.

    Raises:
        FileNotFoundError if the local file doesn't exist.
        Exception if all retries are exhausted.
    """
    local_path = Path(local_path)
    if not local_path.exists():
        raise FileNotFoundError(f"File not found for R2 upload: {local_path}")

    size = local_path.stat().st_size
    logger.info(f"R2 upload: {local_path.name} -> {r2_key} ({size:,} bytes)")

    last_exc: Optional[Exception] = None

    for attempt in range(1, _MAX_RETRIES + 1):
        def _do_upload():
            client = _make_client()
            config = _make_transfer_config()
            client.upload_file(
                str(local_path),
                R2_BUCKET_NAME,
                r2_key,
                ExtraArgs={"ContentType": content_type},
                Config=config,
            )

        try:
            await asyncio.get_event_loop().run_in_executor(None, _do_upload)
            logger.info(f"R2 upload complete: {r2_key} attempt={attempt}")
            return r2_key

        except Exception as exc:
            last_exc = exc
            if _is_connection_error(exc) and attempt < _MAX_RETRIES:
                wait = _RETRY_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    f"R2 upload connection error (attempt {attempt}/{_MAX_RETRIES}), "
                    f"retrying in {wait:.0f}s: {exc}"
                )
                await asyncio.sleep(wait)
                continue
            logger.error(f"R2 upload failed after {attempt} attempt(s): {r2_key} — {exc}")
            raise

    raise RuntimeError(f"R2 upload exhausted {_MAX_RETRIES} retries: {r2_key}") from last_exc


async def object_exists(r2_key: str) -> bool:
    """
    Return True if the object exists in R2 (HTTP 200 on HEAD).
    False for 404 / NoSuchKey; re-raises on other errors.
    """
    from botocore.exceptions import ClientError

    def _head() -> bool:
        client = _make_client()
        try:
            client.head_object(Bucket=R2_BUCKET_NAME, Key=r2_key)
            return True
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "") or ""
            if code in ("404", "NoSuchKey", "NotFound") or "404" in str(code):
                return False
            raise

    try:
        return await asyncio.get_event_loop().run_in_executor(None, _head)
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "") or ""
        if code in ("404", "NoSuchKey", "NotFound") or "404" in str(code):
            return False
        raise


async def delete_file(r2_key: str) -> None:
    """Delete a file from R2. Non-fatal — logs warning on failure."""
    def _do_delete():
        client = _make_client()
        client.delete_object(Bucket=R2_BUCKET_NAME, Key=r2_key)

    try:
        await asyncio.get_event_loop().run_in_executor(None, _do_delete)
        logger.info(f"R2 delete complete: {r2_key}")
    except Exception as exc:
        logger.warning(f"R2 delete failed (non-fatal): {r2_key} — {exc}")


def generate_presigned_url(r2_key: str, expires: int = 3600) -> str:
    """
    Generate a presigned GET URL for an R2 object.

    Args:
        r2_key:  Object key.
        expires: TTL in seconds (default 1 hour).

    Returns:
        Presigned URL string.
    """
    client = _make_client()
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": R2_BUCKET_NAME, "Key": r2_key},
        ExpiresIn=expires,
    )


def generate_presigned_upload_url(
    r2_key: str,
    content_type: str = "application/octet-stream",
    expires: int = 3600,
) -> str:
    """
    Generate a presigned PUT URL for direct browser-to-R2 uploads.

    Args:
        r2_key:       Destination key.
        content_type: MIME type.
        expires:      TTL in seconds.

    Returns:
        Presigned PUT URL string.
    """
    client = _make_client()
    return client.generate_presigned_url(
        "put_object",
        Params={
            "Bucket": R2_BUCKET_NAME,
            "Key": r2_key,
            "ContentType": content_type,
        },
        ExpiresIn=expires,
    )


def get_public_url(r2_key: str) -> Optional[str]:
    """Return the public CDN URL for an R2 object if R2_PUBLIC_URL is set."""
    if not R2_PUBLIC_URL:
        return None
    return f"{R2_PUBLIC_URL.rstrip('/')}/{r2_key}"
