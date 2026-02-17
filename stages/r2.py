"""
UploadM8 R2 Storage Stage
==========================
Cloudflare R2 object storage operations for the worker pipeline.

Exports:
  - download_file(r2_key, local_path)
  - upload_file(local_path, r2_key, content_type)
  - delete_file(r2_key)
  - generate_presigned_url(r2_key, expires)
"""

import os
import logging
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger("uploadm8-worker")

# R2 Configuration
R2_ACCOUNT_ID = os.environ.get("R2_ACCOUNT_ID", "")
R2_ACCESS_KEY_ID = os.environ.get("R2_ACCESS_KEY_ID", "")
R2_SECRET_ACCESS_KEY = os.environ.get("R2_SECRET_ACCESS_KEY", "")
R2_BUCKET_NAME = os.environ.get("R2_BUCKET_NAME", "uploadm8")
R2_ENDPOINT = os.environ.get("R2_ENDPOINT", "")
R2_PUBLIC_URL = os.environ.get("R2_PUBLIC_URL", "")

# Lazy-init S3 client
_s3_client = None


def _get_s3_client():
    """Get or create boto3 S3 client for R2."""
    global _s3_client
    if _s3_client is not None:
        return _s3_client

    import boto3
    from botocore.config import Config

    endpoint = R2_ENDPOINT or f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"

    _s3_client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        config=Config(
            signature_version="s3v4",
            retries={"max_attempts": 3, "mode": "adaptive"},
        ),
        region_name="auto",
    )
    return _s3_client


async def download_file(r2_key: str, local_path: Path) -> Path:
    """
    Download a file from R2 to a local path.

    Args:
        r2_key: Object key in R2 bucket.
        local_path: Where to save the file locally.

    Returns:
        The local_path for convenience.

    Raises:
        Exception: If download fails.
    """
    import asyncio

    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"R2 download: {r2_key} -> {local_path}")

    def _download():
        client = _get_s3_client()
        client.download_file(R2_BUCKET_NAME, r2_key, str(local_path))

    await asyncio.get_event_loop().run_in_executor(None, _download)

    if not local_path.exists():
        raise FileNotFoundError(f"R2 download produced no file: {r2_key}")

    size = local_path.stat().st_size
    logger.info(f"R2 download complete: {r2_key} ({size} bytes)")
    return local_path


async def upload_file(
    local_path: Path,
    r2_key: str,
    content_type: str = "application/octet-stream",
) -> str:
    """
    Upload a local file to R2.

    Args:
        local_path: Path to local file.
        r2_key: Destination key in R2 bucket.
        content_type: MIME type of the file.

    Returns:
        The r2_key for convenience.

    Raises:
        Exception: If upload fails.
    """
    import asyncio

    local_path = Path(local_path)
    if not local_path.exists():
        raise FileNotFoundError(f"File not found for upload: {local_path}")

    size = local_path.stat().st_size
    logger.info(f"R2 upload: {local_path} -> {r2_key} ({size} bytes, {content_type})")

    def _upload():
        client = _get_s3_client()
        client.upload_file(
            str(local_path),
            R2_BUCKET_NAME,
            r2_key,
            ExtraArgs={"ContentType": content_type},
        )

    await asyncio.get_event_loop().run_in_executor(None, _upload)
    logger.info(f"R2 upload complete: {r2_key}")
    return r2_key


async def delete_file(r2_key: str):
    """Delete a file from R2."""
    import asyncio

    logger.info(f"R2 delete: {r2_key}")

    def _delete():
        client = _get_s3_client()
        client.delete_object(Bucket=R2_BUCKET_NAME, Key=r2_key)

    await asyncio.get_event_loop().run_in_executor(None, _delete)


def generate_presigned_url(r2_key: str, expires: int = 3600) -> str:
    """Generate a presigned URL for R2 object access."""
    client = _get_s3_client()
    url = client.generate_presigned_url(
        "get_object",
        Params={"Bucket": R2_BUCKET_NAME, "Key": r2_key},
        ExpiresIn=expires,
    )
    return url


def get_public_url(r2_key: str) -> Optional[str]:
    """Get public URL if R2_PUBLIC_URL is configured."""
    if not R2_PUBLIC_URL:
        return None
    base = R2_PUBLIC_URL.rstrip("/")
    return f"{base}/{r2_key}"
