"""
UploadM8 R2 Storage Stage
=========================
Cloudflare R2 (S3-compatible) operations.
"""

import os
import logging
from pathlib import Path
from typing import Optional

import boto3
from botocore.config import Config

from .errors import R2Error, ErrorCode


logger = logging.getLogger("uploadm8-worker")


# Configuration
R2_ACCOUNT_ID = os.environ.get("R2_ACCOUNT_ID", "")
R2_ACCESS_KEY_ID = os.environ.get("R2_ACCESS_KEY_ID", "")
R2_SECRET_ACCESS_KEY = os.environ.get("R2_SECRET_ACCESS_KEY", "")
R2_BUCKET_NAME = os.environ.get("R2_BUCKET_NAME", "uploadm8-media")
R2_ENDPOINT_URL = os.environ.get("R2_ENDPOINT_URL", "")


def get_s3_client():
    """Get configured S3 client for R2."""
    endpoint = R2_ENDPOINT_URL or f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )


async def download_file(key: str, local_path: Path) -> Path:
    """
    Download file from R2 to local path.
    
    Args:
        key: R2 object key
        local_path: Local destination path
        
    Returns:
        Path to downloaded file
        
    Raises:
        R2Error: If download fails
    """
    try:
        s3 = get_s3_client()
        s3.download_file(R2_BUCKET_NAME, key, str(local_path))
        logger.info(f"Downloaded {key} to {local_path}")
        return local_path
    except Exception as e:
        error_msg = str(e)
        if "404" in error_msg or "NoSuchKey" in error_msg:
            raise R2Error(
                f"File not found: {key}",
                code=ErrorCode.R2_NOT_FOUND,
                detail=error_msg
            )
        raise R2Error(
            f"Failed to download {key}",
            code=ErrorCode.R2_DOWNLOAD_FAILED,
            detail=error_msg
        )


async def upload_file(
    local_path: Path,
    key: str,
    content_type: str = "video/mp4"
) -> str:
    """
    Upload file from local path to R2.
    
    Args:
        local_path: Local source path
        key: R2 destination key
        content_type: MIME type
        
    Returns:
        R2 object key
        
    Raises:
        R2Error: If upload fails
    """
    try:
        s3 = get_s3_client()
        s3.upload_file(
            str(local_path),
            R2_BUCKET_NAME,
            key,
            ExtraArgs={"ContentType": content_type}
        )
        logger.info(f"Uploaded {local_path} to {key}")
        return key
    except Exception as e:
        raise R2Error(
            f"Failed to upload to {key}",
            code=ErrorCode.R2_UPLOAD_FAILED,
            detail=str(e)
        )


async def delete_file(key: str) -> bool:
    """
    Delete file from R2.
    
    Args:
        key: R2 object key
        
    Returns:
        True if deleted, False if not found
    """
    try:
        s3 = get_s3_client()
        s3.delete_object(Bucket=R2_BUCKET_NAME, Key=key)
        logger.info(f"Deleted {key}")
        return True
    except Exception as e:
        logger.warning(f"Failed to delete {key}: {e}")
        return False


async def file_exists(key: str) -> bool:
    """Check if file exists in R2."""
    try:
        s3 = get_s3_client()
        s3.head_object(Bucket=R2_BUCKET_NAME, Key=key)
        return True
    except Exception:
        return False


def get_processed_key(source_key: str) -> str:
    """Generate key for processed video."""
    return source_key.replace("uploads/", "processed/", 1)


def generate_presigned_url(key: str, expires_in: int = 3600) -> str:
    """Generate presigned URL for download."""
    s3 = get_s3_client()
    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": R2_BUCKET_NAME, "Key": key},
        ExpiresIn=expires_in
    )
