"""
UploadM8 R2 Storage Stage
=========================
Cloudflare R2 (S3-compatible) file operations.
"""

import os
import logging
from pathlib import Path
from typing import Optional

import boto3
from botocore.config import Config

from .errors import StageError, ErrorCode

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


async def download_file(r2_key: str, local_path: Path) -> Path:
    """
    Download a file from R2 to local path.
    
    Args:
        r2_key: Key in R2 bucket
        local_path: Local path to save file
        
    Returns:
        Path to downloaded file
        
    Raises:
        StageError: If download fails
    """
    try:
        logger.info(f"Downloading {r2_key} to {local_path}")
        s3 = get_s3_client()
        
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        s3.download_file(R2_BUCKET_NAME, r2_key, str(local_path))
        
        if not local_path.exists():
            raise StageError(
                ErrorCode.DOWNLOAD_FAILED,
                f"Downloaded file not found at {local_path}",
                stage="download"
            )
        
        logger.info(f"Downloaded {local_path.stat().st_size} bytes")
        return local_path
        
    except StageError:
        raise
    except Exception as e:
        raise StageError(
            ErrorCode.DOWNLOAD_FAILED,
            f"Failed to download {r2_key}: {str(e)}",
            stage="download",
            retryable=True
        )


async def upload_file(local_path: Path, r2_key: str, content_type: str = None) -> str:
    """
    Upload a file from local path to R2.
    
    Args:
        local_path: Local file path
        r2_key: Target key in R2 bucket
        content_type: Optional content type
        
    Returns:
        R2 key of uploaded file
        
    Raises:
        StageError: If upload fails
    """
    try:
        if not local_path.exists():
            raise StageError(
                ErrorCode.UPLOAD_FAILED,
                f"Local file not found: {local_path}",
                stage="upload"
            )
        
        logger.info(f"Uploading {local_path} to {r2_key}")
        s3 = get_s3_client()
        
        extra_args = {}
        if content_type:
            extra_args["ContentType"] = content_type
        
        s3.upload_file(str(local_path), R2_BUCKET_NAME, r2_key, ExtraArgs=extra_args or None)
        
        logger.info(f"Uploaded {local_path.stat().st_size} bytes to {r2_key}")
        return r2_key
        
    except StageError:
        raise
    except Exception as e:
        raise StageError(
            ErrorCode.UPLOAD_FAILED,
            f"Failed to upload to {r2_key}: {str(e)}",
            stage="upload",
            retryable=True
        )


async def delete_file(r2_key: str):
    """Delete a file from R2."""
    try:
        s3 = get_s3_client()
        s3.delete_object(Bucket=R2_BUCKET_NAME, Key=r2_key)
        logger.info(f"Deleted {r2_key}")
    except Exception as e:
        logger.warning(f"Failed to delete {r2_key}: {e}")


async def file_exists(r2_key: str) -> bool:
    """Check if a file exists in R2."""
    try:
        s3 = get_s3_client()
        s3.head_object(Bucket=R2_BUCKET_NAME, Key=r2_key)
        return True
    except:
        return False


async def get_file_size(r2_key: str) -> Optional[int]:
    """Get size of a file in R2."""
    try:
        s3 = get_s3_client()
        response = s3.head_object(Bucket=R2_BUCKET_NAME, Key=r2_key)
        return response.get("ContentLength")
    except:
        return None


def generate_presigned_url(r2_key: str, ttl: int = 3600, for_upload: bool = False, content_type: str = None) -> str:
    """
    Generate a presigned URL for R2.
    
    Args:
        r2_key: Key in R2 bucket
        ttl: Time to live in seconds
        for_upload: If True, generate PUT URL; otherwise GET
        content_type: Content type for upload
        
    Returns:
        Presigned URL
    """
    s3 = get_s3_client()
    
    if for_upload:
        params = {"Bucket": R2_BUCKET_NAME, "Key": r2_key}
        if content_type:
            params["ContentType"] = content_type
        return s3.generate_presigned_url("put_object", Params=params, ExpiresIn=ttl)
    else:
        return s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": R2_BUCKET_NAME, "Key": r2_key},
            ExpiresIn=ttl
        )


def get_public_url(r2_key: str) -> str:
    """Get public URL for an R2 object (if bucket is public)."""
    public_domain = os.environ.get("R2_PUBLIC_DOMAIN")
    if public_domain:
        return f"https://{public_domain}/{r2_key}"
    return generate_presigned_url(r2_key, ttl=86400)
