"""
UploadM8 Database Stage
=======================
PostgreSQL operations for upload processing.
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime, timezone

import asyncpg

from .errors import DatabaseError, ErrorCode
from .context import JobContext


logger = logging.getLogger("uploadm8-worker")


async def load_upload_record(pool: asyncpg.Pool, upload_id: str) -> Dict[str, Any]:
    """Load upload record from database."""
    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM uploads WHERE id = $1",
                upload_id
            )
            if not row:
                raise DatabaseError(
                    f"Upload {upload_id} not found",
                    code=ErrorCode.DB_RECORD_NOT_FOUND
                )
            return dict(row)
    except DatabaseError:
        raise
    except Exception as e:
        raise DatabaseError(
            f"Failed to load upload: {e}",
            code=ErrorCode.DB_CONNECTION_FAILED,
            detail=str(e)
        )


async def load_user_settings(pool: asyncpg.Pool, user_id: str) -> Dict[str, Any]:
    """Load user settings from database."""
    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM user_settings WHERE user_id = $1",
                user_id
            )
            return dict(row) if row else {}
    except Exception as e:
        logger.warning(f"Failed to load user settings: {e}")
        return {}


async def load_user(pool: asyncpg.Pool, user_id: str) -> Dict[str, Any]:
    """Load user record from database."""
    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM users WHERE id = $1",
                user_id
            )
            if not row:
                raise DatabaseError(
                    f"User {user_id} not found",
                    code=ErrorCode.DB_RECORD_NOT_FOUND
                )
            return dict(row)
    except DatabaseError:
        raise
    except Exception as e:
        raise DatabaseError(
            f"Failed to load user: {e}",
            code=ErrorCode.DB_CONNECTION_FAILED,
            detail=str(e)
        )


async def load_platform_token(
    pool: asyncpg.Pool,
    user_id: str,
    platform: str
) -> Optional[str]:
    """Load encrypted platform token."""
    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT token_blob FROM platform_tokens 
                   WHERE user_id = $1 AND platform = $2""",
                user_id, platform
            )
            return row["token_blob"] if row else None
    except Exception as e:
        logger.warning(f"Failed to load platform token: {e}")
        return None


async def update_upload_status(
    pool: asyncpg.Pool,
    upload_id: str,
    status: str,
    **kwargs
) -> None:
    """Update upload status and optional fields."""
    try:
        updates = ["status = $2", "updated_at = NOW()"]
        values = [upload_id, status]
        idx = 3
        
        field_mapping = {
            "processing_started_at": "processing_started_at",
            "processing_finished_at": "processing_finished_at",
            "completed_at": "completed_at",
            "error_code": "error_code",
            "error_detail": "error_detail",
            "trill_score": "trill_score",
            "processed_r2_key": "processed_r2_key",
            "title": "title",
            "caption": "caption",
        }
        
        for key, field in field_mapping.items():
            if key in kwargs:
                updates.append(f"{field} = ${idx}")
                values.append(kwargs[key])
                idx += 1
        
        sql = f"UPDATE uploads SET {', '.join(updates)} WHERE id = $1"
        
        async with pool.acquire() as conn:
            await conn.execute(sql, *values)
        
        logger.debug(f"Updated upload {upload_id} status to {status}")
    except Exception as e:
        raise DatabaseError(
            f"Failed to update upload status: {e}",
            code=ErrorCode.DB_UPDATE_FAILED,
            detail=str(e)
        )


async def mark_processing_started(pool: asyncpg.Pool, ctx: JobContext) -> None:
    """Mark upload as processing started."""
    await update_upload_status(
        pool,
        ctx.upload_id,
        "processing",
        processing_started_at=datetime.now(timezone.utc)
    )
    ctx.status = "processing"


async def mark_processing_complete(
    pool: asyncpg.Pool,
    ctx: JobContext,
    status: str = "completed"
) -> None:
    """Mark upload as processing complete."""
    kwargs = {
        "processing_finished_at": datetime.now(timezone.utc),
        "completed_at": datetime.now(timezone.utc),
    }
    
    if ctx.trill:
        kwargs["trill_score"] = ctx.trill.score
    
    if ctx.processed_r2_key:
        kwargs["processed_r2_key"] = ctx.processed_r2_key
    
    if ctx.caption:
        if ctx.caption.title:
            kwargs["title"] = ctx.final_title
        if ctx.caption.caption:
            kwargs["caption"] = ctx.final_caption
    
    await update_upload_status(pool, ctx.upload_id, status, **kwargs)
    ctx.status = status


async def mark_processing_failed(
    pool: asyncpg.Pool,
    ctx: JobContext,
    error_code: str,
    error_detail: str
) -> None:
    """Mark upload as failed."""
    await update_upload_status(
        pool,
        ctx.upload_id,
        "failed",
        processing_finished_at=datetime.now(timezone.utc),
        error_code=error_code,
        error_detail=error_detail[:1000] if error_detail else None
    )
    ctx.status = "failed"
    ctx.error_code = error_code
    ctx.error_detail = error_detail


async def mark_partial_success(
    pool: asyncpg.Pool,
    ctx: JobContext
) -> None:
    """Mark upload as partially successful."""
    failed_platforms = ctx.get_failed_platforms()
    error_detail = f"Failed platforms: {', '.join(failed_platforms)}"
    
    await update_upload_status(
        pool,
        ctx.upload_id,
        "partial",
        processing_finished_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
        error_code=ErrorCode.PUBLISH_PARTIAL.value,
        error_detail=error_detail,
        trill_score=ctx.trill.score if ctx.trill else None,
        processed_r2_key=ctx.processed_r2_key
    )
    ctx.status = "partial"
