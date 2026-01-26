"""
UploadM8 Worker Errors
======================
Typed exceptions and error codes for stage processing.
"""

from enum import Enum
from typing import Optional


class ErrorCode(str, Enum):
    """Standardized error codes for tracking and debugging."""
    # General
    UNKNOWN = "unknown_error"
    TIMEOUT = "timeout"
    
    # R2/Storage
    R2_DOWNLOAD_FAILED = "r2_download_failed"
    R2_UPLOAD_FAILED = "r2_upload_failed"
    R2_NOT_FOUND = "r2_not_found"
    
    # Database
    DB_CONNECTION_FAILED = "db_connection_failed"
    DB_RECORD_NOT_FOUND = "db_record_not_found"
    DB_UPDATE_FAILED = "db_update_failed"
    
    # Telemetry
    TELEMETRY_PARSE_FAILED = "telemetry_parse_failed"
    TELEMETRY_INVALID_FORMAT = "telemetry_invalid_format"
    TELEMETRY_EMPTY = "telemetry_empty"
    
    # Transcode
    FFMPEG_NOT_FOUND = "ffmpeg_not_found"
    FFMPEG_FAILED = "ffmpeg_failed"
    VIDEO_CORRUPT = "video_corrupt"
    
    # HUD
    HUD_GENERATION_FAILED = "hud_generation_failed"
    
    # Caption
    CAPTION_GENERATION_FAILED = "caption_generation_failed"
    CAPTION_TIER_BLOCKED = "caption_tier_blocked"
    
    # Publish
    PUBLISH_ALL_FAILED = "publish_all_failed"
    PUBLISH_PARTIAL = "publish_partial"
    PLATFORM_NOT_CONNECTED = "platform_not_connected"
    PLATFORM_TOKEN_EXPIRED = "platform_token_expired"
    PLATFORM_TOKEN_DECRYPT_FAILED = "platform_token_decrypt_failed"
    TIKTOK_UPLOAD_FAILED = "tiktok_upload_failed"
    YOUTUBE_UPLOAD_FAILED = "youtube_upload_failed"
    INSTAGRAM_UPLOAD_FAILED = "instagram_upload_failed"
    FACEBOOK_UPLOAD_FAILED = "facebook_upload_failed"
    
    # Notification
    DISCORD_NOTIFY_FAILED = "discord_notify_failed"


class StageError(Exception):
    """Base exception for stage processing errors."""
    
    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.UNKNOWN,
        detail: Optional[str] = None,
        recoverable: bool = False
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.detail = detail
        self.recoverable = recoverable
    
    def to_dict(self) -> dict:
        return {
            "error": self.message,
            "code": self.code.value,
            "detail": self.detail,
            "recoverable": self.recoverable,
        }


class SkipStage(Exception):
    """Raised when a stage should be skipped (not an error)."""
    
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class R2Error(StageError):
    """R2/S3 storage errors."""
    pass


class DatabaseError(StageError):
    """Database operation errors."""
    pass


class TelemetryError(StageError):
    """Telemetry processing errors."""
    pass


class TranscodeError(StageError):
    """Video transcoding errors."""
    pass


class HUDError(StageError):
    """HUD overlay generation errors."""
    pass


class CaptionError(StageError):
    """Caption/title generation errors."""
    pass


class PublishError(StageError):
    """Platform publishing errors."""
    pass


class NotifyError(StageError):
    """Notification delivery errors."""
    pass
