"""
UploadM8 Stage Errors
=====================
Centralized error handling for the processing pipeline.
"""

from enum import Enum
from typing import Optional


class ErrorCode(str, Enum):
    """Standardized error codes for debugging and frontend display."""
    # Generic
    UNKNOWN = "UNKNOWN"
    INTERNAL = "INTERNAL"
    TIMEOUT = "TIMEOUT"
    CANCELLED = "CANCELLED"
    
    # Auth
    AUTH_FAILED = "AUTH_FAILED"
    TOKEN_EXPIRED = "TOKEN_EXPIRED"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    
    # Upload
    UPLOAD_FAILED = "UPLOAD_FAILED"
    UPLOAD_TOO_LARGE = "UPLOAD_TOO_LARGE"
    INVALID_FILE_TYPE = "INVALID_FILE_TYPE"
    QUOTA_EXCEEDED = "QUOTA_EXCEEDED"
    PRESIGN_FAILED = "PRESIGN_FAILED"
    
    # Processing
    DOWNLOAD_FAILED = "DOWNLOAD_FAILED"
    TRANSCODE_FAILED = "TRANSCODE_FAILED"
    FFMPEG_FAILED = "FFMPEG_FAILED"
    TELEMETRY_PARSE_FAILED = "TELEMETRY_PARSE"
    HUD_FAILED = "HUD_FAILED"
    WATERMARK_FAILED = "WATERMARK_FAILED"
    THUMBNAIL_FAILED = "THUMBNAIL_FAILED"
    
    # AI
    AI_CAPTION_FAILED = "AI_CAPTION_FAILED"
    AI_THUMBNAIL_FAILED = "AI_THUMBNAIL_FAILED"
    AI_HASHTAG_FAILED = "AI_HASHTAG_FAILED"
    OPENAI_RATE_LIMIT = "OPENAI_RATE_LIMIT"
    OPENAI_ERROR = "OPENAI_ERROR"
    
    # Platform
    PLATFORM_AUTH_FAILED = "PLATFORM_AUTH"
    PLATFORM_UPLOAD_FAILED = "PLATFORM_UPLOAD"
    PLATFORM_RATE_LIMIT = "PLATFORM_RATE_LIMIT"
    TIKTOK_FAILED = "TIKTOK_FAILED"
    YOUTUBE_FAILED = "YOUTUBE_FAILED"
    INSTAGRAM_FAILED = "INSTAGRAM_FAILED"
    FACEBOOK_FAILED = "FACEBOOK_FAILED"
    
    # Entitlements
    TIER_BLOCKED = "TIER_BLOCKED"
    FEATURE_DISABLED = "FEATURE_DISABLED"
    
    # Database
    DB_ERROR = "DB_ERROR"
    NOT_FOUND = "NOT_FOUND"
    
    # Network
    NETWORK_ERROR = "NETWORK_ERROR"
    DNS_ERROR = "DNS_ERROR"
    CORS_ERROR = "CORS_ERROR"


class StageError(Exception):
    """
    Error raised during stage processing.
    
    Attributes:
        code: Standardized error code
        message: Human-readable error message
        details: Optional additional context
        retryable: Whether this error can be retried
        stage: Which stage raised the error
    """
    
    def __init__(
        self,
        code: ErrorCode,
        message: str,
        details: Optional[dict] = None,
        retryable: bool = False,
        stage: str = "unknown"
    ):
        self.code = code
        self.message = message
        self.details = details or {}
        self.retryable = retryable
        self.stage = stage
        super().__init__(f"[{code.value}] {message}")
    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging/API response."""
        return {
            "error_code": self.code.value,
            "error_message": self.message,
            "details": self.details,
            "retryable": self.retryable,
            "stage": self.stage,
        }


class SkipStage(Exception):
    """
    Raised when a stage should be skipped (not an error).
    
    Examples:
    - No telemetry file provided
    - AI captions not enabled for tier
    - User disabled HUD overlay
    """
    
    def __init__(self, reason: str, stage: str = "unknown"):
        self.reason = reason
        self.stage = stage
        super().__init__(f"Stage skipped: {reason}")


class CancelRequested(Exception):
    """Raised when a cancel has been requested for this job."""
    
    def __init__(self, upload_id: str):
        self.upload_id = upload_id
        super().__init__(f"Cancel requested for upload {upload_id}")


def error_from_exception(e: Exception, stage: str = "unknown") -> StageError:
    """Convert a generic exception to a StageError."""
    if isinstance(e, StageError):
        return e
    
    error_msg = str(e)
    
    # Try to categorize common errors
    if "timeout" in error_msg.lower():
        return StageError(ErrorCode.TIMEOUT, error_msg, retryable=True, stage=stage)
    if "connection" in error_msg.lower() or "network" in error_msg.lower():
        return StageError(ErrorCode.NETWORK_ERROR, error_msg, retryable=True, stage=stage)
    if "quota" in error_msg.lower() or "limit" in error_msg.lower():
        return StageError(ErrorCode.QUOTA_EXCEEDED, error_msg, retryable=False, stage=stage)
    if "auth" in error_msg.lower() or "unauthorized" in error_msg.lower():
        return StageError(ErrorCode.AUTH_FAILED, error_msg, retryable=False, stage=stage)
    if "ffmpeg" in error_msg.lower():
        return StageError(ErrorCode.FFMPEG_FAILED, error_msg, retryable=False, stage=stage)
    
    return StageError(ErrorCode.UNKNOWN, error_msg, stage=stage)


# HTTP status code mapping
ERROR_HTTP_STATUS = {
    ErrorCode.UNKNOWN: 500,
    ErrorCode.INTERNAL: 500,
    ErrorCode.TIMEOUT: 504,
    ErrorCode.CANCELLED: 499,
    ErrorCode.AUTH_FAILED: 401,
    ErrorCode.TOKEN_EXPIRED: 401,
    ErrorCode.UNAUTHORIZED: 401,
    ErrorCode.FORBIDDEN: 403,
    ErrorCode.UPLOAD_FAILED: 500,
    ErrorCode.UPLOAD_TOO_LARGE: 413,
    ErrorCode.INVALID_FILE_TYPE: 415,
    ErrorCode.QUOTA_EXCEEDED: 429,
    ErrorCode.TIER_BLOCKED: 403,
    ErrorCode.FEATURE_DISABLED: 403,
    ErrorCode.NOT_FOUND: 404,
    ErrorCode.NETWORK_ERROR: 502,
}


def get_http_status(code: ErrorCode) -> int:
    """Get HTTP status code for an error code."""
    return ERROR_HTTP_STATUS.get(code, 500)

# Backward compatibility alias
TelemetryError = StageError
