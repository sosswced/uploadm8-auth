"""UploadM8 stage error primitives.

This module is the single source of truth for pipeline exception types.
Keep it dependency-free (no imports from other stages modules) to avoid
worker boot-time circular dependencies.

Canonical exports:
  - ErrorCode
  - StageError  (base)
  - SkipStage
  - CancelRequested
  - PublishError, StorageError, TelemetryError, TranscodeError,
    ThumbnailError, CaptionError, WatermarkError, VerifyError, HUDError
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class ErrorCode(str, Enum):
    # Generic
    INTERNAL = "INTERNAL"
    CANCELLED = "CANCELLED"
    SKIPPED = "SKIPPED"
    VALIDATION = "VALIDATION"
    NOT_FOUND = "NOT_FOUND"
    UNAUTHORIZED = "UNAUTHORIZED"
    RATE_LIMIT = "RATE_LIMIT"
    UPSTREAM = "UPSTREAM"
    TIMEOUT = "TIMEOUT"
    NETWORK_ERROR = "NETWORK_ERROR"

    # Storage / R2
    STORAGE = "STORAGE"
    DOWNLOAD_FAILED = "DOWNLOAD_FAILED"
    UPLOAD_FAILED = "UPLOAD_FAILED"

    # Processing stages
    PUBLISH = "PUBLISH"
    TELEMETRY = "TELEMETRY"
    TELEMETRY_PARSE_FAILED = "TELEMETRY_PARSE_FAILED"
    TELEMETRY_EMPTY = "TELEMETRY_EMPTY"
    TRANSCODE = "TRANSCODE"
    TRANSCODE_FAILED = "TRANSCODE_FAILED"
    THUMBNAIL = "THUMBNAIL"
    CAPTION = "CAPTION"
    WATERMARK = "WATERMARK"
    VERIFY = "VERIFY"
    HUD = "HUD"
    HUD_GENERATION_FAILED = "HUD_GENERATION_FAILED"

    # FFmpeg
    FFMPEG_FAILED = "FFMPEG_FAILED"
    FFMPEG_NOT_FOUND = "FFMPEG_NOT_FOUND"

    # AI / OpenAI
    AI_CAPTION_FAILED = "AI_CAPTION_FAILED"
    OPENAI_ERROR = "OPENAI_ERROR"
    OPENAI_RATE_LIMIT = "OPENAI_RATE_LIMIT"

    # Platform
    PLATFORM_AUTH_FAILED = "PLATFORM_AUTH_FAILED"
    PLATFORM_UPLOAD_FAILED = "PLATFORM_UPLOAD_FAILED"
    PLATFORM_RATE_LIMIT = "PLATFORM_RATE_LIMIT"

    # Entitlements
    TIER_BLOCKED = "TIER_BLOCKED"
    FEATURE_DISABLED = "FEATURE_DISABLED"
    QUOTA_EXCEEDED = "QUOTA_EXCEEDED"

    # Auth
    AUTH_FAILED = "AUTH_FAILED"
    TOKEN_EXPIRED = "TOKEN_EXPIRED"

    # Database
    DB_ERROR = "DB_ERROR"


@dataclass
class StageError(Exception):
    """Base pipeline error."""

    code: Any = ErrorCode.INTERNAL
    message: str = "Stage error"
    stage: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None
    retryable: bool = False
    detail: Optional[str] = None

    def __post_init__(self):
        super().__init__(str(self))

    def __str__(self) -> str:
        base = f"{self.code}: {self.message}"
        if self.stage:
            base = f"[{self.stage}] {base}"
        return base

    def to_dict(self) -> dict:
        return {
            "error_code": self.code.value if hasattr(self.code, "value") else str(self.code),
            "error_message": self.message,
            "stage": self.stage,
            "retryable": self.retryable,
            "detail": self.detail,
        }


class SkipStage(Exception):
    """Raise to intentionally skip a stage without failing the pipeline."""

    def __init__(self, message: str = "Stage skipped", meta: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.reason = message  # worker.py accesses .reason
        self.message = message
        self.meta = meta or {}


class CancelRequested(Exception):
    """Raise to stop processing due to explicit cancellation."""

    def __init__(self, upload_id: str = ""):
        self.upload_id = upload_id
        super().__init__(f"Cancel requested for upload {upload_id}")


# ---- Specialized errors (all derive from StageError) ----


class PublishError(StageError):
    def __init__(self, message: str = "Publish failed", *, code=ErrorCode.PUBLISH,
                 meta: Optional[Dict[str, Any]] = None, retryable: bool = True, detail: str = None):
        super().__init__(code=code, message=message, stage="publish", meta=meta, retryable=retryable, detail=detail)


class StorageError(StageError):
    def __init__(self, message: str = "Storage/R2 operation failed", *, code=ErrorCode.STORAGE,
                 meta: Optional[Dict[str, Any]] = None, retryable: bool = True, detail: str = None):
        super().__init__(code=code, message=message, stage="storage", meta=meta, retryable=retryable, detail=detail)


class TelemetryError(StageError):
    def __init__(self, message: str = "Telemetry processing failed", *, code=ErrorCode.TELEMETRY,
                 meta: Optional[Dict[str, Any]] = None, retryable: bool = False, detail: str = None):
        super().__init__(code=code, message=message, stage="telemetry", meta=meta, retryable=retryable, detail=detail)


class TranscodeError(StageError):
    def __init__(self, message: str = "Transcode failed", *, code=ErrorCode.TRANSCODE,
                 meta: Optional[Dict[str, Any]] = None, retryable: bool = True, detail: str = None):
        super().__init__(code=code, message=message, stage="transcode", meta=meta, retryable=retryable, detail=detail)


class ThumbnailError(StageError):
    def __init__(self, message: str = "Thumbnail generation failed", *, code=ErrorCode.THUMBNAIL,
                 meta: Optional[Dict[str, Any]] = None, retryable: bool = False, detail: str = None):
        super().__init__(code=code, message=message, stage="thumbnail", meta=meta, retryable=retryable, detail=detail)


class CaptionError(StageError):
    def __init__(self, message: str = "Caption generation failed", *, code=ErrorCode.CAPTION,
                 meta: Optional[Dict[str, Any]] = None, retryable: bool = False, detail: str = None):
        super().__init__(code=code, message=message, stage="caption", meta=meta, retryable=retryable, detail=detail)


class WatermarkError(StageError):
    def __init__(self, message: str = "Watermark stage failed", *, code=ErrorCode.WATERMARK,
                 meta: Optional[Dict[str, Any]] = None, retryable: bool = False, detail: str = None):
        super().__init__(code=code, message=message, stage="watermark", meta=meta, retryable=retryable, detail=detail)


class VerifyError(StageError):
    def __init__(self, message: str = "Verification failed", *, code=ErrorCode.VERIFY,
                 meta: Optional[Dict[str, Any]] = None, retryable: bool = True, detail: str = None):
        super().__init__(code=code, message=message, stage="verify", meta=meta, retryable=retryable, detail=detail)


class HUDError(StageError):
    def __init__(self, message: str = "HUD generation failed", *, code=ErrorCode.HUD,
                 meta: Optional[Dict[str, Any]] = None, retryable: bool = False, detail: str = None):
        super().__init__(code=code, message=message, stage="hud", meta=meta, retryable=retryable, detail=detail)


__all__ = [
    "ErrorCode",
    "StageError",
    "SkipStage",
    "CancelRequested",
    "PublishError",
    "StorageError",
    "TelemetryError",
    "TranscodeError",
    "ThumbnailError",
    "CaptionError",
    "WatermarkError",
    "VerifyError",
    "HUDError",
]
