"""UploadM8 stage error primitives.

This module MUST be import-safe (no imports from other stages modules) to avoid
worker boot-time circular dependencies.

Canonical exports used across the worker/stages:
  - ErrorCode
  - StageError
  - SkipStage
  - CancelRequested
  - PublishError
  - StorageError
  - TelemetryError
  - TranscodeError
  - ThumbnailError
  - CaptionError
  - WatermarkError
  - VerifyError
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


class ErrorCode(str, Enum):
    INTERNAL = "INTERNAL"
    CANCELLED = "CANCELLED"
    SKIPPED = "SKIPPED"
    VALIDATION = "VALIDATION"
    NOT_FOUND = "NOT_FOUND"
    UNAUTHORIZED = "UNAUTHORIZED"
    RATE_LIMIT = "RATE_LIMIT"
    UPSTREAM = "UPSTREAM"
    STORAGE = "STORAGE"
    PUBLISH = "PUBLISH"
    TELEMETRY = "TELEMETRY"
    TRANSCODE = "TRANSCODE"
    THUMBNAIL = "THUMBNAIL"
    CAPTION = "CAPTION"
    WATERMARK = "WATERMARK"
    VERIFY = "VERIFY"


@dataclass
class StageError(Exception):
    """Base pipeline error."""
    code: Any = ErrorCode.INTERNAL
    message: str = "Stage error"
    stage: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None
    retryable: bool = False

    def __str__(self) -> str:
        base = f"{self.code}: {self.message}"
        if self.stage:
            base = f"[{self.stage}] {base}"
        return base


class SkipStage(Exception):
    """Raise to intentionally skip a stage without failing the pipeline."""
    def __init__(self, message: str = "Stage skipped", meta: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.meta = meta or {}


class CancelRequested(Exception):
    """Raise to stop processing due to explicit cancellation."""
    pass


# ---- Specialized errors (all derive from StageError) ----

class PublishError(StageError):
    def __init__(self, message: str = "Publish failed", *, meta: Optional[Dict[str, Any]] = None, retryable: bool = False):
        super().__init__(ErrorCode.PUBLISH, message, stage="publish", meta=meta, retryable=retryable)


class StorageError(StageError):
    def __init__(self, message: str = "Storage/R2 operation failed", *, meta: Optional[Dict[str, Any]] = None, retryable: bool = True):
        super().__init__(ErrorCode.STORAGE, message, stage="storage", meta=meta, retryable=retryable)


class TelemetryError(StageError):
    def __init__(self, message: str = "Telemetry processing failed", *, meta: Optional[Dict[str, Any]] = None, retryable: bool = False):
        super().__init__(ErrorCode.TELEMETRY, message, stage="telemetry", meta=meta, retryable=retryable)


class TranscodeError(StageError):
    def __init__(self, message: str = "Transcode failed", *, meta: Optional[Dict[str, Any]] = None, retryable: bool = True):
        super().__init__(ErrorCode.TRANSCODE, message, stage="transcode", meta=meta, retryable=retryable)


class ThumbnailError(StageError):
    def __init__(self, message: str = "Thumbnail generation failed", *, meta: Optional[Dict[str, Any]] = None, retryable: bool = False):
        super().__init__(ErrorCode.THUMBNAIL, message, stage="thumbnail", meta=meta, retryable=retryable)


class CaptionError(StageError):
    def __init__(self, message: str = "Caption generation failed", *, meta: Optional[Dict[str, Any]] = None, retryable: bool = False):
        super().__init__(ErrorCode.CAPTION, message, stage="caption", meta=meta, retryable=retryable)


class WatermarkError(StageError):
    def __init__(self, message: str = "Watermark stage failed", *, meta: Optional[Dict[str, Any]] = None, retryable: bool = False):
        super().__init__(ErrorCode.WATERMARK, message, stage="watermark", meta=meta, retryable=retryable)


class VerifyError(StageError):
    def __init__(self, message: str = "Verification failed", *, meta: Optional[Dict[str, Any]] = None, retryable: bool = True):
        super().__init__(ErrorCode.VERIFY, message, stage="verify", meta=meta, retryable=retryable)


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
]
