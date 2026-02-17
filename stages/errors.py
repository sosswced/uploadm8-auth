"""UploadM8 stage error primitives.

This module is the single source of truth for pipeline exception types.
Keep it dependency-free to avoid import-time crashes in workers.

Exports:
  - ErrorCode
  - StageError
  - PublishError
  - TelemetryError
  - StorageError
  - SkipStage
  - CancelRequested
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


@dataclass
class StageError(Exception):
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


class PublishError(StageError):
    def __init__(self, message: str = "Publish failed", stage: Optional[str] = "publish",
                 meta: Optional[Dict[str, Any]] = None, retryable: bool = True):
        super().__init__(ErrorCode.PUBLISH, message, stage=stage, meta=meta, retryable=retryable)


class TelemetryError(StageError):
    def __init__(self, message: str = "Telemetry failed", stage: Optional[str] = "telemetry",
                 meta: Optional[Dict[str, Any]] = None, retryable: bool = False):
        super().__init__(ErrorCode.TELEMETRY, message, stage=stage, meta=meta, retryable=retryable)


class StorageError(StageError):
    def __init__(self, message: str = "Storage error", stage: Optional[str] = "storage",
                 meta: Optional[Dict[str, Any]] = None, retryable: bool = True):
        super().__init__(ErrorCode.STORAGE, message, stage=stage, meta=meta, retryable=retryable)


class SkipStage(Exception):
    """Raise to intentionally skip a stage without failing the pipeline."""
    def __init__(self, message: str = "Stage skipped", meta: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.meta = meta or {}


class CancelRequested(Exception):
    """Raise to stop processing due to explicit cancellation."""
    def __init__(self, message: str = "Cancel requested"):
        super().__init__(message)
