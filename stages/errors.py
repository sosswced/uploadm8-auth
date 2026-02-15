"""
UploadM8 Stage Errors
=====================
Canonical exception types used across worker + stages.

Keep this module dependency-light. Do NOT import other stages here.
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


@dataclass
class StageError(Exception):
    """
    Base pipeline error type.
    """
    code: Any = ErrorCode.INTERNAL
    message: str = "Stage error"
    stage: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None
    retryable: bool = False

    def __str__(self) -> str:
        base = f"{self.code}: {self.message}"
        return f"[{self.stage}] {base}" if self.stage else base


class SkipStage(Exception):
    """Raise to intentionally skip a stage without failing the pipeline."""
    def __init__(self, message: str = "Stage skipped", meta: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.meta = meta or {}


class CancelRequested(Exception):
    """Raise to stop processing due to explicit cancellation."""
    def __init__(self, message: str = "Cancel requested"):
        super().__init__(message)


@dataclass
class PublishError(StageError):
    """Errors specific to platform publishing."""
    code: Any = ErrorCode.PUBLISH
    message: str = "Publish error"
    stage: Optional[str] = "publish"


@dataclass
class StorageError(StageError):
    """Errors specific to object storage / R2."""
    code: Any = ErrorCode.STORAGE
    message: str = "Storage error"
    stage: Optional[str] = "storage"
