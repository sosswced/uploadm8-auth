"""UploadM8 stages package.

Keep this file import-light to avoid boot-time circular imports.
"""

# Re-export error primitives only (safe).
from .errors import (
    ErrorCode,
    StageError,
    SkipStage,
    CancelRequested,
    PublishError,
    StorageError,
    TelemetryError,
    TranscodeError,
    ThumbnailError,
    CaptionError,
    WatermarkError,
    VerifyError,
)

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
