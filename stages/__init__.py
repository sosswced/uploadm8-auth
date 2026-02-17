"""stages package initializer.

IMPORTANT: keep this import-light. Importing submodules here can create
circular imports and boot-time crashes.

Worker/stages should import directly from stages.errors, stages.context, etc.
"""

# Optional re-exports (safe). Do not hard-fail if errors module changes.
try:
    from .errors import (
        ErrorCode,
        StageError,
        PublishError,
        TelemetryError,
        StorageError,
        SkipStage,
        CancelRequested,
    )
    __all__ = [
        "ErrorCode",
        "StageError",
        "PublishError",
        "TelemetryError",
        "StorageError",
        "SkipStage",
        "CancelRequested",
    ]
except Exception:
    __all__ = []
