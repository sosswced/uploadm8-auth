"""
UploadM8 stages package.

Keep this module intentionally lightweight.
Rationale: importing heavy modules (context/db/entitlements) here can create circular
imports during service boot (e.g., worker importing stages.errors triggers __init__).
"""

# Export only the error primitives at package import time.
# Other modules should be imported directly (e.g., `from stages.context import JobContext`).

try:
    from .errors import StageError, SkipStage, ErrorCode, CancelRequested
except Exception:
    # Never brick process start due to packaging/import drift
    StageError = Exception  # type: ignore
    SkipStage = Exception   # type: ignore
    CancelRequested = Exception  # type: ignore
    ErrorCode = str  # type: ignore

__all__ = ["StageError", "SkipStage", "ErrorCode", "CancelRequested"]
