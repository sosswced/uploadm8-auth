"""stages package initializer.

IMPORTANT: keep this import-light. Importing submodules here can create
circular imports and boot-time crashes.

Worker/stages should import directly from stages.errors, stages.context, etc.

Dead-code policy: many stage modules export helpers used only from worker.py or
app.py. Before deleting a public function, grep the repo and run the test suite;
imports are the source of truth, not ``stages`` package re-exports.
"""

import logging

_logger = logging.getLogger("uploadm8-worker.stages")

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
except Exception as e:
    _logger.warning("stages package: optional error re-exports failed: %s", e)
    __all__ = []
