"""
UploadM8 Worker Stages
======================
Modular processing pipeline for video uploads.

Stage Execution Order:
1. telemetry_stage - Parse .map files, calculate Trill score
2. caption_stage - Generate AI captions/titles (tier-gated)
3. transcode_stage - Standardize video format
4. hud_stage - Burn speed HUD overlay
5. publish_stage - Distribute to platforms
6. notify_stage - Send Discord notifications

Each stage receives a JobContext and returns an updated context.
"""

from .errors import StageError, SkipStage
from .context import JobContext, create_context

__all__ = [
    'StageError',
    'SkipStage', 
    'JobContext',
    'create_context',
]
