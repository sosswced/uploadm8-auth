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

from .errors import StageError, SkipStage, ErrorCode
from .context import JobContext, create_context
from .entitlements import (
    Entitlements,
    TIER_CONFIG,
    get_entitlements_for_tier,
    get_entitlements_from_user,
    can_user_upload,
    can_user_connect_platform,
)

__all__ = [
    'StageError',
    'SkipStage',
    'ErrorCode',
    'JobContext',
    'create_context',
    'Entitlements',
    'TIER_CONFIG',
    'get_entitlements_for_tier',
    'get_entitlements_from_user',
    'can_user_upload',
    'can_user_connect_platform',
]
