"""
UploadM8 Processing Stages
==========================
Modular pipeline stages for video processing.

Stages:
1. download - Download files from R2
2. telemetry - Parse .map telemetry files
3. thumbnail - Generate thumbnails (FFmpeg + AI)
4. caption - Generate AI captions/titles/hashtags
5. hud - Burn speed HUD overlay
6. watermark - Apply tier-based watermark
7. upload - Upload processed video to R2
8. publish - Publish to platforms
9. notify - Send notifications
"""

try:
    from .errors import StageError, SkipStage, ErrorCode, CancelRequested
except Exception:
    # Fallback exports so worker can boot even if errors.py drifts again.
    StageError = Exception  # type: ignore
    SkipStage = Exception   # type: ignore
    CancelRequested = Exception  # type: ignore
    ErrorCode = str  # type: ignore
from .context import JobContext, PlatformResult, create_context
from .entitlements import (
    Entitlements,
    get_entitlements_for_tier,
    get_entitlements_from_user,
    entitlements_to_dict,
    can_user_upload,
    can_user_connect_platform,
    get_tier_display_name,
    get_tier_from_lookup_key,
    TIER_CONFIG,
)

__all__ = [
    # Errors
    "StageError",
    "SkipStage",
    "ErrorCode",
    "CancelRequested",
    # Context
    "JobContext",
    "PlatformResult",
    "create_context",
    # Entitlements
    "Entitlements",
    "get_entitlements_for_tier",
    "get_entitlements_from_user",
    "entitlements_to_dict",
    "can_user_upload",
    "can_user_connect_platform",
    "get_tier_display_name",
    "get_tier_from_lookup_key",
    "TIER_CONFIG",
]
