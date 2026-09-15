"""
Opt-in AV training pack writer — fail-soft, never fails the upload.

Gate: ``aiServiceRecognitionTraining`` / recognition_training (consent only, no AIC).
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from stages.ai_service_costs import user_pref_ai_service_enabled
from stages.errors import SkipStage

logger = logging.getLogger("uploadm8-worker")


async def run_av_training_pack_stage(ctx: Any) -> Any:
    prefs = getattr(ctx, "user_prefs", None) or getattr(ctx, "user_settings", None) or {}
    if not isinstance(prefs, dict):
        prefs = {}
    if not user_pref_ai_service_enabled(prefs, "recognition_training", default=False):
        raise SkipStage("Recognition training opt-in disabled (aiServiceRecognitionTraining)")

    user_id = getattr(ctx, "user_id", None)
    upload_id = getattr(ctx, "upload_id", None)
    if not user_id or not upload_id:
        raise SkipStage("Missing user_id/upload_id for AV training pack")

    arts = getattr(ctx, "output_artifacts", None)
    if not isinstance(arts, dict):
        ctx.output_artifacts = {}
        arts = ctx.output_artifacts
    identity = arts.get("content_identity_v1")
    if not identity:
        raise SkipStage("No content_identity_v1 — skip AV training pack")

    from services.av_training_pack import (
        AV_TRAINING_PACK_ARTIFACT,
        build_av_training_pack,
        collect_local_keyframe_paths,
        persist_av_training_pack_to_r2,
    )

    pack = build_av_training_pack(ctx, consent=True)
    local_kf = collect_local_keyframe_paths(ctx)
    try:
        _full, meta = persist_av_training_pack_to_r2(
            pack,
            user_id=user_id,
            upload_id=upload_id,
            local_keyframe_paths=local_kf,
        )
    except Exception as e:
        logger.warning("[%s] AV training pack R2 upload failed (non-fatal): %s", upload_id, e)
        # Still persist compact meta without R2 so dataset build can use teacher flags.
        from services.av_training_pack import artifact_meta_from_pack

        meta = artifact_meta_from_pack(pack)
        meta["r2_error"] = str(e)[:200]
        meta["pack_r2_key"] = ""

    arts[AV_TRAINING_PACK_ARTIFACT] = meta
    try:
        from services.pipeline_ai_trace import record_ai_pipeline_trace

        record_ai_pipeline_trace(
            ctx,
            str(upload_id),
            "av_training_pack",
            {
                "status": "ok" if meta.get("pack_r2_key") else "meta_only",
                "pack_bytes": meta.get("pack_bytes"),
                "keyframe_count": meta.get("keyframe_count"),
                "tl_status": meta.get("tl_status"),
                "fusion_source": meta.get("fusion_source"),
            },
        )
    except Exception:
        pass
    return ctx


__all__ = ["run_av_training_pack_stage"]
