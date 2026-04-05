"""
UploadM8 Google Video Intelligence Stage
========================================
Full-video analysis (labels + shot boundaries) via Google Cloud Video Intelligence API.
Complements single-frame Vision API (faces/OCR) and Twelve Labs (semantic description).

Use cases:
  - Segment-level labels (actions, objects, scenes) across the whole clip
  - Shot change timestamps for editing / highlight detection
  - Feed into caption/thumbnail brief via ctx.video_intelligence_context

Input: inline bytes (local file) when size <= VIDEO_INTELLIGENCE_MAX_BYTES (default 10MB),
       OR gs:// URI when VIDEO_INTELLIGENCE_INPUT_URI is set.

Env:
  VIDEO_INTELLIGENCE_ENABLED   (default false — costs apply)
  VIDEO_INTELLIGENCE_MAX_BYTES (default 10485760)
  VIDEO_INTELLIGENCE_TIMEOUT_SEC (default 600)
  VIDEO_INTELLIGENCE_INPUT_URI  Optional gs://... (skips local read)
"""

from __future__ import annotations

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from .errors import SkipStage
from .context import JobContext

logger = logging.getLogger("uploadm8-worker")

VIDEO_INTELLIGENCE_ENABLED = os.environ.get("VIDEO_INTELLIGENCE_ENABLED", "false").lower() == "true"
VIDEO_INTELLIGENCE_MAX_BYTES = int(os.environ.get("VIDEO_INTELLIGENCE_MAX_BYTES", str(10 * 1024 * 1024)))
VIDEO_INTELLIGENCE_TIMEOUT_SEC = int(os.environ.get("VIDEO_INTELLIGENCE_TIMEOUT_SEC", "600"))
VIDEO_INTELLIGENCE_INPUT_URI = (os.environ.get("VIDEO_INTELLIGENCE_INPUT_URI") or "").strip()

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gvi")


def _resolve_gcp_credentials_path() -> Optional[Path]:
    from .vision_stage import _resolve_gcp_credentials_path as _rv

    return _rv()


def _gcp_creds_for_vi():
    from .vision_stage import load_gcp_service_account_credentials

    return load_gcp_service_account_credentials()


def _offset_to_seconds(ts: Any) -> float:
    if ts is None:
        return 0.0
    try:
        sec = float(getattr(ts, "seconds", 0) or 0)
        nanos = float(getattr(ts, "nanos", 0) or 0)
        return sec + nanos / 1e9
    except (TypeError, ValueError, AttributeError):
        return 0.0


def _parse_annotation_result(result: Any) -> Dict[str, Any]:
    """Parse AnnotateVideoResponse into a compact JSON-friendly dict."""
    out: Dict[str, Any] = {
        "segment_labels": [],
        "shot_labels": [],
        "shots": [],
        "summary_text": "",
    }
    try:
        ar_list = getattr(result, "annotation_results", None) or []
        for ar in ar_list:
            for sla in getattr(ar, "segment_label_annotations", None) or []:
                ent = getattr(sla, "entity", None)
                desc = (getattr(ent, "description", None) or "").strip()
                for seg in getattr(sla, "segments", None) or []:
                    conf = float(getattr(seg, "confidence", None) or 0.0)
                    st = getattr(seg, "start_time_offset", None)
                    en = getattr(seg, "end_time_offset", None)
                    out["segment_labels"].append({
                        "description": desc,
                        "confidence": round(conf, 4),
                        "start_s": round(_offset_to_seconds(st), 3),
                        "end_s": round(_offset_to_seconds(en), 3),
                    })
            for sl in getattr(ar, "shot_label_annotations", None) or []:
                ent = getattr(sl, "entity", None)
                desc = (getattr(ent, "description", None) or "").strip()
                for shot in getattr(sl, "shots", None) or []:
                    st = getattr(shot, "start_time_offset", None)
                    en = getattr(shot, "end_time_offset", None)
                    out["shot_labels"].append({
                        "description": desc,
                        "start_s": round(_offset_to_seconds(st), 3),
                        "end_s": round(_offset_to_seconds(en), 3),
                    })
            for shot in getattr(ar, "shot_annotations", None) or []:
                st = getattr(shot, "start_time_offset", None)
                en = getattr(shot, "end_time_offset", None)
                out["shots"].append({
                    "start_s": round(_offset_to_seconds(st), 3),
                    "end_s": round(_offset_to_seconds(en), 3),
                })
    except (TypeError, ValueError, AttributeError) as e:
        logger.warning("[video_intelligence] parse failed: %s", e)
        out["parse_error"] = str(e)

    # Dedupe segment labels by description, keep highest confidence
    best: Dict[str, float] = {}
    for row in out["segment_labels"]:
        d = row.get("description") or ""
        c = float(row.get("confidence") or 0.0)
        if d and c >= best.get(d, 0.0):
            best[d] = c
    top = sorted(best.items(), key=lambda x: -x[1])[:24]
    out["top_labels"] = [f"{d} ({c:.2f})" for d, c in top]
    out["summary_text"] = (
        "Video Intelligence labels: " + ", ".join(d for d, _ in top[:15])
        if top
        else ""
    )
    return out


def _analyze_sync_inline(video_bytes: bytes, creds: Any = None) -> Dict[str, Any]:
    from google.cloud import videointelligence_v1 as vi

    client = (
        vi.VideoIntelligenceServiceClient(credentials=creds)
        if creds is not None
        else vi.VideoIntelligenceServiceClient()
    )
    features = [vi.Feature.LABEL_DETECTION]
    try:
        features.append(vi.Feature.SHOT_CHANGE_DETECTION)
    except AttributeError as e:
        logger.debug("video_intelligence: SHOT_CHANGE_DETECTION unavailable, labels only: %s", e)
    request = vi.AnnotateVideoRequest(
        input_content=video_bytes,
        features=features,
    )
    operation = client.annotate_video(request=request)
    result = operation.result(timeout=VIDEO_INTELLIGENCE_TIMEOUT_SEC)
    return _parse_annotation_result(result)


def _analyze_sync_gcs(uri: str, creds: Any = None) -> Dict[str, Any]:
    from google.cloud import videointelligence_v1 as vi

    client = (
        vi.VideoIntelligenceServiceClient(credentials=creds)
        if creds is not None
        else vi.VideoIntelligenceServiceClient()
    )
    features = [vi.Feature.LABEL_DETECTION]
    try:
        features.append(vi.Feature.SHOT_CHANGE_DETECTION)
    except AttributeError as e:
        logger.debug("video_intelligence: SHOT_CHANGE_DETECTION unavailable (GCS path), labels only: %s", e)
    request = vi.AnnotateVideoRequest(
        input_uri=uri,
        features=features,
    )
    operation = client.annotate_video(request=request)
    result = operation.result(timeout=VIDEO_INTELLIGENCE_TIMEOUT_SEC)
    return _parse_annotation_result(result)


async def run_video_intelligence_stage(ctx: JobContext) -> JobContext:
    """
    Populate ctx.video_intelligence_context with labels + shot boundaries.
    """
    ctx.mark_stage("video_intelligence")

    if not VIDEO_INTELLIGENCE_ENABLED:
        raise SkipStage("Video Intelligence disabled (VIDEO_INTELLIGENCE_ENABLED=false)")

    creds_obj = _gcp_creds_for_vi()
    creds_path = _resolve_gcp_credentials_path()
    if not creds_obj and not creds_path:
        raise SkipStage(
            "GCP credentials not configured for Video Intelligence (same as Vision: "
            "GOOGLE_APPLICATION_CREDENTIALS or social-media-up-*.json in repo root)"
        )

    if creds_path and not (os.environ.get("GCP_SERVICE_ACCOUNT_JSON") or os.environ.get("GOOGLE_CREDENTIALS_JSON")):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(creds_path)

    loop = asyncio.get_event_loop()

    if VIDEO_INTELLIGENCE_INPUT_URI.startswith("gs://"):
        try:
            data = await loop.run_in_executor(
                _executor,
                lambda: _analyze_sync_gcs(VIDEO_INTELLIGENCE_INPUT_URI, creds_obj),
            )
            ctx.video_intelligence_context = data
            logger.info(
                "[video_intelligence]  GCS uri labels=%d shots=%d",
                len(data.get("segment_labels") or []),
                len(data.get("shots") or []),
            )
            return ctx
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("[video_intelligence] GCS analyze failed: %s", e)
            ctx.video_intelligence_context = {"error": str(e)}
            return ctx

    video_path: Optional[Path] = None
    for candidate in (ctx.processed_video_path, ctx.local_video_path):
        if candidate and Path(candidate).exists():
            video_path = Path(candidate)
            break
    if not video_path:
        raise SkipStage("No local video file for Video Intelligence")

    size = video_path.stat().st_size
    if size > VIDEO_INTELLIGENCE_MAX_BYTES:
        raise SkipStage(
            f"Video too large for inline Video Intelligence ({size} bytes > {VIDEO_INTELLIGENCE_MAX_BYTES}); "
            "set VIDEO_INTELLIGENCE_INPUT_URI to gs://... or raise VIDEO_INTELLIGENCE_MAX_BYTES"
        )

    try:
        video_bytes = video_path.read_bytes()
    except (OSError, PermissionError) as e:
        raise SkipStage(f"Cannot read video file: {e}") from e

    try:
        data = await loop.run_in_executor(
            _executor,
            lambda: _analyze_sync_inline(video_bytes, creds_obj),
        )
        ctx.video_intelligence_context = data
        logger.info(
            "[video_intelligence]  inline labels=%d shots=%d",
            len(data.get("segment_labels") or []),
            len(data.get("shots") or []),
        )
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.warning("[video_intelligence] Non-fatal error: %s", e)
        ctx.video_intelligence_context = {"error": str(e)}

    return ctx
