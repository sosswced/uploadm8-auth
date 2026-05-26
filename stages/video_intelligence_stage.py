"""
UploadM8 Google Video Intelligence Stage
========================================
Full-video analysis (labels + shot boundaries) via Google Cloud Video Intelligence API.
Complements single-frame Vision API (faces/OCR) and Twelve Labs (semantic description).

Use cases:
  - Segment-level labels (actions, objects, scenes) across the whole clip
  - Shot change timestamps for editing / highlight detection
  - Feed into caption/thumbnail brief via ctx.video_intelligence_context

Input: inline bytes (local file) when size <= VIDEO_INTELLIGENCE_MAX_BYTES (default 100 MiB),
       OR gs:// URI when VIDEO_INTELLIGENCE_INPUT_URI is set.

GCP: enable the **Video Intelligence API** on the worker service account project
(``gcloud services enable videointelligence.googleapis.com``) or annotate calls
return permission/API-not-enabled errors that surface as ``ctx.video_intelligence_context.error``.

Env:
  VIDEO_INTELLIGENCE_MAX_BYTES (default 104857600 = 100 MiB; clamped 10 MiB - 1 GiB; loads full file into RAM)
  VIDEO_INTELLIGENCE_TIMEOUT_SEC (default 1800)  LRO wait for annotate_video result()
  VIDEO_INTELLIGENCE_INPUT_URI  Optional gs://... (skips local read; best for huge files)
  VIDEO_INTELLIGENCE_MAX_DURATION_SEC  If >0, skip when ffprobe duration exceeds this seconds.
                                       Default 0 = never skip on our side. Google still documents ~3h max
                                       per annotate request; longer clips may fail or need gs:// input.
  VIDEO_INTELLIGENCE_INCLUDE_SHOT_CHANGE  If false, label detection only (often faster than +shots).
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
from .ai_service_costs import user_pref_ai_service_enabled
from services.provider_error_trace import append_provider_error

logger = logging.getLogger("uploadm8-worker")

# Caption-accuracy defaults: fewer inline rejects, long enough LRO wait to match worker stage budget.
_VI_DEFAULT_MAX_BYTES = 100 * 1024 * 1024  # 100 MiB
_VI_MIN_MAX_BYTES = 10 * 1024 * 1024  # env floor
_VI_ABS_MAX_BYTES = 1024 * 1024 * 1024  # 1 GiB ceiling (RAM / safety)
_VI_DEFAULT_TIMEOUT_SEC = 1800  # 30 min
_GOOGLE_VI_ANNOTATE_MAX_DURATION_SEC = 3 * 3600  # Google documents ~3h per annotate (label/shots)


def _parse_vi_max_bytes() -> int:
    raw = (os.environ.get("VIDEO_INTELLIGENCE_MAX_BYTES") or "").strip()
    if not raw:
        return _VI_DEFAULT_MAX_BYTES
    try:
        v = int(raw, 10)
    except ValueError:
        return _VI_DEFAULT_MAX_BYTES
    return max(_VI_MIN_MAX_BYTES, min(v, _VI_ABS_MAX_BYTES))


def _parse_vi_timeout_sec() -> int:
    raw = (os.environ.get("VIDEO_INTELLIGENCE_TIMEOUT_SEC") or "").strip()
    if not raw:
        return _VI_DEFAULT_TIMEOUT_SEC
    try:
        v = int(raw, 10)
    except ValueError:
        return _VI_DEFAULT_TIMEOUT_SEC
    return max(120, min(v, 7200))


def _parse_vi_max_duration_sec() -> float:
    """0 = do not skip by duration (maximize VI coverage). Negative treated as 0."""
    raw = (os.environ.get("VIDEO_INTELLIGENCE_MAX_DURATION_SEC") or "0").strip()
    try:
        v = float(raw)
    except ValueError:
        return 0.0
    return max(0.0, v)


VIDEO_INTELLIGENCE_MAX_BYTES = _parse_vi_max_bytes()
VIDEO_INTELLIGENCE_TIMEOUT_SEC = _parse_vi_timeout_sec()
VIDEO_INTELLIGENCE_INPUT_URI = (os.environ.get("VIDEO_INTELLIGENCE_INPUT_URI") or "").strip()
VIDEO_INTELLIGENCE_MAX_DURATION_SEC = _parse_vi_max_duration_sec()
VIDEO_INTELLIGENCE_STAGE_ENABLED = (
    os.environ.get("VIDEO_INTELLIGENCE_STAGE_ENABLED", "true").lower() == "true"
)
VIDEO_INTELLIGENCE_INCLUDE_SHOT_CHANGE = (
    os.environ.get("VIDEO_INTELLIGENCE_INCLUDE_SHOT_CHANGE", "true").lower() == "true"
)

# ---------------------------------------------------------------------------
# Phase-3 features (default ON — moves us from "labels only" to full coverage).
# Each one can still be disabled per-deployment via env to control GCP spend.
# ---------------------------------------------------------------------------
VIDEO_INTELLIGENCE_INCLUDE_OBJECT_TRACKING = (
    os.environ.get("VIDEO_INTELLIGENCE_INCLUDE_OBJECT_TRACKING", "true").lower() == "true"
)
VIDEO_INTELLIGENCE_INCLUDE_TEXT_DETECTION = (
    os.environ.get("VIDEO_INTELLIGENCE_INCLUDE_TEXT_DETECTION", "true").lower() == "true"
)
VIDEO_INTELLIGENCE_INCLUDE_PERSON_DETECTION = (
    os.environ.get("VIDEO_INTELLIGENCE_INCLUDE_PERSON_DETECTION", "true").lower() == "true"
)
VIDEO_INTELLIGENCE_INCLUDE_LOGO_RECOGNITION = (
    os.environ.get("VIDEO_INTELLIGENCE_INCLUDE_LOGO_RECOGNITION", "true").lower() == "true"
)
# Confidence floors prevent low-quality noise from dominating downstream copy.
VI_OBJECT_CONF_MIN = float(os.environ.get("VIDEO_INTELLIGENCE_OBJECT_CONF_MIN", "0.55") or 0.55)
VI_LOGO_CONF_MIN = float(os.environ.get("VIDEO_INTELLIGENCE_LOGO_CONF_MIN", "0.50") or 0.50)
VI_TEXT_CONF_MIN = float(os.environ.get("VIDEO_INTELLIGENCE_TEXT_CONF_MIN", "0.55") or 0.55)
VI_PERSON_CONF_MIN = float(os.environ.get("VIDEO_INTELLIGENCE_PERSON_CONF_MIN", "0.55") or 0.55)

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gvi")


def _ctx_duration_sec(ctx: JobContext) -> Optional[float]:
    vi = getattr(ctx, "video_info", None)
    if not isinstance(vi, dict):
        return None
    d = vi.get("duration")
    if d is None:
        return None
    try:
        return float(d)
    except (TypeError, ValueError):
        return None


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


def _normalized_to_pct(box: Any) -> Optional[Dict[str, float]]:
    """Convert a NormalizedBoundingBox to a {left,top,right,bottom} 0..1 dict."""
    if box is None:
        return None
    try:
        return {
            "left": round(float(getattr(box, "left", 0.0) or 0.0), 4),
            "top": round(float(getattr(box, "top", 0.0) or 0.0), 4),
            "right": round(float(getattr(box, "right", 0.0) or 0.0), 4),
            "bottom": round(float(getattr(box, "bottom", 0.0) or 0.0), 4),
        }
    except (TypeError, ValueError, AttributeError):
        return None


def _parse_annotation_result(result: Any) -> Dict[str, Any]:
    """Parse AnnotateVideoResponse into a compact JSON-friendly dict.

    Output keys:
      - segment_labels / shot_labels / shots / top_labels / summary_text (legacy)
      - object_tracks         (OBJECT_TRACKING)
      - on_screen_text        (TEXT_DETECTION across full video, not just sampled frames)
      - person_segments       (PERSON_DETECTION timeline + pose hints when available)
      - logos                 (LOGO_RECOGNITION brand callouts: Tesla, In-N-Out, etc.)
    """
    out: Dict[str, Any] = {
        "segment_labels": [],
        "shot_labels": [],
        "shots": [],
        "object_tracks": [],
        "on_screen_text": [],
        "person_segments": [],
        "logos": [],
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

            # ── OBJECT TRACKING ─────────────────────────────────────────────
            for ot in getattr(ar, "object_annotations", None) or []:
                ent = getattr(ot, "entity", None)
                desc = (getattr(ent, "description", None) or "").strip()
                conf = float(getattr(ot, "confidence", None) or 0.0)
                if conf < VI_OBJECT_CONF_MIN:
                    continue
                seg = getattr(ot, "segment", None)
                start_s = _offset_to_seconds(getattr(seg, "start_time_offset", None)) if seg else 0.0
                end_s = _offset_to_seconds(getattr(seg, "end_time_offset", None)) if seg else 0.0
                # Sample first/middle/last frame boxes for thumbnail keyframe selection.
                frames = getattr(ot, "frames", None) or []
                if frames:
                    sample_idx = sorted({0, len(frames) // 2, max(0, len(frames) - 1)})
                    frame_samples = []
                    for i in sample_idx:
                        try:
                            f = frames[i]
                            ts = _offset_to_seconds(getattr(f, "time_offset", None))
                            box = _normalized_to_pct(getattr(f, "normalized_bounding_box", None))
                            frame_samples.append({"t_s": round(ts, 3), "box": box})
                        except (IndexError, AttributeError):
                            continue
                else:
                    frame_samples = []
                out["object_tracks"].append({
                    "description": desc,
                    "confidence": round(conf, 4),
                    "start_s": round(start_s, 3),
                    "end_s": round(end_s, 3),
                    "frames": frame_samples,
                })

            # ── TEXT DETECTION (full-clip OCR) ──────────────────────────────
            for td in getattr(ar, "text_annotations", None) or []:
                text_val = (getattr(td, "text", None) or "").strip()
                if not text_val:
                    continue
                segments = getattr(td, "segments", None) or []
                best_seg = None
                for s in segments:
                    conf = float(getattr(s, "confidence", None) or 0.0)
                    if best_seg is None or conf > best_seg["conf"]:
                        seg_obj = getattr(s, "segment", None)
                        st = getattr(seg_obj, "start_time_offset", None) if seg_obj else None
                        en = getattr(seg_obj, "end_time_offset", None) if seg_obj else None
                        best_seg = {
                            "conf": conf,
                            "start_s": round(_offset_to_seconds(st), 3),
                            "end_s": round(_offset_to_seconds(en), 3),
                        }
                if best_seg and best_seg["conf"] < VI_TEXT_CONF_MIN:
                    continue
                out["on_screen_text"].append({
                    "text": text_val,
                    "confidence": round((best_seg or {"conf": 0.0})["conf"], 4),
                    "start_s": (best_seg or {}).get("start_s", 0.0),
                    "end_s": (best_seg or {}).get("end_s", 0.0),
                })

            # ── PERSON DETECTION (timeline + pose hints) ────────────────────
            for pd in getattr(ar, "person_detection_annotations", None) or []:
                tracks = getattr(pd, "tracks", None) or []
                for tr in tracks:
                    seg = getattr(tr, "segment", None)
                    start_s = _offset_to_seconds(getattr(seg, "start_time_offset", None)) if seg else 0.0
                    end_s = _offset_to_seconds(getattr(seg, "end_time_offset", None)) if seg else 0.0
                    conf = float(getattr(tr, "confidence", None) or 0.0)
                    if conf and conf < VI_PERSON_CONF_MIN:
                        continue
                    # Collect attribute / landmark hints (pose, clothing) when present.
                    attributes: List[str] = []
                    timestamped_objects = getattr(tr, "timestamped_objects", None) or []
                    for tso in timestamped_objects[:8]:
                        for attr in getattr(tso, "attributes", None) or []:
                            name = (getattr(attr, "name", None) or "").strip()
                            val = (getattr(attr, "value", None) or "").strip()
                            if name and val and len(attributes) < 6:
                                attributes.append(f"{name}={val}")
                    out["person_segments"].append({
                        "confidence": round(conf, 4),
                        "start_s": round(start_s, 3),
                        "end_s": round(end_s, 3),
                        "attributes": attributes,
                    })

            # ── LOGO RECOGNITION (brand callouts) ───────────────────────────
            for lr in getattr(ar, "logo_recognition_annotations", None) or []:
                ent = getattr(lr, "entity", None)
                desc = (getattr(ent, "description", None) or "").strip()
                if not desc:
                    continue
                tracks = getattr(lr, "tracks", None) or []
                best_conf = 0.0
                start_s = 0.0
                end_s = 0.0
                for tr in tracks:
                    conf = float(getattr(tr, "confidence", None) or 0.0)
                    if conf >= best_conf:
                        best_conf = conf
                        seg = getattr(tr, "segment", None)
                        if seg is not None:
                            start_s = _offset_to_seconds(getattr(seg, "start_time_offset", None))
                            end_s = _offset_to_seconds(getattr(seg, "end_time_offset", None))
                if best_conf < VI_LOGO_CONF_MIN:
                    continue
                out["logos"].append({
                    "description": desc,
                    "confidence": round(best_conf, 4),
                    "start_s": round(start_s, 3),
                    "end_s": round(end_s, 3),
                })
    except (TypeError, ValueError, AttributeError) as e:
        logger.warning("[video_intelligence] parse failed: %s", e)
        out["parse_error"] = str(e)

    # ── Top-N rollups for downstream consumers ──────────────────────────────
    best: Dict[str, float] = {}
    for row in out["segment_labels"]:
        d = row.get("description") or ""
        c = float(row.get("confidence") or 0.0)
        if d and c >= best.get(d, 0.0):
            best[d] = c
    top = sorted(best.items(), key=lambda x: -x[1])[:24]
    out["top_labels"] = [f"{d} ({c:.2f})" for d, c in top]

    out["object_tracks"].sort(key=lambda x: -float(x.get("confidence") or 0))
    out["object_tracks"] = out["object_tracks"][:24]
    out["on_screen_text"].sort(key=lambda x: -float(x.get("confidence") or 0))
    out["on_screen_text"] = out["on_screen_text"][:32]
    out["person_segments"].sort(key=lambda x: -float(x.get("confidence") or 0))
    out["person_segments"] = out["person_segments"][:16]
    out["logos"].sort(key=lambda x: -float(x.get("confidence") or 0))
    out["logos"] = out["logos"][:16]

    summary_bits = []
    if top:
        summary_bits.append("labels: " + ", ".join(d for d, _ in top[:10]))
    if out["object_tracks"]:
        summary_bits.append(
            "objects: "
            + ", ".join(
                f"{o['description']} ({o['start_s']:.0f}-{o['end_s']:.0f}s)"
                for o in out["object_tracks"][:5]
            )
        )
    if out["logos"]:
        summary_bits.append("logos: " + ", ".join(l["description"] for l in out["logos"][:5]))
    if out["on_screen_text"]:
        summary_bits.append(
            "OCR: "
            + ", ".join(
                (t["text"][:40] + ("…" if len(t["text"]) > 40 else ""))
                for t in out["on_screen_text"][:4]
            )
        )
    if out["person_segments"]:
        summary_bits.append(f"people: {len(out['person_segments'])} segment(s)")
    out["summary_text"] = "Video Intelligence — " + " | ".join(summary_bits) if summary_bits else ""
    return out


def _build_vi_features(vi_module: Any) -> List[Any]:
    """Compose the features list for an annotate_video request.

    Each feature is gated by an env flag so deployments can opt out of the
    pricier ones (object/text/person/logo). Defaults to ALL ON because the
    user explicitly requested full coverage.
    """
    features: List[Any] = [vi_module.Feature.LABEL_DETECTION]
    if VIDEO_INTELLIGENCE_INCLUDE_SHOT_CHANGE:
        try:
            features.append(vi_module.Feature.SHOT_CHANGE_DETECTION)
        except AttributeError as e:
            logger.debug("video_intelligence: SHOT_CHANGE_DETECTION unavailable: %s", e)
    if VIDEO_INTELLIGENCE_INCLUDE_OBJECT_TRACKING:
        try:
            features.append(vi_module.Feature.OBJECT_TRACKING)
        except AttributeError as e:
            logger.debug("video_intelligence: OBJECT_TRACKING unavailable: %s", e)
    if VIDEO_INTELLIGENCE_INCLUDE_TEXT_DETECTION:
        try:
            features.append(vi_module.Feature.TEXT_DETECTION)
        except AttributeError as e:
            logger.debug("video_intelligence: TEXT_DETECTION unavailable: %s", e)
    if VIDEO_INTELLIGENCE_INCLUDE_PERSON_DETECTION:
        try:
            features.append(vi_module.Feature.PERSON_DETECTION)
        except AttributeError as e:
            logger.debug("video_intelligence: PERSON_DETECTION unavailable: %s", e)
    if VIDEO_INTELLIGENCE_INCLUDE_LOGO_RECOGNITION:
        try:
            features.append(vi_module.Feature.LOGO_RECOGNITION)
        except AttributeError as e:
            logger.debug("video_intelligence: LOGO_RECOGNITION unavailable: %s", e)
    return features


def _analyze_sync_inline(video_bytes: bytes, creds: Any = None) -> Dict[str, Any]:
    from google.cloud import videointelligence_v1 as vi

    client = (
        vi.VideoIntelligenceServiceClient(credentials=creds)
        if creds is not None
        else vi.VideoIntelligenceServiceClient()
    )
    features = _build_vi_features(vi)
    request = vi.AnnotateVideoRequest(
        input_content=video_bytes,
        features=features,
    )
    operation = client.annotate_video(request=request)
    result = operation.result(timeout=VIDEO_INTELLIGENCE_TIMEOUT_SEC)
    return _parse_annotation_result(result)


async def _build_vi_inline_proxy(
    video_path: Path,
    *,
    max_bytes: int,
    temp_dir: Optional[Path],
) -> Optional[Path]:
    """FFmpeg a smaller proxy when the source exceeds inline VI byte limits."""
    from stages.ffmpeg_env import resolve_ffmpeg_executable

    out_dir = temp_dir or video_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"_vi_proxy_{video_path.stem}.mp4"
    ffmpeg = resolve_ffmpeg_executable("ffmpeg") or "ffmpeg"

    # Progressively lower resolution / raise CRF until under the inline cap.
    presets = (
        ("1280:-2", 32, "96k"),
        ("854:-2", 34, "64k"),
        ("640:-2", 36, "48k"),
        ("480:-2", 38, "32k"),
    )
    for scale, crf, abit in presets:
        if dst.exists():
            try:
                dst.unlink()
            except OSError:
                pass
        cmd = [
            ffmpeg,
            "-y",
            "-i",
            str(video_path),
            "-vf",
            f"scale={scale}",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            str(crf),
            "-c:a",
            "aac",
            "-b:a",
            abit,
            "-ac",
            "1",
            "-movflags",
            "+faststart",
            str(dst),
        ]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
        except (OSError, FileNotFoundError) as e:
            logger.warning("[video_intelligence] ffmpeg proxy unavailable: %s", e)
            return None
        if proc.returncode != 0:
            logger.debug(
                "[video_intelligence] ffmpeg proxy failed scale=%s crf=%s: %s",
                scale,
                crf,
                (stderr or b"").decode("utf-8", errors="replace")[:400],
            )
            continue
        if dst.exists() and dst.stat().st_size <= max_bytes:
            return dst
    if dst.exists() and dst.stat().st_size < video_path.stat().st_size:
        logger.warning(
            "[video_intelligence] proxy still %.1f MB (> %.1f MB cap); using best-effort proxy",
            dst.stat().st_size / (1024 * 1024),
            max_bytes / (1024 * 1024),
        )
        return dst
    return None


def _analyze_sync_gcs(uri: str, creds: Any = None) -> Dict[str, Any]:
    from google.cloud import videointelligence_v1 as vi

    client = (
        vi.VideoIntelligenceServiceClient(credentials=creds)
        if creds is not None
        else vi.VideoIntelligenceServiceClient()
    )
    features = _build_vi_features(vi)
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

    if not VIDEO_INTELLIGENCE_STAGE_ENABLED:
        raise SkipStage("Video Intelligence stage disabled via env")

    tier_allowed = getattr(ctx.entitlements, "allowed_ai_services", None) if ctx.entitlements else None
    tier_allowed_set = set(tier_allowed) if tier_allowed is not None else None
    if not user_pref_ai_service_enabled(
        ctx.user_settings or {},
        "video_intelligence",
        default=True,
        allowed_services=tier_allowed_set,
    ):
        raise SkipStage("Video Intelligence disabled in upload preferences (aiServiceVideoAnalyzer)")

    dur = _ctx_duration_sec(ctx)
    if VIDEO_INTELLIGENCE_MAX_DURATION_SEC > 0:
        if dur is not None and dur > VIDEO_INTELLIGENCE_MAX_DURATION_SEC:
            raise SkipStage(
                f"Video Intelligence skipped (duration {dur:.0f}s > "
                f"{VIDEO_INTELLIGENCE_MAX_DURATION_SEC:.0f}s VIDEO_INTELLIGENCE_MAX_DURATION_SEC)"
            )
    if dur is not None and dur > _GOOGLE_VI_ANNOTATE_MAX_DURATION_SEC:
        logger.warning(
            "[video_intelligence] duration=%.0fs exceeds Google annotate ~%ds limit; "
            "request may fail. Use VIDEO_INTELLIGENCE_INPUT_URI=gs://bucket/object for large/long assets.",
            dur,
            _GOOGLE_VI_ANNOTATE_MAX_DURATION_SEC,
        )

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
            _publish_recognition_to_ctx(ctx, data)
            logger.info(
                "[video_intelligence] GCS labels=%d shots=%d objects=%d text=%d persons=%d logos=%d",
                len(data.get("segment_labels") or []),
                len(data.get("shots") or []),
                len(data.get("object_tracks") or []),
                len(data.get("on_screen_text") or []),
                len(data.get("person_segments") or []),
                len(data.get("logos") or []),
            )
            return ctx
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("[video_intelligence] GCS analyze failed: %s", e)
            append_provider_error(
                ctx,
                provider="google_video_intelligence",
                stage="video_intelligence_stage",
                operation="annotate_video_gcs",
                message=str(e),
                exception_type=type(e).__name__,
                result_tier="degraded",
            )
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
    vi_input_path = video_path
    if size > VIDEO_INTELLIGENCE_MAX_BYTES:
        proxy = await _build_vi_inline_proxy(
            video_path,
            max_bytes=VIDEO_INTELLIGENCE_MAX_BYTES,
            temp_dir=getattr(ctx, "temp_dir", None),
        )
        if proxy and proxy.exists():
            vi_input_path = proxy
            logger.info(
                "[video_intelligence] inline proxy %s (%.1f MB -> %.1f MB)",
                proxy.name,
                size / (1024 * 1024),
                proxy.stat().st_size / (1024 * 1024),
            )
        else:
            raise SkipStage(
                f"Video too large for inline Video Intelligence ({size} bytes > {VIDEO_INTELLIGENCE_MAX_BYTES}); "
                "set VIDEO_INTELLIGENCE_INPUT_URI to gs://... or raise VIDEO_INTELLIGENCE_MAX_BYTES"
            )

    try:
        video_bytes = vi_input_path.read_bytes()
    except (OSError, PermissionError) as e:
        raise SkipStage(f"Cannot read video file: {e}") from e

    if len(video_bytes) > VIDEO_INTELLIGENCE_MAX_BYTES:
        raise SkipStage(
            f"Video too large for inline Video Intelligence ({len(video_bytes)} bytes > "
            f"{VIDEO_INTELLIGENCE_MAX_BYTES}); set VIDEO_INTELLIGENCE_INPUT_URI to gs://... "
            "or raise VIDEO_INTELLIGENCE_MAX_BYTES"
        )

    try:
        data = await loop.run_in_executor(
            _executor,
            lambda: _analyze_sync_inline(video_bytes, creds_obj),
        )
        ctx.video_intelligence_context = data
        _publish_recognition_to_ctx(ctx, data)
        logger.info(
            "[video_intelligence] inline labels=%d shots=%d objects=%d text=%d persons=%d logos=%d",
            len(data.get("segment_labels") or []),
            len(data.get("shots") or []),
            len(data.get("object_tracks") or []),
            len(data.get("on_screen_text") or []),
            len(data.get("person_segments") or []),
            len(data.get("logos") or []),
        )
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.warning("[video_intelligence] Non-fatal error: %s", e)
        append_provider_error(
            ctx,
            provider="google_video_intelligence",
            stage="video_intelligence_stage",
            operation="annotate_video_inline",
            message=str(e),
            exception_type=type(e).__name__,
            result_tier="degraded",
        )
        ctx.video_intelligence_context = {"error": str(e)}

    return ctx


def _publish_recognition_to_ctx(ctx: JobContext, data: Dict[str, Any]) -> None:
    """Mirror VI structured tracks onto ``ctx.video_intelligence`` for the
    recognition aggregator + hydration enforcer.

    The legacy ``video_intelligence_context`` keeps the full payload (we
    don't want to break callers); ``video_intelligence`` is a smaller view
    optimized for the must_use shortlist and hashtag generation.
    """
    if not isinstance(data, dict):
        return
    flat = {
        "top_labels": list(data.get("top_labels") or []),
        "segment_labels": list(data.get("segment_labels") or []),
        "shot_labels": list(data.get("shot_labels") or []),
        "object_tracks": list(data.get("object_tracks") or []),
        "on_screen_text": list(data.get("on_screen_text") or []),
        "person_segments": list(data.get("person_segments") or []),
        "logos": list(data.get("logos") or []),
        "shots": list(data.get("shots") or []),
        "summary_text": data.get("summary_text") or "",
    }
    setattr(ctx, "video_intelligence", flat)
