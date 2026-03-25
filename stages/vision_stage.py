"""
UploadM8 Vision Stage
=====================
Run Google Cloud Vision API multi-feature analysis on the best video frame.
Extracts:
  - Face detection (bounding boxes, emotion likelihoods)
  - OCR text (scoreboards, signs, labels, product names)
  - Scene labels (people, activities, objects)

Outputs stored in ctx.vision_context, used by:
  - thumbnail_stage: face-priority frame crop, face bounding for overlay
  - caption_stage: OCR text injected into prompt (scoreboards → sports recap)

Cost: $1.50/1K images per feature (3 features = $4.50/1K = $0.0045/upload).
Free: 1,000 units/month per feature.
"""

import asyncio
import base64
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional

from .context import JobContext
from .errors import SkipStage

logger = logging.getLogger("uploadm8-worker")

VISION_STAGE_ENABLED = os.environ.get("VISION_STAGE_ENABLED", "true").lower() == "true"
GCP_CREDENTIALS      = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")

_gcv_client      = None
_vision_module   = None
_gcv_executor    = ThreadPoolExecutor(max_workers=2, thread_name_prefix="gcv")

LIKELIHOOD_MAP = {
    0: "UNKNOWN",
    1: "VERY_UNLIKELY",
    2: "UNLIKELY",
    3: "POSSIBLE",
    4: "LIKELY",
    5: "VERY_LIKELY",
}

POSITIVE_EMOTIONS = {"LIKELY", "VERY_LIKELY"}


def _get_gcv_client():
    """Lazy-load GCV client (avoids import errors when GCP not configured)."""
    global _gcv_client, _vision_module
    if _gcv_client is not None:
        return _gcv_client

    try:
        from google.cloud import vision as v
        _vision_module = v
        _gcv_client    = v.ImageAnnotatorClient()
        logger.info("[vision] Google Cloud Vision client initialized")
        return _gcv_client
    except Exception as e:
        logger.warning(f"[vision] GCV client init failed: {e}")
        return None


def _analyze_sync(image_bytes: bytes) -> Dict[str, Any]:
    """
    Run GCV multi-feature analysis synchronously.
    Must be called from run_in_executor.
    """
    v      = _vision_module
    client = _get_gcv_client()

    if not client or not v:
        return {}

    image    = v.Image(content=image_bytes)
    features = [
        v.Feature(type_=v.Feature.Type.FACE_DETECTION,  max_results=10),
        v.Feature(type_=v.Feature.Type.TEXT_DETECTION),
        v.Feature(type_=v.Feature.Type.LABEL_DETECTION, max_results=20),
    ]
    request  = v.AnnotateImageRequest(image=image, features=features)
    response = client.annotate_image(request=request)

    # ── Face annotations ──────────────────────────────────────────────────────
    faces = []
    for face in response.face_annotations:
        vertices = [(v_pt.x, v_pt.y) for v_pt in face.bounding_poly.vertices]
        joy_score    = LIKELIHOOD_MAP.get(face.joy_likelihood, "UNKNOWN")
        surprise     = LIKELIHOOD_MAP.get(face.surprise_likelihood, "UNKNOWN")
        sorrow       = LIKELIHOOD_MAP.get(face.sorrow_likelihood, "UNKNOWN")
        anger        = LIKELIHOOD_MAP.get(face.anger_likelihood, "UNKNOWN")

        # Score face expressiveness (higher = better thumbnail candidate)
        expressive_score = 0
        if joy_score in POSITIVE_EMOTIONS:      expressive_score += 3
        if surprise in POSITIVE_EMOTIONS:       expressive_score += 3
        if sorrow in POSITIVE_EMOTIONS:         expressive_score += 1
        if anger in POSITIVE_EMOTIONS:          expressive_score += 1

        faces.append({
            "confidence":      face.detection_confidence,
            "bounding_poly":   vertices,
            "center_x":        sum(v[0] for v in vertices) // 4,
            "center_y":        sum(v[1] for v in vertices) // 4,
            "joy":             joy_score,
            "surprise":        surprise,
            "sorrow":          sorrow,
            "anger":           anger,
            "roll_angle":      face.roll_angle,
            "pan_angle":       face.pan_angle,
            "tilt_angle":      face.tilt_angle,
            "expressive_score": expressive_score,
        })

    # ── OCR text ──────────────────────────────────────────────────────────────
    ocr_text = ""
    if response.text_annotations:
        ocr_text = response.text_annotations[0].description.strip()

    # ── Scene labels ──────────────────────────────────────────────────────────
    labels = [
        {"description": lbl.description, "score": round(lbl.score, 3)}
        for lbl in response.label_annotations
        if lbl.score > 0.7
    ]

    return {
        "faces":         faces,
        "face_count":    len(faces),
        "has_faces":     len(faces) > 0,
        "best_face":     max(faces, key=lambda f: f["expressive_score"]) if faces else None,
        "expressive":    any(f["expressive_score"] >= 3 for f in faces),
        "ocr_text":      ocr_text,
        "labels":        labels,
        "label_names":   [lbl["description"] for lbl in labels],
    }


async def run_vision_stage(ctx: JobContext) -> JobContext:
    """
    Analyze the best available video frame with Google Cloud Vision.
    Stores results in ctx.vision_context.
    Non-fatal — pipeline continues on any error.
    """
    ctx.mark_stage("vision")

    if not VISION_STAGE_ENABLED:
        raise SkipStage("Vision stage disabled via env")

    if not GCP_CREDENTIALS:
        raise SkipStage("GOOGLE_APPLICATION_CREDENTIALS not set")

    # Find best frame to analyze
    frame_to_analyze = _find_best_frame(ctx)

    if not frame_to_analyze or not frame_to_analyze.exists():
        # Extract one frame when pipeline runs vision before thumbnail
        frame_to_analyze = await _extract_frame_for_vision(ctx)
    if not frame_to_analyze or not frame_to_analyze.exists():
        raise SkipStage("No frame available for vision analysis")

    try:
        image_bytes = frame_to_analyze.read_bytes()
        loop        = asyncio.get_event_loop()

        result = await loop.run_in_executor(
            _gcv_executor,
            partial(_analyze_sync, image_bytes),
        )

        ctx.vision_context = result

        logger.info(
            f"[vision] ✓ faces={result.get('face_count', 0)} "
            f"expressive={result.get('expressive', False)} "
            f"ocr_chars={len(result.get('ocr_text', ''))} "
            f"labels={result.get('label_names', [])[:3]}"
        )

        return ctx

    except SkipStage:
        raise
    except Exception as e:
        logger.warning(f"[vision] Non-fatal error: {e}")
        ctx.vision_context = {}
        return ctx


def _find_best_frame(ctx: JobContext) -> Optional[Path]:
    """Find the best frame already extracted in temp_dir."""
    if not ctx.temp_dir:
        return None

    temp_dir = Path(ctx.temp_dir)

    # Prefer thumbnail frame (highest quality extraction)
    for pattern in ["thumbnail_final.jpg", "cand_*.jpg", "frame_*.jpg", "cap_frame_*.jpg", "thumb_*.jpg"]:
        matches = sorted(temp_dir.glob(pattern))
        if matches:
            return matches[0]

    return None


async def _extract_frame_for_vision(ctx: JobContext) -> Optional[Path]:
    """Extract one frame from video when none exist (for pipeline order: vision before thumbnail)."""
    video_path = None
    for c in (ctx.processed_video_path, ctx.local_video_path):
        if c and Path(c).exists():
            video_path = Path(c)
            break
    if not video_path or not ctx.temp_dir:
        return None

    out_path = Path(ctx.temp_dir) / "cand_000.jpg"
    FFMPEG_PATH = os.environ.get("FFMPEG_PATH", "ffmpeg")
    FFPROBE_PATH = os.environ.get("FFPROBE_PATH", "ffprobe")
    try:
        # Use ffprobe to get duration, extract at 30%
        import json
        proc = await asyncio.create_subprocess_exec(
            FFPROBE_PATH,
            "-v", "quiet", "-show_entries", "format=duration", "-of", "json",
            str(video_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        duration = 1.0
        if proc.returncode == 0 and stdout:
            try:
                data = json.loads(stdout.decode())
                d = data.get("format", {}).get("duration", 1)
                duration = float(d) if d else 1.0
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
        offset = max(0.5, duration * 0.30)

        cmd = [FFMPEG_PATH, "-y", "-ss", f"{offset:.2f}", "-i", str(video_path),
               "-vframes", "1", "-q:v", "2", "-vf", "scale=1280:-1", str(out_path)]
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        await proc.communicate()
        return out_path if out_path.exists() and out_path.stat().st_size > 1000 else None
    except Exception as e:
        logger.warning(f"[vision] Frame extraction failed: {e}")
        return None


def get_vision_ocr_for_caption(vision_ctx: Dict) -> str:
    """
    Extract OCR text relevant for caption generation.
    Filters out noise, returns clean string.
    """
    if not vision_ctx:
        return ""

    text = vision_ctx.get("ocr_text", "")
    if not text:
        return ""

    # Basic cleaning — remove single chars, very short tokens
    lines = [line.strip() for line in text.split("\n") if len(line.strip()) > 2]
    clean = " | ".join(lines[:5])  # Max 5 lines

    return clean[:300]  # Cap at 300 chars for prompt injection


def get_face_crop_region(vision_ctx: Dict, frame_width: int, frame_height: int) -> Optional[Dict]:
    """
    Return a crop region centered on the most expressive face.
    Used by thumbnail_stage to focus composition.
    Returns dict: {x, y, w, h} normalized 0–1 or None.
    """
    if not vision_ctx or not vision_ctx.get("best_face"):
        return None

    face = vision_ctx["best_face"]
    poly = face.get("bounding_poly", [])

    if len(poly) < 4:
        return None

    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]

    face_x = min(xs)
    face_y = min(ys)
    face_w = max(xs) - face_x
    face_h = max(ys) - face_y

    # Expand crop region by 60% to include shoulders/context
    pad_x = face_w * 0.6
    pad_y = face_h * 0.6

    crop_x = max(0, face_x - pad_x)
    crop_y = max(0, face_y - pad_y)
    crop_w = min(frame_width,  face_w + pad_x * 2)
    crop_h = min(frame_height, face_h + pad_y * 2)

    return {
        "x": int(crop_x), "y": int(crop_y),
        "w": int(crop_w), "h": int(crop_h),
    }
