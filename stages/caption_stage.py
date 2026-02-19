"""
UploadM8 Caption Stage
=======================
Generate AI-powered titles, captions, and hashtags using OpenAI.

GROUNDING RULES (enforced):
  - All generated content MUST be based on the actual FFmpeg thumbnail image
    and/or real telemetry / Trill data parsed from the .map file.
  - If neither a thumbnail nor telemetry is available, caption generation is
    skipped rather than producing fabricated content.
  - GPT-4o (vision) is used when a thumbnail is present so the model can
    describe what it actually sees in the frame.
  - Trill score, speeds, distance, and hashtags from the .map file are
    injected verbatim into the prompt so the model has real evidence.

Exports: run_caption_stage(ctx)
"""

import base64
import json
import logging
import os
from pathlib import Path
from typing import Optional, List

from .errors import SkipStage, CaptionError, ErrorCode
from .context import JobContext, TrillScore, TelemetryData

logger = logging.getLogger("uploadm8-worker.caption")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
# Use vision model when image is available; fall back to mini for text-only
OPENAI_VISION_MODEL = os.environ.get("OPENAI_VISION_MODEL", "gpt-4o")
OPENAI_TEXT_MODEL = os.environ.get("OPENAI_TEXT_MODEL", "gpt-4o-mini")


# ---------------------------------------------------------------------------
# OpenAI helpers
# ---------------------------------------------------------------------------

async def _call_openai(
    prompt: str,
    system: str = "",
    max_tokens: int = 500,
    image_b64: Optional[str] = None,
) -> Optional[str]:
    """
    Make an OpenAI API call.

    - If image_b64 is provided, uses the vision model with the image embedded.
    - Returns response text or None on failure.
    """
    if not OPENAI_API_KEY:
        return None

    import httpx

    model = OPENAI_VISION_MODEL if image_b64 else OPENAI_TEXT_MODEL

    try:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})

        if image_b64:
            # Vision message: interleave image + text
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}",
                            "detail": "low",   # "low" = 1 tile = ~85 tokens, sufficient for scene reading
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            })
        else:
            messages.append({"role": "user", "content": prompt})

        async with httpx.AsyncClient(timeout=45) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": 0.7,
                },
            )

            if resp.status_code == 429:
                logger.warning("OpenAI rate limited")
                return None

            if resp.status_code != 200:
                logger.warning(f"OpenAI error {resp.status_code}: {resp.text[:200]}")
                return None

            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()

    except Exception as e:
        logger.warning(f"OpenAI call failed: {e}")
        return None


def _load_thumbnail_b64(ctx: JobContext) -> Optional[str]:
    """
    Load the thumbnail image as a base64 string if available.
    Returns None if no thumbnail has been generated yet.
    """
    path: Optional[Path] = ctx.thumbnail_path
    if not path:
        return None
    try:
        p = Path(path)
        if p.exists() and p.stat().st_size > 0:
            return base64.b64encode(p.read_bytes()).decode("utf-8")
    except Exception as e:
        logger.warning(f"Could not load thumbnail for vision: {e}")
    return None


def _build_telemetry_context(ctx: JobContext) -> str:
    """
    Build a human-readable telemetry summary from .map-derived data.
    Returns an empty string if no telemetry is available.
    """
    trill: Optional[TrillScore] = ctx.trill_score or ctx.trill
    telem: Optional[TelemetryData] = ctx.telemetry_data or ctx.telemetry

    lines = []

    if telem:
        if telem.max_speed_mph:
            lines.append(f"Max speed: {telem.max_speed_mph:.1f} mph")
        if telem.avg_speed_mph:
            lines.append(f"Avg speed: {telem.avg_speed_mph:.1f} mph")
        if telem.total_distance_miles:
            lines.append(f"Distance: {telem.total_distance_miles:.2f} miles")
        if telem.duration_seconds:
            mins = int(telem.duration_seconds // 60)
            secs = int(telem.duration_seconds % 60)
            lines.append(f"Clip duration: {mins}m {secs}s")
        if telem.max_altitude_ft:
            lines.append(f"Max altitude: {telem.max_altitude_ft:.0f} ft")
        if telem.speeding_seconds:
            lines.append(f"Time above speed limit: {telem.speeding_seconds:.0f}s")
        if telem.euphoria_seconds:
            lines.append(f"Euphoria/high-G seconds: {telem.euphoria_seconds:.0f}s")

    if trill:
        lines.append(f"Trill score: {trill.score}/100 ({trill.bucket})")
        if trill.title_modifier:
            lines.append(f"Speed character: {trill.title_modifier}")
        if trill.hashtags:
            lines.append(f"Trill hashtags: {', '.join(trill.hashtags)}")
        if trill.excessive_speed:
            lines.append("Note: excessive speed detected in clip")

    return "\n".join(lines)


def _build_grounding_evidence(ctx: JobContext, has_image: bool) -> str:
    """
    Assemble all factual evidence available for this clip.
    This is injected into every prompt so the model cannot hallucinate.
    """
    parts = []

    if has_image:
        parts.append("A thumbnail frame extracted from the video is attached.")

    telem_ctx = _build_telemetry_context(ctx)
    if telem_ctx:
        parts.append("Telemetry data from the .map file:\n" + telem_ctx)

    if ctx.filename:
        parts.append(f"Original filename: {ctx.filename}")

    if ctx.title:
        parts.append(f"User-supplied title: {ctx.title}")

    if ctx.caption:
        parts.append(f"User notes: {ctx.caption}")

    if ctx.platforms:
        parts.append(f"Target platforms: {', '.join(ctx.platforms)}")

    return "\n\n".join(parts) if parts else ""


# ---------------------------------------------------------------------------
# Individual generators
# ---------------------------------------------------------------------------

async def generate_title(ctx: JobContext, image_b64: Optional[str], evidence: str) -> Optional[str]:
    """Generate a grounded title. Returns None if generation fails or evidence is empty."""
    if not evidence:
        return None

    system = (
        "You are a social media content expert specialising in driving and dashcam videos. "
        "Generate ONLY the title text — no quotes, no explanation, under 100 characters. "
        "Base the title strictly on the visual content and/or the telemetry data provided. "
        "Do not invent events or sensationalise beyond what the evidence shows."
    )

    prompt = (
        "Generate a short, engaging title (max 100 characters) for this dashcam/driving video.\n\n"
        f"Evidence:\n{evidence}"
    )

    return await _call_openai(prompt, system, max_tokens=60, image_b64=image_b64)


async def generate_caption(ctx: JobContext, image_b64: Optional[str], evidence: str) -> Optional[str]:
    """Generate a grounded caption. Returns None if evidence is empty."""
    if not evidence:
        return None

    system = (
        "You are a social media content expert specialising in driving and dashcam videos. "
        "Write ONLY the caption text — 2-3 sentences, no quotes, no preamble. "
        "Describe what is actually shown in the video frame and/or what the telemetry reveals. "
        "Do not fabricate road conditions, events, or emotions not supported by the evidence."
    )

    prompt = (
        "Write a short, engaging social media caption for this dashcam/driving video.\n\n"
        f"Evidence:\n{evidence}"
    )

    return await _call_openai(prompt, system, max_tokens=200, image_b64=image_b64)


async def generate_hashtags(ctx: JobContext, image_b64: Optional[str], evidence: str) -> List[str]:
    """Generate grounded hashtags. Returns empty list if evidence is empty."""
    if not evidence:
        return []

    # Seed with any Trill hashtags from the .map file — these are always included
    trill = ctx.trill_score or ctx.trill
    trill_tags: List[str] = list(trill.hashtags) if trill and trill.hashtags else []

    system = (
        "You generate hashtags for social media dashcam/driving videos. "
        "Return ONLY a JSON array of strings, each starting with #. "
        "Example: [\"#dashcam\", \"#driving\", \"#fyp\"]. "
        "Base hashtags on the visual content and telemetry data provided. "
        "Do not include generic filler tags unrelated to the content."
    )

    prompt = (
        "Generate 5-10 relevant hashtags for this dashcam/driving video. "
        "Return them as a JSON array of strings.\n\n"
        f"Evidence:\n{evidence}"
    )

    result = await _call_openai(prompt, system, max_tokens=100, image_b64=image_b64)
    if not result:
        return trill_tags

    try:
        clean = result.strip().strip("`").strip()
        if clean.startswith("json"):
            clean = clean[4:].strip()
        ai_tags = json.loads(clean)
        if isinstance(ai_tags, list):
            ai_tags = [str(t) for t in ai_tags if isinstance(t, str)]
        else:
            ai_tags = []
    except (json.JSONDecodeError, ValueError):
        ai_tags = [w.strip() for w in result.split() if w.startswith("#")][:10]

    # Merge Trill tags first (they are evidence-based from the .map file)
    merged = list(dict.fromkeys(trill_tags + ai_tags))
    return merged


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------

async def run_caption_stage(ctx: JobContext) -> JobContext:
    """
    Execute AI caption generation stage.

    Requirements:
    - Must run AFTER thumbnail_stage so ctx.thumbnail_path is populated.
    - Must run AFTER telemetry_stage so Trill / .map data is in ctx.
    - Will skip if neither thumbnail nor telemetry is available (no evidence = no generation).
    - Uses GPT-4o vision when a thumbnail exists; GPT-4o-mini for text-only fallback.

    Produces:
    - ctx.ai_title
    - ctx.ai_caption
    - ctx.ai_hashtags
    """
    ctx.mark_stage("caption")

    # Tier check
    if ctx.entitlements and not ctx.entitlements.can_ai:
        raise SkipStage("AI captions not available for this tier")

    if not OPENAI_API_KEY:
        raise SkipStage("OpenAI API key not configured")

    # ------------------------------------------------------------------ #
    # Load evidence — thumbnail image + telemetry data                    #
    # ------------------------------------------------------------------ #
    image_b64 = _load_thumbnail_b64(ctx)
    evidence = _build_grounding_evidence(ctx, has_image=image_b64 is not None)

    if not evidence:
        # No thumbnail and no telemetry — refusing to generate unsupported content
        logger.info("Skipping caption generation: no thumbnail or telemetry evidence available")
        raise SkipStage("No visual or telemetry evidence available for grounded caption generation")

    has_image = image_b64 is not None
    has_telem = bool(_build_telemetry_context(ctx))
    logger.info(
        f"Generating AI captions for upload {ctx.upload_id} "
        f"[image={'yes' if has_image else 'no'}, telemetry={'yes' if has_telem else 'no'}]"
    )

    # ------------------------------------------------------------------ #
    # Generate title                                                       #
    # ------------------------------------------------------------------ #
    try:
        ai_title = await generate_title(ctx, image_b64, evidence)
        if ai_title:
            ctx.ai_title = ai_title
            logger.info(f"AI title: {ai_title[:80]}")
    except Exception as e:
        logger.warning(f"AI title generation failed (non-fatal): {e}")

    # ------------------------------------------------------------------ #
    # Generate caption                                                     #
    # ------------------------------------------------------------------ #
    try:
        ai_caption = await generate_caption(ctx, image_b64, evidence)
        if ai_caption:
            ctx.ai_caption = ai_caption
            logger.info(f"AI caption: {ai_caption[:80]}")
    except Exception as e:
        logger.warning(f"AI caption generation failed (non-fatal): {e}")

    # ------------------------------------------------------------------ #
    # Generate hashtags                                                    #
    # ------------------------------------------------------------------ #
    try:
        ai_hashtags = await generate_hashtags(ctx, image_b64, evidence)
        if ai_hashtags:
            ctx.ai_hashtags = ai_hashtags
            logger.info(f"AI hashtags ({len(ai_hashtags)}): {ai_hashtags}")
    except Exception as e:
        logger.warning(f"AI hashtag generation failed (non-fatal): {e}")

    # Track AIC cost (1 token per generation batch)
    ctx.aic_cost += 1

    return ctx
