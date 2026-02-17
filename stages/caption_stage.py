"""
UploadM8 Caption Stage
=======================
Generate AI-powered titles, captions, and hashtags using OpenAI.

Exports: run_caption_stage(ctx)
"""

import os
import json
import logging
from typing import Optional, List

from .errors import SkipStage, CaptionError, ErrorCode
from .context import JobContext

logger = logging.getLogger("uploadm8-worker")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


async def _call_openai(prompt: str, system: str = "", max_tokens: int = 500) -> Optional[str]:
    """Make an OpenAI API call. Returns response text or None on failure."""
    if not OPENAI_API_KEY:
        return None

    import httpx

    try:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": OPENAI_MODEL,
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


async def generate_title(ctx: JobContext) -> Optional[str]:
    """Generate an engaging title for the video."""
    # Build context for the AI
    parts = []
    if ctx.filename:
        parts.append(f"Filename: {ctx.filename}")
    if ctx.title:
        parts.append(f"Original title: {ctx.title}")
    if ctx.caption:
        parts.append(f"User caption: {ctx.caption}")
    if ctx.platforms:
        parts.append(f"Target platforms: {', '.join(ctx.platforms)}")

    # Add Trill data if available
    trill = getattr(ctx, "trill_score", None) or getattr(ctx, "trill", None)
    if trill and hasattr(trill, "score"):
        parts.append(f"Trill score: {trill.score} ({trill.bucket})")
        if hasattr(trill, "title_modifier") and trill.title_modifier:
            parts.append(f"Speed modifier: {trill.title_modifier}")

    if not parts:
        return None

    prompt = (
        f"Generate a short, engaging video title (max 100 characters) for social media. "
        f"Make it catchy and platform-appropriate.\n\n"
        f"Context:\n" + "\n".join(parts)
    )

    system = (
        "You are a social media content expert. Generate only the title text, "
        "no quotes, no explanation. Keep it under 100 characters."
    )

    return await _call_openai(prompt, system, max_tokens=60)


async def generate_caption(ctx: JobContext) -> Optional[str]:
    """Generate a caption/description for the video."""
    parts = []
    if ctx.title:
        parts.append(f"Title: {ctx.title}")
    if ctx.caption:
        parts.append(f"User notes: {ctx.caption}")
    if ctx.platforms:
        parts.append(f"Platforms: {', '.join(ctx.platforms)}")

    trill = getattr(ctx, "trill_score", None) or getattr(ctx, "trill", None)
    if trill and hasattr(trill, "score"):
        parts.append(f"Trill score: {trill.score}/100 ({trill.bucket})")

    if not parts:
        return None

    prompt = (
        f"Write a short, engaging social media caption (2-3 sentences max) for this video.\n\n"
        f"Context:\n" + "\n".join(parts)
    )

    system = (
        "You are a social media content expert. Write only the caption text, "
        "no quotes, no explanation. Keep it concise and engaging."
    )

    return await _call_openai(prompt, system, max_tokens=150)


async def generate_hashtags(ctx: JobContext) -> List[str]:
    """Generate relevant hashtags for the video."""
    parts = []
    if ctx.title:
        parts.append(f"Title: {ctx.title}")
    if ctx.caption:
        parts.append(f"Caption: {ctx.caption}")
    if ctx.platforms:
        parts.append(f"Platforms: {', '.join(ctx.platforms)}")

    trill = getattr(ctx, "trill_score", None) or getattr(ctx, "trill", None)
    if trill and hasattr(trill, "hashtags"):
        parts.append(f"Trill hashtags: {', '.join(trill.hashtags)}")

    if not parts:
        return []

    prompt = (
        f"Generate 5-10 relevant hashtags for this social media video. "
        f"Return them as a JSON array of strings.\n\n"
        f"Context:\n" + "\n".join(parts)
    )

    system = (
        "You generate hashtags. Return ONLY a JSON array of strings, "
        "each starting with #. Example: [\"#viral\", \"#fyp\", \"#trending\"]"
    )

    result = await _call_openai(prompt, system, max_tokens=100)
    if not result:
        return []

    # Parse JSON response
    try:
        # Strip markdown code fences if present
        clean = result.strip().strip("`").strip()
        if clean.startswith("json"):
            clean = clean[4:].strip()
        tags = json.loads(clean)
        if isinstance(tags, list):
            return [str(t) for t in tags if isinstance(t, str)]
    except (json.JSONDecodeError, ValueError):
        # Try to extract hashtags from plain text
        tags = [word.strip() for word in result.split() if word.startswith("#")]
        return tags[:10]

    return []


async def run_caption_stage(ctx: JobContext) -> JobContext:
    """
    Execute AI caption generation stage.

    Generates:
    - AI title (ctx.ai_title)
    - AI caption (ctx.ai_caption)
    - AI hashtags (ctx.ai_hashtags)

    Raises:
        SkipStage: If AI is not available or not enabled for tier.
    """
    ctx.mark_stage("caption")

    # Check if AI is enabled for this tier
    if ctx.entitlements and not ctx.entitlements.can_ai:
        raise SkipStage("AI captions not available for this tier")

    # Check if OpenAI is configured
    if not OPENAI_API_KEY:
        raise SkipStage("OpenAI API key not configured")

    logger.info(f"Generating AI captions for upload {ctx.upload_id}")

    # Generate title
    try:
        ai_title = await generate_title(ctx)
        if ai_title:
            ctx.ai_title = ai_title
            logger.info(f"AI title: {ai_title[:80]}")
    except Exception as e:
        logger.warning(f"AI title generation failed (non-fatal): {e}")

    # Generate caption
    try:
        ai_caption = await generate_caption(ctx)
        if ai_caption:
            ctx.ai_caption = ai_caption
            logger.info(f"AI caption: {ai_caption[:80]}")
    except Exception as e:
        logger.warning(f"AI caption generation failed (non-fatal): {e}")

    # Generate hashtags
    try:
        ai_hashtags = await generate_hashtags(ctx)
        if ai_hashtags:
            # Merge with Trill hashtags if present
            trill = getattr(ctx, "trill_score", None) or getattr(ctx, "trill", None)
            if trill and hasattr(trill, "hashtags") and trill.hashtags:
                ai_hashtags = list(dict.fromkeys(trill.hashtags + ai_hashtags))  # dedupe
            ctx.ai_hashtags = ai_hashtags
            logger.info(f"AI hashtags: {ai_hashtags}")
    except Exception as e:
        logger.warning(f"AI hashtag generation failed (non-fatal): {e}")

    # Track AIC token cost (rough estimate)
    ctx.aic_cost += 1  # 1 AIC token per generation batch

    return ctx
