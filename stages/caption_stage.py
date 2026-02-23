"""
UploadM8 Caption Stage
======================
Generates AI-powered titles, captions, and hashtags using OpenAI GPT-4o-mini.

Key behaviors:
- Injects location_name (from telemetry reverse geocoding) into the AI prompt
- Injects Trill score context when telemetry data exists
- Returns hashtags as a proper List[str] — never iterates a string
- Non-fatal: pipeline continues on AI failure
"""

import os
import json
import base64
import logging
import asyncio
from pathlib import Path
from typing import List, Optional, Dict, Any

import httpx

from .errors import SkipStage, StageError, ErrorCode
from .context import JobContext

logger = logging.getLogger("uploadm8-worker")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_CAPTION_MODEL", "gpt-4o-mini")

# Cost per 1K tokens (gpt-4o-mini)
COST_PER_1K_INPUT = 0.000150
COST_PER_1K_OUTPUT = 0.000600
COST_PER_IMAGE = 0.00765  # low detail


def get_max_hashtags(entitlements) -> int:
    if entitlements is None:
        return 5
    return getattr(entitlements, "max_hashtags", 5) or 5


async def extract_video_frames(
    video_path: Path,
    temp_dir: Path,
    num_frames: int = 4
) -> List[Path]:
    """Extract frames from video for AI analysis using ffmpeg."""
    frames = []
    try:
        duration_cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_streams", str(video_path)
        ]
        proc = await asyncio.create_subprocess_exec(
            *duration_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await proc.communicate()
        data = json.loads(stdout.decode())
        duration = float(
            next(
                (s["duration"] for s in data.get("streams", [])
                 if s.get("codec_type") == "video"),
                30.0
            )
        )

        interval = max(1.0, duration / (num_frames + 1))
        for i in range(1, num_frames + 1):
            ts = interval * i
            frame_path = temp_dir / f"frame_{i:02d}.jpg"
            cmd = [
                "ffmpeg", "-ss", str(ts),
                "-i", str(video_path),
                "-frames:v", "1",
                "-q:v", "5",
                "-y", str(frame_path)
            ]
            proc2 = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )
            await proc2.wait()
            if frame_path.exists() and frame_path.stat().st_size > 0:
                frames.append(frame_path)

    except Exception as e:
        logger.warning(f"Frame extraction failed: {e}")

    return frames


def build_ai_prompt(
    filename: str,
    existing_title: str,
    existing_caption: str,
    platform_hints: List[str],
    generate_title: bool,
    generate_caption: bool,
    generate_hashtags: bool,
    hashtag_count: int,
    location_name: Optional[str] = None,
    trill_score: Optional[int] = None,
    trill_bucket: Optional[str] = None,
    max_speed: Optional[float] = None,
) -> str:
    """
    Build the AI generation prompt with location and Trill context injected.
    """
    tasks = []
    if generate_title:
        tasks.append("1. An attention-grabbing TITLE (max 100 characters)")
    if generate_caption:
        tasks.append("2. An engaging CAPTION for social media (max 280 characters)")
    if generate_hashtags and hashtag_count > 0:
        tasks.append(
            f"3. Exactly {hashtag_count} relevant HASHTAGS "
            f"(return as JSON array of strings, WITHOUT the # symbol, "
            f"each hashtag as a complete word like 'dashcam' or 'gloryboy' — "
            f"NEVER return individual characters)"
        )

    platform_str = ", ".join(platform_hints) if platform_hints else "social media"

    # Build context block
    context_lines = [
        f"- Filename: {filename}",
        f"- Target platforms: {platform_str}",
    ]
    if existing_title:
        context_lines.append(f"- User-provided title: {existing_title}")
    if existing_caption:
        context_lines.append(f"- User-provided caption: {existing_caption}")

    # Inject location if available
    if location_name:
        context_lines.append(f"- FILMING LOCATION: {location_name}")
        context_lines.append(
            f"  → Incorporate the location '{location_name}' naturally into the title, "
            f"caption, and/or hashtags (e.g. city name as a hashtag, mention in caption)"
        )

    # Inject Trill/driving context if available
    if trill_score is not None:
        context_lines.append(f"- Trill driving score: {trill_score}/100 (bucket: {trill_bucket})")
        if max_speed:
            context_lines.append(f"- Max speed recorded: {max_speed:.0f} mph")
        context_lines.append(
            "- This is dashcam/driving footage. Make the content feel authentic, "
            "exciting, and road-trip/car culture oriented."
        )

    context_block = "\n".join(context_lines)
    tasks_block = "\n".join(tasks)

    prompt = f"""Analyze this video content and generate the following for {platform_str}:

{tasks_block}

Context:
{context_block}

Important rules:
- Be engaging and encourage interaction
- Use emojis sparingly but effectively  
- Make content feel authentic, NOT AI-generated
- For TikTok/Instagram: be trendy and casual
- For YouTube: be more descriptive
- HASHTAGS RULE: Return each hashtag as a COMPLETE WORD (e.g. "dashcam", "gloryboy", "roadtrip") 
  NOT individual letters. This is critical.
- If location was provided, use it: add city/state as a hashtag, reference it in caption

Respond in this EXACT JSON format only (no markdown, no explanation):
{{"title": "...", "caption": "...", "hashtags": ["completeword1", "completeword2", ...]}}

If not generating a field, use null for that key."""

    return prompt


async def generate_ai_content(
    frames: List[Path],
    generate_title: bool,
    generate_caption: bool,
    generate_hashtags: bool,
    hashtag_count: int,
    existing_title: str,
    existing_caption: str,
    filename: str,
    platform_hints: List[str],
    location_name: Optional[str] = None,
    trill_score: Optional[int] = None,
    trill_bucket: Optional[str] = None,
    max_speed: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Call OpenAI to generate title, caption, and hashtags.

    Returns dict with keys: title, caption, hashtags (List[str]), tokens
    """
    result: Dict[str, Any] = {"title": None, "caption": None, "hashtags": [], "tokens": {}}

    if not OPENAI_API_KEY:
        logger.warning("No OPENAI_API_KEY — skipping AI content generation")
        return result

    prompt = build_ai_prompt(
        filename=filename,
        existing_title=existing_title,
        existing_caption=existing_caption,
        platform_hints=platform_hints,
        generate_title=generate_title,
        generate_caption=generate_caption,
        generate_hashtags=generate_hashtags,
        hashtag_count=hashtag_count,
        location_name=location_name,
        trill_score=trill_score,
        trill_bucket=trill_bucket,
        max_speed=max_speed,
    )

    # Build message content
    content: List[Dict] = [{"type": "text", "text": prompt}]

    # Attach frames (max 3 for cost)
    for frame in frames[:3]:
        try:
            with open(frame, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}",
                    "detail": "low"
                }
            })
        except Exception as e:
            logger.warning(f"Could not attach frame {frame}: {e}")

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": OPENAI_MODEL,
                    "messages": [{"role": "user", "content": content}],
                    "max_tokens": 600,
                    "temperature": 0.7,
                },
            )

            if response.status_code != 200:
                logger.error(
                    f"OpenAI API error: {response.status_code} — {response.text[:300]}"
                )
                return result

            data = response.json()
            usage = data.get("usage", {})
            result["tokens"] = {
                "prompt": usage.get("prompt_tokens", 0),
                "completion": usage.get("completion_tokens", 0),
            }

            answer = data["choices"][0]["message"]["content"]

            # Strip markdown fences if present
            if "```json" in answer:
                answer = answer.split("```json")[1].split("```")[0]
            elif "```" in answer:
                answer = answer.split("```")[1].split("```")[0]

            try:
                parsed = json.loads(answer.strip())
            except json.JSONDecodeError:
                logger.warning(f"AI response not valid JSON: {answer[:300]}")
                return result

            if parsed.get("title"):
                result["title"] = str(parsed["title"])[:100]

            if parsed.get("caption"):
                result["caption"] = str(parsed["caption"])[:500]

            if parsed.get("hashtags") is not None:
                raw_tags = parsed["hashtags"]

                # ── CRITICAL FIX ────────────────────────────────────────────
                # Ensure we have a list. If OpenAI returned a string instead
                # of a list (e.g., "dashcam carlife roadtrip"), split it.
                if isinstance(raw_tags, str):
                    raw_tags = [t.strip() for t in raw_tags.replace(",", " ").split() if t.strip()]
                elif not isinstance(raw_tags, list):
                    raw_tags = []

                cleaned = []
                for tag in raw_tags:
                    tag = str(tag).strip().lstrip('#').lower()
                    # Skip single characters — these are NOT valid hashtags
                    if len(tag) < 2:
                        continue
                    # Skip if it looks like the model returned a sentence fragment
                    if ' ' in tag:
                        # Split on spaces and take parts
                        parts = [p.lstrip('#').lower() for p in tag.split() if len(p.lstrip('#')) >= 2]
                        cleaned.extend(parts)
                    else:
                        cleaned.append(tag)

                result["hashtags"] = cleaned[:hashtag_count]
                logger.info(f"AI hashtags (cleaned): {result['hashtags']}")
                # ─────────────────────────────────────────────────────────────

    except Exception as e:
        logger.error(f"OpenAI API call failed: {e}")

    return result


def estimate_cost(tokens: dict, num_images: int) -> float:
    """Estimate OpenAI API cost in USD."""
    return (
        (tokens.get("prompt", 0) / 1000) * COST_PER_1K_INPUT +
        (tokens.get("completion", 0) / 1000) * COST_PER_1K_OUTPUT +
        num_images * COST_PER_IMAGE
    )


async def run_caption_stage(ctx: JobContext) -> JobContext:
    """
    Execute AI caption generation stage.

    Uses:
    - ctx.location_name (from telemetry_stage reverse geocoding)
    - ctx.trill_score (from telemetry_stage)
    - ctx.telemetry_data (for max speed)
    - ctx.user_settings (for hashtag count, feature flags)
    - ctx.entitlements (for tier gating)

    Updates:
    - ctx.ai_title
    - ctx.ai_caption
    - ctx.ai_hashtags (proper List[str])
    """
    # Feature gate checks
    if not ctx.entitlements:
        raise SkipStage("No entitlements", stage="captions")

    generate_title = getattr(ctx.entitlements, "ai_captions_enabled", False)
    generate_caption = getattr(ctx.entitlements, "ai_captions_enabled", False)
    generate_hashtags = getattr(ctx.entitlements, "ai_hashtags_enabled", False)

    # Check user settings overrides
    if not ctx.user_settings.get("auto_generate_captions", True):
        generate_caption = False
    if not ctx.user_settings.get("auto_generate_hashtags", True):
        generate_hashtags = False

    if not (generate_title or generate_caption or generate_hashtags):
        raise SkipStage("AI generation not enabled for this tier", stage="captions")

    if not ctx.local_video_path or not ctx.local_video_path.exists():
        raise SkipStage("No local video file", stage="captions")

    try:
        frames = await extract_video_frames(ctx.local_video_path, ctx.temp_dir)
        if not frames:
            logger.warning("No frames extracted — generating captions without visual context")

        max_hashtags = get_max_hashtags(ctx.entitlements)
        default_count = int(ctx.user_settings.get("default_hashtag_count", 5))
        hashtag_count = min(default_count, max_hashtags) if generate_hashtags else 0

        # Pull Trill context if available
        trill_score_val: Optional[int] = None
        trill_bucket_val: Optional[str] = None
        max_speed_val: Optional[float] = None

        if ctx.trill_score:
            trill_score_val = ctx.trill_score.total
            # Derive bucket from score
            s = trill_score_val
            trill_bucket_val = (
                "gloryBoy" if s >= 90 else
                "euphoric" if s >= 80 else
                "sendIt" if s >= 60 else
                "spirited" if s >= 40 else
                "chill"
            )
        if ctx.telemetry_data:
            max_speed_val = ctx.telemetry_data.max_speed_mph

        result = await generate_ai_content(
            frames=frames,
            generate_title=generate_title,
            generate_caption=generate_caption,
            generate_hashtags=generate_hashtags,
            hashtag_count=hashtag_count,
            existing_title=ctx.title,
            existing_caption=ctx.caption,
            filename=ctx.filename,
            platform_hints=ctx.platforms,
            location_name=ctx.location_name,      # ← LOCATION INJECTED HERE
            trill_score=trill_score_val,
            trill_bucket=trill_bucket_val,
            max_speed=max_speed_val,
        )

        if result.get("title") and generate_title:
            ctx.ai_title = result["title"]
            logger.info(f"AI title: {ctx.ai_title[:80]}")

        if result.get("caption") and generate_caption:
            ctx.ai_caption = result["caption"]
            logger.info(f"AI caption: {ctx.ai_caption[:80]}")

        if result.get("hashtags") and generate_hashtags:
            ctx.ai_hashtags = result["hashtags"][:max_hashtags]
            logger.info(f"AI hashtags ({len(ctx.ai_hashtags)}): {ctx.ai_hashtags}")

        tokens_used = result.get("tokens", {})
        cost = estimate_cost(tokens_used, len(frames))
        logger.info(f"OpenAI cost estimate: ${cost:.4f}")

        return ctx

    except SkipStage:
        raise
    except StageError:
        raise
    except Exception as e:
        logger.error(f"Caption generation failed (non-fatal): {e}")
        ctx.mark_error(ErrorCode.AI_CAPTION_FAILED.value if hasattr(ErrorCode, 'AI_CAPTION_FAILED') else "AI_CAPTION_FAILED", str(e))
        return ctx
