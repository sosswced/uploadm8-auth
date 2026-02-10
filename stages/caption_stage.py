"""
UploadM8 Caption Stage
======================
Generate AI captions, titles, and hashtags using OpenAI.

Features:
- Auto-generate title from video content
- Auto-generate engaging caption
- Auto-generate relevant hashtags (tier-gated limits)
- Respects user settings for auto-generation
- Tracks OpenAI API costs
"""

import os
import asyncio
import logging
import base64
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
import httpx

from .context import JobContext
from .errors import StageError, SkipStage, ErrorCode
from .entitlements import (
    should_generate_captions, 
    should_generate_hashtags,
    get_max_hashtags
)

logger = logging.getLogger("uploadm8-worker")

# Configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
FFMPEG_PATH = os.environ.get("FFMPEG_PATH", "ffmpeg")


def _norm_tag(t: str) -> str:
    return str(t or "").strip().lstrip("#").strip()

def _dedupe_case_insensitive(tags: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for t in tags:
        nt = _norm_tag(t)
        if not nt:
            continue
        key = nt.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(nt)
    return out


# Cost tracking (approximate)
COST_PER_1K_INPUT = 0.00015  # gpt-4o-mini input
COST_PER_1K_OUTPUT = 0.0006  # gpt-4o-mini output
COST_PER_IMAGE = 0.000425   # low detail image


async def run_caption_stage(ctx: JobContext) -> JobContext:
    """
    Generate AI captions, titles, and hashtags.
    
    Process:
    1. Check entitlements and settings
    2. Extract video frames for analysis
    3. Call OpenAI to generate content
    4. Apply hashtag limits
    5. Track costs
    """
    ctx.mark_stage("captions")
    
    if not ctx.entitlements:
        raise SkipStage("No entitlements loaded", stage="captions")
    
    if not OPENAI_API_KEY:
        raise SkipStage("OpenAI API key not configured", stage="captions")
    
    # Check what needs to be generated
    has_title = bool(ctx.title and ctx.title.strip())
    has_caption = bool(ctx.caption and ctx.caption.strip())
    has_hashtags = bool(ctx.hashtags)
    
    generate_title = not has_title and ctx.entitlements.ai_captions_enabled
    generate_caption = should_generate_captions(ctx.entitlements, has_caption)
    generate_hashtags = should_generate_hashtags(
        ctx.entitlements, 
        has_hashtags,
        ctx.user_settings.get("always_use_hashtags", False)
    )
    
    # Check user settings
    auto_captions = ctx.user_settings.get("auto_generate_captions", True)
    auto_hashtags = ctx.user_settings.get("auto_generate_hashtags", True)
    
    if not auto_captions:
        generate_title = False
        generate_caption = False
    
    if not auto_hashtags:
        generate_hashtags = False
    
    if not generate_title and not generate_caption and not generate_hashtags:
        raise SkipStage("No content generation needed", stage="captions")
    
    if not ctx.local_video_path or not ctx.local_video_path.exists():
        raise SkipStage("No local video file", stage="captions")
    
    try:
        # Extract frames for analysis
        frames = await extract_video_frames(ctx.local_video_path, ctx.temp_dir)
        
        if not frames:
            logger.warning("No frames extracted for caption generation")
            # Generate without visual context
            frames = []
        
        # Get hashtag limit and default count
        max_hashtags = get_max_hashtags(ctx.entitlements)
        default_count = ctx.user_settings.get("default_hashtag_count", 5)
        hashtag_count = min(default_count, max_hashtags) if generate_hashtags else 0
        
        # Generate content
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
        )
        
        # Apply results
        if result.get("title") and generate_title:
            ctx.ai_title = result["title"]
            logger.info(f"AI title: {ctx.ai_title[:50]}...")
        
        if result.get("caption") and generate_caption:
            ctx.ai_caption = result["caption"]
            logger.info(f"AI caption: {ctx.ai_caption[:50]}...")
        
        if result.get("hashtags") and generate_hashtags:
            # Get AI generated tags
            ai_tags = result["hashtags"]
            
            # Apply user hashtag preferences
            # 1. Add always-include hashtags
            always_tags = ctx.user_settings.get("always_hashtags", [])
            combined_tags = list(ai_tags) + list(always_tags)
            
            # 2. Remove blocked hashtags
            blocked_tags = set(ctx.user_settings.get("blocked_hashtags", []))
            # Normalize for comparison (lowercase)
            combined_tags = [
                tag for tag in combined_tags 
                if tag.lower() not in {b.lower() for b in blocked_tags}
            ]
            
            # 3. Apply hashtag style preference
            hashtag_style = ctx.user_settings.get("ai_hashtag_style", "mixed")
            if hashtag_style == "lowercase":
                combined_tags = [tag.lower() for tag in combined_tags]
            elif hashtag_style == "capitalized":
                combined_tags = [tag.capitalize() for tag in combined_tags]
            elif hashtag_style == "camelcase":
                # Convert to camelCase (first word lowercase, rest capitalized)
                combined_tags = [
                    tag[0].lower() + tag[1:].title().replace(" ", "") if len(tag) > 1 else tag.lower()
                    for tag in combined_tags
                ]
            # else: mixed - keep as is
            
            # 4. Deduplicate and limit
            seen = set()
            unique_tags = []
            for tag in combined_tags:
                tag_lower = tag.lower()
                if tag_lower not in seen:
                    seen.add(tag_lower)
                    unique_tags.append(tag)
            
            ctx.ai_hashtags = unique_tags[:max_hashtags]
            logger.info(f"AI hashtags (after preferences): {ctx.ai_hashtags}")
        
        # Track cost (estimated)
        tokens_used = result.get("tokens", {})
        cost = estimate_cost(tokens_used, len(frames))
        
        # Would track to DB here if we had the pool
        logger.info(f"OpenAI cost estimate: ${cost:.4f}")


    except SkipStage:
        raise
    except StageError:
        raise
    except Exception as e:
        logger.exception("Caption stage failed")
        raise StageError(
            ErrorCode.AI_CAPTION_FAILED,
            "AI metadata generation failed",
            details={"error": str(e)},
            retryable=True,
            stage="captions",
        )

    # ---------------------------------------------------------------------
    # FINAL HASHTAG RESOLUTION (base + per-platform) — deterministic + UI-ready
    # ---------------------------------------------------------------------
    always_tags = ctx.user_settings.get("always_hashtags", []) or []
    blocked = set(_norm_tag(b).lower() for b in (ctx.user_settings.get("blocked_hashtags", []) or []))
    platform_prefs = ctx.user_settings.get("platform_hashtags", {}) or {}

    # Base input precedence: user override > AI generated > empty
    if ctx.hashtags:
        base_input = list(ctx.hashtags)
    elif ctx.ai_hashtags:
        base_input = list(ctx.ai_hashtags)
    else:
        base_input = []

    base_combined = _dedupe_case_insensitive(list(base_input) + list(always_tags))
    base_filtered = [t for t in base_combined if _norm_tag(t).lower() not in blocked]

    max_tags = ctx.user_settings.get("max_hashtags", 15)
    ctx.final_hashtags = base_filtered[:max_tags]

    platform_map: Dict[str, List[str]] = {}
    for p in (ctx.platforms or []):
        extra = platform_prefs.get(p, []) or []
        merged = _dedupe_case_insensitive(list(ctx.final_hashtags) + list(extra))
        merged = [t for t in merged if _norm_tag(t).lower() not in blocked]
        platform_map[p] = merged[:max_tags]

    ctx.platform_hashtags_map = platform_map
    logger.info(f"Resolved base hashtags: {ctx.final_hashtags}")
    logger.info(f"Resolved platform hashtags: {ctx.platform_hashtags_map}")

    return ctx

async def extract_video_frames(video_path: Path, temp_dir: Path, num_frames: int = 4) -> List[Path]:
    """
    Extract frames from video for AI analysis.
    Args:
        video_path: Path to video file
        temp_dir: Directory to save frames
        num_frames: Number of frames to extract
    Returns:
        List of paths to frame images
    """
    frames = []
    try:
        # Get video duration first
        cmd = [
            "ffprobe", "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "json",
            str(video_path)
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await proc.communicate()
        data = json.loads(stdout)
        duration = float(data.get("format", {}).get("duration", 30))
        # Calculate timestamps (avoid very start and end)
        interval = duration / (num_frames + 1)
        timestamps = [interval * (i + 1) for i in range(num_frames)]
        # Extract frames
        for i, ts in enumerate(timestamps):
            output = temp_dir / f"frame_{i:03d}.jpg"
            cmd = [
                FFMPEG_PATH,
                "-ss", str(ts),
                "-i", str(video_path),
                "-vframes", "1",
                "-q:v", "5",  # Medium quality for analysis
                "-vf", "scale=512:-1",  # Resize for API efficiency
                "-y",
                str(output)
            ]
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await proc.communicate()
            if output.exists():
                frames.append(output)
    except Exception as e:
        logger.warning(f"Frame extraction error: {e}")
    return frames
async def generate_ai_content(
    frames: List[Path],
    generate_title: bool,
    generate_caption: bool,
    generate_hashtags: bool,
    hashtag_count: int,
    existing_title: str = "",
    existing_caption: str = "",
    filename: str = "",
    platform_hints: List[str] = None,
) -> Dict[str, Any]:
    """
    Generate title, caption, and hashtags using OpenAI.
    Returns dict with keys: title, caption, hashtags, tokens
    """
    result = {"title": None, "caption": None, "hashtags": [], "tokens": {}}
    if not OPENAI_API_KEY:
        return result
    # Build prompt
    tasks = []
    if generate_title:
        tasks.append("1. An attention-grabbing TITLE (max 100 characters)")
    if generate_caption:
        tasks.append("2. An engaging CAPTION for social media (max 280 characters)")
    if generate_hashtags and hashtag_count > 0:
        tasks.append(f"3. Exactly {hashtag_count} relevant HASHTAGS (just the words, no # symbol)")
    if not tasks:
        return result
    platform_str = ", ".join(platform_hints) if platform_hints else "social media"
    prompt = f"""Analyze this video content and generate the following for {platform_str}:
{chr(10).join(tasks)}
Context:
- Filename: {filename}
- Existing title: {existing_title or 'None'}
- Existing caption: {existing_caption or 'None'}
Important:
- Be engaging and encourage interaction
- Use emojis sparingly but effectively
- Make content feel authentic, not AI-generated
- For TikTok/Instagram, be trendy and casual
- For YouTube, be more descriptive
Respond in this exact JSON format:
{{"title": "...", "caption": "...", "hashtags": ["tag1", "tag2", ...]}}
If you're not generating something, use null for that field."""
    # Build message content
    content = [{"type": "text", "text": prompt}]
    # Add frames if available (limit to 3 for cost)
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
        except:
            pass
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": content}],
                    "max_tokens": 500,
                    "temperature": 0.7,
                }
            )
            if response.status_code != 200:
                logger.error(f"OpenAI API error: {response.status_code} - {response.text[:200]}")
                return result
            data = response.json()
            # Track tokens
            usage = data.get("usage", {})
            result["tokens"] = {
                "prompt": usage.get("prompt_tokens", 0),
                "completion": usage.get("completion_tokens", 0),
            }
            # Parse response
            answer = data["choices"][0]["message"]["content"]
            # Try to extract JSON
            try:
                # Handle markdown code blocks
                if "```json" in answer:
                    answer = answer.split("```json")[1].split("```")[0]
                elif "```" in answer:
                    answer = answer.split("```")[1].split("```")[0]
                parsed = json.loads(answer.strip())
                if parsed.get("title"):
                    result["title"] = str(parsed["title"])[:100]
                if parsed.get("caption"):
                    result["caption"] = str(parsed["caption"])[:500]
                if parsed.get("hashtags"):
                    result["hashtags"] = [
                        str(h).lower().strip().lstrip('#') 
                        for h in parsed["hashtags"] 
                        if h
                    ][:hashtag_count]
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse OpenAI response as JSON: {answer[:200]}")
    except Exception as e:
        logger.error(f"OpenAI API call failed: {e}")
    return result
def estimate_cost(tokens: dict, num_images: int) -> float:
    """Estimate OpenAI API cost."""
    input_tokens = tokens.get("prompt", 0)
    output_tokens = tokens.get("completion", 0)
    cost = (
        (input_tokens / 1000) * COST_PER_1K_INPUT +
        (output_tokens / 1000) * COST_PER_1K_OUTPUT +
        num_images * COST_PER_IMAGE
    )
    return cost
