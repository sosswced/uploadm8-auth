"""
UploadM8 Caption Stage — Narrative Story Edition
=================================================
Generates AI-powered titles, captions, and hashtags using OpenAI GPT-4o.

Key features:
  - Multi-frame story: sends up to N frames chronologically to GPT-4o, prompting
    a narrative arc (beginning → middle → end of the drive)
  - Trill telemetry as story beats: score bucket, peak speed, euphoria moments
    are injected as CRITICAL STORY BEATS the AI must weave into the caption
  - Full user preferences: caption_style, caption_tone, ai_hashtag_count,
    ai_hashtag_style, always_hashtags, blocked_hashtags all sourced from
    user_settings (saved via settings page)
  - Hashtag merging: AI hashtags are ADDITIVE — never replace always/preset tags
  - Non-fatal: pipeline continues on AI failure

Exports: run_caption_stage(ctx)
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

logger = logging.getLogger("uploadm8-worker.caption")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
# Default model — overridden by user_settings["trillOpenaiModel"] when set
OPENAI_MODEL_DEFAULT = os.environ.get("OPENAI_CAPTION_MODEL", "gpt-4o-mini")

# Pricing per 1K tokens (gpt-4o-mini defaults; gpt-4o is ~10x)
COST_PER_1K_INPUT  = 0.000150
COST_PER_1K_OUTPUT = 0.000600
COST_PER_IMAGE     = 0.00765   # low-detail vision


# ============================================================
# Frame Collection — prefers multi-frame story from thumbnail stage
# ============================================================

async def _collect_story_frames(ctx: JobContext, max_frames: int) -> List[Path]:
    """
    Collect frames for AI analysis.

    Priority:
      1. ctx.thumbnail_paths  (already extracted by thumbnail_stage, multi-frame)
         → sorted by filename so they stay in chronological order
      2. ctx.thumbnail_path   (single frame fallback)
      3. Live extraction from video (last resort)

    Returns at most max_frames paths.
    """
    # Option 1: multi-frame from thumbnail_stage
    if ctx.thumbnail_paths:
        existing = [p for p in ctx.thumbnail_paths if Path(p).exists()]
        if existing:
            # Ensure chronological order (thumbnail_stage names them _00, _01, etc.)
            existing.sort(key=lambda p: str(p))
            logger.debug(f"Using {len(existing)} pre-extracted frames from thumbnail_stage")
            return existing[:max_frames]

    # Option 2: single thumbnail
    if ctx.thumbnail_path and Path(ctx.thumbnail_path).exists():
        logger.debug("Using single thumbnail_path for caption AI")
        return [ctx.thumbnail_path]

    # Option 3: live extraction
    video_path = ctx.local_video_path or ctx.processed_video_path
    if not video_path or not Path(video_path).exists():
        return []

    logger.debug("No pre-extracted frames — running live extraction for caption AI")
    return await _live_extract_frames(Path(video_path), ctx.temp_dir, max_frames)


async def _live_extract_frames(video_path: Path, temp_dir: Path, n: int) -> List[Path]:
    """Fallback: extract N frames from video on-the-fly."""
    frames: List[Path] = []
    try:
        # Get duration
        dur_cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_streams", str(video_path)
        ]
        proc = await asyncio.create_subprocess_exec(
            *dur_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        data = json.loads(stdout.decode())
        duration = float(next(
            (s["duration"] for s in data.get("streams", [])
             if s.get("codec_type") == "video"),
            30.0
        ))

        interval = max(1.0, duration / (n + 1))
        for i in range(1, n + 1):
            ts = interval * i
            frame_path = temp_dir / f"caption_frame_{i:02d}.jpg"
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
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc2.wait()
            if frame_path.exists() and frame_path.stat().st_size > 1024:
                frames.append(frame_path)
    except Exception as e:
        logger.warning(f"Live frame extraction failed: {e}")
    return frames


# ============================================================
# Trill Story Beat Builder
# ============================================================

def _build_trill_beat(ctx: JobContext) -> Optional[str]:
    """
    Construct the Trill story beat injection string.

    This is injected into the AI prompt as a CRITICAL instruction so the
    AI weaves real driving telemetry data into the narrative.
    """
    ts = ctx.trill_score
    td = ctx.telemetry_data or ctx.telemetry

    if not ts and not td:
        return None

    lines = []

    if ts:
        score = ts.score
        bucket = ts.bucket or (
            "gloryBoy" if score >= 90 else
            "euphoric" if score >= 80 else
            "sendIt"   if score >= 60 else
            "spirited" if score >= 40 else
            "chill"
        )

        bucket_label = {
            "gloryBoy": "🏆 Glory Boy — elite-level drive",
            "euphoric":  "⚡ Euphoric — high-intensity run",
            "sendIt":    "🔥 Send It — aggressive, thrilling",
            "spirited":  "💨 Spirited — energetic and engaging",
            "chill":     "😎 Chill — smooth, relaxed cruise",
        }.get(bucket, bucket)

        lines.append(f"Trill Score: {score}/100 — {bucket_label}")
        if ts.excessive_speed:
            lines.append("Flagged: Excessive speed detected — convey raw energy without glorifying recklessness")

    if td:
        if td.max_speed_mph and td.max_speed_mph > 0:
            lines.append(f"Peak speed: {td.max_speed_mph:.0f} mph")
        if td.avg_speed_mph and td.avg_speed_mph > 0:
            lines.append(f"Average speed: {td.avg_speed_mph:.0f} mph")
        if td.total_distance_miles and td.total_distance_miles > 0:
            lines.append(f"Distance covered: {td.total_distance_miles:.1f} miles")
        if td.euphoria_seconds and td.euphoria_seconds > 0:
            lines.append(f"Euphoria time (high-speed sustained): {td.euphoria_seconds:.0f}s")
        if td.location_display:
            lines.append(f"Location: {td.location_display}")
        if td.location_road:
            lines.append(f"Road/highway: {td.location_road}")

    if not lines:
        return None

    beat = "\n".join(f"  • {l}" for l in lines)
    return (
        "━━ TRILL STORY BEAT (MUST appear in caption) ━━\n"
        f"{beat}\n"
        "━━ Use these real numbers to make the story AUTHENTIC. "
        "Reference the score bucket personality, the peak speed moment, "
        "and the location to create a narrative that feels lived-in and real. ━━"
    )


# ============================================================
# Prompt Builder
# ============================================================

def _build_narrative_prompt(
    ctx: JobContext,
    num_frames: int,
    generate_title: bool,
    generate_caption: bool,
    generate_hashtags: bool,
    hashtag_count: int,
    caption_style: str,
    caption_tone: str,
    hashtag_style: str,
) -> str:
    """
    Build the full AI prompt with multi-frame narrative instructions,
    Trill beat, location, and user preference signals.
    """
    us = ctx.user_settings or {}
    platform_str = ", ".join(ctx.platforms) if ctx.platforms else "social media"

    # ── Task list ────────────────────────────────────────────────────────────
    tasks = []
    if generate_title:
        tasks.append('1. A punchy TITLE (max 100 characters)')
    if generate_caption:
        caption_length = {
            "story":   "150–280 characters — tell a narrative arc",
            "punchy":  "under 120 characters — hook in the first 3 words",
            "factual": "100–200 characters — lead with the most impressive stat",
        }.get(caption_style, "150–280 characters")
        tasks.append(f'2. A CAPTION ({caption_length})')
    if generate_hashtags and hashtag_count > 0:
        style_hint = {
            "trending": "prioritise viral trending tags",
            "niche":    "prioritise specific niche community tags",
            "mixed":    "mix viral and niche tags",
        }.get(hashtag_style, "mix viral and niche tags")
        tasks.append(
            f'3. Exactly {hashtag_count} HASHTAGS ({style_hint}) — '
            f'return as JSON array of complete words WITHOUT the # symbol. '
            f'NEVER return single letters or fragments.'
        )
    tasks_block = "\n".join(tasks) if tasks else "(none requested)"

    # ── Tone instructions ────────────────────────────────────────────────────
    tone_instruction = {
        "hype":       "Energy is HIGH. Use power words, exclamation, urgency. Make viewers stop scrolling.",
        "calm":       "Measured and confident. Let the footage speak. Understated cool.",
        "cinematic":  "Poetic, atmospheric. Paint a picture with words. Think film trailer voiceover.",
        "authentic":  "Real talk, first-person, no fluff. Like texting a friend about the drive.",
    }.get(caption_tone, "Engaging and authentic — feel free to use the tone that best fits the footage.")

    # ── Multi-frame narrative instruction ───────────────────────────────────
    if num_frames > 1:
        frame_instruction = (
            f"You are being shown {num_frames} frames from this video in CHRONOLOGICAL ORDER "
            f"(frame 1 = start of video, frame {num_frames} = near the end). "
            f"Use the visual progression to tell a STORY — describe how the drive EVOLVED. "
            f"What changed? What built up? What was the peak moment? "
            f"The caption should have a beginning, middle, and end feeling."
        )
    else:
        frame_instruction = (
            "You are being shown 1 frame from this video. "
            "Generate content based on the visual context and telemetry data provided."
        )

    # ── Context block ────────────────────────────────────────────────────────
    context_lines = [
        f"Filename: {ctx.filename}",
        f"Target platforms: {platform_str}",
    ]
    if ctx.title:
        context_lines.append(f"User-provided title: {ctx.title}")
    if ctx.caption:
        context_lines.append(f"User-provided caption hint: {ctx.caption}")
    if ctx.location_name:
        context_lines.append(f"Filming location: {ctx.location_name}")
        context_lines.append(
            f"  → Naturally weave '{ctx.location_name}' into the caption and/or hashtags"
        )

    context_block = "\n".join(f"• {l}" for l in context_lines)

    # ── Trill beat ───────────────────────────────────────────────────────────
    trill_beat = _build_trill_beat(ctx)
    trill_section = f"\n\n{trill_beat}" if trill_beat else ""

    # ── Full prompt ──────────────────────────────────────────────────────────
    prompt = f"""You are a social media content specialist for {platform_str}.

{frame_instruction}

TONE DIRECTIVE: {tone_instruction}

CAPTION STYLE: {caption_style.upper()} — follow this style strictly.

Generate the following for this video:
{tasks_block}

Context:
{context_block}{trill_section}

Rules:
- Content must feel AUTHENTIC — not AI-generated
- For TikTok/Instagram/YouTube Shorts: hook in the first 3 words
- Use emojis sparingly but strategically (1–3 max)
- HASHTAGS: each must be a complete word (e.g. "dashcam", "gloryboy", "roadtrip")
  NEVER return single characters or word fragments
- If location is provided: include city/state as a hashtag, reference it in caption
- If Trill data is provided: the caption MUST reference at least one real data point
  (speed, score bucket, location, or distance) to make it feel authentic

Respond ONLY in this exact JSON format (no markdown, no explanation):
{{"title": "...", "caption": "...", "hashtags": ["word1", "word2", ...]}}

Use null for any key you are not generating."""

    return prompt


# ============================================================
# OpenAI API Call
# ============================================================

async def _call_openai(
    frames: List[Path],
    prompt: str,
    model: str,
    hashtag_count: int,
) -> Dict[str, Any]:
    """
    Call OpenAI with the prompt and attached frames.
    Returns dict: {title, caption, hashtags, tokens}
    """
    result: Dict[str, Any] = {
        "title": None, "caption": None, "hashtags": [], "tokens": {}
    }

    if not OPENAI_API_KEY:
        logger.warning("No OPENAI_API_KEY — skipping AI content generation")
        return result

    # Build message content
    content: List[Dict] = [{"type": "text", "text": prompt}]

    # Attach frames (cap at 6 for cost sanity; gpt-4o-mini handles vision)
    frames_to_send = frames[:6]
    for frame in frames_to_send:
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
            logger.warning(f"Could not attach frame {frame.name}: {e}")

    try:
        async with httpx.AsyncClient(timeout=90) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": content}],
                    "max_tokens": 800,
                    "temperature": 0.75,
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
                "prompt":     usage.get("prompt_tokens", 0),
                "completion": usage.get("completion_tokens", 0),
            }

            answer = data["choices"][0]["message"]["content"]

            # Strip markdown fences
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

            raw_tags = parsed.get("hashtags")
            if raw_tags is not None:
                # Normalise — OpenAI sometimes returns a string or partial results
                if isinstance(raw_tags, str):
                    raw_tags = [t.strip() for t in raw_tags.replace(",", " ").split() if t.strip()]
                elif not isinstance(raw_tags, list):
                    raw_tags = []

                cleaned: List[str] = []
                for tag in raw_tags:
                    tag = str(tag).strip().lstrip("#").lower()
                    if len(tag) < 2:
                        continue  # skip single chars
                    if " " in tag:
                        # Split multi-word accidents
                        for part in tag.split():
                            p = part.lstrip("#").lower()
                            if len(p) >= 2:
                                cleaned.append(p)
                    else:
                        cleaned.append(tag)

                result["hashtags"] = cleaned[:hashtag_count]
                logger.info(f"AI hashtags ({len(result['hashtags'])}): {result['hashtags']}")

    except Exception as e:
        logger.error(f"OpenAI API call failed: {e}")

    return result


# ============================================================
# Hashtag Finalisation
# ============================================================

def _finalise_hashtags(
    ai_tags: List[str],
    base_tags: List[str],
    blocked: List[str],
    max_total: int,
) -> List[str]:
    """
    Merge AI tags onto base/always tags, remove blocked, cap at max_total.
    Base tags (always_hashtags + preset) come first, AI tags are additive.
    """
    blocked_set = {t.lower().lstrip("#") for t in (blocked or [])}
    seen: set = set()
    merged: List[str] = []

    for tag in list(base_tags or []) + list(ai_tags or []):
        t = str(tag).strip().lstrip("#").lower()
        if not t or t in seen or t in blocked_set:
            continue
        seen.add(t)
        merged.append(f"#{t}" if not str(tag).startswith("#") else str(tag))

    return merged[:max_total]


# ============================================================
# Cost Estimation
# ============================================================

def _estimate_cost(tokens: dict, num_images: int) -> float:
    return (
        (tokens.get("prompt", 0)     / 1000) * COST_PER_1K_INPUT  +
        (tokens.get("completion", 0) / 1000) * COST_PER_1K_OUTPUT +
        num_images * COST_PER_IMAGE
    )


# ============================================================
# Stage Entry Point
# ============================================================

async def run_caption_stage(ctx: JobContext) -> JobContext:
    """
    Execute AI caption generation stage.

    Reads from ctx:
      - ctx.entitlements           — plan gates (can_ai, max_caption_frames)
      - ctx.user_settings          — all user preferences from settings page
      - ctx.thumbnail_paths        — multi-frame story frames (from thumbnail_stage)
      - ctx.trill_score            — Trill telemetry beat
      - ctx.telemetry_data         — raw speed/distance/location data
      - ctx.location_name          — reverse-geocoded location display string

    Writes to ctx:
      - ctx.ai_title
      - ctx.ai_caption
      - ctx.ai_hashtags            — AI additions (merged into base tags via get_effective_hashtags)
    """
    # ── Plan gate ────────────────────────────────────────────────────────────
    if not ctx.entitlements:
        raise SkipStage("No entitlements")

    can_ai = getattr(ctx.entitlements, "can_ai", False)
    if not can_ai:
        raise SkipStage("AI not available on this plan")

    us = ctx.user_settings or {}

    # ── User preference toggles ──────────────────────────────────────────────
    # Support both camelCase (saved by settings.html) and snake_case
    generate_caption  = bool(us.get("autoCaptions") or us.get("auto_captions") or False)
    generate_title    = generate_caption  # title always follows caption toggle
    generate_hashtags = bool(us.get("aiHashtagsEnabled") or us.get("ai_hashtags_enabled") or False)

    # Explicit override keys (some code paths send these separately)
    if us.get("auto_generate_captions") is not None:
        generate_caption = bool(us["auto_generate_captions"])
        generate_title   = generate_caption
    if us.get("auto_generate_hashtags") is not None:
        generate_hashtags = bool(us["auto_generate_hashtags"])

    if not (generate_title or generate_caption or generate_hashtags):
        raise SkipStage("AI generation not enabled by user settings")

    # ── Read user style/tone preferences ─────────────────────────────────────
    # caption_style: story | punchy | factual  (default: story)
    caption_style = str(
        us.get("captionStyle") or us.get("caption_style") or "story"
    ).lower()
    if caption_style not in ("story", "punchy", "factual"):
        caption_style = "story"

    # caption_tone: hype | calm | cinematic | authentic  (default: authentic)
    caption_tone = str(
        us.get("captionTone") or us.get("caption_tone") or "authentic"
    ).lower()
    if caption_tone not in ("hype", "calm", "cinematic", "authentic"):
        caption_tone = "authentic"

    # hashtag_style: trending | niche | mixed  (default: mixed)
    hashtag_style = str(
        us.get("aiHashtagStyle") or us.get("ai_hashtag_style") or "mixed"
    ).lower()
    if hashtag_style not in ("trending", "niche", "mixed"):
        hashtag_style = "mixed"

    # hashtag count: user preference, bounded by sane limits
    pref_max = int(
        us.get("aiHashtagCount") or
        us.get("ai_hashtag_count") or
        us.get("maxHashtags") or
        us.get("max_hashtags") or
        15
    )
    pref_max = max(1, min(pref_max, 50))
    hashtag_count = pref_max if generate_hashtags else 0

    # OpenAI model: user can pick per-settings (gpt-4o for better quality)
    model = str(us.get("trillOpenaiModel") or us.get("openai_model") or OPENAI_MODEL_DEFAULT)

    # Max frames from entitlements
    max_caption_frames = getattr(ctx.entitlements, "max_caption_frames", 3) or 3
    # User can set lower via settings slider (captionFrameCount)
    user_frame_count = int(us.get("captionFrameCount") or us.get("caption_frame_count") or max_caption_frames)
    num_frames = min(user_frame_count, max_caption_frames)
    num_frames = max(1, num_frames)

    logger.info(
        f"Caption stage: style={caption_style}, tone={caption_tone}, "
        f"hashtag_style={hashtag_style}, hashtag_count={hashtag_count}, "
        f"model={model}, num_frames={num_frames}"
    )

    try:
        # ── Collect story frames ─────────────────────────────────────────────
        frames = await _collect_story_frames(ctx, num_frames)
        if not frames:
            logger.warning("No frames available — generating captions without visual context")

        # ── Build prompt ─────────────────────────────────────────────────────
        prompt = _build_narrative_prompt(
            ctx=ctx,
            num_frames=len(frames),
            generate_title=generate_title,
            generate_caption=generate_caption,
            generate_hashtags=generate_hashtags,
            hashtag_count=hashtag_count,
            caption_style=caption_style,
            caption_tone=caption_tone,
            hashtag_style=hashtag_style,
        )

        # ── Call OpenAI ──────────────────────────────────────────────────────
        result = await _call_openai(
            frames=frames,
            prompt=prompt,
            model=model,
            hashtag_count=hashtag_count,
        )

        # ── Apply results ────────────────────────────────────────────────────
        if result.get("title") and generate_title:
            ctx.ai_title = result["title"]
            logger.info(f"AI title: {ctx.ai_title[:80]}")

        if result.get("caption") and generate_caption:
            ctx.ai_caption = result["caption"]
            logger.info(f"AI caption: {ctx.ai_caption[:80]}")

        if result.get("hashtags") and generate_hashtags:
            # Store raw AI tags in ctx.ai_hashtags
            # get_effective_hashtags() in context.py will merge with base/always tags
            ctx.ai_hashtags = result["hashtags"][:pref_max]
            logger.info(f"AI hashtags ({len(ctx.ai_hashtags)}): {ctx.ai_hashtags}")

        tokens_used = result.get("tokens", {})
        cost = _estimate_cost(tokens_used, len(frames))
        logger.info(
            f"OpenAI usage: prompt_tokens={tokens_used.get('prompt', 0)}, "
            f"completion_tokens={tokens_used.get('completion', 0)}, "
            f"estimated_cost=${cost:.4f}"
        )

        return ctx

    except SkipStage:
        raise
    except StageError:
        raise
    except Exception as e:
        logger.error(f"Caption generation failed (non-fatal): {e}")
        err_code = (
            ErrorCode.AI_CAPTION_FAILED.value
            if hasattr(ErrorCode, "AI_CAPTION_FAILED")
            else "AI_CAPTION_FAILED"
        )
        ctx.mark_error(err_code, str(e))
        return ctx
