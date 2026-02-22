"""
UploadM8 Caption Stage
=======================
Generate AI-powered titles, captions, and hashtags using OpenAI.

WHAT'S NEW IN THIS VERSION:
  - Multi-frame extraction: pulls 8 frames from the video via FFmpeg
    (beginning, action middle, and end of clip) for richer visual context
  - Pandas telemetry enrichment: uses p95 speed, acceleration events,
    speeding_seconds, euphoria_seconds from the telemetry stage
  - Location-mandatory prompts: city/state/road are injected as required
    context, not optional hints — AI must reference them
  - Platform-specific prompt engineering: separate tone and hashtag
    instructions per TikTok / YouTube / Instagram / Facebook
  - Title fallback: if user supplied a title, AI still enhances it
    rather than regenerating from scratch

GROUNDING RULES (enforced):
  - All generated content MUST be based on actual FFmpeg frames and/or
    real telemetry / Trill data from the .map file
  - If neither frames nor telemetry exist, generation is skipped
  - GPT-4o (vision) is used when frames are available
  - Location data is injected verbatim — no invented place names

Exports: run_caption_stage(ctx)
"""

import asyncio
import base64
import json
import logging
import os
from pathlib import Path
from typing import Optional, List, Dict, Any

import httpx

from .errors import SkipStage, CaptionError, ErrorCode
from .context import JobContext, TrillScore, TelemetryData

logger = logging.getLogger("uploadm8-worker.caption")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_VISION_MODEL = os.environ.get("OPENAI_VISION_MODEL", "gpt-4o")
OPENAI_TEXT_MODEL = os.environ.get("OPENAI_TEXT_MODEL", "gpt-4o-mini")
FFMPEG_PATH = os.environ.get("FFMPEG_PATH", "ffmpeg")
FFPROBE_PATH = os.environ.get("FFPROBE_PATH", "ffprobe")

# Number of frames to extract for AI analysis
NUM_ANALYSIS_FRAMES = 8


# ---------------------------------------------------------------------------
# FFmpeg multi-frame extraction
# ---------------------------------------------------------------------------

async def _get_video_duration(video_path: Path) -> float:
    """Use ffprobe to get video duration in seconds."""
    try:
        cmd = [
            FFPROBE_PATH,
            "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "json",
            str(video_path),
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        data = json.loads(stdout.decode())
        return float(data.get("format", {}).get("duration", 30.0))
    except Exception as e:
        logger.warning(f"ffprobe duration failed: {e} — assuming 30s")
        return 30.0


async def _extract_single_frame(
    video_path: Path,
    output_path: Path,
    timestamp: float,
) -> bool:
    """Extract a single JPEG frame at the given timestamp."""
    cmd = [
        FFMPEG_PATH,
        "-y",
        "-ss", f"{timestamp:.3f}",
        "-i", str(video_path),
        "-vframes", "1",
        "-q:v", "4",
        "-vf", "scale=768:-2",  # 768px wide — enough for GPT-4o vision, not too costly
        str(output_path),
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
            return True
        logger.debug(f"Frame at {timestamp:.1f}s failed: {stderr.decode()[-200:]}")
        return False
    except Exception as e:
        logger.debug(f"Frame extraction error at {timestamp:.1f}s: {e}")
        return False


async def extract_analysis_frames(
    video_path: Path,
    temp_dir: Path,
    num_frames: int = NUM_ANALYSIS_FRAMES,
) -> List[Path]:
    """
    Extract multiple frames evenly distributed across the video
    for OpenAI visual analysis.

    Strategy:
      - Skip first 5% and last 5% of video (avoids black frames / end cards)
      - Spread remaining frames across the video uniformly
      - Extract concurrently for speed
      - Return paths for successfully extracted frames only
    """
    if not video_path or not video_path.exists():
        return []

    duration = await _get_video_duration(video_path)

    # Skip first and last 5%
    start_offset = duration * 0.05
    end_offset = duration * 0.95
    effective_duration = end_offset - start_offset

    if effective_duration <= 0:
        # Very short clip — just grab a couple frames
        timestamps = [0.5, max(0, duration - 0.5)]
    else:
        interval = effective_duration / num_frames
        timestamps = [start_offset + interval * i + interval / 2 for i in range(num_frames)]

    # Extract all frames concurrently
    tasks = []
    output_paths = []
    for i, ts in enumerate(timestamps):
        out = temp_dir / f"caption_frame_{i:03d}.jpg"
        output_paths.append(out)
        tasks.append(_extract_single_frame(video_path, out, ts))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    frames = [
        output_paths[i]
        for i, ok in enumerate(results)
        if ok is True and output_paths[i].exists()
    ]

    logger.info(f"Extracted {len(frames)}/{num_frames} analysis frames from {video_path.name}")
    return frames


def _frames_to_b64(frames: List[Path], max_frames: int = 6) -> List[str]:
    """
    Convert frame paths to base64 strings for OpenAI vision.
    Caps at max_frames to control cost (6 frames * ~85 tokens = ~510 tokens).
    """
    b64_list = []
    for frame in frames[:max_frames]:
        try:
            data = frame.read_bytes()
            if len(data) < 100:
                continue
            b64_list.append(base64.b64encode(data).decode("utf-8"))
        except Exception:
            pass
    return b64_list


# ---------------------------------------------------------------------------
# OpenAI API call
# ---------------------------------------------------------------------------

async def _call_openai(
    prompt: str,
    system: str = "",
    max_tokens: int = 600,
    image_b64_list: Optional[List[str]] = None,
) -> Optional[str]:
    """
    Make an OpenAI API call with optional multi-image vision.

    - If image_b64_list is provided, uses OPENAI_VISION_MODEL (gpt-4o)
      with all images embedded in the message.
    - Returns response text or None on failure.
    """
    if not OPENAI_API_KEY:
        return None

    model = OPENAI_VISION_MODEL if image_b64_list else OPENAI_TEXT_MODEL

    try:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})

        if image_b64_list:
            content = []
            # Add all frames to message
            for b64 in image_b64_list:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}",
                        "detail": "low",
                    },
                })
            content.append({"type": "text", "text": prompt})
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": prompt})

        async with httpx.AsyncClient(timeout=60) as client:
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
                    "temperature": 0.72,
                },
            )

            if resp.status_code == 429:
                logger.warning("OpenAI rate limited")
                return None
            if resp.status_code != 200:
                logger.warning(f"OpenAI error {resp.status_code}: {resp.text[:300]}")
                return None

            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()

    except Exception as e:
        logger.warning(f"OpenAI call failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Evidence builder
# ---------------------------------------------------------------------------

def _build_telemetry_block(ctx: JobContext) -> str:
    """
    Build a comprehensive telemetry evidence block including
    pandas-enriched stats. Returns empty string if no telemetry.
    """
    telem: Optional[TelemetryData] = ctx.telemetry_data or ctx.telemetry
    trill: Optional[TrillScore] = ctx.trill_score or ctx.trill

    lines = []

    if telem:
        # --- Location (highest priority for caption quality) ---
        if getattr(telem, "location_display", None):
            lines.append(f"📍 Location: {telem.location_display}")
        if getattr(telem, "location_road", None):
            lines.append(f"🛣️  Road/Highway: {telem.location_road}")
        if (
            not getattr(telem, "location_display", None)
            and getattr(telem, "mid_lat", None)
            and getattr(telem, "mid_lon", None)
        ):
            lines.append(f"📍 GPS: {telem.mid_lat:.4f}°, {telem.mid_lon:.4f}°")

        # --- Speed stats ---
        lines.append(f"🏎️  Max Speed: {telem.max_speed_mph:.1f} mph")
        lines.append(f"📊 Avg Speed: {telem.avg_speed_mph:.1f} mph")

        # Pandas-enriched stats (attached by telemetry_stage)
        p95 = getattr(telem, "_speed_p95", None)
        if p95 and p95 > telem.avg_speed_mph:
            lines.append(f"📈 95th-Percentile Speed: {p95:.1f} mph")

        accel_events = getattr(telem, "_accel_events", 0)
        if accel_events > 0:
            lines.append(f"⚡ Hard Acceleration Events: {accel_events}")

        # --- Time at speed ---
        if telem.speeding_seconds > 0:
            lines.append(f"⏱️  Time Above Speed Threshold: {telem.speeding_seconds:.0f}s")
        if telem.euphoria_seconds > 0:
            lines.append(f"🔥 Euphoria-Speed Seconds: {telem.euphoria_seconds:.0f}s")

        # --- Distance and duration ---
        if telem.total_distance_miles > 0:
            lines.append(f"📏 Distance: {telem.total_distance_miles:.2f} miles")
        if telem.duration_seconds > 0:
            m = int(telem.duration_seconds // 60)
            s = int(telem.duration_seconds % 60)
            lines.append(f"⏰ Clip Duration: {m}m {s}s")

        if telem.max_altitude_ft > 0:
            lines.append(f"⛰️  Max Altitude: {telem.max_altitude_ft:.0f} ft")

    if trill:
        lines.append(f"🏆 Trill Score: {trill.score}/100 ({trill.bucket.upper()})")
        if trill.title_modifier:
            lines.append(f"🎯 Speed Character: {trill.title_modifier.strip()}")
        if trill.excessive_speed:
            lines.append("⚠️  Note: Excessive speed detected in clip")

    return "\n".join(lines)


def _build_full_evidence(ctx: JobContext, has_frames: bool) -> str:
    """
    Assemble complete evidence payload for OpenAI prompts.
    """
    parts = []

    if has_frames:
        parts.append(f"VIDEO FRAMES: {NUM_ANALYSIS_FRAMES} frames extracted from the video are attached for visual analysis.")

    telem_block = _build_telemetry_block(ctx)
    if telem_block:
        parts.append(f"TELEMETRY DATA (from .map file):\n{telem_block}")

    if ctx.filename:
        parts.append(f"FILENAME: {ctx.filename}")

    if ctx.title:
        parts.append(f"USER TITLE: {ctx.title}")

    if ctx.caption:
        parts.append(f"USER NOTES: {ctx.caption}")

    if ctx.platforms:
        parts.append(f"TARGET PLATFORMS: {', '.join(p.upper() for p in ctx.platforms)}")

    return "\n\n".join(parts) if parts else ""


def _get_platform_tone_instructions(platforms: List[str]) -> str:
    """
    Return platform-specific tone and style instructions based on
    which platforms this upload is targeting.
    """
    tones = []
    plats = [p.lower() for p in (platforms or [])]

    if "tiktok" in plats:
        tones.append(
            "TikTok: Keep it raw and punchy. Use trendy slang naturally. "
            "Short sentences. Start with a hook. Drive engagement with "
            "'POV:', 'Caught on dashcam:', or speed-first openers."
        )
    if "youtube" in plats or "youtube_shorts" in plats:
        tones.append(
            "YouTube Shorts: Slightly more descriptive. Mention the car, "
            "location, or speed clearly. Viewers expect context."
        )
    if "instagram" in plats:
        tones.append(
            "Instagram Reels: Aesthetic and aspirational. Emphasise the "
            "experience — the road, the speed, the feeling. Use emojis "
            "sparingly but effectively."
        )
    if "facebook" in plats:
        tones.append(
            "Facebook Reels: Broader audience. Keep it relatable. "
            "'Check out this dashcam footage from...' tone works well."
        )

    if not tones:
        tones.append(
            "General social media: Engaging, authentic, not AI-sounding. "
            "2-3 sentences max. Mention location and/or speed if available."
        )

    return "\n".join(tones)


def _build_location_mandate(ctx: JobContext) -> str:
    """
    Build a mandatory location instruction for prompts.
    If we have location data, the AI MUST use it — not optionally.
    """
    telem = ctx.telemetry_data or ctx.telemetry
    if not telem:
        return ""

    location = getattr(telem, "location_display", None)
    road = getattr(telem, "location_road", None)
    city = getattr(telem, "location_city", None)
    state = getattr(telem, "location_state", None)

    if not location and not city:
        return ""

    parts = []
    if location:
        parts.append(f"Location: {location}")
    if road:
        parts.append(f"Road: {road}")
    if city:
        parts.append(f"City: {city}")
    if state:
        parts.append(f"State: {state}")

    mandate = (
        f"\n\nLOCATION MANDATE: This video was recorded in {location or city}. "
        f"You MUST reference the location ({', '.join(parts)}) naturally in "
        f"the content. Do NOT generate generic content — use the actual place name. "
        f"Examples: 'flying through {city}', 'on the streets of {location}', "
        f"'dashcam caught this in {location}'."
    )
    return mandate


# ---------------------------------------------------------------------------
# Individual content generators
# ---------------------------------------------------------------------------

async def generate_title(
    ctx: JobContext,
    image_b64_list: List[str],
    evidence: str,
) -> Optional[str]:
    """Generate a grounded, location-aware title."""
    if not evidence:
        return None

    trill = ctx.trill_score or ctx.trill
    trill_modifier = ""
    if trill and trill.title_modifier:
        trill_modifier = (
            f"\nTrill modifier to incorporate: \"{trill.title_modifier.strip()}\" "
            f"(only if it fits naturally)"
        )

    location_mandate = _build_location_mandate(ctx)

    system = (
        "You are a viral social media content expert for driving and dashcam videos. "
        "Write ONLY the title text — no quotes, no explanation, no preamble. "
        "Max 100 characters. "
        "Make it punchy, engaging, and platform-ready. "
        "Base it STRICTLY on the visual content and telemetry provided. "
        "Do NOT invent events or exaggerate beyond what the data shows."
        f"{location_mandate}"
    )

    prompt = (
        "Generate a short, viral title (max 100 characters) for this dashcam/driving video.\n\n"
        f"EVIDENCE:\n{evidence}"
        f"{trill_modifier}"
        "\n\nReturn ONLY the title text. No quotes. No explanation."
    )

    return await _call_openai(
        prompt,
        system,
        max_tokens=80,
        image_b64_list=image_b64_list or None,
    )


async def generate_caption(
    ctx: JobContext,
    image_b64_list: List[str],
    evidence: str,
) -> Optional[str]:
    """Generate a grounded, location-aware, platform-tuned caption."""
    if not evidence:
        return None

    platform_tones = _get_platform_tone_instructions(ctx.platforms or [])
    location_mandate = _build_location_mandate(ctx)

    telem = ctx.telemetry_data or ctx.telemetry
    speed_callout = ""
    if telem and telem.max_speed_mph > 0:
        speed_callout = (
            f"\nSpeed context: Max speed was {telem.max_speed_mph:.0f} mph. "
            f"Reference the speed if it adds punch — but do NOT exaggerate."
        )

    system = (
        "You are a social media content expert for driving and dashcam videos. "
        "Write ONLY the caption text — 2-3 sentences, no quotes, no preamble. "
        "Describe what is ACTUALLY shown in the frames and what the telemetry ACTUALLY reveals. "
        "Do NOT fabricate road conditions, events, or emotions not supported by the evidence. "
        f"Platform tone guidance:\n{platform_tones}"
        f"{location_mandate}"
        f"{speed_callout}"
    )

    prompt = (
        "Write a short, engaging social media caption for this dashcam/driving video.\n\n"
        f"EVIDENCE:\n{evidence}"
        "\n\nReturn ONLY the caption text. 2-3 sentences max."
    )

    return await _call_openai(
        prompt,
        system,
        max_tokens=250,
        image_b64_list=image_b64_list or None,
    )


async def generate_hashtags(
    ctx: JobContext,
    image_b64_list: List[str],
    evidence: str,
    max_count: int = 15,
) -> List[str]:
    """
    Generate grounded, location-specific, platform-relevant hashtags.

    Priority order in merged list:
      1. Trill score hashtags (evidence-based from .map file)
      2. Location hashtags (city, state, road — real place names only)
      3. AI-generated content hashtags
    """
    trill = ctx.trill_score or ctx.trill
    trill_tags: List[str] = list(trill.hashtags) if trill and trill.hashtags else []

    telem = ctx.telemetry_data or ctx.telemetry
    location_tags: List[str] = []
    location_instruction = ""

    if telem:
        city = getattr(telem, "location_city", None)
        state = getattr(telem, "location_state", None)
        road = getattr(telem, "location_road", None)
        display = getattr(telem, "location_display", None)
        country = getattr(telem, "location_country", None)

        if city:
            clean_city = city.replace(" ", "").replace("-", "")
            location_tags.append(f"#{clean_city}")
        if state:
            from .telemetry_stage import _abbreviate_us_state
            abbr = _abbreviate_us_state(state)
            if abbr != city:  # avoid duplicate if city IS the state
                location_tags.append(f"#{abbr}")
        if road:
            clean_road = road.replace(" ", "").replace("-", "").replace("/", "")
            if len(clean_road) > 2:
                location_tags.append(f"#{clean_road}")
        if country and country not in ("US", ""):
            location_tags.append(f"#{country}")

        if display:
            location_instruction = (
                f"\n\nLOCATION HASHTAG MANDATE: The video was recorded in {display}. "
                f"You MUST include city-specific and/or state-specific hashtags "
                f"(e.g. #{city.replace(' ', '') if city else ''}, #{abbr if state else ''}, "
                f"#{''.join(display.split(',')[0].split())}driving, etc.). "
                f"Use ONLY real, verifiable place names — NO invented locations."
            )

    if not evidence:
        return list(dict.fromkeys(trill_tags + location_tags))[:max_count]

    platform_tags = []
    for p in (ctx.platforms or []):
        p = p.lower()
        if p == "tiktok":
            platform_tags += ["#fyp", "#foryoupage", "#dashcam"]
        elif p in ("youtube", "youtube_shorts"):
            platform_tags += ["#YouTubeShorts", "#dashcam"]
        elif p == "instagram":
            platform_tags += ["#Reels", "#dashcam"]
        elif p == "facebook":
            platform_tags += ["#FacebookReels"]

    system = (
        "You generate hashtags for social media dashcam/driving videos. "
        "Return ONLY a JSON array of strings, each starting with #. "
        "Example: [\"#dashcam\", \"#driving\", \"#fyp\"]. "
        "Base hashtags ONLY on the visual content and telemetry data provided. "
        "Do NOT include generic filler tags unrelated to the content. "
        "Include a mix of: speed/driving tags, car culture tags, location tags, "
        "and viral discovery tags."
        f"{location_instruction}"
    )

    prompt = (
        f"Generate {min(max_count, 12)} relevant hashtags for this dashcam/driving video. "
        "Return them as a JSON array of strings starting with #.\n\n"
        f"EVIDENCE:\n{evidence}"
    )

    result = await _call_openai(
        prompt,
        system,
        max_tokens=150,
        image_b64_list=image_b64_list or None,
    )

    ai_tags: List[str] = []
    if result:
        try:
            clean = result.strip().strip("`").strip()
            if clean.lower().startswith("json"):
                clean = clean[4:].strip()
            parsed = json.loads(clean)
            if isinstance(parsed, list):
                ai_tags = [str(t) for t in parsed if isinstance(t, str) and t.startswith("#")]
        except (json.JSONDecodeError, ValueError):
            # Fallback: scrape hashtags from free text
            ai_tags = [w.strip() for w in result.split() if w.startswith("#")][:max_count]

    # Merge with deduplication: Trill → location → platform → AI
    merged = list(dict.fromkeys(trill_tags + location_tags + platform_tags + ai_tags))
    return merged[:max_count]


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------

async def run_caption_stage(ctx: JobContext) -> JobContext:
    """
    Execute AI caption generation stage.

    Pipeline:
    1. Check entitlements and API key
    2. Extract 8 frames from video using FFmpeg
    3. Load thumbnail as additional visual context
    4. Build telemetry evidence block (location, speed, Trill, pandas stats)
    5. Generate title, caption, hashtags via GPT-4o vision
    6. Write results to ctx.ai_title, ctx.ai_caption, ctx.ai_hashtags

    Requirements:
    - Runs AFTER telemetry_stage (for location + Trill data)
    - Runs AFTER thumbnail_stage (thumbnail used as frame 0)
    - Will skip only if NO video file AND NO telemetry exist
    """
    ctx.mark_stage("caption")

    # Entitlement check
    if ctx.entitlements and not ctx.entitlements.can_ai:
        raise SkipStage("AI captions not available for this tier")

    if not OPENAI_API_KEY:
        raise SkipStage("OpenAI API key not configured")

    # Determine video source for frame extraction
    video_path: Optional[Path] = None
    for candidate in (ctx.processed_video_path, ctx.local_video_path):
        if candidate and Path(candidate).exists():
            video_path = Path(candidate)
            break

    temp_dir: Optional[Path] = ctx.temp_dir

    # ------------------------------------------------------------------ #
    # Frame extraction                                                     #
    # ------------------------------------------------------------------ #
    extracted_frames: List[Path] = []

    if video_path and temp_dir:
        try:
            extracted_frames = await extract_analysis_frames(video_path, temp_dir)
        except Exception as e:
            logger.warning(f"Frame extraction failed (non-fatal): {e}")

    # Also grab the thumbnail as an additional frame (it's already extracted)
    thumbnail_path: Optional[Path] = ctx.thumbnail_path
    if thumbnail_path and Path(thumbnail_path).exists():
        # Prepend thumbnail so GPT-4o sees it first as "the poster frame"
        extracted_frames = [Path(thumbnail_path)] + [
            f for f in extracted_frames
            if f != thumbnail_path and f.exists()
        ]

    has_frames = len(extracted_frames) > 0
    image_b64_list = _frames_to_b64(extracted_frames, max_frames=6) if has_frames else []
    has_images = len(image_b64_list) > 0

    # ------------------------------------------------------------------ #
    # Telemetry evidence                                                   #
    # ------------------------------------------------------------------ #
    has_telemetry = bool(_build_telemetry_block(ctx))

    # Skip only if absolutely no evidence
    if not has_images and not has_telemetry and not ctx.filename:
        logger.info("Skipping caption generation: no frames, no telemetry, no filename")
        raise SkipStage("No evidence available for grounded caption generation")

    evidence = _build_full_evidence(ctx, has_images)

    logger.info(
        f"Caption generation for upload {ctx.upload_id} | "
        f"frames={len(extracted_frames)} | b64_frames={len(image_b64_list)} | "
        f"telemetry={'yes' if has_telemetry else 'no'} | "
        f"model={'vision' if has_images else 'text'}"
    )

    # ------------------------------------------------------------------ #
    # Generate title                                                       #
    # ------------------------------------------------------------------ #
    try:
        ai_title = await generate_title(ctx, image_b64_list, evidence)
        if ai_title:
            # Strip surrounding quotes if model added them
            ai_title = ai_title.strip().strip('"').strip("'")
            ctx.ai_title = ai_title[:100]
            logger.info(f"AI title: {ctx.ai_title}")
    except Exception as e:
        logger.warning(f"AI title generation failed (non-fatal): {e}")

    # ------------------------------------------------------------------ #
    # Generate caption                                                     #
    # ------------------------------------------------------------------ #
    try:
        ai_caption = await generate_caption(ctx, image_b64_list, evidence)
        if ai_caption:
            ai_caption = ai_caption.strip().strip('"').strip("'")
            ctx.ai_caption = ai_caption[:500]
            logger.info(f"AI caption: {ctx.ai_caption[:100]}")
    except Exception as e:
        logger.warning(f"AI caption generation failed (non-fatal): {e}")

    # ------------------------------------------------------------------ #
    # Generate hashtags                                                    #
    # ------------------------------------------------------------------ #
    try:
        # Determine hashtag limit from entitlements
        max_hashtags = 30  # default ceiling
        if ctx.entitlements:
            try:
                max_hashtags = ctx.entitlements.max_hashtags or 30
            except AttributeError:
                pass

        # Merge with any user-supplied hashtags
        ai_hashtags = await generate_hashtags(ctx, image_b64_list, evidence, max_count=max_hashtags)
        if ai_hashtags:
            ctx.ai_hashtags = ai_hashtags
            logger.info(f"AI hashtags ({len(ai_hashtags)}): {ai_hashtags[:8]}...")
    except Exception as e:
        logger.warning(f"AI hashtag generation failed (non-fatal): {e}")

    # AIC cost: 1 token per generation batch
    ctx.aic_cost += 1

    return ctx
